import subprocess
import os
import re
import fnmatch
import shlex
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from agent_orchestrator.config import WORKSPACES_DIR
from agent_orchestrator.worktree import ensure_worktree

def _get_safe_repo_path(repo: str, config: RunnableConfig, filepath: str = "") -> Path:
    """
    Validates and returns a safe absolute path within the requested repository's worktree.
    Enforces strict path traversal guarding.
    """
    thread_id = config["configurable"]["thread_id"]
    worktree_dir = ensure_worktree(repo, thread_id).resolve()
    
    # Ensure the target file path is within the worktree directory
    target_path = (worktree_dir / filepath).resolve()
    if not target_path.is_relative_to(worktree_dir):
        raise ValueError(f"Path traversal detected! Cannot access: {filepath}")
        
    return target_path

@tool
def list_repo_files(repo: str, config: RunnableConfig, path: str = ".") -> list[str]:
    """Lists files and directories in a specific path within a repository."""
    target_dir = _get_safe_repo_path(repo, config, path)
    if not target_dir.exists() or not target_dir.is_dir():
        return []
    return [str(p.relative_to(target_dir)) for p in target_dir.iterdir()]

@tool
def grep_search(repo: str, pattern: str, config: RunnableConfig, file_glob: str = "*") -> str:
    """Searches for a regular expression pattern within files matching file_glob in the repository."""
    worktree_dir = _get_safe_repo_path(repo, config)
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regular expression pattern - {e}"

    exclude_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
    results = []
    match_count = 0
    MAX_MATCHES = 50

    for root, dirs, files in os.walk(worktree_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if not fnmatch.fnmatch(file, file_glob):
                continue
                
            file_path = Path(root) / file
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if regex.search(line):
                            rel_path = file_path.relative_to(worktree_dir)
                            # Remove trailing newline for cleaner output
                            results.append(f"{rel_path}:{i+1}: {line.rstrip()}")
                            match_count += 1
                            
                            if match_count >= MAX_MATCHES:
                                results.append(f"\n... (Output truncated at {MAX_MATCHES} matches)")
                                return "\n".join(results)
            except UnicodeDecodeError:
                # Skip binary files or files with unknown encodings
                continue

    if not results:
        return f"No matches found for pattern '{pattern}'."
        
    return "\n".join(results)

@tool
def read_repo_file(repo: str, filepath: str, config: RunnableConfig) -> str:
    """Reads and returns the content of a file in a repository."""
    target_file = _get_safe_repo_path(repo, config, filepath)
    if not target_file.exists() or not target_file.is_file():
        raise FileNotFoundError(f"File not found: {filepath} in {repo}")
    return target_file.read_text(encoding="utf-8")

@tool
def write_repo_file(repo: str, filepath: str, content: str, config: RunnableConfig) -> str:
    """Writes content to a file in a repository. Creates parent directories if missing."""
    target_file = _get_safe_repo_path(repo, config, filepath)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {filepath} in {repo}"

@tool
def replace_in_file(repo: str, relative_path: str, old_text: str, new_text: str, config: RunnableConfig) -> str:
    """Replaces exactly ONE occurrence of old_text with new_text in the specified file."""
    target_file = _get_safe_repo_path(repo, config, relative_path)
    
    if not target_file.exists() or not target_file.is_file():
        return f"Error: File not found - {relative_path}"
        
    try:
        content = target_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: Unable to read {relative_path} as utf-8 text."
    
    occurrences = content.count(old_text)
    
    if occurrences == 0:
        return (
            f"Error: `old_text` not found in {relative_path}.\n"
            "Please read the file and ensure your `old_text` matches the file content EXACTLY, "
            "including spaces, indentation, and newlines."
        )
    elif occurrences > 1:
        return (
            f"Error: Found {occurrences} occurrences of `old_text` in {relative_path}.\n"
            "Please include more surrounding context (lines before and after) in your `old_text` "
            "to uniquely identify the exact section you want to change."
        )
        
    new_content = content.replace(old_text, new_text)
    target_file.write_text(new_content, encoding="utf-8")
    
    return f"Successfully replaced 1 occurrence in {relative_path}."

@tool
def run_git_diff(repo: str, config: RunnableConfig) -> str:
    """Runs 'git diff' in the specified repository to see unstaged changes."""
    repo_dir = _get_safe_repo_path(repo, config)
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repository not found: {repo}")
        
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout if result.stdout.strip() else "No unstaged changes."
    except subprocess.CalledProcessError as e:
        return f"Error running git diff: {e.stderr}"
    except FileNotFoundError:
        return "Git is not installed or not found in PATH."

@tool
def run_repo_command(repo: str, command: str, config: RunnableConfig) -> str:
    """Runs an allowed test, lint, or build command in the repository."""
    # Security: check forbidden characters
    forbidden = ["&&", ";", "|", ">", "<"]
    if any(char in command for char in forbidden):
        return "Error: Command contains forbidden shell operators (&&, ;, |, >, <)."

    # Security: Allowlist
    allowed_prefixes = (
        "pytest", "mypy", "ruff", "flake8", 
        "npm test", "npm run lint", "npm run build", 
        "pnpm test", "pnpm lint", "pnpm typecheck", 
        "npx tsc", "go test"
    )
    if not command.strip().startswith(allowed_prefixes):
        return f"Error: Command not allowed. Must start with one of: {', '.join(allowed_prefixes)}"
    
    worktree_dir = _get_safe_repo_path(repo, config)
    if not worktree_dir.exists():
         return f"Error: Repository worktree not found: {repo}"
         
    try:
        args = shlex.split(command)
        result = subprocess.run(
            args,
            cwd=worktree_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = f"Exit Code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        if len(output) > 4000:
            output = output[:4000] + "\n... (Output truncated at 4000 chars)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"
