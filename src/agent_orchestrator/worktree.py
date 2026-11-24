import subprocess
import shutil
from pathlib import Path
from agent_orchestrator.config import WORKSPACES_DIR, WORKTREES_DIR, get_available_repos

def ensure_worktree(repo_name: str, thread_id: str) -> Path:
    """Ensures a worktree exists for the given thread and repo. Returns the path."""
    repo_dir = WORKSPACES_DIR / repo_name
    
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise ValueError(f"Repository not found: {repo_name}")
        
    if not (repo_dir / ".git").exists():
        raise ValueError(f"Not a git repository: {repo_name}")

    # Ensure repository has at least one commit
    try:
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"Repository '{repo_name}' has no commits yet. Please create an initial commit in the main repo first.")

    worktree_path = WORKTREES_DIR / thread_id / repo_name
    branch_name = f"agent/{thread_id}"

    if worktree_path.exists():
        return worktree_path

    # Create worktree folder hierarchy if missing
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try creating the worktree and branch
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, str(worktree_path)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        # If the branch already exists, just attach the worktree to the existing branch
        if "already exists" in e.stderr or "fatal: A branch named" in e.stderr:
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), branch_name],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
        else:
            raise RuntimeError(f"Failed to create worktree: {e.stderr}")
            
    return worktree_path

def commit_worktree(repo_name: str, thread_id: str, message: str) -> str:
    """Commits changes in the worktree of a repository."""
    worktree_path = WORKTREES_DIR / thread_id / repo_name
    if not worktree_path.exists():
        return f"No worktree found for {repo_name}"

    # Stage all changes
    subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True)
    
    # Commit changes
    res = subprocess.run(
        ["git", "commit", "-m", message], 
        cwd=worktree_path, 
        capture_output=True, 
        text=True
    )
    
    if "nothing to commit" in res.stdout:
        return f"No changes to commit in {repo_name}."
        
    return f"Successfully committed in {repo_name} (Branch: agent/{thread_id})."

def cleanup_thread_worktrees(thread_id: str):
    """Force removes all worktrees for a specific thread."""
    thread_dir = WORKTREES_DIR / thread_id
    if not thread_dir.exists():
        return

    # Remove each repo worktree
    for repo_dir in thread_dir.iterdir():
        if repo_dir.is_dir():
            # Run git worktree remove
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(repo_dir)], 
                capture_output=True
            )

    # Remove the thread directory
    try:
        shutil.rmtree(thread_dir)
    except Exception:
        pass

def prune_all_worktrees():
    """Prunes dead worktree links for all known repositories."""
    for repo in get_available_repos():
        repo_dir = WORKSPACES_DIR / repo
        if (repo_dir / ".git").exists():
            subprocess.run(["git", "worktree", "prune"], cwd=repo_dir, capture_output=True)
