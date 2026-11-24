import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Resolve WORKSPACES_DIR to an absolute path for security
_workspaces_env = os.getenv("WORKSPACES_DIR", "../workspaces")
WORKSPACES_DIR = Path(_workspaces_env).resolve()

# Resolve WORKTREES_DIR for agent isolation
_worktrees_env = os.getenv("WORKTREES_DIR", "../agent-worktrees")
WORKTREES_DIR = Path(_worktrees_env).resolve()

CHECKPOINT_DB_PATH = Path("agent_checkpoints.db").resolve()

def get_available_repos() -> list[str]:
    """Scans WORKSPACES_DIR for available repository folders."""
    if not WORKSPACES_DIR.exists() or not WORKSPACES_DIR.is_dir():
        return []
    
    # Return names of all directories inside WORKSPACES_DIR
    return sorted([d.name for d in WORKSPACES_DIR.iterdir() if d.is_dir()])
