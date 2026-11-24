from typing import Annotated, TypedDict, Literal
import subprocess
import sqlite3
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_orchestrator.prompts import PLANNER_PROMPT, CODER_PROMPT, VERIFIER_PROMPT
from agent_orchestrator.tools import (
    list_repo_files, 
    read_repo_file, 
    write_repo_file, 
    run_git_diff,
    grep_search,
    replace_in_file,
    run_repo_command
)
from agent_orchestrator.config import WORKTREES_DIR, CHECKPOINT_DB_PATH
from agent_orchestrator.worktree import commit_worktree, cleanup_thread_worktrees

db_conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)

def list_saved_threads() -> list[str]:
    """Returns a list of unique thread_ids from the SQLite database."""
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        # Table might not exist yet
        return []

# Define the state for the Graph
class MultiRepoState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    verification_attempts: int

# Tool collections
read_only_tools = [list_repo_files, read_repo_file, grep_search]
all_tools = [list_repo_files, read_repo_file, write_repo_file, replace_in_file, run_git_diff, grep_search, run_repo_command]

# Initialize LLM with bound tools
# Ensure GOOGLE_API_KEY is in your environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
planner_llm = llm.bind_tools(read_only_tools)
coder_llm = llm.bind_tools(all_tools)
verifier_llm = llm.bind_tools([run_repo_command, read_repo_file])

def planner_node(state: MultiRepoState) -> Command[Literal["planner_tools", "plan_review"]]:
    """Generates the execution plan using read-only tools."""
    messages_with_sys = [SystemMessage(content=PLANNER_PROMPT)] + state.get("messages", [])
    response = planner_llm.invoke(messages_with_sys)
    
    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="planner_tools")
        
    # Task planning complete, pass the plan forward
    return Command(update={"messages": [response], "plan": response.content}, goto="plan_review")

def plan_review_node(state: MultiRepoState) -> Command[Literal["coder", "planner"]]:
    """Interrupts execution to allow human review of the proposed plan."""
    feedback = interrupt({
        "type": "plan_review",
        "plan": state.get("plan", ""),
        "message": "Please review the execution plan. Approve to proceed, or provide feedback to revise."
    })
    
    if feedback.get("action") == "approve":
        # Reset verification attempts for the new plan
        return Command(update={"verification_attempts": 0}, goto="coder")
    else:
        user_feedback_msg = f"Feedback on the plan: {feedback.get('feedback', 'Please revise the plan.')}"
        return Command(update={"messages": [HumanMessage(content=user_feedback_msg)]}, goto="planner")

def coder_node(state: MultiRepoState) -> Command[Literal["coder_tools", "verifier"]]:
    """Executes the approved plan and makes code modifications."""
    current_plan = state.get("plan", "No plan provided.")
    sys_msg = SystemMessage(content=CODER_PROMPT + f"\n\n### APPROVED PLAN ###\n{current_plan}")
    messages_with_sys = [sys_msg] + state.get("messages", [])
    
    response = coder_llm.invoke(messages_with_sys)
    
    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="coder_tools")
        
    # Finished coding, route to verifier
    return Command(update={"messages": [response]}, goto="verifier")

def verifier_node(state: MultiRepoState) -> Command[Literal["verifier_tools", "coder", "human_review"]]:
    """Verifies the code changes by running tests or linters."""
    attempts = state.get("verification_attempts", 0)
    
    if attempts >= 3:
        return Command(
            update={"messages": [SystemMessage(content="Verification attempts exhausted. Proceeding to human review.")]}, 
            goto="human_review"
        )
        
    sys_msg = SystemMessage(content=VERIFIER_PROMPT)
    messages_with_sys = [sys_msg] + state.get("messages", [])
    response = verifier_llm.invoke(messages_with_sys)
    
    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="verifier_tools")
        
    if "VERIFICATION_PASSED" in response.content:
        return Command(update={"messages": [response]}, goto="human_review")
    else:
        # Pass feedback to coder and increment attempts
        return Command(
            update={
                "messages": [response], 
                "verification_attempts": attempts + 1
            }, 
            goto="coder"
        )

def human_review_node(state: MultiRepoState, config: RunnableConfig) -> Command[Literal["coder", "commit_and_cleanup"]]:
    """A node that generates git diffs from active worktrees and pauses for human approval via interrupt()."""
    thread_id = config["configurable"]["thread_id"]
    thread_dir = WORKTREES_DIR / thread_id
    
    # Collect diffs from worktrees for this thread
    diffs = {}
    if thread_dir.exists():
        for repo_dir in thread_dir.iterdir():
            if repo_dir.is_dir():
                repo = repo_dir.name
                try:
                    res = subprocess.run(
                        ["git", "diff"], cwd=repo_dir, capture_output=True, text=True
                    )
                    if res.stdout.strip():
                        diffs[repo] = res.stdout
                except Exception:
                    pass
            
    # Pause the graph execution and send data to the user interface
    feedback = interrupt({
        "type": "code_review",
        "diffs": diffs,
        "message": "Please review the git diffs. Approve to finish, or provide feedback to request changes."
    })
    
    # Process the human feedback
    if feedback.get("action") == "approve":
        return Command(goto="commit_and_cleanup")
    else:
        # Send feedback back to the coder node
        user_feedback_msg = f"Feedback from Code Review: {feedback.get('feedback', 'Please revise your changes.')}"
        # Reset verification attempts so it can be verified again
        return Command(update={"messages": [HumanMessage(content=user_feedback_msg)], "verification_attempts": 0}, goto="coder")

def commit_and_cleanup_node(state: MultiRepoState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """Commits all worktree changes and cleans up the worktrees."""
    thread_id = config["configurable"]["thread_id"]
    thread_dir = WORKTREES_DIR / thread_id
    
    commit_messages = []
    if thread_dir.exists():
        for repo_dir in thread_dir.iterdir():
            if repo_dir.is_dir():
                repo = repo_dir.name
                msg = commit_worktree(repo, thread_id, f"Agent auto-commit for task {thread_id}")
                commit_messages.append(f"- {repo}: {msg}")
                
    # Force remove all worktrees for this thread
    cleanup_thread_worktrees(thread_id)
    
    summary = "Approved and changes committed successfully.\n" + "\n".join(commit_messages)
    return Command(update={"messages": [AIMessage(content=summary)]}, goto="__end__")

def create_graph():
    """Builds and compiles the StateGraph with a checkpointer."""
    graph_builder = StateGraph(MultiRepoState)
    
    # Add Core Nodes
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("plan_review", plan_review_node)
    graph_builder.add_node("coder", coder_node)
    graph_builder.add_node("verifier", verifier_node)
    graph_builder.add_node("human_review", human_review_node)
    graph_builder.add_node("commit_and_cleanup", commit_and_cleanup_node)
    
    # Add Tool Nodes
    graph_builder.add_node("planner_tools", ToolNode(read_only_tools))
    graph_builder.add_node("coder_tools", ToolNode(all_tools))
    graph_builder.add_node("verifier_tools", ToolNode([run_repo_command, read_repo_file]))
    
    # Define Edges
    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner_tools", "planner")
    graph_builder.add_edge("coder_tools", "coder")
    graph_builder.add_edge("verifier_tools", "verifier")
    
    # Note: State transitions between planner <-> plan_review <-> coder <-> verifier <-> human_review 
    # are handled via the Command pattern within the nodes.
    
    # Memory saver for state persistence (essential for Streamlit and interrupts)
    memory = SqliteSaver(db_conn)
    memory.setup()
    
    # Compile the graph
    return graph_builder.compile(checkpointer=memory)