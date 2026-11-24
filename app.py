import streamlit as st
import uuid6
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from agent_orchestrator.graph import create_graph, list_saved_threads
from agent_orchestrator.config import get_available_repos
from agent_orchestrator.worktree import cleanup_thread_worktrees, prune_all_worktrees

st.set_page_config(page_title="Agent Orchestrator", layout="wide")

# Cache the compiled graph and its checkpointer
@st.cache_resource
def get_graph():
    return create_graph()

graph = get_graph()

def reset_session():
    """Starts a new task thread."""
    if "thread_id" in st.session_state:
        cleanup_thread_worktrees(st.session_state.thread_id)
    st.session_state.thread_id = str(uuid6.uuid7())

# Initialize session thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid6.uuid7())

def switch_thread():
    new_thread = st.session_state.session_selector
    if new_thread != "Current Session" and new_thread != st.session_state.thread_id:
        st.session_state.thread_id = new_thread

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("⚙️ Orchestrator")
    st.button("New Task / Clear Session", on_click=reset_session, use_container_width=True)
    
    saved_threads = [t for t in list_saved_threads() if t != st.session_state.thread_id]
    if saved_threads:
        st.selectbox("📜 Recent Sessions", ["Current Session"] + saved_threads, key="session_selector", on_change=switch_thread)
    
    if st.button("Clean Unused Worktrees", use_container_width=True):
        prune_all_worktrees()
        st.success("Pruned unused worktrees!")
    
    st.divider()
    st.info(f"**Current Branch:** `agent/{st.session_state.thread_id}`")
    
    st.subheader("📦 Available Workspaces")
    repos = get_available_repos()
    if repos:
        for r in repos:
            st.markdown(f"- `{r}`")
    else:
        st.warning("No repositories found in WORKSPACES_DIR")

# ----------------- Main Chat UI -----------------
st.title("🤖 Cross-Repo Coding Agent")

# Define execution configuration for LangGraph
config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "recursion_limit": 30
}

# Fetch the current state
state = graph.get_state(config)
messages = state.values.get("messages", [])

# Render conversation history
for msg in messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            st.chat_message("assistant").write(msg.content)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            with st.chat_message("assistant"):
                for tc in msg.tool_calls:
                    st.code(f"🔨 Tool call: {tc['name']}\nArguments: {tc['args']}", language="json")
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool"):
            st.text(f"Result from {msg.name}:\n{msg.content[:1000]}..." if len(msg.content) > 1000 else f"Result from {msg.name}:\n{msg.content}")

def process_stream(input_data):
    """Processes graph stream and updates UI in real-time."""
    with st.status("Agent is working...", expanded=True) as status:
        try:
            for event in graph.stream(input_data, config, stream_mode="updates"):
                for node, updates in event.items():
                    if node == "planner":
                        st.write("🗺️ Planner formulated an execution plan.")
                    elif node == "coder":
                        st.write("💻 Coder updated the workspace.")
                    elif node == "verifier":
                        st.write("✅ Verifier checked the code.")
                    elif node == "commit_and_cleanup":
                        st.write("🧹 Committing and cleaning up worktrees...")
                    elif node.endswith("tools"):
                        msgs = updates.get("messages", [])
                        if msgs:
                            for m in msgs:
                                if isinstance(m, ToolMessage):
                                    st.write(f"🔧 Tool Result: `{m.name}`")
                        else:
                            st.write(f"🔧 Executing tools in `{node}`...")
            status.update(label="Task Paused or Completed", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Error: {e}", state="error", expanded=False)
            st.error(str(e))

# ----------------- Execution & Interrupts -----------------
if state.next and state.next[0] in ["plan_review", "human_review"]:
    node_name = state.next[0]
    interrupt_data = state.tasks[0].interrupts[0].value
    int_type = interrupt_data.get("type")
    
    if int_type == "plan_review":
        st.warning("📋 **Agent paused for Plan Review**")
        st.subheader("Proposed Execution Plan")
        st.markdown(interrupt_data.get("plan", ""))
        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ Approve Plan", use_container_width=True):
                process_stream(Command(resume={"action": "approve"}))
                st.rerun()
        with col2:
            with st.form("plan_feedback_form"):
                feedback = st.text_area("Revise Plan Feedback:", placeholder="e.g. Please also include changes to X...")
                if st.form_submit_button("Submit Feedback", use_container_width=True):
                    if not feedback.strip():
                        st.error("Please enter feedback before submitting.")
                    else:
                        process_stream(Command(resume={"action": "reject", "feedback": feedback}))
                        st.rerun()
                        
    elif int_type == "code_review":
        st.warning("✋ **Agent paused for Code Review**")
        diffs = interrupt_data.get("diffs", {})
        
        if diffs:
            st.subheader("Git Diffs")
            for repo, diff in diffs.items():
                with st.expander(f"Diff: {repo}", expanded=True):
                    st.code(diff, language="diff")
        else:
            st.info("No git changes detected across workspaces.")
            
        st.divider()
        
        # Approval & Feedback Controls
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("✅ Approve & Commit", use_container_width=True):
                process_stream(Command(resume={"action": "approve"}))
                st.rerun()
                
        with col2:
            with st.form("code_feedback_form"):
                feedback = st.text_area("Request Changes (Feedback):", placeholder="e.g. Please extract the logic into a separate helper function...")
                if st.form_submit_button("Submit Feedback", use_container_width=True):
                    if not feedback.strip():
                        st.error("Please enter feedback before submitting.")
                    else:
                        process_stream(Command(resume={"action": "reject", "feedback": feedback}))
                        st.rerun()

else:
    # Render chat input only if not paused
    if prompt := st.chat_input("Enter your request here (e.g. 'Refactor the auth flow in repo A and repo B')..."):
        st.chat_message("user").write(prompt)
        
        input_data = Command(resume=None) if state.next else {"messages": [HumanMessage(content=prompt)]}
        process_stream(input_data)
        st.rerun()
