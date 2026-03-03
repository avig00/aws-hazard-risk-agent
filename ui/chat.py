"""
Conversation history and session state management for the Streamlit chat UI.
"""
import streamlit as st


def init_session():
    """Initialize session state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tool_mode" not in st.session_state:
        st.session_state.tool_mode = "Auto"
    if "year_range" not in st.session_state:
        st.session_state.year_range = (2015, 2023)


def add_message(role: str, content: str, metadata: dict = None):
    """Append a message to the conversation history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "metadata": metadata or {},
    })


def render_history():
    """Render all past messages in the chat container."""
    from ui.components import render_tool_badges
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("metadata", {})
            if meta.get("tool_used"):
                render_tool_badges(meta["tool_used"], intent=meta.get("intent", ""))


def clear_history():
    """Reset the conversation."""
    st.session_state.messages = []
