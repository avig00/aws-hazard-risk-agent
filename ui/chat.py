"""
Conversation history and session state management for the Streamlit chat UI.
"""
import streamlit as st

_NO_DATA_PHRASES = ("no data was found", "no matching records", "no results")


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
    """Render all past messages including rich content (charts, tables, predictions)."""
    from ui.components import (
        render_analytics_table, render_citations, render_no_data,
        render_prediction_card, render_sql_expander, render_tool_badges,
        render_trend_chart,
    )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            meta = msg.get("metadata", {})

            if msg["role"] != "assistant":
                st.markdown(content)
                continue

            tools_used = meta.get("tool_used", [])
            tool_outputs = meta.get("tool_outputs", {})
            sources = meta.get("sources", [])
            intent = meta.get("intent", "")

            # Text answer
            if content and content != "See results above.":
                if any(p in content.lower() for p in _NO_DATA_PHRASES):
                    render_no_data(content)
                else:
                    st.markdown(content)

            # Analytics table + chart
            if "query" in tools_used and "results" in tool_outputs.get("query", {}):
                query_out = tool_outputs["query"]
                results_data = query_out.get("results", [])
                sql = query_out.get("sql_executed", "")
                q_intent = query_out.get("intent", "") or intent

                if results_data:
                    if "trend" in q_intent or q_intent == "hazard_trend_by_year":
                        render_trend_chart(results_data)
                    else:
                        render_analytics_table(results_data, title="Analytics Results")

                if sql:
                    render_sql_expander(sql)

            # ML prediction card
            pred_out = tool_outputs.get("predict", {})
            if "predict" in tools_used and "risk_tier" in pred_out and not pred_out.get("_no_county"):
                render_prediction_card(pred_out)

            # RAG citations
            if sources:
                render_citations(sources)

            # Tool badges
            if tools_used:
                render_tool_badges(tools_used, intent=intent)


def clear_history():
    """Reset the conversation."""
    st.session_state.messages = []
