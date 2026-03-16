"""
Hazard Risk Intelligence Agent — Streamlit application.

Hosted on Streamlit Cloud. Calls AWS services directly via boto3
using credentials stored in st.secrets.

Run locally:
    streamlit run app.py

Deploy:
    Connect GitHub repo on share.streamlit.io → set AWS secrets in app settings.
"""
import os

import boto3
import streamlit as st

from agent.orchestrator import run_agent
from ui.chat import add_message, clear_history, init_session, render_history
from ui.components import (
    render_analytics_table,
    render_citations,
    render_error,
    render_no_data,
    render_prediction_card,
    render_sql_expander,
    render_tool_badges,
    render_trend_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hazard Risk Intelligence Agent",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AWS credentials from Streamlit secrets ────────────────────────────────────
def _configure_aws():
    """Inject AWS credentials from st.secrets into environment variables."""
    if hasattr(st, "secrets") and "aws" in st.secrets:
        os.environ["AWS_ACCESS_KEY_ID"]     = st.secrets["aws"]["access_key_id"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["aws"]["secret_access_key"]
        os.environ["AWS_DEFAULT_REGION"]    = st.secrets["aws"].get("region", "us-east-1")
    if hasattr(st, "secrets") and "pinecone" in st.secrets:
        os.environ["PINECONE_API_KEY"] = st.secrets["pinecone"]["api_key"]


_configure_aws()


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Chat avatars — blue/teal theme; override Streamlit's default orange/red.
       Target both the legacy data-testid and the newer stChatMessageAvatarUser/Assistant
       selectors used in recent Streamlit versions. */

    /* User avatar — dark navy background, sky-blue border + icon */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #1E40AF !important;
        border: 2px solid #4F9CF9 !important;
        color: #E0F2FE !important;
    }
    [data-testid="chatAvatarIcon-user"] svg,
    [data-testid="chatAvatarIcon-user"] svg path,
    [data-testid="stChatMessageAvatarUser"] svg,
    [data-testid="stChatMessageAvatarUser"] svg path {
        fill: #E0F2FE !important;
        color: #E0F2FE !important;
        stroke: #E0F2FE !important;
    }

    /* Assistant avatar — teal background, lighter teal border + icon */
    [data-testid="chatAvatarIcon-assistant"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0369A1 !important;
        border: 2px solid #38BDF8 !important;
        color: #E0F2FE !important;
    }
    [data-testid="chatAvatarIcon-assistant"] svg,
    [data-testid="chatAvatarIcon-assistant"] svg path,
    [data-testid="stChatMessageAvatarAssistant"] svg,
    [data-testid="stChatMessageAvatarAssistant"] svg path {
        fill: #E0F2FE !important;
        color: #E0F2FE !important;
        stroke: #E0F2FE !important;
    }

    /* Sidebar secondary buttons → subtle clickable link-row style.
       Clear conversation uses type="primary" and is unaffected. */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(30, 58, 95, 0.55) !important;
        border-radius: 6px !important;
        color: #9CA3AF !important;
        text-align: left !important;
        font-size: 0.79rem !important;
        font-weight: 400 !important;
        padding: 5px 10px !important;
        margin: 1px 0 !important;
        min-height: unset !important;
        line-height: 1.45 !important;
        transition: border-color 0.15s, color 0.15s, background 0.15s !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(22, 27, 39, 0.9) !important;
        border-color: #2563EB !important;
        color: #E6EDF3 !important;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Bedrock LLM callable (injected into orchestrator) ─────────────────────────
@st.cache_resource
def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def bedrock_call(system: str, user_message: str) -> str:
    """Call Bedrock Amazon Nova Lite via Converse API and return the text response."""
    bedrock = _get_bedrock_client()
    response = bedrock.converse(
        modelId="us.amazon.nova-lite-v1:0",
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 12px">'
        '<span style="font-size:1.5rem">🌪️</span>'
        '<span style="font-size:1.1rem;font-weight:700;color:#E6EDF3;margin-left:8px">'
        'Hazard Risk Agent</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("County-level disaster risk intelligence powered by ML + RAG")
    st.divider()

    tool_mode = st.selectbox(
        "Tool mode",
        options=["Auto (Recommended)", "Analytics (/query)", "Document Q&A (/ask)", "Risk Prediction (/predict)"],
        help="Auto selects the best tool based on your question",
    )

    with st.expander("⚙️ Advanced settings"):
        top_k = st.slider("RAG context chunks", min_value=3, max_value=10, value=5,
                          help="Number of document chunks retrieved for /ask questions")
        row_limit = st.slider("Max analytics rows", min_value=5, max_value=50, value=15,
                              help="Maximum rows returned from Athena queries")

    st.divider()
    # type="primary" keeps this button visually distinct from the example buttons below
    if st.button("🗑️ Clear conversation", use_container_width=True, type="primary"):
        clear_history()
        st.rerun()

    st.divider()
    st.markdown(
        '<p style="color:#4B5563;font-size:0.75rem;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:6px">Example questions</p>',
        unsafe_allow_html=True,
    )
    _EXAMPLES = [
        ("🔎", "Which states had the most FEMA declarations in 2020–2023?"),
        ("🔎", "Which counties had the highest tornado fatalities?"),
        ("🤖", "Predict the risk tier for Miami-Dade County, Florida"),
        ("📈", "Show year-over-year tornado event trends since 2010"),
        ("📚", "What is the NRI expected loss methodology?"),
    ]
    for i, (icon, ex) in enumerate(_EXAMPLES):
        if st.button(f"{icon}  {ex}", key=f"sidebar_ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:2px">Hazard Risk Intelligence Agent</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#6B7280;font-size:0.9rem;margin-top:0">Ask questions about county-level disaster risk. '
    'The agent combines predictive ML, structured analytics, and document retrieval.</p>',
    unsafe_allow_html=True,
)

init_session()

# Consume any pending question injected by sidebar example buttons
_pending = st.session_state.pop("pending_question", None)

# Welcome state — shown only before the first message
if not st.session_state.get("messages"):
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
            <div style="text-align:center;padding:2rem 1rem">
              <div style="font-size:3rem;margin-bottom:0.5rem">🌍</div>
              <h3 style="color:#E6EDF3;margin:0 0 0.4rem">Ready to analyze hazard risk</h3>
              <p style="color:#6B7280;font-size:0.9rem;margin:0">
                Ask a question in the box below, or use an example from the sidebar.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#4B5563;font-size:0.78rem;margin-top:4px">'
            'Click an example in the sidebar to get started.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # Three quick-start cards
    c1, c2, c3 = st.columns(3)
    _QUICKSTART = [
        ("🔎", "Analytics", "Which counties saw the largest increase in flood events 2015–2023?"),
        ("🤖", "ML Prediction", "Show top 10 counties by predicted risk and property damage"),
        ("📚", "Document Q&A", "Why are coastal counties more vulnerable to hurricanes?"),
    ]
    for col, (icon, label, question) in zip([c1, c2, c3], _QUICKSTART):
        with col:
            st.markdown(
                f'<div style="background:#161B27;border:1px solid #1E2535;border-radius:8px;'
                f'padding:14px 16px;min-height:100px">'
                f'<div style="font-size:1.2rem">{icon}</div>'
                f'<div style="color:#4F9CF9;font-size:0.72rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin:4px 0">{label}</div>'
                f'<div style="color:#9CA3AF;font-size:0.82rem">{question}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

render_history()

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a hazard risk question…") or _pending

if prompt:
    add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Analyzing your question…", expanded=False) as _status:
            _status.write("Routing to tools…")
            try:
                result = run_agent(
                    question=prompt,
                    top_k=top_k,
                    limit=row_limit,
                    bedrock_call_fn=bedrock_call,
                    pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
                )
                _status.write("Synthesizing response…")
                _status.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                _status.update(label="Error", state="error", expanded=False)
                render_error(str(exc))
                st.stop()

        tools_used = result.get("tool_used", [])
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        tool_outputs = result.get("tool_outputs", {})

        # Tool badges first — routing context before the answer
        if tools_used:
            render_tool_badges(
                tools_used,
                intent=tool_outputs.get("query", {}).get("intent", ""),
            )

        # Detect "no data found" answers — render as warning, not prose
        _NO_DATA_PHRASES = ("no data was found", "no matching records", "no results")
        if answer and any(p in answer.lower() for p in _NO_DATA_PHRASES):
            render_no_data(answer)
        elif answer:
            st.markdown(answer)

        # Analytics table + chart
        if "query" in tools_used and "results" in tool_outputs.get("query", {}):
            query_out = tool_outputs["query"]
            results_data = query_out.get("results", [])
            sql = query_out.get("sql_executed", "")
            intent = query_out.get("intent", "")

            if results_data:
                if "trend" in intent or intent == "hazard_trend_by_year":
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

        # Save to conversation history (including rich content for re-render)
        add_message(
            "assistant",
            answer or "See results above.",
            metadata={
                "tool_used": tools_used,
                "intent": tool_outputs.get("query", {}).get("intent", ""),
                "tool_outputs": tool_outputs,
                "sources": sources,
            },
        )
