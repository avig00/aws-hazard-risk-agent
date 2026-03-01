"""
Hazard Risk Intelligence Agent — Streamlit application.

Hosted on Streamlit Cloud. Calls AWS services directly via boto3
using credentials stored in st.secrets.

Run locally:
    streamlit run app.py

Deploy:
    Connect GitHub repo on share.streamlit.io → set AWS secrets in app settings.
"""
import json
import os

import boto3
import streamlit as st

from agent.orchestrator import run_agent
from ui.chat import add_message, clear_history, init_session, render_history
from ui.components import (
    render_analytics_table,
    render_citations,
    render_error,
    render_prediction_card,
    render_sql_expander,
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


# ── Bedrock LLM callable (injected into orchestrator) ─────────────────────────
@st.cache_resource
def _get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def bedrock_call(system: str, user_message: str) -> str:
    """Call Bedrock Claude Sonnet and return the text response."""
    bedrock = _get_bedrock_client()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
    })
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(response["body"].read())["content"][0]["text"]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌪️ Hazard Risk Agent")
    st.caption("County-level disaster risk intelligence powered by ML + RAG")
    st.divider()

    tool_mode = st.selectbox(
        "Tool mode",
        options=["Auto (Recommended)", "Analytics (/query)", "Document Q&A (/ask)", "Risk Prediction (/predict)"],
        help="Auto selects the best tool based on your question",
    )

    st.divider()
    year_min, year_max = st.slider(
        "Year range",
        min_value=2000,
        max_value=2023,
        value=(2015, 2023),
        step=1,
    )

    top_k = st.slider("RAG context chunks", min_value=3, max_value=10, value=5)
    row_limit = st.slider("Max analytics rows", min_value=5, max_value=50, value=15)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        clear_history()
        st.rerun()

    st.divider()
    st.markdown(
        "**Example questions:**\n"
        "- Which counties saw the largest increase in flood events 2015–2023?\n"
        "- Show top 10 counties by predicted risk and property damage\n"
        "- Compare Harris County vs Miami-Dade hazard distributions\n"
        "- Why are coastal counties more vulnerable to hurricanes?\n"
        "- What is the NRI expected loss methodology?"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Hazard Risk Intelligence Agent")
st.caption(
    "Ask questions about county-level disaster risk. "
    "The agent combines predictive ML, structured analytics, and document retrieval."
)

init_session()
render_history()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a hazard risk question…"):
    add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_agent(
                    question=prompt,
                    top_k=top_k,
                    limit=row_limit,
                    bedrock_call_fn=bedrock_call,
                    pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
                )
            except Exception as exc:
                render_error(str(exc))
                st.stop()

        tools_used = result.get("tool_used", [])
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        tool_outputs = result.get("tool_outputs", {})

        # ── Render answer text ─────────────────────────────────────────────
        if answer:
            st.markdown(answer)

        # ── Render structured data from analytics tool ─────────────────────
        if "query" in tools_used and "results" in tool_outputs.get("query", {}):
            query_out = tool_outputs["query"]
            results_data = query_out.get("results", [])
            sql = query_out.get("sql_executed", "")
            intent = query_out.get("intent", "")

            # Choose chart type based on intent
            if "trend" in intent or "year" in intent.lower():
                render_trend_chart(results_data)
            else:
                render_analytics_table(results_data, title="Analytics Results")

            if sql:
                render_sql_expander(sql)

        # ── Render prediction card ─────────────────────────────────────────
        if "predict" in tools_used and "predictions" in tool_outputs.get("predict", {}):
            render_prediction_card(tool_outputs["predict"])

        # ── Render citations ───────────────────────────────────────────────
        if sources:
            render_citations(sources)

        # ── Routing info ───────────────────────────────────────────────────
        routing = result.get("routing", {})
        st.caption(
            f"Tools: `{'` + `'.join(tools_used)}`"
            + (f" | {routing.get('reason', '')}" if routing.get("reason") else "")
        )

        # Save to conversation history
        add_message(
            "assistant",
            answer or "See results above.",
            metadata={
                "tool_used": tools_used,
                "intent": tool_outputs.get("query", {}).get("intent", ""),
            },
        )
