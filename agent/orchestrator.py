"""
Agent orchestrator: top-level loop that routes questions,
calls the appropriate tools, and synthesizes a unified response.

This is the single entry point used by both the FastAPI /agent endpoint
and the Streamlit app.
"""
import logging

from agent.router import RoutingDecision, route
from analytics.query_engine import run_query
from ml.inference.inference_service import predict_from_endpoint
from rag.prompts.ask_template import SYSTEM_PROMPT, build_ask_prompt, build_citations
from rag.retrieval.retrieve import retrieve_similar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_agent(
    question: str,
    top_k: int = 5,
    limit: int = 20,
    bedrock_call_fn=None,
    opensearch_endpoint: str = None,
    sagemaker_endpoint: str = "hazard-risk-model",
) -> dict:
    """
    Full agent execution loop:
    1. Route question to tool(s)
    2. Execute tool calls
    3. Synthesize response

    Args:
        question: User's natural-language question.
        top_k: Number of RAG chunks to retrieve.
        limit: Max rows for analytics queries.
        bedrock_call_fn: Callable(system, user_message) → str for LLM synthesis.
                         Must be injected (avoids circular import with app.py).
        opensearch_endpoint: Override for OpenSearch; reads env var if None.
        sagemaker_endpoint: SageMaker endpoint name for /predict calls.

    Returns:
        dict with keys: answer, tool_used, data, sources, routing_reason
    """
    decision: RoutingDecision = route(question)
    logger.info("Routing: %s → tools=%s", question[:80], decision.tools)

    result = {
        "question": question,
        "routing": {
            "tools": decision.tools,
            "reason": decision.reasoning,
            "is_hybrid": decision.is_hybrid,
        },
    }

    tool_outputs = {}

    # ── Execute each tool ─────────────────────────────────────────────────────
    if "predict" in decision.tools:
        try:
            pred = predict_from_endpoint(
                payload={"instances": [{}]},
                endpoint_name=sagemaker_endpoint,
            )
            tool_outputs["predict"] = pred
        except Exception as exc:
            logger.warning("Predict tool failed: %s", exc)
            tool_outputs["predict"] = {"error": str(exc)}

    if "query" in decision.tools:
        try:
            query_result = run_query(question, limit=limit)
            tool_outputs["query"] = query_result
        except Exception as exc:
            logger.warning("Query tool failed: %s", exc)
            tool_outputs["query"] = {"error": str(exc)}

    if "ask" in decision.tools:
        try:
            chunks = retrieve_similar(
                question=question,
                k=top_k,
                opensearch_endpoint=opensearch_endpoint,
            )
            tool_outputs["ask"] = {
                "chunks": chunks,
                "citations": build_citations(chunks),
            }
        except Exception as exc:
            logger.warning("Ask/retrieve tool failed: %s", exc)
            tool_outputs["ask"] = {"error": str(exc), "chunks": [], "citations": []}

    result["tool_outputs"] = tool_outputs

    # ── Synthesize answer ─────────────────────────────────────────────────────
    if bedrock_call_fn and "ask" in decision.tools:
        chunks = tool_outputs["ask"].get("chunks", [])

        # For hybrid queries, prepend structured data as additional context
        if "query" in decision.tools and "results" in tool_outputs.get("query", {}):
            rows = tool_outputs["query"]["results"][:10]
            structured_context = _format_table_as_text(rows)
            augmented_question = (
                f"{question}\n\nStructured analytics results:\n{structured_context}"
            )
        else:
            augmented_question = question

        user_message = build_ask_prompt(augmented_question, chunks)
        try:
            answer = bedrock_call_fn(SYSTEM_PROMPT, user_message)
        except Exception as exc:
            logger.error("LLM synthesis failed: %s", exc)
            answer = _fallback_answer(tool_outputs, decision.tools)

        result["answer"] = answer
        result["sources"] = tool_outputs["ask"].get("citations", [])
    else:
        # No LLM synthesis — return raw structured data
        result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = []

    result["tool_used"] = decision.tools
    return result


def _format_table_as_text(rows: list) -> str:
    """Convert a list of dicts (Athena results) to a readable text table."""
    if not rows:
        return "No results."
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def _fallback_answer(tool_outputs: dict, tools: list) -> str:
    """Return a plain-text summary when LLM synthesis is unavailable."""
    parts = []
    if "query" in tools and "results" in tool_outputs.get("query", {}):
        rows = tool_outputs["query"]["results"]
        parts.append(f"Analytics results ({len(rows)} rows returned):")
        parts.append(_format_table_as_text(rows[:5]))
    if "predict" in tools and "predictions" in tool_outputs.get("predict", {}):
        parts.append(f"Risk predictions: {tool_outputs['predict']['predictions']}")
    if "ask" in tools and tool_outputs.get("ask", {}).get("error"):
        parts.append(f"Retrieval error: {tool_outputs['ask']['error']}")
    return "\n\n".join(parts) if parts else "No results available."
