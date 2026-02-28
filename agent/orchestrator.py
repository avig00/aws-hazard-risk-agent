"""
Agent orchestrator: top-level loop that routes questions,
calls the appropriate tools, and synthesizes a unified response.

This is the single entry point used by both the FastAPI /agent endpoint
and the Streamlit app.
"""
import logging

from agent.router import RoutingDecision, route
from analytics.query_engine import run_query, run_tag_query
from ml.inference.inference_service import predict_from_endpoint
from rag.prompts.ask_template import SYSTEM_PROMPT, build_ask_prompt, build_citations
from rag.prompts.tag_template import TAG_SYSTEM_PROMPT, build_tag_prompt
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
            if bedrock_call_fn:
                # TAG: Athena results + LLM synthesis in one step
                query_result = run_tag_query(
                    question=question,
                    synthesize_fn=bedrock_call_fn,
                    limit=limit,
                )
            else:
                # No LLM available — raw Athena results only
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
    query_out = tool_outputs.get("query", {})
    ask_out = tool_outputs.get("ask", {})

    if "query" in decision.tools and "ask" not in decision.tools:
        # Pure query route: TAG already ran inside run_tag_query().
        # Use its answer directly — no second LLM call needed.
        result["answer"] = query_out.get("answer", _fallback_answer(tool_outputs, decision.tools))
        result["sources"] = []

    elif "ask" in decision.tools and "query" not in decision.tools:
        # Pure RAG route: standard ask_template synthesis.
        if bedrock_call_fn:
            chunks = ask_out.get("chunks", [])
            user_message = build_ask_prompt(question, chunks)
            try:
                result["answer"] = bedrock_call_fn(SYSTEM_PROMPT, user_message)
            except Exception as exc:
                logger.error("RAG synthesis failed: %s", exc)
                result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        else:
            result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = ask_out.get("citations", [])

    elif "query" in decision.tools and "ask" in decision.tools:
        # Hybrid route: combine TAG answer (structured data) with RAG chunks
        # for a single unified response that draws on both sources.
        if bedrock_call_fn:
            tag_answer = query_out.get("answer", "")
            rag_chunks = ask_out.get("chunks", [])

            # Augment the RAG prompt with the TAG narrative as additional context
            augmented_question = question
            if tag_answer:
                augmented_question = (
                    f"{question}\n\n"
                    f"[Structured analytics answer]\n{tag_answer}"
                )

            user_message = build_ask_prompt(augmented_question, rag_chunks)
            try:
                result["answer"] = bedrock_call_fn(SYSTEM_PROMPT, user_message)
            except Exception as exc:
                logger.error("Hybrid synthesis failed: %s", exc)
                result["answer"] = tag_answer or _fallback_answer(tool_outputs, decision.tools)
        else:
            result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = ask_out.get("citations", [])

    else:
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
