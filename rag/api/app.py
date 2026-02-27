"""
Unified Agent API — FastAPI application exposing four endpoints:

  POST /predict  — ML risk prediction via SageMaker endpoint
  POST /ask      — RAG-based Q&A using OpenSearch + Bedrock Claude
  POST /query    — Governed NL→SQL analytics over Athena Gold layer
  POST /agent    — Orchestrates all tools; routes or combines as needed
  GET  /health   — Health check
"""
import json
import logging
import os
from pathlib import Path

import boto3
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.inference.inference_service import predict_from_endpoint
from rag.prompts.ask_template import SYSTEM_PROMPT, build_ask_prompt, build_citations
from rag.retrieval.retrieve import retrieve_similar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT", "hazard-risk-model")

app = FastAPI(
    title="Hazard Risk Intelligence Agent",
    description="Agentic AI for county-level disaster risk Q&A",
    version="1.0.0",
)


def load_rag_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Request / Response models ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: dict
    county_id: str = ""


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryRequest(BaseModel):
    question: str
    limit: int = 20


class AgentRequest(BaseModel):
    question: str
    top_k: int = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

def call_bedrock_claude(system: str, user_message: str, config: dict) -> str:
    """Invoke Bedrock Claude and return the text response."""
    llm_cfg = config.get("llm", {})
    model_id = llm_cfg.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
    region = config["rag"].get("region", "us-east-1")
    max_tokens = llm_cfg.get("max_tokens", 1024)

    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_message}],
    })

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def classify_intent(question: str) -> str:
    """
    Simple keyword-based intent classifier to route agent requests.
    Returns one of: 'predict', 'query', 'ask', 'hybrid'
    """
    q = question.lower()

    predict_signals = ["predicted risk", "risk score", "risk prediction", "forecast risk",
                       "predicted loss", "model score"]
    query_signals = ["top", "highest", "lowest", "trend", "average", "compare", "ranking",
                     "how many", "count", "total", "list", "show me counties"]
    ask_signals = ["why", "what is", "explain", "describe", "how does", "what are the causes",
                   "documentation", "report", "narrative"]

    is_predict = any(s in q for s in predict_signals)
    is_query = any(s in q for s in query_signals)
    is_ask = any(s in q for s in ask_signals)

    if is_predict and is_query:
        return "hybrid"
    if is_predict:
        return "predict"
    if is_query:
        return "query"
    return "ask"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Call the SageMaker endpoint and return a county risk score + bucket.
    """
    try:
        result = predict_from_endpoint(
            payload=request.features,
            endpoint_name=ENDPOINT_NAME,
        )
        return {
            "county_id": request.county_id,
            "prediction": result["predictions"][0] if result["predictions"] else None,
            "risk_bucket": result["risk_buckets"][0] if result["risk_buckets"] else None,
            "endpoint": result["endpoint_name"],
        }
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask")
def ask(request: AskRequest):
    """
    RAG pipeline: embed question → retrieve chunks → call Bedrock Claude → return answer.
    """
    config = load_rag_config()
    try:
        chunks = retrieve_similar(question=request.question, k=request.top_k, config=config)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    user_message = build_ask_prompt(request.question, chunks)
    try:
        answer = call_bedrock_claude(SYSTEM_PROMPT, user_message, config)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    citations = build_citations(chunks)
    return {
        "answer": answer,
        "sources": citations,
        "chunks_retrieved": len(chunks),
        "tool": "ask",
    }


@app.post("/query")
def query(request: QueryRequest):
    """
    Governed NL→SQL: parse intent, compile SQL template, execute via Athena.
    """
    # Import here to avoid circular dependency at startup
    from analytics.query_engine import run_query
    try:
        result = run_query(request.question, limit=request.limit)
        return result
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent")
def agent(request: AgentRequest):
    """
    Top-level agent endpoint: classify intent, call appropriate tool(s),
    synthesize a unified grounded response.
    """
    intent = classify_intent(request.question)
    logger.info("Agent routing: '%s' → intent=%s", request.question[:80], intent)

    response = {"question": request.question, "intent": intent}

    if intent == "predict":
        result = predict(PredictRequest(features={}, county_id=""))
        response.update({"tool": "predict", "data": result})

    elif intent == "query":
        result = query(QueryRequest(question=request.question))
        response.update({"tool": "query", "data": result})

    elif intent == "hybrid":
        # Call both predict (batch scores) and query (structured analytics), merge
        query_result = query(QueryRequest(question=request.question))
        response.update({
            "tool": "hybrid",
            "analytics": query_result,
        })

    else:
        # Default: RAG ask
        ask_result = ask(AskRequest(question=request.question, top_k=request.top_k))
        response.update({"tool": "ask", "data": ask_result})

    return response
