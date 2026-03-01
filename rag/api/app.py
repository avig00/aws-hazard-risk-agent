"""
Unified Agent API — FastAPI application exposing four endpoints:

  POST /predict  — ML risk prediction via SageMaker endpoint
  POST /ask      — RAG-based Q&A using Pinecone + Bedrock Claude
  POST /query    — Governed NL→SQL + TAG synthesis (Table Augmented Generation)
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
    synthesize: bool = True   # Set False to skip TAG and return raw Athena rows


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
    TAG pipeline: governed NL→SQL over Athena Gold layer, then Bedrock Claude
    synthesizes a narrative analyst answer from the query results.

    Set synthesize=False in the request body to skip LLM synthesis and return
    raw Athena rows (useful for programmatic consumers or cost control).
    """
    from analytics.query_engine import run_query, run_tag_query
    config = load_rag_config()

    try:
        if request.synthesize:
            # Partial application of call_bedrock_claude with config baked in
            def _synthesize(system: str, user_msg: str) -> str:
                return call_bedrock_claude(system, user_msg, config)

            result = run_tag_query(
                question=request.question,
                synthesize_fn=_synthesize,
                limit=request.limit,
            )
        else:
            result = run_query(request.question, limit=request.limit)

        return result
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/agent")
def agent(request: AgentRequest):
    """
    Top-level agent endpoint: routes the question through the full agent orchestrator,
    combining /predict, /query (with TAG synthesis), and /ask tools as needed.
    """
    from agent.orchestrator import run_agent
    config = load_rag_config()

    def _bedrock_call(system: str, user_msg: str) -> str:
        return call_bedrock_claude(system, user_msg, config)

    try:
        result = run_agent(
            question=request.question,
            top_k=request.top_k,
            bedrock_call_fn=_bedrock_call,
        )
    except Exception as exc:
        logger.error("Agent failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return result
