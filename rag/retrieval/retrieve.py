"""
Vector retrieval: embed a user query and find the top-k most similar
document chunks in Pinecone using cosine similarity search.

Pinecone free tier: 1 index, 100K vectors, 5 GB storage — sufficient for
the FEMA/NOAA hazard document corpus used in this project.
"""
import json
import logging
import os
from pathlib import Path

import boto3
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def embed_query(text: str, model_id: str, region: str) -> list:
    """Embed a query string using Amazon Titan Embeddings via Bedrock."""
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(response["body"].read())["embedding"]


def retrieve_similar(
    question: str,
    k: int = None,
    min_score: float = None,
    config: dict = None,
    pinecone_api_key: str = None,
) -> list:
    """
    Embed the question and return the top-k most relevant document chunks from Pinecone.

    Args:
        question: Natural-language question from the user.
        k: Number of results (defaults to config top_k).
        min_score: Minimum cosine similarity score to include.
        config: RAG config dict (loaded from rag_config.yml if not provided).
        pinecone_api_key: Pinecone API key (falls back to PINECONE_API_KEY env var).

    Returns:
        List of dicts: [{text, score, metadata}]
    """
    from pinecone import Pinecone

    if config is None:
        config = load_config()

    rag_cfg = config["rag"]
    pinecone_cfg = config.get("pinecone", {})

    region = rag_cfg.get("region", "us-east-1")
    model_id = rag_cfg["embedding_model"]
    index_name = pinecone_cfg.get("index_name", rag_cfg.get("vector_index", "hazard-risk-docs"))
    top_k = k or rag_cfg.get("top_k", 5)
    score_threshold = min_score if min_score is not None else rag_cfg.get("min_score", 0.0)

    api_key = (
        pinecone_api_key
        or os.environ.get("PINECONE_API_KEY")
        or pinecone_cfg.get("api_key", "")
    )
    if not api_key:
        raise ValueError(
            "Pinecone API key required. Set PINECONE_API_KEY env var "
            "or pinecone.api_key in rag_config.yml"
        )

    query_embedding = embed_query(question, model_id, region)

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    chunks = []
    for match in results.matches:
        score = float(match.score)
        if score < score_threshold:
            continue
        chunks.append({
            "text": match.metadata.get("text", ""),
            "score": round(score, 4),
            "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
        })

    logger.info(
        "Retrieved %d chunks (top score=%.4f)",
        len(chunks),
        chunks[0]["score"] if chunks else 0.0,
    )
    return sorted(chunks, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    sample_q = "Which counties are most at risk for flooding?"
    try:
        chunks = retrieve_similar(sample_q)
        for i, c in enumerate(chunks):
            print(f"[{i+1}] score={c['score']} | {c['text'][:120]}...")
    except Exception as exc:
        print(f"Retrieval failed (expected without live endpoint): {exc}")
