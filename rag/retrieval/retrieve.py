"""
Vector retrieval: embed a user query and find the top-k most similar
document chunks in OpenSearch Serverless using kNN search.
"""
import json
import logging
import os
from pathlib import Path

import boto3
import yaml
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_os_client(endpoint: str, region: str) -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = endpoint.replace("https://", "").rstrip("/")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


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
    opensearch_endpoint: str = None,
) -> list:
    """
    Embed the question and return the top-k most relevant document chunks.

    Args:
        question: Natural-language question from the user.
        k: Number of results (defaults to config top_k).
        min_score: Minimum cosine similarity score to include.
        config: RAG config dict (loaded from rag_config.yml if not provided).
        opensearch_endpoint: Override; also reads OPENSEARCH_ENDPOINT env var.

    Returns:
        List of dicts: [{text, score, metadata}]
    """
    if config is None:
        config = load_config()

    rag_cfg = config["rag"]
    os_cfg = config.get("opensearch", {})

    region = rag_cfg.get("region", "us-east-1")
    model_id = rag_cfg["embedding_model"]
    index_name = os_cfg.get("index_name", rag_cfg["vector_index"])
    top_k = k or rag_cfg.get("top_k", 5)
    score_threshold = min_score if min_score is not None else rag_cfg.get("min_score", 0.0)

    endpoint = (
        opensearch_endpoint
        or os.environ.get("OPENSEARCH_ENDPOINT")
        or os_cfg.get("collection_endpoint", "")
    )
    if not endpoint:
        raise ValueError(
            "OpenSearch endpoint required. Set OPENSEARCH_ENDPOINT env var "
            "or opensearch.collection_endpoint in rag_config.yml"
        )

    query_embedding = embed_query(question, model_id, region)

    os_client = _get_os_client(endpoint, region)
    search_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        },
        "_source": ["text", "metadata"],
    }

    response = os_client.search(index=index_name, body=search_body)
    hits = response["hits"]["hits"]

    results = []
    for hit in hits:
        score = float(hit["_score"])
        if score < score_threshold:
            continue
        results.append({
            "text": hit["_source"]["text"],
            "score": round(score, 4),
            "metadata": hit["_source"].get("metadata", {}),
        })

    logger.info(
        "Retrieved %d chunks (top score=%.4f)",
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return results


if __name__ == "__main__":
    sample_q = "Which counties are most at risk for flooding?"
    try:
        chunks = retrieve_similar(sample_q)
        for i, c in enumerate(chunks):
            print(f"[{i+1}] score={c['score']} | {c['text'][:120]}...")
    except Exception as exc:
        print(f"Retrieval failed (expected without live endpoint): {exc}")
