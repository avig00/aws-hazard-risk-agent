"""
Embed document chunks using Amazon Titan Embeddings (via Bedrock)
and bulk-index them into OpenSearch Serverless.

Run this once to build the vector index, then run again incrementally
to add new documents.
"""
import json
import logging
import os
import time
from pathlib import Path

import boto3
import yaml
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers

from rag.indexing.chunk_documents import build_corpus_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_opensearch_client(endpoint: str, region: str = "us-east-1") -> OpenSearch:
    """Build an authenticated OpenSearch Serverless client using SigV4."""
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service="aoss")

    host = endpoint.replace("https://", "").rstrip("/")
    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )
    return client


def create_index_if_not_exists(client: OpenSearch, index_name: str, dimensions: int) -> None:
    """Create a kNN vector index if it doesn't already exist."""
    if client.indices.exists(index=index_name):
        logger.info("Index '%s' already exists — skipping creation", index_name)
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                "text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }

    client.indices.create(index=index_name, body=mapping)
    logger.info("Created kNN index '%s' with %d dimensions", index_name, dimensions)


def embed_text(text: str, model_id: str, bedrock_client) -> list:
    """Call Amazon Titan Embeddings via Bedrock and return the embedding vector."""
    body = json.dumps({"inputText": text})
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def embed_chunks_batch(
    chunks: list,
    model_id: str,
    bedrock_client,
    batch_delay: float = 0.1,
) -> list:
    """
    Embed all chunks, adding the 'embedding' key to each record.
    Adds a small delay between calls to respect Bedrock rate limits.
    """
    logger.info("Embedding %d chunks using %s...", len(chunks), model_id)
    embedded = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk["text"], model_id, bedrock_client)
        embedded.append({**chunk, "embedding": embedding})

        if (i + 1) % 50 == 0:
            logger.info("Embedded %d / %d chunks", i + 1, len(chunks))
            time.sleep(batch_delay)

    return embedded


def bulk_index(client: OpenSearch, index_name: str, embedded_chunks: list) -> dict:
    """Bulk-index all embedded chunks into OpenSearch."""
    actions = [
        {
            "_index": index_name,
            "_id": chunk["id"],
            "_source": {
                "embedding": chunk["embedding"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            },
        }
        for chunk in embedded_chunks
    ]

    success, errors = helpers.bulk(client, actions, raise_on_error=False, chunk_size=200)
    logger.info("Indexed %d documents | Errors: %d", success, len(errors))
    if errors:
        logger.warning("First error: %s", errors[0])
    return {"indexed": success, "errors": len(errors)}


def run_indexing(config: dict = None, opensearch_endpoint: str = None) -> dict:
    """
    Full indexing pipeline:
    1. Chunk all corpus documents
    2. Embed each chunk via Bedrock Titan
    3. Bulk-index into OpenSearch Serverless

    opensearch_endpoint can also be set via the OPENSEARCH_ENDPOINT env var.
    """
    if config is None:
        config = load_config()

    rag_cfg = config["rag"]
    os_cfg = config.get("opensearch", {})

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

    region = rag_cfg.get("region", "us-east-1")
    index_name = os_cfg.get("index_name", rag_cfg["vector_index"])
    dimensions = rag_cfg.get("embedding_dimensions", 1536)
    model_id = rag_cfg["embedding_model"]

    # Clients
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    os_client = get_opensearch_client(endpoint, region)

    create_index_if_not_exists(os_client, index_name, dimensions)

    # Build corpus chunks
    chunks = build_corpus_chunks(config)
    if not chunks:
        logger.warning("No chunks produced — check corpus S3 path")
        return {"indexed": 0, "errors": 0}

    # Embed
    embedded_chunks = embed_chunks_batch(chunks, model_id, bedrock)

    # Index
    result = bulk_index(os_client, index_name, embedded_chunks)
    return result


if __name__ == "__main__":
    result = run_indexing()
    print(f"Indexing result: {result}")
