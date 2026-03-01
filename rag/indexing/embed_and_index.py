"""
Embed document chunks using Amazon Titan Embeddings (via Bedrock)
and bulk-upsert them into Pinecone.

Pinecone free tier: 1 index, 100K vectors, 5 GB storage.
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

from rag.indexing.chunk_documents import build_corpus_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_pinecone_index(api_key: str, index_name: str, dimensions: int):
    """Return a Pinecone Index, creating it if it doesn't exist."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (%d dims, cosine)", index_name, dimensions)
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for Pinecone index to be ready...")
            time.sleep(3)

    return pc.Index(index_name)


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


def upsert_to_pinecone(index, embedded_chunks: list, batch_size: int = 100) -> dict:
    """Upsert all embedded chunks into a Pinecone index in batches."""
    vectors = [
        {
            "id": chunk["id"],
            "values": chunk["embedding"],
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        }
        for chunk in embedded_chunks
    ]

    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)
        logger.info("Upserted %d / %d vectors", total_upserted, len(vectors))

    logger.info("Pinecone upsert complete: %d vectors", total_upserted)
    return {"indexed": total_upserted, "errors": 0}


def run_indexing(config: dict = None, pinecone_api_key: str = None) -> dict:
    """
    Full indexing pipeline:
    1. Chunk all corpus documents
    2. Embed each chunk via Bedrock Titan
    3. Upsert into Pinecone

    pinecone_api_key can also be set via the PINECONE_API_KEY env var.
    """
    if config is None:
        config = load_config()

    rag_cfg = config["rag"]
    pinecone_cfg = config.get("pinecone", {})

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

    region = rag_cfg.get("region", "us-east-1")
    index_name = pinecone_cfg.get("index_name", rag_cfg.get("vector_index", "hazard-risk-docs"))
    dimensions = rag_cfg.get("embedding_dimensions", 1536)
    model_id = rag_cfg["embedding_model"]

    # Clients
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    pinecone_index = _get_pinecone_index(api_key, index_name, dimensions)

    # Build corpus chunks
    chunks = build_corpus_chunks(config)
    if not chunks:
        logger.warning("No chunks produced — check corpus S3 path")
        return {"indexed": 0, "errors": 0}

    # Embed
    embedded_chunks = embed_chunks_batch(chunks, model_id, bedrock)

    # Upsert
    result = upsert_to_pinecone(pinecone_index, embedded_chunks)
    return result


if __name__ == "__main__":
    result = run_indexing()
    print(f"Indexing result: {result}")
