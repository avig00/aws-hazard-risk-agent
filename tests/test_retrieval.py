"""
Unit tests for rag/retrieval/retrieve.py

Tests use mocked boto3 and Pinecone clients — no AWS or Pinecone connectivity required.
"""
from unittest.mock import MagicMock, patch

import pytest

from rag.retrieval.retrieve import embed_query, retrieve_similar


MOCK_EMBEDDING = [0.1] * 1536

MOCK_CONFIG = {
    "rag": {
        "embedding_model": "amazon.titan-embed-text-v1",
        "embedding_dimensions": 1536,
        "vector_index": "hazard-risk-docs",
        "top_k": 5,
        "min_score": 0.4,
        "region": "us-east-1",
    },
    "pinecone": {
        "index_name": "hazard-risk-docs",
        "api_key": "mock-api-key",
    },
}


def _make_match(score, text, **meta):
    """Build a mock Pinecone match object."""
    m = MagicMock()
    m.score = score
    m.metadata = {"text": text, **meta}
    return m


MOCK_MATCHES = [
    _make_match(
        0.92,
        "Harris County Texas has experienced significant flood events.",
        source="s3://hazard/docs/fema_report_2022.pdf",
        hazard_type="flood",
        section="chunk_3",
    ),
    _make_match(
        0.78,
        "Miami-Dade County faces elevated hurricane risk due to coastal exposure.",
        source="s3://hazard/docs/nri_documentation.pdf",
        hazard_type="hurricane",
        section="chunk_7",
    ),
    _make_match(
        0.30,  # Below default min_score of 0.4 → should be filtered
        "Low relevance document.",
        source="s3://hazard/docs/misc.txt",
        hazard_type="general",
    ),
]


@patch("rag.retrieval.retrieve.boto3")
def test_embed_query_returns_vector(mock_boto3):
    """embed_query should return a list of floats."""
    import json

    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = {
        "body": MagicMock(
            read=lambda: json.dumps({"embedding": MOCK_EMBEDDING}).encode()
        ),
    }

    result = embed_query("flood risk in Texas", "amazon.titan-embed-text-v1", "us-east-1")
    assert isinstance(result, list)
    assert len(result) == 1536


@patch("pinecone.Pinecone")
@patch("rag.retrieval.retrieve.embed_query")
def test_retrieve_similar_returns_chunks(mock_embed, mock_pinecone_class):
    """retrieve_similar should return filtered, scored chunks."""
    mock_embed.return_value = MOCK_EMBEDDING

    mock_pc = MagicMock()
    mock_pinecone_class.return_value = mock_pc
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_results = MagicMock()
    mock_results.matches = MOCK_MATCHES
    mock_index.query.return_value = mock_results

    results = retrieve_similar(
        question="Which counties are most flood-prone?",
        k=5,
        min_score=0.4,
        config=MOCK_CONFIG,
        pinecone_api_key="mock-api-key",
    )

    # Third chunk (score=0.30) should be filtered out by min_score
    assert len(results) == 2
    assert results[0]["score"] == 0.92
    assert results[1]["score"] == 0.78
    assert "text" in results[0]
    assert "metadata" in results[0]


@patch("pinecone.Pinecone")
@patch("rag.retrieval.retrieve.embed_query")
def test_retrieve_returns_sorted_by_score(mock_embed, mock_pinecone_class):
    """Results should be ordered by score descending."""
    mock_embed.return_value = MOCK_EMBEDDING

    mock_pc = MagicMock()
    mock_pinecone_class.return_value = mock_pc
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_results = MagicMock()
    mock_results.matches = MOCK_MATCHES
    mock_index.query.return_value = mock_results

    results = retrieve_similar(
        question="hurricane risk",
        k=5,
        min_score=0.0,  # No filtering
        config=MOCK_CONFIG,
        pinecone_api_key="mock-api-key",
    )

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_raises_without_api_key():
    """Should raise ValueError when no API key is configured."""
    config_no_key = {
        "rag": {**MOCK_CONFIG["rag"]},
        "pinecone": {"index_name": "hazard-risk-docs", "api_key": ""},
    }

    with pytest.raises(ValueError, match="Pinecone API key required"):
        retrieve_similar(
            question="test",
            config=config_no_key,
            pinecone_api_key=None,
        )
