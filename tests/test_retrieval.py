"""
Unit tests for rag/retrieval/retrieve.py

Tests use mocked boto3 and OpenSearch clients — no AWS connectivity required.
"""
from unittest.mock import MagicMock, patch

import pytest

from rag.retrieval.retrieve import embed_query, retrieve_similar


MOCK_EMBEDDING = [0.1] * 1536

MOCK_OS_RESPONSE = {
    "hits": {
        "hits": [
            {
                "_score": 0.92,
                "_source": {
                    "text": "Harris County Texas has experienced significant flood events.",
                    "metadata": {
                        "source": "s3://hazard/docs/fema_report_2022.pdf",
                        "hazard_type": "flood",
                        "section": "chunk_3",
                    },
                },
            },
            {
                "_score": 0.78,
                "_source": {
                    "text": "Miami-Dade County faces elevated hurricane risk due to coastal exposure.",
                    "metadata": {
                        "source": "s3://hazard/docs/nri_documentation.pdf",
                        "hazard_type": "hurricane",
                        "section": "chunk_7",
                    },
                },
            },
            {
                "_score": 0.30,  # Below default min_score of 0.4 → should be filtered
                "_source": {
                    "text": "Low relevance document.",
                    "metadata": {"source": "s3://hazard/docs/misc.txt", "hazard_type": "general"},
                },
            },
        ]
    }
}

MOCK_CONFIG = {
    "rag": {
        "embedding_model": "amazon.titan-embed-text-v1",
        "embedding_dimensions": 1536,
        "vector_index": "hazard-index",
        "top_k": 5,
        "min_score": 0.4,
        "region": "us-east-1",
    },
    "opensearch": {
        "index_name": "hazard-index",
        "collection_endpoint": "https://mock.us-east-1.aoss.amazonaws.com",
    },
}


@patch("rag.retrieval.retrieve.boto3")
def test_embed_query_returns_vector(mock_boto3):
    """embed_query should return a list of floats."""
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = {
        "body": MagicMock(read=lambda: b'{"embedding": ' + str(MOCK_EMBEDDING).encode() + b"}"),
    }

    import json
    mock_client.invoke_model.return_value["body"].read = lambda: json.dumps(
        {"embedding": MOCK_EMBEDDING}
    ).encode()

    result = embed_query("flood risk in Texas", "amazon.titan-embed-text-v1", "us-east-1")
    assert isinstance(result, list)
    assert len(result) == 1536


@patch("rag.retrieval.retrieve._get_os_client")
@patch("rag.retrieval.retrieve.embed_query")
def test_retrieve_similar_returns_chunks(mock_embed, mock_os_client):
    """retrieve_similar should return filtered, scored chunks."""
    mock_embed.return_value = MOCK_EMBEDDING

    mock_client = MagicMock()
    mock_client.search.return_value = MOCK_OS_RESPONSE
    mock_os_client.return_value = mock_client

    results = retrieve_similar(
        question="Which counties are most flood-prone?",
        k=5,
        min_score=0.4,
        config=MOCK_CONFIG,
        opensearch_endpoint="https://mock.us-east-1.aoss.amazonaws.com",
    )

    # Third chunk (score=0.30) should be filtered out by min_score
    assert len(results) == 2
    assert results[0]["score"] == 0.92
    assert results[1]["score"] == 0.78
    assert "text" in results[0]
    assert "metadata" in results[0]


@patch("rag.retrieval.retrieve._get_os_client")
@patch("rag.retrieval.retrieve.embed_query")
def test_retrieve_returns_sorted_by_score(mock_embed, mock_os_client):
    """Results should be ordered by score descending (OpenSearch guarantees this)."""
    mock_embed.return_value = MOCK_EMBEDDING
    mock_client = MagicMock()
    mock_client.search.return_value = MOCK_OS_RESPONSE
    mock_os_client.return_value = mock_client

    results = retrieve_similar(
        question="hurricane risk",
        k=5,
        min_score=0.0,  # No filtering
        config=MOCK_CONFIG,
        opensearch_endpoint="https://mock.us-east-1.aoss.amazonaws.com",
    )

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_raises_without_endpoint():
    """Should raise ValueError when no endpoint is configured."""
    config_no_endpoint = {
        "rag": {**MOCK_CONFIG["rag"]},
        "opensearch": {"index_name": "hazard-index", "collection_endpoint": ""},
    }

    with pytest.raises(ValueError, match="OpenSearch endpoint required"):
        retrieve_similar(
            question="test",
            config=config_no_endpoint,
            opensearch_endpoint=None,
        )
