"""
End-to-end agent routing tests.

Uses mocked tool functions — no AWS connectivity required.
Tests that the router + orchestrator correctly direct questions
to the right tool(s) and return properly structured responses.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.router import RoutingDecision, route


# ── router ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_tools", [
    ("Show top 10 counties by predicted risk score", ["predict"]),
    ("Which counties have the highest flood event count since 2015?", ["query"]),
    ("Why are coastal counties more vulnerable to hurricanes?", ["ask"]),
    ("Show top 10 counties by predicted risk and property damage", ["predict", "query"]),
    ("What are the trend results and why do they show increasing floods?", ["query", "ask"]),
])
def test_router_tools(question, expected_tools):
    decision = route(question)
    for tool in expected_tools:
        assert tool in decision.tools, (
            f"Expected tool '{tool}' in routing for: '{question}', got {decision.tools}"
        )


def test_router_returns_routing_decision():
    result = route("Top counties by risk")
    assert isinstance(result, RoutingDecision)
    assert len(result.tools) >= 1
    assert result.reasoning != ""


def test_router_hybrid_flag():
    decision = route("Show top counties by predicted risk and property damage")
    assert decision.is_hybrid is True


def test_router_non_hybrid():
    decision = route("Why are floods increasing?")
    assert decision.is_hybrid is False


# ── orchestrator (with mocked tools) ─────────────────────────────────────────

@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_query")
@patch("agent.orchestrator.predict_risk")
def test_orchestrator_query_route(mock_predict, mock_query, mock_retrieve):
    """Analytics question → only /query tool is called."""
    mock_query.return_value = {
        "results": [{"county_name": "Harris County", "avg_expected_loss": 52000}],
        "sql_executed": "SELECT ...",
        "intent": "top_counties_by_risk",
        "row_count": 1,
        "tool": "query",
    }

    from agent.orchestrator import run_agent
    result = run_agent(
        question="Top 10 counties by risk from 2015 to 2023",
        bedrock_call_fn=None,
    )

    mock_query.assert_called_once()
    mock_predict.assert_not_called()
    assert "query" in result["tool_used"]
    assert "results" in result["tool_outputs"]["query"]


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_query")
def test_orchestrator_ask_route(mock_query, mock_retrieve):
    """Open-ended question → only /ask tool is called."""
    mock_retrieve.return_value = [
        {
            "text": "Harris County has high flood risk.",
            "score": 0.85,
            "metadata": {"source": "fema_report.pdf", "hazard_type": "flood"},
        }
    ]

    mock_bedrock = MagicMock(return_value="Harris County is highly flood-prone due to...")

    from agent.orchestrator import run_agent
    result = run_agent(
        question="Why is Harris County so prone to flooding?",
        bedrock_call_fn=mock_bedrock,
    )

    mock_retrieve.assert_called_once()
    mock_query.assert_not_called()
    assert "ask" in result["tool_used"]
    assert result.get("answer") is not None


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_query")
@patch("agent.orchestrator.predict_risk")
def test_orchestrator_returns_structured_response(mock_predict, mock_query, mock_retrieve):
    """Response must always include required keys."""
    mock_query.return_value = {"results": [], "sql_executed": "", "intent": "top_counties_by_risk"}
    mock_retrieve.return_value = []

    from agent.orchestrator import run_agent
    result = run_agent(
        question="Show me county risk trends",
        bedrock_call_fn=None,
    )

    required_keys = {"question", "routing", "tool_outputs", "tool_used"}
    assert required_keys.issubset(result.keys())
