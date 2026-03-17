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
    ("Show top 10 counties by predicted risk score", ["query"]),   # no county → query-only
    ("Which counties have the highest flood event count since 2015?", ["query"]),
    ("Why are coastal counties more vulnerable to hurricanes?", ["ask"]),
    ("What is the predicted risk and property damage for Harris County TX?", ["predict", "query"]),
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
    # Hybrid predict+query requires a named county — without one the ML endpoint can't run
    decision = route("What is the predicted risk and property damage for Harris County TX?")
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


# ── edge cases & guardrails ────────────────────────────────────────────────────

@patch("agent.orchestrator._check_county_ambiguity", return_value=[])
@patch("agent.orchestrator._fetch_county_features", return_value=({}, "Anytown", ""))
@patch("agent.orchestrator.predict_risk")
def test_12_1_county_not_found(mock_predict, mock_features, mock_ambiguity):
    """12.1 — Non-existent county: returns not-found message, never calls SageMaker."""
    from agent.orchestrator import run_agent
    result = run_agent(
        question="What is the risk for Anytown County?",
        bedrock_call_fn=None,
    )
    mock_predict.assert_not_called()
    assert "not found" in result["answer"].lower()


@patch("agent.orchestrator._check_county_ambiguity",
       return_value=["Alabama", "Arkansas", "Colorado", "Florida", "Georgia",
                     "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
                     "Mississippi", "Missouri", "Montana", "New York",
                     "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
                     "Tennessee", "Texas", "Washington", "West Virginia"])
@patch("agent.orchestrator.predict_risk")
def test_12_7_ambiguous_county_no_state(mock_predict, mock_ambiguity):
    """12.7 — 'Jefferson County' (no state) → clarification listing all states, no SageMaker call."""
    from agent.orchestrator import run_agent
    result = run_agent(
        question="Predict the risk for Jefferson County",
        bedrock_call_fn=None,
    )
    mock_predict.assert_not_called()
    answer = result["answer"]
    # Must mention multiple states in the clarification
    assert "jefferson" in answer.lower()
    assert "alabama" in answer.lower() or "texas" in answer.lower()
    # Must ask user to specify state
    assert "state" in answer.lower() or "specify" in answer.lower()


@patch("agent.orchestrator._check_county_ambiguity", return_value=[])
@patch("agent.orchestrator._fetch_county_features",
       return_value=({"state": "Missouri", "flood_events": 5.0}, "Harrison County", "29085"))
@patch("agent.orchestrator.predict_risk",
       return_value={"risk_tier": "LOW", "probability": {"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.1}})
def test_12_8_state_abbreviation_hint(mock_predict, mock_features, mock_ambiguity):
    """12.8 — 'Harrison County MO': state abbreviation resolves correctly; predicts for Missouri."""
    from agent.orchestrator import run_agent
    result = run_agent(
        question="Predict the risk tier for Harrison County MO",
        bedrock_call_fn=None,
    )
    mock_predict.assert_called_once()
    # _fetch_county_features must be called with state_hint="Missouri" (full name, not abbr)
    _, kwargs = mock_features.call_args
    assert kwargs.get("state_hint", "").lower() == "missouri"
    assert "missouri" in result["answer"].lower() or "harrison" in result["answer"].lower()


@patch("agent.orchestrator._check_county_ambiguity", return_value=[])
@patch("agent.orchestrator._fetch_county_features",
       return_value=({"state": "Missouri", "flood_events": 5.0}, "Harrison County", "29085"))
@patch("agent.orchestrator.predict_risk",
       return_value={"risk_tier": "LOW", "probability": {"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.1}})
def test_12_9_full_state_name_hint(mock_predict, mock_features, mock_ambiguity):
    """12.9 — 'Harrison County Missouri': full state name resolves correctly."""
    from agent.orchestrator import run_agent
    result = run_agent(
        question="Predict the risk tier for Harrison County Missouri",
        bedrock_call_fn=None,
    )
    mock_predict.assert_called_once()
    _, kwargs = mock_features.call_args
    assert kwargs.get("state_hint", "").lower() == "missouri"


@patch("agent.orchestrator._check_county_ambiguity")
@patch("agent.orchestrator._fetch_county_features",
       return_value=({"state": "Texas", "flood_events": 120.0}, "Harris County", "48201"))
@patch("agent.orchestrator.predict_risk",
       return_value={"risk_tier": "HIGH", "probability": {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.8}})
def test_12_10_fips_bypasses_ambiguity(mock_predict, mock_features, mock_ambiguity):
    """12.10 — FIPS code input bypasses ambiguity check entirely."""
    from agent.orchestrator import run_agent
    result = run_agent(
        question="Predict the risk for county 48201",
        bedrock_call_fn=None,
    )
    mock_ambiguity.assert_not_called()
    mock_predict.assert_called_once()
    assert result["answer"] != ""
