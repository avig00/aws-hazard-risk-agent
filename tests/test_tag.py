"""
Tests for Table Augmented Generation (TAG) pipeline.

Covers:
- TAG prompt construction (tag_template.py)
- run_tag_query() synthesis integration (query_engine.py)
- Orchestrator uses TAG for pure-query routes, not a second RAG call
- Graceful fallback when LLM synthesis fails
"""
from unittest.mock import MagicMock, call, patch

import pytest

from rag.prompts.tag_template import (
    TAG_SYSTEM_PROMPT,
    _describe_columns,
    _format_results_table,
    build_tag_prompt,
)


# ── tag_template.py ───────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    {"county_name": "Harris County", "state": "TX", "avg_expected_loss": 52340.0, "years_on_record": 8},
    {"county_name": "Miami-Dade",    "state": "FL", "avg_expected_loss": 48120.0, "years_on_record": 8},
    {"county_name": "Broward",       "state": "FL", "avg_expected_loss": 41500.0, "years_on_record": 7},
]

SAMPLE_SQL = "SELECT county_name, state, AVG(nri_eal_score) AS avg_expected_loss FROM gold_hazard.risk_feature_mart GROUP BY county_name, state ORDER BY avg_expected_loss DESC LIMIT 10;"


def test_build_tag_prompt_contains_question():
    question = "Top 3 counties by expected loss"
    prompt = build_tag_prompt(question, SAMPLE_ROWS, SAMPLE_SQL, "top_counties_by_risk", 3)
    assert question in prompt


def test_build_tag_prompt_contains_sql():
    prompt = build_tag_prompt("test", SAMPLE_ROWS, SAMPLE_SQL, "top_counties_by_risk", 3)
    assert "gold_hazard" in prompt
    assert "SELECT" in prompt


def test_build_tag_prompt_contains_row_data():
    prompt = build_tag_prompt("test", SAMPLE_ROWS, SAMPLE_SQL, "top_counties_by_risk", 3)
    assert "Harris County" in prompt
    assert "52340" in prompt


def test_build_tag_prompt_contains_column_definitions():
    prompt = build_tag_prompt("test", SAMPLE_ROWS, SAMPLE_SQL, "top_counties_by_risk", 3)
    # avg_expected_loss is a known column — its description should appear
    assert "Expected Annual Loss" in prompt


def test_build_tag_prompt_empty_results():
    prompt = build_tag_prompt("test", [], SAMPLE_SQL, "top_counties_by_risk", 0)
    assert "No rows returned" in prompt


def test_format_results_table_truncates():
    rows = [{"county": f"County{i}", "score": i * 1000} for i in range(50)]
    table = _format_results_table(rows, max_rows=10)
    assert "more rows truncated" in table


def test_format_results_table_header_row():
    table = _format_results_table(SAMPLE_ROWS)
    assert "county_name" in table
    assert "avg_expected_loss" in table


def test_describe_columns_known_col():
    result = _describe_columns(["avg_expected_loss", "county_name"])
    assert "Expected Annual Loss" in result
    assert "County name" in result


def test_describe_columns_unknown_col():
    # Unknown columns should be silently omitted, not raise
    result = _describe_columns(["unknown_column_xyz"])
    assert "unknown_column_xyz" not in result


def test_tag_system_prompt_not_empty():
    assert len(TAG_SYSTEM_PROMPT) > 100


# ── run_tag_query() ───────────────────────────────────────────────────────────

MOCK_QUERY_RESULT = {
    "results": SAMPLE_ROWS,
    "sql_executed": SAMPLE_SQL,
    "intent": "top_counties_by_risk",
    "row_count": 3,
    "tool": "query",
}


@patch("analytics.query_engine.run_query")
def test_run_tag_query_calls_synthesize_fn(mock_run_query):
    """synthesize_fn must be called once with TAG system prompt and built user message."""
    mock_run_query.return_value = MOCK_QUERY_RESULT

    mock_synthesize = MagicMock(return_value="Harris County leads with $52,340 expected annual loss...")

    from analytics.query_engine import run_tag_query
    result = run_tag_query(
        question="Top counties by expected loss",
        synthesize_fn=mock_synthesize,
        limit=10,
    )

    mock_synthesize.assert_called_once()
    args = mock_synthesize.call_args[0]
    assert args[0] == TAG_SYSTEM_PROMPT          # system prompt
    assert "Harris County" in args[1]            # user message includes table data
    assert "Top counties by expected loss" in args[1]


@patch("analytics.query_engine.run_query")
def test_run_tag_query_returns_answer_key(mock_run_query):
    """Result must contain an 'answer' key with the synthesized text."""
    mock_run_query.return_value = MOCK_QUERY_RESULT
    mock_synthesize = MagicMock(return_value="Harris County is highest risk.")

    from analytics.query_engine import run_tag_query
    result = run_tag_query("test", mock_synthesize)

    assert "answer" in result
    assert result["answer"] == "Harris County is highest risk."
    assert result["tag_enabled"] is True


@patch("analytics.query_engine.run_query")
def test_run_tag_query_preserves_raw_results(mock_run_query):
    """Original query result keys must be preserved alongside the TAG answer."""
    mock_run_query.return_value = MOCK_QUERY_RESULT
    mock_synthesize = MagicMock(return_value="Synthesis.")

    from analytics.query_engine import run_tag_query
    result = run_tag_query("test", mock_synthesize)

    assert result["results"] == SAMPLE_ROWS
    assert result["sql_executed"] == SAMPLE_SQL
    assert result["intent"] == "top_counties_by_risk"
    assert result["row_count"] == 3


@patch("analytics.query_engine.run_query")
def test_run_tag_query_empty_results_skips_synthesis(mock_run_query):
    """synthesize_fn must NOT be called when Athena returns zero rows."""
    mock_run_query.return_value = {**MOCK_QUERY_RESULT, "results": [], "row_count": 0}
    mock_synthesize = MagicMock()

    from analytics.query_engine import run_tag_query
    result = run_tag_query("test", mock_synthesize)

    mock_synthesize.assert_not_called()
    assert "answer" in result


@patch("analytics.query_engine.run_query")
def test_run_tag_query_graceful_synthesis_failure(mock_run_query):
    """If synthesize_fn raises, run_tag_query should NOT propagate the exception."""
    mock_run_query.return_value = MOCK_QUERY_RESULT
    mock_synthesize = MagicMock(side_effect=Exception("Bedrock timeout"))

    from analytics.query_engine import run_tag_query
    result = run_tag_query("test", mock_synthesize)

    assert "answer" in result
    assert "unavailable" in result["answer"].lower() or "error" in result["answer"].lower()


# ── Orchestrator: TAG used for pure-query routes ──────────────────────────────

@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
@patch("agent.orchestrator.run_query")
def test_orchestrator_uses_tag_for_query_route(mock_run_query, mock_run_tag, mock_retrieve):
    """When bedrock_call_fn is provided and route is query-only, run_tag_query is called."""
    mock_run_tag.return_value = {
        **MOCK_QUERY_RESULT,
        "answer": "Harris County leads with $52,340 expected annual loss.",
        "tag_enabled": True,
    }
    mock_bedrock = MagicMock(return_value="irrelevant")

    from agent.orchestrator import run_agent
    result = run_agent(
        question="Top 10 counties by expected loss 2015–2023",
        bedrock_call_fn=mock_bedrock,
    )

    mock_run_tag.assert_called_once()
    mock_run_query.assert_not_called()   # raw run_query should NOT be called
    assert "Harris County leads" in result["answer"]


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
@patch("agent.orchestrator.run_query")
def test_orchestrator_uses_raw_query_without_bedrock(mock_run_query, mock_run_tag, mock_retrieve):
    """When bedrock_call_fn is None, raw run_query is called (no TAG)."""
    mock_run_query.return_value = MOCK_QUERY_RESULT

    from agent.orchestrator import run_agent
    result = run_agent(
        question="Top 10 counties by expected loss",
        bedrock_call_fn=None,
    )

    mock_run_query.assert_called_once()
    mock_run_tag.assert_not_called()


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
def test_orchestrator_pure_query_answer_not_double_synthesized(mock_run_tag, mock_retrieve):
    """
    For pure query routes, the TAG answer is used as-is.
    bedrock_call_fn must NOT be called a second time in the synthesis block.
    """
    mock_run_tag.return_value = {
        **MOCK_QUERY_RESULT,
        "answer": "TAG answer from run_tag_query.",
        "tag_enabled": True,
    }
    mock_bedrock = MagicMock(return_value="should not be called again")

    from agent.orchestrator import run_agent
    run_agent(
        question="Top 10 counties by expected loss",
        bedrock_call_fn=mock_bedrock,
    )

    # bedrock_call_fn is passed into run_tag_query as synthesize_fn,
    # but the orchestrator's synthesis block must not call it a second time.
    # run_tag_query itself calls it once internally — that's the only call.
    assert mock_bedrock.call_count <= 1, (
        f"bedrock_call_fn was called {mock_bedrock.call_count} times; expected ≤1"
    )
