"""
Stress test suite for the Hazard Risk Intelligence Agent.

Covers every routing path, intent, hazard synonym, and tool combination.

Run modes
---------
  # Fast — all mocked, no AWS required (default pytest run)
  pytest tests/stress_test.py -v

  # Live — real Athena + Bedrock + Pinecone calls (requires AWS credentials)
  STRESS_LIVE=1 pytest tests/stress_test.py -v -m live

  # Standalone summary report
  python tests/stress_test.py
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Shared fixtures / helpers ─────────────────────────────────────────────────

LIVE = os.getenv("STRESS_LIVE", "").lower() in ("1", "true", "yes")

SAMPLE_QUERY_RESULT = {
    "results": [
        {"county_fips": "48201", "county_name": "Harris County", "state": "TX",
         "avg_expected_loss": "52340.0", "avg_risk_score": "0.9936",
         "avg_vulnerability": "0.679", "avg_resilience": "0.178", "years_on_record": "8"},
        {"county_fips": "12086", "county_name": "Miami-Dade County", "state": "FL",
         "avg_expected_loss": "48120.0", "avg_risk_score": "0.9618",
         "avg_vulnerability": "0.759", "avg_resilience": "0.429", "years_on_record": "8"},
    ],
    "sql_executed": "SELECT * FROM gold_hazard.risk_feature_mart LIMIT 10;",
    "intent": "top_counties_by_risk",
    "row_count": 2,
    "tool": "query",
}

SAMPLE_HAZARD_INCREASE_RESULT = {
    "results": [
        {"county_fips": "48201", "county_name": "Harris County", "state": "TX",
         "events_early_period": "120", "events_recent_period": "198",
         "absolute_increase": "78", "pct_increase": "65.0"},
    ],
    "sql_executed": "SELECT ... FROM gold_hazard.hazard_event_summary_current ...",
    "intent": "hazard_event_increase",
    "row_count": 1,
    "tool": "query",
}

SAMPLE_RAG_CHUNKS = [
    {"text": "Coastal counties face storm surge and wind from hurricanes.",
     "score": 0.88, "metadata": {"source": "noaa_report.pdf", "hazard_type": "Tropical Storm"}},
    {"text": "FEMA's National Risk Index quantifies expected annual loss.", "score": 0.81,
     "metadata": {"source": "nri_methodology.pdf", "hazard_type": "general"}},
]

SAMPLE_PREDICT_RESULT = {
    "risk_tier": "HIGH",
    "probabilities": {"LOW": 0.05, "MEDIUM": 0.20, "HIGH": 0.75},
    "county_name": "Harris County",
    "county_fips": "48201",
}


def _mock_bedrock(system: str, user_message: str) -> str:
    """Lightweight mock that returns a realistic-looking synthesis."""
    if "Harris County" in user_message:
        return "Harris County, TX leads with the highest Expected Annual Loss of $52,340 per year."
    if "coastal" in user_message.lower() or "hurricane" in user_message.lower():
        return "Coastal counties are more vulnerable due to storm surge, wind exposure, and low elevation."
    if "NRI" in user_message or "methodology" in user_message.lower():
        return "The NRI Expected Annual Loss is calculated by combining hazard frequency, exposure, and vulnerability."
    return "Based on the retrieved data, the answer is: [synthesized result from Nova Lite]."


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Intent classifier: routing and parameter extraction
# ═══════════════════════════════════════════════════════════════════════════════

from analytics.intent_classifier import (
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    _HAZARD_SYNONYMS,
    _extract_hazard_type,
    _extract_limit,
    _extract_years,
    classify_intent,
)


@pytest.mark.parametrize("question,expected_template", [
    # Standard top-counties queries
    ("Show top 10 counties by risk from 2015 to 2023",          "top_counties_by_risk"),
    ("Which counties have the highest expected annual loss?",    "top_counties_by_risk"),
    ("Worst counties for disaster risk",                         "top_counties_by_risk"),
    # All-hazard increase (no specific hazard → largest_increase)
    ("Which counties saw the biggest increase in events from 2010 to 2020?", "largest_increase"),
    # Hazard-specific increase → must route to hazard_event_increase
    ("Which counties saw the largest increase in flood events 2015–2023?",   "hazard_event_increase"),
    ("Biggest increase in tornado events 2010 to 2023",                       "hazard_event_increase"),
    ("Largest rise in wildfire events from 2010 to 2022",                     "hazard_event_increase"),
    ("Most growth in hurricane disaster events 2010–2022",                    "hazard_event_increase"),
    ("Counties with biggest increase in drought events 2015 2023",            "hazard_event_increase"),
    # County comparison
    ("Compare Harris County vs Miami-Dade from 2015 to 2023",   "county_comparison"),
    ("Comparison between LA County and Cook County",             "county_comparison"),
    # Trend by year
    ("Show the annual trend of hazard events from 2015 to 2023", "hazard_trend_by_year"),
    ("Year-over-year change in events 2015 to 2023",             "hazard_trend_by_year"),
    ("Events by year over time 2010 to 2023",                    "hazard_trend_by_year"),
])
def test_classify_intent_template(question, expected_template):
    intent = classify_intent(question)
    assert intent.template == expected_template, (
        f"Q: '{question}'\n  Expected: {expected_template}\n  Got: {intent.template}"
    )


@pytest.mark.parametrize("question,expected_start,expected_end", [
    ("Show top 10 counties by risk from 2015 to 2023", 2015, 2023),
    ("Events from 2018 to 2022",                        2018, 2022),
    # No years → defaults
    ("Top counties by risk",                            DEFAULT_START_YEAR, DEFAULT_END_YEAR),
])
def test_year_extraction(question, expected_start, expected_end):
    intent = classify_intent(question)
    assert intent.params["start_year"] == expected_start, (
        f"Q: '{question}' — expected start_year={expected_start}, got {intent.params['start_year']}"
    )
    assert intent.params["end_year"] == expected_end, (
        f"Q: '{question}' — expected end_year={expected_end}, got {intent.params['end_year']}"
    )


def test_year_defaults_are_hardcoded():
    """DEFAULT_START_YEAR / DEFAULT_END_YEAR must be hardcoded, not from datetime.now()."""
    assert DEFAULT_START_YEAR == 2010, "DEFAULT_START_YEAR must be 2010 to match Gold-layer data"
    assert DEFAULT_END_YEAR == 2023,   "DEFAULT_END_YEAR must be 2023 to match Gold-layer data"


@pytest.mark.parametrize("question,expected_limit", [
    ("Show top 10 counties by risk", 10),
    ("Top 25 counties by expected loss", 25),
    ("Top 5 counties", 5),
    ("Show counties by risk",  10),   # default
])
def test_limit_extraction(question, expected_limit):
    intent = classify_intent(question)
    assert intent.params["limit"] == expected_limit


@pytest.mark.parametrize("phrase,expected_canonical", [
    ("flood",          "Flood"),
    ("flooding",       "Flood"),
    ("flash flood",    "Flash Flood"),
    ("flash flooding", "Flash Flood"),
    ("tornado",        "Tornado"),
    ("tornadoes",      "Tornado"),
    ("twister",        "Tornado"),
    ("hurricane",      "Tropical Storm"),
    ("tropical storm", "Tropical Storm"),
    ("cyclone",        "Tropical Storm"),
    ("typhoon",        "Tropical Storm"),
    ("wildfire",       "Wildfire"),
    ("forest fire",    "Wildfire"),
    ("fire",           "Wildfire"),
    ("drought",        "Drought"),
    ("hail",           "Hail"),
    ("hailstorm",      "Hail"),
    ("lightning",      "Lightning"),
    ("winter storm",   "Heavy Snow"),
    ("blizzard",       "Heavy Snow"),
    ("snow",           "Heavy Snow"),
    ("heat wave",      "Excessive Heat"),
    ("extreme heat",   "Excessive Heat"),
    ("high wind",      "High Wind"),
    ("strong wind",    "Strong Wind"),
    ("heavy rain",     "Heavy Rain"),
    ("debris flow",    "Debris Flow"),
    ("landslide",      "Debris Flow"),
    ("mudslide",       "Debris Flow"),
    ("dust storm",     "Dust Storm"),
    ("fog",            "Dense Fog"),
])
def test_hazard_synonym_mapping(phrase, expected_canonical):
    question = f"Which counties saw the largest increase in {phrase} events 2015-2023"
    result = _extract_hazard_type(question)
    assert result == expected_canonical, (
        f"'{phrase}' → expected '{expected_canonical}', got '{result}'"
    )


def test_no_hazard_returns_all():
    result = _extract_hazard_type("Which counties have the highest risk?")
    assert result == "all"


def test_period_split_for_hazard_increase():
    """For hazard_event_increase, period boundaries must be auto-set from start/end years."""
    intent = classify_intent("Largest increase in flood events 2015 to 2023")
    assert intent.template == "hazard_event_increase"
    p = intent.params
    # mid = (2015 + 2023) // 2 = 2019
    assert p["period_a_start"] == 2015
    assert p["period_a_end"] == 2019
    assert p["period_b_start"] == 2020
    assert p["period_b_end"] == 2023


def test_all_hazard_increase_uses_largest_increase_not_hazard_event():
    """All-hazard increase queries must NOT route to hazard_event_increase."""
    intent = classify_intent("Which counties saw the biggest increase in disaster events 2010 to 2022?")
    assert intent.template == "largest_increase"
    assert intent.params.get("hazard_type") == "all"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Router: tool selection logic
# ═══════════════════════════════════════════════════════════════════════════════

from agent.router import route


@pytest.mark.parametrize("question,must_include", [
    ("Show top 10 counties by predicted risk score",             ["predict"]),
    ("What is Harris County's predicted risk tier?",             ["predict"]),
    ("Top 10 counties by risk from 2015 to 2023",                ["query"]),
    ("Which counties have the highest flood event count?",       ["query"]),
    ("Year-over-year events trend 2015 to 2023",                 ["query"]),
    ("Why are coastal counties more vulnerable to hurricanes?",   ["ask"]),
    ("What is the NRI expected loss methodology?",               ["ask"]),
    ("How does FEMA calculate disaster declarations?",           ["ask"]),
    # Hybrid: predict + query
    ("Show top 10 counties by predicted risk and property damage", ["predict", "query"]),
])
def test_router_tools(question, must_include):
    decision = route(question)
    for tool in must_include:
        assert tool in decision.tools, (
            f"Expected tool '{tool}' for: '{question}'\nGot: {decision.tools}"
        )


def test_router_always_returns_routing_decision():
    from agent.router import RoutingDecision
    result = route("Some completely random question about hazards")
    assert isinstance(result, RoutingDecision)
    assert len(result.tools) >= 1
    assert result.reasoning


def test_router_hybrid_sets_flag():
    decision = route("Show top counties by predicted risk and property damage")
    assert decision.is_hybrid is True


def test_router_single_tool_not_hybrid():
    decision = route("Why are floods increasing?")
    assert decision.is_hybrid is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — TAG pipeline (mocked)
# ═══════════════════════════════════════════════════════════════════════════════

@patch("analytics.query_engine.run_query")
def test_tag_synthesizes_answer(mock_run_query):
    mock_run_query.return_value = SAMPLE_QUERY_RESULT
    mock_synth = MagicMock(return_value="Harris County leads with $52,340 EAL.")

    from analytics.query_engine import run_tag_query
    result = run_tag_query("Top counties by risk", mock_synth)

    assert "answer" in result
    assert result["tag_enabled"] is True
    assert result["results"] == SAMPLE_QUERY_RESULT["results"]
    mock_synth.assert_called_once()


@patch("analytics.query_engine.run_query")
def test_tag_no_rows_skips_synthesis(mock_run_query):
    mock_run_query.return_value = {**SAMPLE_QUERY_RESULT, "results": [], "row_count": 0}
    mock_synth = MagicMock()

    from analytics.query_engine import run_tag_query
    result = run_tag_query("Top counties by risk", mock_synth)

    mock_synth.assert_not_called()
    assert "no data" in result["answer"].lower() or "not contain" in result["answer"].lower()


@patch("analytics.query_engine.run_query")
def test_tag_synthesis_failure_is_graceful(mock_run_query):
    mock_run_query.return_value = SAMPLE_QUERY_RESULT
    mock_synth = MagicMock(side_effect=RuntimeError("Bedrock timeout"))

    from analytics.query_engine import run_tag_query
    result = run_tag_query("Top counties by risk", mock_synth)

    assert "answer" in result
    assert "unavailable" in result["answer"].lower() or "error" in result["answer"].lower()


@patch("analytics.query_engine.run_query")
def test_tag_injects_data_note_for_hazard_question(mock_run_query):
    """When user asks about a specific hazard but intent is risk_feature_mart, data note is injected."""
    mock_run_query.return_value = {**SAMPLE_QUERY_RESULT, "intent": "top_counties_by_risk"}
    captured_messages = []

    def capture_synth(system, user_message):
        captured_messages.append(user_message)
        return "Mocked answer with caveat."

    from analytics.query_engine import run_tag_query
    run_tag_query("Top counties by flood risk 2015-2023", capture_synth)

    assert captured_messages, "synthesize_fn was never called"
    assert "DATA LIMITATION" in captured_messages[0]


@patch("analytics.query_engine.run_query")
def test_tag_no_data_note_for_hazard_event_increase(mock_run_query):
    """hazard_event_increase intent must NOT inject data limitation note."""
    mock_run_query.return_value = {**SAMPLE_HAZARD_INCREASE_RESULT}
    captured_messages = []

    def capture_synth(system, user_message):
        captured_messages.append(user_message)
        return "Mocked answer."

    from analytics.query_engine import run_tag_query
    run_tag_query("Largest increase in flood events 2015-2023", capture_synth)

    assert captured_messages
    assert "DATA LIMITATION" not in captured_messages[0], (
        "hazard_event_increase uses per-hazard table — no all-hazard caveat should appear"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Orchestrator: full agent loop (all tools mocked)
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_RESPONSE_KEYS = {"question", "routing", "tool_outputs", "tool_used"}


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
@patch("agent.orchestrator.run_query")
@patch("agent.orchestrator.predict_risk")
def test_orchestrator_query_only_uses_tag(mock_predict, mock_raw_query, mock_tag, mock_retrieve):
    """Pure query route: run_tag_query must be called, raw run_query must NOT be called."""
    mock_tag.return_value = {**SAMPLE_QUERY_RESULT, "answer": "Harris County leads.", "tag_enabled": True}

    from agent.orchestrator import run_agent
    result = run_agent("Top 10 counties by risk 2015-2023", bedrock_call_fn=_mock_bedrock)

    mock_tag.assert_called_once()
    mock_raw_query.assert_not_called()
    assert "query" in result["tool_used"]
    assert "Harris County leads" in result["answer"]
    assert REQUIRED_RESPONSE_KEYS.issubset(result.keys())


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_query")
def test_orchestrator_ask_route(mock_raw_query, mock_retrieve):
    """Pure ask route: retrieve_similar must be called; query must NOT be called."""
    mock_retrieve.return_value = SAMPLE_RAG_CHUNKS

    from agent.orchestrator import run_agent
    result = run_agent(
        "Why are coastal counties more vulnerable to hurricanes?",
        bedrock_call_fn=_mock_bedrock,
    )

    mock_retrieve.assert_called_once()
    mock_raw_query.assert_not_called()
    assert "ask" in result["tool_used"]
    assert result["answer"]
    assert REQUIRED_RESPONSE_KEYS.issubset(result.keys())


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
@patch("agent.orchestrator.predict_risk")
@patch("agent.orchestrator._fetch_county_features")
def test_orchestrator_hybrid_calls_both_tools(mock_fetch, mock_predict, mock_tag, mock_retrieve):
    """Hybrid question: both predict and query tools must be invoked."""
    mock_fetch.return_value = ({}, "Harris County", "48201")
    mock_predict.return_value = SAMPLE_PREDICT_RESULT
    mock_tag.return_value = {**SAMPLE_QUERY_RESULT, "answer": "Harris: $52k EAL.", "tag_enabled": True}
    mock_retrieve.return_value = []

    from agent.orchestrator import run_agent
    result = run_agent(
        "Show top counties by predicted risk and property damage",
        bedrock_call_fn=_mock_bedrock,
    )

    assert "predict" in result["tool_used"]
    assert "query" in result["tool_used"]
    assert result["routing"]["is_hybrid"] is True


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
@patch("agent.orchestrator.run_query")
def test_orchestrator_no_bedrock_uses_raw_query(mock_raw_query, mock_tag, mock_retrieve):
    """Without bedrock_call_fn (no LLM), raw run_query must be used (no TAG)."""
    mock_raw_query.return_value = SAMPLE_QUERY_RESULT

    from agent.orchestrator import run_agent
    result = run_agent("Top 10 counties by risk", bedrock_call_fn=None)

    mock_raw_query.assert_called_once()
    mock_tag.assert_not_called()


@patch("agent.orchestrator.retrieve_similar")
@patch("agent.orchestrator.run_tag_query")
def test_orchestrator_tag_answer_not_double_synthesized(mock_tag, mock_retrieve):
    """For pure query route, bedrock_call_fn must not be called a second time after TAG."""
    mock_tag.return_value = {**SAMPLE_QUERY_RESULT, "answer": "TAG answer.", "tag_enabled": True}
    mock_bedrock = MagicMock(return_value="should not appear twice")

    from agent.orchestrator import run_agent
    run_agent("Top 10 counties by risk", bedrock_call_fn=mock_bedrock)

    # TAG passes bedrock as synthesize_fn → 1 call inside run_tag_query.
    # Orchestrator synthesis block must NOT make an additional call.
    assert mock_bedrock.call_count <= 1, (
        f"bedrock_call_fn called {mock_bedrock.call_count} times — expected ≤ 1"
    )


@pytest.mark.parametrize("question", [
    "Top 10 counties by risk",
    "Why are floods increasing?",
    "Compare Harris County vs Miami-Dade",
    "Show me data for county 48201",
    "What is a unicorn hazard?",  # out-of-domain
])
@patch("agent.orchestrator.retrieve_similar", return_value=[])
@patch("agent.orchestrator.run_tag_query", return_value={**SAMPLE_QUERY_RESULT, "answer": "ok", "tag_enabled": True})
@patch("agent.orchestrator.run_query", return_value=SAMPLE_QUERY_RESULT)
@patch("agent.orchestrator.predict_risk", return_value=SAMPLE_PREDICT_RESULT)
@patch("agent.orchestrator._fetch_county_features", return_value=({}, "", ""))
def test_orchestrator_always_returns_valid_structure(
    mock_fetch, mock_predict, mock_raw_query, mock_tag, mock_retrieve, question
):
    """Every question must produce a response with all required keys — no exceptions."""
    from agent.orchestrator import run_agent
    result = run_agent(question, bedrock_call_fn=_mock_bedrock)
    assert REQUIRED_RESPONSE_KEYS.issubset(result.keys()), (
        f"Missing keys for '{question}': {REQUIRED_RESPONSE_KEYS - result.keys()}"
    )
    assert result["answer"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — SQL template rendering (no AWS, pure string compilation)
# ═══════════════════════════════════════════════════════════════════════════════

from analytics.query_engine import _compile_sql, _enforce_guardrails, _load_template, _sanitize_params


@pytest.mark.parametrize("template_name,params", [
    ("top_counties_by_risk", {"start_year": 2015, "end_year": 2023, "limit": 10}),
    ("hazard_trend_by_year",  {"start_year": 2015, "end_year": 2023}),
    ("county_comparison",
     {"county_fips_list": "'48201','12086'", "start_year": 2015, "end_year": 2023, "limit": 10}),
    ("largest_increase",
     {"period_a_start": 2015, "period_a_end": 2019,
      "period_b_start": 2020, "period_b_end": 2023, "limit": 10}),
    ("hazard_event_increase",
     {"period_a_start": 2015, "period_a_end": 2019,
      "period_b_start": 2020, "period_b_end": 2023,
      "hazard_type": "Flood", "limit": 10}),
])
def test_sql_template_compiles_cleanly(template_name, params):
    """Every template must compile without KeyError and pass guardrails."""
    raw = _load_template(template_name)
    clean_params = _sanitize_params(params)
    compiled = _compile_sql(raw, clean_params)
    safe = _enforce_guardrails(compiled)
    assert "gold_hazard" in safe.lower()
    assert "LIMIT" in safe.upper()


def test_hazard_event_increase_references_correct_table():
    """hazard_event_increase.sql must reference hazard_event_summary_current, not the old name."""
    raw = _load_template("hazard_event_increase")
    assert "hazard_event_summary_current" in raw, (
        "Table must be hazard_event_summary_current (not hazard_event_summary)"
    )
    assert "hazard_event_summary\n" not in raw  # no bare reference to the wrong name


def test_guardrails_block_ddl():
    for keyword in ["DROP TABLE", "DELETE FROM", "INSERT INTO", "CREATE TABLE"]:
        with pytest.raises(ValueError, match="Forbidden"):
            _enforce_guardrails(f"{keyword} gold_hazard.risk_feature_mart LIMIT 10")


def test_guardrails_require_gold_hazard():
    with pytest.raises(ValueError, match="gold_hazard"):
        _enforce_guardrails("SELECT * FROM some_other_db.table LIMIT 10")


def test_sanitize_year_out_of_range():
    with pytest.raises(ValueError, match="Year out of range"):
        _sanitize_params({"start_year": 1800})


def test_sanitize_hazard_type_rejects_sql_injection():
    with pytest.raises(ValueError, match="Invalid hazard_type"):
        _sanitize_params({"hazard_type": "Flood'; DROP TABLE users; --"})


def test_sanitize_fips_list_rejects_invalid():
    with pytest.raises(ValueError):
        _sanitize_params({"county_fips_list": "UNION SELECT * FROM users"})


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Live end-to-end tests (skipped unless STRESS_LIVE=1)
# ═══════════════════════════════════════════════════════════════════════════════

live_only = pytest.mark.skipif(not LIVE, reason="Set STRESS_LIVE=1 to run live AWS tests")


@live_only
def test_live_top_counties_query():
    """Real Athena query: top counties by risk 2015-2023 must return rows."""
    from analytics.query_engine import run_tag_query
    result = run_tag_query("Show top 10 counties by risk from 2015 to 2023", _mock_bedrock)
    assert result["row_count"] > 0, "Live query returned 0 rows"
    assert "county_name" in result["results"][0]


@live_only
def test_live_flood_event_increase():
    """Real Athena query against hazard_event_summary_current for flood increases."""
    from analytics.query_engine import run_tag_query
    result = run_tag_query(
        "Which counties saw the largest increase in flood events 2015–2023?",
        _mock_bedrock,
    )
    assert result["intent"] == "hazard_event_increase", (
        f"Expected hazard_event_increase, got {result['intent']}"
    )
    assert result["row_count"] > 0, (
        "Live per-hazard query returned 0 rows — check hazard_event_summary_current table name and schema"
    )
    first = result["results"][0]
    assert "events_early_period" in first or "absolute_increase" in first


@live_only
def test_live_tornado_event_increase():
    """Tornado synonym routing: 'tornado' → 'Tornado' → hazard_event_summary_current."""
    from analytics.query_engine import run_tag_query
    result = run_tag_query(
        "Biggest increase in tornado events 2010 to 2023",
        _mock_bedrock,
    )
    assert result["intent"] == "hazard_event_increase"
    assert result["row_count"] > 0


@live_only
def test_live_hurricane_routes_to_tropical_storm():
    """'hurricane' must map to 'Tropical Storm' and return real event data."""
    from analytics.query_engine import run_tag_query
    result = run_tag_query(
        "Which counties saw the largest increase in hurricane events 2010-2023?",
        _mock_bedrock,
    )
    assert "Tropical Storm" in result["sql_executed"], (
        "SQL should filter on 'Tropical Storm', not 'hurricane'"
    )
    assert result["row_count"] > 0


@live_only
def test_live_county_comparison():
    """Real Athena comparison: Harris County vs Miami-Dade."""
    from analytics.query_engine import run_tag_query
    result = run_tag_query(
        "Compare Harris County vs Miami-Dade from 2015 to 2023",
        _mock_bedrock,
    )
    assert result["row_count"] > 0
    fips_values = [r.get("county_fips") for r in result["results"]]
    assert any(f in ("48201", "12086") for f in fips_values)


@live_only
def test_live_rag_coastal_vulnerability():
    """Real Pinecone + Bedrock call for open-ended domain question."""
    from agent.orchestrator import run_agent
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    def real_bedrock_call(system, user_message):
        resp = bedrock.converse(
            modelId="us.amazon.nova-lite-v1:0",
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
        return resp["output"]["message"]["content"][0]["text"]

    result = run_agent(
        "Why are coastal counties more vulnerable to hurricanes?",
        bedrock_call_fn=real_bedrock_call,
    )
    assert result["answer"], "Live RAG response was empty"
    assert "ask" in result["tool_used"]
    # Answer should mention storm or coast or hurricane — not a refusal
    answer_lower = result["answer"].lower()
    assert any(w in answer_lower for w in ["storm", "coast", "hurricane", "surge", "wind", "flood"]), (
        f"Answer doesn't mention any relevant hazard terms:\n{result['answer']}"
    )


@live_only
def test_live_nri_methodology_uses_domain_knowledge():
    """NRI methodology question with sparse RAG — LLM must answer from expertise, not refuse."""
    from agent.orchestrator import run_agent
    import boto3

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    def real_bedrock_call(system, user_message):
        resp = bedrock.converse(
            modelId="us.amazon.nova-lite-v1:0",
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
        return resp["output"]["message"]["content"][0]["text"]

    result = run_agent(
        "What is the NRI expected loss methodology?",
        bedrock_call_fn=real_bedrock_call,
    )
    assert result["answer"]
    answer_lower = result["answer"].lower()
    refuse_phrases = ["cannot answer", "insufficient context", "i don't have", "no information"]
    assert not any(p in answer_lower for p in refuse_phrases), (
        f"LLM refused to answer instead of using domain knowledge:\n{result['answer']}"
    )
    assert any(w in answer_lower for w in ["loss", "nri", "expected", "hazard", "risk"]), (
        f"Answer doesn't mention relevant NRI terms:\n{result['answer']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone runner — prints a summary table
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    args = [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"]
    if LIVE:
        args += ["-m", "live or not live"]
    print(f"Running: {' '.join(args)}\n")
    result = subprocess.run(args)
    sys.exit(result.returncode)
