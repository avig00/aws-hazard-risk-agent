"""
Unit tests for analytics/intent_classifier.py and analytics/query_engine.py

Tests SQL template rendering and guardrail enforcement without Athena connectivity.
"""
import pytest

from analytics.intent_classifier import QueryIntent, classify_intent
from analytics.query_engine import (
    _compile_sql,
    _enforce_guardrails,
    _sanitize_params,
)


# ── intent_classifier ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_template", [
    ("Show the top 10 counties by risk from 2015 to 2023", "top_counties_by_risk"),
    ("Which counties had the highest average expected loss?", "top_counties_by_risk"),
    ("Which counties saw the largest increase in flood events from 2010–2022?", "hazard_event_increase"),
    # flood-specific trend → hazard_trend_specific (per-hazard table, not all-hazard)
    ("Show me flood event trends by year 2015–2023", "hazard_trend_specific"),
    # all-hazard trend → hazard_trend_by_year (risk_feature_mart)
    ("Show the annual trend of all hazard events from 2015 to 2023", "hazard_trend_by_year"),
    ("Compare Harris County vs Miami-Dade over the last 5 years", "county_comparison"),
])
def test_intent_classification(question, expected_template):
    intent = classify_intent(question)
    assert intent.template == expected_template, (
        f"Expected '{expected_template}' for: '{question}', got '{intent.template}'"
    )


def test_intent_extracts_years():
    intent = classify_intent("Top 10 counties by risk from 2018 to 2023")
    assert intent.params["start_year"] == 2018
    assert intent.params["end_year"] == 2023


def test_intent_extracts_limit():
    intent = classify_intent("Show top 25 counties by flood events")
    assert intent.params["limit"] == 25


def test_intent_extracts_hazard_type():
    intent = classify_intent("Year-over-year trend for flood events 2015–2023")
    assert intent.params["hazard_type"] == "Flood"  # canonical DB value from _HAZARD_SYNONYMS


def test_intent_default_hazard_type():
    intent = classify_intent("Show annual trends from 2015 to 2023")
    assert intent.params["hazard_type"] == "all"


def test_intent_returns_query_intent_type():
    result = classify_intent("Top 5 counties by risk")
    assert isinstance(result, QueryIntent)


# ── _sanitize_params ──────────────────────────────────────────────────────────

def test_sanitize_valid_params():
    params = {"start_year": 2015, "end_year": 2023, "limit": 10, "hazard_type": "flood"}
    result = _sanitize_params(params)
    assert result["start_year"] == 2015
    assert result["hazard_type"] == "flood"


def test_sanitize_clamps_limit():
    params = {"limit": 999}
    result = _sanitize_params(params)
    assert result["limit"] <= 100


def test_sanitize_rejects_invalid_year():
    with pytest.raises(ValueError, match="Year out of range"):
        _sanitize_params({"start_year": 1850})


def test_sanitize_rejects_invalid_hazard():
    with pytest.raises(ValueError, match="Invalid hazard_type"):
        _sanitize_params({"hazard_type": "flood'; DROP TABLE--"})


def test_sanitize_rejects_invalid_fips():
    with pytest.raises(ValueError, match="Invalid county_fips_list"):
        _sanitize_params({"county_fips_list": "'; DROP TABLE counties;--"})


def test_sanitize_valid_fips():
    params = {"county_fips_list": "'48201','12086'"}
    result = _sanitize_params(params)
    assert result["county_fips_list"] == "'48201','12086'"


# ── _compile_sql ──────────────────────────────────────────────────────────────

def test_compile_sql_binds_params():
    template = "SELECT * FROM hazard_gold.risk_feature_mart WHERE year BETWEEN {start_year} AND {end_year} LIMIT {limit};"
    result = _compile_sql(template, {"start_year": 2015, "end_year": 2023, "limit": 10})
    assert "2015" in result
    assert "2023" in result
    assert "LIMIT 10" in result


def test_compile_sql_raises_on_missing_param():
    template = "SELECT * FROM t WHERE year = {year};"
    with pytest.raises(ValueError, match="Missing template parameter"):
        _compile_sql(template, {})


# ── _enforce_guardrails ───────────────────────────────────────────────────────

def test_guardrails_allows_valid_sql():
    sql = "SELECT * FROM gold_hazard.risk_feature_mart LIMIT 10;"
    result = _enforce_guardrails(sql)
    assert "LIMIT" in result


def test_guardrails_adds_missing_limit():
    sql = "SELECT * FROM gold_hazard.risk_feature_mart WHERE year = 2023"
    result = _enforce_guardrails(sql)
    assert "LIMIT" in result


def test_guardrails_blocks_drop():
    with pytest.raises(ValueError, match="Forbidden SQL keyword"):
        _enforce_guardrails("DROP TABLE gold_hazard.risk_feature_mart;")


def test_guardrails_blocks_non_gold_table():
    with pytest.raises(ValueError, match="Gold-layer only"):
        _enforce_guardrails("SELECT * FROM raw_bronze.events LIMIT 10;")
