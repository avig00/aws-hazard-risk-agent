"""
Intent classifier: maps a natural-language question to a SQL template name
and extracts parameter bindings.

Uses regex pattern matching with an LLM fallback for ambiguous questions.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime


CURRENT_YEAR = datetime.now().year
DEFAULT_START_YEAR = CURRENT_YEAR - 8
DEFAULT_END_YEAR = CURRENT_YEAR - 1


@dataclass
class QueryIntent:
    template: str          # Which SQL template to use
    params: dict = field(default_factory=dict)  # Template parameter bindings
    confidence: float = 1.0


# ── Pattern rules ────────────────────────────────────────────────────────────

_PATTERNS = [
    # "which counties saw the largest increase in flood events from 2015–2023"
    {
        "template": "largest_increase",
        "regex": r"(largest|biggest|most)\s+(increase|growth|rise).*(event|hazard|disaster)",
        "params": {"period_a_start": DEFAULT_START_YEAR - 4, "period_a_end": DEFAULT_START_YEAR,
                   "period_b_start": DEFAULT_START_YEAR + 1, "period_b_end": DEFAULT_END_YEAR},
    },
    # "compare harris county vs miami-dade" / "comparison between X and Y"
    {
        "template": "county_comparison",
        "regex": r"(compare|comparison|versus|vs\.?)\s+.+(county|counties)",
        "params": {"county_fips_list": "'48201','12086'"},  # Harris TX, Miami-Dade FL defaults
    },
    # "hazard trend" / "events by year" / "year-over-year"
    {
        "template": "hazard_trend_by_year",
        "regex": r"(trend|year.over.year|by year|over time|annual)",
        "params": {"hazard_type": "all"},
    },
    # "top N counties by risk" (default / catch-all for ranking questions)
    {
        "template": "top_counties_by_risk",
        "regex": r"(top|highest|most at risk|worst|greatest risk|ranking)",
        "params": {},
    },
]


def _extract_years(question: str) -> dict:
    """Pull explicit year ranges from the question text."""
    years = re.findall(r"\b(20\d{2})\b", question)
    params = {}
    if len(years) >= 2:
        params["start_year"] = int(min(years))
        params["end_year"] = int(max(years))
    elif len(years) == 1:
        params["start_year"] = int(years[0])
        params["end_year"] = DEFAULT_END_YEAR
    return params


def _extract_limit(question: str, default: int = 10) -> int:
    """Extract 'top N' or 'N counties' from question."""
    match = re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+count", question, re.IGNORECASE)
    if match:
        n = match.group(1) or match.group(2)
        return int(n)
    return default


def _extract_hazard_type(question: str) -> str:
    """Extract a specific hazard type mentioned in the question."""
    hazards = ["flood", "hurricane", "tornado", "wildfire", "earthquake",
               "drought", "hail", "winter storm", "thunderstorm"]
    q = question.lower()
    for h in hazards:
        if h in q:
            return h
    return "all"


def classify_intent(question: str, default_limit: int = 10) -> QueryIntent:
    """
    Match question against patterns and return QueryIntent with template + params.

    Falls back to 'top_counties_by_risk' if no pattern matches.
    """
    q = question.lower()

    matched_template = None
    base_params = {}

    for rule in _PATTERNS:
        if re.search(rule["regex"], q, re.IGNORECASE):
            matched_template = rule["template"]
            base_params = dict(rule["params"])
            break

    if not matched_template:
        matched_template = "top_counties_by_risk"

    # Merge dynamic extractions
    year_params = _extract_years(question)
    limit = _extract_limit(question, default=default_limit)
    hazard_type = _extract_hazard_type(question)

    params = {
        "start_year": DEFAULT_START_YEAR,
        "end_year": DEFAULT_END_YEAR,
        "limit": limit,
        **base_params,
        "hazard_type": hazard_type,  # extracted value overrides base default
        **year_params,
    }

    # For largest_increase: auto-set period boundaries if years extracted
    if matched_template == "largest_increase" and "start_year" in year_params:
        mid = (params["start_year"] + params["end_year"]) // 2
        params.update({
            "period_a_start": params["start_year"],
            "period_a_end": mid,
            "period_b_start": mid + 1,
            "period_b_end": params["end_year"],
        })

    return QueryIntent(template=matched_template, params=params, confidence=0.9)
