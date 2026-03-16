"""
Intent classifier: maps a natural-language question to a SQL template name
and extracts parameter bindings.

Uses regex pattern matching with an LLM fallback for ambiguous questions.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime


# Hardcoded to match the actual Gold-layer data range (risk_feature_mart + hazard_event_summary).
# Do NOT compute from datetime.now() — the data pipeline runs annually and the latest
# complete year is 2023.  Updating this constant is the only change needed when new data arrives.
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2023


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
    # "most FEMA declarations by state" / "states with the most disaster declarations"
    # NOTE: Must come before top_counties_by_risk — "most" is also in the ranking pattern.
    {
        "template": "fema_declarations_by_state",
        "regex": r"(fema\s+declaration|disaster\s+declaration|federal\s+declaration)",
        "params": {},
    },
    # "top N counties by risk / highest expected loss / worst counties"
    # NOTE: Must come before hazard_trend_by_year — "annual" in "expected annual loss"
    # would otherwise match the trend pattern before this ranking pattern.
    {
        "template": "top_counties_by_risk",
        "regex": r"(top|highest|most at risk|worst|greatest risk|ranking)",
        "params": {},
    },
    # "hazard trend" / "events by year" / "year-over-year"
    {
        "template": "hazard_trend_by_year",
        "regex": r"(trend|year.over.year|by year|over time|annual)",
        "params": {"hazard_type": "all"},
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


# Maps user-supplied terms to the canonical hazard_type values in the Gold layer.
# Gold-layer values: Drought, Heavy Snow, Tropical Storm, Tornado, Heavy Rain,
# Funnel Cloud, High Wind, Excessive Heat, Thunderstorm Wind, Flash Flood, Flood,
# Debris Flow, Dust Devil, Heat, Dust Storm, Dense Fog, Hail, Lightning,
# Strong Wind, Wildfire
# Maps canonical Gold-layer hazard_type → the dedicated event-count column in
# risk_feature_mart_current.  Used to route trend queries to the full-history table
# (2010–2023) instead of hazard_event_summary_current (limited recent coverage).
_HAZARD_TYPE_TO_FEATURE_COL: dict[str, str] = {
    "Wildfire":         "wildfire_events",
    "Tornado":          "tornado_events",
    "Flood":            "flood_events",
    "Flash Flood":      "flood_events",
    "Hail":             "hail_events",
    "Lightning":        "lightning_events",
    "High Wind":        "wind_events",
    "Strong Wind":      "wind_events",
    "Thunderstorm Wind":"wind_events",
    "Tropical Storm":   "tropical_events",
    "Heavy Snow":       "winter_events",
    "Excessive Heat":   "heat_events",
    "Heat":             "heat_events",
    "Debris Flow":      "debris_flow_events",
}

_HAZARD_SYNONYMS: dict[str, str] = {
    "hurricane":        "Tropical Storm",
    "tropical storm":   "Tropical Storm",
    "tropical":         "Tropical Storm",
    "cyclone":          "Tropical Storm",
    "typhoon":          "Tropical Storm",
    "flood":            "Flood",
    "flooding":         "Flood",
    "flash flood":      "Flash Flood",
    "flash flooding":   "Flash Flood",
    "tornado":          "Tornado",
    "tornadoes":        "Tornado",
    "twister":          "Tornado",
    "wildfire":         "Wildfire",
    "fire":             "Wildfire",
    "forest fire":      "Wildfire",
    "drought":          "Drought",
    "hail":             "Hail",
    "hailstorm":        "Hail",
    "lightning":        "Lightning",
    "thunderstorm":     "Thunderstorm Wind",
    "thunder":          "Thunderstorm Wind",
    "winter storm":     "Heavy Snow",
    "blizzard":         "Heavy Snow",
    "snowstorm":        "Heavy Snow",
    "snow":             "Heavy Snow",
    "heat wave":        "Excessive Heat",
    "extreme heat":     "Excessive Heat",
    "high wind":        "High Wind",
    "strong wind":      "Strong Wind",
    "heavy rain":       "Heavy Rain",
    "debris flow":      "Debris Flow",
    "landslide":        "Debris Flow",
    "mudslide":         "Debris Flow",
    "dust storm":       "Dust Storm",
    "fog":              "Dense Fog",
}


def _extract_hazard_type(question: str) -> str:
    """
    Extract a specific hazard type from the question and normalize it to the
    canonical Gold-layer value.  Returns 'all' if no hazard is detected.
    """
    q = question.lower()
    # Sort by length descending so multi-word phrases (e.g. "flash flood") match before
    # shorter substrings (e.g. "flood")
    for phrase in sorted(_HAZARD_SYNONYMS, key=len, reverse=True):
        if phrase in q:
            return _HAZARD_SYNONYMS[phrase]
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

    # Route hazard-specific increase queries to hazard_event_summary (per-hazard table)
    # rather than risk_feature_mart (all-hazard aggregate).
    if matched_template == "largest_increase" and params.get("hazard_type", "all") != "all":
        matched_template = "hazard_event_increase"

    # Route hazard-specific top-N queries (e.g. "highest tornado events by county")
    # to hazard_event_summary rather than NRI scores in risk_feature_mart.
    if matched_template == "top_counties_by_risk" and params.get("hazard_type", "all") != "all":
        matched_template = "top_counties_by_hazard"
        # Choose sort column based on what the question is asking for
        q_lower = question.lower()
        if any(w in q_lower for w in ("fatal", "death", "deaths", "killed", "casualties")):
            params["order_col"] = "total_fatalities"
        elif any(w in q_lower for w in ("injur", "hurt", "wounded")):
            params["order_col"] = "total_injuries"
        else:
            params["order_col"] = "total_events"

    # Route hazard-specific trend queries.
    # Prefer hazard_trend_by_feature (uses risk_feature_mart_current dedicated columns,
    # full 2010–2023 history) when the hazard maps to a known feature column.
    # Fall back to hazard_trend_specific (hazard_event_summary_current) only for hazard
    # types without a dedicated column — coverage may be limited in that table.
    if matched_template == "hazard_trend_by_year" and params.get("hazard_type", "all") != "all":
        hazard_col = _HAZARD_TYPE_TO_FEATURE_COL.get(params["hazard_type"])
        if hazard_col:
            matched_template = "hazard_trend_by_feature"
        else:
            matched_template = "hazard_trend_specific"

    return QueryIntent(template=matched_template, params=params, confidence=0.9)
