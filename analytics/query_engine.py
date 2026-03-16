"""
Governed Athena query engine for the /query analytics tool.

Enforces:
- Gold-layer table access only (gold_hazard database)
- Template-based SQL compilation (no free-form SQL injection)
- Automatic LIMIT enforcement
- Partition filtering (year range)
- Scan-cost cap (~100 MB by default)
"""
import logging
import os
import re
from pathlib import Path

import time

import boto3
import yaml

from analytics.intent_classifier import QueryIntent, classify_intent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yml"
TEMPLATE_DIR = Path(__file__).parent / "sql_templates"

ALLOWED_DATABASE = "gold_hazard"
MAX_SCAN_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_LIMIT = 100

# Whitelist of per-hazard feature columns in risk_feature_mart_current.
# Used by hazard_trend_by_feature.sql to avoid injecting arbitrary column names.
ALLOWED_HAZARD_FEATURE_COLS = {
    "wildfire_events", "tornado_events", "flood_events", "hail_events",
    "lightning_events", "wind_events", "tropical_events", "winter_events",
    "heat_events", "debris_flow_events",
}
MIN_YEAR = 2000
MAX_YEAR = 2025


def _load_template(template_name: str) -> str:
    """Load a SQL template file. Raises if template name is not in the allowed set."""
    allowed_templates = {f.stem for f in TEMPLATE_DIR.glob("*.sql")}
    if template_name not in allowed_templates:
        raise ValueError(
            f"Unknown template '{template_name}'. Allowed: {sorted(allowed_templates)}"
        )
    path = TEMPLATE_DIR / f"{template_name}.sql"
    return path.read_text()


def _sanitize_params(params: dict) -> dict:
    """
    Validate and sanitize all template parameters before SQL compilation.
    Prevents injection by enforcing types and value ranges.
    """
    clean = {}
    for key, value in params.items():
        if key in {"start_year", "end_year", "period_a_start", "period_a_end",
                   "period_b_start", "period_b_end"}:
            year = int(value)
            if not (MIN_YEAR <= year <= MAX_YEAR):
                raise ValueError(f"Year out of range: {year}")
            clean[key] = year

        elif key == "limit":
            limit = min(int(value), MAX_LIMIT)
            clean[key] = limit

        elif key == "hazard_type":
            # Allow alpha, space, and hyphen — covers canonical DB values like
            # "Tropical Storm", "Flash Flood", "Thunderstorm Wind"
            if not re.match(r"^[a-zA-Z \-]+$", str(value)):
                raise ValueError(f"Invalid hazard_type: {value}")
            clean[key] = str(value)

        elif key == "county_fips_list":
            # Must be a comma-separated list of quoted 5-digit FIPS codes
            fips_pattern = r"^('[0-9]{5}'(,'[0-9]{5}')*)$"
            val = str(value).replace(" ", "")
            if not re.match(fips_pattern, val):
                raise ValueError(f"Invalid county_fips_list format: {value}")
            clean[key] = val

        elif key == "hazard_col":
            # Must be an exact match against the whitelist — injected as a column name, not a value
            if value not in ALLOWED_HAZARD_FEATURE_COLS:
                raise ValueError(f"Unknown hazard_col '{value}'. Allowed: {sorted(ALLOWED_HAZARD_FEATURE_COLS)}")
            clean[key] = value

        elif key == "order_col":
            # Whitelist of sortable columns in top_counties_by_hazard
            allowed_order_cols = {"total_events", "total_fatalities", "total_injuries"}
            if value not in allowed_order_cols:
                raise ValueError(f"Invalid order_col '{value}'. Allowed: {sorted(allowed_order_cols)}")
            clean[key] = value

        else:
            # Generic string: strip any SQL-dangerous characters
            clean[key] = re.sub(r"[;'\"\-\-]", "", str(value))

    return clean


def _compile_sql(template: str, params: dict) -> str:
    """Bind sanitized parameters into the SQL template using str.format()."""
    try:
        return template.format(**params)
    except KeyError as exc:
        raise ValueError(f"Missing template parameter: {exc}")


def _expand_combined_hazards(sql: str) -> str:
    """
    Expand single-value hazard_type filters to IN clauses for hazards that NOAA
    splits across multiple event type codes representing the same phenomenon.

    Only Heat / Excessive Heat are combined — these are two NOAA codes for the
    same hazard; all other hazard types remain as single-value equality filters.
    """
    sql = sql.replace(
        "hazard_type = 'Excessive Heat'",
        "hazard_type IN ('Excessive Heat', 'Heat')",
    ).replace(
        "hazard_type = 'Heat'",
        "hazard_type IN ('Excessive Heat', 'Heat')",
    )
    return sql


def _enforce_guardrails(sql: str) -> str:
    """
    Final safety checks on compiled SQL:
    - Must reference only gold_hazard tables
    - Must contain a LIMIT clause
    - Must not contain DDL or dangerous keywords
    """
    sql_upper = sql.upper()

    forbidden = ["DROP ", "DELETE ", "INSERT ", "UPDATE ", "CREATE ", "ALTER ", "TRUNCATE "]
    for keyword in forbidden:
        if keyword in sql_upper:
            raise ValueError(f"Forbidden SQL keyword: {keyword.strip()}")

    if "LIMIT" not in sql_upper:
        sql = sql.rstrip("; \n") + f"\nLIMIT {MAX_LIMIT};"

    if "GOLD_HAZARD" not in sql_upper:
        raise ValueError("Query must reference gold_hazard database (Gold-layer only)")

    return sql


def run_query(
    question: str,
    limit: int = 20,
    s3_output: str = None,
    region: str = "us-east-1",
) -> dict:
    """
    Full governed query pipeline:
    1. Classify question intent → SQL template
    2. Sanitize + bind parameters
    3. Enforce safety guardrails
    4. Execute via Athena with cost cap
    5. Return structured results

    Returns:
        dict with keys: results, sql_executed, intent, scan_bytes, row_count
    """
    # 1. Classify intent
    intent: QueryIntent = classify_intent(question, default_limit=min(limit, MAX_LIMIT))
    logger.info("Intent: template=%s params=%s", intent.template, intent.params)

    # 2. Load + compile template
    raw_template = _load_template(intent.template)
    clean_params = _sanitize_params(intent.params)
    compiled_sql = _compile_sql(raw_template, clean_params)

    # 3. Expand combined hazard types (e.g. Heat + Excessive Heat), then guardrails
    compiled_sql = _expand_combined_hazards(compiled_sql)
    safe_sql = _enforce_guardrails(compiled_sql)
    logger.info("Compiled SQL:\n%s", safe_sql)

    # 4. Execute via Athena (raw boto3 — no awswrangler dependency)
    output_location = (
        s3_output
        or os.environ.get("ATHENA_OUTPUT_LOCATION", "s3://aws-hazard-risk-vigamogh-dev/athena-results/")
    )

    athena = boto3.client("athena", region_name=region)
    start_resp = athena.start_query_execution(
        QueryString=safe_sql,
        QueryExecutionContext={"Database": ALLOWED_DATABASE},
        ResultConfiguration={"OutputLocation": output_location},
    )
    execution_id = start_resp["QueryExecutionId"]

    # Poll until terminal state
    for _ in range(60):
        status_resp = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status_resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        reason = status_resp["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query {state}: {reason}")

    # Paginate results
    paginator = athena.get_paginator("get_query_results")
    rows_all = []
    col_names = None
    for page in paginator.paginate(QueryExecutionId=execution_id):
        page_rows = page["ResultSet"]["Rows"]
        if col_names is None:
            col_names = [c["VarCharValue"] for c in page_rows[0]["Data"]]
            page_rows = page_rows[1:]  # skip header row
        for row in page_rows:
            values = [c.get("VarCharValue", "") for c in row["Data"]]
            rows_all.append(dict(zip(col_names, values)))

    results = rows_all
    logger.info("Query returned %d rows", len(results))

    # For templates with a hardcoded ORDER BY (not a {order_col} parameter), inject
    # the sort column name so the TAG prompt can tell the LLM what the ranking criterion is.
    _FIXED_ORDER_COLS = {
        "hazard_event_increase": "absolute_increase",
        "largest_increase":      "absolute_increase",
    }
    order_col = clean_params.get("order_col", "") or _FIXED_ORDER_COLS.get(intent.template, "")

    return {
        "results": results,
        "sql_executed": safe_sql,
        "intent": intent.template,
        "row_count": len(results),
        "tool": "query",
        "order_col": order_col,
    }


# Columns in risk_feature_mart that carry NOAA per-event metrics.
# These are all-hazard aggregates — there is no per-hazard filter on this table.
_NOAA_METRIC_COLS = [
    "total_events", "avg_property_damage", "total_fatalities",
    "fema_property_damage", "fema_claim_count",
]

# Gold-layer canonical hazard names (from hazard_event_summary_current).
# Surfaced in data quality notes so the LLM can give the user correct terminology.
_DB_HAZARD_TYPES = (
    "Drought, Heavy Snow, Tropical Storm, Tornado, Heavy Rain, Funnel Cloud, "
    "High Wind, Excessive Heat, Thunderstorm Wind, Flash Flood, Flood, "
    "Debris Flow, Dust Devil, Heat, Dust Storm, Dense Fog, Hail, Lightning, "
    "Strong Wind, Wildfire"
)


def _data_quality_note(question: str, results: list, intent_template: str) -> str:
    """
    Detect known Gold-layer data limitation patterns and return a plain-English
    note for the LLM to relay to the user.  Returns "" when no issue detected.

    Patterns detected:
    1. User named a hazard but query runs against risk_feature_mart (all-hazard NRI/FEMA).
    2. All NOAA event/damage columns are zero across all rows.
    3. Wildfire query against hazard_event_summary — NOAA systematically underreports fires.
    4. User asked about "risk/vulnerability" for a hazard, results show event counts only.
    """
    if not results:
        return ""

    from analytics.intent_classifier import _HAZARD_SYNONYMS

    q_lower = question.lower()
    # Identify the hazard the user mentioned and its canonical DB name
    db_hazard = None
    user_term = None
    for phrase in sorted(_HAZARD_SYNONYMS, key=len, reverse=True):
        if phrase in q_lower:
            user_term = phrase
            db_hazard = _HAZARD_SYNONYMS[phrase]
            break

    # Pattern 1: user named a specific hazard, but risk_feature_mart stores aggregate NRI/FEMA
    # metrics across ALL hazard types.  Skip this note when the intent routed to
    # hazard_event_increase, which queries hazard_event_summary with a hazard_type filter
    # and therefore DOES return hazard-specific results.
    if db_hazard and intent_template not in ("hazard_event_increase", "fema_declarations_by_state", "hazard_trend_specific", "hazard_trend_by_feature", "top_counties_by_hazard"):
        return (
            f"DATA LIMITATION — must tell the user: "
            f"The user asked about '{user_term}' (Gold-layer canonical term: "
            f"'{db_hazard}'). All analytics queries run against risk_feature_mart, which "
            f"stores composite NRI scores (Expected Annual Loss, Social Vulnerability, "
            f"Resilience) and FEMA declaration counts aggregated across ALL hazard types — "
            f"it does NOT filter by individual hazard type. "
            f"The results shown therefore reflect all-hazard risk, NOT '{user_term}'-specific "
            f"risk. Do NOT attribute the EAL scores or rankings specifically to '{user_term}'. "
            f"Instead explain: (1) the data shows all-hazard composite risk; "
            f"(2) '{user_term}' in this dataset is called '{db_hazard}'; "
            f"(3) per-hazard event breakdowns are available in a separate event summary table "
            f"but are not surfaced in this query. "
            f"Gold-layer hazard types: {_DB_HAZARD_TYPES}."
        )

    # Pattern 3: Wildfire queries against hazard_event_summary — NOAA Storm Events
    # systematically underreports wildfires because fire data is primarily tracked by
    # USFS, CAL FIRE, and NIFC, not NOAA. Warn the user so they don't treat these
    # event counts as a complete picture of wildfire risk.
    if db_hazard == "Wildfire" and intent_template in (
        "top_counties_by_hazard", "hazard_event_increase", "hazard_trend_specific",
    ):
        return (
            "DATA LIMITATION — must tell the user: "
            "These results are sourced from the NOAA Storm Events database, which "
            "systematically underreports wildfires. NOAA is a meteorological agency; "
            "comprehensive wildfire records are maintained by USFS, CAL FIRE, and the "
            "National Interagency Fire Center (NIFC) — none of which feed into NOAA Storm "
            "Events. A county only appears in this data if a local NWS Weather Forecast "
            "Office filed a storm event report for a fire, so major fire counties (e.g., "
            "Butte County CA — 2018 Camp Fire; Shasta, Plumas, Mariposa CA) may rank low "
            "or not appear at all. The event counts shown reflect NOAA-reported fire "
            "incidents, NOT total wildfire destruction or acreage. Treat these rankings as "
            "a lower-bound indicator only, and recommend the user consult NIFC or CAL FIRE "
            "data for a complete wildfire picture."
        )

    # Pattern 4: user asked about "risk" or "vulnerability" for a specific hazard,
    # but the results show event counts (top_counties_by_hazard).
    # Event frequency is a useful proxy, but it is NOT the same as composite risk —
    # it doesn't capture community exposure, social vulnerability, or resilience.
    _RISK_VOCAB = ("at risk", "vulnerable", "vulnerability", "risk level", "risk score")
    if (
        db_hazard
        and intent_template == "top_counties_by_hazard"
        and any(phrase in q_lower for phrase in _RISK_VOCAB)
    ):
        return (
            f"CONTEXT NOTE — tell the user: "
            f"The user asked about '{user_term}' risk or vulnerability. These results show "
            f"NOAA Storm Event counts for '{db_hazard}' — counties ranked by how often this "
            f"hazard has occurred. Event frequency is a useful indicator of hazard exposure, "
            f"but composite risk also depends on community social vulnerability, built "
            f"environment exposure, and resilience capacity (all captured in the NRI). "
            f"Explain this distinction briefly, then present the event-count rankings as the "
            f"best available hazard-specific data."
        )

    # Pattern 2: any template where all NOAA event/damage columns are zero.
    present_cols = [c for c in _NOAA_METRIC_COLS if c in results[0]]
    if present_cols:
        all_zero = all(
            float(r.get(col) or 0) == 0
            for col in present_cols
            for r in results
        )
        if all_zero:
            return (
                f"DATA LIMITATION — must tell the user: "
                f"The columns {present_cols} are all zero in these results. "
                f"The Gold-layer mart stores aggregate NRI scores and FEMA declaration "
                f"counts but does not carry detailed per-event NOAA damage figures for "
                f"this query. The NRI expected-loss and FEMA declaration data shown above "
                f"is still valid and useful."
            )

    return ""


def run_tag_query(
    question: str,
    synthesize_fn,
    limit: int = 20,
    s3_output: str = None,
    region: str = "us-east-1",
) -> dict:
    """
    Table Augmented Generation (TAG) pipeline:
    1. Run the governed Athena query via run_query()
    2. Detect data quality issues and build a plain-English note for the LLM
    3. Pass results + note to the LLM for narrative synthesis
    4. Return both the raw data and the synthesized analyst answer

    Args:
        question:      User's natural-language question.
        synthesize_fn: Callable(system_prompt, user_message) → str.
                       Injected to keep query_engine.py free of boto3 imports.
        limit:         Max rows to return from Athena.
        s3_output:     S3 path for Athena result staging.
        region:        AWS region.

    Returns:
        dict with all keys from run_query() plus:
          - answer:        LLM-synthesized narrative (TAG output)
          - tag_enabled:   True — flag for downstream consumers
    """
    from rag.prompts.tag_template import TAG_SYSTEM_PROMPT, build_tag_prompt

    # Step 1: governed Athena query
    query_result = run_query(question, limit=limit, s3_output=s3_output, region=region)

    results = query_result["results"]
    sql = query_result["sql_executed"]
    intent = query_result["intent"]
    order_col = query_result.get("order_col", "")
    row_count = query_result["row_count"]

    # Step 1b: for fatality/injury sorts, drop rows where the sort column is 0.
    # Done in Python (not SQL HAVING) to avoid Athena alias-in-HAVING edge cases.
    # If all rows are zero the result set becomes empty and the no-data handler fires,
    # which is the correct behaviour: don't present "0 fatalities" as a ranking answer.
    if order_col in ("total_fatalities", "total_injuries") and results:
        results = [r for r in results if float(r.get(order_col) or 0) > 0]
        query_result = {**query_result, "results": results, "row_count": len(results)}
        row_count = len(results)

    # Step 2: graceful no-rows handling
    if not results:
        logger.info("TAG synthesis skipped: no rows returned")
        if order_col in ("total_fatalities", "total_injuries"):
            metric = "fatalities" if order_col == "total_fatalities" else "injuries"
            no_data_msg = (
                f"No counties with recorded {metric} were found for this hazard type "
                f"in the NOAA Storm Events data (2010–2023). NOAA Storm Events may not "
                f"capture all {metric} for this hazard — for example, wildfire, tropical "
                f"storm, and heat fatalities are often underreported or attributed to "
                f"different event types. Try querying by total events instead, or check "
                f"a primary source (e.g. NIFC for wildfire, NHC for tropical storms)."
            )
        else:
            no_data_msg = (
                "No data was found for this query. The Gold-layer dataset may not "
                "contain records matching your filters (hazard type, year range, or "
                "county). Try broadening your search."
            )
        return {
            **query_result,
            "answer": no_data_msg,
            "tag_enabled": True,
        }

    # Step 3: detect data quality / scope issues; inject a note for the LLM
    data_note = _data_quality_note(question, results, intent)
    if data_note:
        logger.info("Data quality note injected for intent=%s", intent)

    user_message = build_tag_prompt(
        question=question,
        results=results,
        sql_executed=sql,
        intent=intent,
        row_count=row_count,
        data_note=data_note,
        order_col=order_col,
    )

    try:
        answer = synthesize_fn(TAG_SYSTEM_PROMPT, user_message)
        logger.info("TAG synthesis complete (%d chars)", len(answer))
    except Exception as exc:
        logger.warning("TAG synthesis failed, returning raw results: %s", exc)
        answer = f"Analytics results ({row_count} rows). LLM synthesis unavailable: {exc}"

    return {**query_result, "answer": answer, "tag_enabled": True}


if __name__ == "__main__":
    questions = [
        "Show the top 10 counties by risk from 2015 to 2023",
        "Which counties saw the largest increase in flood events from 2010–2022?",
        "Compare Harris County vs Miami-Dade over the last 5 years",
    ]
    for q in questions:
        intent = classify_intent(q)
        print(f"Q: {q}")
        print(f"  → template={intent.template} | params={intent.params}\n")
