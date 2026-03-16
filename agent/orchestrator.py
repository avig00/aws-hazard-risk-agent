"""
Agent orchestrator: top-level loop that routes questions,
calls the appropriate tools, and synthesizes a unified response.

This is the single entry point used by both the FastAPI /agent endpoint
and the Streamlit app.
"""
import logging
import os
import re
import time

import boto3

from agent.router import RoutingDecision, route
from analytics.query_engine import run_query, run_tag_query
from ml.inference.inference_service import predict_risk
from rag.prompts.ask_template import SYSTEM_PROMPT, build_ask_prompt, build_citations
from rag.prompts.tag_template import TAG_SYSTEM_PROMPT, build_tag_prompt
from rag.retrieval.retrieve import retrieve_similar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
_STATE_ABBRS = {v for v in _STATE_NAME_TO_ABBR.values()}


def _extract_state_hint(question: str) -> str:
    """
    Extract a U.S. state abbreviation from the question.
    Matches full state names ('Texas') and two-letter abbreviations when
    they appear after a comma or the word 'in' (e.g. 'Harris County, TX').
    Returns a 2-letter abbreviation, or '' if none found.
    """
    q_lower = question.lower()
    # Full state name
    for name, abbr in sorted(_STATE_NAME_TO_ABBR.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(name) + r"\b", q_lower):
            return abbr
    # Two-letter abbreviation after comma or "in " (e.g. "Harris County, TX")
    m = re.search(r"(?:,\s*|\bin\s+)([A-Z]{2})\b", question)
    if m and m.group(1) in _STATE_ABBRS:
        return m.group(1)
    return ""


def _extract_county_name(question: str):
    """Extract a county name or FIPS code from a question. Returns str or None."""
    # 5-digit FIPS code
    m = re.search(r"\b(\d{5})\b", question)
    if m:
        return m.group(1)
    # "X County" or "X-Y County" pattern — requires uppercase start (proper noun)
    m = re.search(
        r"([A-Z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*)?)\s+[Cc]ounty",
        question,
    )
    if m:
        return m.group(1).strip()
    return None


def _fetch_county_features(
    county_identifier: str,
    region: str = "us-east-1",
    s3_output: str = None,
    state_hint: str = "",
) -> tuple:
    """
    Query Athena for cross-sectional feature data for a county.

    Mirrors the ML training query: aggregates risk_feature_mart_current (FEMA/demographic)
    and hazard_event_summary_current (per-hazard event counts) to produce the same
    feature vector the model was trained on.

    county_identifier is either a 5-digit FIPS string or a county name fragment.
    Handles hyphenated names (e.g. "Miami-Dade" matches "Miami Dade" in county_dim).
    state_hint: 2-letter state abbreviation to narrow lookup (prevents e.g. "Harris"
                matching "Harrison County, MO" when user clearly means Harris County, TX).
    Returns (features_dict, county_name, county_fips).
    Returns ({}, county_identifier, "") on failure or no results.
    """
    if s3_output is None:
        s3_output = os.environ.get(
            "ATHENA_OUTPUT_LOCATION",
            "s3://aws-hazard-risk-vigamogh-dev/hazard/athena-results/",
        )

    if re.match(r"^\d{5}$", county_identifier):
        county_filter = f"r.county_fips = '{county_identifier}'"
    else:
        safe_name = re.sub(r"[^a-zA-Z \-]", "", county_identifier)[:40].strip()
        # Handle hyphenated names: try exact match, hyphen→space variant, and prefix LIKE.
        # Use safe_lower (e.g. 'miami-dade%') not just the first word ('miami%') to avoid
        # matching unrelated counties like Miami County, OH when asking for Miami-Dade, FL.
        safe_lower = safe_name.lower()
        safe_spaces = safe_lower.replace("-", " ")
        county_filter = (
            f"(LOWER(d.county_name) = '{safe_lower}' OR "
            f"LOWER(d.county_name) = '{safe_spaces}' OR "
            f"LOWER(d.county_name) LIKE '{safe_lower}%' OR "
            f"LOWER(d.county_name) LIKE '{safe_spaces}%')"
        )
        # Narrow by state when the question provides a clear state reference.
        # Prevents ambiguous names (e.g. "Harris" → Harrison MO) when state context exists.
        if state_hint and re.match(r"^[A-Z]{2}$", state_hint):
            county_filter += f" AND UPPER(d.state) = '{state_hint}'"

    sql = f"""
WITH base AS (
    SELECT
        r.county_fips,
        d.county_name,
        d.state,
        AVG(r.population_total)              AS population_total,
        AVG(r.median_household_income)       AS median_household_income,
        AVG(r.median_home_value)             AS median_home_value,
        AVG(r.in_labor_force)                AS in_labor_force,
        AVG(r.unemployed)                    AS unemployed,
        AVG(r.education_universe_total)      AS education_universe_total,
        AVG(r.high_school_grad)              AS high_school_grad,
        AVG(r.bachelors)                     AS bachelors,
        AVG(r.graduate_degree)               AS graduate_degree
    FROM gold_hazard.risk_feature_mart_current r
    JOIN gold_hazard.county_dim d ON r.county_fips = d.county_fips
    WHERE {county_filter}
    GROUP BY r.county_fips, d.county_name, d.state
    LIMIT 1
),
events AS (
    SELECT
        e.county_fips,
        SUM(e.event_count)                                                              AS noaa_event_count,
        SUM(e.total_fatalities)                                                         AS noaa_total_fatalities,
        SUM(e.total_injuries)                                                           AS noaa_total_injuries,
        SUM(CASE WHEN e.hazard_type IN ('Flood','Flash Flood','Heavy Rain')
                 THEN e.event_count ELSE 0 END)                                        AS flood_events,
        SUM(CASE WHEN e.hazard_type IN ('High Wind','Strong Wind','Thunderstorm Wind')
                 THEN e.event_count ELSE 0 END)                                        AS wind_events,
        SUM(CASE WHEN e.hazard_type IN ('Tornado','Funnel Cloud')
                 THEN e.event_count ELSE 0 END)                                        AS tornado_events,
        SUM(CASE WHEN e.hazard_type = 'Hail'
                 THEN e.event_count ELSE 0 END)                                        AS hail_events,
        SUM(CASE WHEN e.hazard_type = 'Lightning'
                 THEN e.event_count ELSE 0 END)                                        AS lightning_events,
        SUM(CASE WHEN e.hazard_type = 'Debris Flow'
                 THEN e.event_count ELSE 0 END)                                        AS debris_flow_events,
        SUM(CASE WHEN e.hazard_type = 'Wildfire'
                 THEN e.event_count ELSE 0 END)                                        AS wildfire_events,
        SUM(CASE WHEN e.hazard_type IN ('Excessive Heat','Heat')
                 THEN e.event_count ELSE 0 END)                                        AS heat_events,
        SUM(CASE WHEN e.hazard_type = 'Tropical Storm'
                 THEN e.event_count ELSE 0 END)                                        AS tropical_events,
        SUM(CASE WHEN e.hazard_type = 'Heavy Snow'
                 THEN e.event_count ELSE 0 END)                                        AS winter_events
    FROM gold_hazard.hazard_event_summary_current e
    WHERE e.county_fips IN (SELECT county_fips FROM base)
    GROUP BY e.county_fips
)
SELECT
    b.county_fips,
    b.county_name,
    b.state,
    b.population_total,
    b.median_household_income,
    b.median_home_value,
    b.in_labor_force,
    b.unemployed,
    b.education_universe_total,
    b.high_school_grad,
    b.bachelors,
    b.graduate_degree,
    COALESCE(e.noaa_event_count, 0)        AS noaa_event_count,
    COALESCE(e.noaa_total_fatalities, 0)   AS noaa_total_fatalities,
    COALESCE(e.noaa_total_injuries, 0)     AS noaa_total_injuries,
    COALESCE(e.flood_events, 0)            AS flood_events,
    COALESCE(e.wind_events, 0)             AS wind_events,
    COALESCE(e.tornado_events, 0)          AS tornado_events,
    COALESCE(e.hail_events, 0)             AS hail_events,
    COALESCE(e.lightning_events, 0)        AS lightning_events,
    COALESCE(e.debris_flow_events, 0)      AS debris_flow_events,
    COALESCE(e.wildfire_events, 0)         AS wildfire_events,
    COALESCE(e.heat_events, 0)             AS heat_events,
    COALESCE(e.tropical_events, 0)         AS tropical_events,
    COALESCE(e.winter_events, 0)           AS winter_events,
    CASE WHEN b.in_labor_force > 0
         THEN b.unemployed / b.in_labor_force END               AS unemployment_rate,
    CASE WHEN b.education_universe_total > 0
         THEN b.bachelors / b.education_universe_total END       AS bachelors_rate,
    CASE WHEN b.population_total > 0
         THEN COALESCE(e.noaa_event_count, 0) / b.population_total END AS events_per_capita
FROM base b
LEFT JOIN events e ON b.county_fips = e.county_fips
"""

    try:
        client = boto3.client("athena", region_name=region)
        resp = client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": "gold_hazard"},
            ResultConfiguration={"OutputLocation": s3_output},
        )
        exec_id = resp["QueryExecutionId"]

        for _ in range(60):
            status = client.get_query_execution(QueryExecutionId=exec_id)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(2)

        if state != "SUCCEEDED":
            logger.warning("Athena county lookup failed: state=%s", state)
            return {}, county_identifier, ""

        result = client.get_query_results(QueryExecutionId=exec_id)
        rows = result["ResultSet"]["Rows"]
        if len(rows) < 2:
            return {}, county_identifier, ""

        headers = [c["VarCharValue"] for c in rows[0]["Data"]]
        vals = [c.get("VarCharValue", "") for c in rows[1]["Data"]]
        row_dict = dict(zip(headers, vals))

        county_name = row_dict.get("county_name", county_identifier)
        county_fips = row_dict.get("county_fips", "")

        features = {}
        for k, v in row_dict.items():
            if v == "":
                features[k] = None  # becomes NaN in DataFrame; fillna(0) handles it
            else:
                try:
                    features[k] = float(v)
                except (ValueError, TypeError):
                    features[k] = v  # keep string for categorical cols (e.g. "state")

        logger.info("Fetched features for county=%s fips=%s", county_name, county_fips)
        return features, county_name, county_fips

    except Exception as exc:
        logger.warning("County features lookup error: %s", exc)
        return {}, county_identifier, ""


def run_agent(
    question: str,
    top_k: int = 5,
    limit: int = 20,
    bedrock_call_fn=None,
    pinecone_api_key: str = None,
    sagemaker_endpoint: str = "hazard-risk-model",
) -> dict:
    """
    Full agent execution loop:
    1. Route question to tool(s)
    2. Execute tool calls
    3. Synthesize response

    Args:
        question: User's natural-language question.
        top_k: Number of RAG chunks to retrieve.
        limit: Max rows for analytics queries.
        bedrock_call_fn: Callable(system, user_message) → str for LLM synthesis.
                         Must be injected (avoids circular import with app.py).
        pinecone_api_key: Pinecone API key; reads PINECONE_API_KEY env var if None.
        sagemaker_endpoint: SageMaker endpoint name for /predict calls.

    Returns:
        dict with keys: answer, tool_used, data, sources, routing_reason
    """
    decision: RoutingDecision = route(question)
    logger.info("Routing: %s → tools=%s", question[:80], decision.tools)

    result = {
        "question": question,
        "routing": {
            "tools": decision.tools,
            "reason": decision.reasoning,
            "is_hybrid": decision.is_hybrid,
        },
    }

    tool_outputs = {}

    # ── Execute each tool ─────────────────────────────────────────────────────
    if "predict" in decision.tools:
        try:
            features = {}
            county_name = ""
            county_fips = ""
            county_id = _extract_county_name(question)
            if county_id:
                state_hint = _extract_state_hint(question)
                features, county_name, county_fips = _fetch_county_features(
                    county_id, state_hint=state_hint
                )
            pred = predict_risk(features=features, endpoint_name=sagemaker_endpoint)
            pred["county_name"] = county_name
            pred["county_fips"] = county_fips
            pred["county_state"] = features.get("state", "")
            if not county_id:
                pred["_no_county"] = True
            tool_outputs["predict"] = pred
        except Exception as exc:
            logger.warning("Predict tool failed: %s", exc)
            tool_outputs["predict"] = {"error": str(exc)}

    if "query" in decision.tools:
        try:
            if bedrock_call_fn:
                # TAG: Athena results + LLM synthesis in one step
                query_result = run_tag_query(
                    question=question,
                    synthesize_fn=bedrock_call_fn,
                    limit=limit,
                )
            else:
                # No LLM available — raw Athena results only
                query_result = run_query(question, limit=limit)
            tool_outputs["query"] = query_result
        except Exception as exc:
            logger.warning("Query tool failed: %s", exc)
            tool_outputs["query"] = {"error": str(exc)}

    if "ask" in decision.tools:
        try:
            chunks = retrieve_similar(
                question=question,
                k=top_k,
                pinecone_api_key=pinecone_api_key,
            )
            tool_outputs["ask"] = {
                "chunks": chunks,
                "citations": build_citations(chunks),
            }
        except Exception as exc:
            logger.warning("Ask/retrieve tool failed: %s", exc)
            tool_outputs["ask"] = {"error": str(exc), "chunks": [], "citations": []}

    result["tool_outputs"] = tool_outputs

    # ── Synthesize answer ─────────────────────────────────────────────────────
    query_out = tool_outputs.get("query", {})
    ask_out = tool_outputs.get("ask", {})

    if "query" in decision.tools and "ask" not in decision.tools:
        # Pure query route: TAG already ran inside run_tag_query().
        # Use its answer directly — no second LLM call needed.
        result["answer"] = query_out.get("answer", _fallback_answer(tool_outputs, decision.tools))
        result["sources"] = []

    elif "ask" in decision.tools and "query" not in decision.tools:
        # Pure RAG route: standard ask_template synthesis.
        if bedrock_call_fn:
            chunks = ask_out.get("chunks", [])
            user_message = build_ask_prompt(question, chunks)
            try:
                result["answer"] = bedrock_call_fn(SYSTEM_PROMPT, user_message)
            except Exception as exc:
                logger.error("RAG synthesis failed: %s", exc)
                result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        else:
            result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = ask_out.get("citations", [])

    elif "query" in decision.tools and "ask" in decision.tools:
        # Hybrid route: combine TAG answer (structured data) with RAG chunks.
        # Uses a dedicated hybrid prompt — NOT the ask/RAG system prompt — so the LLM
        # leads with the specific data findings (top-ranked county, event counts, etc.)
        # and then adds contextual explanation from retrieved documents.
        if bedrock_call_fn:
            tag_answer = query_out.get("answer", "")
            rag_chunks = ask_out.get("chunks", [])
            rag_text = "\n\n".join(
                c.get("text", "") for c in rag_chunks[:3] if c.get("text", "").strip()
            )

            hybrid_system = (
                "You are a senior hazard risk analyst combining structured data findings "
                "with document-grounded context.\n\n"
                "Rules:\n"
                "- Open with 1–2 sentences citing the specific county names and figures "
                "from the DATA ANSWER. The top-ranked county MUST be named in your first sentence.\n"
                "- Follow with 1–2 sentences of contextual explanation using the DOCUMENT CONTEXT.\n"
                "- Do NOT re-state or paraphrase the full data table — you are writing a verbal "
                "complement to the table already shown to the user, not a substitute for it.\n"
                "- Do NOT use section headers like 'Data Answer:' or 'Document Context:'.\n"
                "- If no document context was retrieved, answer using only the data findings."
            )

            hybrid_user = (
                f"QUESTION: {question}\n\n"
                f"DATA ANSWER (cite these specific values — lead with the top result):\n"
                f"{tag_answer}\n\n"
                f"DOCUMENT CONTEXT (use for explanation only):\n"
                f"{rag_text or 'No documents retrieved.'}"
            )

            try:
                result["answer"] = bedrock_call_fn(hybrid_system, hybrid_user)
            except Exception as exc:
                logger.error("Hybrid synthesis failed: %s", exc)
                result["answer"] = tag_answer or _fallback_answer(tool_outputs, decision.tools)
        else:
            result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = ask_out.get("citations", [])

    elif "predict" in decision.tools and "query" not in decision.tools and "ask" not in decision.tools:
        # Pure predict route
        pred_out = tool_outputs.get("predict", {})
        if "error" in pred_out:
            result["answer"] = f"Prediction error: {pred_out['error']}"
        elif pred_out.get("_no_county"):
            result["answer"] = (
                "Please specify a county name to get an ML risk prediction. "
                "Example: *What is the predicted risk for Harris County, TX?*"
            )
        elif "risk_tier" in pred_out:
            county = pred_out.get("county_name") or pred_out.get("county_fips") or "the county"
            state = pred_out.get("county_state", "")
            county_label = f"{county}, {state}" if state else county
            result["answer"] = (
                f"**ML Risk Prediction**\n\n"
                f"Based on the XGBoost classifier trained on NOAA storm history, "
                f"demographics, and NRI risk scores:\n\n"
                f"**{county_label}** → predicted risk tier: **{pred_out['risk_tier']}**"
            )
        else:
            result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = []

    else:
        result["answer"] = _fallback_answer(tool_outputs, decision.tools)
        result["sources"] = []

    result["tool_used"] = decision.tools
    return result


def _format_table_as_text(rows: list) -> str:
    """Convert a list of dicts (Athena results) to a readable text table."""
    if not rows:
        return "No results."
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def _fallback_answer(tool_outputs: dict, tools: list) -> str:
    """Return a plain-text summary when LLM synthesis is unavailable."""
    parts = []
    if "query" in tools and "results" in tool_outputs.get("query", {}):
        rows = tool_outputs["query"]["results"]
        parts.append(f"Analytics results ({len(rows)} rows returned):")
        parts.append(_format_table_as_text(rows[:5]))
    if "predict" in tools and "risk_tier" in tool_outputs.get("predict", {}):
        p = tool_outputs["predict"]
        county = p.get("county_name") or p.get("county_fips") or "County"
        parts.append(f"ML Prediction: {county} → {p['risk_tier']}")
    if "ask" in tools and tool_outputs.get("ask", {}).get("error"):
        parts.append(f"Retrieval error: {tool_outputs['ask']['error']}")
    return "\n\n".join(parts) if parts else "No results available."
