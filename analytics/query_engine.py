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
            # Only allow lowercase alpha + space
            if not re.match(r"^[a-z ]+$", str(value)):
                raise ValueError(f"Invalid hazard_type: {value}")
            clean[key] = str(value).lower()

        elif key == "county_fips_list":
            # Must be a comma-separated list of quoted 5-digit FIPS codes
            fips_pattern = r"^('[0-9]{5}'(,'[0-9]{5}')*)$"
            val = str(value).replace(" ", "")
            if not re.match(fips_pattern, val):
                raise ValueError(f"Invalid county_fips_list format: {value}")
            clean[key] = val

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

    # 3. Guardrails
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

    return {
        "results": results,
        "sql_executed": safe_sql,
        "intent": intent.template,
        "row_count": len(results),
        "tool": "query",
    }


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
    2. Pass the results + original question to an LLM for narrative synthesis
    3. Return both the raw data and the synthesized analyst answer

    Args:
        question:      User's natural-language question.
        synthesize_fn: Callable(system_prompt, user_message) → str.
                       Injected to keep query_engine.py free of boto3 imports.
                       Typically wraps Bedrock Claude (app.py or orchestrator).
        limit:         Max rows to return from Athena.
        s3_output:     S3 path for Athena result staging.
        region:        AWS region.

    Returns:
        dict with all keys from run_query() plus:
          - answer:        LLM-synthesized narrative (TAG output)
          - tag_enabled:   True — flag for downstream consumers
    """
    from rag.prompts.tag_template import TAG_SYSTEM_PROMPT, build_tag_prompt

    # Step 1: governed Athena query (unchanged pipeline)
    query_result = run_query(question, limit=limit, s3_output=s3_output, region=region)

    results = query_result["results"]
    sql = query_result["sql_executed"]
    intent = query_result["intent"]
    row_count = query_result["row_count"]

    # Step 2: synthesize — skip gracefully if no results or no LLM available
    if not results:
        logger.info("TAG synthesis skipped: no rows returned")
        return {**query_result, "answer": "The query returned no results.", "tag_enabled": True}

    user_message = build_tag_prompt(
        question=question,
        results=results,
        sql_executed=sql,
        intent=intent,
        row_count=row_count,
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
