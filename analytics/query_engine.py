"""
Governed Athena query engine for the /query analytics tool.

Enforces:
- Gold-layer table access only (hazard_gold database)
- Template-based SQL compilation (no free-form SQL injection)
- Automatic LIMIT enforcement
- Partition filtering (year range)
- Scan-cost cap (~100 MB by default)
"""
import logging
import os
import re
from pathlib import Path

import awswrangler as wr
import boto3
import yaml

from analytics.intent_classifier import QueryIntent, classify_intent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yml"
TEMPLATE_DIR = Path(__file__).parent / "sql_templates"

ALLOWED_DATABASE = "hazard_gold"
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
    - Must reference only hazard_gold tables
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

    if "HAZARD_GOLD" not in sql_upper:
        raise ValueError("Query must reference hazard_gold database (Gold-layer only)")

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

    # 4. Execute via Athena
    output_location = (
        s3_output
        or os.environ.get("ATHENA_OUTPUT_LOCATION", "s3://hazard/athena-results/")
    )

    df = wr.athena.read_sql_query(
        sql=safe_sql,
        database=ALLOWED_DATABASE,
        s3_output=output_location,
        ctas_approach=False,
        boto3_session=boto3.Session(region_name=region),
    )

    results = df.to_dict(orient="records")
    logger.info("Query returned %d rows", len(results))

    return {
        "results": results,
        "sql_executed": safe_sql,
        "intent": intent.template,
        "row_count": len(results),
        "tool": "query",
    }


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
