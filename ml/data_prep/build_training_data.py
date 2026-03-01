"""
Pull Gold-layer hazard risk data from Athena, split into train/test,
and save as Parquet files for ML training.

Gold layer source: s3://aws-hazard-risk-vigamogh-dev/hazard/gold/risk_feature_mart/
"""
import io
import json
import logging
import time
from pathlib import Path

import boto3
import awswrangler as wr
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"
ATHENA_OUTPUT = "s3://aws-hazard-risk-vigamogh-dev/athena-results/"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _run_athena_query(sql: str, database: str) -> pd.DataFrame:
    """Execute SQL via boto3 Athena client and return results as a DataFrame.

    Uses boto3 directly to avoid awswrangler's Ray distributed backend,
    which has a pydantic v2 incompatibility in version 3.x.
    """
    athena = boto3.client("athena", region_name="us-east-1")
    s3 = boto3.client("s3", region_name="us-east-1")

    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )
    execution_id = response["QueryExecutionId"]
    logger.info("Athena query started: %s", execution_id)

    # Poll until complete
    for _ in range(120):  # max 10 minutes
        status = athena.get_query_execution(QueryExecutionId=execution_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
            raise RuntimeError(f"Athena query {state}: {reason}")
        time.sleep(5)
    else:
        raise RuntimeError("Athena query timed out after 10 minutes")

    # Result CSV is at ATHENA_OUTPUT/<execution_id>.csv
    bucket = "aws-hazard-risk-vigamogh-dev"
    key = f"athena-results/{execution_id}.csv"
    logger.info("Reading result from s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def pull_gold_data(database: str, table: str) -> pd.DataFrame:
    """Query the Gold-layer risk feature mart from Athena.

    nri_eal_score is a static FEMA NRI snapshot (same value per county across all years).
    We aggregate NOAA/FEMA event features by county (SUM/AVG across all available years)
    to build a cross-sectional dataset: one row per county.
    This prevents trivial autoregressive prediction and produces a meaningful model.
    """
    logger.info("Querying Athena: %s.%s (cross-sectional aggregate)", database, table)
    query = f"""
        SELECT
            county_fips,
            -- NOAA event totals across all years
            SUM(noaa_event_count)           AS noaa_event_count,
            SUM(noaa_total_fatalities)      AS noaa_total_fatalities,
            SUM(noaa_total_injuries)        AS noaa_total_injuries,
            AVG(noaa_avg_property_damage)   AS noaa_avg_property_damage,
            -- FEMA assistance totals across all years
            SUM(fema_valid_registrations)   AS fema_valid_registrations,
            SUM(fema_total_damage)          AS fema_total_damage,
            SUM(fema_total_approved_ihp_amount) AS fema_total_approved_ihp_amount,
            SUM(fema_declaration_count)     AS fema_declaration_count,
            SUM(fema_repair_replace_amount) AS fema_repair_replace_amount,
            SUM(fema_rental_amount)         AS fema_rental_amount,
            SUM(fema_other_needs_amount)    AS fema_other_needs_amount,
            SUM(fema_total_inspected)       AS fema_total_inspected,
            -- Socioeconomic (stable, use latest available)
            AVG(population_total)           AS population_total,
            AVG(median_household_income)    AS median_household_income,
            AVG(median_home_value)          AS median_home_value,
            AVG(in_labor_force)             AS in_labor_force,
            AVG(unemployed)                 AS unemployed,
            AVG(high_school_grad)           AS high_school_grad,
            AVG(bachelors)                  AS bachelors,
            AVG(graduate_degree)            AS graduate_degree,
            -- NRI sub-scores (static, take max — all values equal per county)
            MAX(nri_sovi_score)             AS nri_sovi_score,
            MAX(nri_resl_score)             AS nri_resl_score,
            -- Target: static NRI EAL score (same value per county)
            MAX(nri_eal_score)              AS nri_eal_score
        FROM {database}.{table}
        WHERE nri_eal_score IS NOT NULL
        GROUP BY county_fips
        ORDER BY county_fips
    """
    df = _run_athena_query(query, database)
    logger.info("Cross-sectional dataset: %d counties, %d columns", len(df), len(df.columns))
    return df


def add_risk_bucket(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Bin the continuous target into LOW/MEDIUM/HIGH for stratified splitting."""
    df = df.copy()
    df["risk_bucket"] = pd.qcut(
        df[target_col],
        q=3,
        labels=["LOW", "MEDIUM", "HIGH"],
        duplicates="drop",
    )
    return df


def split_and_save(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    stratify_col: str,
    train_ratio: float,
    output_dir: str,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split and persist to Parquet (local or S3)."""
    cols = feature_cols + [target_col, stratify_col]
    cols = [c for c in cols if c in df.columns]
    df_clean = df[cols].dropna(subset=[target_col])

    train_df, test_df = train_test_split(
        df_clean,
        train_size=train_ratio,
        stratify=df_clean[stratify_col],
        random_state=random_state,
    )
    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    is_s3 = output_dir.startswith("s3://")
    if is_s3:
        wr.s3.to_parquet(train_df, path=f"{output_dir}train.parquet")
        wr.s3.to_parquet(test_df, path=f"{output_dir}test.parquet")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        test_df.to_parquet(f"{output_dir}/test.parquet", index=False)

    logger.info("Saved train/test Parquet to %s", output_dir)
    return train_df, test_df


def save_feature_list(feature_cols: list, target_col: str, output_dir: str) -> None:
    """Persist the feature schema for downstream use (inference, monitoring)."""
    schema = {"features": feature_cols, "target": target_col}
    is_s3 = output_dir.startswith("s3://")
    if is_s3:
        wr.s3.to_json(
            pd.DataFrame([schema]),
            path=f"{output_dir}feature_list.json",
        )
    else:
        with open(f"{output_dir}/feature_list.json", "w") as f:
            json.dump(schema, f, indent=2)
    logger.info("Saved feature_list.json")


def run(output_dir=None):
    config = load_config()
    data_cfg = config["data"]
    feat_cfg = config["features"]

    df = pull_gold_data(data_cfg["database"], data_cfg["table"])
    df = add_risk_bucket(df, data_cfg["target_column"])

    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])
    feature_cols = [c for c in feature_cols if c in df.columns]

    out = output_dir or data_cfg["output_dir"]
    train_df, test_df = split_and_save(
        df=df,
        feature_cols=feature_cols,
        target_col=data_cfg["target_column"],
        stratify_col=data_cfg["stratify_column"],
        train_ratio=data_cfg["train_ratio"],
        output_dir=out,
        random_state=config["model"].get("random_state", 42),
    )
    save_feature_list(feature_cols, data_cfg["target_column"], out)
    return train_df, test_df


if __name__ == "__main__":
    run()
