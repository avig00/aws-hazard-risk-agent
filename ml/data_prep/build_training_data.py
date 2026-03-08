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
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

S3_CLIENT = None


def _s3():
    global S3_CLIENT
    if S3_CLIENT is None:
        S3_CLIENT = boto3.client("s3", region_name="us-east-1")
    return S3_CLIENT


def _s3_put_parquet(df: pd.DataFrame, s3_path: str) -> None:
    """Write DataFrame as Parquet to S3 without awswrangler (avoids Ray)."""
    # s3_path format: s3://bucket/key
    _, _, rest = s3_path.partition("s3://")
    bucket, _, key = rest.partition("/")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info("Wrote s3://%s/%s (%d bytes)", bucket, key, buf.tell())

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
    logger.info("Querying Athena: cross-sectional from risk_feature_mart_current + hazard_event_summary_current")
    query = f"""
        WITH base AS (
            -- Cross-sectional FEMA/demographic features from risk_feature_mart_current
            SELECT
                r.county_fips,
                d.state,
                SUM(r.fema_valid_registrations)          AS fema_valid_registrations,
                SUM(r.fema_total_damage)                 AS fema_total_damage,
                SUM(r.fema_total_approved_ihp_amount)    AS fema_total_approved_ihp_amount,
                SUM(r.fema_declaration_count)            AS fema_declaration_count,
                SUM(r.fema_repair_replace_amount)        AS fema_repair_replace_amount,
                SUM(r.fema_rental_amount)                AS fema_rental_amount,
                SUM(r.fema_other_needs_amount)           AS fema_other_needs_amount,
                SUM(r.fema_total_inspected)              AS fema_total_inspected,
                AVG(r.population_total)                  AS population_total,
                AVG(r.median_household_income)           AS median_household_income,
                AVG(r.median_home_value)                 AS median_home_value,
                AVG(r.in_labor_force)                    AS in_labor_force,
                AVG(r.unemployed)                        AS unemployed,
                AVG(r.education_universe_total)          AS education_universe_total,
                AVG(r.high_school_grad)                  AS high_school_grad,
                AVG(r.bachelors)                         AS bachelors,
                AVG(r.graduate_degree)                   AS graduate_degree,
                MAX(r.nri_eal_score)                     AS nri_eal_score,
                MAX(r.nri_risk_score)                    AS nri_risk_score,
                MAX(r.nri_sovi_score)                    AS nri_sovi_score,
                MAX(r.nri_resl_score)                    AS nri_resl_score
            FROM {database}.risk_feature_mart_current r
            LEFT JOIN {database}.county_dim d ON r.county_fips = d.county_fips
            WHERE r.nri_eal_score IS NOT NULL
            GROUP BY r.county_fips, d.state
        ),
        events AS (
            -- Per-hazard-type event pivots from hazard_event_summary_current
            -- Including total NOAA property damage (from NOAA estimates, independent of FEMA)
            SELECT
                county_fips,
                SUM(event_count)                                              AS noaa_event_count,
                SUM(total_fatalities)                                         AS noaa_total_fatalities,
                SUM(total_injuries)                                           AS noaa_total_injuries,
                AVG(avg_property_damage)                                      AS noaa_avg_property_damage,
                -- Total NOAA property damage estimate (events × avg_damage per type)
                SUM(CAST(event_count AS double) * COALESCE(avg_property_damage, 0)) AS noaa_total_property_damage,
                -- Flood-related
                SUM(CASE WHEN hazard_type IN ('Flood','Flash Flood','Heavy Rain')
                         THEN event_count ELSE 0 END)                         AS flood_events,
                SUM(CASE WHEN hazard_type IN ('Flood','Flash Flood','Heavy Rain')
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS flood_total_damage,
                -- Wind events
                SUM(CASE WHEN hazard_type IN ('High Wind','Strong Wind','Thunderstorm Wind')
                         THEN event_count ELSE 0 END)                         AS wind_events,
                SUM(CASE WHEN hazard_type IN ('High Wind','Strong Wind','Thunderstorm Wind')
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS wind_total_damage,
                -- Tornado
                SUM(CASE WHEN hazard_type IN ('Tornado','Funnel Cloud')
                         THEN event_count ELSE 0 END)                         AS tornado_events,
                SUM(CASE WHEN hazard_type IN ('Tornado','Funnel Cloud')
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS tornado_total_damage,
                -- Hail (second-highest event volume — now includes damage total)
                SUM(CASE WHEN hazard_type = 'Hail'
                         THEN event_count ELSE 0 END)                         AS hail_events,
                SUM(CASE WHEN hazard_type = 'Hail'
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS hail_total_damage,
                -- Lightning (6 K+ events — storm intensity proxy)
                SUM(CASE WHEN hazard_type = 'Lightning'
                         THEN event_count ELSE 0 END)                         AS lightning_events,
                SUM(CASE WHEN hazard_type = 'Lightning'
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS lightning_total_damage,
                -- Debris Flow (1 400+ events — slope/wildfire-adjacent risk)
                SUM(CASE WHEN hazard_type = 'Debris Flow'
                         THEN event_count ELSE 0 END)                         AS debris_flow_events,
                SUM(CASE WHEN hazard_type = 'Debris Flow'
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS debris_flow_total_damage,
                -- Wildfire
                SUM(CASE WHEN hazard_type = 'Wildfire'
                         THEN event_count ELSE 0 END)                         AS wildfire_events,
                SUM(CASE WHEN hazard_type = 'Wildfire'
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS wildfire_total_damage,
                -- Heat
                SUM(CASE WHEN hazard_type IN ('Excessive Heat','Heat')
                         THEN event_count ELSE 0 END)                         AS heat_events,
                -- Tropical
                SUM(CASE WHEN hazard_type = 'Tropical Storm'
                         THEN event_count ELSE 0 END)                         AS tropical_events,
                SUM(CASE WHEN hazard_type = 'Tropical Storm'
                         THEN CAST(event_count AS double) * COALESCE(avg_property_damage,0)
                         ELSE 0 END)                                          AS tropical_total_damage,
                -- Winter
                SUM(CASE WHEN hazard_type = 'Heavy Snow'
                         THEN event_count ELSE 0 END)                         AS winter_events
            FROM {database}.hazard_event_summary_current
            GROUP BY county_fips
        )
        SELECT
            b.*,
            COALESCE(e.noaa_event_count, 0)            AS noaa_event_count,
            COALESCE(e.noaa_total_fatalities, 0)       AS noaa_total_fatalities,
            COALESCE(e.noaa_total_injuries, 0)         AS noaa_total_injuries,
            COALESCE(e.noaa_avg_property_damage, 0)    AS noaa_avg_property_damage,
            -- Total NOAA damage (independent of FEMA — key predictor)
            COALESCE(e.noaa_total_property_damage, 0)  AS noaa_total_property_damage,
            -- Per-hazard-type counts
            COALESCE(e.flood_events, 0)                AS flood_events,
            COALESCE(e.flood_total_damage, 0)          AS flood_total_damage,
            COALESCE(e.wind_events, 0)                 AS wind_events,
            COALESCE(e.wind_total_damage, 0)           AS wind_total_damage,
            COALESCE(e.tornado_events, 0)              AS tornado_events,
            COALESCE(e.tornado_total_damage, 0)        AS tornado_total_damage,
            COALESCE(e.hail_events, 0)                 AS hail_events,
            COALESCE(e.hail_total_damage, 0)           AS hail_total_damage,
            COALESCE(e.lightning_events, 0)            AS lightning_events,
            COALESCE(e.lightning_total_damage, 0)      AS lightning_total_damage,
            COALESCE(e.debris_flow_events, 0)          AS debris_flow_events,
            COALESCE(e.debris_flow_total_damage, 0)    AS debris_flow_total_damage,
            COALESCE(e.wildfire_events, 0)             AS wildfire_events,
            COALESCE(e.wildfire_total_damage, 0)       AS wildfire_total_damage,
            COALESCE(e.heat_events, 0)                 AS heat_events,
            COALESCE(e.tropical_events, 0)             AS tropical_events,
            COALESCE(e.tropical_total_damage, 0)       AS tropical_total_damage,
            COALESCE(e.winter_events, 0)               AS winter_events,
            CASE WHEN b.in_labor_force > 0
                 THEN b.unemployed / b.in_labor_force END               AS unemployment_rate,
            CASE WHEN b.education_universe_total > 0
                 THEN b.bachelors / b.education_universe_total END       AS bachelors_rate,
            CASE WHEN b.population_total > 0
                 THEN COALESCE(e.noaa_event_count,0) / b.population_total END AS events_per_capita
        FROM base b
        LEFT JOIN events e ON b.county_fips = e.county_fips
        ORDER BY b.county_fips
    """
    df = _run_athena_query(query, database)
    logger.info("Cross-sectional dataset: %d counties, %d columns", len(df), len(df.columns))
    return df


def add_risk_bucket(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Bin the continuous target into LOW/MEDIUM/HIGH for stratified splitting.

    Falls back to rank-based tertiles when many ties (e.g., many zero-damage counties)
    prevent qcut from forming 3 distinct quantile bins.
    """
    df = df.copy()
    labels = ["LOW", "MEDIUM", "HIGH"]
    try:
        df["risk_bucket"] = pd.qcut(
            df[target_col], q=len(labels), labels=labels, duplicates="drop"
        )
    except ValueError:
        # Too many duplicate values — use rank-based equal-count tertiles
        df["risk_bucket"] = pd.cut(
            df[target_col].rank(method="first"),
            bins=len(labels),
            labels=labels,
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
    # Deduplicate while preserving order (target_col and stratify_col are often the same)
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
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
        _s3_put_parquet(train_df, f"{output_dir}train.parquet")
        _s3_put_parquet(test_df, f"{output_dir}test.parquet")
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
        _, _, rest = f"{output_dir}feature_list.json".partition("s3://")
        bucket, _, key = rest.partition("/")
        _s3().put_object(Bucket=bucket, Key=key, Body=json.dumps(schema, indent=2).encode())
    else:
        with open(f"{output_dir}/feature_list.json", "w") as f:
            json.dump(schema, f, indent=2)
    logger.info("Saved feature_list.json")


def run(output_dir=None):
    config = load_config()
    data_cfg = config["data"]
    feat_cfg = config["features"]

    df = pull_gold_data(data_cfg["database"], data_cfg["table"])
    # risk_bucket is derived from fema_total_damage tertiles (classification target).
    # Skip if the gold layer already contains risk_bucket (e.g. after a data repair that
    # pre-computed the column); otherwise derive it from fema_total_damage.
    if "risk_bucket" not in df.columns:
        df = add_risk_bucket(df, "fema_total_damage")

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
