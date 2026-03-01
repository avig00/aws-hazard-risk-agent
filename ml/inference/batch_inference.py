"""
Batch inference: score all counties in the Gold layer and write predictions
back to S3 for use by the analytics tool and Streamlit dashboard.

Can run as a standalone script or be triggered by a SageMaker Pipeline
BatchTransformStep / Lambda on a schedule.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import awswrangler as wr
import boto3
import pandas as pd
import yaml

from ml.data_prep.feature_engineering import engineer_features
from ml.inference.inference_service import predict_batch_local, predict_from_endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"
OUTPUT_PATH = "s3://aws-hazard-risk-vigamogh-dev/hazard/gold/batch_predictions/"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_inference_data(database: str, table: str, year: int = None) -> pd.DataFrame:
    """Pull the latest county feature data from Athena for scoring."""
    year_filter = f"AND year = {year}" if year else "AND year = (SELECT MAX(year) FROM {database}.{table})"
    query = f"""
        SELECT *
        FROM {database}.{table}
        WHERE nri_eal_score IS NOT NULL
        {year_filter}
    """
    logger.info("Loading inference data from Athena (year=%s)...", year or "latest")
    df = wr.athena.read_sql_query(sql=query, database=database, ctas_approach=False)
    logger.info("Loaded %d counties for batch scoring", len(df))
    return df


def run_batch_via_endpoint(df: pd.DataFrame, feature_cols: list, endpoint_name: str) -> list:
    """Score counties in batches of 100 via the SageMaker endpoint."""
    results = []
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        payload = {"instances": batch[feature_cols].to_dict(orient="records")}
        try:
            response = predict_from_endpoint(payload, endpoint_name=endpoint_name)
            for j, (score, bucket) in enumerate(
                zip(response["predictions"], response["risk_buckets"])
            ):
                results.append({
                    "county_fips": batch.iloc[j].get("county_fips", ""),
                    "county_name": batch.iloc[j].get("county_name", ""),
                    "state": batch.iloc[j].get("state", ""),
                    "predicted_loss": round(score, 4),
                    "risk_bucket": bucket,
                })
        except Exception as exc:
            logger.error("Batch %d failed: %s", i // batch_size, exc)
    return results


def run_batch_inference(
    input_path: str = None,
    output_path: str = OUTPUT_PATH,
    endpoint_name: str = "hazard-risk-model",
    year: int = None,
    use_endpoint: bool = True,
) -> pd.DataFrame:
    """
    Full batch inference pipeline:
    1. Load county feature data from Athena (or local Parquet)
    2. Apply feature engineering
    3. Score via SageMaker endpoint or local model
    4. Write predictions back to S3 as Parquet partitioned by run_date
    """
    config = load_config()
    data_cfg = config["data"]

    if input_path and not input_path.startswith("s3://"):
        df = pd.read_parquet(input_path)
    else:
        df = load_inference_data(data_cfg["database"], data_cfg["table"], year=year)

    df_features = engineer_features(df, config)

    feat_cfg = config["features"]
    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])
    feature_cols = [c for c in feature_cols if c in df_features.columns]

    if use_endpoint:
        results = run_batch_via_endpoint(df_features, feature_cols, endpoint_name)
    else:
        # Local fallback — requires a pre-loaded model object (used in tests)
        raise RuntimeError("Local batch scoring requires a model object. Pass use_endpoint=True.")

    predictions_df = pd.DataFrame(results)
    predictions_df["run_date"] = datetime.utcnow().strftime("%Y-%m-%d")
    predictions_df["model_endpoint"] = endpoint_name

    is_s3 = output_path.startswith("s3://")
    if is_s3:
        wr.s3.to_parquet(
            df=predictions_df,
            path=output_path,
            dataset=True,
            partition_cols=["run_date"],
        )
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(
            f"{output_path}/predictions_{datetime.utcnow().strftime('%Y%m%d')}.parquet",
            index=False,
        )

    logger.info(
        "Batch inference complete: %d counties scored → %s",
        len(predictions_df),
        output_path,
    )
    return predictions_df


if __name__ == "__main__":
    df = run_batch_inference()
    print(df.head())
    print(f"Risk distribution:\n{df['risk_bucket'].value_counts()}")
