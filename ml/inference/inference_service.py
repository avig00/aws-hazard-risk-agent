"""
SageMaker real-time endpoint invocation for county-level risk classification.

The endpoint accepts raw county feature values, applies the same feature
engineering pipeline used at training time, and returns a risk tier prediction
(LOW / MEDIUM / HIGH).

Feature alignment:
  - Model expects exactly 89 features (stored in S3 as model_feature_cols.json).
  - Feature engineering (log transforms + one-hot state encoding) is applied
    locally before calling the endpoint.
  - Missing state dummies are filled with 0.

Content type: text/csv (SageMaker XGBoost built-in algorithm)
Response: single integer (0=LOW, 1=MEDIUM, 2=HIGH)
"""
import io
import json
import logging

import boto3
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "hazard-risk-model"
BUCKET = "aws-hazard-risk-vigamogh-dev"
FEATURE_COLS_KEY = "hazard/ml/features/model_feature_cols.json"
LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]

# Cached at module level to avoid repeated S3 calls
_feature_cols: list = None


def _get_feature_cols() -> list:
    global _feature_cols
    if _feature_cols is None:
        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=BUCKET, Key=FEATURE_COLS_KEY)
        _feature_cols = json.loads(obj["Body"].read())["features"]
        logger.info("Loaded %d model feature columns from S3", len(_feature_cols))
    return _feature_cols


def predict_risk(
    features: dict,
    config: dict = None,
    endpoint_name: str = ENDPOINT_NAME,
    region: str = "us-east-1",
) -> dict:
    """
    Classify a single county's risk tier via the SageMaker endpoint.

    Args:
        features: Raw feature dict (county_fips, year, NOAA counts, demo stats, etc.)
                  Must include at least the numeric features and 'state'.
        config:   model_config.yml dict (loaded if not provided).
        endpoint_name: SageMaker endpoint name.
        region:   AWS region.

    Returns:
        dict with:
          - risk_tier: str  (LOW / MEDIUM / HIGH)
          - class_id:  int  (0 / 1 / 2)
          - endpoint_name: str
    """
    if config is None:
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Build a single-row DataFrame
    row_df = pd.DataFrame([features])

    # Apply feature engineering without dropna (inference doesn't need lag filtering)
    from ml.data_prep.feature_engineering import (
        encode_categoricals, impute_numeric, log_transform, drop_non_features,
    )
    feat_cfg = config["features"]
    engineered = row_df.copy()
    engineered = impute_numeric(engineered, feat_cfg.get("numeric", []))
    engineered = log_transform(engineered, feat_cfg.get("log_transform", []))
    engineered = encode_categoricals(engineered, feat_cfg.get("categorical", []))
    engineered = drop_non_features(engineered)

    # Align to exact model feature columns (fill missing/NaN with 0)
    feature_cols = _get_feature_cols()
    aligned = engineered.reindex(columns=feature_cols, fill_value=0).fillna(0)

    # Serialize as CSV
    csv_body = ",".join(str(v) for v in aligned.iloc[0].values)

    # Call endpoint
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=csv_body,
    )
    raw = response["Body"].read().decode("utf-8").strip()
    class_id = int(float(raw))
    risk_tier = LABEL_NAMES[class_id]

    logger.info(
        "Endpoint=%s | county_fips=%s | predicted=%s (%d)",
        endpoint_name,
        features.get("county_fips", "unknown"),
        risk_tier,
        class_id,
    )

    return {
        "risk_tier": risk_tier,
        "class_id": class_id,
        "endpoint_name": endpoint_name,
    }


def predict_batch(
    df: pd.DataFrame,
    config: dict = None,
    endpoint_name: str = ENDPOINT_NAME,
    region: str = "us-east-1",
) -> pd.DataFrame:
    """
    Classify multiple counties via repeated endpoint calls.
    Returns original DataFrame with 'predicted_risk_tier' and 'predicted_class_id' columns.
    """
    if config is None:
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

    from ml.data_prep.feature_engineering import (
        encode_categoricals, impute_numeric, log_transform, drop_non_features,
    )
    feat_cfg = config["features"]
    engineered = df.copy()
    engineered = impute_numeric(engineered, feat_cfg.get("numeric", []))
    engineered = log_transform(engineered, feat_cfg.get("log_transform", []))
    engineered = encode_categoricals(engineered, feat_cfg.get("categorical", []))
    engineered = drop_non_features(engineered)

    feature_cols = _get_feature_cols()
    aligned = engineered.reindex(columns=feature_cols, fill_value=0).fillna(0)

    runtime = boto3.client("sagemaker-runtime", region_name=region)
    class_ids = []
    for _, row in aligned.iterrows():
        csv_body = ",".join(str(v) for v in row.values)
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=csv_body,
        )
        class_id = int(float(response["Body"].read().decode("utf-8").strip()))
        class_ids.append(class_id)

    result = df.copy()
    result["predicted_class_id"] = class_ids
    result["predicted_risk_tier"] = [LABEL_NAMES[c] for c in class_ids]
    return result


if __name__ == "__main__":
    # Quick smoke test — uses first row from test.parquet
    import boto3 as _boto3

    s3 = _boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(
        Bucket=BUCKET,
        Key="hazard/ml/features/test.parquet",
    )
    test_df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    sample = test_df.iloc[0].to_dict()
    actual = sample.get("risk_bucket", "unknown")

    result = predict_risk(sample)
    print(f"Predicted: {result['risk_tier']} | Actual: {actual} | Match: {result['risk_tier'] == actual}")
