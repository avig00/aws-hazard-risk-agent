"""
SageMaker real-time endpoint invocation for county-level risk prediction.

The endpoint accepts a dict of feature values per county and returns
a risk score (NRI_ExpectedLoss predicted value) plus a risk bucket label.
"""
import json
import logging

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "hazard-risk-model"


def predict_from_endpoint(
    payload: dict,
    endpoint_name: str = ENDPOINT_NAME,
    region: str = "us-east-1",
) -> dict:
    """
    Call the deployed SageMaker endpoint and return a structured prediction.

    Args:
        payload: dict mapping feature names to values for one or more counties.
                 Expects key "instances" → list of feature dicts.
                 OR a flat feature dict for a single county.
        endpoint_name: SageMaker endpoint name.
        region: AWS region.

    Returns:
        dict with keys:
          - predictions: list of floats (expected loss scores)
          - risk_buckets: list of str (LOW / MEDIUM / HIGH)
          - endpoint_name: str
    """
    # Normalize payload to {"instances": [...]}
    if "instances" not in payload:
        payload = {"instances": [payload]}

    client = boto3.client("sagemaker-runtime", region_name=region)

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )

    raw = json.loads(response["Body"].read())

    # Normalize response — SageMaker XGBoost built-in returns {"predictions": [...]}
    predictions = raw.get("predictions", raw) if isinstance(raw, dict) else raw
    if isinstance(predictions, (int, float)):
        predictions = [predictions]

    risk_buckets = [_score_to_bucket(float(p)) for p in predictions]

    logger.info(
        "Endpoint=%s | %d predictions | scores=%s",
        endpoint_name,
        len(predictions),
        [round(p, 2) for p in predictions[:5]],
    )

    return {
        "predictions": [float(p) for p in predictions],
        "risk_buckets": risk_buckets,
        "endpoint_name": endpoint_name,
    }


def _score_to_bucket(score: float) -> str:
    """Convert continuous expected-loss score to LOW / MEDIUM / HIGH label.

    Thresholds are approximate NRI percentile boundaries — override these
    once baseline statistics are computed from the training data.
    """
    if score < 5_000:
        return "LOW"
    elif score < 25_000:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_batch_local(df, model, feature_cols: list) -> list:
    """
    Local (non-endpoint) prediction for batch inference or testing.
    Used when SageMaker endpoint is not deployed.
    """
    import pandas as pd
    X = df[feature_cols] if hasattr(df, "__getitem__") else df
    scores = model.predict(X).tolist()
    return [
        {"score": round(float(s), 4), "risk_bucket": _score_to_bucket(float(s))}
        for s in scores
    ]


if __name__ == "__main__":
    # Local smoke test (will fail without a live endpoint — expected)
    sample_payload = {
        "instances": [{
            "total_events": 12,
            "fema_claim_count": 30,
            "NRI_Exposure": 0.8,
            "NRI_SocialVulnerability": 0.65,
            "NRI_CommunityResilience": 0.5,
            "census_median_income": 52000,
        }]
    }
    try:
        result = predict_from_endpoint(sample_payload)
        print("Prediction:", result)
    except Exception as exc:
        print(f"Endpoint not available (expected in dev): {exc}")
