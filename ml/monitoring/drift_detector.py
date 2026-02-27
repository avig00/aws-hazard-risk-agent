"""
Data and prediction drift detection for the Hazard Risk model.

Computes KL divergence between baseline feature distributions (from training)
and recent inference distributions (from SageMaker data capture logs).

Run on a daily/weekly schedule via EventBridge → Lambda or a cron job.
"""
import json
import logging
from pathlib import Path

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"

BASELINE_PATH = "s3://hazard/ml/monitoring/baseline_stats.json"
CAPTURE_PATH = "s3://hazard/ml/data-capture/"
DRIFT_THRESHOLD_KL = 0.1       # Flag if KL divergence exceeds this
DRIFT_SNS_TOPIC = "arn:aws:sns:us-east-1:ACCOUNT_ID:hazard-model-alerts"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_baseline_stats(train_path: str, feature_cols: list) -> dict:
    """
    Compute per-feature statistics from training data and save as baseline.
    Call once after initial model training.
    """
    if train_path.startswith("s3://"):
        df = wr.s3.read_parquet(path=train_path)
    else:
        df = pd.read_parquet(train_path)

    stats = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "histogram": _compute_histogram(series),
        }

    # Save to S3
    s3 = boto3.client("s3")
    bucket, key = BASELINE_PATH.replace("s3://", "").split("/", 1)
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(stats))
    logger.info("Baseline stats saved for %d features", len(stats))
    return stats


def _compute_histogram(series: pd.Series, bins: int = 20) -> dict:
    """Compute a normalized histogram for KL divergence comparison."""
    counts, edges = np.histogram(series.dropna(), bins=bins, density=False)
    total = counts.sum()
    probs = (counts / total).tolist() if total > 0 else [0.0] * bins
    return {"counts": counts.tolist(), "edges": edges.tolist(), "probs": probs}


def load_baseline_stats() -> dict:
    """Load saved baseline statistics from S3."""
    s3 = boto3.client("s3")
    bucket, key = BASELINE_PATH.replace("s3://", "").split("/", 1)
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def load_recent_capture(days_back: int = 1) -> pd.DataFrame:
    """
    Read recent SageMaker data capture logs from S3.
    Returns a DataFrame of captured inference inputs.
    """
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    logger.info("Loading capture data since %s", cutoff.isoformat())

    try:
        df = wr.s3.read_json(
            path=CAPTURE_PATH,
            boto3_session=boto3.Session(),
        )
        df["capture_time"] = pd.to_datetime(df.get("eventMetadata", {}).get("eventId", ""))
        return df
    except Exception as exc:
        logger.warning("Failed to load capture data: %s", exc)
        return pd.DataFrame()


def kl_divergence(p: list, q: list, epsilon: float = 1e-10) -> float:
    """KL divergence D(P||Q) — measures how P differs from baseline Q."""
    p_arr = np.array(p, dtype=float) + epsilon
    q_arr = np.array(q, dtype=float) + epsilon
    p_norm = p_arr / p_arr.sum()
    q_norm = q_arr / q_arr.sum()
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def detect_drift(
    recent_df: pd.DataFrame,
    baseline_stats: dict,
    feature_cols: list,
    threshold: float = DRIFT_THRESHOLD_KL,
) -> dict:
    """
    Compare recent inference feature distributions against baseline.
    Returns drift report per feature.
    """
    drift_report = {}

    for col in feature_cols:
        if col not in recent_df.columns or col not in baseline_stats:
            continue

        series = recent_df[col].dropna()
        if len(series) < 10:
            continue

        baseline = baseline_stats[col]
        bins = len(baseline["histogram"]["edges"]) - 1
        counts, _ = np.histogram(series, bins=baseline["histogram"]["edges"])
        total = counts.sum()
        recent_probs = (counts / total).tolist() if total > 0 else [0.0] * bins

        kl = kl_divergence(recent_probs, baseline["histogram"]["probs"])
        is_drifted = kl > threshold

        drift_report[col] = {
            "kl_divergence": round(kl, 6),
            "is_drifted": is_drifted,
            "recent_mean": float(series.mean()),
            "baseline_mean": baseline["mean"],
            "mean_shift": round(float(series.mean()) - baseline["mean"], 4),
        }

        if is_drifted:
            logger.warning(
                "DRIFT DETECTED: %s | KL=%.4f (threshold=%.4f)", col, kl, threshold
            )

    drifted_features = [k for k, v in drift_report.items() if v["is_drifted"]]
    drift_report["_summary"] = {
        "total_features_checked": len(drift_report) - 1,
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
    }

    return drift_report


def publish_drift_alert(drift_report: dict, sns_topic: str = DRIFT_SNS_TOPIC) -> None:
    """Send an SNS notification when drift is detected."""
    summary = drift_report.get("_summary", {})
    if not summary.get("drift_detected"):
        return

    sns = boto3.client("sns")
    message = (
        f"Model drift detected in {len(summary['drifted_features'])} feature(s): "
        f"{summary['drifted_features']}\n\n"
        f"Full report: {json.dumps({k: v for k, v in drift_report.items() if k != '_summary'}, indent=2)}"
    )

    sns.publish(
        TopicArn=sns_topic,
        Subject="[Hazard Risk Agent] Model Drift Alert",
        Message=message,
    )
    logger.info("Drift alert published to SNS")


def run_drift_check(days_back: int = 1) -> dict:
    """Full drift detection pipeline: load → compare → alert."""
    config = load_config()
    feature_cols = config["features"]["numeric"]

    baseline_stats = load_baseline_stats()
    recent_df = load_recent_capture(days_back=days_back)

    if recent_df.empty:
        logger.warning("No recent capture data — skipping drift check")
        return {"_summary": {"drift_detected": False, "reason": "no_capture_data"}}

    drift_report = detect_drift(recent_df, baseline_stats, feature_cols)

    if drift_report["_summary"]["drift_detected"]:
        publish_drift_alert(drift_report)

    return drift_report


if __name__ == "__main__":
    report = run_drift_check()
    summary = report.get("_summary", {})
    print(f"Drift detected: {summary.get('drift_detected', False)}")
    print(f"Drifted features: {summary.get('drifted_features', [])}")
