"""
SageMaker Processing script — Evaluation step for the classification pipeline.

Runs inside a SageMaker SKLearnProcessor container.
Reads:
  /opt/ml/processing/model/   — model.tar.gz (XGBoost binary)
  /opt/ml/processing/test/    — test.parquet

Writes:
  /opt/ml/processing/evaluation/evaluation.json
      {"metrics": {"balanced_accuracy": {"value": 0.687}, "accuracy": {"value": 0.687}, ...}}
"""
import json
import logging
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def load_model(model_dir: str) -> xgb.XGBClassifier:
    """Extract model.tar.gz and load the XGBoost binary."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    extract_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
        logger.info("Extracted: %s", tar.getnames())

    model_file = os.path.join(extract_dir, "xgboost-model")
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    logger.info("Model loaded from %s", model_file)
    return model


def load_test_data(test_dir: str) -> pd.DataFrame:
    """Load test.parquet from the processing input directory."""
    parquet_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {test_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    logger.info("Loaded test data: %d rows, %d cols", len(df), len(df.columns))
    return df


def prepare_xy(df: pd.DataFrame, target_col: str = "risk_bucket"):
    """Extract feature matrix X and integer-encoded target y."""
    passthrough = {
        "county_fips", "year", target_col,
        "risk_bucket", "damage_bucket", "fema_total_damage",
    }
    feature_cols = [
        c for c in df.columns
        if c not in passthrough and df[c].dtype != object
    ]
    X = df[feature_cols]
    y = df[target_col].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Unmapped labels: {df[target_col].unique()}")
    return X, y.astype(int)


def evaluate(model_dir: str, test_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_dir)
    test_df = load_test_data(test_dir)

    X_test, y_test = prepare_xy(test_df)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    logger.info(
        "Accuracy=%.4f | Balanced Acc=%.4f | F1 weighted=%.4f | F1 macro=%.4f",
        accuracy, balanced_acc, f1_weighted, f1_macro,
    )
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=LABEL_NAMES),
    )

    # SageMaker PropertyFile format — read by ConditionStep via JsonGet
    evaluation_report = {
        "metrics": {
            "balanced_accuracy": {"value": balanced_acc},
            "accuracy": {"value": accuracy},
            "f1_weighted": {"value": f1_weighted},
            "f1_macro": {"value": f1_macro},
        }
    }

    report_path = os.path.join(output_dir, "evaluation.json")
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("Evaluation report written to %s", report_path)


if __name__ == "__main__":
    evaluate(MODEL_DIR, TEST_DIR, OUTPUT_DIR)
