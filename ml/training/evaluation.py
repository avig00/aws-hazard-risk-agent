"""
Model evaluation: accuracy, balanced accuracy, F1, confusion matrix,
SHAP feature importance -- all logged to MLflow.
"""
import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]


def evaluate_model(model, X_test, y_test, feature_cols: list = None) -> dict:
    """
    Compute classification metrics and log everything to the active MLflow run.
    Also computes SHAP feature importance if shap is installed.

    Returns:
        dict with accuracy, balanced_accuracy, f1_weighted, f1_macro, top_features
    """
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    mlflow.log_metrics({
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    })
    logger.info(
        "Accuracy=%.4f | Balanced Acc=%.4f | F1 weighted=%.4f | F1 macro=%.4f",
        accuracy, balanced_acc, f1_weighted, f1_macro,
    )

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        output_dict=False,
    )
    logger.info("Classification report:\n%s", report)
    report_path = "/tmp/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="evaluation")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_path = "/tmp/confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="evaluation")
    logger.info("Confusion matrix:\n%s", cm_df.to_string())

    # XGBoost native feature importance
    top_features = []
    if feature_cols and hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        top_features = importance_df.head(20)["feature"].tolist()

        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="evaluation")
        logger.info("Top 5 features: %s", top_features[:5])

    # SHAP values (optional -- skipped gracefully if shap not installed)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500])  # cap for speed

        # Multi-class SHAP can return either:
        #   - 3D ndarray (n_samples, n_features, n_classes)  — newer shap
        #   - list of 2D arrays [(n_samples, n_features), ...]  — older shap
        # In both cases reduce to a 1D (n_features,) mean-abs importance.
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        elif isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_importance = pd.DataFrame({
            "feature": feature_cols if feature_cols else X_test.columns.tolist(),
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_path = "/tmp/shap_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path, artifact_path="evaluation")
        logger.info("SHAP importance logged for %d features", len(shap_importance))
        logger.info("Top 5 SHAP: %s", shap_importance.head(5)["feature"].tolist())

    except ImportError:
        logger.info("shap not installed -- skipping SHAP values")
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "top_features": top_features,
    }
