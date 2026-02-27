"""
Model evaluation: RMSE, MAE, R², SHAP feature importance — all logged to MLflow.
"""
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, feature_cols: list = None) -> dict:
    """
    Compute regression metrics and log everything to the active MLflow run.
    Also computes SHAP feature importance if shap is installed.

    Returns:
        dict with rmse, mae, r2, and top_features
    """
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    logger.info("RMSE=%.4f | MAE=%.4f | R²=%.4f", rmse, mae, r2)

    # XGBoost native feature importance
    top_features = []
    if feature_cols and hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        top_features = importance_df.head(20)["feature"].tolist()

        # Log as MLflow artifact
        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="evaluation")
        logger.info("Top 5 features: %s", top_features[:5])

    # SHAP values (optional — skipped gracefully if shap not installed)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500])  # cap for speed

        shap_importance = pd.DataFrame({
            "feature": feature_cols if feature_cols else X_test.columns.tolist(),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        shap_path = "/tmp/shap_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path, artifact_path="evaluation")
        logger.info("SHAP importance logged for %d features", len(shap_importance))

    except ImportError:
        logger.info("shap not installed — skipping SHAP values")
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "top_features": top_features,
    }
