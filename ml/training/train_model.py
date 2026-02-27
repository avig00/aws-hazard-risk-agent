"""
XGBoost model training with MLflow experiment tracking.

Loads train/test Parquet, applies feature engineering, trains XGBoost
with cross-validation, logs all params/metrics/artifacts to MLflow.
"""
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.model_selection import KFold, cross_val_score

from ml.data_prep.feature_engineering import engineer_features
from ml.training.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data(train_path: str, test_path: str) -> tuple:
    """Load pre-built Parquet splits (S3 or local)."""
    if train_path.startswith("s3://"):
        import awswrangler as wr
        train_df = wr.s3.read_parquet(path=train_path)
        test_df = wr.s3.read_parquet(path=test_path)
    else:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    logger.info("Loaded train=%d rows, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple:
    """Split DataFrame into feature matrix X and target vector y."""
    passthrough = {"county_fips", "year", target_col}
    feature_cols = [c for c in df.columns if c not in passthrough and df[c].dtype != object]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_model(config: dict = None, train_path: str = None, test_path: str = None):
    """
    Full training pipeline:
    1. Load Parquet splits
    2. Feature engineering (transforms, lags, encoding)
    3. XGBoost training with MLflow autolog
    4. 5-fold cross-validation
    5. Evaluation on held-out test set (RMSE, MAE, R², SHAP)
    6. Artifacts saved to MLflow run
    """
    if config is None:
        config = load_config()

    model_cfg = config["model"]
    data_cfg = config["data"]
    mlflow_cfg = config.get("mlflow", {})

    base_dir = data_cfg["output_dir"]
    _train_path = train_path or f"{base_dir}train.parquet"
    _test_path = test_path or f"{base_dir}test.parquet"

    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "hazard-risk-xgboost"))
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    train_df, test_df = load_data(_train_path, _test_path)
    target_col = data_cfg["target_column"]

    train_df = engineer_features(train_df, config)
    test_df = engineer_features(test_df, config)

    # Align columns after one-hot encoding (test may have different dummies)
    train_df, test_df = train_df.align(test_df, join="inner", axis=1)

    X_train, y_train, feature_cols = prepare_xy(train_df, target_col)
    X_test, y_test, _ = prepare_xy(test_df, target_col)

    xgb_params = {
        k: v for k, v in model_cfg.items()
        if k not in {"type", "random_state", "early_stopping_rounds", "eval_metric"}
    }
    xgb_params["random_state"] = model_cfg.get("random_state", 42)

    with mlflow.start_run() as run:
        mlflow.xgboost.autolog(log_models=True, log_input_examples=True)
        mlflow.log_params(xgb_params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model = xgb.XGBRegressor(**xgb_params)

        # 5-fold CV on training set
        logger.info("Running 5-fold cross-validation...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse = -cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        mlflow.log_metric("cv_rmse_mean", float(np.mean(cv_rmse)))
        mlflow.log_metric("cv_rmse_std", float(np.std(cv_rmse)))
        logger.info("CV RMSE: %.4f ± %.4f", np.mean(cv_rmse), np.std(cv_rmse))

        # Final fit on full training set
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=model_cfg.get("early_stopping_rounds", 50),
            verbose=False,
        )

        metrics = evaluate_model(model, X_test, y_test, feature_cols)
        logger.info(
            "Test — RMSE: %.4f | MAE: %.4f | R²: %.4f",
            metrics["rmse"], metrics["mae"], metrics["r2"],
        )

        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

    return model, feature_cols, run_id


if __name__ == "__main__":
    model, features, run_id = train_model()
    print(f"Training complete. Run ID: {run_id}")
    print(f"Features used ({len(features)}): {features[:10]} ...")
