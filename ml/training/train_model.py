import mlflow
import xgboost as xgb
import pandas as pd

def train_model(config: dict):
    """
    Train XGBoost model, track with MLflow, save artifacts.
    """
    print("Training model...")
    with mlflow.start_run():
        mlflow.log_params(config)

        # Placeholder
        model = xgb.XGBRegressor(**config)
        mlflow.xgboost.log_model(model, "model")

    return model

if __name__ == "__main__":
    config = {
        "max_depth": 5,
        "learning_rate": 0.1
    }
    train_model(config)
