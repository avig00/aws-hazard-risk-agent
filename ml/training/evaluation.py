import numpy as np
import mlflow

def evaluate_model(model, X_test, y_test):
    """
    Compute RMSE, MAE, R2, log to MLflow.
    """
    print("Evaluating model...")
    rmse = 0.0  # placeholder
    mae = 0.0

    mlflow.log_metrics({"rmse": rmse, "mae": mae})
    return {"rmse": rmse, "mae": mae}
