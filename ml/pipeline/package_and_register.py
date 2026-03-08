"""
Train → Save → Package → Register → Deploy

Runs the full post-training workflow in one shot:
  1. Train XGBoost on S3 Parquet splits (uses current config)
  2. Save the XGBoost booster binary to /tmp/xgboost-model
  3. Create model.tar.gz and upload to S3
  4. Write model_feature_cols.json to S3
  5. Create a new SageMaker Model pointing to the artifact
  6. Register the model in the Model Package Group
  7. Approve the new model version
  8. Update the serverless endpoint to use the new model

Usage:
    PYTHONPATH=. python ml/pipeline/package_and_register.py
"""
import io
import json
import logging
import os
import tarfile
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGION = "us-east-1"
BUCKET = "aws-hazard-risk-vigamogh-dev"
ARTIFACT_S3_KEY = "hazard/ml/artifacts/model.tar.gz"
FEATURE_COLS_S3_KEY = "hazard/ml/features/model_feature_cols.json"
MODEL_GROUP = "hazard-risk-model-group"
ENDPOINT_NAME = "hazard-risk-model"
ENDPOINT_CONFIG_BASE = "hazard-risk-model-cfg"

ROLE_ARN = os.environ.get(
    "SAGEMAKER_ROLE_ARN",
    "arn:aws:iam::945919380353:role/hazard-sagemaker-execution-role",
)


# ── Step 1: Train ────────────────────────────────────────────────────────────

def train() -> tuple:
    """Return (model, feature_cols, run_id)."""
    from ml.training.train_model import train_model
    logger.info("Training model...")
    model, feature_cols, run_id = train_model()
    logger.info("Training complete. run_id=%s, features=%d", run_id, len(feature_cols))
    return model, feature_cols, run_id


# ── Step 2+3: Save & Package ─────────────────────────────────────────────────

def package_model(model, feature_cols: list) -> str:
    """Save XGBoost binary, tar it, upload to S3.  Returns s3 URI."""
    local_model_path = "/tmp/xgboost-model"
    local_tar_path = "/tmp/model.tar.gz"

    logger.info("Saving XGBoost booster to %s", local_model_path)
    model.save_model(local_model_path)

    logger.info("Creating %s", local_tar_path)
    with tarfile.open(local_tar_path, "w:gz") as tar:
        tar.add(local_model_path, arcname="xgboost-model")

    s3 = boto3.client("s3", region_name=REGION)

    logger.info("Uploading model.tar.gz to s3://%s/%s", BUCKET, ARTIFACT_S3_KEY)
    s3.upload_file(local_tar_path, BUCKET, ARTIFACT_S3_KEY)

    # ── Step 4: Feature columns ──────────────────────────────────────────────
    feature_json = json.dumps({"features": feature_cols}, indent=2).encode()
    logger.info("Writing model_feature_cols.json (%d features)", len(feature_cols))
    s3.put_object(
        Bucket=BUCKET,
        Key=FEATURE_COLS_S3_KEY,
        Body=feature_json,
        ContentType="application/json",
    )

    return f"s3://{BUCKET}/{ARTIFACT_S3_KEY}"


# ── Step 5+6+7: Register ─────────────────────────────────────────────────────

def register_model(model_uri: str) -> str:
    """Create a SageMaker Model + register in Model Group.  Returns model package ARN."""
    sm = boto3.client("sagemaker", region_name=REGION)

    # Create a SageMaker Model resource
    model_name = f"hazard-risk-xgb-{_short_ts()}"
    logger.info("Creating SageMaker model: %s", model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": _xgb_image(),
            "ModelDataUrl": model_uri,
            "Environment": {},
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    # Register into model package group
    logger.info("Registering model in group: %s", MODEL_GROUP)
    pkg_resp = sm.create_model_package(
        ModelPackageGroupName=MODEL_GROUP,
        ModelPackageDescription="XGBoost county risk classifier — no-NRI clean features",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": _xgb_image(),
                    "ModelDataUrl": model_uri,
                }
            ],
            "SupportedContentTypes": ["application/json", "text/csv"],
            "SupportedResponseMIMETypes": ["application/json", "text/csv"],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        },
        ModelApprovalStatus="Approved",
    )
    pkg_arn = pkg_resp["ModelPackageArn"]
    logger.info("Model package registered + approved: %s", pkg_arn)
    return model_name, pkg_arn


# ── Step 8: Update endpoint ──────────────────────────────────────────────────

def update_endpoint(model_name: str, model_uri: str):
    """Create a new endpoint config and update the live endpoint."""
    sm = boto3.client("sagemaker", region_name=REGION)

    cfg_name = f"{ENDPOINT_CONFIG_BASE}-{_short_ts()}"
    logger.info("Creating endpoint config: %s", cfg_name)
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": 2048,
                    "MaxConcurrency": 5,
                },
            }
        ],
    )

    logger.info("Updating endpoint %s → config %s", ENDPOINT_NAME, cfg_name)
    sm.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=cfg_name,
    )

    # Wait for InService
    logger.info("Waiting for endpoint to reach InService state...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 15, "MaxAttempts": 40},
    )
    logger.info("Endpoint %s is InService", ENDPOINT_NAME)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _short_ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _xgb_image() -> str:
    """Return the AWS-managed XGBoost 1.7 container URI for us-east-1."""
    return (
        "683313688378.dkr.ecr.us-east-1.amazonaws.com"
        "/sagemaker-xgboost:1.7-1"
    )


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, feature_cols, run_id = train()
    model_uri = package_model(model, feature_cols)
    model_name, pkg_arn = register_model(model_uri)
    update_endpoint(model_name, model_uri)

    print("\n✓ Pipeline complete")
    print(f"  MLflow run   : {run_id}")
    print(f"  Model URI    : {model_uri}")
    print(f"  Features     : {len(feature_cols)}")
    print(f"  Package ARN  : {pkg_arn}")
    print(f"  Endpoint     : {ENDPOINT_NAME} (InService)")
