"""
Automated retraining trigger.

Invoked by EventBridge rule → Lambda when:
  - Drift check detects feature/prediction drift exceeding threshold
  - Scheduled weekly retraining (monthly baseline refresh)

Starts a new SageMaker Pipeline execution to re-train and re-register the model.
"""
import json
import logging

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_NAME = "hazard-risk-pipeline"
REGION = "us-east-1"


def trigger_retraining(
    reason: str = "scheduled",
    pipeline_name: str = PIPELINE_NAME,
    region: str = REGION,
) -> str:
    """
    Start a new SageMaker Pipeline execution.

    Args:
        reason: Why retraining was triggered ('drift_detected' or 'scheduled').
        pipeline_name: SageMaker Pipeline name.
        region: AWS region.

    Returns:
        Pipeline execution ARN.
    """
    sm = boto3.client("sagemaker", region_name=region)

    response = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=f"retrain-{reason}",
        PipelineExecutionDescription=f"Automated retrain triggered by: {reason}",
        PipelineParameters=[
            # Use latest Gold layer data
            {"Name": "InputDataUri", "Value": "s3://hazard/gold/risk_feature_mart/"},
        ],
    )

    arn = response["PipelineExecutionArn"]
    logger.info("Pipeline execution started: %s (reason=%s)", arn, reason)
    return arn


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    Expected event payload:
    {
        "reason": "drift_detected" | "scheduled",
        "drifted_features": [...],  # Optional
    }
    """
    reason = event.get("reason", "scheduled")
    drifted_features = event.get("drifted_features", [])

    logger.info(
        "Retrain trigger invoked: reason=%s | drifted_features=%s",
        reason,
        drifted_features,
    )

    arn = trigger_retraining(reason=reason)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Retraining pipeline started",
            "execution_arn": arn,
            "reason": reason,
        }),
    }


if __name__ == "__main__":
    # Local test
    arn = trigger_retraining(reason="manual_test")
    print(f"Started: {arn}")
