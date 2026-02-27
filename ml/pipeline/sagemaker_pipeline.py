"""
SageMaker Pipeline: 4-step ML workflow for the Hazard Risk model.

Steps:
  1. ProcessingStep  — pull Gold data from Athena, run feature engineering
  2. TrainingStep    — XGBoost training, log params/metrics to MLflow
  3. EvaluationStep  — compute RMSE/R²/MAE, write evaluation.json
  4. ConditionStep   — register model only if RMSE <= threshold

Usage:
    python ml/pipeline/sagemaker_pipeline.py --action create   # upsert pipeline definition
    python ml/pipeline/sagemaker_pipeline.py --action execute  # start a run
    python ml/pipeline/sagemaker_pipeline.py --action status   # check latest execution
"""
import argparse
import logging
from pathlib import Path

import boto3
import sagemaker
import yaml
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost import XGBoost, XGBoostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"
PIPELINE_NAME = "hazard-risk-pipeline"
REGION = "us-east-1"
BASE_JOB_PREFIX = "hazard-risk"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_pipeline(role_arn: str, bucket: str) -> Pipeline:
    """Build and return the SageMaker Pipeline object."""
    config = load_config()
    model_cfg = config["model"]

    session = PipelineSession(default_bucket=bucket)

    # Pipeline parameters (overridable at execution time)
    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{bucket}/gold/risk_feature_mart/",
    )
    rmse_threshold = ParameterFloat(name="RmseThreshold", default_value=15000.0)

    # ── Step 1: Data Prep ────────────────────────────────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{BASE_JOB_PREFIX}-data-prep",
        role=role_arn,
        sagemaker_session=session,
    )

    step_data_prep = ProcessingStep(
        name="DataPrep",
        processor=processor,
        inputs=[ProcessingInput(source=input_data_uri, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/ml/features/train/",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/ml/features/test/",
            ),
        ],
        code="ml/data_prep/build_training_data.py",
    )

    # ── Step 2: Training ─────────────────────────────────────────────────────
    xgb_estimator = XGBoost(
        entry_point="ml/training/train_model.py",
        framework_version="1.7-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role_arn,
        base_job_name=f"{BASE_JOB_PREFIX}-training",
        sagemaker_session=session,
        hyperparameters={
            "max_depth": model_cfg["max_depth"],
            "learning_rate": model_cfg["learning_rate"],
            "n_estimators": model_cfg["n_estimators"],
            "objective": model_cfg["objective"],
            "subsample": model_cfg["subsample"],
            "colsample_bytree": model_cfg["colsample_bytree"],
        },
    )

    step_training = TrainingStep(
        name="TrainXGBoost",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ── Step 3: Evaluation ───────────────────────────────────────────────────
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{BASE_JOB_PREFIX}-eval",
        role=role_arn,
        sagemaker_session=session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_training.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_data_prep.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/ml/evaluation/",
            )
        ],
        code="ml/training/evaluation.py",
        property_files=[evaluation_report],
    )

    # ── Step 4: Conditional Model Registration ───────────────────────────────
    xgb_model = XGBoostModel(
        model_data=step_training.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=session,
        framework_version="1.7-1",
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=xgb_model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=f"{BASE_JOB_PREFIX}-model-group",
            approval_status="PendingManualApproval",
        ),
    )

    step_condition = ConditionStep(
        name="CheckRmseThreshold",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=step_eval.name,
                    property_file=evaluation_report,
                    json_path="metrics.rmse.value",
                ),
                right=rmse_threshold,
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data_uri, rmse_threshold],
        steps=[step_data_prep, step_training, step_eval, step_condition],
        sagemaker_session=session,
    )


def create_or_update_pipeline(role_arn: str, bucket: str) -> str:
    pipeline = get_pipeline(role_arn, bucket)
    response = pipeline.upsert(role_arn=role_arn)
    arn = response["PipelineArn"]
    logger.info("Pipeline upserted: %s", arn)
    return arn


def execute_pipeline(role_arn: str, bucket: str) -> str:
    pipeline = get_pipeline(role_arn, bucket)
    execution = pipeline.start()
    logger.info("Execution started: %s", execution.arn)
    return execution.arn


def get_latest_status(pipeline_name: str = PIPELINE_NAME, region: str = REGION) -> str:
    sm = boto3.client("sagemaker", region_name=region)
    executions = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        SortOrder="Descending",
        MaxResults=1,
    )["PipelineExecutionSummaries"]

    if not executions:
        return "No executions found"

    status = executions[0]["PipelineExecutionStatus"]
    logger.info("Latest status: %s", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["create", "execute", "status"], default="create")
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--bucket", default="hazard")
    args = parser.parse_args()

    if args.action == "status":
        print(get_latest_status())
    else:
        role = args.role_arn or sagemaker.get_execution_role()
        if args.action == "create":
            create_or_update_pipeline(role, args.bucket)
        elif args.action == "execute":
            execute_pipeline(role, args.bucket)
