###############################################################################
# SageMaker — model package group + endpoint configuration
#
# The actual pipeline runs are triggered via Python SDK (ml/pipeline/sagemaker_pipeline.py).
# Terraform manages the static infrastructure: model group, endpoint config, endpoint.
#
# Phase 1 (no model_data_uri): only model_package_group is created.
# Phase 2 (after training): set model_data_uri=s3://... to deploy model + endpoint.
###############################################################################

# SageMaker Model object — container image + artifact pointer required by endpoint config.
# Only created when model_data_uri is set (Phase 2 onwards).
resource "aws_sagemaker_model" "hazard_risk" {
  count              = local.deploy_endpoint ? 1 : 0
  name               = var.sagemaker_model_name
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    # XGBoost 1.7 built-in container (us-east-1)
    image          = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"
    model_data_url = var.model_data_uri
  }

  tags = local.common_tags
}

resource "aws_sagemaker_model_package_group" "hazard_risk" {
  model_package_group_name        = "hazard-risk-model-group"
  model_package_group_description = "XGBoost county hazard risk regression models"

  tags = local.common_tags
}

# Endpoint configuration — only created when model artifact is available.
resource "aws_sagemaker_endpoint_configuration" "hazard_risk" {
  count = local.deploy_endpoint ? 1 : 0
  name  = "hazard-risk-endpoint-config"

  # Serverless Inference: scales to zero when idle — no charge between requests.
  # Cold start: ~60–120 seconds. Suitable for demo/portfolio workloads.
  production_variants {
    variant_name = "primary"
    model_name   = aws_sagemaker_model.hazard_risk[0].name

    serverless_config {
      max_concurrency   = 5
      memory_size_in_mb = 2048
    }
  }

  tags = local.common_tags
}

resource "aws_sagemaker_endpoint" "hazard_risk" {
  count                = local.deploy_endpoint ? 1 : 0
  name                 = "hazard-risk-model"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.hazard_risk[0].name

  tags = local.common_tags

  lifecycle {
    ignore_changes = [endpoint_config_name]
  }
}

output "sagemaker_endpoint_name" {
  value = local.deploy_endpoint ? aws_sagemaker_endpoint.hazard_risk[0].name : "not-yet-deployed"
}
