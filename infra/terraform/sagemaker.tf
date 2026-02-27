###############################################################################
# SageMaker — model package group + endpoint configuration
#
# The actual pipeline runs are triggered via Python SDK (ml/pipeline/sagemaker_pipeline.py).
# Terraform manages the static infrastructure: model group, endpoint config, endpoint.
###############################################################################

resource "aws_sagemaker_model_package_group" "hazard_risk" {
  model_package_group_name        = "hazard-risk-model-group"
  model_package_group_description = "XGBoost county hazard risk regression models"

  tags = local.common_tags
}

# Endpoint configuration — references the latest approved model package
# (model_data_url is populated after a pipeline run approves the model)
resource "aws_sagemaker_endpoint_configuration" "hazard_risk" {
  name = "hazard-risk-endpoint-config"

  production_variants {
    variant_name           = "primary"
    model_name             = var.sagemaker_model_name
    initial_instance_count = 1
    instance_type          = "ml.m5.large"
    initial_variant_weight = 1.0
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 20
    destination_s3_uri          = "s3://hazard/ml/data-capture/"

    capture_options {
      capture_mode = "Input"
    }
    capture_options {
      capture_mode = "Output"
    }
  }

  tags = local.common_tags
}

resource "aws_sagemaker_endpoint" "hazard_risk" {
  name                 = "hazard-risk-model"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.hazard_risk.name

  tags = local.common_tags

  lifecycle {
    ignore_changes = [endpoint_config_name]
  }
}

output "sagemaker_endpoint_name" {
  value = aws_sagemaker_endpoint.hazard_risk.name
}
