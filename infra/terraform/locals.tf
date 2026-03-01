locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  # True only after model artifact is uploaded and model_data_uri is set.
  # Controls conditional creation of SageMaker model, endpoint, and monitoring resources.
  deploy_endpoint = var.model_data_uri != ""
}
