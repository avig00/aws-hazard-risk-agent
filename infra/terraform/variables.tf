variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "sagemaker_model_name" {
  description = "Name of the approved SageMaker model (set after pipeline run)"
  type        = string
  default     = "hazard-risk-xgboost-v1"
}

variable "model_data_uri" {
  description = "S3 URI of the trained model artifact (model.tar.gz) — set after training"
  type        = string
  default     = ""
}

variable "project" {
  description = "Project name tag applied to all resources"
  type        = string
  default     = "hazard-risk-agent"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"
}
