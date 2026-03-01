###############################################################################
# IAM — Roles and policies for SageMaker, ECS, and Bedrock access
###############################################################################

data "aws_caller_identity" "current" {}

# ── SageMaker execution role ──────────────────────────────────────────────────
resource "aws_iam_role" "sagemaker_execution" {
  name = "hazard-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "hazard-sagemaker-s3"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
      Resource = ["arn:aws:s3:::hazard", "arn:aws:s3:::hazard/*"]
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_athena" {
  name = "hazard-sagemaker-athena"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "athena:StartQueryExecution", "athena:GetQueryExecution",
        "athena:GetQueryResults", "athena:StopQueryExecution",
        "glue:GetTable", "glue:GetPartitions", "glue:GetDatabase",
      ]
      Resource = "*"
    }]
  })
}

# ── ECS task execution role ───────────────────────────────────────────────────
resource "aws_iam_role" "ecs_task_execution" {
  name = "hazard-ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── ECS task role (runtime permissions) ──────────────────────────────────────
resource "aws_iam_role" "ecs_task" {
  name = "hazard-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_bedrock" {
  name = "hazard-ecs-bedrock"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_sagemaker" {
  name = "hazard-ecs-sagemaker"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sagemaker:InvokeEndpoint"]
      Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:endpoint/hazard-risk-model"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_athena" {
  name = "hazard-ecs-athena"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution", "athena:GetQueryExecution",
          "athena:GetQueryResults", "glue:GetTable",
          "glue:GetPartitions", "glue:GetDatabase",
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = ["arn:aws:s3:::hazard", "arn:aws:s3:::hazard/*"]
      },
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_secrets" {
  name = "hazard-ecs-secrets"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:hazard/*"
    }]
  })
}

# ── Lambda retraining role ────────────────────────────────────────────────────
resource "aws_iam_role" "lambda_retrain" {
  name = "hazard-lambda-retrain-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_retrain.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_retrain_sagemaker" {
  name = "hazard-lambda-sagemaker-pipeline"
  role = aws_iam_role.lambda_retrain.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution"]
      Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.current.account_id}:pipeline/hazard-risk-pipeline"
    }]
  })
}
