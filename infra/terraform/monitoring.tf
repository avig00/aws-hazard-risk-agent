###############################################################################
# Monitoring — CloudWatch dashboards, alarms, SageMaker Model Monitor,
# EventBridge retraining trigger, SNS alert topic
###############################################################################

# ── SNS topic for alerts ──────────────────────────────────────────────────────
resource "aws_sns_topic" "model_alerts" {
  name = "hazard-model-alerts"
  tags = local.common_tags
}

# ── CloudWatch alarms ─────────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "sagemaker_invocation_errors" {
  alarm_name          = "hazard-sagemaker-invocation-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "Invocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "SageMaker endpoint returning 4XX errors"
  alarm_actions       = [aws_sns_topic.model_alerts.arn]

  dimensions = {
    EndpointName = "hazard-risk-model"
    VariantName  = "primary"
  }
}

# ── CloudWatch dashboard ──────────────────────────────────────────────────────
resource "aws_cloudwatch_dashboard" "hazard_risk" {
  dashboard_name = "HazardRiskAgent"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 24
        height = 6
        properties = {
          title   = "SageMaker Endpoint — Invocations & Model Latency"
          region  = var.region
          metrics = [
            ["AWS/SageMaker", "Invocations",         "EndpointName", "hazard-risk-model", "VariantName", "primary"],
            ["AWS/SageMaker", "ModelLatency",         "EndpointName", "hazard-risk-model", "VariantName", "primary"],
            ["AWS/SageMaker", "Invocation4XXErrors",  "EndpointName", "hazard-risk-model", "VariantName", "primary"],
          ]
          period = 60
          stat   = "Average"
          view   = "timeSeries"
        }
      },
    ]
  })
}

# ── SageMaker Model Monitor ───────────────────────────────────────────────────
# NOTE: SageMaker Serverless Inference endpoints do not support DataCapture,
# which is required for the built-in Model Monitor data quality jobs.
# Drift detection is handled instead via ml/monitoring/drift_detector.py
# (scheduled by EventBridge → Lambda) which computes feature statistics
# from Athena predictions and stores them in S3.

# ── EventBridge retraining trigger ───────────────────────────────────────────
resource "aws_cloudwatch_event_rule" "monthly_retrain" {
  name                = "hazard-monthly-retrain"
  description         = "Trigger model retraining on the 1st of each month"
  schedule_expression = "cron(0 7 1 * ? *)"
  tags                = local.common_tags
}

resource "aws_cloudwatch_event_target" "retrain_lambda" {
  rule      = aws_cloudwatch_event_rule.monthly_retrain.name
  target_id = "hazard-retrain-lambda"
  arn       = aws_lambda_function.retrain_trigger.arn

  input = jsonencode({ reason = "scheduled" })
}

data "archive_file" "retrain_trigger_zip" {
  type        = "zip"
  source_file = "${path.module}/../../ml/monitoring/retrain_trigger.py"
  output_path = "${path.module}/../../lambda_retrain.zip"
}

resource "aws_lambda_function" "retrain_trigger" {
  function_name = "hazard-retrain-trigger"
  role          = aws_iam_role.lambda_retrain.arn
  handler       = "retrain_trigger.lambda_handler"
  runtime       = "python3.11"
  timeout       = 60

  filename         = data.archive_file.retrain_trigger_zip.output_path
  source_code_hash = data.archive_file.retrain_trigger_zip.output_base64sha256

  environment {
    variables = {
      PIPELINE_NAME    = "hazard-risk-pipeline"
      SAGEMAKER_REGION = var.region
    }
  }

  tags = local.common_tags
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retrain_trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.monthly_retrain.arn
}
