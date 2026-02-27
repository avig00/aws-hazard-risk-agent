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

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "hazard-api-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "5XXError"
  namespace           = "AWS/ApiGateway"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "More than 10 5XX errors per minute on the Hazard Risk API"
  alarm_actions       = [aws_sns_topic.model_alerts.arn]

  dimensions = {
    ApiId = aws_apigatewayv2_api.hazard.id
    Stage = aws_apigatewayv2_stage.prod.name
  }
}

resource "aws_cloudwatch_metric_alarm" "api_latency" {
  alarm_name          = "hazard-api-p99-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "IntegrationLatency"
  namespace           = "AWS/ApiGateway"
  period              = 60
  extended_statistic  = "p99"
  threshold           = 5000  # 5 seconds
  alarm_description   = "P99 latency exceeds 5s on the Hazard Risk API"
  alarm_actions       = [aws_sns_topic.model_alerts.arn]

  dimensions = {
    ApiId = aws_apigatewayv2_api.hazard.id
    Stage = aws_apigatewayv2_stage.prod.name
  }
}

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
        type       = "metric"
        x          = 0; y = 0; width = 12; height = 6
        properties = {
          title   = "API Gateway — Request Count & Error Rate"
          metrics = [
            ["AWS/ApiGateway", "Count",     "ApiId", aws_apigatewayv2_api.hazard.id, "Stage", "prod"],
            ["AWS/ApiGateway", "5XXError",  "ApiId", aws_apigatewayv2_api.hazard.id, "Stage", "prod"],
            ["AWS/ApiGateway", "4XXError",  "ApiId", aws_apigatewayv2_api.hazard.id, "Stage", "prod"],
          ]
          period = 60
          stat   = "Sum"
          view   = "timeSeries"
        }
      },
      {
        type       = "metric"
        x          = 12; y = 0; width = 12; height = 6
        properties = {
          title   = "API Gateway — Latency (P50 / P99)"
          metrics = [
            [{ expression = "SELECT AVG(IntegrationLatency) FROM \"AWS/ApiGateway\" GROUP BY ApiId" }],
          ]
          period = 60
          stat   = "p99"
          view   = "timeSeries"
        }
      },
      {
        type       = "metric"
        x          = 0; y = 6; width = 12; height = 6
        properties = {
          title   = "SageMaker Endpoint — Invocations & Model Latency"
          metrics = [
            ["AWS/SageMaker", "Invocations",          "EndpointName", "hazard-risk-model", "VariantName", "primary"],
            ["AWS/SageMaker", "ModelLatency",          "EndpointName", "hazard-risk-model", "VariantName", "primary"],
            ["AWS/SageMaker", "Invocation4XXErrors",   "EndpointName", "hazard-risk-model", "VariantName", "primary"],
          ]
          period = 60
          stat   = "Average"
          view   = "timeSeries"
        }
      },
    ]
  })
}

# ── SageMaker Model Monitor schedule ─────────────────────────────────────────
resource "aws_sagemaker_data_quality_job_definition" "hazard_monitor" {
  name     = "hazard-data-quality-monitor"
  role_arn = aws_iam_role.sagemaker_execution.arn

  data_quality_app_specification {
    image_uri = "156813124566.dkr.ecr.${var.region}.amazonaws.com/sagemaker-model-monitor-analyzer"
  }

  data_quality_baseline_config {
    statistics_resource {
      s3_uri = "s3://hazard/ml/monitoring/baseline_stats.json"
    }
    constraints_resource {
      s3_uri = "s3://hazard/ml/monitoring/baseline_constraints.json"
    }
  }

  data_quality_job_input {
    endpoint_input {
      endpoint_name              = "hazard-risk-model"
      local_path                 = "/opt/ml/processing/input/endpoint"
      s3_input_mode              = "File"
      s3_data_distribution_type  = "FullyReplicated"
    }
  }

  data_quality_job_output_config {
    monitoring_outputs {
      s3_output {
        local_path    = "/opt/ml/processing/output"
        s3_uri        = "s3://hazard/ml/monitoring/reports/"
        s3_upload_mode = "EndOfJob"
      }
    }
  }

  job_resources {
    cluster_config {
      instance_count    = 1
      instance_type     = "ml.m5.large"
      volume_size_in_gb = 20
    }
  }
}

resource "aws_sagemaker_monitoring_schedule" "hazard_daily" {
  name = "hazard-daily-data-quality"

  monitoring_schedule_config {
    monitoring_job_definition_name = aws_sagemaker_data_quality_job_definition.hazard_monitor.name
    monitoring_type                = "DataQuality"
    schedule_config {
      schedule_expression = "cron(0 6 * * ? *)"  # Daily at 6 AM UTC
    }
  }

  tags = local.common_tags
}

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

resource "aws_lambda_function" "retrain_trigger" {
  function_name = "hazard-retrain-trigger"
  role          = aws_iam_role.sagemaker_execution.arn
  handler       = "ml.monitoring.retrain_trigger.lambda_handler"
  runtime       = "python3.11"
  timeout       = 60

  filename         = "${path.module}/../../lambda_retrain.zip"
  source_code_hash = filebase64sha256("${path.module}/../../lambda_retrain.zip")

  environment {
    variables = {
      PIPELINE_NAME = "hazard-risk-pipeline"
      AWS_REGION    = var.region
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
