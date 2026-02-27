###############################################################################
# API Gateway — HTTP API routing to ALB (ECS FastAPI service)
###############################################################################

resource "aws_apigatewayv2_api" "hazard" {
  name          = "hazard-risk-api"
  protocol_type = "HTTP"
  description   = "Hazard Risk Intelligence Agent — unified tool API"

  cors_configuration {
    allow_headers = ["Content-Type", "Authorization"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_origins = ["*"]
    max_age       = 300
  }

  tags = local.common_tags
}

resource "aws_apigatewayv2_vpc_link" "hazard" {
  name               = "hazard-vpc-link"
  security_group_ids = [aws_security_group.ecs_api.id]
  subnet_ids         = aws_subnet.public[*].id

  tags = local.common_tags
}

resource "aws_apigatewayv2_integration" "hazard_alb" {
  api_id             = aws_apigatewayv2_api.hazard.id
  integration_type   = "HTTP_PROXY"
  integration_uri    = aws_lb_listener.hazard_api.arn
  integration_method = "ANY"
  connection_type    = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.hazard.id
}

# Routes
locals {
  api_routes = ["/predict", "/ask", "/query", "/agent", "/health"]
}

resource "aws_apigatewayv2_route" "hazard" {
  for_each  = toset(local.api_routes)
  api_id    = aws_apigatewayv2_api.hazard.id
  route_key = "ANY ${each.value}"
  target    = "integrations/${aws_apigatewayv2_integration.hazard_alb.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.hazard.id
  name        = "prod"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      responseLength = "$context.responseLength"
      duration       = "$context.integrationLatency"
    })
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/hazard-risk-api"
  retention_in_days = 14
  tags              = local.common_tags
}

output "api_gateway_url" {
  description = "API Gateway invoke URL"
  value       = "${aws_apigatewayv2_stage.prod.invoke_url}"
}
