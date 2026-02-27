###############################################################################
# ECR + ECS Fargate — container registry and service for the FastAPI agent API
###############################################################################

# ── ECR repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "hazard_api" {
  name                 = "hazard-risk-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "hazard_api" {
  repository = aws_ecr_repository.hazard_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# ── ECS cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "hazard" {
  name = "hazard-risk-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

# ── CloudWatch log group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs_api" {
  name              = "/ecs/hazard-risk-api"
  retention_in_days = 30
  tags              = local.common_tags
}

# ── ECS task definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "hazard_api" {
  family                   = "hazard-risk-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "hazard-api"
    image = "${aws_ecr_repository.hazard_api.repository_url}:latest"
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "SAGEMAKER_ENDPOINT",      value = "hazard-risk-model" },
      { name = "ATHENA_OUTPUT_LOCATION",  value = "s3://hazard/athena-results/" },
      { name = "AWS_DEFAULT_REGION",      value = var.region },
    ]
    secrets = [{
      name      = "OPENSEARCH_ENDPOINT"
      valueFrom = aws_secretsmanager_secret.opensearch_endpoint.arn
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_api.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = local.common_tags
}

# ── ECS service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "hazard_api" {
  name            = "hazard-risk-api"
  cluster         = aws_ecs_cluster.hazard.id
  task_definition = aws_ecs_task_definition.hazard_api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_api.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.hazard_api.arn
    container_name   = "hazard-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.hazard_api]

  tags = local.common_tags
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "hazard_api" {
  name               = "hazard-risk-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = local.common_tags
}

resource "aws_lb_target_group" "hazard_api" {
  name        = "hazard-risk-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.hazard.id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "hazard_api" {
  load_balancer_arn = aws_lb.hazard_api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.hazard_api.arn
  }
}

output "alb_dns_name" {
  description = "ALB DNS name for the FastAPI agent API"
  value       = aws_lb.hazard_api.dns_name
}
