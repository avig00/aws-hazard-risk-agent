###############################################################################
# Network — VPC, subnets, internet gateway, security groups
###############################################################################

resource "aws_vpc" "hazard" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, { Name = "hazard-vpc" })
}

resource "aws_internet_gateway" "hazard" {
  vpc_id = aws_vpc.hazard.id
  tags   = merge(local.common_tags, { Name = "hazard-igw" })
}

# Two public subnets across AZs for ALB high-availability
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.hazard.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, { Name = "hazard-public-${count.index}" })
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.hazard.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.hazard.id
  }

  tags = merge(local.common_tags, { Name = "hazard-public-rt" })
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Security groups ───────────────────────────────────────────────────────────

resource "aws_security_group" "alb" {
  name        = "hazard-alb-sg"
  description = "Allow HTTP traffic to ALB"
  vpc_id      = aws_vpc.hazard.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "hazard-alb-sg" })
}

resource "aws_security_group" "ecs_api" {
  name        = "hazard-ecs-api-sg"
  description = "Allow ALB to ECS task on port 8000"
  vpc_id      = aws_vpc.hazard.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "hazard-ecs-sg" })
}
