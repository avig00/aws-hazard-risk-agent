###############################################################################
# Secrets Manager — store sensitive config values referenced by ECS tasks
###############################################################################

resource "aws_secretsmanager_secret" "opensearch_endpoint" {
  name                    = "hazard/opensearch-endpoint"
  description             = "OpenSearch Serverless collection endpoint URL"
  recovery_window_in_days = 7

  tags = local.common_tags
}

# Populated after OpenSearch collection is created
resource "aws_secretsmanager_secret_version" "opensearch_endpoint" {
  secret_id     = aws_secretsmanager_secret.opensearch_endpoint.id
  secret_string = aws_opensearchserverless_collection.hazard_index.collection_endpoint

  depends_on = [aws_opensearchserverless_collection.hazard_index]
}
