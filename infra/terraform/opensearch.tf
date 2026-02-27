###############################################################################
# OpenSearch Serverless — vector collection for RAG document index
###############################################################################

resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "hazard-index-encryption"
  type        = "encryption"
  description = "AWS-managed KMS encryption for hazard vector collection"

  policy = jsonencode({
    Rules       = [{ Resource = ["collection/hazard-index"], ResourceType = "collection" }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name        = "hazard-index-network"
  type        = "network"
  description = "Public endpoint access for hazard vector collection"

  policy = jsonencode([{
    Rules = [
      { Resource = ["collection/hazard-index"], ResourceType = "collection" },
      { Resource = ["dashboard/hazard-index"],  ResourceType = "dashboard" },
    ]
    AllowFromPublic = true
  }])
}

resource "aws_opensearchserverless_collection" "hazard_index" {
  name        = "hazard-index"
  type        = "VECTORSEARCH"
  description = "RAG vector index for county hazard documents"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]

  tags = local.common_tags
}

resource "aws_opensearchserverless_access_policy" "hazard_index" {
  name        = "hazard-index-access"
  type        = "data"
  description = "Allow ECS task role and SageMaker role to read/write hazard index"

  policy = jsonencode([{
    Rules = [
      {
        Resource     = ["index/hazard-index/*"]
        Permission   = [
          "aoss:CreateIndex", "aoss:DeleteIndex", "aoss:UpdateIndex",
          "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument",
        ]
        ResourceType = "index"
      },
      {
        Resource     = ["collection/hazard-index"]
        Permission   = ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems"]
        ResourceType = "collection"
      },
    ]
    Principal = [
      aws_iam_role.ecs_task.arn,
      aws_iam_role.sagemaker_execution.arn,
    ]
  }])
}

output "opensearch_collection_endpoint" {
  description = "OpenSearch Serverless collection endpoint URL"
  value       = aws_opensearchserverless_collection.hazard_index.collection_endpoint
}
