# ML System Architecture (Project 2)

## 1. Data Input
Training data is pulled from:
s3://hazard/gold/risk_feature_mart/

Tools:
- Athena SQL queries
- Pandas or PySpark (EMR/SageMaker Processing)

---

## 2. Feature Engineering
Scripts:
- ml/data_prep/build_training_data.py
- ml/data_prep/feature_engineering.py

Outputs:
- train.parquet
- test.parquet
- feature_list.json

---

## 3. Model Training
Use SageMaker Pipeline with steps:
1. Data prep
2. Training
3. Evaluation
4. Model registration

Models considered:
- XGBoost (default)
- LightGBM
- CatBoost
- TabTransformer (optional)

Tracking:
- MLflow experiment tracking

---

## 4. Model Deployment
Deploy best model via:
- SageMaker Endpoint (real-time)
or
- ECS Fargate container with FastAPI

---

## 5. Model Monitoring
Plan:
- Data drift detection (input feature distribution)
- Prediction drift
- Performance decay alerts

Stored in:
notes/model_monitoring_plan.md

---

## 6. RAG Integration
- Vector store: OpenSearch Serverless
- Embeddings model: Amazon Titan Embeddings or local instructor-base
- LLM: Bedrock Claude/Sonnet, or Llama2/3 on ECS

Unified API returns:
- Model prediction
- Retrieved context
- LLM-generated explanation
