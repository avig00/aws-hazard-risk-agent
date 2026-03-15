# ML System Architecture

## 1. Data Input

Training data is pulled from the Gold layer via Athena:

```
s3://aws-hazard-risk-vigamogh-dev/hazard/gold/risk_feature_mart/
```

Tools: Athena SQL + boto3 directly. awswrangler was removed — it pulls in Ray which conflicts with pydantic v2. PySpark/EMR not used.

Scripts:
- `ml/data_prep/build_training_data.py` — Athena → Parquet train/test splits

---

## 2. Feature Engineering

Script: `ml/data_prep/feature_engineering.py`

Steps:
1. Load train/test Parquet
2. Impute missing numeric values (median)
3. Log-transform `median_household_income`
4. Encode `state` as one-hot dummies
5. Derive rate features (`events_per_capita`)
6. Drop rows with missing target (`risk_bucket`)
7. Whitelist enforcement — drops any column not in the config's `numeric` list, passthrough IDs, `_log` suffixes, or `state_` prefixes. Prevents unlisted columns from silently entering the feature matrix.

Outputs:
- `train.parquet` / `test.parquet` (local, then passed to training)
- `model_feature_cols.json` → `s3://aws-hazard-risk-vigamogh-dev/hazard/ml/features/model_feature_cols.json`

Feature count: 74 (storm event counts + non-dollar severity + demographics + geography + state dummies)

Dollar damage columns (`*_total_damage`) are explicitly excluded — using same-period NOAA damage co-variates to predict FEMA damage tiers is circular reasoning. Whitelist enforces this at step 7.

---

## 3. Model Training

Model: XGBoost (`multi:softmax`, 3 classes). LightGBM, CatBoost, and TabTransformer were not pursued — XGBoost is the right fit for tabular county-level data at this scale.

Target: `risk_bucket` — LOW / MEDIUM / HIGH, derived from equal tertiles of `fema_total_damage`. Random baseline = 33%.

Hyperparameters (from `config/model_config.yml`):
- `max_depth=4`, `learning_rate=0.05`, `n_estimators=800`
- `subsample=0.8`, `colsample_bytree=0.6`

Validation: 5-fold stratified cross-validation

Results (v3):
- Test accuracy: 69.3%
- CV balanced accuracy: 67.9% ± 2.3%
- F1 macro: 0.69

Tracking: MLflow autolog, local SQLite (`mlflow.db`), experiment `hazard-risk-xgboost`
- v3 run ID: `2423fda164fd4dbd9ccc35913b5d0e28`

Actual training workflow: `ml/pipeline/package_and_register.py`
- train → save model as `xgboost-model` → tar.gz → upload to S3 → register in SageMaker Model Registry → update endpoint

SageMaker Pipeline (`ml/pipeline/sagemaker_pipeline.py`) is defined but the `ml.m5.large` processing quota is 0 on this account, so pipeline execution fails at the DataPrep step. `package_and_register.py` is the working substitute.

---

## 4. Model Deployment

Endpoint: `hazard-risk-model` (SageMaker Serverless, InService)
- Serverless config: memory 2048 MB, max concurrency 5
- Model artifact: `s3://aws-hazard-risk-vigamogh-dev/hazard/ml/artifacts/model.tar.gz`
- Execution role: `arn:aws:iam::945919380353:role/hazard-sagemaker-execution-role`

Model registry: `hazard-risk-model-group`, version 4, status Approved

At inference time, `inference_service.py` loads `model_feature_cols.json` from S3 to align the 74-feature vector. Missing demographic columns are filled with 0.

ECS Fargate + FastAPI is in the Terraform infra definition but Streamlit calls boto3 directly — no intermediary API layer at runtime.

---

## 5. Model Monitoring

Implemented via Terraform:
- CloudWatch alarms on endpoint latency and error rate
- EventBridge rule → Lambda → SageMaker Pipeline re-execution on drift alerts
- Drift detection: KL divergence on feature distributions (daily schedule)

SageMaker DataCapture was removed — serverless endpoints do not support it.

---

## 6. RAG Integration

Vector store: Pinecone (index: `hazard-risk-docs`, 1536-dim cosine, 24 vectors)
- OpenSearch Serverless was the original plan; switched to Pinecone to avoid provisioned capacity costs

Embeddings: Amazon Titan Embeddings v1 (`amazon.titan-embed-text-v1`)

LLM: `us.amazon.nova-lite-v1:0` via Bedrock Converse API
- All Anthropic models are blocked on this account (missing Bedrock use-case agreement)
- Nova Lite validated at equivalent quality: 4.5/5.0 avg on LLM-as-Judge eval, 8 PASS / 2 WARN / 0 FAIL

Unified agent response includes:
- ML prediction (tier + class probabilities)
- TAG result (governed Athena SQL → LLM narrative)
- RAG result (Pinecone kNN retrieval → LLM synthesis with citations)
