# AWS Hazard Risk Intelligence Agent

An enterprise-grade agentic AI system that answers complex county-level disaster risk questions by combining **predictive ML**, **governed NL→SQL analytics**, and **RAG-based document retrieval** — with a live Streamlit dashboard.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app)

---

## What It Does

The agent answers questions like:

> *"Which counties saw the largest increase in flood events from 2015–2023?"*
> *"Show the top 10 counties by predicted risk and highest average property damage."*
> *"Why are coastal counties more vulnerable to hurricanes?"*

These queries require real computation over structured datasets and document corpora — not just retrieval.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Agent Orchestrator                  │
│         (Intent routing · Tool selection)            │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
     ┌─────▼─────┐  ┌─────▼──────┐  ┌───▼──────────┐
     │  /predict  │  │  /query    │  │   /ask        │
     │            │  │            │  │               │
     │ XGBoost ML │  │ NL → SQL   │  │ OpenSearch    │
     │ SageMaker  │  │ Athena     │  │ kNN retrieval │
     │ Endpoint   │  │ Gold Layer │  │ + Bedrock LLM │
     └─────┬──────┘  └─────┬──────┘  └──────┬────────┘
           │               │                 │
           └───────────────┴─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  Streamlit   │
                    │  Dashboard   │
                    │(Cloud-hosted)│
                    └──────────────┘
```

### Tool Routing Logic

| Question type | Tools invoked |
|---|---|
| "Predicted risk score for X county" | `/predict` |
| "Top N counties by damage 2015–2023" | `/query` + TAG synthesis |
| "Why are floods increasing in Harris County?" | `/ask` (RAG) |
| "Top counties by predicted risk and damage" | `/predict` + `/query` + TAG (hybrid) |

### TAG vs RAG

The system uses two distinct LLM grounding strategies:

- **RAG (`/ask`)** — retrieves relevant document chunks from OpenSearch and instructs the LLM to answer *only from those sources* with citations. Best for narrative questions about hazard concepts, policy, and reports.
- **TAG (`/query`) — Table Augmented Generation** — runs a governed SQL query over Athena, then passes the *structured results table* to the LLM for interpretation. The LLM adds domain context and analytical insight on top of the computed numbers. Best for data-driven ranking, trend, and comparison questions.

TAG and RAG are combined in the hybrid route: the TAG narrative is prepended into the RAG prompt as additional context, giving the LLM one coherent input that draws from both structured data and documents.

---

## System Components

| Layer | Technology |
|---|---|
| ML Training + Deployment | SageMaker Pipelines + XGBoost |
| Experiment Tracking | MLflow |
| Feature + Gold Analytics | Athena (Gold-layer only) |
| Analytics Synthesis (TAG) | Bedrock Claude 3 Sonnet over Athena results |
| Vector Database | OpenSearch Serverless (kNN, cosine) |
| Document Synthesis (RAG) | Bedrock Claude 3 Sonnet over retrieved chunks |
| Serving Layer | ECS Fargate + API Gateway |
| Infrastructure as Code | Terraform |
| Frontend | Streamlit Cloud |

---

## Project Structure

```
aws-hazard-risk-agent/
├── app.py                          # Streamlit entrypoint (Streamlit Cloud deploy target)
├── requirements.txt
├── config/
│   ├── model_config.yml            # XGBoost hyperparams, feature schema, MLflow settings
│   └── rag_config.yml              # Embedding model, chunk size, OpenSearch config
├── ml/
│   ├── data_prep/
│   │   ├── build_training_data.py  # Athena → Parquet train/test splits
│   │   └── feature_engineering.py # Log transforms, lag features, imputation, encoding
│   ├── training/
│   │   ├── train_model.py          # XGBoost + 5-fold CV + MLflow autolog
│   │   └── evaluation.py          # RMSE/MAE/R² + SHAP feature importance
│   ├── inference/
│   │   ├── inference_service.py   # SageMaker real-time endpoint invocation
│   │   └── batch_inference.py     # Batch scoring → S3 Gold layer
│   ├── pipeline/
│   │   └── sagemaker_pipeline.py  # 4-step SageMaker Pipeline SDK definition
│   └── monitoring/
│       ├── drift_detector.py      # KL divergence drift detection on data capture
│       └── retrain_trigger.py     # EventBridge → Lambda → pipeline re-execution
├── rag/
│   ├── indexing/
│   │   ├── chunk_documents.py     # PDF/TXT → overlapping chunks + metadata
│   │   └── embed_and_index.py     # Titan embeddings → OpenSearch bulk index
│   ├── retrieval/
│   │   └── retrieve.py            # kNN vector search with score filtering
│   ├── prompts/
│   │   ├── ask_template.py        # RAG prompt templates + citation builder
│   │   └── tag_template.py        # TAG system prompt + table formatter + column legend
│   └── api/
│       └── app.py                 # FastAPI: /predict, /ask, /query, /agent
├── analytics/
│   ├── intent_classifier.py       # NL → SQL template routing
│   ├── query_engine.py            # Governed Athena executor with guardrails
│   └── sql_templates/             # Parameterized SQL templates (Gold-only)
├── agent/
│   ├── router.py                  # Multi-signal intent routing rules
│   └── orchestrator.py            # Top-level agent loop + tool execution
├── infra/terraform/               # IAM, VPC, ECS, OpenSearch, API GW, monitoring
├── ui/
│   ├── chat.py                    # Session state + conversation history
│   └── components.py             # Tables, charts, citation cards, prediction panels
└── tests/                        # Unit tests (no AWS required)
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- AWS credentials with access to: Athena, Bedrock, SageMaker, OpenSearch Serverless, S3

```bash
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your AWS credentials and OpenSearch endpoint
```

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the FastAPI backend

```bash
uvicorn rag.api.app:app --reload --port 8000
```

### Train the model

```bash
# Pull Gold layer data from Athena and build train/test splits
python ml/data_prep/build_training_data.py

# Train XGBoost with MLflow experiment tracking
python ml/training/train_model.py

# (Optional) Run the full 4-step SageMaker Pipeline
python ml/pipeline/sagemaker_pipeline.py --action create
python ml/pipeline/sagemaker_pipeline.py --action execute
```

### Index the RAG corpus

```bash
python rag/indexing/embed_and_index.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Governance & Safety

The `/query` endpoint supports two response modes:

- `synthesize: true` (default) — runs the governed Athena query, then calls Bedrock Claude to produce a narrative analyst answer from the results table (TAG pipeline)
- `synthesize: false` — returns raw Athena rows as JSON, useful for programmatic consumers or cost control

The `/query` analytics tool enforces strict guardrails:

- **Gold-layer only** — queries must reference `hazard_gold` database
- **Template-based SQL** — no free-form input, preventing injection
- **Automatic LIMIT** — every query capped at 100 rows
- **Year range validation** — prevents full-table scans
- **Scan-cost cap** — ~100 MB Athena scan limit per query

---

## MLOps

| Concern | Implementation |
|---|---|
| Experiment tracking | MLflow autolog (params, metrics, SHAP artifacts) |
| Model registry | SageMaker Model Registry with manual approval gate |
| Data capture | SageMaker `DataCaptureConfig` on endpoint (20% sampling) |
| Drift detection | KL divergence on feature distributions (daily schedule) |
| Retraining | EventBridge → Lambda → SageMaker Pipeline (monthly + on-drift) |
| Alerting | CloudWatch alarms → SNS → email (latency, errors, drift) |

---

## Snowflake-Native Equivalent Architecture

This project uses AWS-native tooling. The same design maps directly onto Snowflake's AI platform:

| This Project (AWS) | Snowflake Equivalent |
|---|---|
| Athena + Gold-layer S3 tables | Snowflake SQL + Gold-layer tables |
| Governed NL→SQL `/query` tool | **Cortex Analyst** |
| OpenSearch Serverless vector search | **Cortex Search** |
| Bedrock Claude LLM synthesis | **Cortex LLM Functions** (`COMPLETE`) |
| SageMaker Pipelines + MLflow | **Snowflake ML** (feature store, model registry) |
| Streamlit Cloud frontend | **Streamlit in Snowflake** |

The architectural pattern — governed analytics + vector retrieval + LLM synthesis + Streamlit UI — is identical to Snowflake Cortex AI, built from first principles on AWS.

---

## Data Design

The agent operates exclusively on the **Gold layer** — curated analytical tables with one row per county per year:

- `gold_hazard_event_summary` — event counts per county, hazard type, year
- `gold_county_risk_scores` — NRI expected loss, exposure, vulnerability, resilience
- `gold_risk_feature_mart` — ML-ready feature set combining all Gold sources

**ML target:** `NRI_ExpectedLoss` (continuous regression)
**ML features:** Hazard event frequencies, FEMA claim volumes/amounts, NRI indicators, Census socioeconomic variables

---

## Example Questions

| Question | Tool(s) |
|---|---|
| "Top 10 counties by flood risk 2015–2023" | `/query` |
| "Counties with largest increase in tornado events" | `/query` |
| "Show predicted risk vs property damage for Texas" | `/predict` + `/query` |
| "Why is Harris County flood risk so high?" | `/ask` |
| "What does NRI expected loss measure?" | `/ask` |
| "Compare Harris County and Miami-Dade hazard profiles" | `/query` |
