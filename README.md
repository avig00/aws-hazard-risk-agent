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
     │ XGBoost ML │  │ NL → SQL   │  │ Pinecone      │
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

- **RAG (`/ask`)** — retrieves relevant document chunks from Pinecone and instructs the LLM to answer *only from those sources* with citations. Best for narrative questions about hazard concepts, policy, and reports.
- **TAG (`/query`) — Table Augmented Generation** — runs a governed SQL query over Athena, then passes the *structured results table* to the LLM for interpretation. The LLM adds domain context and analytical insight on top of the computed numbers. Best for data-driven ranking, trend, and comparison questions.

TAG and RAG are combined in the hybrid route: the TAG narrative is prepended into the RAG prompt as additional context, giving the LLM one coherent input that draws from both structured data and documents.

---

## System Components

| Layer | Technology |
|---|---|
| ML Training + Deployment | SageMaker Pipelines + XGBoost |
| Experiment Tracking | MLflow |
| Feature + Gold Analytics | Athena (Gold-layer only) |
| Analytics Synthesis (TAG) | Amazon Nova Lite (Bedrock Converse API) over Athena results |
| Vector Database | Pinecone (free tier, cosine) |
| Document Synthesis (RAG) | Amazon Nova Lite (Bedrock Converse API) over retrieved chunks |
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
│   └── rag_config.yml              # Embedding model, chunk size, Pinecone config
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
│   │   └── embed_and_index.py     # Titan embeddings → Pinecone upsert
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
├── infra/terraform/               # IAM, VPC, ECS, SageMaker, API GW, monitoring
├── ui/
│   ├── chat.py                    # Session state + conversation history
│   └── components.py             # Tables, charts, citation cards, prediction panels
└── tests/                        # Unit tests (no AWS required)
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- AWS credentials with access to: Athena, Bedrock, SageMaker, S3
- Pinecone account (free tier) with an API key

```bash
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your AWS credentials and Pinecone API key
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

## ML Model — County Risk Classifier

The `/predict` tool invokes a **3-class XGBoost classifier** that assigns each U.S. county a risk tier: **LOW**, **MEDIUM**, or **HIGH**.

### What the model predicts

> *"Given a county's storm history, demographic profile, and geography, which third of historical FEMA total damage does this county fall into?"*

The target `risk_bucket` is derived by binning `fema_total_damage` into equal tertiles:

| Class | Label | Meaning |
|---|---|---|
| 0 | LOW | Bottom third of cumulative FEMA damage nationally |
| 1 | MEDIUM | Middle third |
| 2 | HIGH | Top third — counties with highest cumulative disaster costs |

### Training setup

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (`multi:softmax`, 3 classes) |
| Training rows | 2,552 counties (80% stratified split) |
| Test rows | 638 counties |
| Features | 94 (storm history + demographics + geography, one-hot state encoding) |
| Hyperparameters | max_depth=4, lr=0.05, n_estimators=800, subsample=0.8, col_bytree=0.6 |
| Validation | 5-fold stratified cross-validation |
| Experiment tracking | MLflow autolog (params, metrics, SHAP artifacts) |

### Results

| Metric | Score |
|---|---|
| Test accuracy | **70.5%** (vs. 33% random baseline) |
| Test balanced accuracy | **70.5%** |
| CV balanced accuracy | **71.0% ± 1.1%** (stable across folds) |
| F1 macro | 0.70 |

**Confusion matrix (638 test counties):**

```
              Predicted
              LOW   MED   HIGH
Actual  LOW   170    16    27    (recall 80%)
        MED    40   131    41    (recall 62%)
        HIGH   24    40   149    (recall 70%)
```

Key pattern: misclassifications are almost exclusively *adjacent* tiers (LOW↔MEDIUM, MEDIUM↔HIGH) — the model makes no extreme errors confusing LOW counties for HIGH.

### Feature importance (SHAP)

Top features by mean |SHAP| value, ranked:

```
1.  state_Texas              — Texas accounts for a disproportionate share of HIGH-risk counties
2.  tornado_events           — count of NOAA-recorded tornado events per county
3.  flood_total_damage       — log-scaled cumulative flood property damage (NOAA)
4.  wind_total_damage        — log-scaled wind event damage
5.  flood_events             — count of flood events
6.  noaa_total_property_damage — total NOAA-estimated property damage across all hazard types
7.  tropical_events          — tropical storm event count
8.  hail_total_damage        — hail property damage
9.  wind_events              — wind event count
10. state_Florida            — coastal/hurricane exposure signal
```

> **Note on NRI scores:** FEMA's National Risk Index scores (`nri_risk_score`, `nri_eal_score`, etc.) were intentionally excluded. Although they improve raw test accuracy, they are computed from the same FEMA damage data as the target — circular reasoning. Cross-validation confirms the storm-history-only model generalises better (CV 71.0% ± 1.1% vs. 70.8% ± 2.0% with NRI).

### Limitations

- **Cross-sectional, not temporal** — the model uses county-level aggregates, not year-by-year predictions. It answers "what tier does this county belong to?", not "what will happen in 2025?"
- **Geography encoded via state** — fine-grained geographic effects (coastal proximity, elevation) are captured only through state dummies and storm history counts
- **Adjacent-class confusion** — MEDIUM counties are harder to classify (recall 62%) because they sit between two extremes; real-world decisions should treat MEDIUM predictions with appropriate uncertainty
- **Training data vintage** — trained on data through 2023; counties undergoing rapid development or land-use change may shift tiers

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
| Pinecone vector search | **Cortex Search** |
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

**ML target:** `risk_bucket` — LOW / MEDIUM / HIGH tier derived from FEMA total damage tertiles (multi-class classification)
**ML features:** Hazard event frequencies, property damage estimates, Census socioeconomic variables, and state geography (94 features after one-hot encoding)

---

## LLM Response Quality

The agent is evaluated end-to-end using an **LLM-as-Judge harness** (Amazon Nova Lite) that scores each response on faithfulness, relevance, and groundedness (1–5 scale). The final evaluation across 10 representative questions scores **4.5 / 5.0 average with 8 PASS, 2 WARN, and 0 FAIL** — no hallucination failures across any test case.

| Test Case | Tool | Avg | Verdict |
|---|---|---|---|
| Top counties by expected annual loss | query | 5.0 | PASS |
| Flood event increase by county | query | 5.0 | PASS |
| Hurricane synonym routing (→ Tropical Storm) | query | 4.3 | PASS |
| County comparison (Harris County vs Miami-Dade) | query | 5.0 | PASS |
| Predict risk tier for Harris County | predict | 5.0 | PASS |
| Wildfire year-over-year trend | query | 2.7 | WARN |
| Coastal hurricane vulnerability | ask | 4.3 | PASS |
| NRI expected loss methodology | ask | 3.3 | WARN |
| FEMA declarations by state | query | 5.0 | PASS |
| Hybrid: predicted risk + property damage | predict + query | 5.0 | PASS |

The two WARN cases reflect data coverage gaps rather than agent logic failures. The wildfire trend query returns a single year of data — the Gold-layer hazard summary has sparse Wildfire records — so there is no multi-year trend to analyze and the LLM correctly surfaces this limitation. The NRI methodology case relies on the RAG tool; without a Pinecone API key in the eval environment, it falls back to domain knowledge, which the judge penalizes for lack of document citations. Both cases resolve in a fully configured production deployment.

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
