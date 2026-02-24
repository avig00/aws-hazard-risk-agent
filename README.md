# AWS Hazard Risk Intelligence Agent (ML + Analytics + RAG)

This project builds an agentic hazard risk intelligence system that answers complex county-level disaster risk questions by combining:

- Predictive Machine Learning risk scoring  
- Governed natural-language analytics over curated Gold datasets (NL → SQL)  
- Retrieval-Augmented Generation (RAG) over disaster documentation  
- Multi-tool orchestration through a unified agent API  

Unlike a standard chatbot, this system behaves as an AI agent: it interprets user goals, selects the correct toolchain, executes structured actions, and returns grounded, data-backed answers.

---

## Purpose

Using curated Gold-layer hazard and claims data, the agent enables:

- County-level hazard risk prediction  
- Trend and change detection across years  
- Multi-metric county ranking (risk × damage × exposure)  
- Evidence-backed disaster summaries from retrieved documents  

The end goal is an end-to-end risk intelligence assistant for hazard modeling, insurance analytics, and resilience planning.

---

## Agent Capabilities

The agent can answer questions such as:

- “Which counties saw the largest increase in flood events from 2015–2023?”  
- “Show the top 10 counties by predicted risk and highest average property damage.”  
- “Compare hazard type distributions for Harris County vs Miami-Dade over the last 20 years.”  

These queries require real computation over structured datasets — not just document retrieval.

---

## Agent Execution Loop

The system follows a tool-based agent workflow:

1. Intent Understanding  
   Parses the user’s question into a structured analytic goal.

2. Tool Selection + Planning  
   Routes the request to the appropriate execution path:

   - ML prediction  
   - Gold-layer analytics query  
   - Document retrieval  
   - Hybrid multi-step reasoning  

3. Action Execution  
   Executes governed tool calls (SQL templates, vector retrieval, model inference).

4. Grounded Response Synthesis  
   Combines computed metrics and retrieved evidence into a final answer.

---

## Agent Tooling (Unified API)

The agent exposes a single interface with multiple tools:

### `/predict` — Risk Prediction Tool  
Returns county-level hazard risk scores from the deployed ML model.

### `/query` — Analytics Tool (NL → Governed SQL)  
Executes structured analytic questions over Gold datasets via Athena.

- Only approved query intents  
- Gold-only table access  
- Partition and cost guardrails  
- Deterministic, testable outputs  

### `/ask` — Retrieval + Reasoning Tool  
Answers open-ended hazard questions using:

- OpenSearch vector retrieval  
- Document chunking and embeddings  
- LLM-based grounded synthesis  

---

## System Components

| Layer | Technology |
|------|------------|
| ML Training + Deployment | SageMaker Pipelines |
| Experiment Tracking | MLflow |
| Feature + Gold Analytics | Athena (Gold-only) |
| Vector Database | OpenSearch Serverless |
| LLM Generation | Bedrock or local OSS models |
| Serving Layer | ECS + API Gateway |
| Infrastructure as Code | Terraform |

---

## Data Design (Gold Layer)

The agent operates only on curated Gold business-ready marts, enabling:

- One row per county per year  
- No downstream joins required  
- Direct support for analytics and ML inference  

Example marts:

- `gold_hazard_event_summary`  
- `gold_county_risk_scores`  
- `gold_risk_feature_mart`  

---

## Governance and Safety

To ensure production-grade reliability, the analytics agent enforces:

- No raw Bronze access  
- Gold-only query execution  
- Template-based SQL compilation (no free-form SQL)  
- Automatic LIMIT and partition filtering  
- Scan-cost controls for Athena queries  

This prevents hallucinated or expensive queries while maintaining flexibility.

---

## Monitoring and MLOps

The system includes end-to-end ML lifecycle support:

- Data drift monitoring  
- Model drift detection  
- Automated retraining triggers  
- Model registry and promotion workflows  
- Audit-ready experiment tracking in MLflow  

---

## Project Status

This repository currently contains:

- Full design documentation  
- Folder structure and scaffolding  
- Terraform infrastructure plan  
- MLflow and pipeline architecture  
- Agent tool schema and query intent templates  

Implementation begins after the holiday break.

---

## Outcome

This project demonstrates how to build a real-world agentic AI system that integrates:

- Predictive modeling  
- Structured analytics  
- Retrieval-based reasoning  
- Governed tool execution  
- Cloud-native deployment  

It is designed as a portfolio-grade example of an enterprise-style risk intelligence agent for insurance and hazard analytics.

---
