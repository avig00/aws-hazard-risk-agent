# AWS Hazard Risk ML System + RAG Assistant

This project builds a full Machine Learning and Retrieval-Augmented Generation (RAG) system using:
- SageMaker Pipelines (training & deployment)
- MLflow (experiment tracking)
- OpenSearch Serverless as vector database
- Bedrock or local OSS LLMs for generation
- ECS + API Gateway for serving the unified ML + RAG API
- Terraform for full infrastructure automation

## Purpose
To use curated Gold data (from Project 1) to:
1. Train a risk-scoring ML model (predict hazard risk at county level)
2. Build a RAG assistant that provides hazard summaries by retrieving disaster documents + ML predictions

## Components
- ML Pipeline (feature prep → training → evaluation → registry → deployment)
- ML monitoring (data drift + model drift)
- Vector store (OpenSearch Serverless)
- Chunking & embeddings for documents
- Unified API exposing:
  - `/predict` → ML model
  - `/ask` → RAG query endpoint

## Status
This repo contains all design documents, folder structure, pseudo-code, MLflow plan, and Terraform scaffolding.
Implementation starts after holiday break.
