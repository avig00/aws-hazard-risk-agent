# Dataset & Targets for ML System

## Data Source
Gold layer from Project 1:
s3://hazard/gold/risk_feature_mart/

Fields include:
- hazard event counts (NOAA)
- FEMA claim counts & severity
- NRI expected loss + vulnerability + resilience indices
- Census socioeconomic variables

## ML Targets
Two options for supervised modeling:

### 1. Risk Score Regression (default)
Predict continuous **expected loss score** for each county-year.
Target: NRI_ExpectedLoss

### 2. Risk Bucket Classification (optional)
Labels: LOW / MEDIUM / HIGH risk
Based on percentile bins of NRI Expected Loss or model outputs.

## Features
- hazard event frequencies (multi-year)
- fatalities, injuries
- FEMA claim volume, assistance amounts
- NRI indicators (exposure, social vulnerability, resilience)
- Census demographic + income + education + housing
