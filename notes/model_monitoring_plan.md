# Model Monitoring Plan

## Monitored Metrics
- Data drift in key features (KL divergence)
- Prediction drift (label distribution shifts)
- Model performance decay (based on periodic retraining)
- Outliers in NRI Expected Loss

## Tools
- SageMaker Model Monitor
- Athena queries on inference logs
- CloudWatch alarms for:
  - high latency
  - error spikes
  - drift thresholds

## Schedule
Daily:
- Run drift checks on batch inference logs

Weekly:
- Aggregate performance metrics
- Trigger evaluation & notify if drift detected

Monthly:
- Optional retraining pipeline
