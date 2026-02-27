"""
Feature engineering transforms applied to the Gold-layer risk feature mart.

Responsibilities:
- Log-transform skewed financial/loss features
- Impute missing values
- Create multi-year lag features
- One-hot encode categoricals
- Return transformed DataFrame ready for XGBoost
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply log1p to skewed financial features to reduce right-skew."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            logger.debug("Log-transformed: %s", col)
    return df


def impute_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Median imputation for numeric features."""
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed %s with median=%.4f", col, median_val)
    return df


def add_lag_features(df: pd.DataFrame, columns: list, lags: list = None) -> pd.DataFrame:
    """
    Create lag features per county (sorted by year).
    Requires 'county_fips' and 'year' columns.
    """
    if "county_fips" not in df.columns or "year" not in df.columns:
        logger.warning("Skipping lag features: county_fips or year column not found")
        return df

    if lags is None:
        lags = [1, 2, 3]

    df = df.sort_values(["county_fips", "year"]).copy()
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby("county_fips")[col].shift(lag)
            logger.debug("Created lag feature: %s", lag_col)
    return df


def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode low-cardinality categorical columns."""
    df = df.copy()
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df
    df = pd.get_dummies(df, columns=existing, drop_first=True, dtype=float)
    logger.info("One-hot encoded: %s", existing)
    return df


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier/metadata columns that should not be model inputs."""
    drop_cols = ["county_name", "state_name", "geometry", "risk_bucket"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def engineer_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Applies log transforms, imputation, lag features, and encoding.
    Returns a model-ready DataFrame.
    """
    if config is None:
        config = load_config()

    feat_cfg = config["features"]

    logger.info("Starting feature engineering on %d rows", len(df))

    # 1. Log-transform skewed columns
    df = log_transform(df, feat_cfg.get("log_transform", []))

    # 2. Median imputation for numeric features
    df = impute_numeric(df, feat_cfg.get("numeric", []))

    # 3. Lag features for temporal signals
    df = add_lag_features(df, feat_cfg.get("lag_features", []))

    # 4. One-hot encode categoricals
    df = encode_categoricals(df, feat_cfg.get("categorical", []))

    # 5. Drop any remaining NaN rows (from lag features on early years)
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    if rows_before != rows_after:
        logger.info("Dropped %d rows with NaN (likely lag boundary rows)", rows_before - rows_after)

    # 6. Remove non-feature identifier columns
    df = drop_non_features(df)

    logger.info("Feature engineering complete: %d rows, %d columns", len(df), len(df.columns))
    return df


if __name__ == "__main__":
    # Smoke test with synthetic data
    sample = pd.DataFrame({
        "county_fips": ["01001", "01001", "01001", "01003", "01003", "01003"],
        "year": [2019, 2020, 2021, 2019, 2020, 2021],
        "total_events": [5, 7, 6, 12, 14, 11],
        "fema_property_damage": [100000, 250000, 180000, 500000, 750000, 600000],
        "fema_total_assistance": [50000, 120000, 90000, 200000, 350000, 280000],
        "NRI_ExpectedLoss": [15000, 18000, 16000, 45000, 52000, 48000],
        "NRI_Exposure": [0.6, 0.65, 0.63, 0.8, 0.82, 0.81],
        "fema_claim_count": [10, 14, 12, 25, 30, 27],
        "census_median_income": [52000, 53000, 54000, 61000, 62000, 63000],
        "state": ["AL", "AL", "AL", "AL", "AL", "AL"],
    })

    config = load_config()
    result = engineer_features(sample, config)
    print(result.head())
    print("Columns:", result.columns.tolist())
