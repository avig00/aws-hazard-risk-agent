"""
Unit tests for ml/data_prep/feature_engineering.py

Tests run without any AWS connectivity — uses synthetic in-memory DataFrames.
"""
import numpy as np
import pandas as pd
import pytest

from ml.data_prep.feature_engineering import (
    add_lag_features,
    encode_categoricals,
    engineer_features,
    impute_numeric,
    log_transform,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "county_fips": ["01001"] * 4 + ["01003"] * 4,
        "year": [2019, 2020, 2021, 2022] * 2,
        "total_events": [5, 7, 6, 8, 12, 14, 11, 15],
        "fema_property_damage": [100_000, 250_000, None, 300_000,
                                  500_000, 750_000, 600_000, 800_000],
        "fema_total_assistance": [50_000, 120_000, 90_000, 110_000,
                                   200_000, 350_000, 280_000, 320_000],
        "NRI_ExpectedLoss": [15_000, 18_000, 16_000, 19_000,
                              45_000, 52_000, 48_000, 55_000],
        "census_median_income": [52_000, 53_000, 54_000, 55_000,
                                  61_000, 62_000, 63_000, 64_000],
        "fema_claim_count": [10, 14, 12, 16, 25, 30, 27, 32],
        "state": ["AL"] * 8,
    })


# ── log_transform ─────────────────────────────────────────────────────────────

def test_log_transform_creates_log_columns(sample_df):
    result = log_transform(sample_df, ["fema_property_damage"])
    assert "fema_property_damage_log" in result.columns


def test_log_transform_non_negative(sample_df):
    result = log_transform(sample_df, ["fema_property_damage"])
    assert (result["fema_property_damage_log"].dropna() >= 0).all()


def test_log_transform_ignores_missing_columns(sample_df):
    # Should not raise even if column doesn't exist
    result = log_transform(sample_df, ["nonexistent_column"])
    assert "nonexistent_column_log" not in result.columns


# ── impute_numeric ────────────────────────────────────────────────────────────

def test_impute_fills_nulls(sample_df):
    assert sample_df["fema_property_damage"].isnull().any()
    result = impute_numeric(sample_df, ["fema_property_damage"])
    assert result["fema_property_damage"].isnull().sum() == 0


def test_impute_uses_median(sample_df):
    series = sample_df["fema_property_damage"].dropna()
    expected_median = series.median()
    result = impute_numeric(sample_df, ["fema_property_damage"])
    # The null row should now be close to the median
    imputed_val = result.loc[sample_df["fema_property_damage"].isnull(), "fema_property_damage"].iloc[0]
    assert abs(imputed_val - expected_median) < 1.0


def test_impute_no_op_on_complete_column(sample_df):
    result = impute_numeric(sample_df, ["total_events"])
    assert result["total_events"].equals(sample_df["total_events"])


# ── add_lag_features ──────────────────────────────────────────────────────────

def test_lag_features_created(sample_df):
    result = add_lag_features(sample_df, ["total_events"], lags=[1, 2])
    assert "total_events_lag1" in result.columns
    assert "total_events_lag2" in result.columns


def test_lag1_values_correct(sample_df):
    result = add_lag_features(sample_df, ["total_events"], lags=[1])
    # For county 01001, year 2020, lag1 should be year 2019's value (5)
    row = result[(result["county_fips"] == "01001") & (result["year"] == 2020)]
    assert row["total_events_lag1"].iloc[0] == 5


def test_lag_nan_at_boundary(sample_df):
    result = add_lag_features(sample_df, ["total_events"], lags=[1])
    # First year per county should have NaN lag
    first_rows = result[result["year"] == 2019]
    assert first_rows["total_events_lag1"].isnull().all()


def test_lag_skipped_without_required_columns():
    df = pd.DataFrame({"total_events": [1, 2, 3]})
    result = add_lag_features(df, ["total_events"], lags=[1])
    # No county_fips / year → should return unchanged
    assert "total_events_lag1" not in result.columns


# ── encode_categoricals ───────────────────────────────────────────────────────

def test_encode_creates_dummy_columns(sample_df):
    result = encode_categoricals(sample_df, ["state"])
    # Original 'state' column should be dropped, dummy created
    assert "state" not in result.columns


def test_encode_no_op_on_missing_column(sample_df):
    result = encode_categoricals(sample_df, ["nonexistent"])
    # Should not raise and return unchanged
    assert "nonexistent" not in result.columns


# ── engineer_features (end-to-end) ────────────────────────────────────────────

def test_engineer_features_returns_dataframe(sample_df):
    config = {
        "features": {
            "numeric": ["total_events", "fema_property_damage", "NRI_ExpectedLoss"],
            "log_transform": ["fema_property_damage", "NRI_ExpectedLoss"],
            "lag_features": ["total_events"],
            "categorical": ["state"],
        }
    }
    result = engineer_features(sample_df, config)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_engineer_features_no_nulls(sample_df):
    config = {
        "features": {
            "numeric": ["total_events", "fema_property_damage"],
            "log_transform": ["fema_property_damage"],
            "lag_features": ["total_events"],
            "categorical": [],
        }
    }
    result = engineer_features(sample_df, config)
    assert result.isnull().sum().sum() == 0
