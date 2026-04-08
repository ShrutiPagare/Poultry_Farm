"""
Unit tests for preprocessing and model pipeline.
Run:  pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.data_loader import load_raw, clean, encode_categoricals, get_X_y
from configs.config import TARGET_PROFIT, TARGET_RISK


@pytest.fixture(scope="module")
def df_clean():
    return clean(load_raw())


def test_load_raw_shape(df_clean):
    assert len(df_clean) > 100, "Dataset should have >100 rows"
    assert "estimated_profit" in df_clean.columns


def test_no_nulls_after_clean(df_clean):
    numeric_nulls = df_clean.select_dtypes(include="number").isnull().sum().sum()
    assert numeric_nulls == 0, f"Found {numeric_nulls} null numeric values after cleaning"


def test_mortality_rate_bounds(df_clean):
    assert df_clean["mortality_rate"].between(0, 1).all(), "Mortality rate out of bounds [0,1]"


def test_feed_efficiency_positive(df_clean):
    assert (df_clean["feed_efficiency"] > 0).all(), "Feed efficiency must be positive"


def test_risk_labels(df_clean):
    assert set(df_clean["risk_label"].unique()).issubset({"Low", "Medium", "High"})


def test_farm_score_range(df_clean):
    assert df_clean["farm_score"].between(0, 100).all(), "Farm score must be in [0, 100]"


def test_encode_categoricals(df_clean):
    df_enc, encoders = encode_categoricals(df_clean)
    for col in ["farm_size", "contract_type", "sale_type", "state"]:
        assert df_enc[col].dtype in [np.int32, np.int64, int], f"{col} not encoded to int"


def test_get_X_y_shape(df_clean):
    X, y = get_X_y(df_clean, TARGET_PROFIT)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] >= 10


def test_profit_range(df_clean):
    # Extreme losses or gains above 500% revenue are likely errors
    ratio = df_clean["estimated_profit"] / df_clean["total_revenue"]
    assert ratio.between(-1.5, 1.5).mean() > 0.95, "Too many extreme profit ratios"
