"""
Data loading, cleaning, and feature engineering pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from configs.config import RAW_CSV, ALL_FEATURES, CATEGORICAL_FEATURES


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # Ensure categoricals are strings
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remove duplicates
    df.drop_duplicates(subset="farm_id", inplace=True)
    # Clip extreme values
    df["mortality_rate"]   = df["mortality_rate"].clip(0, 1)
    df["feed_efficiency"]  = df["feed_efficiency"].clip(1, 10)
    # Fill any nulls
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def encode_categoricals(df: pd.DataFrame):
    """Label-encode categorical columns; return df + encoder dict."""
    encoders = {}
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def get_X_y(df: pd.DataFrame, target: str):
    df_enc, _ = encode_categoricals(df)
    X = df_enc[ALL_FEATURES]
    y = df_enc[target]
    return X, y


def get_time_series_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily records into monthly profit time-series."""
    df = df.copy()
    # Simulate a contract_date from index for ordering
    np.random.seed(42)
    start = pd.Timestamp("2022-01-01")
    df["contract_date"] = pd.date_range(start=start, periods=len(df), freq="D")
    monthly = (
        df.set_index("contract_date")["estimated_profit"]
        .resample("ME")
        .sum()
        .reset_index()
    )
    monthly.columns = ["month", "total_profit"]
    return monthly
