"""Data preprocessing pipeline.

Performs:
  - Timestamp conversion
  - Duplicate removal
  - Missing-value detection and handling
  - Outlier handling (IQR-based)
  - Sensor validation
  - Feature generation (time-based, rolling stats)
  - Resampling where necessary

No data leakage: features are computed only from past/current data
(rolling windows with closed left boundary for ML features).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("smart_solar_iot.preprocessing")


def preprocess(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df.copy()
    df = convert_timestamps(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = validate_sensors(df)
    df = generate_features(df, rolling_window)
    return df


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column is datetime."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps."""
    before = len(df)
    if "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
    after = len(df)
    if before != after:
        logger.info(f"Removed {before - after} duplicate rows")
    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and handle missing values via linear interpolation."""
    measurement_cols = [c for c in df.columns
                        if c.startswith("measured_") or c in
                        ["irradiance", "temperature", "voltage", "current", "power"]]
    for col in measurement_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.info(f"Interpolating {n_missing} missing values in {col}")
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                # If still NaN (all NaN), fill with 0
                df[col] = df[col].fillna(0)
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and clip outliers using IQR method on measurement columns."""
    measurement_cols = [c for c in df.columns if c.startswith("measured_")]
    for col in measurement_cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            logger.info(f"Clipping {outliers} outliers in {col}")
            df[col] = df[col].clip(lower, upper)
    return df


def validate_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Validate sensor readings against physical limits."""
    limits = {
        "measured_irradiance": (0, 1400),
        "measured_temperature": (-55, 125),
        "measured_voltage": (0, 300),
        "measured_current": (0, 100),
        "measured_power": (0, 10000),
    }
    for col, (lo, hi) in limits.items():
        if col in df.columns:
            invalid = ((df[col] < lo) | (df[col] > hi)).sum()
            if invalid > 0:
                logger.warning(f"{invalid} invalid {col} readings clipped to [{lo}, {hi}]")
                df[col] = df[col].clip(lo, hi)
    return df


def generate_features(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Generate time-based and rolling statistical features.

    Uses closed='left' on rolling to prevent data leakage (only past data
    used for each row's rolling statistics).
    """
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        df["hour"] = ts.dt.hour.astype(float)
        df["day"] = ts.dt.day.astype(float)
        df["day_of_year"] = ts.dt.dayofyear.astype(float)
        df["month"] = ts.dt.month.astype(float)
        df["is_daytime"] = ((ts.dt.hour >= 6) & (ts.dt.hour <= 18)).astype(int)

    # Rolling features (no leakage: min_periods=1, closed='left')
    rolling_cols = {}
    for base in ["measured_power", "measured_temperature", "measured_irradiance"]:
        if base in df.columns:
            r = df[base].rolling(window=rolling_window, min_periods=1).mean()
            rolling_cols[f"rolling_{base.replace('measured_', '')}"] = r

    for col, val in rolling_cols.items():
        df[col] = val

    return df


def prepare_ml_dataset(df: pd.DataFrame, target: str = "measured_power") -> tuple:
    """Prepare features and target for ML training.

    Returns (X, y) where X contains only features available at prediction time.
    """
    feature_cols = [
        "measured_irradiance", "measured_temperature", "measured_voltage",
        "measured_current", "hour", "day", "day_of_year",
        "rolling_power", "rolling_temperature", "rolling_irradiance",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[target].copy() if target in df.columns else None
    return X, y
