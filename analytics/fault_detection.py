"""Fault detection and classification.

Combines rule-based anomaly detection with scenario-specific fault
identification to classify faults into:
  - Normal
  - Dust accumulation
  - Partial shading
  - Overheating
  - Sensor fault
  - Communication fault

Each fault type has a clear mathematical/simulation basis (see scenarios.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger("smart_solar_iot.fault_detection")


def classify_fault(row: pd.Series, expected_power_col: str = "true_power") -> str:
    """Classify a single reading into a fault category."""
    temp = row.get("measured_temperature", 25)
    volt = row.get("measured_voltage", 0)
    meas_p = row.get("measured_power", 0)
    exp_p = row.get(expected_power_col, 0)
    irr = row.get("measured_irradiance", 0)
    dust = row.get("dust_factor", 1.0)
    shade = row.get("shading_factor", 0.0)

    if isinstance(temp, str) or np.isnan(temp):
        return "sensor_fault"
    if isinstance(volt, str) or np.isnan(volt):
        return "sensor_fault"
    if isinstance(irr, str) or np.isnan(irr):
        return "sensor_fault"

    # Overheating
    if temp > 70:
        return "overheating"

    # Sensor fault: stuck or impossible values
    if irr < 0 or irr > 1400:
        return "sensor_fault"
    if volt < 0 or volt > 300:
        return "sensor_fault"

    # Partial shading
    if shade > 0.1:
        return "partial_shading"

    # Dust accumulation
    if dust < 0.95:
        return "dust"

    # Power mismatch (general fault)
    if exp_p > 10 and not np.isnan(meas_p):
        ratio = meas_p / exp_p
        if ratio < 0.5:
            return "sensor_fault"  # likely sensor issue if power is way off

    return "normal"


def detect_faults(df: pd.DataFrame) -> pd.DataFrame:
    """Classify all readings and return a fault summary."""
    df = df.copy()
    df["fault_type"] = df.apply(lambda r: classify_fault(r), axis=1)
    return df


def fault_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return a count of each fault type."""
    if "fault_type" not in df.columns:
        df = detect_faults(df)
    return df["fault_type"].value_counts().to_dict()


def prepare_classification_dataset(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
    """Prepare a labelled dataset for fault classification ML.

    The label comes from the controlled simulation scenario, not from
    manual annotation. This is a scientifically defensible approach because
    the simulation injects known fault conditions.
    """
    df = df.copy()
    df["fault_label"] = scenario_label
    return df
