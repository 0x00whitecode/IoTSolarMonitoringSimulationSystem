"""Anomaly detection module.

Two approaches:
  1. Rule-based: physical thresholds and expected-power comparison
  2. ML-based: Isolation Forest

Rule-based rules:
  - temperature > threshold -> overheating
  - voltage outside normal range -> voltage anomaly
  - measured_power << expected_power -> possible PV fault
  - sensor value physically impossible -> sensor anomaly
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("smart_solar_iot.anomaly_detection")

# Physical thresholds
TEMP_THRESHOLD = 70.0       # °C - panel overheating
VOLTAGE_LOW = 5.0            # V
VOLTAGE_HIGH = 60.0          # V
POWER_RATIO_THRESHOLD = 0.5  # measured/expected below this -> fault


def detect_anomalies_rule_based(df: pd.DataFrame,
                                 expected_power_col: str = "true_power",
                                 measured_power_col: str = "measured_power",
                                 temp_col: str = "measured_temperature",
                                 voltage_col: str = "measured_voltage",
                                 irradiance_col: str = "measured_irradiance") -> pd.DataFrame:
    """Rule-based anomaly detection. Returns DataFrame of anomalies."""
    anomalies = []

    for idx, row in df.iterrows():
        ts = row.get("timestamp", idx)
        temp = row.get(temp_col, 25)
        volt = row.get(voltage_col, 0)
        meas_p = row.get(measured_power_col, 0)
        exp_p = row.get(expected_power_col, 0)
        irr = row.get(irradiance_col, 0)

        if np.isnan(temp) or np.isnan(volt) or np.isnan(meas_p):
            anomalies.append({
                "timestamp": ts, "anomaly_type": "missing_data",
                "severity": "warning", "value": 0, "expected_value": 0,
            })
            continue

        # Temperature check
        if temp > TEMP_THRESHOLD:
            anomalies.append({
                "timestamp": ts, "anomaly_type": "overheating",
                "severity": "critical", "value": float(temp),
                "expected_value": TEMP_THRESHOLD,
            })

        # Voltage range check
        if volt < VOLTAGE_LOW or volt > VOLTAGE_HIGH:
            anomalies.append({
                "timestamp": ts, "anomaly_type": "voltage_anomaly",
                "severity": "critical", "value": float(volt),
                "expected_value": float(VOLTAGE_LOW if volt < VOLTAGE_LOW else VOLTAGE_HIGH),
            })

        # Power vs expected (PV fault)
        if exp_p > 10 and meas_p < exp_p * POWER_RATIO_THRESHOLD:
            anomalies.append({
                "timestamp": ts, "anomaly_type": "possible_pv_fault",
                "severity": "critical", "value": float(meas_p),
                "expected_value": float(exp_p),
            })

        # Physical impossibility
        if irr < 0 or irr > 1400:
            anomalies.append({
                "timestamp": ts, "anomaly_type": "sensor_anomaly",
                "severity": "critical", "value": float(irr),
                "expected_value": 0,
            })

    return pd.DataFrame(anomalies)


def detect_anomalies_isolation_forest(df: pd.DataFrame,
                                       contamination: float = 0.05,
                                       seed: int = 42) -> pd.DataFrame:
    """ML-based anomaly detection using Isolation Forest."""
    feature_cols = ["measured_irradiance", "measured_temperature",
                    "measured_voltage", "measured_current", "measured_power"]
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()

    X = df[available].copy()
    # Fill NaN with column mean
    for col in available:
        X[col] = X[col].fillna(X[col].mean())

    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=100)
    labels = iso.fit_predict(X)
    # -1 = anomaly, 1 = normal
    anomalies = df[labels == -1].copy()
    anomalies["anomaly_type"] = "isolation_forest"
    anomalies["severity"] = "warning"
    anomalies["value"] = anomalies.get("measured_power", 0)
    anomalies["expected_value"] = df["measured_power"].median() if "measured_power" in df else 0
    return anomalies[["timestamp", "anomaly_type", "severity", "value", "expected_value"]] \
        if "timestamp" in anomalies else anomalies


def compare_methods(df: pd.DataFrame) -> Dict[str, int]:
    """Compare rule-based and ML-based anomaly detection counts."""
    rb = detect_anomalies_rule_based(df)
    ml = detect_anomalies_isolation_forest(df)
    return {
        "rule_based_count": len(rb),
        "isolation_forest_count": len(ml),
        "rule_based_types": rb["anomaly_type"].value_counts().to_dict() if len(rb) > 0 else {},
    }
