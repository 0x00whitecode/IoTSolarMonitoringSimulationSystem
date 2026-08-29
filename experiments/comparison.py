"""IoT vs traditional monitoring comparison.

Quantitative comparison across:
  - Monitoring accuracy (MAPE)
  - Data availability (%)
  - Communication (latency, packet loss, delivery rate)
  - Fault detection (accuracy, detection time, false alarms)
  - Energy analysis (power/energy estimation error)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict

from analytics.efficiency import (compute_monitoring_accuracy,
                                     compute_power_estimation_error,
                                     compute_energy_estimation_error)
from analytics.statistics import percentage_improvement

logger = logging.getLogger("smart_solar_iot.comparison")


def compare_iot_vs_traditional(iot_df: pd.DataFrame, traditional_df: pd.DataFrame,
                                true_df: pd.DataFrame,
                                iot_stats: dict, trad_metrics: dict,
                                dt_hours_iot: float, dt_hours_trad: float) -> Dict:
    """Full quantitative comparison of IoT vs traditional monitoring."""
    # Monitoring accuracy
    iot_accuracy = compute_monitoring_accuracy(true_df, iot_df)
    trad_accuracy = compute_monitoring_accuracy(true_df, traditional_df)

    # Power estimation error
    true_power = true_df.get("true_power", true_df.get("measured_power", pd.Series(dtype=float))).values
    iot_power = iot_df.get("measured_power", pd.Series(dtype=float)).values
    trad_power = traditional_df.get("measured_power", pd.Series(dtype=float)).values

    iot_power_err = compute_power_estimation_error(true_power[:len(iot_power)], iot_power)
    trad_power_err = compute_power_estimation_error(true_power[:len(trad_power)], trad_power)

    # Energy estimation error
    true_energy = true_df.get("true_energy", pd.Series(dtype=float)).values
    iot_energy = iot_df.get("measured_energy", pd.Series(dtype=float)).values
    trad_energy_vals = traditional_df.get("measured_energy", pd.Series(dtype=float)).values

    iot_energy_err = compute_energy_estimation_error(true_energy[:len(iot_energy)], iot_energy)
    trad_energy_err = compute_energy_estimation_error(true_energy[:len(trad_energy_vals)], trad_energy_vals)

    # Communication
    iot_comm = {
        "latency_ms": iot_stats.get("avg_latency_ms", 0),
        "packet_loss_pct": iot_stats.get("packet_loss", 0),
        "delivery_rate_pct": iot_stats.get("delivery_rate", 0),
    }
    trad_comm = {
        "latency_ms": trad_metrics.get("response_time_s", 0) * 1000,  # traditional has no real-time comm
        "packet_loss_pct": 0,  # no continuous transmission
        "delivery_rate_pct": 100,  # manual readings always recorded
    }

    # Fault detection
    iot_fault = {
        "detection_rate_pct": iot_stats.get("fault_detection_rate", 0),
        "response_time_s": iot_stats.get("response_time", 0),
        "false_alarms": iot_stats.get("false_alarms", 0),
    }
    trad_fault = {
        "detection_rate_pct": trad_metrics.get("fault_detection_rate_pct", 0),
        "response_time_s": trad_metrics.get("response_time_s", 0),
        "false_alarms": 0,
    }

    # Build comparison table
    comparison = {
        "monitoring_accuracy": {
            "iot_mape_power": iot_accuracy.get("mape_power", 0),
            "traditional_mape_power": trad_accuracy.get("mape_power", 0),
            "improvement_pct": percentage_improvement(
                iot_accuracy.get("mape_power", 0),
                trad_accuracy.get("mape_power", 0)),
        },
        "data_availability": {
            "iot_pct": iot_accuracy.get("data_availability_pct", 0),
            "traditional_pct": trad_accuracy.get("data_availability_pct", 0),
            "improvement_pct": percentage_improvement(
                trad_accuracy.get("data_availability_pct", 0),  # higher is better, so reversed
                iot_accuracy.get("data_availability_pct", 0)),
        },
        "communication": {
            "iot_latency_ms": iot_comm["latency_ms"],
            "traditional_latency_ms": trad_comm["latency_ms"],
            "iot_packet_loss_pct": iot_comm["packet_loss_pct"],
            "iot_delivery_rate_pct": iot_comm["delivery_rate_pct"],
        },
        "fault_detection": {
            "iot_detection_rate_pct": iot_fault["detection_rate_pct"],
            "traditional_detection_rate_pct": trad_fault["detection_rate_pct"],
            "iot_response_time_s": iot_fault["response_time_s"],
            "traditional_response_time_s": trad_fault["response_time_s"],
        },
        "energy_analysis": {
            "iot_power_mae": iot_power_err["mae"],
            "traditional_power_mae": trad_power_err["mae"],
            "iot_energy_error_pct": iot_energy_err["percentage_error"],
            "traditional_energy_error_pct": trad_energy_err["percentage_error"],
            "power_error_improvement_pct": percentage_improvement(
                iot_power_err["mae"], trad_power_err["mae"]),
            "energy_error_improvement_pct": percentage_improvement(
                iot_energy_err["percentage_error"], trad_energy_err["percentage_error"]),
        },
    }

    return comparison


def comparison_to_dataframe(comparison: Dict) -> pd.DataFrame:
    """Flatten comparison dict into a table for CSV export."""
    rows = []
    for category, metrics in comparison.items():
        for metric, value in metrics.items():
            rows.append({"category": category, "metric": metric, "value": value})
    return pd.DataFrame(rows)
