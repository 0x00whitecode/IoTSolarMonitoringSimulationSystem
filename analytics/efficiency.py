"""Efficiency analysis comparing IoT vs traditional monitoring.

Computes:
  - Power estimation error
  - Energy estimation error
  - Data availability
  - Measurement accuracy
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def compute_monitoring_accuracy(true_df: pd.DataFrame, measured_df: pd.DataFrame) -> Dict[str, float]:
    """Compute monitoring accuracy metrics.

    MAPE for each measured variable, and overall data availability.
    """
    metrics = {}
    var_map = {
        "irradiance": ("true_irradiance", "measured_irradiance"),
        "temperature": ("true_panel_temp", "measured_temperature"),
        "voltage": ("true_voltage", "measured_voltage"),
        "current": ("true_current", "measured_current"),
        "power": ("true_power", "measured_power"),
    }
    for name, (true_col, meas_col) in var_map.items():
        if true_col not in true_df.columns or meas_col not in measured_df.columns:
            continue
        true = true_df[true_col].values
        meas = measured_df[meas_col].values
        # Align lengths: compare only overlapping rows
        min_len = min(len(true), len(meas))
        true, meas = true[:min_len], meas[:min_len]
        mask = ~np.isnan(meas) & ~np.isnan(true) & (np.abs(true) > 0.01)
        if mask.sum() == 0:
            metrics[f"mape_{name}"] = 0.0
            continue
        mape = np.mean(np.abs((meas[mask] - true[mask]) / true[mask])) * 100
        metrics[f"mape_{name}"] = float(mape)

    # Data availability
    total = len(measured_df)
    non_null = measured_df["measured_power"].notna().sum() if "measured_power" in measured_df else total
    metrics["data_availability_pct"] = float(non_null / max(total, 1) * 100)
    return metrics


def compute_power_estimation_error(true_power: np.ndarray,
                                     estimated_power: np.ndarray) -> Dict[str, float]:
    """Compute error between true and estimated power."""
    mask = ~np.isnan(estimated_power) & ~np.isnan(true_power)
    tp = true_power[mask]
    ep = estimated_power[mask]
    if len(tp) == 0:
        return {"mae": 0, "rmse": 0, "mape": 0}
    mae = float(np.mean(np.abs(ep - tp)))
    rmse = float(np.sqrt(np.mean((ep - tp) ** 2)))
    nonzero = np.abs(tp) > 0.01
    mape = float(np.mean(np.abs((ep[nonzero] - tp[nonzero]) / tp[nonzero])) * 100) if nonzero.any() else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compute_energy_estimation_error(true_energy: np.ndarray,
                                      estimated_energy: np.ndarray) -> Dict[str, float]:
    """Compute error between true and estimated cumulative energy."""
    if len(true_energy) == 0:
        return {"mae": 0, "rmse": 0, "percentage_error": 0}
    mae = float(np.mean(np.abs(estimated_energy - true_energy)))
    rmse = float(np.sqrt(np.mean((estimated_energy - true_energy) ** 2)))
    true_final = true_energy[-1] if true_energy[-1] > 0.01 else 0.01
    pct_err = float(abs(estimated_energy[-1] - true_energy[-1]) / true_final * 100)
    return {"mae": mae, "rmse": rmse, "percentage_error": pct_err}
