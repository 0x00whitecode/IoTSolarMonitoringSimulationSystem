"""PV performance analytics.

Calculates electrical parameters and PV performance metrics:
  - Voltage, current, power, energy
  - Efficiency: eta = P_out / (G * A) * 100
  - Capacity factor: CF = mean(P) / P_stc
  - Energy yield: Y = sum(P * dt) / P_stc
  - Performance ratio: PR = P_actual / P_expected
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def compute_performance_metrics(df: pd.DataFrame, panel_area: float,
                                 nominal_power: float, dt_hours: float) -> Dict[str, float]:
    """Compute summary performance metrics from a simulation dataset."""
    power = df.get("measured_power", df.get("true_power"))
    if power is None:
        raise ValueError("No power column found")

    irradiance = df.get("measured_irradiance", df.get("true_irradiance"))
    if irradiance is None:
        raise ValueError("No irradiance column found")

    power = np.asarray(power, dtype=float)
    irradiance = np.asarray(irradiance, dtype=float)
    # Align lengths
    min_len = min(len(power), len(irradiance))
    power, irradiance = power[:min_len], irradiance[:min_len]
    # Remove NaN pairs-wise
    mask = ~np.isnan(power) & ~np.isnan(irradiance)
    power, irradiance = power[mask], irradiance[mask]

    total_energy = float(np.sum(power * dt_hours))
    mean_power = float(np.mean(power))
    max_power = float(np.max(power))
    capacity_factor = mean_power / nominal_power
    energy_yield = total_energy / nominal_power

    # Efficiency from measured data
    G_safe = np.maximum(irradiance, 1.0)
    efficiency = power / (G_safe * panel_area) * 100.0
    mean_efficiency = float(np.mean(efficiency[~np.isnan(efficiency)]))

    # Performance ratio
    P_expected = nominal_power * (irradiance / 1000.0)
    P_expected_safe = np.where(P_expected > 0.1, P_expected, 0.1)
    pr = power / P_expected_safe
    mean_pr = float(np.mean(pr[~np.isnan(pr)]))

    return {
        "total_energy_wh": total_energy,
        "mean_power_w": mean_power,
        "max_power_w": max_power,
        "mean_efficiency_pct": mean_efficiency,
        "capacity_factor": capacity_factor,
        "energy_yield_wh_wp": energy_yield,
        "performance_ratio": mean_pr,
    }


def compute_time_series_metrics(df: pd.DataFrame, panel_area: float,
                                  dt_hours: float) -> pd.DataFrame:
    """Compute per-timestep efficiency and performance ratio."""
    power = df.get("measured_power", df.get("true_power"))
    irradiance = df.get("measured_irradiance", df.get("true_irradiance"))

    G_safe = np.maximum(np.asarray(irradiance, dtype=float), 1.0)
    power_arr = np.asarray(power, dtype=float)

    efficiency = power_arr / (G_safe * panel_area) * 100.0
    P_expected = 400.0 * (G_safe / 1000.0)
    P_expected_safe = np.where(P_expected > 0.1, P_expected, 0.1)
    pr = power_arr / P_expected_safe

    return pd.DataFrame({
        "timestamp": df["timestamp"] if "timestamp" in df else df.index,
        "efficiency_pct": efficiency,
        "performance_ratio": pr,
    })


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for key variables."""
    cols = ["measured_irradiance", "measured_temperature", "measured_voltage",
            "measured_current", "measured_power"]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr()
