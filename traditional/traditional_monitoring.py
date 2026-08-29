"""Traditional solar monitoring baseline model.

Represents conventional (non-IoT) solar monitoring:
  - Periodic/manual measurements (hourly instead of every 5 minutes)
  - Delayed analysis (not real-time)
  - Manual fault identification (no automated detection)
  - No continuous cloud analytics
  - No real-time dashboard alerts

This baseline provides a quantitative comparison point for the IoT system.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger("smart_solar_iot.traditional")


class TraditionalMonitoring:
    """Simulates traditional periodic/manual solar monitoring."""

    def __init__(self, sampling_interval_minutes: int = 60,
                 measurement_error_multiplier: float = 2.0,
                 analysis_delay_minutes: int = 60,
                 manual_fault_detection: bool = True,
                 seed: int = 42):
        self.sampling_interval = sampling_interval_minutes
        self.error_multiplier = measurement_error_multiplier
        self.analysis_delay = analysis_delay_minutes
        self.manual_fault_detection = manual_fault_detection
        self.rng = np.random.default_rng(seed)

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample to traditional periodic measurements.

        Traditional systems take readings every hour (or longer) rather
        than every 5 minutes. This means missed data between samples.
        """
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Resample to hourly (or configured interval)
            freq = f"{self.sampling_interval}min"
            df = df.set_index("timestamp").resample(freq).first().reset_index()
            logger.info(f"Traditional monitoring: downsampled to {self.sampling_interval}min "
                        f"({len(df)} samples from original)")

        # Add higher measurement noise (less precise equipment)
        for col in ["measured_irradiance", "measured_temperature", "measured_voltage",
                    "measured_current", "measured_power"]:
            if col in df.columns:
                noise = self.rng.normal(0, df[col].std() * 0.01 * self.error_multiplier, len(df))
                df[col] = df[col] + noise
        return df

    def detect_faults_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual fault detection: simple threshold checks, no ML.

        Traditional monitoring relies on human inspection of periodic data.
        This is modelled as basic threshold checks with delayed response.
        """
        anomalies = []
        for idx, row in df.iterrows():
            temp = row.get("measured_temperature", 25)
            power = row.get("measured_power", 0)
            if isinstance(temp, str) or np.isnan(temp):
                continue
            if temp > 75:  # Higher threshold (manual inspection less sensitive)
                anomalies.append({
                    "timestamp": row.get("timestamp", idx),
                    "anomaly_type": "manual_overheating",
                    "severity": "critical",
                    "value": float(temp), "expected_value": 75.0,
                    "detection_delay_min": self.analysis_delay,
                })
            if power < 10 and row.get("measured_irradiance", 0) > 200:
                anomalies.append({
                    "timestamp": row.get("timestamp", idx),
                    "anomaly_type": "manual_low_power",
                    "severity": "warning",
                    "value": float(power), "expected_value": 50.0,
                    "detection_delay_min": self.analysis_delay,
                })
        return pd.DataFrame(anomalies)

    def estimate_energy(self, df: pd.DataFrame, dt_hours: float) -> float:
        """Estimate total energy from periodic samples.

        Traditional systems estimate energy from sparse readings,
        leading to higher error than continuous IoT monitoring.
        """
        power = df.get("measured_power", pd.Series(dtype=float))
        if len(power) == 0:
            return 0.0
        # Linear interpolation between samples (coarser = more error)
        return float(np.sum(power * dt_hours))

    def compute_metrics(self, true_df: pd.DataFrame, traditional_df: pd.DataFrame,
                        dt_hours_traditional: float, true_energy: float) -> Dict[str, float]:
        """Compute traditional monitoring performance metrics."""
        # Data availability: fewer samples = lower availability
        data_availability = len(traditional_df) / max(len(true_df), 1) * 100

        # Energy estimation error
        trad_energy = self.estimate_energy(traditional_df, dt_hours_traditional)
        energy_error_pct = abs(trad_energy - true_energy) / max(true_energy, 0.01) * 100

        # Fault detection: manual misses many faults
        true_anomalies = 0
        for _, row in true_df.iterrows():
            temp = row.get("measured_temperature", 25)
            if isinstance(temp, str) or np.isnan(temp):
                continue
            if temp > 70:
                true_anomalies += 1

        trad_anomalies = len(self.detect_faults_manual(traditional_df))
        detection_rate = trad_anomalies / max(true_anomalies, 1) * 100

        # Response time: traditional has built-in delay
        response_time = self.analysis_delay * 60  # seconds

        return {
            "data_availability_pct": data_availability,
            "energy_estimation_error_pct": energy_error_pct,
            "fault_detection_rate_pct": detection_rate,
            "response_time_s": response_time,
            "sampling_interval_min": self.sampling_interval,
            "total_samples": len(traditional_df),
        }
