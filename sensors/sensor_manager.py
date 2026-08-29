"""Sensor manager: coordinates virtual sensors and calibration.

Provides a unified interface for reading all virtual sensors and
computing calibration metrics (MAE, RMSE, bias, percentage error).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sensors.virtual_pzem017 import VirtualPZEM017
from sensors.virtual_bh1750 import VirtualBH1750
from sensors.virtual_ds18b20 import VirtualDS18B20


class SensorManager:
    """Manages all virtual sensors and provides calibration analysis."""

    def __init__(self, pzem: VirtualPZEM017, bh1750: VirtualBH1750,
                 ds18b20: VirtualDS18B20):
        self.pzem = pzem
        self.bh1750 = bh1750
        self.ds18b20 = ds18b20

    @staticmethod
    def calibration_metrics(true: np.ndarray, measured: np.ndarray) -> Dict[str, float]:
        """Compute sensor calibration metrics.

        MAE  = mean(|true - measured|)
        RMSE = sqrt(mean((true - measured)^2))
        bias = mean(true - measured)
        MAPE = mean(|true - measured| / |true|) * 100
        """
        true = np.asarray(true, dtype=float)
        measured = np.asarray(measured, dtype=float)
        mask = ~np.isnan(measured)
        true, measured = true[mask], measured[mask]
        if len(true) == 0:
            return {"mae": 0, "rmse": 0, "bias": 0, "mape": 0}
        error = measured - true
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        bias = float(np.mean(error))
        nonzero = np.abs(true) > 0.01
        mape = float(np.mean(np.abs(error[nonzero]) / np.abs(true[nonzero])) * 100) if nonzero.any() else 0.0
        return {"mae": mae, "rmse": rmse, "bias": bias, "mape": mape}

    def calibrate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute calibration metrics for all sensors.

        df must contain both true_* and measured_* columns.
        Returns a DataFrame with one row per sensor.
        """
        results = []
        for sensor, true_col, meas_col, unit in [
            ("BH1750 (irradiance)", "true_irradiance", "measured_irradiance", "W/m^2"),
            ("DS18B20 (temperature)", "true_panel_temp", "measured_temperature", "°C"),
            ("PZEM-017 (voltage)", "true_voltage", "measured_voltage", "V"),
            ("PZEM-017 (current)", "true_current", "measured_current", "A"),
            ("PZEM-017 (power)", "true_power", "measured_power", "W"),
        ]:
            m = self.calibration_metrics(df[true_col].values, df[meas_col].values)
            results.append({
                "sensor": sensor,
                "unit": unit,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "mape": m["mape"],
            })
        return pd.DataFrame(results)
