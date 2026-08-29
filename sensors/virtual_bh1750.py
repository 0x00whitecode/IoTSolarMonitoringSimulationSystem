"""Virtual BH1750 ambient light sensor.

Simulates the BH1750 I2C light sensor for irradiance measurement.
The BH1750 measures illuminance (lux), which we convert to irradiance (W/m^2)
using the standard approximation: 1 W/m^2 ~ 126.7 lux (for solar spectrum).

Measurement model:
  X_measured = X_true + epsilon,  epsilon ~ N(0, sigma^2)

No physical hardware is used; this is a software simulation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class BH1750Reading:
    sensor_id: str
    timestamp: str
    irradiance: float
    raw_lux: float
    status: str = "ok"


class VirtualBH1750:
    """Simulated BH1750 light/irradiance sensor."""

    SENSOR_ID = "BH1750_001"
    LUX_TO_IRRADIANCE = 1.0 / 126.7  # W/m^2 per lux

    def __init__(self, irradiance_noise: float = 15.0,
                 calibration_factor: float = 1.0, seed: int = 42):
        self.irradiance_noise = irradiance_noise
        self.calibration_factor = calibration_factor
        self.rng = np.random.default_rng(seed)

    def measure(self, true_irradiance: np.ndarray) -> np.ndarray:
        """Measure irradiance with Gaussian noise."""
        noise = self.rng.normal(0, self.irradiance_noise, len(true_irradiance))
        measured = (true_irradiance + noise) * self.calibration_factor
        return np.maximum(measured, 0.0)

    def to_lux(self, irradiance: np.ndarray) -> np.ndarray:
        """Convert irradiance (W/m^2) to lux."""
        return irradiance / self.LUX_TO_IRRADIANCE

    def validate_reading(self, irradiance: float) -> str:
        """Validate reading against physical limits."""
        if irradiance < 0:
            return "negative_irradiance"
        if irradiance > 1400:
            return "irradiance_exceeds_physical_max"
        return "ok"
