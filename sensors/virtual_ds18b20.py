"""Virtual DS18B20 temperature sensor.

Simulates the DS18B20 1-Wire digital temperature sensor.
Range: -55°C to +125°C, ±0.5°C accuracy.

Measurement model:
  X_measured = X_true + epsilon,  epsilon ~ N(0, sigma^2)

No physical hardware is used; this is a software simulation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class DS18B20Reading:
    sensor_id: str
    timestamp: str
    temperature: float
    status: str = "ok"


class VirtualDS18B20:
    """Simulated DS18B20 temperature sensor."""

    SENSOR_ID = "DS18B20_001"

    def __init__(self, temperature_noise: float = 0.5,
                 calibration_factor: float = 1.0, seed: int = 42):
        self.temperature_noise = temperature_noise
        self.calibration_factor = calibration_factor
        self.rng = np.random.default_rng(seed)

    def measure(self, true_temperature: np.ndarray) -> np.ndarray:
        """Measure temperature with Gaussian noise."""
        noise = self.rng.normal(0, self.temperature_noise, len(true_temperature))
        measured = (true_temperature + noise) * self.calibration_factor
        return np.clip(measured, -55.0, 125.0)

    def validate_reading(self, temperature: float) -> str:
        """Validate reading against physical limits."""
        if temperature < -55 or temperature > 125:
            return "temperature_out_of_range"
        return "ok"
