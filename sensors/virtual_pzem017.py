"""Virtual PZEM-017 sensor (DC energy meter).

Simulates the PZEM-017 Modbus DC energy meter that measures:
  - DC voltage (0-300V)
  - DC current (0-100A)
  - Power (computed as V*I)
  - Energy (accumulated)

Measurement model:
  X_measured = X_true + epsilon,  epsilon ~ N(0, sigma^2)

The PZEM-017 is a real Modbus RTU device. This is a software simulation
of its measurement behavior — no physical hardware is used.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PZEMReading:
    sensor_id: str
    timestamp: str
    voltage: float
    current: float
    power: float
    energy: float
    status: str = "ok"


class VirtualPZEM017:
    """Simulated PZEM-017 DC energy meter."""

    SENSOR_ID = "PZEM017_001"

    def __init__(self, voltage_noise: float = 0.2, current_noise: float = 0.05,
                 power_noise: float = 2.0, calibration_factor: float = 1.0,
                 seed: int = 42):
        self.voltage_noise = voltage_noise
        self.current_noise = current_noise
        self.power_noise = power_noise
        self.calibration_factor = calibration_factor
        self.rng = np.random.default_rng(seed)
        self._energy_accum = 0.0

    def measure_voltage(self, true_voltage: np.ndarray) -> np.ndarray:
        """Measure DC voltage with Gaussian noise."""
        noise = self.rng.normal(0, self.voltage_noise, len(true_voltage))
        measured = (true_voltage + noise) * self.calibration_factor
        return np.clip(measured, 0, 300)

    def measure_current(self, true_current: np.ndarray) -> np.ndarray:
        """Measure DC current with Gaussian noise."""
        noise = self.rng.normal(0, self.current_noise, len(true_current))
        measured = (true_current + noise) * self.calibration_factor
        return np.clip(measured, 0, 100)

    def measure_power(self, true_power: np.ndarray) -> np.ndarray:
        """Measure DC power with Gaussian noise."""
        noise = self.rng.normal(0, self.power_noise, len(true_power))
        measured = (true_power + noise) * self.calibration_factor
        return np.maximum(measured, 0.0)

    def compute_energy(self, power: np.ndarray, dt_hours: float) -> np.ndarray:
        """Accumulate energy [Wh] from power readings."""
        return np.cumsum(power * dt_hours)

    def validate_reading(self, voltage: float, current: float) -> str:
        """Validate a reading against physical limits."""
        if voltage < 0 or voltage > 300:
            return "voltage_out_of_range"
        if current < 0 or current > 100:
            return "current_out_of_range"
        return "ok"
