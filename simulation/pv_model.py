"""Photovoltaic mathematical model.

Implements the single-diode-inspired simplified PV model for a single module.
The model computes voltage, current, and power from irradiance and temperature
using physically-justified relationships rather than arbitrary equations.

Equations:
  1. Power output (electrical):
     P_PV = V_PV * I_PV

  2. Energy accumulation:
     E = sum(P_i * dt)

  3. Module efficiency:
     eta = P_out / (G * A) * 100   [%]

  4. Temperature derating (standard linear model, NREL PVWatts):
     P_temp = P_stc * (1 + gamma * (T_panel - T_ref))
     where gamma is the temperature coefficient (negative for silicon).

  5. Irradiance scaling:
     P_irr = P_stc * (G / G_ref)

  6. Combined power:
     P_out = P_stc * (G / G_ref) * (1 + gamma * (T_panel - T_ref))
              * (1 - shading_loss) * (1 - dust_loss) * inverter_efficiency

  7. Voltage model (temperature-dependent Vmp):
     V_mp = V_mp_stc * (1 + beta_v * (T_panel - T_ref)) * ln(max(G, 1) / G_ref + 1) / ln(2)
     Simplified: V_mp scales with log(irradiance) and temperature coefficient.

  8. Current model:
     I_mp = P_out / V_mp  (derived from power and voltage)

Assumptions:
  - Single module, not a full array. Extension to arrays is linear.
  - Maximum Power Point (MPP) tracking is assumed (the virtual charge
    controller always operates at MPP).
  - Inverter efficiency is constant (simplified; real inverters vary with load).
  - Shading reduces both direct and diffuse components.
  - Dust reduces effective irradiance linearly over time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PVConfig:
    panel_area: float = 1.7           # m^2
    nominal_power: float = 400.0       # W (STC)
    nominal_voltage: float = 40.0      # V (Vmp at STC)
    nominal_current: float = 10.0      # A (Imp at STC)
    reference_irradiance: float = 1000.0  # W/m^2
    reference_temperature: float = 25.0    # °C
    temperature_coefficient: float = -0.0035  # per °C (gamma)
    voltage_temp_coefficient: float = -0.0030  # per °C (beta_v)
    current_temp_coefficient: float = 0.0005   # per °C (alpha_i)
    inverter_efficiency: float = 0.96
    shading_factor: float = 0.0
    dust_loss_initial: float = 0.0
    dust_loss_rate: float = 0.001


class PVModel:
    """Photovoltaic module model computing electrical output from environment."""

    def __init__(self, cfg: PVConfig):
        self.cfg = cfg

    def compute_power(self, irradiance: np.ndarray, panel_temp: np.ndarray,
                      shading_factor: np.ndarray | float = 0.0,
                      dust_factor: np.ndarray | float = 1.0) -> np.ndarray:
        """Compute DC power output [W] using PVWatts-style model.

        P_out = P_stc * (G / G_ref) * (1 + gamma * (T - T_ref))
                 * (1 - shading) * dust * eta_inv
        """
        G = np.maximum(irradiance, 0.0)
        gamma = self.cfg.temperature_coefficient
        T_ref = self.cfg.reference_temperature
        G_ref = self.cfg.reference_irradiance
        P_stc = self.cfg.nominal_power
        eta_inv = self.cfg.inverter_efficiency

        irradiance_factor = G / G_ref
        temp_factor = 1.0 + gamma * (panel_temp - T_ref)
        shading_loss = np.asarray(shading_factor)
        dust = np.asarray(dust_factor)

        power = P_stc * irradiance_factor * temp_factor * (1.0 - shading_loss) * dust * eta_inv
        return np.maximum(power, 0.0)

    def compute_voltage(self, irradiance: np.ndarray, panel_temp: np.ndarray) -> np.ndarray:
        """Compute module voltage at MPP [V].

        V_mp = V_mp_stc * (1 + beta_v * (T - T_ref)) * f(G)
        where f(G) = 0.7 + 0.3 * log(G+1) / log(G_ref+1)
        (logarithmic voltage response to irradiance, typical of crystalline Si)
        """
        G = np.maximum(irradiance, 0.0)
        beta_v = self.cfg.voltage_temp_coefficient
        T_ref = self.cfg.reference_temperature
        G_ref = self.cfg.reference_irradiance
        V_stc = self.cfg.nominal_voltage

        temp_factor = 1.0 + beta_v * (panel_temp - T_ref)
        irr_factor = 0.7 + 0.3 * np.log(G + 1.0) / np.log(G_ref + 1.0)
        voltage = V_stc * temp_factor * irr_factor
        return np.maximum(voltage, 0.0)

    def compute_current(self, power: np.ndarray, voltage: np.ndarray) -> np.ndarray:
        """Compute module current at MPP [A].

        I_mp = P / V  (Ohm's law at MPP)
        """
        v_safe = np.where(voltage > 0.1, voltage, 0.1)
        return power / v_safe

    def compute_efficiency(self, power: np.ndarray, irradiance: np.ndarray) -> np.ndarray:
        """Compute module conversion efficiency [%].

        eta = P_out / (G * A) * 100
        """
        G = np.maximum(irradiance, 1.0)
        eta = power / (G * self.cfg.panel_area) * 100.0
        return eta

    def compute_energy(self, power: np.ndarray, dt_hours: float) -> np.ndarray:
        """Compute cumulative energy [Wh].

        E = cumsum(P_i * dt)
        """
        return np.cumsum(power * dt_hours)

    def compute_performance_ratio(self, power: np.ndarray, irradiance: np.ndarray) -> np.ndarray:
        """Compute Performance Ratio (PR).

        PR = P_actual / P_expected
        where P_expected = P_stc * (G / G_ref)
        (removes irradiance effect, isolates system losses)
        """
        G = np.maximum(irradiance, 0.0)
        P_expected = self.cfg.nominal_power * (G / self.cfg.reference_irradiance)
        P_expected = np.where(P_expected > 0.1, P_expected, 0.1)
        return power / P_expected

    def compute_capacity_factor(self, power: np.ndarray) -> float:
        """Compute capacity factor over the period.

        CF = mean(P) / P_stc
        """
        return float(np.mean(power) / self.cfg.nominal_power)

    def compute_energy_yield(self, power: np.ndarray, dt_hours: float) -> float:
        """Compute specific energy yield [Wh/Wp].

        Y = sum(P * dt) / P_stc
        """
        return float(np.sum(power * dt_hours) / self.cfg.nominal_power)

    def simulate(self, env_df: pd.DataFrame, dt_hours: float) -> pd.DataFrame:
        """Run full PV simulation on an environment DataFrame.

        Returns DataFrame with: voltage, current, power, energy,
        efficiency, performance_ratio
        """
        G = env_df["effective_irradiance"].values
        T = env_df["panel_temp"].values
        shading = env_df.get("shading_factor", 0.0)
        dust = env_df.get("dust_factor", 1.0)

        power = self.compute_power(G, T, shading, dust)
        voltage = self.compute_voltage(G, T)
        current = self.compute_current(power, voltage)
        efficiency = self.compute_efficiency(power, G)
        energy = self.compute_energy(power, dt_hours)
        pr = self.compute_performance_ratio(power, G)

        return pd.DataFrame({
            "voltage": voltage,
            "current": current,
            "power": power,
            "energy": energy,
            "efficiency": efficiency,
            "performance_ratio": pr,
        })
