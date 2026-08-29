"""Main simulator orchestrating environment + PV + sensors + IoT + storage.

Usage:
    python -m simulation.simulator
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from simulation.solar_environment import SolarEnvironment
from simulation.pv_model import PVModel
from simulation.scenarios import get_scenario, all_scenario_names, Scenario
from sensors.sensor_manager import SensorManager
from sensors.virtual_pzem017 import VirtualPZEM017
from sensors.virtual_bh1750 import VirtualBH1750
from sensors.virtual_ds18b20 import VirtualDS18B20


class SolarSimulator:
    """Orchestrates the full simulation pipeline for a single scenario."""

    def __init__(self, scenario: Scenario, sensor_noise_cfg: dict):
        self.scenario = scenario
        self.env = SolarEnvironment(scenario.env_config)
        self.pv = PVModel(scenario.pv_config)
        self.sensor_noise_cfg = sensor_noise_cfg
        self.dt_hours = scenario.env_config.sampling_interval_minutes / 60.0

    def run(self) -> pd.DataFrame:
        """Run the full simulation: environment -> PV -> virtual sensors.

        Returns a DataFrame combining environment, PV output, and sensor
        measurements with both 'true' and 'measured' columns.
        """
        env_df = self.env.generate()
        pv_df = self.pv.simulate(env_df, self.dt_hours)

        # Create virtual sensors
        sn = self.sensor_noise_cfg
        pzem = VirtualPZEM017(
            voltage_noise=sn.get("voltage_noise", 0.2),
            current_noise=sn.get("current_noise", 0.05),
            power_noise=sn.get("power_noise", 2.0),
            seed=self.scenario.env_config.random_seed + 1,
        )
        bh1750 = VirtualBH1750(
            irradiance_noise=sn.get("irradiance_noise", 15.0),
            seed=self.scenario.env_config.random_seed + 2,
        )
        ds18b20 = VirtualDS18B20(
            temperature_noise=sn.get("temperature_noise", 0.5),
            seed=self.scenario.env_config.random_seed + 3,
        )

        # Generate sensor measurements
        irr_meas = bh1750.measure(env_df["effective_irradiance"].values)
        temp_meas = ds18b20.measure(env_df["panel_temp"].values)
        v_meas = pzem.measure_voltage(pv_df["voltage"].values)
        i_meas = pzem.measure_current(pv_df["current"].values)
        p_meas = pzem.measure_power(pv_df["power"].values)
        e_meas = pzem.compute_energy(p_meas, self.dt_hours)

        result = pd.DataFrame({
            "timestamp": env_df["timestamp"],
            # True values
            "true_irradiance": env_df["effective_irradiance"].values,
            "true_ambient_temp": env_df["ambient_temp"].values,
            "true_panel_temp": env_df["panel_temp"].values,
            "true_voltage": pv_df["voltage"].values,
            "true_current": pv_df["current"].values,
            "true_power": pv_df["power"].values,
            "true_efficiency": pv_df["efficiency"].values,
            "true_energy": pv_df["energy"].values,
            # Sensor measurements
            "measured_irradiance": irr_meas,
            "measured_temperature": temp_meas,
            "measured_voltage": v_meas,
            "measured_current": i_meas,
            "measured_power": p_meas,
            "measured_energy": e_meas,
            # Environmental context
            "ghi": env_df["ghi"].values,
            "cloud_factor": env_df["cloud_factor"].values,
            "dust_factor": env_df["dust_factor"].values,
            "shading_factor": env_df["shading_factor"].values,
            "solar_zenith": env_df["solar_zenith"].values,
        })

        # Inject sensor faults if scenario calls for it
        if self.scenario.fault_type == "sensor":
            result = self._inject_sensor_faults(result)

        return result

    def _inject_sensor_faults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject controlled sensor faults: stuck values, abnormal, missing."""
        rng = np.random.default_rng(self.scenario.env_config.random_seed + 100)
        n = len(df)
        # 5% stuck values on voltage
        stuck_idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[stuck_idx, "measured_voltage"] = df["measured_voltage"].iloc[0]
        # 3% abnormal temperature spikes
        spike_idx = rng.choice(n, size=max(1, n // 33), replace=False)
        df.loc[spike_idx, "measured_temperature"] += rng.uniform(15, 25, size=len(spike_idx))
        # 2% missing readings (NaN)
        miss_idx = rng.choice(n, size=max(1, n // 50), replace=False)
        df.loc[miss_idx, "measured_irradiance"] = np.nan
        return df


def run_scenario(scenario_name: str, cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """Convenience: run a named scenario from config."""
    sim_cfg = cfg["simulation"]
    sensor_cfg = cfg["sensors"]
    s = seed if seed is not None else sim_cfg["random_seed"]
    scenario = get_scenario(
        scenario_name,
        seed=s,
        duration_days=sim_cfg["duration_days"],
        sampling_interval=sim_cfg["sampling_interval_minutes"],
    )
    sim = SolarSimulator(scenario, sensor_cfg)
    return sim.run()


if __name__ == "__main__":
    from utils import load_config, setup_logging, ensure_dir
    import sys

    logger = setup_logging()
    cfg = load_config()
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "normal_clear_sky"
    logger.info(f"Running simulation scenario: {scenario_name}")
    df = run_scenario(scenario_name, cfg)
    out_dir = ensure_dir("data/generated")
    out_path = out_dir / f"simulation_{scenario_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} samples to {out_path}")
    print(df.describe())
