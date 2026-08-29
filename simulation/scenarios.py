"""Scenario definitions for controlled experiments.

Each scenario modifies the environment and PV configuration to simulate
a specific operating condition. Scenarios map directly to Experiments 1-9
in the research methodology.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
from simulation.solar_environment import EnvironmentConfig
from simulation.pv_model import PVConfig


@dataclass
class Scenario:
    name: str
    description: str
    env_config: EnvironmentConfig
    pv_config: PVConfig
    fault_type: str = "none"  # none, dust, shading, overheating, sensor, communication
    experiment_id: int = 0


def _base_env(seed: int = 42) -> EnvironmentConfig:
    return EnvironmentConfig(random_seed=seed)


def _base_pv() -> PVConfig:
    return PVConfig()


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "normal_clear_sky": {
        "description": "Experiment 1: Clear-sky operation, no faults. Baseline reference.",
        "cloud_cover_mean": 0.0,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 28.0,
        "ambient_temp_amplitude": 8.0,
        "fault_type": "none",
        "experiment_id": 1,
    },
    "low_irradiance": {
        "description": "Experiment 2: Low irradiance due to heavy overcast conditions.",
        "cloud_cover_mean": 0.7,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 22.0,
        "ambient_temp_amplitude": 5.0,
        "fault_type": "none",
        "experiment_id": 2,
    },
    "high_temperature": {
        "description": "Experiment 3: High ambient temperature, clear sky. Tests thermal derating.",
        "cloud_cover_mean": 0.0,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 40.0,
        "ambient_temp_amplitude": 10.0,
        "noct": 50.0,
        "fault_type": "overheating",
        "experiment_id": 3,
    },
    "cloud_variation": {
        "description": "Experiment 4: Variable cloud cover (intermittent irradiance).",
        "cloud_cover_mean": 0.4,
        "cloud_cover_var": 0.3,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 26.0,
        "ambient_temp_amplitude": 6.0,
        "fault_type": "none",
        "experiment_id": 4,
    },
    "dust_accumulation": {
        "description": "Experiment 5: Progressive dust accumulation over simulation period.",
        "cloud_cover_mean": 0.0,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.05,
        "dust_loss_rate": 0.02,
        "ambient_temp_mean": 30.0,
        "ambient_temp_amplitude": 8.0,
        "fault_type": "dust",
        "experiment_id": 5,
    },
    "partial_shading": {
        "description": "Experiment 6: Partial shading of panel (30% blocked).",
        "cloud_cover_mean": 0.0,
        "shading_factor": 0.3,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 28.0,
        "ambient_temp_amplitude": 8.0,
        "fault_type": "shading",
        "experiment_id": 6,
    },
    "sensor_failure": {
        "description": "Experiment 7: Sensor faults (stuck values, abnormal readings, missing data).",
        "cloud_cover_mean": 0.1,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 28.0,
        "ambient_temp_amplitude": 8.0,
        "fault_type": "sensor",
        "experiment_id": 7,
    },
    "communication_failure": {
        "description": "Experiment 8: Communication faults (packet loss, latency, outage).",
        "cloud_cover_mean": 0.1,
        "shading_factor": 0.0,
        "dust_loss_initial": 0.0,
        "dust_loss_rate": 0.0,
        "ambient_temp_mean": 28.0,
        "ambient_temp_amplitude": 8.0,
        "fault_type": "communication",
        "experiment_id": 8,
    },
    "mixed_conditions": {
        "description": "Experiment 9: Mixed environmental conditions (clouds, dust, high temp).",
        "cloud_cover_mean": 0.3,
        "shading_factor": 0.1,
        "dust_loss_initial": 0.03,
        "dust_loss_rate": 0.01,
        "ambient_temp_mean": 35.0,
        "ambient_temp_amplitude": 9.0,
        "noct": 48.0,
        "fault_type": "none",
        "experiment_id": 9,
    },
}


def get_scenario(name: str, seed: int = 42, duration_days: int = 1,
                 sampling_interval: int = 5) -> Scenario:
    """Build a Scenario from preset definitions."""
    preset = SCENARIO_PRESETS[name]
    env = EnvironmentConfig(
        random_seed=seed,
        duration_days=duration_days,
        sampling_interval_minutes=sampling_interval,
        cloud_cover_mean=preset.get("cloud_cover_mean", 0.0),
        cloud_cover_var=preset.get("cloud_cover_var", 0.0),
        shading_factor=preset.get("shading_factor", 0.0),
        dust_loss_initial=preset.get("dust_loss_initial", 0.0),
        dust_loss_rate=preset.get("dust_loss_rate", 0.0),
        ambient_temp_mean=preset.get("ambient_temp_mean", 28.0),
        ambient_temp_amplitude=preset.get("ambient_temp_amplitude", 8.0),
        noct=preset.get("noct", 45.0),
    )
    pv = PVConfig(
        shading_factor=preset.get("shading_factor", 0.0),
        dust_loss_initial=preset.get("dust_loss_initial", 0.0),
        dust_loss_rate=preset.get("dust_loss_rate", 0.0),
    )
    return Scenario(
        name=name,
        description=preset["description"],
        env_config=env,
        pv_config=pv,
        fault_type=preset.get("fault_type", "none"),
        experiment_id=preset["experiment_id"],
    )


def all_scenario_names() -> list[str]:
    return list(SCENARIO_PRESETS.keys())
