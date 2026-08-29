"""Experiment configuration and scenario registry.

Maps experiments to their IDs, scenarios, and expected metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

EXPERIMENTS: Dict[int, Dict] = {
    1: {"name": "normal_clear_sky", "scenario": "normal_clear_sky",
        "description": "Baseline clear-sky operation with no faults."},
    2: {"name": "low_irradiance", "scenario": "low_irradiance",
        "description": "Low irradiance due to heavy overcast conditions."},
    3: {"name": "high_temperature", "scenario": "high_temperature",
        "description": "High ambient temperature testing thermal derating."},
    4: {"name": "cloud_variation", "scenario": "cloud_variation",
        "description": "Variable cloud cover (intermittent irradiance)."},
    5: {"name": "dust_accumulation", "scenario": "dust_accumulation",
        "description": "Progressive dust accumulation over time."},
    6: {"name": "partial_shading", "scenario": "partial_shading",
        "description": "Partial shading of PV panel (30% blocked)."},
    7: {"name": "sensor_failure", "scenario": "sensor_failure",
        "description": "Sensor faults: stuck values, abnormal readings, missing data."},
    8: {"name": "communication_failure", "scenario": "communication_failure",
        "description": "Communication faults: packet loss, latency, outage."},
    9: {"name": "mixed_conditions", "scenario": "mixed_conditions",
        "description": "Mixed environmental conditions (clouds, dust, high temp)."},
}


def get_experiment_list() -> List[Dict]:
    return [{"id": k, **v} for k, v in EXPERIMENTS.items()]
