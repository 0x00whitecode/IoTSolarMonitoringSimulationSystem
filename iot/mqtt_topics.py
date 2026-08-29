"""MQTT topic definitions for the solar IoT system.

Topic structure follows a hierarchical naming convention:
  solar/<device_id>/<parameter>

This mirrors the structure used in real IoT deployments.
"""
from __future__ import annotations

DEVICE_ID = "ESP32_SOLAR_001"


def topic_irradiance(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/irradiance"


def topic_temperature(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/temperature"


def topic_voltage(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/voltage"


def topic_current(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/current"


def topic_power(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/power"


def topic_status(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/status"


def topic_all(device_id: str = DEVICE_ID) -> str:
    return f"solar/{device_id}/all"


ALL_TOPICS = [
    topic_irradiance(),
    topic_temperature(),
    topic_voltage(),
    topic_current(),
    topic_power(),
    topic_status(),
    topic_all(),
]
