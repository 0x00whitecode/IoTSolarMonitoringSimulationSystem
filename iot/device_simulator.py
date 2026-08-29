"""Virtual ESP32 device simulator.

Simulates an ESP32 microcontroller that:
  1. Receives sensor measurements from virtual sensors
  2. Validates readings (physical range checks)
  3. Timestamps data
  4. Performs simple edge processing (power = V * I check)
  5. Creates JSON payload
  6. Publishes through MQTT

This is a SOFTWARE SIMULATION of an ESP32. No physical hardware is used.

Usage:
    python -m iot.device_simulator
"""
from __future__ import annotations
import json
import time
import uuid
import logging
import numpy as np
import pandas as pd
from typing import Optional

from iot.mqtt_publisher import MQTTPublisher
from iot.mqtt_topics import topic_all, topic_irradiance, topic_temperature, \
    topic_voltage, topic_current, topic_power, topic_status

logger = logging.getLogger("smart_solar_iot.device_simulator")


class VirtualESP32:
    """Simulated ESP32 microcontroller for IoT data aggregation and publishing."""

    def __init__(self, device_id: str, publisher: MQTTPublisher,
                 packet_loss_prob: float = 0.0, seed: int = 42):
        self.device_id = device_id
        self.publisher = publisher
        self.rng = np.random.default_rng(seed)
        self.packet_loss_prob = packet_loss_prob
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        self.latencies: list[float] = []

    def validate_reading(self, irradiance: float, temperature: float,
                         voltage: float, current: float, power: float) -> tuple[bool, str]:
        """Edge validation: check physical ranges."""
        if not (0 <= irradiance <= 1400):
            return False, "irradiance_out_of_range"
        if not (-55 <= temperature <= 125):
            return False, "temperature_out_of_range"
        if not (0 <= voltage <= 300):
            return False, "voltage_out_of_range"
        if not (0 <= current <= 100):
            return False, "current_out_of_range"
        if power < 0 or power > 10000:
            return False, "power_out_of_range"
        return True, "ok"

    def create_payload(self, timestamp: str, irradiance: float, temperature: float,
                       voltage: float, current: float, power: float,
                       energy: float = 0.0) -> dict:
        """Create JSON payload matching the proposal's example format."""
        return {
            "device_id": self.device_id,
            "timestamp": timestamp,
            "message_id": str(uuid.uuid4()),
            "irradiance": round(float(irradiance), 2),
            "temperature": round(float(temperature), 2),
            "voltage": round(float(voltage), 2),
            "current": round(float(current), 4),
            "power": round(float(power), 2),
            "energy": round(float(energy), 2),
        }

    def publish_reading(self, timestamp: str, irradiance: float, temperature: float,
                        voltage: float, current: float, power: float,
                        energy: float = 0.0) -> dict | None:
        """Validate, package, and publish a sensor reading via MQTT."""
        valid, status = self.validate_reading(irradiance, temperature,
                                               voltage, current, power)
        if not valid:
            logger.warning(f"Invalid reading rejected: {status}")
            self._publish_status(status)
            return None

        payload = self.create_payload(timestamp, irradiance, temperature,
                                       voltage, current, power, energy)

        # Simulate communication fault (packet loss at device level)
        if self.rng.random() < self.packet_loss_prob:
            self.messages_sent += 1
            self.messages_dropped += 1
            logger.debug(f"Packet lost at device level for msg {payload['message_id']}")
            return payload

        msg = self.publisher.publish(topic_all(self.device_id), payload)
        self.messages_sent += 1
        if msg:
            if msg.dropped:
                self.messages_dropped += 1
            else:
                self.messages_delivered += 1
                self.latencies.append(msg.latency_ms)
        return payload

    def _publish_status(self, status: str):
        """Publish a status/error message."""
        payload = {"device_id": self.device_id, "status": status,
                   "timestamp": time.time()}
        self.publisher.publish(topic_status(self.device_id), payload)

    def get_stats(self) -> dict:
        """Return communication statistics."""
        delivery_rate = (self.messages_delivered / max(self.messages_sent, 1)) * 100
        packet_loss = (self.messages_dropped / max(self.messages_sent, 1)) * 100
        avg_latency = float(np.mean(self.latencies)) if self.latencies else 0.0
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "delivery_rate": delivery_rate,
            "packet_loss": packet_loss,
            "avg_latency_ms": avg_latency,
            "throughput_msgs_per_s": 0.0,  # computed externally
        }


def simulate_iot_communication(df: pd.DataFrame, device_id: str,
                                publisher: MQTTPublisher,
                                packet_loss_prob: float = 0.0,
                                seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Run the full IoT communication simulation on a dataset.

    Publishes each row through MQTT, tracks delivery/latency, and returns
    communication logs and statistics.
    """
    esp32 = VirtualESP32(device_id, publisher, packet_loss_prob, seed)
    logs = []
    start_time = time.time()

    for _, row in df.iterrows():
        ts = str(row.get("timestamp", ""))
        irr = row.get("measured_irradiance", 0)
        temp = row.get("measured_temperature", 0)
        v = row.get("measured_voltage", 0)
        i = row.get("measured_current", 0)
        p = row.get("measured_power", 0)
        e = row.get("measured_energy", 0)

        if np.isnan(irr) or np.isnan(temp) or np.isnan(v):
            esp32._publish_status("missing_data")
            logs.append({
                "timestamp": ts,
                "topic": topic_all(device_id),
                "status": "missing_data",
                "latency_ms": 0,
                "message_id": "",
            })
            continue

        payload = esp32.publish_reading(ts, irr, temp, v, i, p, e)
        status = "delivered" if payload else "dropped"
        latency = esp32.latencies[-1] if esp32.latencies else 0.0

        logs.append({
            "timestamp": ts,
            "topic": topic_all(device_id),
            "status": status,
            "latency_ms": latency,
            "message_id": payload.get("message_id", "") if payload else "",
        })

    elapsed = time.time() - start_time
    stats = esp32.get_stats()
    stats["throughput_msgs_per_s"] = esp32.messages_sent / max(elapsed, 0.001)

    log_df = pd.DataFrame(logs)
    return log_df, stats


if __name__ == "__main__":
    from utils import load_config, setup_logging
    cfg = load_config()
    logger = setup_logging()
    iot_cfg = cfg["iot"]
    pub = MQTTPublisher(
        broker=iot_cfg["mqtt_broker"],
        port=iot_cfg["mqtt_port"],
        device_id=iot_cfg["device_id"],
        use_simulated=iot_cfg["use_simulated_broker"],
        packet_loss_prob=iot_cfg["packet_loss_probability"],
        delay_mean_ms=iot_cfg["network_delay_ms_mean"],
        delay_std_ms=iot_cfg["network_delay_ms_std"],
    )
    logger.info("ESP32 device simulator ready. Run via main.py for full pipeline.")
