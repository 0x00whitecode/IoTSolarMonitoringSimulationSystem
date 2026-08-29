"""MQTT publisher with simulated broker fallback.

When use_simulated_broker=True, messages are passed through an in-memory
queue instead of a real MQTT broker. This allows the full pipeline to run
without external dependencies.

Simulates network conditions:
  - Packet loss: messages are dropped with configurable probability
  - Network delay: latency is added from a normal distribution
  - Reconnect: simulated connection failures and retries

Metrics tracked:
  - latency (ms)
  - throughput (msg/s)
  - delivery_rate (%)
  - packet_loss (%)
"""
from __future__ import annotations
import json
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger("smart_solar_iot.mqtt_publisher")


@dataclass
class MQTTMessage:
    message_id: str
    topic: str
    payload: dict
    timestamp: float
    delivered: bool = False
    latency_ms: float = 0.0
    dropped: bool = False


class SimulatedMQTTBroker:
    """In-memory message broker that mimics MQTT pub/sub behavior.

    When a real broker is unavailable, this allows the full pipeline to
    run locally. Messages pass through a queue with simulated network
    conditions (delay, packet loss).
    """

    def __init__(self, packet_loss_prob: float = 0.02,
                 delay_mean_ms: float = 50.0, delay_std_ms: float = 20.0,
                 seed: int = 42):
        self.packet_loss_prob = packet_loss_prob
        self.delay_mean = delay_mean_ms / 1000.0
        self.delay_std = delay_std_ms / 1000.0
        self.rng = __import__("numpy").random.default_rng(seed)
        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._message_log: deque[MQTTMessage] = deque(maxlen=100000)
        self._running = True

    def subscribe(self, topic: str, callback: Callable):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, payload: dict, timestamp: float) -> MQTTMessage:
        msg_id = str(uuid.uuid4())
        msg = MQTTMessage(
            message_id=msg_id,
            topic=topic,
            payload=payload,
            timestamp=timestamp,
        )
        # Simulate packet loss
        if self.rng.random() < self.packet_loss_prob:
            msg.dropped = True
            msg.delivered = False
            self._message_log.append(msg)
            return msg

        # Simulate network delay
        delay = max(0, self.rng.normal(self.delay_mean, self.delay_std))
        msg.latency_ms = delay * 1000.0

        # Deliver to subscribers after delay (in a thread for async)
        def _deliver():
            time.sleep(delay)
            msg.delivered = True
            with self._lock:
                for cb in self._subscribers.get(topic, []):
                    try:
                        cb(topic, payload, msg)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")
                self._message_log.append(msg)

        t = threading.Thread(target=_deliver, daemon=True)
        t.start()
        return msg

    def get_all_messages(self) -> list[MQTTMessage]:
        return list(self._message_log)

    def stop(self):
        self._running = False


class MQTTPublisher:
    """Publishes sensor data to MQTT topics (real or simulated broker)."""

    def __init__(self, broker: str = "localhost", port: int = 1883,
                 device_id: str = "ESP32_SOLAR_001",
                 use_simulated: bool = True,
                 packet_loss_prob: float = 0.02,
                 delay_mean_ms: float = 50.0, delay_std_ms: float = 20.0,
                 seed: int = 42):
        self.device_id = device_id
        self.use_simulated = use_simulated
        self.broker_host = broker
        self.broker_port = port
        self._client = None
        self._sim_broker: Optional[SimulatedMQTTBroker] = None
        self._connected = False
        self._reconnect_attempts = 5
        self._reconnect_delay = 2.0

        if use_simulated:
            self._sim_broker = SimulatedMQTTBroker(
                packet_loss_prob, delay_mean_ms, delay_std_ms, seed
            )
            self._connected = True
            logger.info("Using simulated MQTT broker (no external broker required)")
        else:
            self._connect_real_broker()

    def _connect_real_broker(self):
        """Attempt to connect to a real MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id=f"publisher_{self.device_id}")
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect(self.broker_host, self.broker_port, 60)
            self._client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            logger.warning(f"Could not connect to real MQTT broker: {e}. "
                           "Falling back to simulated broker.")
            self.use_simulated = True
            self._sim_broker = SimulatedMQTTBroker()
            self._connected = True

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT publisher connected")
        else:
            logger.error(f"MQTT connect failed with code {rc}")
            self._connected = False

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.warning(f"MQTT disconnected (rc={rc}). Attempting reconnect...")
        self._reconnect()

    def _reconnect(self):
        for attempt in range(self._reconnect_attempts):
            logger.info(f"Reconnect attempt {attempt + 1}/{self._reconnect_attempts}")
            try:
                self._client.reconnect()
                return
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
                time.sleep(self._reconnect_delay)
        logger.error("All reconnect attempts failed. Switching to simulated broker.")
        self.use_simulated = True
        self._sim_broker = SimulatedMQTTBroker()
        self._connected = True

    def publish(self, topic: str, payload: dict) -> MQTTMessage | None:
        """Publish a message to a topic."""
        ts = time.time()
        if self.use_simulated and self._sim_broker:
            return self._sim_broker.publish(topic, payload, ts)
        elif self._client and self._connected:
            import paho.mqtt.client as mqtt
            info = self._client.publish(topic, json.dumps(payload), qos=1)
            return MQTTMessage(
                message_id=str(uuid.uuid4()),
                topic=topic,
                payload=payload,
                timestamp=ts,
                delivered=(info.rc == mqtt.MQTT_ERR_SUCCESS),
            )
        return None

    def get_sim_broker(self) -> Optional[SimulatedMQTTBroker]:
        return self._sim_broker

    def disconnect(self):
        if self._client and not self.use_simulated:
            self._client.loop_stop()
            self._client.disconnect()
        if self._sim_broker:
            self._sim_broker.stop()
