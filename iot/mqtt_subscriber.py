"""MQTT subscriber for ingesting sensor data into the database.

Subscribes to all solar IoT topics and forwards received messages
to a callback (typically the database ingestion function).
"""
from __future__ import annotations
import json
import time
import logging
import threading
from typing import Callable, Optional
from iot.mqtt_publisher import SimulatedMQTTBroker
from iot.mqtt_topics import ALL_TOPICS, topic_all

logger = logging.getLogger("smart_solar_iot.mqtt_subscriber")


class MQTTSubscriber:
    """Subscribes to MQTT topics and processes incoming messages."""

    def __init__(self, sim_broker: Optional[SimulatedMQTTBroker] = None,
                 on_message: Callable | None = None):
        self.sim_broker = sim_broker
        self.on_message = on_message
        self.messages_received = 0
        self._client = None
        self._running = False

    def start(self):
        """Start subscribing to topics."""
        if self.sim_broker:
            for t in ALL_TOPICS:
                self.sim_broker.subscribe(t, self._handle_message)
            logger.info(f"Subscribed to {len(ALL_TOPICS)} topics on simulated broker")
        else:
            logger.warning("No broker available for subscription")

    def _handle_message(self, topic: str, payload: dict, msg):
        """Callback for received messages."""
        self.messages_received += 1
        if self.on_message:
            try:
                self.on_message(topic, payload, msg)
            except Exception as e:
                logger.error(f"Error processing message on {topic}: {e}")

    def stop(self):
        self._running = False
