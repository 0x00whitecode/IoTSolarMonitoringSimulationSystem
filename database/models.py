"""SQLAlchemy ORM models for the solar IoT database.

Tables:
  - sensor_measurements: all sensor readings
  - communication_logs: MQTT message delivery records
  - anomaly_logs: detected anomalies
  - experiment_results: experiment metrics
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Index
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SensorMeasurement(Base):
    __tablename__ = "sensor_measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    device_id = Column(String(50), index=True, nullable=False)
    irradiance = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    voltage = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    power = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    status = Column(String(50), default="ok")
    scenario = Column(String(50), index=True, default="normal")

    __table_args__ = (
        Index("ix_sensor_timestamp_device", "timestamp", "device_id"),
    )


class CommunicationLog(Base):
    __tablename__ = "communication_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    message_id = Column(String(64), nullable=True)
    topic = Column(String(200), nullable=True)
    latency = Column(Float, default=0.0)  # ms
    status = Column(String(50), default="delivered")
    device_id = Column(String(50), default="ESP32_SOLAR_001")


class AnomalyLog(Base):
    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    anomaly_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="warning")  # info, warning, critical
    value = Column(Float, nullable=True)
    expected_value = Column(Float, nullable=True)
    device_id = Column(String(50), default="ESP32_SOLAR_001")
    scenario = Column(String(50), default="normal")


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    scenario = Column(String(100), nullable=False, index=True)
    metric = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
