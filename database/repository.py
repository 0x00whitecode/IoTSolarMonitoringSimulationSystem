"""Data access layer for storing and retrieving sensor/IoT data."""
from __future__ import annotations
import pandas as pd
import logging
from typing import Optional
from sqlalchemy.orm import Session
from database.models import (SensorMeasurement, CommunicationLog,
                             AnomalyLog, ExperimentResult)

logger = logging.getLogger("smart_solar_iot.repository")


def insert_measurements(session: Session, df: pd.DataFrame,
                        device_id: str = "ESP32_SOLAR_001",
                        scenario: str = "normal") -> int:
    """Bulk insert sensor measurements from a DataFrame."""
    count = 0
    for _, row in df.iterrows():
        ts = row.get("timestamp")
        if pd.isna(ts):
            continue
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        rec = SensorMeasurement(
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            device_id=device_id,
            irradiance=_safe_float(row.get("measured_irradiance")),
            temperature=_safe_float(row.get("measured_temperature")),
            voltage=_safe_float(row.get("measured_voltage")),
            current=_safe_float(row.get("measured_current")),
            power=_safe_float(row.get("measured_power")),
            energy=_safe_float(row.get("measured_energy")),
            status="ok",
            scenario=scenario,
        )
        session.add(rec)
        count += 1
    session.commit()
    logger.info(f"Inserted {count} sensor measurements (scenario={scenario})")
    return count


def insert_comm_logs(session: Session, log_df: pd.DataFrame,
                      device_id: str = "ESP32_SOLAR_001") -> int:
    """Bulk insert communication logs."""
    count = 0
    for _, row in log_df.iterrows():
        ts = row.get("timestamp")
        if pd.isna(ts):
            continue
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        rec = CommunicationLog(
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            message_id=str(row.get("message_id", "")),
            topic=str(row.get("topic", "")),
            latency=_safe_float(row.get("latency_ms")),
            status=str(row.get("status", "unknown")),
            device_id=device_id,
        )
        session.add(rec)
        count += 1
    session.commit()
    logger.info(f"Inserted {count} communication logs")
    return count


def insert_anomaly(session: Session, timestamp, anomaly_type: str,
                    severity: str, value: float, expected_value: float,
                    device_id: str = "ESP32_SOLAR_001",
                    scenario: str = "normal"):
    """Insert a single anomaly record."""
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)
    rec = AnomalyLog(
        timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
        anomaly_type=anomaly_type,
        severity=severity,
        value=value,
        expected_value=expected_value,
        device_id=device_id,
        scenario=scenario,
    )
    session.add(rec)
    session.commit()


def insert_experiment_result(session: Session, experiment_id: int,
                              scenario: str, metric: str, value: float,
                              notes: str = ""):
    """Insert an experiment result."""
    rec = ExperimentResult(
        experiment_id=experiment_id,
        scenario=scenario,
        metric=metric,
        value=value,
        notes=notes,
    )
    session.add(rec)
    session.commit()


def query_measurements(session: Session, scenario: Optional[str] = None,
                        limit: int | None = None) -> pd.DataFrame:
    """Query sensor measurements as a DataFrame."""
    q = session.query(SensorMeasurement)
    if scenario:
        q = q.filter(SensorMeasurement.scenario == scenario)
    if limit:
        q = q.limit(limit)
    rows = q.all()
    return pd.DataFrame([{
        "id": r.id, "timestamp": r.timestamp, "device_id": r.device_id,
        "irradiance": r.irradiance, "temperature": r.temperature,
        "voltage": r.voltage, "current": r.current, "power": r.power,
        "energy": r.energy, "status": r.status, "scenario": r.scenario,
    } for r in rows])


def query_comm_logs(session: Session, limit: int | None = None) -> pd.DataFrame:
    """Query communication logs as a DataFrame."""
    q = session.query(CommunicationLog)
    if limit:
        q = q.limit(limit)
    rows = q.all()
    return pd.DataFrame([{
        "id": r.id, "timestamp": r.timestamp, "message_id": r.message_id,
        "topic": r.topic, "latency": r.latency, "status": r.status,
        "device_id": r.device_id,
    } for r in rows])


def query_anomalies(session: Session, scenario: Optional[str] = None) -> pd.DataFrame:
    """Query anomaly logs as a DataFrame."""
    q = session.query(AnomalyLog)
    if scenario:
        q = q.filter(AnomalyLog.scenario == scenario)
    rows = q.all()
    return pd.DataFrame([{
        "id": r.id, "timestamp": r.timestamp, "anomaly_type": r.anomaly_type,
        "severity": r.severity, "value": r.value, "expected_value": r.expected_value,
        "device_id": r.device_id, "scenario": r.scenario,
    } for r in rows])


def query_experiment_results(session: Session, scenario: Optional[str] = None) -> pd.DataFrame:
    """Query experiment results as a DataFrame."""
    q = session.query(ExperimentResult)
    if scenario:
        q = q.filter(ExperimentResult.scenario == scenario)
    rows = q.all()
    return pd.DataFrame([{
        "id": r.id, "experiment_id": r.experiment_id, "scenario": r.scenario,
        "metric": r.metric, "value": r.value, "timestamp": r.timestamp,
        "notes": r.notes,
    } for r in rows])


def _safe_float(val) -> float | None:
    """Convert to float, returning None for NaN/missing."""
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
