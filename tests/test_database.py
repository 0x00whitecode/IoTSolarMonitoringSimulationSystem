"""Unit tests for database operations."""
import pandas as pd
import numpy as np
import tempfile
import os
from database.database import init_db, get_session
from database.repository import (insert_measurements, insert_comm_logs,
                                   query_measurements, query_comm_logs,
                                   insert_anomaly, query_anomalies)


def _make_test_df(n=5):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "measured_irradiance": np.random.uniform(100, 1000, n),
        "measured_temperature": np.random.uniform(20, 40, n),
        "measured_voltage": np.random.uniform(30, 40, n),
        "measured_current": np.random.uniform(5, 10, n),
        "measured_power": np.random.uniform(100, 400, n),
        "measured_energy": np.cumsum(np.random.uniform(100, 400, n) * 0.083),
    })


def test_database_insert_and_query():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = init_db(db_path)
        session = get_session(engine)
        df = _make_test_df(5)
        count = insert_measurements(session, df, device_id="TEST_001", scenario="test")
        assert count == 5
        result = query_measurements(session, scenario="test")
        assert len(result) == 5
        assert "power" in result.columns
        session.close()
    finally:
        os.unlink(db_path)


def test_comm_log_insert():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = init_db(db_path)
        session = get_session(engine)
        log_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
            "message_id": ["a", "b", "c"],
            "topic": ["solar/test/all"] * 3,
            "latency_ms": [50.0, 60.0, 55.0],
            "status": ["delivered"] * 3,
        })
        count = insert_comm_logs(session, log_df, device_id="TEST_001")
        assert count == 3
        result = query_comm_logs(session)
        assert len(result) == 3
        session.close()
    finally:
        os.unlink(db_path)


def test_anomaly_insert():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = init_db(db_path)
        session = get_session(engine)
        insert_anomaly(session, pd.Timestamp("2024-01-01 12:00"),
                       "overheating", "critical", 75.0, 70.0)
        result = query_anomalies(session)
        assert len(result) == 1
        assert result.iloc[0]["anomaly_type"] == "overheating"
        session.close()
    finally:
        os.unlink(db_path)
