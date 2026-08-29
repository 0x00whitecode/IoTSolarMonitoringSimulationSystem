"""Unit tests for analytics modules."""
import numpy as np
import pandas as pd
from analytics.preprocessing import preprocess
from analytics.performance import compute_performance_metrics, correlation_analysis
from analytics.anomaly_detection import detect_anomalies_rule_based, detect_anomalies_isolation_forest
from analytics.fault_detection import classify_fault
from analytics.statistics import compute_summary_stats, check_normality, percentage_improvement


def _make_test_df(n=100):
    ts = pd.date_range("2024-01-01 06:00", periods=n, freq="5min")
    irr = np.linspace(100, 1000, n)
    temp = np.linspace(20, 45, n)
    v = np.linspace(30, 40, n)
    i = np.linspace(2, 10, n)
    p = v * i
    return pd.DataFrame({
        "timestamp": ts,
        "true_irradiance": irr, "true_panel_temp": temp,
        "true_voltage": v, "true_current": i, "true_power": p,
        "true_efficiency": p / (irr * 1.7) * 100,
        "true_energy": np.cumsum(p * 0.083),
        "measured_irradiance": irr + np.random.normal(0, 5, n),
        "measured_temperature": temp + np.random.normal(0, 0.5, n),
        "measured_voltage": v + np.random.normal(0, 0.2, n),
        "measured_current": i + np.random.normal(0, 0.05, n),
        "measured_power": p + np.random.normal(0, 2, n),
        "measured_energy": np.cumsum((p + np.random.normal(0, 2, n)) * 0.083),
        "dust_factor": np.ones(n), "shading_factor": np.zeros(n),
    })


def test_preprocessing():
    df = _make_test_df(50)
    processed = preprocess(df)
    assert "hour" in processed.columns
    assert "rolling_power" in processed.columns


def test_performance_metrics():
    df = _make_test_df(100)
    metrics = compute_performance_metrics(df, panel_area=1.7, nominal_power=400, dt_hours=0.083)
    assert "total_energy_wh" in metrics
    assert metrics["total_energy_wh"] > 0


def test_correlation_analysis():
    df = _make_test_df(50)
    corr = correlation_analysis(df)
    assert not corr.empty
    assert "measured_power" in corr.columns


def test_anomaly_detection_rule_based():
    df = _make_test_df(50)
    anomalies = detect_anomalies_rule_based(df)
    assert isinstance(anomalies, pd.DataFrame)


def test_anomaly_detection_isolation_forest():
    df = _make_test_df(100)
    anomalies = detect_anomalies_isolation_forest(df)
    assert isinstance(anomalies, pd.DataFrame)


def test_fault_classification():
    row = pd.Series({
        "measured_temperature": 75.0, "measured_voltage": 35.0,
        "measured_power": 200.0, "true_power": 200.0,
        "measured_irradiance": 800.0,
        "dust_factor": 1.0, "shading_factor": 0.0,
    })
    assert classify_fault(row) == "overheating"
    row2 = pd.Series({
        "measured_temperature": 30.0, "measured_voltage": 35.0,
        "measured_power": 200.0, "true_power": 200.0,
        "measured_irradiance": 800.0,
        "dust_factor": 1.0, "shading_factor": 0.0,
    })
    assert classify_fault(row2) == "normal"


def test_statistics():
    vals = np.random.normal(100, 10, 50)
    stats = compute_summary_stats(vals)
    assert "mean" in stats and "ci_lower" in stats
    is_normal, p = check_normality(vals)
    assert isinstance(is_normal, bool)
    assert percentage_improvement(80, 100) == 20.0
