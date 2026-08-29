"""Unit tests for ML pipeline."""
import numpy as np
import pandas as pd
from ml.power_prediction import get_model, train_model, evaluate_model
from analytics.preprocessing import preprocess, prepare_ml_dataset


def _make_ml_test_df(n=200):
    ts = pd.date_range("2024-01-01 06:00", periods=n, freq="5min")
    irr = np.linspace(100, 1000, n) + np.random.normal(0, 20, n)
    temp = np.linspace(20, 45, n) + np.random.normal(0, 1, n)
    v = 40 * (0.7 + 0.3 * np.log(np.maximum(irr, 1) + 1) / np.log(1001))
    i = (np.maximum(irr, 0) / 1000) * 10
    p = v * i
    return pd.DataFrame({
        "timestamp": ts,
        "measured_irradiance": np.maximum(irr, 0), "measured_temperature": temp,
        "measured_voltage": v, "measured_current": i,
        "measured_power": p, "measured_energy": np.cumsum(p * 0.083),
    })


def test_model_instantiation():
    for name in ["linear_regression", "random_forest", "gradient_boosting"]:
        model = get_model(name, random_state=42)
        assert model is not None


def test_train_and_evaluate():
    df = _make_ml_test_df(200)
    processed = preprocess(df)
    X, y = prepare_ml_dataset(processed, target="measured_power")
    assert len(X) > 0 and len(y) > 0
    n_train = int(len(X) * 0.8)
    model, scaler = train_model(X.values[:n_train], y.values[:n_train], "linear_regression", 42)
    metrics = evaluate_model(model, scaler, X.values[n_train:], y.values[n_train:])
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["r2"] > 0
