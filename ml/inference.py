"""ML inference module for real-time power prediction."""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from ml.power_prediction import load_model

logger = logging.getLogger("smart_solar_iot.inference")


def predict_power(features: dict, model_name: str = "random_forest",
                   model_dir: str = "models/ml") -> float:
    """Predict power output from a feature dictionary."""
    model, scaler, feature_names = load_model(model_name, model_dir)
    X = np.array([[features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return float(max(pred, 0.0))


def predict_batch(df: pd.DataFrame, model_name: str = "random_forest",
                  model_dir: str = "models/ml") -> np.ndarray:
    """Predict power for a batch of samples."""
    model, scaler, feature_names = load_model(model_name, model_dir)
    X = df[feature_names].fillna(0).values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return np.maximum(preds, 0.0)
