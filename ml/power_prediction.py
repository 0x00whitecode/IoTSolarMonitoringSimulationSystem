"""Power prediction models.

Supervised regression to predict solar PV power output from
environmental and electrical features.

Models evaluated:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost (optional)

Target: measured_power
Features: irradiance, temperature, voltage, current, time-based, rolling
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("smart_solar_iot.power_prediction")


def get_model(name: str, random_state: int = 42) -> Any:
    """Instantiate a regression model by name."""
    if name == "linear_regression":
        return LinearRegression()
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
        except ImportError:
            logger.warning("XGBoost not available, using Gradient Boosting")
            return GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif name == "svm":
        return SVR(kernel="rbf", C=1.0)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                model_name: str, random_state: int = 42) -> Tuple[Any, StandardScaler]:
    """Train a regression model with feature scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = get_model(model_name, random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model: Any, scaler: StandardScaler,
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a regression model. Returns MAE, RMSE, R2."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2),
            "predictions": y_pred.tolist()}


def save_model(model: Any, scaler: StandardScaler, feature_names: list,
               model_name: str, output_dir: str = "models/ml"):
    """Save trained model, scaler, and feature list."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{model_name}.joblib")
    joblib.dump(scaler, out / f"{model_name}_scaler.joblib")
    with open(out / f"{model_name}_features.txt", "w") as f:
        f.write("\n".join(feature_names))
    logger.info(f"Saved {model_name} model to {out}")


def load_model(model_name: str, model_dir: str = "models/ml") -> Tuple[Any, StandardScaler, list]:
    """Load a trained model, scaler, and feature list."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / f"{model_name}.joblib")
    scaler = joblib.load(model_dir / f"{model_name}_scaler.joblib")
    with open(model_dir / f"{model_name}_features.txt", "r") as f:
        features = [line.strip() for line in f if line.strip()]
    return model, scaler, features
