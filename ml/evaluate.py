"""ML evaluation utilities for computing and saving metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("smart_solar_iot.ml_evaluate")


def evaluate_all_power_models(results: Dict) -> pd.DataFrame:
    """Convert power model results dict to a comparison DataFrame."""
    rows = []
    for name, res in results.items():
        if "error" in res:
            rows.append({"model": name, "mae": None, "rmse": None, "r2": None, "status": "error"})
            continue
        rows.append({
            "model": name,
            "mae": res.get("mae"), "rmse": res.get("rmse"), "r2": res.get("r2"),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def evaluate_all_classifiers(results: Dict) -> pd.DataFrame:
    """Convert classifier results dict to a comparison DataFrame."""
    rows = []
    for name, res in results.items():
        if not isinstance(res, dict) or "error" in res:
            rows.append({"model": name, "accuracy": None, "precision": None,
                         "recall": None, "f1": None, "status": "error"})
            continue
        rows.append({
            "model": name,
            "accuracy": res.get("accuracy"), "precision": res.get("precision"),
            "recall": res.get("recall"), "f1": res.get("f1"),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def save_predictions(results: Dict, output_dir: str = "results/predictions"):
    """Save model predictions and actual values for visualization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        if "predictions" not in res or "y_test" not in res:
            continue
        df = pd.DataFrame({
            "actual": res["y_test"],
            "predicted": res["predictions"],
        })
        df.to_csv(out / f"power_prediction_{name}.csv", index=False)
        logger.info(f"Saved predictions for {name}")


def save_metrics_table(df: pd.DataFrame, filename: str,
                       output_dir: str = "results/tables"):
    """Save a metrics DataFrame as CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / filename, index=False)
    logger.info(f"Saved metrics table: {filename}")
