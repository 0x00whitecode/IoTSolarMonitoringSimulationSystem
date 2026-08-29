"""ML training pipeline for power prediction and fault classification.

Usage:
    python -m ml.train
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from analytics.preprocessing import preprocess, prepare_ml_dataset
from ml.power_prediction import train_model, evaluate_model, save_model

logger = logging.getLogger("smart_solar_iot.ml_train")


def train_power_models(df: pd.DataFrame, cfg: dict, output_dir: str = "models/ml") -> dict:
    """Train and evaluate all power prediction models."""
    logger.info("Training power prediction models...")
    processed = preprocess(df)
    X, y = prepare_ml_dataset(processed, target="measured_power")

    if y is None or len(X) == 0:
        raise ValueError("No data available for training")

    ml_cfg = cfg.get("machine_learning", {})
    test_size = ml_cfg.get("test_size", 0.2)
    random_state = ml_cfg.get("random_state", 42)
    model_names = ml_cfg.get("models", ["linear_regression", "random_forest",
                                         "gradient_boosting", "xgboost"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    results = {}
    for name in model_names:
        logger.info(f"Training {name}...")
        try:
            model, scaler = train_model(X_train.values, y_train.values, name, random_state)
            metrics = evaluate_model(model, scaler, X_test.values, y_test.values)
            save_model(model, scaler, list(X.columns), name, output_dir)
            results[name] = {
                "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"],
                "predictions": metrics["predictions"],
                "y_test": y_test.values.tolist(),
            }
            logger.info(f"{name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def train_fault_classifier(df: pd.DataFrame, cfg: dict,
                           output_dir: str = "models/ml") -> dict:
    """Train fault classification models.

    Labels come from the classify_fault() function, which uses known
    simulation conditions. This is a controlled, defensible labelling approach.
    """
    from analytics.fault_detection import detect_faults

    logger.info("Training fault classification models...")
    processed = preprocess(df)
    labelled = detect_faults(processed)

    feature_cols = ["measured_irradiance", "measured_temperature", "measured_voltage",
                    "measured_current", "hour", "day_of_year",
                    "rolling_power", "rolling_temperature", "rolling_irradiance"]
    available = [c for c in feature_cols if c in labelled.columns]

    X = labelled[available].copy()
    # Fill NaN
    for col in available:
        X[col] = X[col].fillna(X[col].median() if X[col].median() == X[col].median() else 0)

    le = LabelEncoder()
    y = le.fit_transform(labelled["fault_type"])

    if len(set(y)) < 2:
        logger.warning("Only one fault class in data. Skipping classification training.")
        return {"error": "insufficient_classes", "classes": list(le.classes_)}

    ml_cfg = cfg.get("machine_learning", {})
    test_size = ml_cfg.get("test_size", 0.2)
    random_state = ml_cfg.get("random_state", 42)
    model_names = ml_cfg.get("classification_models", ["random_forest", "gradient_boosting", "svm"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for name in model_names:
        logger.info(f"Training classifier: {name}...")
        try:
            if name == "random_forest":
                clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif name == "gradient_boosting":
                clf = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
            elif name == "svm":
                clf = SVC(kernel="rbf", random_state=random_state)
            else:
                logger.warning(f"Unknown classifier {name}, skipping")
                continue

            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()

            # Save
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, out / f"clf_{name}.joblib")
            joblib.dump(scaler, out / f"clf_{name}_scaler.joblib")
            joblib.dump(le, out / f"clf_label_encoder.joblib")
            with open(out / f"clf_{name}_features.txt", "w") as f:
                f.write("\n".join(available))

            results[name] = {
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "f1": float(f1),
                "confusion_matrix": cm,
                "classes": list(le.classes_),
            }
            logger.info(f"{name}: acc={acc:.3f}, f1={f1:.3f}")
        except Exception as e:
            logger.error(f"Failed to train classifier {name}: {e}")
            results[name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    from utils import load_config, setup_logging
    cfg = load_config()
    logger = setup_logging()
    # Load most recent simulation data
    data_path = Path("data/generated")
    csvs = sorted(data_path.glob("simulation_*.csv"))
    if csvs:
        df = pd.read_csv(csvs[-1])
        power_results = train_power_models(df, cfg)
        fault_results = train_fault_classifier(df, cfg)
        print("Power prediction results:", {k: {m: v for m, v in val.items()
                                                if m not in ("predictions", "y_test")}
                                            for k, val in power_results.items()})
        print("Fault classification results:", {k: {m: v for m, v in val.items()
                                                    if m != "confusion_matrix"}
                                                for k, val in fault_results.items()})
    else:
        logger.error("No simulation data found. Run simulation first.")
