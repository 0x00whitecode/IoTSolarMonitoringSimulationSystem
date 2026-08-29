"""Research figure generation: orchestrates all publication-quality figures.

Usage:
    python3 -m visualization.research_figures
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict

from visualization.plots import (
    plot_irradiance_vs_time, plot_temperature_vs_time, plot_voltage_vs_time,
    plot_current_vs_time, plot_power_vs_time, plot_energy_vs_time,
    plot_efficiency_vs_time, plot_power_vs_irradiance, plot_efficiency_vs_temperature,
    plot_correlation_matrix, plot_sensor_error_distribution,
    plot_mqtt_latency, plot_packet_loss,
    plot_prediction_actual_vs_predicted, plot_prediction_residuals,
    plot_confusion_matrix, plot_fault_detection_results,
    plot_traditional_vs_iot, plot_scenario_comparison,
    plot_system_performance_comparison,
)

logger = logging.getLogger("smart_solar_iot.research_figures")


def generate_all_figures(df: pd.DataFrame, comm_logs: pd.DataFrame,
                          ml_results: Dict, classifier_results: Dict,
                          summary_df: pd.DataFrame,
                          output_dir: str = "results/figures") -> list[str]:
    """Generate all research figures and return list of saved paths."""
    saved = []

    for fn in [plot_irradiance_vs_time, plot_temperature_vs_time,
              plot_voltage_vs_time, plot_current_vs_time,
              plot_power_vs_time, plot_energy_vs_time,
              plot_efficiency_vs_time]:
        try:
            path = fn(df, output_dir)
            if path:
                saved.append(path)
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")

    for fn in [plot_power_vs_irradiance, plot_efficiency_vs_temperature]:
        try:
            saved.append(fn(df, output_dir))
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")

    try:
        saved.append(plot_correlation_matrix(df, output_dir))
    except Exception as e:
        logger.error(f"Correlation matrix failed: {e}")

    try:
        saved.append(plot_sensor_error_distribution(df, output_dir))
    except Exception as e:
        logger.error(f"Sensor error distribution failed: {e}")

    if comm_logs is not None and len(comm_logs) > 0:
        try:
            saved.append(plot_mqtt_latency(comm_logs, output_dir))
        except Exception as e:
            logger.error(f"MQTT latency plot failed: {e}")
        try:
            saved.append(plot_packet_loss(comm_logs, output_dir))
        except Exception as e:
            logger.error(f"Packet loss plot failed: {e}")

    for name, res in ml_results.items():
        if "predictions" not in res or "y_test" not in res:
            continue
        try:
            y_pred = np.array(res["predictions"])
            y_true = np.array(res["y_test"])
            saved.append(plot_prediction_actual_vs_predicted(y_true, y_pred, name, output_dir))
            saved.append(plot_prediction_residuals(y_true, y_pred, name, output_dir))
        except Exception as e:
            logger.error(f"Prediction plot failed for {name}: {e}")

    for name, res in classifier_results.items():
        if "confusion_matrix" not in res:
            continue
        try:
            saved.append(plot_confusion_matrix(res["confusion_matrix"],
                                               res.get("classes", []), name, output_dir))
        except Exception as e:
            logger.error(f"Confusion matrix failed for {name}: {e}")

    try:
        from analytics.fault_detection import detect_faults
        faults = detect_faults(df)
        saved.append(plot_fault_detection_results(faults, output_dir))
    except Exception as e:
        logger.error(f"Fault detection plot failed: {e}")

    if summary_df is not None and not summary_df.empty:
        try:
            saved.append(plot_traditional_vs_iot(summary_df, output_dir))
        except Exception as e:
            logger.error(f"Traditional vs IoT plot failed: {e}")
        try:
            saved.append(plot_scenario_comparison(summary_df, output_dir))
        except Exception as e:
            logger.error(f"Scenario comparison plot failed: {e}")
        try:
            saved.append(plot_system_performance_comparison(summary_df, output_dir))
        except Exception as e:
            logger.error(f"System performance plot failed: {e}")

    logger.info(f"Generated {len(saved)} figures in {output_dir}")
    return saved


if __name__ == "__main__":
    from utils import load_config, setup_logging
    cfg = load_config()
    logger = setup_logging()
    data_path = Path("data/generated")
    csvs = sorted(data_path.glob("simulation_*.csv"))
    if csvs:
        df = pd.read_csv(csvs[-1])
        generate_all_figures(df, pd.DataFrame(), {}, {}, pd.DataFrame())
    else:
        logger.error("No simulation data found.")
