"""Visualization utilities for generating plots.

All figures use consistent academic styling and are saved to
results/figures/ with descriptive filenames.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Consistent academic style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_palette("muted")


def setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def save_fig(fig, filename: str, output_dir: str = "results/figures"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out / filename)


def plot_time_series(df: pd.DataFrame, col: str, title: str, ylabel: str,
                     filename: str, output_dir: str = "results/figures") -> str:
    """Plot a time series."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df else df.index
    ax.plot(ts, df[col], linewidth=1.2, color="#2b6cb0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    fig.autofmt_xdate()
    return save_fig(fig, filename, output_dir)


def plot_irradiance_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_irradiance",
                            "Fig. 1: Solar Irradiance vs Time",
                            "Irradiance (W/m²)", "fig01_irradiance_vs_time.png", output_dir)


def plot_temperature_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_temperature",
                            "Fig. 2: Panel Temperature vs Time",
                            "Temperature (°C)", "fig02_temperature_vs_time.png", output_dir)


def plot_voltage_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_voltage",
                            "Fig. 3: DC Voltage vs Time",
                            "Voltage (V)", "fig03_voltage_vs_time.png", output_dir)


def plot_current_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_current",
                            "Fig. 4: DC Current vs Time",
                            "Current (A)", "fig04_current_vs_time.png", output_dir)


def plot_power_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_power",
                            "Fig. 5: Power Output vs Time",
                            "Power (W)", "fig05_power_vs_time.png", output_dir)


def plot_energy_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    return plot_time_series(df, "measured_energy",
                            "Fig. 6: Cumulative Energy vs Time",
                            "Energy (Wh)", "fig06_energy_vs_time.png", output_dir)


def plot_efficiency_vs_time(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    eff = df.get("true_efficiency", df.get("efficiency_pct"))
    if eff is None:
        return ""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df else df.index
    ax.plot(ts, eff, linewidth=1.2, color="#38a169")
    ax.set_title("Fig. 7: PV Efficiency vs Time")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Time")
    fig.autofmt_xdate()
    return save_fig(fig, "fig07_efficiency_vs_time.png", output_dir)


def plot_power_vs_irradiance(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["measured_irradiance"], df["measured_power"], alpha=0.5, s=15, color="#2b6cb0")
    ax.set_title("Fig. 8: Power Output vs Irradiance")
    ax.set_xlabel("Irradiance (W/m²)")
    ax.set_ylabel("Power (W)")
    return save_fig(fig, "fig08_power_vs_irradiance.png", output_dir)


def plot_efficiency_vs_temperature(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    eff = df.get("true_efficiency", df.get("efficiency_pct"))
    if eff is None:
        return ""
    ax.scatter(df["measured_temperature"], eff, alpha=0.5, s=15, color="#dd6b20")
    ax.set_title("Fig. 9: Efficiency vs Temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Efficiency (%)")
    return save_fig(fig, "fig09_efficiency_vs_temperature.png", output_dir)


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    from analytics.performance import correlation_analysis
    setup_style()
    corr = correlation_analysis(df)
    if corr.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Fig. 10: Correlation Matrix")
    return save_fig(fig, "fig10_correlation_matrix.png", output_dir)


def plot_sensor_error_distribution(df: pd.DataFrame, output_dir: str = "results/figures") -> str:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    pairs = [
        ("true_irradiance", "measured_irradiance", "Irradiance"),
        ("true_panel_temp", "measured_temperature", "Temperature"),
        ("true_voltage", "measured_voltage", "Voltage"),
        ("true_current", "measured_current", "Current"),
        ("true_power", "measured_power", "Power"),
    ]
    for idx, (true_col, meas_col, label) in enumerate(pairs):
        if true_col not in df or meas_col not in df:
            continue
        ax = axes[idx // 3, idx % 3]
        error = df[meas_col] - df[true_col]
        ax.hist(error.dropna(), bins=30, edgecolor="black", color="#3182ce", alpha=0.7)
        ax.set_title(f"{label} Error")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
    axes[1, 2].axis("off")
    fig.suptitle("Fig. 11: Sensor Error Distributions", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "fig11_sensor_error_distribution.png", output_dir)


def plot_mqtt_latency(comm_logs: pd.DataFrame, output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    lat = comm_logs["latency_ms"].dropna()
    if len(lat) == 0:
        lat = pd.Series([0])
    ax.hist(lat, bins=40, edgecolor="black", color="#805ad5", alpha=0.7)
    ax.set_title("Fig. 12: MQTT Latency Distribution")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    return save_fig(fig, "fig12_mqtt_latency_distribution.png", output_dir)


def plot_packet_loss(comm_logs: pd.DataFrame, output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    status_counts = comm_logs["status"].value_counts()
    ax.bar(status_counts.index, status_counts.values, color=["#48bb78", "#f56565"])
    ax.set_title("Fig. 13: Packet Delivery Analysis")
    ax.set_ylabel("Message Count")
    ax.set_xlabel("Status")
    return save_fig(fig, "fig13_packet_loss_analysis.png", output_dir)


def plot_prediction_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                                        model_name: str,
                                        output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15, color="#2b6cb0")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_title(f"Fig. 14: Power Prediction - Actual vs Predicted ({model_name})")
    ax.set_xlabel("Actual Power (W)")
    ax.set_ylabel("Predicted Power (W)")
    return save_fig(fig, f"fig14_prediction_actual_vs_predicted_{model_name}.png", output_dir)


def plot_prediction_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str,
                               output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    residuals = np.array(y_true) - np.array(y_pred)
    ax.scatter(range(len(residuals)), residuals, alpha=0.5, s=10, color="#e53e3e")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Fig. 15: Prediction Residuals ({model_name})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (W)")
    return save_fig(fig, f"fig15_prediction_residuals_{model_name}.png", output_dir)


def plot_confusion_matrix(cm: list, classes: list, model_name: str,
                          output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"Fig. 16: Confusion Matrix ({model_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return save_fig(fig, f"fig16_confusion_matrix_{model_name}.png", output_dir)


def plot_fault_detection_results(faults_df: pd.DataFrame,
                                  output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    if "fault_type" not in faults_df:
        return ""
    counts = faults_df["fault_type"].value_counts()
    ax.bar(counts.index, counts.values, color="#e53e3e")
    ax.set_title("Fig. 17: Fault Detection Results")
    ax.set_ylabel("Count")
    ax.set_xlabel("Fault Type")
    plt.xticks(rotation=30, ha="right")
    return save_fig(fig, "fig17_fault_detection_results.png", output_dir)


def plot_traditional_vs_iot(comparison_df: pd.DataFrame,
                             output_dir: str = "results/figures") -> str:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("Data Availability (%)", "data_availability_pct", "traditional_data_availability_pct"),
        ("Fault Detection Rate (%)", "iot_fault_detection_rate", "traditional_fault_detection_rate"),
        ("Energy Error (%)", "iot_energy_error", "traditional_energy_error_pct"),
    ]
    for ax, (title, iot_col, trad_col) in zip(axes, metrics):
        if iot_col in comparison_df and trad_col in comparison_df:
            ax.bar(["IoT", "Traditional"], [comparison_df[iot_col].mean(),
                                              comparison_df[trad_col].mean()],
                   color=["#3182ce", "#a0aec0"])
            ax.set_title(title)
    fig.suptitle("Fig. 18: Traditional vs IoT Monitoring Comparison", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "fig18_traditional_vs_iot.png", output_dir)


def plot_scenario_comparison(summary_df: pd.DataFrame,
                               output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = summary_df["scenario"].unique()
    means = summary_df.groupby("scenario")["total_energy_wh"].mean()
    stds = summary_df.groupby("scenario")["total_energy_wh"].std()
    ax.bar(scenarios, means, yerr=stds, capsize=5, color="#2b6cb0", alpha=0.8)
    ax.set_title("Fig. 19: Energy Output by Scenario")
    ax.set_ylabel("Total Energy (Wh)")
    ax.set_xlabel("Scenario")
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    return save_fig(fig, "fig19_scenario_comparison.png", output_dir)


def plot_system_performance_comparison(summary_df: pd.DataFrame,
                                         output_dir: str = "results/figures") -> str:
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["mean_efficiency_pct", "performance_ratio", "iot_delivery_rate_pct"]
    x = np.arange(len(summary_df["scenario"].unique()))
    width = 0.25
    for i, m in enumerate(metrics):
        vals = summary_df.groupby("scenario")[m].mean()
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(summary_df["scenario"].unique(), rotation=40, ha="right")
    ax.set_title("Fig. 20: Overall System Performance Comparison")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return save_fig(fig, "fig20_system_performance_comparison.png", output_dir)
