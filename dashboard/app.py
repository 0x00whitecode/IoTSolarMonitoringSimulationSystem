"""Streamlit dashboard for the Smart Solar IoT Monitoring System.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from utils import load_config

st.set_page_config(page_title="Smart Solar IoT Monitoring", page_icon="☀️", layout="wide")


@st.cache_data
def load_data():
    data = {}
    gen_dir = PROJECT_ROOT / "data" / "generated"
    csvs = sorted(gen_dir.glob("simulation_*.csv"))
    if csvs:
        data["simulations"] = {f.stem: pd.read_csv(f) for f in csvs}
    tables_dir = PROJECT_ROOT / "results" / "tables"
    if tables_dir.exists():
        for f in tables_dir.glob("*.csv"):
            data[f.stem] = pd.read_csv(f)
    return data


def main():
    st.title("☀️ Smart Solar IoT Monitoring System")
    st.markdown("Real-time monitoring dashboard for IoT-based solar PV simulation")
    data = load_data()
    if not data:
        st.warning("No data found. Run `python3 main.py` first.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", "Real-Time Monitoring", "Analytics",
        "Fault Detection", "ML Prediction", "Communication", "Experiments"
    ])

    sim_key = list(data.get("simulations", {}).keys())[0] if data.get("simulations") else None
    df = data["simulations"][sim_key] if sim_key else pd.DataFrame()

    with tab1:
        st.header("System Overview")
        if df.empty:
            st.warning("No simulation data available.")
            return
        col1, col2, col3, col4 = st.columns(4)
        latest = df.iloc[-1]
        col1.metric("Current Voltage", f"{latest.get('measured_voltage', 0):.1f} V")
        col2.metric("Current Current", f"{latest.get('measured_current', 0):.2f} A")
        col3.metric("Current Power", f"{latest.get('measured_power', 0):.1f} W")
        col4.metric("Energy Generated", f"{latest.get('measured_energy', 0):.1f} Wh")
        col5, col6, col7 = st.columns(3)
        col5.metric("Irradiance", f"{latest.get('measured_irradiance', 0):.1f} W/m²")
        col6.metric("Temperature", f"{latest.get('measured_temperature', 0):.1f} °C")
        col7.metric("System Status", "✅ Normal")
        st.subheader("Recent Readings")
        st.dataframe(df.tail(10), use_container_width=True)

    with tab2:
        st.header("Real-Time Monitoring")
        if df.empty:
            return
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df else df.index
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("Power (W)", "Voltage (V)", "Current (A)"))
        fig.add_trace(go.Scatter(x=ts, y=df["measured_power"], name="Power"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts, y=df["measured_voltage"], name="Voltage"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ts, y=df["measured_current"], name="Current"), row=3, col=1)
        fig.update_layout(height=600, title="Real-Time Sensor Data")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Temperature (°C)", "Irradiance (W/m²)"))
        fig2.add_trace(go.Scatter(x=ts, y=df["measured_temperature"], name="Temp"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=ts, y=df["measured_irradiance"], name="Irr"), row=2, col=1)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.header("Analytics")
        if "pv_performance" in data:
            st.subheader("PV Performance Metrics")
            st.dataframe(data["pv_performance"], use_container_width=True)
        if "sensor_accuracy" in data:
            st.subheader("Sensor Accuracy")
            st.dataframe(data["sensor_accuracy"], use_container_width=True)
        if not df.empty:
            import plotly.express as px
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.scatter(df, x="measured_irradiance", y="measured_power",
                                title="Power vs Irradiance"), use_container_width=True)
            with col2:
                st.plotly_chart(px.scatter(df, x="measured_temperature", y="measured_power",
                                title="Power vs Temperature"), use_container_width=True)

    with tab4:
        st.header("Fault Detection")
        if "anomaly_detection" in data:
            st.dataframe(data["anomaly_detection"], use_container_width=True)
        if "fault_detection" in data:
            st.dataframe(data["fault_detection"], use_container_width=True)
        if not df.empty:
            from analytics.fault_detection import detect_faults
            faults = detect_faults(df)
            st.bar_chart(faults["fault_type"].value_counts())

    with tab5:
        st.header("ML Prediction")
        if "ml_model_comparison" in data:
            st.subheader("Model Comparison")
            st.dataframe(data["ml_model_comparison"], use_container_width=True)
        pred_dir = PROJECT_ROOT / "results" / "predictions"
        if pred_dir.exists():
            pred_csvs = sorted(pred_dir.glob("*.csv"))
            if pred_csvs:
                import plotly.express as px
                pdf = pd.read_csv(pred_csvs[0])
                st.plotly_chart(px.scatter(pdf, x="actual", y="predicted",
                                title="Actual vs Predicted Power"), use_container_width=True)

    with tab6:
        st.header("Communication")
        if "communication_performance" in data:
            st.dataframe(data["communication_performance"], use_container_width=True)

    with tab7:
        st.header("Experiments")
        if "scenario_comparison" in data:
            st.subheader("Scenario Comparison")
            st.dataframe(data["scenario_comparison"], use_container_width=True)
            import plotly.express as px
            st.plotly_chart(px.bar(data["scenario_comparison"], x="scenario",
                                  y="total_energy_wh", title="Energy Output by Scenario"),
                           use_container_width=True)
        if "traditional_vs_iot" in data:
            st.subheader("Traditional vs IoT")
            st.dataframe(data["traditional_vs_iot"], use_container_width=True)


if __name__ == "__main__":
    main()
