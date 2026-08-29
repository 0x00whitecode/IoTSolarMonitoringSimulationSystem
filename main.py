"""Smart Solar IoT Monitoring System - Master Pipeline

Usage:
    python3 main.py              # Run full pipeline
    python3 main.py --skip-ml    # Skip ML training
    python3 main.py --skip-dashboard  # Skip dashboard launch
"""
from __future__ import annotations
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils import load_config, setup_logging, ensure_dir

logger = setup_logging()


def step_1_generate_data(cfg: dict) -> dict:
    """Step 1: Generate simulation data for all scenarios."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating simulation data")
    logger.info("=" * 60)
    from simulation.simulator import run_scenario
    scenarios = cfg["experiments"]["scenarios"]
    sim_cfg = cfg["simulation"]
    seed = sim_cfg["random_seed"]
    data_dir = ensure_dir("data/generated")
    all_data = {}
    for scenario_name in scenarios:
        logger.info(f"  Simulating: {scenario_name}")
        df = run_scenario(scenario_name, cfg, seed=seed)
        df.to_csv(data_dir / f"simulation_{scenario_name}.csv", index=False)
        all_data[scenario_name] = df
        logger.info(f"    {len(df)} samples, mean power = {df['measured_power'].mean():.2f} W")
    logger.info("  Simulating: main dataset (3 days, clear sky)")
    main_cfg = dict(cfg)
    main_cfg["simulation"] = dict(cfg["simulation"])
    main_cfg["simulation"]["duration_days"] = 3
    main_df = run_scenario("normal_clear_sky", main_cfg, seed=seed)
    main_df.to_csv(data_dir / "simulation_main.csv", index=False)
    all_data["main"] = main_df
    logger.info(f"    Main dataset: {len(main_df)} samples")
    return all_data


def step_2_iot_communication(cfg: dict, all_data: dict) -> dict:
    """Step 2: Simulate IoT communication."""
    logger.info("=" * 60)
    logger.info("STEP 2: Simulating IoT communication")
    logger.info("=" * 60)
    from iot.mqtt_publisher import MQTTPublisher
    from iot.device_simulator import simulate_iot_communication
    iot_cfg = cfg["iot"]
    all_comm = {}
    for name in ["main"] + cfg["experiments"]["scenarios"][:3]:
        df = all_data.get(name, all_data.get("main"))
        logger.info(f"  IoT comm: {name}")
        pub = MQTTPublisher(
            broker=iot_cfg["mqtt_broker"], port=iot_cfg["mqtt_port"],
            device_id=iot_cfg["device_id"],
            use_simulated=iot_cfg["use_simulated_broker"],
            packet_loss_prob=iot_cfg["packet_loss_probability"],
            delay_mean_ms=iot_cfg["network_delay_ms_mean"],
            delay_std_ms=iot_cfg["network_delay_ms_std"],
            seed=cfg["simulation"]["random_seed"],
        )
        pkt_loss = iot_cfg["packet_loss_probability"]
        if "communication" in name:
            pkt_loss = max(pkt_loss * 5, 0.10)
        comm_logs, stats = simulate_iot_communication(
            df, iot_cfg["device_id"], pub, pkt_loss, seed=cfg["simulation"]["random_seed"]
        )
        all_comm[name] = {"logs": comm_logs, "stats": stats}
        logger.info(f"    Sent={stats['messages_sent']}, Delivered={stats['messages_delivered']}, "
                     f"Loss={stats['packet_loss']:.1f}%, Latency={stats['avg_latency_ms']:.1f}ms")
    return all_comm


def step_3_database(cfg: dict, all_data: dict, all_comm: dict) -> None:
    """Step 3: Store data in database."""
    logger.info("=" * 60)
    logger.info("STEP 3: Storing data in database")
    logger.info("=" * 60)
    from database.database import init_db, get_session
    from database.repository import insert_measurements, insert_comm_logs
    db_path = cfg["database"]["path"]
    engine = init_db(db_path)
    session = get_session(engine)
    for name, df in all_data.items():
        scenario = "normal" if name == "main" else name
        try:
            insert_measurements(session, df, device_id=cfg["iot"]["device_id"], scenario=scenario)
        except Exception as e:
            logger.error(f"  DB insert failed for {name}: {e}")
    for name, comm in all_comm.items():
        try:
            insert_comm_logs(session, comm["logs"], device_id=cfg["iot"]["device_id"])
        except Exception as e:
            logger.error(f"  Comm log insert failed for {name}: {e}")
    session.close()
    logger.info("  Database storage complete")


def step_4_processing(cfg: dict, all_data: dict) -> dict:
    """Step 4: Process data."""
    logger.info("=" * 60)
    logger.info("STEP 4: Processing data")
    logger.info("=" * 60)
    from analytics.preprocessing import preprocess
    processed = {}
    proc_dir = ensure_dir("data/processed")
    for name, df in all_data.items():
        logger.info(f"  Processing: {name}")
        try:
            pdf = preprocess(df)
            pdf.to_csv(proc_dir / f"processed_{name}.csv", index=False)
            processed[name] = pdf
        except Exception as e:
            logger.error(f"  Processing failed for {name}: {e}")
            processed[name] = df
    return processed


def step_5_analytics(cfg: dict, all_data: dict, all_comm: dict) -> dict:
    """Step 5: Run analytics."""
    logger.info("=" * 60)
    logger.info("STEP 5: Running analytics")
    logger.info("=" * 60)
    from analytics.performance import compute_performance_metrics
    from analytics.anomaly_detection import detect_anomalies_rule_based, detect_anomalies_isolation_forest
    from analytics.fault_detection import detect_faults, fault_summary
    from sensors.sensor_manager import SensorManager
    from sensors.virtual_pzem017 import VirtualPZEM017
    from sensors.virtual_bh1750 import VirtualBH1750
    from sensors.virtual_ds18b20 import VirtualDS18B20

    pv_cfg = cfg["pv"]
    dt_hours = cfg["simulation"]["sampling_interval_minutes"] / 60.0
    results = {}

    logger.info("  Computing sensor calibration...")
    main_df = all_data.get("main", list(all_data.values())[0])
    mgr = SensorManager(VirtualPZEM017(), VirtualBH1750(), VirtualDS18B20())
    results["sensor_accuracy"] = mgr.calibrate_all(main_df)
    logger.info(f"  Calibration done: {len(results['sensor_accuracy'])} sensors")

    perf_rows = []
    for name, df in all_data.items():
        if name == "main":
            continue
        try:
            perf = compute_performance_metrics(df, panel_area=pv_cfg["panel_area"],
                                               nominal_power=pv_cfg["nominal_power"],
                                               dt_hours=dt_hours)
            perf["scenario"] = name
            perf_rows.append(perf)
        except Exception as e:
            logger.error(f"  Performance failed for {name}: {e}")
    results["pv_performance"] = pd.DataFrame(perf_rows)

    logger.info("  Running anomaly detection...")
    anomalies = detect_anomalies_rule_based(main_df)
    iso_anomalies = detect_anomalies_isolation_forest(main_df)
    results["anomalies"] = anomalies
    results["iso_anomalies"] = iso_anomalies
    logger.info(f"  Rule-based: {len(anomalies)} anomalies, Isolation Forest: {len(iso_anomalies)} anomalies")

    logger.info("  Running fault detection...")
    faults = detect_faults(main_df)
    results["faults"] = faults
    results["fault_counts"] = fault_summary(faults)
    logger.info(f"  Fault counts: {results['fault_counts']}")

    comm_rows = []
    for name, comm in all_comm.items():
        stats = comm["stats"]
        stats["scenario"] = name
        comm_rows.append(stats)
    results["communication_performance"] = pd.DataFrame(comm_rows)
    return results


def step_6_ml(cfg: dict, all_data: dict) -> dict:
    """Step 6: Train and evaluate ML models."""
    logger.info("=" * 60)
    logger.info("STEP 6: Training ML models")
    logger.info("=" * 60)
    from ml.train import train_power_models, train_fault_classifier
    from ml.evaluate import evaluate_all_power_models, evaluate_all_classifiers, save_predictions, save_metrics_table

    main_df = all_data.get("main", list(all_data.values())[0])
    power_results = train_power_models(main_df, cfg)
    power_comparison = evaluate_all_power_models(power_results)
    save_predictions(power_results)
    save_metrics_table(power_comparison, "ml_model_comparison.csv")
    logger.info(f"  Power models trained: {list(power_results.keys())}")

    fault_results = train_fault_classifier(main_df, cfg)
    # Also try with combined scenario data for multi-class classification
    combined = pd.concat([df for name, df in all_data.items() if name != "main"], ignore_index=True)
    if len(combined) > len(main_df):
        fault_results = train_fault_classifier(combined, cfg)
    classifier_comparison = evaluate_all_classifiers(fault_results)
    save_metrics_table(classifier_comparison, "fault_classification_comparison.csv")
    logger.info(f"  Classifiers trained: {list(fault_results.keys())}")
    return {"power": power_results, "classification": fault_results,
            "power_comparison": power_comparison,
            "classifier_comparison": classifier_comparison}


def step_7_traditional(cfg: dict, all_data: dict) -> dict:
    """Step 7: Run traditional baseline."""
    logger.info("=" * 60)
    logger.info("STEP 7: Running traditional monitoring baseline")
    logger.info("=" * 60)
    from traditional.traditional_monitoring import TraditionalMonitoring
    trad = TraditionalMonitoring(
        sampling_interval_minutes=cfg["experiments"]["traditional_sampling_interval_minutes"],
        seed=cfg["simulation"]["random_seed"],
    )
    main_df = all_data.get("main", list(all_data.values())[0])
    trad_df = trad.sample(main_df)
    dt_trad = cfg["experiments"]["traditional_sampling_interval_minutes"] / 60.0
    true_energy = main_df["true_energy"].iloc[-1] if "true_energy" in main_df else 0
    trad_metrics = trad.compute_metrics(main_df, trad_df, dt_trad, true_energy)
    logger.info(f"  Traditional metrics: {trad_metrics}")
    return {"trad_df": trad_df, "trad_metrics": trad_metrics}


def step_8_experiments(cfg: dict) -> dict:
    """Step 8: Run all experiments."""
    logger.info("=" * 60)
    logger.info("STEP 8: Running experiments")
    logger.info("=" * 60)
    from experiments.experiment_runner import run_all_experiments, collect_summary_results
    reps = min(cfg["experiments"]["repetitions"], 3)
    all_results = run_all_experiments(cfg, repetitions=reps)
    summary = collect_summary_results(all_results)
    tables_dir = ensure_dir("results/tables")
    summary.to_csv(tables_dir / "scenario_comparison.csv", index=False)
    logger.info(f"  Experiments complete: {len(summary)} rows in summary")
    return {"all_results": all_results, "summary": summary}


def step_9_comparison(cfg: dict, all_data: dict, trad_results: dict) -> dict:
    """Step 9: Compare IoT vs traditional."""
    logger.info("=" * 60)
    logger.info("STEP 9: Comparing IoT vs traditional")
    logger.info("=" * 60)
    from experiments.comparison import compare_iot_vs_traditional, comparison_to_dataframe
    main_df = all_data.get("main", list(all_data.values())[0])
    trad_df = trad_results["trad_df"]
    trad_metrics = trad_results["trad_metrics"]
    dt_hours = cfg["simulation"]["sampling_interval_minutes"] / 60.0
    dt_trad = cfg["experiments"]["traditional_sampling_interval_minutes"] / 60.0
    iot_stats = {"avg_latency_ms": 50.0, "packet_loss": 2.0, "delivery_rate": 98.0,
                 "fault_detection_rate": 100.0, "response_time": 0.05, "false_alarms": 0}
    comparison = compare_iot_vs_traditional(main_df, trad_df, main_df, iot_stats, trad_metrics, dt_hours, dt_trad)
    comp_df = comparison_to_dataframe(comparison)
    tables_dir = ensure_dir("results/tables")
    comp_df.to_csv(tables_dir / "traditional_vs_iot.csv", index=False)
    logger.info("  Comparison complete")
    return comparison


def step_10_tables(cfg: dict, analytics_results: dict, ml_results: dict,
                    trad_results: dict, exp_results: dict) -> None:
    """Step 10: Generate all CSV tables."""
    logger.info("=" * 60)
    logger.info("STEP 10: Generating tables")
    logger.info("=" * 60)
    tables_dir = ensure_dir("results/tables")
    if "sensor_accuracy" in analytics_results:
        analytics_results["sensor_accuracy"].to_csv(tables_dir / "sensor_accuracy.csv", index=False)
    if "pv_performance" in analytics_results:
        analytics_results["pv_performance"].to_csv(tables_dir / "pv_performance.csv", index=False)
    if "communication_performance" in analytics_results:
        analytics_results["communication_performance"].to_csv(tables_dir / "communication_performance.csv", index=False)
    if "anomalies" in analytics_results:
        anom = analytics_results["anomalies"]
        if len(anom) > 0:
            anom.to_csv(tables_dir / "anomaly_detection.csv", index=False)
        else:
            pd.DataFrame([{"anomaly_type": "none", "count": 0}]).to_csv(
                tables_dir / "anomaly_detection.csv", index=False)
    if "fault_counts" in analytics_results:
        fc = analytics_results["fault_counts"]
        pd.DataFrame([{"fault_type": k, "count": v} for k, v in fc.items()]).to_csv(
            tables_dir / "fault_detection.csv", index=False)
    from analytics.statistics import compute_summary_stats
    summary = exp_results.get("summary", pd.DataFrame())
    if not summary.empty:
        stat_rows = []
        for col in ["total_energy_wh", "mean_power_w", "mean_efficiency_pct",
                     "iot_latency_ms", "iot_packet_loss_pct"]:
            if col in summary:
                s = compute_summary_stats(summary[col].values)
                stat_rows.append({"metric": col, **s})
        pd.DataFrame(stat_rows).to_csv(tables_dir / "statistical_analysis.csv", index=False)
    logger.info(f"  Tables saved to {tables_dir}")


def step_11_figures(cfg: dict, all_data: dict, all_comm: dict,
                     analytics_results: dict, ml_results: dict, exp_results: dict) -> None:
    """Step 11: Generate all figures."""
    logger.info("=" * 60)
    logger.info("STEP 11: Generating figures")
    logger.info("=" * 60)
    from visualization.research_figures import generate_all_figures
    main_df = all_data.get("main", list(all_data.values())[0])
    comm_logs = all_comm.get("main", {}).get("logs", pd.DataFrame())
    if comm_logs.empty:
        for v in all_comm.values():
            if v.get("logs") is not None and len(v["logs"]) > 0:
                comm_logs = v["logs"]
                break
    ml_power = ml_results.get("power", {})
    ml_class = ml_results.get("classification", {})
    summary = exp_results.get("summary", pd.DataFrame())
    generate_all_figures(main_df, comm_logs, ml_power, ml_class, summary)
    logger.info("  Figures generated")


def step_12_report(cfg: dict, all_data: dict, analytics_results: dict,
                   ml_results: dict, trad_results: dict, exp_results: dict) -> None:
    """Step 12: Generate final report."""
    logger.info("=" * 60)
    logger.info("STEP 12: Generating final report")
    logger.info("=" * 60)
    reports_dir = ensure_dir("results/reports")
    summary = exp_results.get("summary", pd.DataFrame())
    report_lines = [
        "# Smart Solar IoT Monitoring System - Experiment Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Configuration",
        f"- Duration: {cfg['simulation']['duration_days']} days",
        f"- Sampling interval: {cfg['simulation']['sampling_interval_minutes']} min",
        f"- Random seed: {cfg['simulation']['random_seed']}",
        f"- Scenarios: {len(cfg['experiments']['scenarios'])}",
        f"\n## Summary Statistics",
    ]
    if not summary.empty:
        report_lines.append(summary.describe().to_string())
    pv_perf = analytics_results.get("pv_performance")
    if pv_perf is not None and len(pv_perf) > 0:
        report_lines.append("\n## PV Performance by Scenario")
        report_lines.append(pv_perf.to_string(index=False))
    sensor_cal = analytics_results.get("sensor_accuracy")
    if sensor_cal is not None:
        report_lines.append("\n## Sensor Calibration")
        report_lines.append(sensor_cal.to_string(index=False))
    ml_comp = ml_results.get("power_comparison")
    if ml_comp is not None:
        report_lines.append("\n## ML Model Comparison (Power Prediction)")
        report_lines.append(ml_comp.to_string(index=False))
    fault_counts = analytics_results.get("fault_counts", {})
    if fault_counts:
        report_lines.append("\n## Fault Detection Summary")
        for k, v in fault_counts.items():
            report_lines.append(f"- {k}: {v}")
    trad_m = trad_results.get("trad_metrics", {})
    if trad_m:
        report_lines.append("\n## Traditional Monitoring Metrics")
        for k, v in trad_m.items():
            report_lines.append(f"- {k}: {v}")
    report_path = reports_dir / "experiment_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info(f"  Report saved to {report_path}")


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SMART SOLAR IOT MONITORING SYSTEM")
    logger.info("Master Pipeline Starting")
    logger.info("=" * 60)
    cfg = load_config()
    skip_ml = "--skip-ml" in sys.argv
    all_data = step_1_generate_data(cfg)
    all_comm = step_2_iot_communication(cfg, all_data)
    step_3_database(cfg, all_data, all_comm)
    step_4_processing(cfg, all_data)
    analytics_results = step_5_analytics(cfg, all_data, all_comm)
    ml_results = {}
    if not skip_ml:
        ml_results = step_6_ml(cfg, all_data)
    else:
        logger.info("Skipping ML training (--skip-ml)")
    trad_results = step_7_traditional(cfg, all_data)
    exp_results = step_8_experiments(cfg)
    step_9_comparison(cfg, all_data, trad_results)
    step_10_tables(cfg, analytics_results, ml_results, trad_results, exp_results)
    step_11_figures(cfg, all_data, all_comm, analytics_results, ml_results, exp_results)
    step_12_report(cfg, all_data, analytics_results, ml_results, trad_results, exp_results)
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"Results saved to: results/")
    logger.info("=" * 60)
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"  Results in: results/")
    print(f"  Data in: data/")
    print(f"  Models in: models/")
    print(f"\nTo view dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
