"""Unit tests for virtual sensors."""
import numpy as np
from sensors.virtual_pzem017 import VirtualPZEM017
from sensors.virtual_bh1750 import VirtualBH1750
from sensors.virtual_ds18b20 import VirtualDS18B20
from sensors.sensor_manager import SensorManager


def test_pzem017_voltage_noise():
    pzem = VirtualPZEM017(voltage_noise=0.1, seed=42)
    true_v = np.array([40.0] * 100)
    meas_v = pzem.measure_voltage(true_v)
    assert abs(np.mean(meas_v) - 40.0) < 1.0
    assert np.std(meas_v) < 1.0


def test_pzem017_current_noise():
    pzem = VirtualPZEM017(current_noise=0.01, seed=42)
    true_i = np.array([10.0] * 100)
    meas_i = pzem.measure_current(true_i)
    assert abs(np.mean(meas_i) - 10.0) < 0.5


def test_bh1750_irradiance():
    bh = VirtualBH1750(irradiance_noise=5.0, seed=42)
    true_g = np.array([800.0] * 100)
    meas_g = bh.measure(true_g)
    assert abs(np.mean(meas_g) - 800.0) < 10.0


def test_ds18b20_temperature():
    ds = VirtualDS18B20(temperature_noise=0.5, seed=42)
    true_t = np.array([35.0] * 100)
    meas_t = ds.measure(true_t)
    assert abs(np.mean(meas_t) - 35.0) < 1.0


def test_sensor_calibration_metrics():
    mgr = SensorManager(VirtualPZEM017(), VirtualBH1750(), VirtualDS18B20())
    true = np.array([100, 200, 300], dtype=float)
    meas = np.array([101, 199, 305], dtype=float)
    m = mgr.calibration_metrics(true, meas)
    assert "mae" in m and "rmse" in m and "bias" in m
    assert m["mae"] > 0


def test_sensor_validation():
    pzem = VirtualPZEM017()
    assert pzem.validate_reading(40.0, 10.0) == "ok"
    assert pzem.validate_reading(400.0, 10.0) == "voltage_out_of_range"
    bh = VirtualBH1750()
    assert bh.validate_reading(800.0) == "ok"
    assert bh.validate_reading(2000.0) == "irradiance_exceeds_physical_max"
    ds = VirtualDS18B20()
    assert ds.validate_reading(35.0) == "ok"
    assert ds.validate_reading(200.0) == "temperature_out_of_range"
