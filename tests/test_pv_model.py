"""Unit tests for PV model calculations."""
import numpy as np
from simulation.pv_model import PVModel, PVConfig


def test_power_calculation():
    pv = PVModel(PVConfig())
    G = np.array([1000.0, 500.0])
    T = np.array([25.0, 25.0])
    power = pv.compute_power(G, T)
    assert len(power) == 2
    assert power[0] > power[1]


def test_efficiency_calculation():
    pv = PVModel(PVConfig(panel_area=1.7, nominal_power=400))
    G = np.array([1000.0])
    T = np.array([25.0])
    power = pv.compute_power(G, T)
    eff = pv.compute_efficiency(power, G)
    assert eff[0] > 0
    assert eff[0] < 100


def test_temperature_effect():
    pv = PVModel(PVConfig(temperature_coefficient=-0.0035))
    G = np.array([1000.0, 1000.0])
    T = np.array([25.0, 60.0])
    power = pv.compute_power(G, T)
    assert power[1] < power[0]


def test_irradiance_effect():
    pv = PVModel(PVConfig())
    G = np.array([200.0, 800.0, 1000.0])
    T = np.array([25.0, 25.0, 25.0])
    power = pv.compute_power(G, T)
    assert power[0] < power[1] < power[2]


def test_energy_accumulation():
    pv = PVModel(PVConfig())
    power = np.array([100.0, 200.0, 300.0])
    energy = pv.compute_energy(power, dt_hours=0.5)
    assert energy[0] == 50.0
    assert energy[1] == 150.0
    assert energy[2] == 300.0


def test_current_from_power_voltage():
    pv = PVModel(PVConfig())
    power = np.array([400.0])
    voltage = np.array([40.0])
    current = pv.compute_current(power, voltage)
    assert abs(current[0] - 10.0) < 0.01
