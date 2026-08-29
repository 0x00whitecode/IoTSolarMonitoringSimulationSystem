"""Solar environment simulator.

Generates realistic solar irradiance, ambient temperature, panel temperature,
and weather/cloud/shading effects for a given location and time period.

Uses pvlib for solar position calculations. All randomness is seeded for
reproducibility.

Equations / assumptions:
  - Clear-sky irradiance from pvlib's Ineichen model (physically-based,
    accounts for atmosphere, altitude, and solar zenith angle).
  - Ambient temperature: sinusoidal daily profile with configurable
    amplitude and offset, plus stochastic noise.
  - Panel temperature: T_panel = T_ambient + (NOCT - 20)/800 * G
    (standard NOCT-based model, IEC 61215).
  - Cloud effects: multiplicative reduction factor on irradiance, drawn
    from a Beta distribution to mimic intermittent cloud cover.
  - Shading: configurable fraction of panel area blocked, reducing
    effective irradiance proportionally.
  - Dust: progressive linear degradation factor applied to effective
    irradiance over time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pvlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConfig:
    latitude: float = 7.3775
    longitude: float = -3.9779
    altitude_m: float = 300.0
    timezone: str = "UTC"
    start_date: str = "2024-06-15"
    duration_days: int = 1
    sampling_interval_minutes: int = 5
    random_seed: int = 42
    ambient_temp_mean: float = 28.0
    ambient_temp_amplitude: float = 8.0
    noct: float = 45.0  # Nominal Operating Cell Temperature
    cloud_cover_mean: float = 0.0
    cloud_cover_var: float = 0.0
    shading_factor: float = 0.0
    dust_loss_initial: float = 0.0
    dust_loss_rate: float = 0.001  # per day
    wind_speed_mean: float = 2.0  # m/s


class SolarEnvironment:
    """Simulates solar environmental variables over a time period."""

    def __init__(self, cfg: EnvironmentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.times = self._generate_time_index()
        self.solar_pos = self._compute_solar_position()
        self.clear_sky_irradiance = self._compute_clear_sky()

    def _generate_time_index(self) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.cfg.start_date, tz=self.cfg.timezone)
        periods = int(self.cfg.duration_days * 24 * 60 / self.cfg.sampling_interval_minutes)
        return pd.date_range(start=start, periods=periods,
                             freq=f"{self.cfg.sampling_interval_minutes}min")

    def _compute_solar_position(self) -> pd.DataFrame:
        return pvlib.solarposition.get_solarposition(
            self.times,
            latitude=self.cfg.latitude,
            longitude=self.cfg.longitude,
            altitude=self.cfg.altitude_m,
            temperature=self.cfg.ambient_temp_mean,
            pressure=101325.0,
        )

    def _compute_clear_sky(self) -> pd.DataFrame:
        """Use Ineichen clear-sky model for physically-based GHI, DNI, DHI."""
        apparent_zenith = self.solar_pos["apparent_zenith"]
        airmass_absolute = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
        cs = pvlib.clearsky.ineichen(
            apparent_zenith,
            airmass_absolute,
            linke_turbidity=3.0,
            altitude=self.cfg.altitude_m,
        )
        return cs

    def _ambient_temperature(self) -> np.ndarray:
        """Sinusoidal daily temperature profile + noise."""
        hours = self.times.hour + self.times.minute / 60.0
        # Peak at ~14:00 local
        temp = (self.cfg.ambient_temp_mean
                + self.cfg.ambient_temp_amplitude * np.sin(2 * np.pi * (hours - 8) / 24.0))
        noise = self.rng.normal(0, 1.0, len(hours))
        return temp + noise

    def _cloud_factor(self) -> np.ndarray:
        """Multiplicative cloud reduction factor in [0, 1].

        When cloud_cover_mean is 0, returns all ones (clear sky).
        Uses Beta distribution for realistic intermittency.
        """
        n = len(self.times)
        if self.cfg.cloud_cover_mean <= 0:
            return np.ones(n)
        # cloud_cover_mean is the mean reduction (0=clear, 1=fully overcast)
        alpha = max(self.cfg.cloud_cover_mean * 10, 0.5)
        beta = max((1 - self.cfg.cloud_cover_mean) * 10, 0.5)
        cloud = self.rng.beta(alpha, beta, n)
        # Smooth with a rolling mean to avoid per-timestep jitter
        series = pd.Series(cloud).rolling(window=5, min_periods=1, center=True).mean()
        return (1.0 - series.values).clip(0.1, 1.0)

    def _dust_factor(self) -> np.ndarray:
        """Progressive linear dust degradation over the simulation period.

        dust_factor(t) = 1 - dust_loss_initial - dust_loss_rate * t_days
        Clipped to a minimum of 0.5 (50% output).
        """
        days_elapsed = np.arange(len(self.times)) * self.cfg.sampling_interval_minutes / (24 * 60)
        factor = 1.0 - self.cfg.dust_loss_initial - self.cfg.dust_loss_rate * days_elapsed
        return np.clip(factor, 0.5, 1.0)

    def generate(self) -> pd.DataFrame:
        """Generate the full environmental dataset.

        Returns a DataFrame with columns:
          timestamp, ghi, dni, dhi, ambient_temp, panel_temp,
          wind_speed, cloud_factor, dust_factor, shading_factor,
          effective_irradiance, solar_zenith, solar_azimuth
        """
        ghi_clear = self.clear_sky_irradiance["ghi"].fillna(0).values
        dni_clear = self.clear_sky_irradiance["dni"].fillna(0).values
        dhi_clear = self.clear_sky_irradiance["dhi"].fillna(0).values

        cloud = self._cloud_factor()
        dust = self._dust_factor()
        shading = np.full(len(self.times), self.cfg.shading_factor)

        ghi = ghi_clear * cloud * dust * (1.0 - shading)
        dhi = dhi_clear * cloud * dust * (1.0 - shading)
        # DNI is more affected by shading (direct beam)
        dni = dni_clear * cloud * (1.0 - shading * 0.8) * dust

        ambient = self._ambient_temperature()
        wind = self.rng.normal(self.cfg.wind_speed_mean, 0.5, len(self.times)).clip(0, 15)

        # Panel temperature via NOCT model:
        # T_panel = T_ambient + (NOCT - 20) / 800 * G
        # With wind correction: subtract ~1°C per m/s above 1 m/s
        panel_temp = ambient + (self.cfg.noct - 20.0) / 800.0 * ghi
        panel_temp -= 0.5 * np.clip(wind - 1.0, 0, None)

        # Effective irradiance accounts for shading on the panel surface
        effective_irr = ghi  # already includes cloud, dust, shading

        df = pd.DataFrame({
            "timestamp": self.times,
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "ambient_temp": ambient,
            "panel_temp": panel_temp,
            "wind_speed": wind,
            "cloud_factor": cloud,
            "dust_factor": dust,
            "shading_factor": shading,
            "effective_irradiance": effective_irr,
            "solar_zenith": self.solar_pos["zenith"].values,
            "solar_azimuth": self.solar_pos["azimuth"].values,
        })
        return df

    @property
    def n_samples(self) -> int:
        return len(self.times)
