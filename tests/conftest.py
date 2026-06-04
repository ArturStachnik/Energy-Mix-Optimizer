"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_mix_optimizer.data.locations import IBERIAN_PANEL


@pytest.fixture
def utc_now() -> datetime:
    return datetime(2024, 6, 15, 0, 0, tzinfo=UTC)


@pytest.fixture
def hourly_index(utc_now: datetime) -> pd.DatetimeIndex:
    return pd.date_range(
        start=utc_now - timedelta(days=10),
        end=utc_now + timedelta(days=1),
        freq="1h",
        tz="UTC",
    )


@pytest.fixture
def synthetic_weather_long(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a small weather table covering all panel points."""
    rng = np.random.default_rng(seed=0)
    rows: list[pd.DataFrame] = []
    n = len(hourly_index)
    hour = hourly_index.hour.to_numpy()
    daily_solar = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    for point in IBERIAN_PANEL:
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": hourly_index,
                    "location": point.name,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "temperature_2m": 18.0 + rng.normal(0, 2, n),
                    "wind_speed_10m": 4.0 + rng.normal(0, 1, n).clip(min=0),
                    "wind_speed_100m": 7.0 + rng.normal(0, 1.5, n).clip(min=0),
                    "shortwave_radiation": 800.0 * daily_solar + rng.normal(0, 30, n),
                    "direct_radiation": 600.0 * daily_solar + rng.normal(0, 20, n),
                    "diffuse_radiation": 200.0 * daily_solar + rng.normal(0, 10, n),
                    "cloud_cover": (40.0 + rng.normal(0, 15, n)).clip(0, 100),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def synthetic_target(hourly_index: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(seed=1)
    hour = hourly_index.hour.to_numpy()
    daily = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    values = 15_000.0 * daily + rng.normal(0, 200, len(hourly_index))
    values = np.clip(values, 0, None)
    return pd.Series(values, index=hourly_index, name="solar")


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def energy_charts_payload() -> dict:
    """Minimal but realistic Energy-Charts ``/public_power`` payload for Spain.

    The Energy-Charts API returns one ``unix_seconds`` array and one entry
    per production type, plus a ``Load`` series for demand. We model a few
    hours so that resampling to hourly produces clean values.
    """
    # 4 hourly timestamps starting 2024-06-15 00:00 UTC.
    base = 1718409600  # 2024-06-15T00:00:00+00:00 in unix seconds
    return {
        "unix_seconds": [base, base + 3600, base + 7200, base + 10800],
        "production_types": [
            {"name": "Solar", "data": [0.0, 0.0, 100.0, 800.0]},
            {"name": "Wind onshore", "data": [8000.0, 8200.0, 8400.0, 8100.0]},
            {"name": "Nuclear", "data": [5800.0, 5800.0, 5800.0, 5800.0]},
            {"name": "Fossil gas", "data": [3000.0, 2900.0, 2700.0, 2500.0]},
            {"name": "Hydro Run-of-River", "data": [1200.0, 1180.0, 1190.0, 1210.0]},
            {"name": "Hydro water reservoir", "data": [800.0, 820.0, 810.0, 840.0]},
            {"name": "Load", "data": [28000.0, 28500.0, 29000.0, 29500.0]},
            # These should be ignored / collapsed by the parser.
            {"name": "Cross border electricity trading", "data": [-500.0, -400.0, -300.0, -200.0]},
            {"name": "Residual load", "data": [12000.0, 12500.0, 13000.0, 13500.0]},
        ],
        "deprecated": False,
    }


@pytest.fixture
def open_meteo_payload() -> dict:
    return {
        "latitude": 37.39,
        "longitude": -5.99,
        "timezone": "UTC",
        "hourly_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "m/s",
            "wind_speed_100m": "m/s",
            "shortwave_radiation": "W/m²",
            "cloud_cover": "%",
            "direct_radiation": "W/m²",
            "diffuse_radiation": "W/m²",
        },
        "hourly": {
            "time": ["2024-06-15T00:00", "2024-06-15T01:00", "2024-06-15T02:00"],
            "temperature_2m": [22.0, 21.5, 21.0],
            "wind_speed_10m": [3.5, 3.6, 3.7],
            "wind_speed_100m": [6.5, 6.7, 6.9],
            "shortwave_radiation": [0.0, 0.0, 0.0],
            "cloud_cover": [10.0, 12.0, 15.0],
            "direct_radiation": [0.0, 0.0, 0.0],
            "diffuse_radiation": [0.0, 0.0, 0.0],
        },
    }
