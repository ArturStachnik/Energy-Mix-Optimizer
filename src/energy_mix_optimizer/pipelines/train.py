"""Training pipeline.

Usage::

    python -m energy_mix_optimizer.pipelines.train \\
        --start 2022-01-01 --end 2024-12-31 \\
        --artifacts-dir artifacts

or, equivalently::

    emo-train --start 2022-01-01 --end 2024-12-31

The pipeline downloads:

* observed generation and demand from Energy-Charts (no auth required),
* historical weather from Open-Meteo Archive (no auth required),

builds features for solar, wind and demand, fits three XGBoost forecasters,
evaluates them with TimeSeriesSplit cross-validation plus a final hold-out
test, and persists the resulting artifacts together with their metadata.

A ``--synthetic`` flag is available for development and CI: it generates
plausible-looking data offline so the rest of the pipeline can be exercised
without hitting any external service.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from energy_mix_optimizer.config import Settings, get_settings
from energy_mix_optimizer.data.energy_charts_client import EnergyChartsClient
from energy_mix_optimizer.data.feature_engineering import (
    DEMAND_SPEC,
    SOLAR_SPEC,
    WIND_SPEC,
    FeatureSpec,
    build_feature_matrix,
)
from energy_mix_optimizer.data.locations import IBERIAN_PANEL
from energy_mix_optimizer.data.open_meteo_client import OpenMeteoClient
from energy_mix_optimizer.logging_config import configure_logging
from energy_mix_optimizer.models.forecaster import ModelMetadata, TimeSeriesForecaster

logger = logging.getLogger(__name__)


# ---------- Data fetching ----------------------------------------------------


async def fetch_real_data(
    *,
    start: datetime,
    end: datetime,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(generation_df, demand_series, weather_long_df)``.

    Generation and demand come from the Energy-Charts public API
    (Fraunhofer ISE, no auth). Weather comes from Open-Meteo Archive.
    """
    async with EnergyChartsClient(
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        user_agent=settings.http_user_agent,
    ) as energy_charts, OpenMeteoClient(
        forecast_url=settings.open_meteo_forecast_url,
        archive_url=settings.open_meteo_archive_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        user_agent=settings.http_user_agent,
    ) as weather:
        logger.info("Fetching generation and demand from Energy-Charts...")
        generation, demand = await energy_charts.get_generation_and_demand(
            country="es",
            start=start.date(),
            end=end.date(),
        )
        logger.info(
            "Generation rows=%d (techs=%s), demand rows=%d",
            len(generation),
            ", ".join(sorted(generation.columns)),
            len(demand),
        )

        logger.info("Fetching historical weather from Open-Meteo archive...")
        weather_long = await weather.get_archive(
            IBERIAN_PANEL,
            start_date=start.date(),
            end_date=end.date(),
        )
        logger.info("Weather rows=%d across %d points", len(weather_long), len(IBERIAN_PANEL))

    return generation, demand, weather_long


def generate_synthetic_data(
    *, start: datetime, end: datetime
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate fake but plausible data so the pipeline can run offline."""
    rng = np.random.default_rng(seed=2024)
    timestamps = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    n = len(timestamps)
    hour = timestamps.hour.to_numpy()
    doy = timestamps.dayofyear.to_numpy()

    daily_solar = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)

    solar = 18_000.0 * daily_solar * seasonal + rng.normal(0, 400, n)
    solar = np.clip(solar, 0, None)

    wind = (
        12_000.0
        + 4_000.0 * np.sin(2 * np.pi * doy / 365.25 + 1.0)
        + 3_000.0 * np.cos(2 * np.pi * hour / 24.0 - 0.5)
        + rng.normal(0, 1_200, n)
    )
    wind = np.clip(wind, 0, None)

    demand = (
        28_000.0
        + 4_000.0 * np.cos(2 * np.pi * (doy - 10) / 365.25)
        + 5_000.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        + rng.normal(0, 700, n)
    )

    generation = pd.DataFrame(
        {
            "solar_pv": solar,
            "wind": wind,
            "hydro": 3_500.0 + rng.normal(0, 200, n),
            "nuclear": 6_500.0 + rng.normal(0, 100, n),
            "combined_cycle": np.clip(demand - solar - wind - 3_500.0 - 6_500.0, 0, None),
        },
        index=timestamps,
    )
    demand_series = pd.Series(demand, index=timestamps, name="demand_mw")

    weather_rows: list[pd.DataFrame] = []
    for point in IBERIAN_PANEL:
        local = pd.DataFrame(
            {
                "timestamp": timestamps,
                "location": point.name,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "temperature_2m": 15.0 + 10.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
                + 5.0 * np.sin(2 * np.pi * (hour - 14) / 24.0)
                + rng.normal(0, 1.5, n),
                "wind_speed_10m": np.clip(4.0 + 3.0 * np.sin(2 * np.pi * doy / 365.25)
                + rng.normal(0, 1.0, n), 0, None),
                "wind_speed_100m": np.clip(7.0 + 4.0 * np.sin(2 * np.pi * doy / 365.25 + 0.5)
                + rng.normal(0, 1.5, n), 0, None),
                "shortwave_radiation": np.maximum(0.0, 850.0 * daily_solar * seasonal
                + rng.normal(0, 40, n)),
                "direct_radiation": np.maximum(0.0, 600.0 * daily_solar * seasonal
                + rng.normal(0, 30, n)),
                "diffuse_radiation": np.maximum(0.0, 250.0 * daily_solar * seasonal
                + rng.normal(0, 20, n)),
                "cloud_cover": np.clip(40.0 + rng.normal(0, 20, n), 0, 100),
            }
        )
        weather_rows.append(local)
    weather_long = pd.concat(weather_rows, ignore_index=True)

    return generation, demand_series, weather_long


# ---------- Training ---------------------------------------------------------


def _select_target(generation: pd.DataFrame, demand: pd.Series, kind: str) -> pd.Series:
    if kind == "solar":
        return generation["solar_pv"].rename("solar")
    if kind == "wind":
        return generation["wind"].rename("wind")
    if kind == "demand":
        return demand.rename("demand")
    raise ValueError(f"Unknown target kind: {kind}")


def train_one(
    *,
    target_kind: str,
    spec: FeatureSpec,
    generation: pd.DataFrame,
    demand: pd.Series,
    weather_long: pd.DataFrame,
    artifacts_dir: Path,
    n_cv_splits: int,
) -> ModelMetadata:
    target = _select_target(generation, demand, target_kind)
    fm = build_feature_matrix(weather_long=weather_long, target=target, spec=spec)
    if fm.y is None or fm.X.empty:
        raise RuntimeError(f"No usable training rows for target '{target_kind}'")

    logger.info(
        "Training '%s' on %d rows with %d features",
        target_kind,
        len(fm.X),
        len(fm.feature_columns),
    )
    forecaster = TimeSeriesForecaster(target_kind=target_kind)
    metadata = forecaster.fit(fm.X, fm.y, n_cv_splits=n_cv_splits)
    forecaster.save(artifacts_dir)
    return metadata


def run_training(
    *,
    start: datetime,
    end: datetime,
    artifacts_dir: Path,
    use_synthetic: bool,
    n_cv_splits: int,
    settings: Settings,
) -> dict[str, ModelMetadata]:
    if use_synthetic:
        logger.warning(
            "Using SYNTHETIC data. Models will only learn the toy distribution."
        )
        generation, demand, weather_long = generate_synthetic_data(start=start, end=end)
    else:
        generation, demand, weather_long = asyncio.run(
            fetch_real_data(start=start, end=end, settings=settings)
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, ModelMetadata] = {}
    for kind, spec in (
        ("solar", SOLAR_SPEC),
        ("wind", WIND_SPEC),
        ("demand", DEMAND_SPEC),
    ):
        metadata[kind] = train_one(
            target_kind=kind,
            spec=spec,
            generation=generation,
            demand=demand,
            weather_long=weather_long,
            artifacts_dir=artifacts_dir,
            n_cv_splits=n_cv_splits,
        )
    return metadata


# ---------- CLI --------------------------------------------------------------


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Energy Mix Optimizer forecasters.")
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=(datetime.now(UTC) - timedelta(days=365 * 2)).replace(
            hour=0, minute=0, second=0, microsecond=0
        ),
        help="Training data start date (YYYY-MM-DD), default = today - 2 years.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
        help="Training data end date (YYYY-MM-DD), default = today.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory where trained models will be written. "
        "Defaults to settings.models_dir.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for cross-validation.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate offline synthetic data instead of calling external APIs.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    artifacts_dir = args.artifacts_dir or settings.models_dir

    metadata = run_training(
        start=args.start,
        end=args.end,
        artifacts_dir=artifacts_dir,
        use_synthetic=args.synthetic,
        n_cv_splits=args.cv_splits,
        settings=settings,
    )

    logger.info("Training complete. Summary:")
    for kind, meta in metadata.items():
        logger.info(
            "  [%s] CV MAE=%.2f, hold-out MAE=%.2f RMSE=%.2f%s",
            kind,
            meta.cv_metrics_mean.mae,
            meta.test_metrics.mae,
            meta.test_metrics.rmse,
            (
                f" NMAE={meta.test_metrics.nmae_percent:.2f}%"
                if meta.test_metrics.nmae_percent is not None
                else ""
            ),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
