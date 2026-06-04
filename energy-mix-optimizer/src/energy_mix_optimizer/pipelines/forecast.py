"""Inference pipeline: produce the next 24h dispatch recommendation.

This is the high-level orchestration used by the FastAPI endpoints and by
the standalone ``emo-forecast`` CLI. The pipeline is fully asynchronous and
self-contained: given an "as-of" timestamp, it pulls the latest Energy-Charts
actuals (for lag features), pulls Open-Meteo forecast weather, runs the
three forecasters and dispatches the result through the LP optimizer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from energy_mix_optimizer.config import Settings, get_settings
from energy_mix_optimizer.data.energy_charts_client import EnergyChartsClient
from energy_mix_optimizer.data.feature_engineering import (
    DEMAND_SPEC,
    SOLAR_SPEC,
    WIND_SPEC,
    FeatureSpec,
    add_lag_features,
    build_feature_matrix,
)
from energy_mix_optimizer.data.locations import IBERIAN_PANEL
from energy_mix_optimizer.data.open_meteo_client import OpenMeteoClient
from energy_mix_optimizer.logging_config import configure_logging
from energy_mix_optimizer.models.forecaster import TimeSeriesForecaster
from energy_mix_optimizer.optimization.dispatch import (
    DispatchResult,
    optimize_dispatch_horizon,
)

logger = logging.getLogger(__name__)


# Lookback used to populate lag features (24h, 48h, 168h require at least 7
# days of past target values).
_LOOKBACK_DAYS_FOR_LAGS = 9


@dataclass
class HourlyForecast:
    """One row of the inference output."""

    timestamp: str
    demand_mw: float
    solar_pv_mw: float
    wind_mw: float
    dispatch_mw: dict[str, float] = field(default_factory=dict)
    marginal_cost_eur_per_mwh: float | None = None
    total_energy_cost_eur: float = 0.0
    total_emissions_t: float = 0.0
    feasible: bool = True


@dataclass
class ForecastBundle:
    """Full output of the inference pipeline."""

    as_of_utc: str
    horizon_hours: int
    carbon_price_eur_per_t: float
    hourly: list[HourlyForecast] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class LoadedModels:
    """The three forecasters used at inference time."""

    solar: TimeSeriesForecaster
    wind: TimeSeriesForecaster
    demand: TimeSeriesForecaster

    @classmethod
    def load(cls, directory: Path) -> LoadedModels:
        return cls(
            solar=TimeSeriesForecaster.load(directory, target_kind="solar"),
            wind=TimeSeriesForecaster.load(directory, target_kind="wind"),
            demand=TimeSeriesForecaster.load(directory, target_kind="demand"),
        )

    def versions(self) -> dict[str, str]:
        return {
            "solar": self.solar.metadata.trained_at_utc,
            "wind": self.wind.metadata.trained_at_utc,
            "demand": self.demand.metadata.trained_at_utc,
        }


# ---------- Public entry points ----------------------------------------------


async def run_forecast(
    *,
    as_of: datetime,
    horizon_hours: int,
    carbon_price_eur_per_t: float,
    models: LoadedModels,
    energy_charts: EnergyChartsClient,
    weather: OpenMeteoClient,
) -> ForecastBundle:
    """Produce the dispatch recommendation for the next ``horizon_hours`` hours.

    ``as_of`` is the inclusive start of the forecast window.
    """
    if horizon_hours <= 0 or horizon_hours > 72:
        raise ValueError("horizon_hours must be between 1 and 72")

    history_start = as_of - timedelta(days=_LOOKBACK_DAYS_FOR_LAGS)
    history_end = as_of - timedelta(hours=1)
    window_start = as_of
    window_end = as_of + timedelta(hours=horizon_hours - 1)

    # ---- Concurrent data fetches ----
    energy_task = energy_charts.get_generation_and_demand(
        country="es",
        start=history_start.date(),
        end=history_end.date(),
    )
    forecast_days = max(2, (window_end.date() - as_of.date()).days + 2)
    weather_task = weather.get_forecast(IBERIAN_PANEL, forecast_days=forecast_days)
    archive_task = weather.get_archive(
        IBERIAN_PANEL,
        start_date=history_start.date(),
        end_date=history_end.date(),
    )

    (generation_df, demand_series), weather_forecast, weather_archive = await asyncio.gather(
        energy_task, weather_task, archive_task
    )

    # The aggregate weather table covers both the past (for lag rows) and
    # the future (for feature rows we are about to predict on).
    weather_long = pd.concat([weather_archive, weather_forecast], ignore_index=True)
    weather_long = weather_long.drop_duplicates(subset=["timestamp", "location"], keep="last")

    # Build the target window index.
    target_index = pd.date_range(start=window_start, end=window_end, freq="1h", tz="UTC")

    # Each forecaster needs (a) features at the target rows and (b) a target
    # series that includes past observations so the lag columns can be
    # populated. We compose this by concatenating the observed history with
    # NaNs at the target horizon, then letting the feature builder do the
    # alignment.
    solar_mw = _predict_target(
        forecaster=models.solar,
        spec=SOLAR_SPEC,
        history=generation_df["solar_pv"].rename("solar"),
        weather_long=weather_long,
        target_index=target_index,
    )
    wind_mw = _predict_target(
        forecaster=models.wind,
        spec=WIND_SPEC,
        history=generation_df["wind"].rename("wind"),
        weather_long=weather_long,
        target_index=target_index,
    )
    demand_mw = _predict_target(
        forecaster=models.demand,
        spec=DEMAND_SPEC,
        history=demand_series.rename("demand"),
        weather_long=weather_long,
        target_index=target_index,
    )

    # ---- Dispatch ----
    renewable_forecasts: list[dict[str, float]] = [
        {"solar_pv": max(0.0, float(s)), "wind": max(0.0, float(w))}
        for s, w in zip(solar_mw.to_numpy(), wind_mw.to_numpy(), strict=True)
    ]
    dispatch_results = optimize_dispatch_horizon(
        demand_mw=[max(0.0, float(d)) for d in demand_mw.to_numpy()],
        renewable_forecast_mw=renewable_forecasts,
        carbon_price_eur_per_t=carbon_price_eur_per_t,
    )

    bundle = _assemble_bundle(
        as_of=as_of,
        horizon_hours=horizon_hours,
        carbon_price=carbon_price_eur_per_t,
        target_index=target_index,
        solar_mw=solar_mw,
        wind_mw=wind_mw,
        demand_mw=demand_mw,
        dispatch_results=dispatch_results,
        models=models,
    )
    return bundle


def _predict_target(
    *,
    forecaster: TimeSeriesForecaster,
    spec: FeatureSpec,
    history: pd.Series,
    weather_long: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Build features for the target window and run the model."""
    # Concatenate observed history with NaN placeholders at the target
    # horizon so the feature builder can populate lags using history.
    future = pd.Series(index=target_index, dtype=float, name=history.name)
    extended_target = pd.concat([history, future]).sort_index()

    feature_matrix = build_feature_matrix(
        weather_long=weather_long, target=None, spec=spec
    )
    # We need to recompute lags using the extended target; the
    # build_feature_matrix function with target=None does NOT add lags, so
    # we add them here using the same logic.
    X_all = add_lag_features(
        feature_matrix.X,
        extended_target.reindex(feature_matrix.X.index),
        lag_hours=spec.lag_hours,
        rolling_mean_hours=spec.rolling_mean_hours,
    )

    # Subset to target window.
    X_target = X_all.reindex(target_index).copy()
    if X_target.isna().any().any():
        # Forward-fill weather columns only; lag columns must remain accurate.
        weather_cols = [c for c in X_target.columns if c in feature_matrix.X.columns]
        if weather_cols:
            X_target[weather_cols] = X_target[weather_cols].ffill()
        nan_count = int(X_target.isna().sum().sum())
        if nan_count > 0:
            logger.warning(
                "Filling %d NaN feature cells for %s with column-wise medians",
                nan_count,
                forecaster.target_kind,
            )
            X_target = X_target.fillna(X_target.median(numeric_only=True))
            X_target = X_target.fillna(0.0)

    predictions = forecaster.predict(X_target)
    return pd.Series(predictions, index=target_index, name=forecaster.target_kind)


def _assemble_bundle(
    *,
    as_of: datetime,
    horizon_hours: int,
    carbon_price: float,
    target_index: pd.DatetimeIndex,
    solar_mw: pd.Series,
    wind_mw: pd.Series,
    demand_mw: pd.Series,
    dispatch_results: list[DispatchResult],
    models: LoadedModels,
) -> ForecastBundle:
    hourly: list[HourlyForecast] = []
    total_cost = 0.0
    total_emissions = 0.0
    total_demand = 0.0
    total_renewable_dispatch = 0.0
    for ts, d, s, w, dispatch in zip(
        target_index, demand_mw, solar_mw, wind_mw, dispatch_results, strict=True
    ):
        hourly.append(
            HourlyForecast(
                timestamp=ts.isoformat(),
                demand_mw=float(d),
                solar_pv_mw=float(s),
                wind_mw=float(w),
                dispatch_mw={k: round(v, 2) for k, v in dispatch.dispatch_mw.items()},
                marginal_cost_eur_per_mwh=dispatch.marginal_cost_eur_per_mwh,
                total_energy_cost_eur=dispatch.total_energy_cost_eur,
                total_emissions_t=dispatch.total_emissions_t,
                feasible=dispatch.feasible,
            )
        )
        total_cost += dispatch.total_energy_cost_eur
        total_emissions += dispatch.total_emissions_t
        total_demand += float(d)
        total_renewable_dispatch += dispatch.dispatch_mw.get(
            "solar_pv", 0.0
        ) + dispatch.dispatch_mw.get("wind", 0.0)

    renewable_share = total_renewable_dispatch / total_demand if total_demand > 0 else 0.0
    summary = {
        "total_demand_mwh": round(total_demand, 2),
        "total_energy_cost_eur": round(total_cost, 2),
        "total_emissions_t": round(total_emissions, 2),
        "renewable_share": round(renewable_share, 4),
        "average_marginal_cost_eur_per_mwh": _safe_mean(
            [r.marginal_cost_eur_per_mwh for r in dispatch_results]
        ),
    }
    return ForecastBundle(
        as_of_utc=as_of.isoformat(),
        horizon_hours=horizon_hours,
        carbon_price_eur_per_t=carbon_price,
        hourly=hourly,
        summary=summary,
        model_versions=models.versions(),
    )


def _safe_mean(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 4)


# ---------- CLI --------------------------------------------------------------


async def _cli_entrypoint(args: argparse.Namespace, settings: Settings) -> ForecastBundle:
    models = LoadedModels.load(args.models_dir)
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
        return await run_forecast(
            as_of=args.as_of,
            horizon_hours=args.horizon_hours,
            carbon_price_eur_per_t=args.carbon_price,
            models=models,
            energy_charts=energy_charts,
            weather=weather,
        )


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        now = datetime.now(UTC)
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the dispatch forecast pipeline.")
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=_parse_as_of(None),
        help="Start of the forecast window (ISO 8601). Defaults to tomorrow 00:00 UTC.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Forecast horizon in hours (1-72).",
    )
    parser.add_argument(
        "--carbon-price",
        type=float,
        default=None,
        help="Carbon price in EUR/tonne. Defaults to the configured value.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing the trained models. Defaults to settings.models_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON result. Otherwise printed to stdout.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    if args.models_dir is None:
        args.models_dir = settings.models_dir
    if args.carbon_price is None:
        args.carbon_price = settings.default_carbon_price_eur_per_t

    bundle = asyncio.run(_cli_entrypoint(args, settings))
    payload = json.dumps(asdict(bundle), indent=2, default=str)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        logger.info("Wrote forecast to %s", args.output)
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
