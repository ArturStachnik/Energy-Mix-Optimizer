"""FastAPI application.

Exposes the inference pipeline and the dispatch optimizer over HTTP.
Models are loaded lazily on startup: if no artifacts are present, the
service still starts so that ``/health`` works, but ``/forecast`` and
``/models/info`` return ``503``.

Run locally with::

    uvicorn energy_mix_optimizer.api.main:app --reload

Run in production via the ``Dockerfile`` (gunicorn + uvicorn workers).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from energy_mix_optimizer import __version__
from energy_mix_optimizer.api.dependencies import (
    EnergyChartsClientDep,
    ModelsDep,
    SettingsDep,
    WeatherClientDep,
)
from energy_mix_optimizer.api.schemas import (
    DispatchRequest,
    DispatchResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HourlyDispatchOut,
    ModelInfoOut,
    ModelsInfoResponse,
    SummaryOut,
    TrainingMetricsOut,
)
from energy_mix_optimizer.config import get_settings
from energy_mix_optimizer.data.energy_charts_client import EnergyChartsClient
from energy_mix_optimizer.data.open_meteo_client import OpenMeteoClient
from energy_mix_optimizer.exceptions import (
    DataSourceError,
    EnergyMixOptimizerError,
    InvalidRequestError,
    ModelArtifactMissingError,
    OptimizationInfeasibleError,
)
from energy_mix_optimizer.logging_config import configure_logging
from energy_mix_optimizer.optimization.dispatch import optimize_dispatch_horizon
from energy_mix_optimizer.pipelines.forecast import LoadedModels, run_forecast

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared clients and load models once per process."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info("Starting Energy Mix Optimizer API v%s", __version__)

    app.state.settings = settings
    app.state.energy_charts_client = EnergyChartsClient(
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        user_agent=settings.http_user_agent,
    )
    app.state.weather_client = OpenMeteoClient(
        forecast_url=settings.open_meteo_forecast_url,
        archive_url=settings.open_meteo_archive_url,
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        user_agent=settings.http_user_agent,
    )

    try:
        app.state.models = LoadedModels.load(settings.models_dir)
        logger.info("Loaded models from %s", settings.models_dir)
    except ModelArtifactMissingError as exc:
        logger.warning(
            "No trained models found at %s: %s. The /forecast and /models/info "
            "endpoints will return 503 until models are trained.",
            settings.models_dir,
            exc,
        )
        app.state.models = None

    try:
        yield
    finally:
        logger.info("Shutting down Energy Mix Optimizer API")
        await app.state.energy_charts_client.close()
        await app.state.weather_client.close()


app = FastAPI(
    title="Energy Mix Optimizer",
    description=(
        "Day-ahead renewable generation forecast and dispatch optimization "
        "for the Iberian power system, using public Energy-Charts and Open-Meteo data."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- Exception handlers -----------------------------------------------


@app.exception_handler(InvalidRequestError)
async def _invalid_request_handler(
    request: Request,  # noqa: ARG001 - FastAPI signature
    exc: InvalidRequestError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_request", "detail": str(exc)},
    )


@app.exception_handler(DataSourceError)
async def _data_source_handler(
    request: Request,  # noqa: ARG001
    exc: DataSourceError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": "upstream_data_error", "detail": str(exc)},
    )


@app.exception_handler(OptimizationInfeasibleError)
async def _infeasible_handler(
    request: Request,  # noqa: ARG001
    exc: OptimizationInfeasibleError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"error": "optimization_infeasible", "detail": str(exc)},
    )


@app.exception_handler(EnergyMixOptimizerError)
async def _generic_domain_handler(
    request: Request,  # noqa: ARG001
    exc: EnergyMixOptimizerError,
) -> JSONResponse:
    logger.exception("Unhandled domain error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": str(exc)},
    )


# ---------- Endpoints --------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=getattr(request.app.state, "models", None) is not None,
    )


@app.get("/models/info", response_model=ModelsInfoResponse, tags=["meta"])
def models_info(models: ModelsDep) -> ModelsInfoResponse:
    out: dict[str, ModelInfoOut] = {}
    for kind, forecaster in (
        ("solar", models.solar),
        ("wind", models.wind),
        ("demand", models.demand),
    ):
        meta = forecaster.metadata
        out[kind] = ModelInfoOut(
            target_kind=meta.target_kind,
            feature_columns=meta.feature_columns,
            trained_at_utc=meta.trained_at_utc,
            training_data_start=meta.training_data_start,
            training_data_end=meta.training_data_end,
            test_metrics=TrainingMetricsOut(**asdict(meta.test_metrics)),
            cv_metrics_mean=TrainingMetricsOut(**asdict(meta.cv_metrics_mean)),
            cv_metrics_std=dict(meta.cv_metrics_std),
            xgboost_version=meta.xgboost_version,
            package_version=meta.package_version,
        )
    return ModelsInfoResponse(models=out)


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
async def forecast_endpoint(
    payload: ForecastRequest,
    models: ModelsDep,
    settings: SettingsDep,
    energy_charts: EnergyChartsClientDep,
    weather: WeatherClientDep,
) -> ForecastResponse:
    as_of = payload.as_of
    if as_of is None:
        as_of = (datetime.now(UTC) + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    carbon_price = (
        payload.carbon_price_eur_per_t
        if payload.carbon_price_eur_per_t is not None
        else settings.default_carbon_price_eur_per_t
    )

    bundle = await run_forecast(
        as_of=as_of,
        horizon_hours=payload.horizon_hours,
        carbon_price_eur_per_t=carbon_price,
        models=models,
        energy_charts=energy_charts,
        weather=weather,
    )
    return _bundle_to_response(bundle)


@app.post("/optimize", response_model=DispatchResponse, tags=["forecast"])
def optimize_endpoint(payload: DispatchRequest, settings: SettingsDep) -> DispatchResponse:
    if not (len(payload.demand_mw) == len(payload.solar_forecast_mw) == len(payload.wind_forecast_mw)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="demand_mw, solar_forecast_mw and wind_forecast_mw must have the same length",
        )

    carbon = (
        payload.carbon_price_eur_per_t
        if payload.carbon_price_eur_per_t is not None
        else settings.default_carbon_price_eur_per_t
    )
    renewables = [
        {"solar_pv": s, "wind": w}
        for s, w in zip(payload.solar_forecast_mw, payload.wind_forecast_mw, strict=True)
    ]
    results = optimize_dispatch_horizon(
        demand_mw=list(payload.demand_mw),
        renewable_forecast_mw=renewables,
        carbon_price_eur_per_t=carbon,
    )

    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    hourly = [
        HourlyDispatchOut(
            timestamp=now + timedelta(hours=i),
            demand_mw=float(payload.demand_mw[i]),
            solar_pv_mw=float(payload.solar_forecast_mw[i]),
            wind_mw=float(payload.wind_forecast_mw[i]),
            dispatch_mw={k: round(v, 2) for k, v in r.dispatch_mw.items()},
            marginal_cost_eur_per_mwh=r.marginal_cost_eur_per_mwh,
            total_energy_cost_eur=r.total_energy_cost_eur,
            total_emissions_t=r.total_emissions_t,
            feasible=r.feasible,
        )
        for i, r in enumerate(results)
    ]
    total_demand = float(sum(payload.demand_mw))
    total_cost = sum(r.total_energy_cost_eur for r in results)
    total_emissions = sum(r.total_emissions_t for r in results)
    total_renewable = sum(
        r.dispatch_mw.get("solar_pv", 0.0) + r.dispatch_mw.get("wind", 0.0) for r in results
    )
    summary = SummaryOut(
        total_demand_mwh=round(total_demand, 2),
        total_energy_cost_eur=round(total_cost, 2),
        total_emissions_t=round(total_emissions, 2),
        renewable_share=round(total_renewable / total_demand, 4) if total_demand > 0 else 0.0,
        average_marginal_cost_eur_per_mwh=_safe_mean(
            [r.marginal_cost_eur_per_mwh for r in results]
        ),
    )
    return DispatchResponse(hourly=hourly, summary=summary)


# ---------- Helpers ----------------------------------------------------------


def _bundle_to_response(bundle: object) -> ForecastResponse:
    bundle_dict = asdict(bundle)  # type: ignore[arg-type]
    hourly = [
        HourlyDispatchOut(
            timestamp=row["timestamp"],
            demand_mw=row["demand_mw"],
            solar_pv_mw=row["solar_pv_mw"],
            wind_mw=row["wind_mw"],
            dispatch_mw=row["dispatch_mw"],
            marginal_cost_eur_per_mwh=row["marginal_cost_eur_per_mwh"],
            total_energy_cost_eur=row["total_energy_cost_eur"],
            total_emissions_t=row["total_emissions_t"],
            feasible=row["feasible"],
        )
        for row in bundle_dict["hourly"]
    ]
    summary_payload = bundle_dict["summary"]
    summary = SummaryOut(
        total_demand_mwh=summary_payload["total_demand_mwh"],
        total_energy_cost_eur=summary_payload["total_energy_cost_eur"],
        total_emissions_t=summary_payload["total_emissions_t"],
        renewable_share=summary_payload["renewable_share"],
        average_marginal_cost_eur_per_mwh=summary_payload.get(
            "average_marginal_cost_eur_per_mwh"
        ),
    )
    return ForecastResponse(
        as_of_utc=bundle_dict["as_of_utc"],
        horizon_hours=bundle_dict["horizon_hours"],
        carbon_price_eur_per_t=bundle_dict["carbon_price_eur_per_t"],
        hourly=hourly,
        summary=summary,
        model_versions=bundle_dict["model_versions"],
    )


def _safe_mean(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 4)
