"""Pydantic schemas for the public API.

Mirrors the dataclasses in ``pipelines.forecast`` but with explicit
validation, documentation and JSON-schema generation. Keeping the two
layers separate avoids leaking SQL-tier or dataclass internals through the
HTTP surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


class ForecastRequest(BaseModel):
    """Body for ``POST /forecast``."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime | None = Field(
        default=None,
        description=(
            "Start of the forecast window (ISO 8601, UTC if no timezone). "
            "Defaults to next-day midnight UTC."
        ),
    )
    horizon_hours: Annotated[int, Field(ge=1, le=72)] = Field(
        default=24,
        description="Length of the forecast horizon in hours.",
    )
    carbon_price_eur_per_t: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Carbon price applied to the cost vector. Defaults to the "
            "configured EMO_DEFAULT_CARBON_PRICE_EUR_PER_T value."
        ),
    )


class DispatchRequest(BaseModel):
    """Body for ``POST /optimize`` (forecast-free dispatch)."""

    model_config = ConfigDict(extra="forbid")

    demand_mw: Annotated[list[NonNegativeFloat], Field(min_length=1, max_length=72)] = Field(
        description="Hourly forecast of demand in MW."
    )
    solar_forecast_mw: Annotated[list[NonNegativeFloat], Field(min_length=1, max_length=72)] = Field(
        description="Hourly forecast of solar availability in MW."
    )
    wind_forecast_mw: Annotated[list[NonNegativeFloat], Field(min_length=1, max_length=72)] = Field(
        description="Hourly forecast of wind availability in MW."
    )
    carbon_price_eur_per_t: NonNegativeFloat | None = Field(
        default=None,
        description="Carbon price in EUR per tonne.",
    )


class HourlyDispatchOut(BaseModel):
    timestamp: datetime
    demand_mw: float
    solar_pv_mw: float
    wind_mw: float
    dispatch_mw: dict[str, float]
    marginal_cost_eur_per_mwh: float | None
    total_energy_cost_eur: float
    total_emissions_t: float
    feasible: bool


class SummaryOut(BaseModel):
    total_demand_mwh: float
    total_energy_cost_eur: float
    total_emissions_t: float
    renewable_share: float
    average_marginal_cost_eur_per_mwh: float | None = None


class ForecastResponse(BaseModel):
    as_of_utc: datetime
    horizon_hours: int
    carbon_price_eur_per_t: float
    hourly: list[HourlyDispatchOut]
    summary: SummaryOut
    model_versions: dict[str, str]


class DispatchResponse(BaseModel):
    hourly: list[HourlyDispatchOut]
    summary: SummaryOut


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


class TrainingMetricsOut(BaseModel):
    n_samples: int
    mae: float
    rmse: float
    nmae_percent: float | None
    mape_percent: float | None


class ModelInfoOut(BaseModel):
    target_kind: str
    feature_columns: list[str]
    trained_at_utc: str
    training_data_start: str
    training_data_end: str
    test_metrics: TrainingMetricsOut
    cv_metrics_mean: TrainingMetricsOut
    cv_metrics_std: dict[str, float]
    xgboost_version: str
    package_version: str


class ModelsInfoResponse(BaseModel):
    models: dict[str, ModelInfoOut]
