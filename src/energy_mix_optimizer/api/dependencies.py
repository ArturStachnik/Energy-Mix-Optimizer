"""FastAPI dependency providers.

The HTTP clients and the loaded model bundle are created once at startup
(``lifespan``) and reused for every request. The functions below expose
them via ``Depends`` so that route handlers receive properly typed
instances and tests can easily override them.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from energy_mix_optimizer.config import Settings
from energy_mix_optimizer.data.energy_charts_client import EnergyChartsClient
from energy_mix_optimizer.data.open_meteo_client import OpenMeteoClient
from energy_mix_optimizer.pipelines.forecast import LoadedModels


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_energy_charts_client(request: Request) -> EnergyChartsClient:
    return request.app.state.energy_charts_client  # type: ignore[no-any-return]


def get_weather_client(request: Request) -> OpenMeteoClient:
    return request.app.state.weather_client  # type: ignore[no-any-return]


def get_models(request: Request) -> LoadedModels:
    models = getattr(request.app.state, "models", None)
    if models is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Forecasting models are not loaded. Run the training pipeline "
                "(`emo-train` or `python -m energy_mix_optimizer.pipelines.train`) "
                "to produce the artifacts."
            ),
        )
    return models  # type: ignore[no-any-return]


SettingsDep = Annotated[Settings, Depends(get_settings)]
EnergyChartsClientDep = Annotated[EnergyChartsClient, Depends(get_energy_charts_client)]
WeatherClientDep = Annotated[OpenMeteoClient, Depends(get_weather_client)]
ModelsDep = Annotated[LoadedModels, Depends(get_models)]
