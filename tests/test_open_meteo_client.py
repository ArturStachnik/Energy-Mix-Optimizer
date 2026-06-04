"""Open-Meteo client tests."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from energy_mix_optimizer.data.locations import IBERIAN_PANEL
from energy_mix_optimizer.data.open_meteo_client import OpenMeteoClient
from energy_mix_optimizer.exceptions import DataSourceError


@respx.mock
@pytest.mark.asyncio
async def test_forecast_aggregates_responses_from_all_points(
    open_meteo_payload: dict,
) -> None:
    route = respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=open_meteo_payload)
    )
    async with OpenMeteoClient() as client:
        df = await client.get_forecast(IBERIAN_PANEL, forecast_days=2)

    # Each panel point triggers one call.
    assert route.call_count == len(IBERIAN_PANEL)
    assert set(df["location"]) == {p.name for p in IBERIAN_PANEL}
    assert "temperature_2m" in df.columns
    assert "timestamp" in df.columns


@respx.mock
@pytest.mark.asyncio
async def test_archive_aggregates_responses(open_meteo_payload: dict) -> None:
    respx.get("https://archive-api.open-meteo.com/v1/archive").mock(
        return_value=httpx.Response(200, json=open_meteo_payload)
    )
    async with OpenMeteoClient() as client:
        df = await client.get_archive(
            IBERIAN_PANEL,
            start_date=date(2024, 6, 15),
            end_date=date(2024, 6, 16),
        )
    # 3 timestamps in fixture, 7 panel points = 21 rows.
    assert len(df) == 3 * len(IBERIAN_PANEL)


@respx.mock
@pytest.mark.asyncio
async def test_malformed_payload_raises_data_source_error() -> None:
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={"latitude": 40.0, "longitude": -3.0})
    )
    async with OpenMeteoClient() as client:
        with pytest.raises(DataSourceError):
            await client.get_forecast(IBERIAN_PANEL, forecast_days=2)
