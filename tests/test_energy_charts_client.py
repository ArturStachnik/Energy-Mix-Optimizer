"""Tests for the Energy-Charts API client.

We use ``respx`` to mock httpx so no network access is required. The
fixtures live in ``conftest.py``; see ``energy_charts_payload``.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import httpx
import pandas as pd
import pytest
import respx

from energy_mix_optimizer.data.energy_charts_client import (
    EnergyChartsClient,
    chunk_date_range,
)
from energy_mix_optimizer.exceptions import DataSourceError

_BASE = "https://api.energy-charts.info"


@pytest.mark.asyncio
@respx.mock
async def test_get_generation_and_demand_parses_payload(
    energy_charts_payload: dict,
) -> None:
    respx.get(f"{_BASE}/public_power").mock(
        return_value=httpx.Response(200, json=energy_charts_payload)
    )

    async with EnergyChartsClient() as client:
        generation, demand = await client.get_generation_and_demand(
            country="es",
            start=date(2024, 6, 15),
            end=date(2024, 6, 15),
            resample_to_hourly=False,
        )

    assert isinstance(generation.index, pd.DatetimeIndex)
    assert generation.index.tz is not None  # UTC-aware

    # The mapping collapses two solar series (we only sent one) and two
    # hydro categories; check the resulting columns.
    cols = set(generation.columns)
    assert {"solar_pv", "wind", "nuclear", "combined_cycle", "hydro"} <= cols

    # Hydro should be ROR + reservoir summed (1200+800 = 2000 at t0).
    assert generation["hydro"].iloc[0] == pytest.approx(2000.0)
    assert generation["nuclear"].iloc[0] == pytest.approx(5800.0)

    # Demand reads from the 'Load' series.
    assert demand.iloc[0] == pytest.approx(28000.0)
    assert demand.iloc[3] == pytest.approx(29500.0)


@pytest.mark.asyncio
@respx.mock
async def test_resample_to_hourly_keeps_means(energy_charts_payload: dict) -> None:
    # Payload is already hourly; resampling should be a no-op except for
    # potential edge alignment.
    respx.get(f"{_BASE}/public_power").mock(
        return_value=httpx.Response(200, json=energy_charts_payload)
    )

    async with EnergyChartsClient() as client:
        generation, demand = await client.get_generation_and_demand(
            country="es",
            start=date(2024, 6, 15),
            end=date(2024, 6, 15),
            resample_to_hourly=True,
        )

    # Same number of rows since the source is already hourly.
    assert len(generation) == 4
    assert len(demand) == 4
    # Hourly means equal raw values for a 1-sample-per-hour series.
    assert generation["nuclear"].iloc[0] == pytest.approx(5800.0)


@pytest.mark.asyncio
@respx.mock
async def test_negative_aggregate_is_clipped(energy_charts_payload: dict) -> None:
    # When all wind sub-series net negative (a reporting glitch), the
    # parser should clip the aggregate to zero so downstream optimizers
    # do not see infeasible inputs. Note: negative pumped-storage
    # consumption is *intentionally* preserved by being folded into
    # ``hydro`` to net out the storage cycle.
    payload = dict(energy_charts_payload)
    payload["production_types"] = [
        # Drop wind onshore and replace with an all-negative offshore series.
        *[s for s in payload["production_types"] if s["name"] != "Wind onshore"],
        {"name": "Wind offshore", "data": [-10.0, -5.0, -2.0, -1.0]},
    ]
    respx.get(f"{_BASE}/public_power").mock(return_value=httpx.Response(200, json=payload))

    async with EnergyChartsClient() as client:
        generation, _ = await client.get_generation_and_demand(
            country="es",
            start=date(2024, 6, 15),
            end=date(2024, 6, 15),
            resample_to_hourly=False,
        )

    # All-negative wind nets to a negative value, which is clipped to 0.
    assert (generation["wind"] == 0.0).all()


@pytest.mark.asyncio
@respx.mock
async def test_offshore_offset_against_onshore_is_preserved(
    energy_charts_payload: dict,
) -> None:
    # Onshore + offshore with mixed signs: the aggregate is positive and
    # should NOT be clipped (we only clip when the net goes negative).
    payload = dict(energy_charts_payload)
    payload["production_types"] = [
        *payload["production_types"],
        {"name": "Wind offshore", "data": [-10.0, -5.0, 0.0, 3.0]},
    ]
    respx.get(f"{_BASE}/public_power").mock(return_value=httpx.Response(200, json=payload))

    async with EnergyChartsClient() as client:
        generation, _ = await client.get_generation_and_demand(
            country="es",
            start=date(2024, 6, 15),
            end=date(2024, 6, 15),
            resample_to_hourly=False,
        )

    # onshore (8000) + offshore (-10) = 7990 (positive, kept as-is).
    assert generation["wind"].iloc[0] == pytest.approx(7990.0)


@pytest.mark.asyncio
@respx.mock
async def test_missing_load_raises_clear_error() -> None:
    payload = {
        "unix_seconds": [1718409600],
        "production_types": [{"name": "Solar", "data": [0.0]}],
        "deprecated": False,
    }
    respx.get(f"{_BASE}/public_power").mock(return_value=httpx.Response(200, json=payload))

    async with EnergyChartsClient() as client:
        with pytest.raises(DataSourceError, match="Load"):
            await client.get_generation_and_demand(
                country="es",
                start=date(2024, 6, 15),
                end=date(2024, 6, 15),
            )


@pytest.mark.asyncio
@respx.mock
async def test_client_does_not_retry_on_4xx() -> None:
    route = respx.get(f"{_BASE}/public_power").mock(
        return_value=httpx.Response(400, json={"detail": "bad request"})
    )

    async with EnergyChartsClient(max_retries=3) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_generation_and_demand(
                country="es",
                start=date(2024, 6, 15),
                end=date(2024, 6, 15),
            )

    # 4xx must fail fast - the retry policy is for 5xx and transport errors.
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_client_retries_on_5xx_then_succeeds(
    energy_charts_payload: dict,
) -> None:
    route = respx.get(f"{_BASE}/public_power").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=energy_charts_payload),
        ]
    )

    async with EnergyChartsClient(max_retries=3) as client:
        generation, demand = await client.get_generation_and_demand(
            country="es",
            start=date(2024, 6, 15),
            end=date(2024, 6, 15),
            resample_to_hourly=False,
        )

    assert route.call_count == 2
    assert not generation.empty
    assert not demand.empty


def test_chunk_date_range_splits_correctly() -> None:
    chunks = chunk_date_range(date(2024, 1, 1), date(2024, 6, 30), max_days=30)
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 31))
    assert chunks[-1][1] == date(2024, 6, 30)
    # No gaps, no overlaps.
    for a, b in pairwise(chunks):
        assert a[1] == b[0]


def test_chunk_date_range_handles_short_window() -> None:
    chunks = chunk_date_range(date(2024, 1, 1), date(2024, 1, 5), max_days=30)
    assert chunks == [(date(2024, 1, 1), date(2024, 1, 5))]
