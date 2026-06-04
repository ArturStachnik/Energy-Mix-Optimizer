"""Client for the Open-Meteo weather API.

Open-Meteo (https://open-meteo.com) is a free, no-auth weather service that
provides global, hourly forecasts and historical reanalysis. We use two
endpoints:

* ``/v1/forecast`` - forecasts up to 16 days ahead.
* ``/v1/archive`` - historical data from 1940 onwards (ERA5-backed).

The API exposes a single endpoint per (latitude, longitude) - we issue one
HTTP request per location in our Iberian panel and concatenate the
responses into a single long-format DataFrame.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from datetime import date
from types import TracebackType
from typing import Any, Self

import httpx
import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from energy_mix_optimizer.data.locations import GeoPoint
from energy_mix_optimizer.exceptions import DataSourceError

logger = logging.getLogger(__name__)


# Hourly variables used by our forecasters. Names match the Open-Meteo API.
DEFAULT_HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "shortwave_radiation",
    "cloud_cover",
    "direct_radiation",
    "diffuse_radiation",
)


class OpenMeteoClient:
    """Asynchronous client for Open-Meteo forecast and archive endpoints."""

    def __init__(
        self,
        *,
        forecast_url: str = "https://api.open-meteo.com/v1/forecast",
        archive_url: str = "https://archive-api.open-meteo.com/v1/archive",
        timeout: float = 30.0,
        max_retries: int = 3,
        user_agent: str = "energy-mix-optimizer/0.2",
        http_client: httpx.AsyncClient | None = None,
        timezone: str = "UTC",
    ) -> None:
        self._forecast_url = forecast_url
        self._archive_url = archive_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._timezone = timezone
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ----- Public endpoints --------------------------------------------------

    async def get_forecast(
        self,
        points: Sequence[GeoPoint],
        *,
        variables: Iterable[str] = DEFAULT_HOURLY_VARIABLES,
        forecast_days: int = 2,
    ) -> pd.DataFrame:
        """Return a long-format DataFrame of forecast weather for each point.

        Columns: ``timestamp`` (UTC), ``location``, ``latitude``,
        ``longitude``, plus one column per variable.
        """
        variables = tuple(variables)
        responses = await asyncio.gather(
            *[
                self._fetch_forecast(point, variables=variables, forecast_days=forecast_days)
                for point in points
            ]
        )
        return self._concatenate(points, responses, variables=variables)

    async def get_archive(
        self,
        points: Sequence[GeoPoint],
        *,
        start_date: date,
        end_date: date,
        variables: Iterable[str] = DEFAULT_HOURLY_VARIABLES,
    ) -> pd.DataFrame:
        """Return a long-format DataFrame of historical weather for each point."""
        variables = tuple(variables)
        responses = await asyncio.gather(
            *[
                self._fetch_archive(
                    point,
                    variables=variables,
                    start_date=start_date,
                    end_date=end_date,
                )
                for point in points
            ]
        )
        return self._concatenate(points, responses, variables=variables)

    # ----- Internals ---------------------------------------------------------

    async def _fetch_forecast(
        self,
        point: GeoPoint,
        *,
        variables: tuple[str, ...],
        forecast_days: int,
    ) -> dict[str, Any]:
        params = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "hourly": ",".join(variables),
            "forecast_days": forecast_days,
            "timezone": self._timezone,
        }
        return await self._get(self._forecast_url, params, label=f"forecast/{point.name}")

    async def _fetch_archive(
        self,
        point: GeoPoint,
        *,
        variables: tuple[str, ...],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        params = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "hourly": ",".join(variables),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": self._timezone,
        }
        return await self._get(self._archive_url, params, label=f"archive/{point.name}")

    async def _get(self, url: str, params: dict[str, Any], *, label: str) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(self._max_retries, 1)),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
            reraise=True,
        ):
            with attempt:
                logger.debug("Open-Meteo GET %s [%s]", url, label)
                response = await self._client.get(url, params=params)
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    raise DataSourceError(
                        f"Open-Meteo returned non-JSON payload for {label}"
                    ) from exc

        raise DataSourceError(f"Open-Meteo request failed for {label}")

    @staticmethod
    def _concatenate(
        points: Sequence[GeoPoint],
        responses: Sequence[dict[str, Any]],
        *,
        variables: tuple[str, ...],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for point, payload in zip(points, responses, strict=True):
            hourly = payload.get("hourly")
            if not hourly or "time" not in hourly:
                raise DataSourceError(
                    f"Open-Meteo response for {point.name} missing 'hourly.time' block"
                )
            timestamps = pd.to_datetime(hourly["time"], utc=True, errors="raise")
            frame = pd.DataFrame({var: hourly.get(var) for var in variables}, index=timestamps)
            frame["location"] = point.name
            frame["latitude"] = point.latitude
            frame["longitude"] = point.longitude
            frames.append(frame.reset_index(names="timestamp"))

        return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "location"])
