"""Client for the Energy-Charts API (Fraunhofer ISE).

Energy-Charts (https://energy-charts.info) is a public, no-auth REST API
maintained by the Fraunhofer Institute for Solar Energy Systems ISE. It
aggregates ENTSO-E Transparency Platform data and re-exposes it under a
permissive CC BY 4.0 license without requiring registration or API keys.

API root: https://api.energy-charts.info
Endpoint used: ``/public_power?country=<code>&start=<date>&end=<date>``

The endpoint returns generation by technology and electricity demand
(``Load``) in a single response, at 15-minute resolution for Germany and
hourly resolution for most other countries. We resample to hourly UTC and
normalize the production-type names to our internal vocabulary.

Reference: https://api.energy-charts.info/
Attribution: data must credit ``Energy-Charts.info``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from types import TracebackType
from typing import Any, Self

import httpx
import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from energy_mix_optimizer.exceptions import DataSourceError

logger = logging.getLogger(__name__)


# Mapping from Energy-Charts production type names to our normalized
# lower_snake_case technology names. Entries are aggregated when several
# Energy-Charts categories collapse to a single internal technology.
_PRODUCTION_TYPE_MAP: dict[str, str] = {
    "Solar": "solar_pv",
    "Wind onshore": "wind",
    "Wind offshore": "wind",
    "Nuclear": "nuclear",
    "Fossil gas": "combined_cycle",
    "Fossil coal-derived gas": "combined_cycle",
    "Fossil hard coal": "coal",
    "Fossil brown coal / lignite": "coal",
    "Fossil oil": "oil",
    "Hydro Run-of-River": "hydro",
    "Hydro water reservoir": "hydro",
    "Hydro pumped storage": "hydro",
    "Biomass": "biomass",
    "Geothermal": "geothermal",
    "Waste": "waste",
    "Others": "other",
}

# These series in ``production_types`` are not generation; we route them
# separately.
_DEMAND_SERIES_NAME = "Load"
_NEGATIVE_PUMPED_CONSUMPTION = "Hydro pumped storage consumption"
_IGNORE_SERIES: frozenset[str] = frozenset(
    {"Cross border electricity trading", "Residual load",
     "Renewable share of load", "Renewable share of generation"}
)


def _is_retriable_exception(exc: BaseException) -> bool:
    """Retry on transport errors and 5xx; fail fast on 4xx."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError))


class EnergyChartsClient:
    """Asynchronous client for the Energy-Charts public API.

    Usage::

        async with EnergyChartsClient() as ec:
            generation_df, demand_series = await ec.get_generation_and_demand(
                country="es",
                start=date(2024, 1, 1),
                end=date(2024, 6, 30),
            )

    Or with an externally managed HTTP client (useful for tests)::

        client = EnergyChartsClient(http_client=httpx.AsyncClient(transport=...))
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.energy-charts.info",
        timeout: float = 60.0,
        max_retries: int = 3,
        user_agent: str = "energy-mix-optimizer/0.3",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
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

    # ----- Public API --------------------------------------------------------

    async def get_generation_and_demand(
        self,
        *,
        country: str = "es",
        start: date,
        end: date,
        resample_to_hourly: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return generation by technology and demand for a date range.

        Parameters
        ----------
        country
            ISO-style country code accepted by Energy-Charts (``es`` for Spain,
            ``de``, ``fr``, ``pt``, ...).
        start
            Inclusive start date.
        end
            Inclusive end date.
        resample_to_hourly
            If True (default), 15-minute series are resampled to hourly means.

        Returns
        -------
        generation_df
            DataFrame indexed by UTC timestamps with one column per
            normalized technology (``solar_pv``, ``wind``, ``nuclear``,
            ``combined_cycle``, ``coal``, ``hydro``, ...). Values are in MW.
        demand_series
            Series indexed by UTC timestamps with electricity demand in MW
            (the Energy-Charts ``Load`` series).
        """
        payload = await self._fetch_public_power(country=country, start=start, end=end)
        generation_df = self._parse_generation(payload)
        demand_series = self._parse_demand(payload)
        if resample_to_hourly:
            generation_df = generation_df.resample("1h").mean().dropna(how="all")
            demand_series = demand_series.resample("1h").mean().dropna()
        return generation_df, demand_series

    # ----- Internals ---------------------------------------------------------

    async def _fetch_public_power(
        self,
        *,
        country: str,
        start: date,
        end: date,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/public_power"
        params = {
            "country": country,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(self._max_retries, 1)),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception(_is_retriable_exception),
            reraise=True,
        ):
            with attempt:
                logger.debug("Energy-Charts GET %s params=%s", url, params)
                response = await self._client.get(url, params=params)
                if response.status_code >= 400:
                    body = response.text[:500] if response.text else "<empty body>"
                    logger.warning(
                        "Energy-Charts /public_power returned %d for country=%s: %s",
                        response.status_code,
                        country,
                        body,
                    )
                response.raise_for_status()
                try:
                    return response.json()
                except ValueError as exc:
                    raise DataSourceError(
                        "Energy-Charts returned non-JSON payload"
                    ) from exc

        raise DataSourceError("Energy-Charts /public_power request failed")

    @staticmethod
    def _parse_generation(payload: dict[str, Any]) -> pd.DataFrame:
        timestamps = _decode_timestamps(payload)
        production_types = payload.get("production_types") or []
        if not production_types:
            raise DataSourceError("Energy-Charts payload has no 'production_types' entries")

        # Aggregate by our normalized technology name.
        by_tech: dict[str, pd.Series] = {}
        for entry in production_types:
            name = entry.get("name", "")
            data = entry.get("data") or []
            if name == _DEMAND_SERIES_NAME or name in _IGNORE_SERIES:
                continue
            if name == _NEGATIVE_PUMPED_CONSUMPTION:
                # Pump consumption is reported as negative MW (storage charge).
                # We currently fold it into hydro to net out the cycle.
                tech = "hydro"
            else:
                tech = _PRODUCTION_TYPE_MAP.get(name)
            if tech is None:
                logger.debug("Ignoring unmapped Energy-Charts series '%s'", name)
                continue

            values = _coerce_floats(data)
            series = pd.Series(values, index=timestamps, name=tech)
            if tech in by_tech:
                by_tech[tech] = by_tech[tech].add(series, fill_value=0.0)
            else:
                by_tech[tech] = series

        if not by_tech:
            raise DataSourceError(
                "Energy-Charts payload contained no recognized generation series"
            )

        df = pd.concat(by_tech.values(), axis=1).sort_index()
        # Replace negatives in non-storage technologies with zero (reporting
        # quirks of net production for ROR hydro can produce small negatives).
        for col in df.columns:
            df[col] = df[col].clip(lower=0.0)
        return df

    @staticmethod
    def _parse_demand(payload: dict[str, Any]) -> pd.Series:
        timestamps = _decode_timestamps(payload)
        for entry in payload.get("production_types") or []:
            if entry.get("name") == _DEMAND_SERIES_NAME:
                values = _coerce_floats(entry.get("data") or [])
                return pd.Series(values, index=timestamps, name="demand_mw").dropna()
        raise DataSourceError("Energy-Charts payload has no 'Load' series")


def _decode_timestamps(payload: dict[str, Any]) -> pd.DatetimeIndex:
    raw = payload.get("unix_seconds")
    if not raw:
        raise DataSourceError("Energy-Charts payload missing 'unix_seconds'")
    return pd.to_datetime(raw, unit="s", utc=True)


def _coerce_floats(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            out.append(float("nan"))
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
    return out


def chunk_date_range(
    start: date, end: date, *, max_days: int
) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into consecutive sub-ranges of at most ``max_days``.

    Exposed so callers can paginate manually if Energy-Charts begins to
    enforce per-call limits in the future. Currently a 6-month single call
    works reliably.
    """
    if end <= start:
        return [(start, end)]
    chunks: list[tuple[date, date]] = []
    cursor = start
    delta = timedelta(days=max_days)
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks
