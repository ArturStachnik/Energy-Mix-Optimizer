"""Spatial aggregation points for weather features.

The Iberian Peninsula is large enough that a single point is a poor proxy
for system-wide renewable generation. We aggregate weather across a small
panel of representative locations, weighted by the rough share of installed
solar and wind capacity in their surrounding regions.

The weights are coarse and only intended for a portfolio-level forecast. A
production deployment would use installation-resolved weights informed by
the actual asset register.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeoPoint:
    """A single weather sampling point."""

    name: str
    latitude: float
    longitude: float
    solar_weight: float
    wind_weight: float


# Weights are normalized to sum to 1.0 within each technology.
# Sources for siting: REE installed-capacity reports for Spain (2023-2024).
IBERIAN_PANEL: tuple[GeoPoint, ...] = (
    # Southern Spain: solar-heavy.
    GeoPoint(name="Sevilla", latitude=37.39, longitude=-5.99, solar_weight=0.25, wind_weight=0.05),
    GeoPoint(name="Badajoz", latitude=38.88, longitude=-6.97, solar_weight=0.20, wind_weight=0.05),
    GeoPoint(name="Murcia", latitude=37.99, longitude=-1.13, solar_weight=0.15, wind_weight=0.05),
    # Castilla-La Mancha: mixed solar and wind.
    GeoPoint(name="Albacete", latitude=38.99, longitude=-1.86, solar_weight=0.15, wind_weight=0.15),
    # Northern interior: wind-heavy.
    GeoPoint(name="Zaragoza", latitude=41.65, longitude=-0.89, solar_weight=0.10, wind_weight=0.25),
    GeoPoint(name="Valladolid", latitude=41.65, longitude=-4.72, solar_weight=0.10, wind_weight=0.20),
    # Atlantic northwest: wind-heavy.
    GeoPoint(name="A Coruna", latitude=43.36, longitude=-8.41, solar_weight=0.05, wind_weight=0.25),
)


def _assert_weights_normalized() -> None:
    """Sanity-check that the weight vectors sum to 1.

    Triggered at import time so a misconfigured panel fails loudly.
    """
    solar_sum = sum(p.solar_weight for p in IBERIAN_PANEL)
    wind_sum = sum(p.wind_weight for p in IBERIAN_PANEL)
    if not (0.999 <= solar_sum <= 1.001):
        raise ValueError(f"IBERIAN_PANEL solar weights sum to {solar_sum}, expected 1.0")
    if not (0.999 <= wind_sum <= 1.001):
        raise ValueError(f"IBERIAN_PANEL wind weights sum to {wind_sum}, expected 1.0")


_assert_weights_normalized()
