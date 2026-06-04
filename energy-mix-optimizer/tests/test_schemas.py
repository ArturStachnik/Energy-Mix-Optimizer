"""Schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from energy_mix_optimizer.api.schemas import DispatchRequest, ForecastRequest


class TestForecastRequest:
    def test_defaults(self) -> None:
        req = ForecastRequest()
        assert req.horizon_hours == 24
        assert req.as_of is None
        assert req.carbon_price_eur_per_t is None

    def test_horizon_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(horizon_hours=200)

    def test_horizon_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(horizon_hours=0)

    def test_negative_carbon_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(carbon_price_eur_per_t=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ForecastRequest(horizon_hours=24, mystery_field=42)  # type: ignore[call-arg]


class TestDispatchRequest:
    def test_minimum_payload_accepted(self) -> None:
        req = DispatchRequest(
            demand_mw=[10_000.0],
            solar_forecast_mw=[1_000.0],
            wind_forecast_mw=[2_000.0],
        )
        assert len(req.demand_mw) == 1

    def test_empty_arrays_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DispatchRequest(
                demand_mw=[],
                solar_forecast_mw=[],
                wind_forecast_mw=[],
            )

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DispatchRequest(
                demand_mw=[10_000.0],
                solar_forecast_mw=[-100.0],
                wind_forecast_mw=[0.0],
            )

    def test_long_horizon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DispatchRequest(
                demand_mw=[10_000.0] * 100,
                solar_forecast_mw=[0.0] * 100,
                wind_forecast_mw=[0.0] * 100,
            )
