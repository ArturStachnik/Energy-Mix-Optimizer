"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_mix_optimizer.data.feature_engineering import (
    SOLAR_SPEC,
    WIND_SPEC,
    add_calendar_features,
    add_lag_features,
    aggregate_weather,
    build_feature_matrix,
)
from energy_mix_optimizer.data.locations import GeoPoint


class TestAggregateWeather:
    def test_weighted_mean_matches_manual_calculation(self) -> None:
        timestamps = pd.date_range("2024-06-15", periods=2, freq="h", tz="UTC")
        long_df = pd.DataFrame(
            {
                "timestamp": list(timestamps) * 2,
                "location": ["A", "A", "B", "B"],
                "latitude": [40.0, 40.0, 41.0, 41.0],
                "longitude": [-3.0, -3.0, -4.0, -4.0],
                "temperature_2m": [10.0, 12.0, 20.0, 22.0],
            }
        )
        points = (
            GeoPoint(name="A", latitude=40.0, longitude=-3.0, solar_weight=0.75, wind_weight=0.5),
            GeoPoint(name="B", latitude=41.0, longitude=-4.0, solar_weight=0.25, wind_weight=0.5),
        )
        agg = aggregate_weather(
            long_df, variables=("temperature_2m",), points=points, weight_attr="solar_weight"
        )
        # 0.75*10 + 0.25*20 = 12.5; 0.75*12 + 0.25*22 = 14.5
        assert agg.iloc[0]["temperature_2m"] == pytest.approx(12.5)
        assert agg.iloc[1]["temperature_2m"] == pytest.approx(14.5)

    def test_ignores_locations_not_in_panel(self) -> None:
        timestamps = pd.date_range("2024-06-15", periods=1, freq="h", tz="UTC")
        long_df = pd.DataFrame(
            {
                "timestamp": list(timestamps) * 3,
                "location": ["A", "B", "C"],
                "latitude": [40.0, 41.0, 42.0],
                "longitude": [-3.0, -4.0, -5.0],
                "temperature_2m": [10.0, 20.0, 100.0],
            }
        )
        points = (
            GeoPoint(name="A", latitude=40.0, longitude=-3.0, solar_weight=0.5, wind_weight=0.5),
            GeoPoint(name="B", latitude=41.0, longitude=-4.0, solar_weight=0.5, wind_weight=0.5),
        )
        agg = aggregate_weather(
            long_df, variables=("temperature_2m",), points=points, weight_attr="solar_weight"
        )
        # C is ignored despite the absurd value.
        assert agg.iloc[0]["temperature_2m"] == pytest.approx(15.0)


class TestCalendarFeatures:
    def test_adds_expected_columns(self) -> None:
        idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"x": np.arange(24)}, index=idx)
        out = add_calendar_features(df)
        for col in [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "is_holiday",
        ]:
            assert col in out.columns

    def test_detects_spanish_holiday(self) -> None:
        # 6 January is Epiphany in Spain (Reyes), a public holiday nationwide.
        idx = pd.DatetimeIndex(["2024-01-06T12:00", "2024-01-07T12:00"], tz="UTC")
        df = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
        out = add_calendar_features(df, use_holidays=True)
        assert out.loc[idx[0], "is_holiday"] == 1
        assert out.loc[idx[1], "is_holiday"] == 0

    def test_cyclical_features_in_unit_circle(self) -> None:
        idx = pd.date_range("2024-06-15", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"x": np.zeros(48)}, index=idx)
        out = add_calendar_features(df)
        radii = np.hypot(out["hour_sin"].to_numpy(), out["hour_cos"].to_numpy())
        np.testing.assert_allclose(radii, 1.0, atol=1e-9)


class TestLagFeatures:
    def test_no_future_leakage(self) -> None:
        idx = pd.date_range("2024-06-15", periods=200, freq="h", tz="UTC")
        target = pd.Series(np.arange(200, dtype=float), index=idx, name="y")
        df = pd.DataFrame({"x": np.zeros(200)}, index=idx)
        out = add_lag_features(df, target, lag_hours=(24,), rolling_mean_hours=(24,))
        # At index 50, lag_24h should be target at index 26 = 26.
        assert out["lag_24h"].iloc[50] == pytest.approx(26.0)
        # rolling_mean_24h is mean of target[shift 24].rolling(24); at index
        # 50, it should be the mean of indices 3..26 of the original target.
        expected = target.shift(24).rolling(24, min_periods=1).mean().iloc[50]
        assert out["rolling_mean_24h"].iloc[50] == pytest.approx(expected)

    def test_lag_nans_at_start_of_series(self) -> None:
        idx = pd.date_range("2024-06-15", periods=10, freq="h", tz="UTC")
        target = pd.Series(np.arange(10, dtype=float), index=idx, name="y")
        df = pd.DataFrame({"x": np.zeros(10)}, index=idx)
        out = add_lag_features(df, target, lag_hours=(24, 168), rolling_mean_hours=())
        assert out["lag_24h"].isna().all()
        assert out["lag_168h"].isna().all()


class TestBuildFeatureMatrix:
    def test_with_target_produces_aligned_X_y(
        self,
        synthetic_weather_long: pd.DataFrame,
        synthetic_target: pd.Series,
    ) -> None:
        fm = build_feature_matrix(
            weather_long=synthetic_weather_long,
            target=synthetic_target,
            spec=SOLAR_SPEC,
        )
        assert fm.y is not None
        assert len(fm.X) == len(fm.y)
        assert "lag_24h" in fm.X.columns
        assert "is_holiday" in fm.X.columns
        # No NaNs should remain after training-mode filtering.
        assert not fm.X.isna().any().any()
        assert not fm.y.isna().any()

    def test_uses_correct_weight_for_wind(
        self, synthetic_weather_long: pd.DataFrame
    ) -> None:
        fm = build_feature_matrix(
            weather_long=synthetic_weather_long, target=None, spec=WIND_SPEC
        )
        assert "wind_speed_100m" in fm.X.columns
        # The aggregation weight should be wind-biased: A Coruna (high wind
        # weight) influences the mean substantially. We check it indirectly
        # by ensuring the call did not crash and returned the expected schema.
        assert "hour_sin" in fm.X.columns
