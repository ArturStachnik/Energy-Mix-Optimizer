"""Feature engineering for renewable and demand forecasting.

Three sets of features are produced:

1. **Calendar features**: hour of day, day of week, month, weekend flag,
   Spanish public holiday flag. Cyclical (sin/cos) encodings of the hour
   and day of year are included; tree-based learners do not strictly need
   them but they make some downstream models (e.g. linear baselines)
   straightforward to fit.
2. **Spatially aggregated weather**: temperature, wind speed at 100m,
   surface shortwave radiation, cloud cover. Weighted by installed
   capacity proxies for the target technology.
3. **Lagged target**: 24h, 48h, 168h lags plus a 24h rolling mean shifted
   forward, all strictly causal so no future information leaks into the
   features.

The output is a single DataFrame indexed by UTC timestamps. When a target
column is provided, the feature DataFrame is aligned to it and rows with
missing lag values are dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

from energy_mix_optimizer.data.locations import IBERIAN_PANEL, GeoPoint


@dataclass(frozen=True)
class FeatureSpec:
    """How to build features for a specific forecasting target."""

    target_kind: str  # "solar", "wind", or "demand"
    weather_variables: tuple[str, ...] = (
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
        "cloud_cover",
    )
    lag_hours: tuple[int, ...] = (24, 48, 168)
    rolling_mean_hours: tuple[int, ...] = (24,)
    use_holidays: bool = True


SOLAR_SPEC = FeatureSpec(
    target_kind="solar",
    weather_variables=(
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
        "temperature_2m",
    ),
)
WIND_SPEC = FeatureSpec(
    target_kind="wind",
    weather_variables=("wind_speed_100m", "wind_speed_10m", "temperature_2m"),
)
DEMAND_SPEC = FeatureSpec(
    target_kind="demand",
    weather_variables=("temperature_2m",),
)


@dataclass
class FeatureMatrix:
    """Aligned features and target."""

    X: pd.DataFrame
    y: pd.Series | None = None
    feature_columns: list[str] = field(default_factory=list)


def aggregate_weather(
    weather_long: pd.DataFrame,
    *,
    variables: Sequence[str],
    points: Sequence[GeoPoint] = IBERIAN_PANEL,
    weight_attr: str = "solar_weight",
) -> pd.DataFrame:
    """Collapse the per-location weather table to a single time series.

    Each variable becomes a single column equal to the capacity-weighted
    mean across the panel. Weights are pulled from ``GeoPoint.<weight_attr>``.

    Parameters
    ----------
    weather_long
        Long-format DataFrame from ``OpenMeteoClient`` with a ``timestamp``
        and ``location`` column.
    variables
        Weather variables to aggregate.
    points
        Reference panel of geographic points and their weights.
    weight_attr
        Either ``"solar_weight"`` or ``"wind_weight"``.
    """
    weights = {p.name: getattr(p, weight_attr) for p in points}
    if any(w < 0 for w in weights.values()):
        raise ValueError("Geographic weights must be non-negative")

    df = weather_long.copy()
    df["weight"] = df["location"].map(weights).astype(float)
    df = df.dropna(subset=["weight"])
    if df.empty:
        raise ValueError("No weather rows matched the supplied panel")

    rows: dict[str, pd.Series] = {}
    for var in variables:
        weighted = df[var].astype(float) * df["weight"]
        numerator = weighted.groupby(df["timestamp"]).sum()
        denominator = df["weight"].groupby(df["timestamp"]).sum()
        rows[var] = (numerator / denominator).rename(var)

    out = pd.concat(rows.values(), axis=1)
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out.sort_index()


def add_calendar_features(
    df: pd.DataFrame,
    *,
    use_holidays: bool = True,
) -> pd.DataFrame:
    """Append calendar features in place-friendly fashion (returns a copy)."""
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex")

    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    if use_holidays:
        # ``holidays.Spain`` returns a dict-like keyed by date.
        years = sorted({int(ts.year) for ts in idx})
        es_holidays = holidays.country_holidays("ES", years=years)
        out["is_holiday"] = idx.normalize().tz_convert(None).date
        out["is_holiday"] = [int(d in es_holidays) for d in out["is_holiday"]]
    else:
        out["is_holiday"] = 0

    return out


def add_lag_features(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    lag_hours: Sequence[int],
    rolling_mean_hours: Sequence[int],
    column_prefix: str = "lag",
) -> pd.DataFrame:
    """Append strictly causal lag and rolling-mean features.

    ``target`` is expected to be aligned to ``df`` index. Lags reference past
    values of ``target``; rolling means use only data up to ``t - 24h`` to
    avoid leakage when the model is run for day-ahead forecasting.
    """
    if not df.index.equals(target.index):
        target = target.reindex(df.index)

    out = df.copy()
    for lag in lag_hours:
        out[f"{column_prefix}_{lag}h"] = target.shift(lag)
    for window in rolling_mean_hours:
        # Shift by 24h so the rolling mean uses only history available to a
        # day-ahead forecaster.
        out[f"rolling_mean_{window}h"] = target.shift(24).rolling(window=window, min_periods=1).mean()
    return out


def build_feature_matrix(
    *,
    weather_long: pd.DataFrame,
    target: pd.Series | None,
    spec: FeatureSpec,
    points: Sequence[GeoPoint] = IBERIAN_PANEL,
) -> FeatureMatrix:
    """End-to-end feature build for a given target kind.

    When ``target`` is ``None`` (inference time), lag features are filled
    with the most recent observed history if available, otherwise NaN. The
    caller is responsible for supplying any required lagged target via the
    ``target`` argument (e.g. observed generation from the previous days).
    """
    weight_attr = "wind_weight" if spec.target_kind == "wind" else "solar_weight"
    weather = aggregate_weather(
        weather_long, variables=spec.weather_variables, points=points, weight_attr=weight_attr
    )
    features = add_calendar_features(weather, use_holidays=spec.use_holidays)

    if target is not None:
        target = target.copy()
        target.index = pd.to_datetime(target.index, utc=True)
        target = target.sort_index()
        # Align indices via inner join so we do not drop weather rows that
        # have no observed target yet (useful at inference).
        joined_index = features.index.union(target.index).sort_values()
        features = features.reindex(joined_index)
        target = target.reindex(joined_index)
        features = add_lag_features(
            features,
            target,
            lag_hours=spec.lag_hours,
            rolling_mean_hours=spec.rolling_mean_hours,
        )
        # For training, drop rows missing any feature or target.
        mask = features.notna().all(axis=1) & target.notna()
        X = features.loc[mask].copy()
        y = target.loc[mask].copy().rename(spec.target_kind)
        return FeatureMatrix(X=X, y=y, feature_columns=list(X.columns))

    return FeatureMatrix(X=features, y=None, feature_columns=list(features.columns))
