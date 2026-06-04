"""End-to-end test of the forecaster on synthetic data.

This is a smoke test for the model wrapper itself: it covers
``fit``, ``predict``, ``save``, ``load`` and the metric reporting path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_mix_optimizer.exceptions import (
    ModelArtifactMissingError,
    ModelNotTrainedError,
)
from energy_mix_optimizer.models.forecaster import TimeSeriesForecaster


def _make_synthetic_dataset(n: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed=0)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    radiation = np.maximum(0.0, 800.0 * np.sin(np.pi * (hour - 6) / 12.0))
    X = pd.DataFrame(
        {
            "radiation": radiation + rng.normal(0, 20, n),
            "temperature": 18.0 + rng.normal(0, 2, n),
            "hour": hour,
        },
        index=idx,
    )
    y = pd.Series(
        15.0 * radiation + 200.0 + rng.normal(0, 50, n), index=idx, name="solar"
    )
    return X, y


class TestForecasterLifecycle:
    def test_predict_before_fit_raises(self) -> None:
        forecaster = TimeSeriesForecaster(target_kind="solar")
        X, _ = _make_synthetic_dataset(n=10)
        with pytest.raises(ModelNotTrainedError):
            forecaster.predict(X)

    def test_metadata_before_fit_raises(self) -> None:
        forecaster = TimeSeriesForecaster(target_kind="solar")
        with pytest.raises(ModelNotTrainedError):
            _ = forecaster.metadata

    def test_fit_predict_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, y = _make_synthetic_dataset(n=600)
        forecaster = TimeSeriesForecaster(
            target_kind="solar",
            hyperparameters={"n_estimators": 60, "max_depth": 4, "learning_rate": 0.1},
        )
        metadata = forecaster.fit(X, y, n_cv_splits=3, test_size_fraction=0.2)
        assert metadata.target_kind == "solar"
        assert metadata.test_metrics.mae > 0
        assert metadata.cv_metrics_mean.mae > 0

        predictions = forecaster.predict(X.iloc[-20:])
        assert predictions.shape == (20,)

        forecaster.save(tmp_path)
        loaded = TimeSeriesForecaster.load(tmp_path, target_kind="solar")
        np.testing.assert_allclose(
            forecaster.predict(X.iloc[-5:]), loaded.predict(X.iloc[-5:])
        )

    def test_load_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelArtifactMissingError):
            TimeSeriesForecaster.load(tmp_path, target_kind="solar")

    def test_predict_validates_columns(self, tmp_path: Path) -> None:
        X, y = _make_synthetic_dataset(n=400)
        forecaster = TimeSeriesForecaster(
            target_kind="solar",
            hyperparameters={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2},
        )
        forecaster.fit(X, y, n_cv_splits=3, test_size_fraction=0.2)

        X_missing = X.iloc[-5:].drop(columns=["radiation"])
        with pytest.raises(ValueError, match="Missing required feature columns"):
            forecaster.predict(X_missing)
