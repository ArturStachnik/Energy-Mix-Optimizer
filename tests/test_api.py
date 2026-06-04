"""FastAPI integration tests.

We exercise the HTTP surface with a real ``TestClient`` but stub out the
upstream data sources and the trained models. The goal is to validate
routing, schemas, error handlers and dependency wiring - not to retrain
the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from energy_mix_optimizer.api import main as api_main
from energy_mix_optimizer.exceptions import ModelArtifactMissingError
from energy_mix_optimizer.pipelines import forecast as forecast_module
from energy_mix_optimizer.pipelines.forecast import LoadedModels


@dataclass
class _StubMetadata:
    target_kind: str
    feature_columns: list[str]
    trained_at_utc: str
    training_data_start: str
    training_data_end: str
    test_metrics: Any
    cv_metrics_mean: Any
    cv_metrics_std: dict[str, float]
    xgboost_version: str
    package_version: str
    hyperparameters: dict[str, Any]


@dataclass
class _StubTrainingMetrics:
    n_samples: int = 100
    mae: float = 1.0
    rmse: float = 1.5
    nmae_percent: float | None = 5.0
    mape_percent: float | None = 7.5


@dataclass
class _StubForecaster:
    target_kind: str

    @property
    def metadata(self) -> _StubMetadata:
        return _StubMetadata(
            target_kind=self.target_kind,
            feature_columns=["x"],
            trained_at_utc="2024-06-15T12:00:00+00:00",
            training_data_start="2022-01-01",
            training_data_end="2024-06-01",
            test_metrics=_StubTrainingMetrics(),
            cv_metrics_mean=_StubTrainingMetrics(),
            cv_metrics_std={"mae": 0.1},
            xgboost_version="2.1.0",
            package_version="0.2.0",
            hyperparameters={"n_estimators": 100},
        )


@dataclass
class _StubLoadedModels:
    solar: _StubForecaster
    wind: _StubForecaster
    demand: _StubForecaster

    def versions(self) -> dict[str, str]:
        return {
            "solar": self.solar.metadata.trained_at_utc,
            "wind": self.wind.metadata.trained_at_utc,
            "demand": self.demand.metadata.trained_at_utc,
        }


@pytest.fixture
def client_without_models() -> TestClient:
    # Build a TestClient that runs lifespan with no models on disk.
    # We monkey-patch LoadedModels.load to raise before TestClient starts the
    # app.
    original = LoadedModels.load

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise ModelArtifactMissingError("no models for test")

    LoadedModels.load = staticmethod(_raise)  # type: ignore[method-assign]
    try:
        with TestClient(api_main.app) as client:
            yield client
    finally:
        LoadedModels.load = original  # type: ignore[method-assign]


@pytest.fixture
def client_with_stub_models(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    stub = _StubLoadedModels(
        solar=_StubForecaster("solar"),
        wind=_StubForecaster("wind"),
        demand=_StubForecaster("demand"),
    )

    def _load_stub(_directory: Any) -> _StubLoadedModels:
        return stub

    monkeypatch.setattr(forecast_module.LoadedModels, "load", _load_stub)

    with TestClient(api_main.app) as client:
        yield client


class TestHealth:
    def test_health_works_without_models(self, client_without_models: TestClient) -> None:
        response = client_without_models.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["models_loaded"] is False

    def test_health_works_with_models(self, client_with_stub_models: TestClient) -> None:
        response = client_with_stub_models.get("/health")
        assert response.status_code == 200
        assert response.json()["models_loaded"] is True


class TestModelsInfo:
    def test_returns_503_without_models(self, client_without_models: TestClient) -> None:
        response = client_without_models.get("/models/info")
        assert response.status_code == 503

    def test_returns_metadata_with_models(self, client_with_stub_models: TestClient) -> None:
        response = client_with_stub_models.get("/models/info")
        assert response.status_code == 200
        body = response.json()
        assert set(body["models"].keys()) == {"solar", "wind", "demand"}
        assert body["models"]["solar"]["test_metrics"]["rmse"] == 1.5


class TestOptimizeEndpoint:
    def test_runs_without_models(self, client_without_models: TestClient) -> None:
        """The dispatch optimizer does not depend on trained models."""
        payload = {
            "demand_mw": [10_000.0, 12_000.0],
            "solar_forecast_mw": [3_000.0, 5_000.0],
            "wind_forecast_mw": [4_000.0, 5_000.0],
            "carbon_price_eur_per_t": 80.0,
        }
        response = client_without_models.post("/optimize", json=payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["hourly"]) == 2
        assert body["summary"]["total_demand_mwh"] == 22_000.0

    def test_rejects_mismatched_lengths(self, client_without_models: TestClient) -> None:
        response = client_without_models.post(
            "/optimize",
            json={
                "demand_mw": [10_000.0, 12_000.0],
                "solar_forecast_mw": [3_000.0],
                "wind_forecast_mw": [4_000.0, 5_000.0],
            },
        )
        assert response.status_code == 400

    def test_rejects_negative_values(self, client_without_models: TestClient) -> None:
        response = client_without_models.post(
            "/optimize",
            json={
                "demand_mw": [10_000.0],
                "solar_forecast_mw": [-5.0],
                "wind_forecast_mw": [4_000.0],
            },
        )
        assert response.status_code == 422


class TestForecastEndpoint:
    def test_returns_503_without_models(self, client_without_models: TestClient) -> None:
        response = client_without_models.post(
            "/forecast",
            json={"horizon_hours": 24},
        )
        assert response.status_code == 503
