"""Generic XGBoost time-series forecaster.

The same class is reused for solar, wind and demand: the differences live
in the feature specifications and hyperparameters. The class is responsible
for cross-validated evaluation, a final hold-out test, and serialization
together with metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from energy_mix_optimizer import __version__ as package_version
from energy_mix_optimizer.exceptions import ModelArtifactMissingError, ModelNotTrainedError

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Out-of-sample metrics for one CV fold or one hold-out test."""

    n_samples: int
    mae: float
    rmse: float
    nmae_percent: float | None = None  # MAE normalized by mean of target
    mape_percent: float | None = None  # only finite when target has no zeros


@dataclass
class ModelMetadata:
    """Everything we want to know about a trained model after the fact."""

    target_kind: str
    feature_columns: list[str]
    trained_at_utc: str
    training_data_start: str
    training_data_end: str
    test_metrics: TrainingMetrics
    cv_metrics_mean: TrainingMetrics
    cv_metrics_std: dict[str, float]
    xgboost_version: str
    package_version: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)


_DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


class TimeSeriesForecaster:
    """A thin wrapper around ``xgb.XGBRegressor`` with TS-aware training.

    The wrapper enforces:

    * A frozen feature column order, validated at predict time.
    * TimeSeriesSplit cross-validation with honest aggregated metrics.
    * A final hold-out test on the last ``test_size_fraction`` of the data.
    """

    def __init__(
        self,
        target_kind: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        self.target_kind = target_kind
        self.hyperparameters = {**_DEFAULT_HYPERPARAMETERS, **(hyperparameters or {})}
        self._model: xgb.XGBRegressor | None = None
        self._feature_columns: list[str] | None = None
        self._metadata: ModelMetadata | None = None

    # ----- Public API --------------------------------------------------------

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise ModelNotTrainedError(f"Forecaster for '{self.target_kind}' is not trained yet")
        return self._metadata

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            raise ModelNotTrainedError(f"Forecaster for '{self.target_kind}' is not trained yet")
        return list(self._feature_columns)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        n_cv_splits: int = 5,
        test_size_fraction: float = 0.15,
    ) -> ModelMetadata:
        if not 0.0 < test_size_fraction < 0.5:
            raise ValueError("test_size_fraction must be in (0, 0.5)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) < n_cv_splits + 50:
            raise ValueError(
                f"Not enough samples ({len(X)}) for {n_cv_splits}-fold CV "
                "plus a hold-out test"
            )

        X = X.sort_index()
        y = y.reindex(X.index)

        cv_metrics = self._cross_validate(X, y, n_cv_splits=n_cv_splits)
        cv_mean = _aggregate_metrics(cv_metrics, agg="mean")
        cv_std = _aggregate_metrics(cv_metrics, agg="std", as_dict=True)

        # Final hold-out: train on the first (1 - test_size_fraction) of the
        # data, evaluate on the last test_size_fraction.
        n_test = max(1, int(len(X) * test_size_fraction))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

        model = self._build_xgboost_model()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        predictions = model.predict(X_test)
        test_metrics = _compute_metrics(y_test.to_numpy(), np.asarray(predictions))

        self._model = model
        self._feature_columns = list(X.columns)
        self._metadata = ModelMetadata(
            target_kind=self.target_kind,
            feature_columns=list(X.columns),
            trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            training_data_start=str(X.index.min()),
            training_data_end=str(X.index.max()),
            test_metrics=test_metrics,
            cv_metrics_mean=cv_mean,
            cv_metrics_std=cv_std,
            xgboost_version=xgb.__version__,
            package_version=package_version,
            hyperparameters=dict(self.hyperparameters),
        )
        logger.info(
            "Trained '%s' on %d samples. CV MAE=%.3f +/- %.3f, hold-out RMSE=%.3f",
            self.target_kind,
            len(X),
            cv_mean.mae,
            cv_std.get("mae", float("nan")),
            test_metrics.rmse,
        )
        return self._metadata

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._feature_columns is None:
            raise ModelNotTrainedError(f"Forecaster for '{self.target_kind}' is not trained yet")

        missing = set(self._feature_columns) - set(X.columns)
        if missing:
            raise ValueError(
                f"Missing required feature columns for '{self.target_kind}': "
                f"{sorted(missing)}"
            )
        # Force the column order to match training.
        X_ordered = X[self._feature_columns]
        return np.asarray(self._model.predict(X_ordered), dtype=float)

    # ----- Persistence -------------------------------------------------------

    def save(self, directory: Path) -> Path:
        if self._model is None or self._metadata is None:
            raise ModelNotTrainedError(f"Cannot save untrained model '{self.target_kind}'")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / f"{self.target_kind}.joblib"
        metadata_path = directory / f"{self.target_kind}.metadata.json"

        joblib.dump(
            {
                "model": self._model,
                "feature_columns": self._feature_columns,
                "target_kind": self.target_kind,
            },
            model_path,
        )
        payload = _replace_nan_with_none(asdict(self._metadata))
        metadata_path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        logger.info("Saved '%s' model to %s", self.target_kind, model_path)
        return model_path

    @classmethod
    def load(cls, directory: Path, target_kind: str) -> TimeSeriesForecaster:
        directory = Path(directory)
        model_path = directory / f"{target_kind}.joblib"
        metadata_path = directory / f"{target_kind}.metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise ModelArtifactMissingError(
                f"Missing artifact for '{target_kind}' in {directory}. "
                "Run the training pipeline first."
            )

        bundle = joblib.load(model_path)
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))

        forecaster = cls(target_kind=bundle["target_kind"])
        forecaster._model = bundle["model"]
        forecaster._feature_columns = list(bundle["feature_columns"])
        forecaster._metadata = _metadata_from_dict(metadata_raw)
        return forecaster

    # ----- Internals ---------------------------------------------------------

    def _build_xgboost_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **self.hyperparameters,
        )

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        n_cv_splits: int,
    ) -> list[TrainingMetrics]:
        splitter = TimeSeriesSplit(n_splits=n_cv_splits)
        fold_metrics: list[TrainingMetrics] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = self._build_xgboost_model()
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            predictions = np.asarray(model.predict(X_val), dtype=float)
            metrics = _compute_metrics(y_val.to_numpy(), predictions)
            fold_metrics.append(metrics)
            logger.debug(
                "CV fold %d/%d: MAE=%.3f RMSE=%.3f", fold_idx, n_cv_splits, metrics.mae, metrics.rmse
            )
        return fold_metrics


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> TrainingMetrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mean_target = float(np.mean(np.abs(y_true))) if len(y_true) else 0.0
    nmae_percent = (mae / mean_target * 100.0) if mean_target > 0 else None

    nonzero = y_true != 0
    if nonzero.all() and len(y_true) > 0:
        mape_percent = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    else:
        mape_percent = None

    return TrainingMetrics(
        n_samples=len(y_true),
        mae=mae,
        rmse=rmse,
        nmae_percent=nmae_percent,
        mape_percent=mape_percent,
    )


def _aggregate_metrics(
    metrics_list: list[TrainingMetrics],
    *,
    agg: str,
    as_dict: bool = False,
) -> TrainingMetrics | dict[str, float]:
    if not metrics_list:
        empty = TrainingMetrics(n_samples=0, mae=float("nan"), rmse=float("nan"))
        return {} if as_dict else empty

    def _val(field_name: str) -> float:
        values = [getattr(m, field_name) for m in metrics_list if getattr(m, field_name) is not None]
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()) if agg == "mean" else float(arr.std(ddof=0))

    if as_dict:
        result = {
            "mae": _val("mae"),
            "rmse": _val("rmse"),
            "nmae_percent": _val("nmae_percent"),
            "mape_percent": _val("mape_percent"),
        }
        # Drop entries that are NaN so the resulting JSON has only real
        # numbers; downstream consumers can rely on every key being finite.
        return {k: v for k, v in result.items() if not np.isnan(v)}

    return TrainingMetrics(
        n_samples=int(np.mean([m.n_samples for m in metrics_list])),
        mae=_val("mae"),
        rmse=_val("rmse"),
        nmae_percent=_val("nmae_percent"),
        mape_percent=_val("mape_percent"),
    )


def _metadata_from_dict(raw: dict[str, Any]) -> ModelMetadata:
    test = raw["test_metrics"]
    cv = raw["cv_metrics_mean"]
    return ModelMetadata(
        target_kind=raw["target_kind"],
        feature_columns=list(raw["feature_columns"]),
        trained_at_utc=raw["trained_at_utc"],
        training_data_start=raw["training_data_start"],
        training_data_end=raw["training_data_end"],
        test_metrics=TrainingMetrics(**test),
        cv_metrics_mean=TrainingMetrics(**cv),
        cv_metrics_std=dict(raw.get("cv_metrics_std", {})),
        xgboost_version=raw["xgboost_version"],
        package_version=raw["package_version"],
        hyperparameters=dict(raw.get("hyperparameters", {})),
    )


def _replace_nan_with_none(value: Any) -> Any:
    """Recursively replace NaN floats with ``None`` so the result is JSON-clean."""
    if isinstance(value, dict):
        return {k: _replace_nan_with_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_nan_with_none(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
