"""Centralized configuration.

Settings are loaded from environment variables (prefixed with ``EMO_``) or
from a local ``.env`` file. All defaults are production-safe and require no
authentication, since both the Energy-Charts API (Fraunhofer ISE) and
Open-Meteo are public APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All fields can be overridden via environment variables using the
    ``EMO_`` prefix (e.g. ``EMO_HTTP_TIMEOUT_SECONDS=60``).
    """

    model_config = SettingsConfigDict(
        env_prefix="EMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # HTTP
    http_timeout_seconds: float = Field(default=30.0, gt=0.0)
    http_max_retries: int = Field(default=3, ge=0, le=10)
    http_user_agent: str = "energy-mix-optimizer/0.3"

    # Energy-Charts API (Fraunhofer ISE, public, no auth required).
    # Reference: https://api.energy-charts.info
    energy_charts_base_url: str = "https://api.energy-charts.info"

    # Open-Meteo (public, no auth required).
    # Reference: https://open-meteo.com/en/docs
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"

    # Model artifacts
    artifacts_dir: Path = Field(default=Path("artifacts"))

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    # Optimization defaults
    default_carbon_price_eur_per_t: float = Field(default=80.0, ge=0.0)

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def _coerce_artifacts_dir(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    @property
    def models_dir(self) -> Path:
        """Directory where trained model artifacts live."""
        return self.artifacts_dir / "models"


def get_settings() -> Settings:
    """Return a fresh ``Settings`` instance.

    Kept as a function (not a module-level singleton) so it can be overridden
    in tests and FastAPI dependencies.
    """
    return Settings()
