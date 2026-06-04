"""Domain-specific exceptions.

A small hierarchy is sufficient. The API layer maps these to HTTP responses.
"""

from __future__ import annotations


class EnergyMixOptimizerError(Exception):
    """Base class for all package exceptions."""


class DataSourceError(EnergyMixOptimizerError):
    """A data source (Energy-Charts, Open-Meteo) returned an error or malformed payload."""


class ModelNotTrainedError(EnergyMixOptimizerError):
    """A forecaster was used before being fitted."""


class ModelArtifactMissingError(EnergyMixOptimizerError):
    """A persisted model artifact was requested but not found on disk."""


class OptimizationInfeasibleError(EnergyMixOptimizerError):
    """The dispatch LP has no feasible solution under the supplied constraints."""


class InvalidRequestError(EnergyMixOptimizerError):
    """Caller-supplied parameters are inconsistent or out of range."""
