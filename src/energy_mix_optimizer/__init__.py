"""Energy Mix Optimizer.

Day-ahead renewable generation forecast and dispatch optimization for the
Iberian power system, using public Energy-Charts and Open-Meteo APIs.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("energy-mix-optimizer")
except PackageNotFoundError:  # pragma: no cover - editable install fallback
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
