"""Hourly economic dispatch via linear programming.

The problem solved at each hour is the standard merit-order economic
dispatch::

    minimize     sum_i (variable_cost_i + carbon_price * emissions_i) * P_i
    subject to   sum_i P_i = demand
                 0 <= P_i <= availability_i           for renewables
                 min_dispatch_i <= P_i <= max_capacity_i for thermal/nuclear/hydro

It is intentionally simple. A production dispatcher would add ramp limits,
minimum up/down times, start-up costs, storage state variables, reserve
requirements, and network constraints. We do **not** model any of those.
The README is explicit about the scope.

Renewable availability comes from the upstream forecasters as a positive
upper bound; the LP is free to dispatch less (curtailment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from energy_mix_optimizer.exceptions import OptimizationInfeasibleError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TechnologySpec:
    """Static parameters of a generation technology.

    Costs and emissions are nominal portfolio averages, not asset-resolved.
    They are intentionally exposed as configuration so a downstream user can
    plug their own merit order without code changes.
    """

    name: str
    variable_cost_eur_per_mwh: float
    emissions_kg_per_mwh: float
    max_capacity_mw: float
    is_dispatchable: bool = True
    min_dispatch_mw: float = 0.0


@dataclass
class DispatchResult:
    """Outcome of a single-hour dispatch."""

    feasible: bool
    dispatch_mw: dict[str, float] = field(default_factory=dict)
    marginal_cost_eur_per_mwh: float | None = None
    total_energy_cost_eur: float = 0.0
    total_emissions_t: float = 0.0
    carbon_cost_eur: float = 0.0
    curtailment_mw: dict[str, float] = field(default_factory=dict)
    unserved_demand_mw: float = 0.0


# --- Default merit order for the Spanish power system ------------------------
#
# These values are sensible defaults for portfolio-level analysis, not asset
# accounting. Variable cost is short-run marginal cost (fuel + variable O&M).
# Emissions are typical lifecycle-equivalent intensities for the operational
# phase. They can be overridden via configuration.
#
# Sources: REE installed-capacity series 2023-2024; IEA fuel and emissions
# factor compendia for combined-cycle and coal plants. The intent is for the
# *order* between technologies to reflect the real Iberian merit order, not
# to forecast a specific spot price.
DEFAULT_TECHNOLOGIES: tuple[TechnologySpec, ...] = (
    TechnologySpec(
        name="solar_pv",
        variable_cost_eur_per_mwh=0.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=25_000.0,
        is_dispatchable=False,
    ),
    TechnologySpec(
        name="wind",
        variable_cost_eur_per_mwh=0.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=30_000.0,
        is_dispatchable=False,
    ),
    TechnologySpec(
        name="hydro",
        variable_cost_eur_per_mwh=5.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=16_000.0,
        is_dispatchable=True,
    ),
    TechnologySpec(
        name="nuclear",
        variable_cost_eur_per_mwh=10.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=7_100.0,
        is_dispatchable=True,
        # Spanish nuclear runs near baseload; we enforce a soft minimum.
        min_dispatch_mw=5_500.0,
    ),
    TechnologySpec(
        name="combined_cycle",
        variable_cost_eur_per_mwh=90.0,
        emissions_kg_per_mwh=370.0,
        max_capacity_mw=26_000.0,
        is_dispatchable=True,
    ),
    TechnologySpec(
        name="coal",
        variable_cost_eur_per_mwh=140.0,
        emissions_kg_per_mwh=900.0,
        max_capacity_mw=2_000.0,
        is_dispatchable=True,
    ),
)


def optimize_dispatch(
    *,
    demand_mw: float,
    renewable_forecast_mw: dict[str, float],
    technologies: tuple[TechnologySpec, ...] = DEFAULT_TECHNOLOGIES,
    carbon_price_eur_per_t: float = 0.0,
) -> DispatchResult:
    """Solve a single-hour economic dispatch.

    Parameters
    ----------
    demand_mw
        Forecast electricity demand for the hour, in MW.
    renewable_forecast_mw
        Mapping from non-dispatchable technology name (e.g. ``"solar_pv"``,
        ``"wind"``) to its forecast available production in MW. Unknown
        technologies are ignored; missing technologies default to zero
        availability.
    technologies
        The full technology set, including dispatchable and non-dispatchable.
    carbon_price_eur_per_t
        Effective EUR/tonne carbon adder applied to the cost vector. Set to
        zero to ignore carbon entirely; the EUA spot has been > 60 EUR/t
        for most of 2024-2025 so the realistic default is non-zero.

    Returns
    -------
    DispatchResult
        Feasible solution with per-technology dispatch in MW, or
        ``feasible=False`` if no solution exists under the constraints.
    """
    if demand_mw < 0:
        raise ValueError("demand_mw must be non-negative")
    if carbon_price_eur_per_t < 0:
        raise ValueError("carbon_price_eur_per_t must be non-negative")

    n = len(technologies)
    # Carbon adder converts to EUR/MWh via 1 kg -> 1e-3 t.
    cost_vector = np.array(
        [
            t.variable_cost_eur_per_mwh + carbon_price_eur_per_t * t.emissions_kg_per_mwh / 1000.0
            for t in technologies
        ],
        dtype=float,
    )

    # Upper bounds depend on whether the tech is dispatchable.
    upper_bounds: list[float] = []
    lower_bounds: list[float] = []
    for tech in technologies:
        if tech.is_dispatchable:
            upper_bounds.append(tech.max_capacity_mw)
            lower_bounds.append(min(tech.min_dispatch_mw, tech.max_capacity_mw))
        else:
            forecast = renewable_forecast_mw.get(tech.name, 0.0)
            forecast = max(0.0, min(forecast, tech.max_capacity_mw))
            upper_bounds.append(forecast)
            lower_bounds.append(0.0)

    bounds = list(zip(lower_bounds, upper_bounds, strict=True))

    # Equality constraint: dispatch must match demand.
    a_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([demand_mw], dtype=float)

    result = linprog(
        c=cost_vector,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # The two realistic failure modes are: demand exceeds total capacity,
        # or the lower-bound sum (must-run) exceeds demand.
        total_max = sum(upper_bounds)
        total_min = sum(lower_bounds)
        unserved = max(0.0, demand_mw - total_max)
        logger.warning(
            "Dispatch infeasible: demand=%.1f, max_capacity=%.1f, min_must_run=%.1f, unserved=%.1f",
            demand_mw,
            total_max,
            total_min,
            unserved,
        )
        return DispatchResult(
            feasible=False,
            dispatch_mw={t.name: 0.0 for t in technologies},
            unserved_demand_mw=unserved,
        )

    dispatch = np.asarray(result.x, dtype=float)
    dispatch_by_name = {t.name: float(dispatch[i]) for i, t in enumerate(technologies)}

    energy_cost = float(
        sum(t.variable_cost_eur_per_mwh * dispatch[i] for i, t in enumerate(technologies))
    )
    emissions_kg = float(sum(t.emissions_kg_per_mwh * dispatch[i] for i, t in enumerate(technologies)))
    emissions_t = emissions_kg / 1000.0
    carbon_cost = carbon_price_eur_per_t * emissions_t

    # Marginal cost is the dual value of the demand-balance constraint, which
    # SciPy/HiGHS reports via the ``eqlin`` field. Per the SciPy convention,
    # ``marginals[i]`` is the change in objective for a unit increase of
    # ``b_eq[i]``; for our demand balance this is exactly the EUR/MWh cost
    # of serving one additional MW.
    marginal_cost: float | None = None
    eq_marginals = getattr(result, "eqlin", None)
    if eq_marginals is not None and getattr(eq_marginals, "marginals", None) is not None:
        marginals = np.asarray(eq_marginals.marginals, dtype=float)
        if marginals.size:
            marginal_cost = float(marginals[0])

    curtailment = {
        t.name: max(0.0, renewable_forecast_mw.get(t.name, 0.0) - dispatch_by_name[t.name])
        for t in technologies
        if not t.is_dispatchable
    }

    return DispatchResult(
        feasible=True,
        dispatch_mw=dispatch_by_name,
        marginal_cost_eur_per_mwh=marginal_cost,
        total_energy_cost_eur=energy_cost,
        total_emissions_t=emissions_t,
        carbon_cost_eur=carbon_cost,
        curtailment_mw=curtailment,
        unserved_demand_mw=0.0,
    )


def optimize_dispatch_horizon(
    *,
    demand_mw: list[float],
    renewable_forecast_mw: list[dict[str, float]],
    technologies: tuple[TechnologySpec, ...] = DEFAULT_TECHNOLOGIES,
    carbon_price_eur_per_t: float = 0.0,
    raise_on_infeasible: bool = False,
) -> list[DispatchResult]:
    """Solve the dispatch independently for each hour of a horizon.

    No inter-temporal coupling (ramps, storage) is modeled. The horizon is
    decomposed into independent single-hour LPs.
    """
    if len(demand_mw) != len(renewable_forecast_mw):
        raise ValueError(
            f"demand_mw and renewable_forecast_mw must have the same length; "
            f"got {len(demand_mw)} and {len(renewable_forecast_mw)}"
        )
    results: list[DispatchResult] = []
    for hour, (d, r) in enumerate(zip(demand_mw, renewable_forecast_mw, strict=True)):
        result = optimize_dispatch(
            demand_mw=d,
            renewable_forecast_mw=r,
            technologies=technologies,
            carbon_price_eur_per_t=carbon_price_eur_per_t,
        )
        if not result.feasible and raise_on_infeasible:
            raise OptimizationInfeasibleError(
                f"Dispatch infeasible at hour {hour}: demand={d:.1f} MW"
            )
        results.append(result)
    return results
