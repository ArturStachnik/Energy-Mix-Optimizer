"""Tests for the dispatch LP."""

from __future__ import annotations

import pytest

from energy_mix_optimizer.exceptions import OptimizationInfeasibleError
from energy_mix_optimizer.optimization.dispatch import (
    DEFAULT_TECHNOLOGIES,
    TechnologySpec,
    optimize_dispatch,
    optimize_dispatch_horizon,
)

# A simplified technology stack with no must-run constraint, used to keep the
# arithmetic in tests trivially verifiable.
_TEST_TECHS = (
    TechnologySpec(
        name="solar_pv",
        variable_cost_eur_per_mwh=0.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=10_000.0,
        is_dispatchable=False,
    ),
    TechnologySpec(
        name="wind",
        variable_cost_eur_per_mwh=0.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=10_000.0,
        is_dispatchable=False,
    ),
    TechnologySpec(
        name="hydro",
        variable_cost_eur_per_mwh=5.0,
        emissions_kg_per_mwh=0.0,
        max_capacity_mw=5_000.0,
    ),
    TechnologySpec(
        name="combined_cycle",
        variable_cost_eur_per_mwh=90.0,
        emissions_kg_per_mwh=370.0,
        max_capacity_mw=20_000.0,
    ),
    TechnologySpec(
        name="coal",
        variable_cost_eur_per_mwh=140.0,
        emissions_kg_per_mwh=900.0,
        max_capacity_mw=5_000.0,
    ),
)


class TestMeritOrder:
    def test_renewables_are_dispatched_first(self) -> None:
        """With ample renewables, the LP should use solar+wind and zero coal."""
        result = optimize_dispatch(
            demand_mw=10_000.0,
            renewable_forecast_mw={"solar_pv": 6_000.0, "wind": 4_000.0},
            technologies=_TEST_TECHS,
        )
        assert result.feasible
        assert result.dispatch_mw["solar_pv"] == pytest.approx(6_000.0)
        assert result.dispatch_mw["wind"] == pytest.approx(4_000.0)
        assert result.dispatch_mw["hydro"] == pytest.approx(0.0)
        assert result.dispatch_mw["combined_cycle"] == pytest.approx(0.0)
        assert result.dispatch_mw["coal"] == pytest.approx(0.0)
        assert result.total_energy_cost_eur == pytest.approx(0.0)
        assert result.total_emissions_t == pytest.approx(0.0)

    def test_fills_residual_with_hydro_before_gas(self) -> None:
        """Residual demand above renewables uses hydro (cheaper) before gas."""
        result = optimize_dispatch(
            demand_mw=8_000.0,
            renewable_forecast_mw={"solar_pv": 2_000.0, "wind": 1_000.0},
            technologies=_TEST_TECHS,
        )
        assert result.feasible
        assert result.dispatch_mw["solar_pv"] == pytest.approx(2_000.0)
        assert result.dispatch_mw["wind"] == pytest.approx(1_000.0)
        assert result.dispatch_mw["hydro"] == pytest.approx(5_000.0)
        assert result.dispatch_mw["combined_cycle"] == pytest.approx(0.0)
        # 5000 MW * 5 EUR/MWh = 25_000 EUR
        assert result.total_energy_cost_eur == pytest.approx(25_000.0)

    def test_uses_gas_when_cheap_options_exhausted(self) -> None:
        result = optimize_dispatch(
            demand_mw=20_000.0,
            renewable_forecast_mw={"solar_pv": 1_000.0, "wind": 1_000.0},
            technologies=_TEST_TECHS,
        )
        assert result.feasible
        # All cheap sources at cap, residual on combined_cycle.
        assert result.dispatch_mw["solar_pv"] == pytest.approx(1_000.0)
        assert result.dispatch_mw["wind"] == pytest.approx(1_000.0)
        assert result.dispatch_mw["hydro"] == pytest.approx(5_000.0)
        assert result.dispatch_mw["combined_cycle"] == pytest.approx(13_000.0)
        assert result.dispatch_mw["coal"] == pytest.approx(0.0)


class TestInfeasibility:
    def test_demand_exceeds_capacity_returns_unserved(self) -> None:
        result = optimize_dispatch(
            demand_mw=999_999.0,
            renewable_forecast_mw={"solar_pv": 0.0, "wind": 0.0},
            technologies=_TEST_TECHS,
        )
        assert not result.feasible
        assert result.unserved_demand_mw > 0

    def test_horizon_raises_when_requested(self) -> None:
        with pytest.raises(OptimizationInfeasibleError):
            optimize_dispatch_horizon(
                demand_mw=[999_999.0],
                renewable_forecast_mw=[{"solar_pv": 0.0, "wind": 0.0}],
                technologies=_TEST_TECHS,
                raise_on_infeasible=True,
            )

    def test_horizon_returns_infeasible_results_by_default(self) -> None:
        results = optimize_dispatch_horizon(
            demand_mw=[10_000.0, 999_999.0],
            renewable_forecast_mw=[
                {"solar_pv": 5_000.0, "wind": 5_000.0},
                {"solar_pv": 0.0, "wind": 0.0},
            ],
            technologies=_TEST_TECHS,
        )
        assert results[0].feasible
        assert not results[1].feasible


class TestCurtailment:
    def test_renewable_curtails_when_excess(self) -> None:
        """If forecast renewable > demand, some renewable is curtailed."""
        result = optimize_dispatch(
            demand_mw=5_000.0,
            renewable_forecast_mw={"solar_pv": 4_000.0, "wind": 4_000.0},
            technologies=_TEST_TECHS,
        )
        assert result.feasible
        served = result.dispatch_mw["solar_pv"] + result.dispatch_mw["wind"]
        assert served == pytest.approx(5_000.0)
        curtailed = sum(result.curtailment_mw.values())
        assert curtailed == pytest.approx(3_000.0)


class TestMustRun:
    def test_nuclear_minimum_dispatch_is_honored(self) -> None:
        """With the default technologies, nuclear must dispatch >= 5500 MW."""
        result = optimize_dispatch(
            demand_mw=20_000.0,
            renewable_forecast_mw={"solar_pv": 10_000.0, "wind": 10_000.0},
            technologies=DEFAULT_TECHNOLOGIES,
        )
        assert result.feasible
        # Total renewables + must-run nuclear meet demand; some renewable
        # must therefore be curtailed (nuclear must-run = 5500 MW).
        assert result.dispatch_mw["nuclear"] >= 5_500.0 - 1e-6


class TestCarbonPrice:
    def test_carbon_price_increases_cost_and_marginal(self) -> None:
        zero_carbon = optimize_dispatch(
            demand_mw=15_000.0,
            renewable_forecast_mw={"solar_pv": 1_000.0, "wind": 1_000.0},
            technologies=_TEST_TECHS,
            carbon_price_eur_per_t=0.0,
        )
        with_carbon = optimize_dispatch(
            demand_mw=15_000.0,
            renewable_forecast_mw={"solar_pv": 1_000.0, "wind": 1_000.0},
            technologies=_TEST_TECHS,
            carbon_price_eur_per_t=100.0,
        )
        # Dispatch should be the same (carbon shifts cost but not order between
        # renewables and hydro and gas; coal is still more expensive than gas).
        assert zero_carbon.dispatch_mw == pytest.approx(with_carbon.dispatch_mw)
        # The marginal cost in EUR/MWh should reflect the carbon adder on gas.
        assert with_carbon.marginal_cost_eur_per_mwh is not None
        assert zero_carbon.marginal_cost_eur_per_mwh is not None
        assert with_carbon.marginal_cost_eur_per_mwh > zero_carbon.marginal_cost_eur_per_mwh
