"""Identical-successor aggregation.

CS011 benchmark contract, "Identical successors": after aligning all successor
primitives, ``P`` and ``F`` aggregate into one effective mark with **summed** physical
and risk-neutral intensities. Failure means incorrect mark aggregation.

The test is at the level of the equations rather than the technologies: two marks whose
payoff and successor wealth coincide must behave exactly as their sum, and the
unspanned residual and investment wedge must vanish.
"""

from __future__ import annotations

import pytest

from ak_partial_ramsey.canonical import investment_wedge, total_valuation_residual
from ak_partial_ramsey.exposure import Mark, solve_private_portfolio, solve_public_exposure

RHO = 0.04
C = 0.4


def test_identical_marks_aggregate_to_summed_intensities(tol):
    J, h = 0.35, 9.0
    pair = (Mark("P", 0.012, 0.010, J, h), Mark("F", 0.008, 0.015, J, h))
    aggregated = (Mark("E", 0.012 + 0.008, 0.010 + 0.015, J, h),)

    a = solve_public_exposure(pair, C, RHO, tol)
    b = solve_public_exposure(aggregated, C, RHO, tol)
    assert a.psi == pytest.approx(b.psi, rel=1e-13)
    assert a.successor_wealth["P"] == pytest.approx(b.successor_wealth["E"], rel=1e-13)


def test_identical_marks_leave_no_unspanned_residual(tol):
    """U = 0 and D = 0: with coincident marks there is nothing orthogonal to the payoff."""
    J, h, K, varphi = 0.35, 9.0, 10.0, 2.5
    pair = (Mark("P", 0.012, 0.010, J, h), Mark("F", 0.008, 0.015, J, h))
    sol = solve_public_exposure(pair, C, RHO, tol)
    U = total_valuation_residual(pair, sol.psi, C, RHO)
    assert U == pytest.approx(0.0, abs=1e-15)
    assert investment_wedge(sol.psi, U, K, varphi) == pytest.approx(0.0, abs=1e-15)


def test_identical_marks_kill_the_payoff_slope_term(tol):
    """sum_j u_j J_{j,K} = J_K * U = 0 when the marks and their slopes coincide."""
    J, h = 0.35, 9.0
    pair = (Mark("P", 0.012, 0.010, J, h), Mark("F", 0.008, 0.015, J, h))
    sol = solve_public_exposure(pair, C, RHO, tol)
    J_K = 0.02  # a common slope, manufactured
    slope_term = sum(sol.fiscal_valuation_residual[m.label] * J_K for m in pair)
    assert slope_term == pytest.approx(0.0, abs=1e-15)


def test_identical_marks_aggregate_in_the_private_portfolio(tol):
    J, h = 0.35, 9.0
    pair = (Mark("P", 0.012, 0.010, J, h), Mark("F", 0.008, 0.015, J, h))
    aggregated = (Mark("E", 0.020, 0.025, J, h),)
    assert solve_private_portfolio(pair, tol).pi == pytest.approx(
        solve_private_portfolio(aggregated, tol).pi, rel=1e-13
    )


def test_distinct_marks_do_not_aggregate(tol):
    """The negative control: with different payoffs the residual is genuinely nonzero."""
    pair = (Mark("P", 0.012, 0.010, 0.1, 10.0), Mark("F", 0.008, 0.015, 0.5, 8.0))
    sol = solve_public_exposure(pair, C, RHO, tol)
    U = total_valuation_residual(pair, sol.psi, C, RHO)
    assert abs(U) > 1e-6
    assert abs(investment_wedge(sol.psi, U, 10.0, 2.5)) > 1e-6
