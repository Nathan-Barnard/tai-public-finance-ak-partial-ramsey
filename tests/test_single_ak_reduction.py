"""Exact reduction of the two-mark equations to the single-AK system.

CS011 benchmark contract, "Single-AK support": as ``p_P`` and ``lambda_P_star`` go to
zero, recover the reconciled one-mark price, value, zero orthogonal residual, and tax
equations. Failure means incorrect nesting, units, or zero-intensity handling.

This is an *algebraic support restriction*, not a limit: the values are set to exactly
zero, the ``P`` mark is retained in the configuration as a first-class inactive branch,
and nothing is divided by either vanishing quantity.
"""

from __future__ import annotations

import pytest

from ak_partial_ramsey.assembly import build_marks, evaluate_state
from ak_partial_ramsey.canonical import (
    consumption_growth,
    costate_e_dot,
    costate_K_dot,
    investment_wedge,
    risk_neutral_drag,
    single_ak_consumption_growth,
    single_ak_exposure_closed_form,
    single_ak_q_dot,
    total_valuation_residual,
)
from ak_partial_ramsey.evaluator import check_single_ak_reduction
from ak_partial_ramsey.exposure import (
    fiscal_valuation_residual,
    solve_public_exposure,
)
from ak_partial_ramsey.recovery import recover_ak_source_tax, recover_source_tax
from ak_partial_ramsey.successors.ak import solve_ak_successor
from ak_partial_ramsey.successors.partial import solve_partial_successor

STATE = {"K": 320.0, "e": -400.0, "q": 1.10, "C": 12.0, "a": 50.0}


@pytest.fixture(scope="module")
def support(support_fixture, tol):
    p = support_fixture.params
    ak = solve_ak_successor(p, support_fixture.ak_root_interval, tol)
    partial = solve_partial_successor(p, support_fixture.partial_capital_interval, tol)
    marks = build_marks(p, ak, partial, STATE["K"], STATE["e"], STATE["q"])
    sol = solve_public_exposure(marks.marks, STATE["C"], p.preferences.rho, tol)
    return p, ak, partial, marks, sol


def test_partial_mark_is_inactive_not_absent(support):
    """The P mark is retained and reported, but carries no weight."""
    _, _, _, marks, _ = support
    labels = {m.label for m in marks.marks}
    assert labels == {"P", "F"}
    P = next(m for m in marks.marks if m.label == "P")
    F = next(m for m in marks.marks if m.label == "F")
    assert P.lambda_physical == 0.0
    assert P.lambda_star == 0.0
    assert not P.is_active
    assert F.is_active
    # The intensity object itself records the restriction.
    assert P.J != 0.0  # the payoff is a real number; only the intensities vanish


def test_u_P_vanishes_identically_without_division(support):
    """u_P = 0 by construction, reached without dividing by p_P or lambda_P_star."""
    p, _, _, marks, sol = support
    P = next(m for m in marks.marks if m.label == "P")
    u_P = fiscal_valuation_residual(P, sol.psi, STATE["C"], p.preferences.rho)
    assert u_P == 0.0


def test_exposure_condition_forces_u_F_to_zero(support):
    """With one active mark and J_F != 0, the projection condition gives u_F = 0."""
    p, _, _, marks, sol = support
    F = next(m for m in marks.marks if m.label == "F")
    assert F.J != 0.0
    u_F = fiscal_valuation_residual(F, sol.psi, STATE["C"], p.preferences.rho)
    assert u_F == pytest.approx(0.0, abs=1e-15)


def test_total_residual_and_investment_wedge_vanish(support):
    """U = D = 0, which is what restores the single-mark tax cancellation."""
    p, _, _, marks, sol = support
    U = total_valuation_residual(marks.marks, sol.psi, STATE["C"], p.preferences.rho)
    D = investment_wedge(sol.psi, U, STATE["K"], p.installation.varphi)
    assert U == pytest.approx(0.0, abs=1e-15)
    assert D == pytest.approx(0.0, abs=1e-15)


def test_fiscal_to_market_alignment(support):
    """P.6: lambda V_{F,e} = lambda_F_star mu_e, i.e. the ratio equals lambda_F*/lambda."""
    p, _, _, _marks, sol = support
    X_F = sol.successor_wealth["F"]
    V_F_e = 1.0 / (p.preferences.rho * X_F)
    mu_e = 1.0 / STATE["C"]
    i = p.intensities
    assert i.lambda_total * V_F_e == pytest.approx(i.lambda_F_star * mu_e, rel=1e-13)
    assert V_F_e / mu_e == pytest.approx(i.lambda_F_star / i.lambda_total, rel=1e-13)


def test_closed_form_exposure_agrees_with_unmultiplied_root(support):
    """P.20's closed form (which divides by J_F) matches the unmultiplied bracket root."""
    p, ak, _, marks, sol = support
    F = next(m for m in marks.marks if m.label == "F")
    # P.14 rearranged: successor wealth is (lambda/lambda_F_star) * X_0 with X_0 = C/rho.
    X_0 = STATE["C"] / p.preferences.rho
    psi_closed = single_ak_exposure_closed_form(X_0, STATE["e"], STATE["K"], ak.q_F, F.J, p)
    assert psi_closed == pytest.approx(sol.psi, rel=1e-11)


def test_consumption_growth_reduces_to_P9(support):
    """The two-mark growth equation collapses to P.9 once U = 0."""
    p, _, _, marks, sol = support
    U = total_valuation_residual(marks.marks, sol.psi, STATE["C"], p.preferences.rho)
    mu_e = 1.0 / STATE["C"]
    general = consumption_growth(p, U, mu_e)
    specialised = single_ak_consumption_growth(p)
    assert general == pytest.approx(specialised, abs=1e-15)


def test_capital_costate_reduces_to_the_single_mark_price_law(support):
    """The two-mark costate pair reproduces P.10's q-law exactly."""
    p, ak, _partial, marks, sol = support
    K, q, C = STATE["K"], STATE["q"], STATE["C"]
    rho = p.preferences.rho
    mu_e = 1.0 / C
    mu_K = q * mu_e  # P.8, which holds because D = 0

    X = {m.label: sol.successor_wealth[m.label] for m in marks.marks if m.is_active}
    V_e = {lab: 1.0 / (rho * X[lab]) for lab in X}
    V_K = {lab: marks.successor_prices[lab] * V_e[lab] for lab in X}
    u = {m.label: fiscal_valuation_residual(m, sol.psi, C, rho) for m in marks.marks if m.is_active}
    J_K = {lab: marks.successor_price_slopes[lab] / q for lab in X}

    mu_e_dot = costate_e_dot(mu_e, V_e, marks.marks, p)
    mu_K_dot = costate_K_dot(K, q, mu_K, mu_e, sol.psi, V_K, u, J_K, marks.marks, p)
    # mu_K = q mu_e implies mu_K' = q' mu_e + q mu_e', so q' is recoverable.
    q_dot_from_costates = (mu_K_dot - q * mu_e_dot) / mu_e
    q_dot_P10 = single_ak_q_dot(K, q, ak.q_F, p)
    assert q_dot_from_costates == pytest.approx(q_dot_P10, rel=1e-11)


def test_zero_source_tax_is_recovered(support):
    """P.12: substituting the P.10 price law into world pricing gives tau_0 = 0 exactly."""
    p, ak, _, marks, _sol = support
    K, q = STATE["K"], STATE["q"]
    q_dot = single_ak_q_dot(K, q, ak.q_F, p)
    Lambda = risk_neutral_drag(marks.marks)

    tau_two_mark = recover_source_tax(K, q, q_dot, Lambda, p)
    tau_single = recover_ak_source_tax(K, q, q_dot, ak.q_F, p)
    assert tau_two_mark == pytest.approx(0.0, abs=1e-14)
    assert tau_single == pytest.approx(0.0, abs=1e-14)
    # The two independently written arrangements agree.
    assert tau_two_mark == pytest.approx(tau_single, abs=1e-15)


def test_risk_neutral_drag_collapses_to_one_mark(support):
    """Lambda = lambda_P_star J_P + lambda_F_star J_F reduces to the F term alone."""
    p, _, _, marks, _ = support
    F = next(m for m in marks.marks if m.label == "F")
    assert risk_neutral_drag(marks.marks) == pytest.approx(
        p.intensities.lambda_F_star * F.J, abs=1e-16
    )


def test_private_portfolio_reduces_to_R2(support, tol):
    """R.2: 1 + pi J_F = lambda/lambda_F_star on the single-mark branch."""
    from ak_partial_ramsey.exposure import solve_private_portfolio

    p, _, _, marks, _ = support
    F = next(m for m in marks.marks if m.label == "F")
    sol = solve_private_portfolio(marks.marks, tol)
    i = p.intensities
    assert 1.0 + sol.pi * F.J == pytest.approx(i.lambda_total / i.lambda_F_star, rel=1e-12)
    assert sol.pi == pytest.approx((i.lambda_total / i.lambda_F_star - 1.0) / F.J, rel=1e-12)


def test_independent_evaluator_confirms_the_reduction(support, tol):
    """The separate evaluator checks the whole collapse by its own formulas."""
    p, ak, _, marks, sol = support
    K, q, C = STATE["K"], STATE["q"], STATE["C"]
    rho = p.preferences.rho
    u = {m.label: fiscal_valuation_residual(m, sol.psi, C, rho) for m in marks.marks if m.is_active}
    U = total_valuation_residual(marks.marks, sol.psi, C, rho)
    D = investment_wedge(sol.psi, U, K, p.installation.varphi)
    q_dot = single_ak_q_dot(K, q, ak.q_F, p)
    report = check_single_ak_reduction(
        u_P=0.0,
        u_F=u["F"],
        U=U,
        D=D,
        consumption_growth_reported=consumption_growth(p, U, 1.0 / C),
        tau_reported=recover_source_tax(K, q, q_dot, risk_neutral_drag(marks.marks), p),
        rho=rho,
        lambda_total=p.intensities.lambda_total,
        lambda_F_star=p.intensities.lambda_F_star,
        r0_bar=p.rates.r0_bar,
        tolerance=tol.identity_atol,
    )
    report.raise_if_failed()


def test_full_state_evaluation_runs_on_the_support_restriction(support_fixture, tol):
    """The two-mark code path runs end to end with a genuinely inactive mark."""
    p = support_fixture.params
    ak = solve_ak_successor(p, support_fixture.ak_root_interval, tol)
    partial = solve_partial_successor(p, support_fixture.partial_capital_interval, tol)
    ev = evaluate_state(
        p,
        ak,
        partial,
        K=STATE["K"],
        e=STATE["e"],
        q=STATE["q"],
        C=STATE["C"],
        a=STATE["a"],
        q_dot=single_ak_q_dot(STATE["K"], STATE["q"], ak.q_F, p),
        tolerances=tol,
    )
    assert ev.U == pytest.approx(0.0, abs=1e-15)
    assert ev.D == pytest.approx(0.0, abs=1e-15)
    assert ev.tau_0 == pytest.approx(0.0, abs=1e-14)
