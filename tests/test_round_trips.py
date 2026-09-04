"""Normalized-to-level event-map and budget round trips.

CS011 benchmark contract, "Level round trip": normalized and level budgets, event maps,
and positions agree at held-out times. Failure means a coordinate or timing defect, which
is a structural failure and not a tolerance question.
"""

from __future__ import annotations

import pytest

from ak_partial_ramsey.evaluator import check_level_budget_round_trip
from ak_partial_ramsey.events import (
    event_level,
    event_normalized,
    jump_payoff,
    jump_payoff_dK,
    jump_payoff_dq,
    level_from_normalized,
    national_event_jump,
    national_net_foreign_assets,
    normalized_from_level,
    normalized_successor_wealth_coordinate,
    owner_event,
    round_trip_diagnostics,
    successor_price_from_jump,
)
from ak_partial_ramsey.recovery import (
    recover_foreign_residual,
    recover_owner,
    recover_positions,
    recover_transfer,
)

CASES = [
    # (e, psi, q, K, q_successor, a, pi)   -- manufactured, no economic interpretation
    (0.2, 0.3, 1.0600839471281598, 1.0, 1.0600839471281598, 0.5, 0.0),
    (-6.0, -4.94654, 1.10, 10.0, 1.21, 20.0, -0.886306),
    (5.0, -1693.0, 1.15, 300.0, 1.162, 50.0, 0.4),
    (0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.25),
    (-400.0, 900.0, 1.30, 320.0, 1.05, 2.0, -0.9),
]


@pytest.mark.parametrize(("e", "psi", "q", "K", "q_j", "a", "pi"), CASES)
def test_coordinate_and_event_round_trips(e, psi, q, K, q_j, a, pi):
    J = jump_payoff(q_j, q)
    rt = round_trip_diagnostics(e=e, psi=psi, q=q, K=K, J=J, q_successor=q_j, a=a, pi=pi)
    assert rt.max_abs < 1e-9, rt.as_dict()


@pytest.mark.parametrize(("e", "psi", "q", "K", "q_j", "a", "pi"), CASES)
def test_level_event_map_reproduces_the_normalized_one(e, psi, q, K, q_j, a, pi):
    """F_j^+ = F + Theta J_j, converted back, equals e + psi J_j."""
    J = jump_payoff(q_j, q)
    lvl = level_from_normalized(e, psi, q, K)
    F_plus = event_level(lvl["F"], lvl["Theta"], J)
    via_level = normalized_successor_wealth_coordinate(F_plus, q_j, K)
    via_normalized = event_normalized(e, psi, J)
    assert via_level == pytest.approx(via_normalized, abs=1e-9)


@pytest.mark.parametrize(("e", "psi", "q", "K", "q_j", "a", "pi"), CASES)
def test_position_recovery_is_consistent(e, psi, q, K, q_j, a, pi):
    """F = e + qK, Theta = psi + qK, B = psi - e, and back again."""
    pos = recover_positions(e, psi, q, K)
    back = normalized_from_level(pos["F"], pos["Theta"], q, K)
    assert back["e"] == pytest.approx(e, abs=1e-9)
    assert back["psi"] == pytest.approx(psi, abs=1e-9)
    assert back["B"] == pytest.approx(pos["B"], abs=1e-12)
    assert pos["B"] == pytest.approx(psi - e, abs=1e-12)


@pytest.mark.parametrize(("e", "psi", "q", "K", "q_j", "a", "pi"), CASES)
def test_both_asset_markets_clear_exactly(e, psi, q, K, q_j, a, pi):
    """Foreign residual positions clear equity and safe markets with zero residual."""
    pos = recover_positions(e, psi, q, K)
    owner = recover_owner(a, pi, 0.01, _params())
    foreign = recover_foreign_residual(q, K, pos["Theta"], owner.vartheta, pos["B"], owner.d_O)
    assert foreign["equity_clearing_residual"] == pytest.approx(0.0, abs=1e-9)
    assert foreign["safe_clearing_residual"] == pytest.approx(0.0, abs=1e-9)
    # The foreign net claim is exactly minus domestic net foreign assets.
    n = national_net_foreign_assets(owner.a, e)
    assert foreign["foreign_net_claim"] == pytest.approx(-n, abs=1e-8)


@pytest.mark.parametrize(("e", "psi", "q", "K", "q_j", "a", "pi"), CASES)
def test_national_event_jump_matches_its_parts(e, psi, q, K, q_j, a, pi):
    """n_j^+ - n = (psi + pi a) J_j, consistent with the public and owner event maps."""
    J = jump_payoff(q_j, q)
    n = national_net_foreign_assets(a, e)
    n_plus_parts = national_net_foreign_assets(owner_event(a, pi, J), event_normalized(e, psi, J))
    assert n_plus_parts - n == pytest.approx(national_event_jump(psi, pi, a, J), abs=1e-8)


def _params():
    from ak_partial_ramsey.fixtures import get_fixture

    return get_fixture("manufactured-two-mark").params


def test_jump_payoff_derivatives():
    """J_{j,q} = -(1+J_j)/q and J_{j,K} = q_j'(K)/q, checked by finite differences."""
    q, q_j = 1.15, 1.30
    J = jump_payoff(q_j, q)
    h = 1e-7
    fd_q = (jump_payoff(q_j, q + h) - jump_payoff(q_j, q - h)) / (2 * h)
    assert jump_payoff_dq(J, q) == pytest.approx(fd_q, rel=1e-6)

    slope = 0.03  # a manufactured q_j'(K)
    fd_K = (jump_payoff(q_j + slope * h, q) - jump_payoff(q_j - slope * h, q)) / (2 * h)
    assert jump_payoff_dK(slope, q) == pytest.approx(fd_K, rel=1e-6)


def test_ak_payoff_slope_is_zero():
    """The selected AK price is constant in K, so J_{F,K} = 0 exactly."""
    assert jump_payoff_dK(0.0, 1.15) == 0.0


def test_successor_price_inverts_the_jump():
    q, q_j = 1.15, 1.30
    assert successor_price_from_jump(jump_payoff(q_j, q), q) == pytest.approx(q_j, rel=1e-15)


def test_independent_evaluator_confirms_the_budget_round_trip(two_mark_fixture, tol):
    """The external-wealth law rebuilt from an independent production block."""
    p = two_mark_fixture.params
    from ak_partial_ramsey.canonical import state_e_dot

    K, e, q, C, psi, Lambda = 320.0, 5.0, 1.15, 10.0, -20.0, 0.0004
    reported = state_e_dot(K, e, q, C, psi, Lambda, p)
    report = check_level_budget_round_trip(
        e=e,
        psi=psi,
        q=q,
        K=K,
        C_W=C,
        Lambda=Lambda,
        Z=p.pre_arrival_technology.Z,
        I_0=p.pre_arrival_technology.I,
        varphi=p.installation.varphi,
        r0_bar=p.rates.r0_bar,
        e_dot_reported=reported,
        tolerance=tol.identity_atol,
    )
    report.raise_if_failed()


def test_transfer_recovery_identity(two_mark_fixture):
    """C^W = W_0(K) + T is an identity, not an approximation."""
    from ak_partial_ramsey.primitives import task_wage

    p = two_mark_fixture.params
    K, C = 320.0, 15.0
    T = recover_transfer(C, K, p)
    assert task_wage(K, p.pre_arrival_technology) + T == pytest.approx(C, rel=1e-15)
