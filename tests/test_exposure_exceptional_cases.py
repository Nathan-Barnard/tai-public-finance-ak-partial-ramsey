"""Exceptional cases of the public and private portfolio equations.

Covers the CS011 benchmark rows "Zero payoff vector", "One zero payoff component", and
the zero-intensity handling required by the single-AK support branch. The governing rule
throughout: use the unmultiplied first-order condition, and never divide by a mark payoff
that may vanish.
"""

from __future__ import annotations

import math

import pytest

from ak_partial_ramsey.errors import DomainError, RankFailure
from ak_partial_ramsey.exposure import (
    Mark,
    fiscal_valuation_residual,
    private_portfolio_residual,
    public_exposure_residual,
    public_exposure_residual_derivative,
    solve_private_portfolio,
    solve_public_exposure,
    successor_wealth_interval,
)

RHO = 0.04
C = 0.4


def marks(
    J_P: float,
    J_F: float,
    *,
    lam_P=0.012,
    lam_F=0.008,
    star_P=0.010,
    star_F=0.015,
    h_P=10.0,
    h_F=8.0,
):
    """Manufactured marks. No economic interpretation."""
    return (
        Mark("P", lam_P, star_P, J_P, h_P),
        Mark("F", lam_F, star_F, J_F, h_F),
    )


# --- zero payoff vector -------------------------------------------------------------


def test_zero_payoff_vector_is_refused(tol):
    """Rank failure: with J = (0, 0) the condition reads 0 = 0 and identifies nothing."""
    with pytest.raises(RankFailure) as exc:
        solve_public_exposure(marks(0.0, 0.0), C, RHO, tol)
    assert exc.value.detail["rank"] == 0
    assert exc.value.detail["payoffs"] == [0.0, 0.0]


def test_zero_payoff_vector_refused_for_the_private_portfolio(tol):
    with pytest.raises(RankFailure):
        solve_private_portfolio(marks(0.0, 0.0), tol)


def test_zero_payoff_vector_makes_the_residual_identically_zero():
    """The refusal is not conservatism: the condition really does carry no information."""
    m = marks(0.0, 0.0)
    for psi in (-100.0, -1.0, 0.0, 1.0, 100.0):
        assert public_exposure_residual(psi, m, C, RHO) == 0.0
        assert public_exposure_residual_derivative(psi, m, C, RHO) == 0.0


# --- one zero payoff component ------------------------------------------------------


def test_one_zero_component_is_solved_and_the_mark_stays_unspanned(tol):
    """J_P = 0, J_F != 0: solve through the unmultiplied FOC, retain the unspanned mark.

    The zero-payoff mark must survive the solve with a generally nonzero ``u_P``. That
    nonzero residual *is* the statement that the mark is unspanned; forcing it to zero,
    or dropping the mark, would be the "hidden division by a jump" the specification
    warns against.
    """
    m = marks(0.0, 0.5)
    sol = solve_public_exposure(m, C, RHO, tol)

    assert sol.unspanned_marks == ("P",)
    assert sol.orthogonality_residual == pytest.approx(0.0, abs=1e-12)
    # The spanned mark's residual is driven to zero...
    assert sol.fiscal_valuation_residual["F"] == pytest.approx(0.0, abs=1e-14)
    # ...while the unspanned mark's is not, and is retained.
    assert sol.fiscal_valuation_residual["P"] != 0.0
    # The zero-payoff mark's wealth does not move with the exposure.
    assert sol.successor_wealth["P"] == pytest.approx(m[0].h, abs=1e-15)


def test_one_zero_component_matches_the_hand_solution(tol):
    """With one spanned mark the root is where u_F = 0, i.e. X_F = lambda_F C/(rho lambda_F*)."""
    m = marks(0.0, 0.5)
    sol = solve_public_exposure(m, C, RHO, tol)
    F = m[1]
    X_expected = F.lambda_physical * C / (RHO * F.lambda_star)
    assert sol.successor_wealth["F"] == pytest.approx(X_expected, rel=1e-12)
    assert sol.psi == pytest.approx((X_expected - F.h) / F.J, rel=1e-12)


def test_zero_payoff_mark_does_not_bound_the_exposure_interval():
    """A J_j = 0 mark has constant successor wealth and so constrains no exposure."""
    lo, hi = successor_wealth_interval(marks(0.0, 0.5))
    # Only the F mark bounds the interval, from below since J_F > 0.
    assert lo == pytest.approx(-8.0 / 0.5, rel=1e-15)
    assert math.isinf(hi)


def test_zero_payoff_mark_with_nonpositive_wealth_is_refused():
    """Its log value is undefined at *every* exposure, so this is a domain failure."""
    with pytest.raises(DomainError) as exc:
        successor_wealth_interval(marks(0.0, 0.5, h_P=-1.0))
    assert exc.value.detail["label"] == "P"


def test_the_other_orientation_also_works(tol):
    """J_P != 0, J_F = 0: the roles simply swap."""
    m = marks(0.3, 0.0)
    sol = solve_public_exposure(m, C, RHO, tol)
    assert sol.unspanned_marks == ("F",)
    assert sol.fiscal_valuation_residual["P"] == pytest.approx(0.0, abs=1e-14)
    assert sol.fiscal_valuation_residual["F"] != 0.0


# --- zero intensities, no division ---------------------------------------------------


def test_inactive_mark_contributes_nothing_and_divides_by_nothing():
    """p_P = lambda_P_star = 0: u_P is zero without forming X_P or dividing by either."""
    inactive = Mark("P", 0.0, 0.0, 0.25, 10.0)
    # Even at an exposure that would make its wealth negative, no division occurs.
    for psi in (-1e6, 0.0, 1e6):
        assert fiscal_valuation_residual(inactive, psi, C, RHO) == 0.0


def test_inactive_mark_does_not_bound_the_interval():
    m = (Mark("P", 0.0, 0.0, 0.5, 8.0), Mark("F", 0.008, 0.015, 0.5, 8.0))
    lo, hi = successor_wealth_interval(m)
    assert lo == pytest.approx(-16.0, rel=1e-15)  # from F alone
    assert math.isinf(hi)


def test_all_marks_inactive_is_a_configuration_error(tol):
    from ak_partial_ramsey.errors import ConfigurationError

    m = (Mark("P", 0.0, 0.0, 0.1, 10.0), Mark("F", 0.0, 0.0, 0.5, 8.0))
    with pytest.raises(ConfigurationError):
        solve_public_exposure(m, C, RHO, tol)


def test_zero_risk_neutral_intensity_does_not_divide_by_zero(tol):
    """lambda_j_star = 0 on an active mark: no root exists, and that is said explicitly."""
    from ak_partial_ramsey.errors import BranchFailure

    m = (Mark("P", 0.012, 0.0, 0.1, 10.0), Mark("F", 0.008, 0.0, 0.5, 8.0))
    # Every u_j is then strictly positive, so sum u_j J_j cannot vanish for J > 0.
    with pytest.raises(BranchFailure) as exc:
        solve_public_exposure(m, C, RHO, tol)
    assert "no interior root" in exc.value.message


# --- monotonicity and uniqueness ----------------------------------------------------


@pytest.mark.parametrize(
    ("J_P", "J_F"),
    [(0.1, 0.5), (-0.1, -0.5), (0.1, -0.5), (-0.1, 0.5), (0.0, 0.5), (0.3, 0.0)],
)
def test_residual_is_strictly_decreasing_and_the_root_is_unique(J_P, J_F, tol):
    """H_{psi psi} < 0 on the positive-wealth interval, so the interior root is unique."""
    m = marks(J_P, J_F)
    lo, hi = successor_wealth_interval(m)
    sol = solve_public_exposure(m, C, RHO, tol)
    assert lo < sol.psi < hi
    assert sol.curvature < 0.0
    # Sample the interval and confirm monotone decrease across the root.
    span_lo = sol.psi - 1.0 if math.isinf(lo) else lo + 0.25 * (sol.psi - lo)
    span_hi = sol.psi + 1.0 if math.isinf(hi) else hi - 0.25 * (hi - sol.psi)
    assert public_exposure_residual(span_lo, m, C, RHO) > 0.0
    assert public_exposure_residual(span_hi, m, C, RHO) < 0.0


def test_private_portfolio_root_is_unique_and_solvent(tol):
    m = marks(0.1, 0.5)
    sol = solve_private_portfolio(m, tol)
    assert all(v > 0.0 for v in sol.event_solvency.values())
    assert sol.residual == pytest.approx(0.0, abs=1e-12)
    # The condition is strictly decreasing in pi on the solvency interval.
    assert private_portfolio_residual(sol.pi - 0.1, m) > 0.0
    assert private_portfolio_residual(sol.pi + 0.1, m) < 0.0


def test_owner_event_solvency_is_refused_not_clipped():
    """1 + pi J_j <= 0 raises; the position is never pulled back into the domain."""
    m = marks(0.1, 0.5)
    with pytest.raises(DomainError) as exc:
        private_portfolio_residual(-3.0, m)  # 1 + (-3)(0.5) = -0.5
    assert exc.value.detail["label"] == "F"


def test_theory_packet_algebraic_root_fixture(tol):
    """The two-mark packet's section 10 trial fixture, reproduced.

    Supplied verbatim: rho = 0.04, (lambda_P, lambda_F) = (0.012, 0.008),
    (lambda_P_star, lambda_F_star) = (0.01, 0.015), (J_P, J_F) = (0.1, 0.5),
    (h_P, h_F) = (10, 8), C = 0.4, e = -6. The packet states the unique admissible root
    and the resulting quantities.
    """
    m = marks(0.1, 0.5)
    sol = solve_public_exposure(m, C, RHO, tol)
    e = -6.0
    assert sol.psi == pytest.approx(-4.94654, abs=5e-6)
    assert sol.psi - e == pytest.approx(1.05346, abs=5e-6)  # B = psi - e
    assert sol.successor_wealth["P"] == pytest.approx(9.50535, abs=5e-6)
    assert sol.successor_wealth["F"] == pytest.approx(5.52673, abs=5e-6)
    assert sol.fiscal_valuation_residual["P"] == pytest.approx(0.00656119, abs=5e-9)
    assert sol.fiscal_valuation_residual["F"] == pytest.approx(-0.00131224, abs=5e-9)
    assert abs(sol.orthogonality_residual) < 1e-17

    priv = solve_private_portfolio(m, tol)
    assert priv.pi == pytest.approx(-0.886306, abs=5e-7)
    assert priv.event_solvency["P"] == pytest.approx(0.91137, abs=5e-6)
    assert priv.event_solvency["F"] == pytest.approx(0.55685, abs=5e-6)
