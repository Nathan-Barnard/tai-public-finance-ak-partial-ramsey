"""Partial-automation successor: stationary point, manifold, quadrature, BVP agreement.

CS011 N1 requires, for the absorbing fixed-``I_P`` successor: the stationary endpoint
with a verified one-stable/one-unstable linearisation; the stable manifold by
high-accuracy integration from a linearised offset with offset-size and
integration-direction checks; productive wealth by quadrature; shape-preserving
interpolants inside their certified domain only; and comparison of selected nodes with
an independent two-point collocation solve.
"""

from __future__ import annotations

import math

import pytest

from ak_partial_ramsey.errors import BranchFailure, DomainError
from ak_partial_ramsey.primitives import capital_growth, omega_Z, task_rental
from ak_partial_ramsey.successors.partial import (
    partial_field,
    partial_linearization,
    partial_stationary_point,
)

# --- stationary point ----------------------------------------------------------------


def test_stationary_point_solves_the_field(two_mark_fixture):
    p = two_mark_fixture.params
    pt = partial_stationary_point(p)
    assert pt.residual_K == pytest.approx(0.0, abs=1e-12)
    assert pt.residual_q == pytest.approx(0.0, abs=1e-12)


def test_stationary_price_is_the_zero_growth_price(two_mark_fixture):
    """K' = 0 forces g(q) = 0, whose only positive solution is q_delta = exp(varphi delta)."""
    p = two_mark_fixture.params
    pt = partial_stationary_point(p)
    expected = math.exp(p.installation.varphi * p.installation.delta)
    assert pt.q_delta == pytest.approx(expected, rel=1e-15)
    assert capital_growth(pt.q_delta, p.installation) == pytest.approx(0.0, abs=1e-15)


def test_stationary_capital_matches_the_closed_form(two_mark_fixture):
    """K_P* = [I_P Omega_Z(I_P)/U_P]^{1/(1-I_P)}, and R_P(K_P*) = U_P."""
    p = two_mark_fixture.params
    tech = p.partial_technology
    pt = partial_stationary_point(p)
    closed = (tech.I * omega_Z(tech) / pt.U_P) ** (1.0 / (1.0 - tech.I))
    assert pt.K_star == pytest.approx(closed, rel=1e-14)
    assert task_rental(pt.K_star, tech) == pytest.approx(pt.U_P, rel=1e-13)


def test_nonpositive_marginal_product_target_is_refused(two_mark_fixture):
    """U_P <= 0 means no positive rest point; the service refuses rather than returning one."""
    import dataclasses

    from ak_partial_ramsey.params import WorldRates

    p = two_mark_fixture.params
    # Choose r_P_bar so that U_P = r_P q_delta + iota_delta is negative.
    q_delta = p.installation.q_delta
    iota_delta = (q_delta - 1.0) / p.installation.varphi
    bad = -2.0 * iota_delta / q_delta
    broken = dataclasses.replace(p, rates=WorldRates(r0_bar=0.03, rF_bar=0.035, rP_bar=bad))
    with pytest.raises(BranchFailure) as exc:
        partial_stationary_point(broken)
    assert exc.value.detail["U_P"] <= 0.0


# --- linearisation --------------------------------------------------------------------


def test_exactly_one_stable_and_one_unstable_root(two_mark_fixture):
    p = two_mark_fixture.params
    lin = partial_linearization(p, partial_stationary_point(p))
    assert lin.nu_minus < 0.0 < lin.nu_plus
    assert lin.spectral_gap > 0.0


def test_characteristic_polynomial_matches_the_packet(two_mark_fixture):
    """nu^2 - r_P nu - (1 - I_P) U_P/(varphi q_delta), stated in the two-mark packet."""
    p = two_mark_fixture.params
    lin = partial_linearization(p, partial_stationary_point(p))
    assert lin.characteristic_residual == pytest.approx(0.0, abs=1e-15)


def test_eigenvalues_match_the_closed_form(two_mark_fixture):
    p = two_mark_fixture.params
    pt = partial_stationary_point(p)
    lin = partial_linearization(p, pt)
    rP = p.rates.rP_bar
    disc = rP**2 + 4.0 * (1.0 - p.partial_technology.I) * pt.U_P / (
        p.installation.varphi * pt.q_delta
    )
    assert lin.nu_plus == pytest.approx((rP + math.sqrt(disc)) / 2.0, rel=1e-13)
    assert lin.nu_minus == pytest.approx((rP - math.sqrt(disc)) / 2.0, rel=1e-13)


def test_manifold_slope_matches_the_packet(two_mark_fixture, partial_two_mark):
    """q_P'(K_P*) = varphi q_delta nu_- / K_P*, and it is negative."""
    p = two_mark_fixture.params
    pt = partial_stationary_point(p)
    lin = partial_linearization(p, pt)
    expected = p.installation.varphi * pt.q_delta * lin.nu_minus / pt.K_star
    assert lin.manifold_slope == pytest.approx(expected, rel=1e-13)
    assert lin.manifold_slope < 0.0
    # The constructed graph reproduces the analytic slope at the rest point.
    assert partial_two_mark.q_P_derivative(pt.K_star) == pytest.approx(expected, rel=1e-6)


# --- manifold construction ------------------------------------------------------------


def test_manifold_passes_through_the_rest_point(two_mark_fixture, partial_two_mark):
    pt = partial_two_mark.point
    assert partial_two_mark.q_P(pt.K_star) == pytest.approx(pt.q_delta, abs=1e-9)


def test_both_branches_reach_the_declared_boundary(two_mark_fixture, partial_two_mark):
    d = partial_two_mark.diagnostics
    for name in ("upper", "lower"):
        assert d["branches"][name]["integration"]["reached_declared_boundary"]
    lo, hi = partial_two_mark.certified_domain
    declared = two_mark_fixture.partial_capital_interval
    assert lo == pytest.approx(declared.K_lo, rel=1e-6)
    assert hi == pytest.approx(declared.K_hi, rel=1e-6)


def test_integration_direction_check_shows_contraction(partial_two_mark):
    """Forward time on the constructed branch contracts, at the rate linear theory predicts."""
    d = partial_two_mark.diagnostics
    for name in ("upper", "lower"):
        dc = d["branches"][name]["direction_check"]
        assert dc["contracts"]
        # Substantial contraction, not merely a marginal decrease.
        assert dc["contraction_ratio"] < 0.5
        # Same order as the pure stable-mode prediction. The band is deliberately wide:
        # the check starts from the far end of the branch, well outside the linear
        # regime, so exact agreement is not expected and would not be meaningful. What
        # the comparison rules out is contamination by the unstable mode, which would
        # show up as a ratio orders of magnitude larger, or a ratio above one.
        lo_band = 0.25 * dc["linear_predicted_ratio"]
        hi_band = 4.0 * dc["linear_predicted_ratio"]
        assert lo_band <= dc["contraction_ratio"] <= hi_band


def test_offset_size_check(partial_two_mark, tol):
    """Halving the linearisation offset does not move the manifold."""
    assert partial_two_mark.diagnostics["max_offset_size_spread"] < tol.cross_route_atol


def test_manifold_invariance(partial_two_mark, two_mark_fixture, tol):
    """q_P'(K) g(q_P) K equals the price drift: the graph is invariant under the flow."""
    p = two_mark_fixture.params
    lo, hi = partial_two_mark.certified_domain
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        K = lo + frac * (hi - lo)
        q = partial_two_mark.q_P(K)
        lhs = partial_two_mark.q_P_derivative(K) * capital_growth(q, p.installation) * K
        rhs = partial_field(K, q, p)[1]
        assert lhs == pytest.approx(rhs, abs=1e-6)


# --- productive wealth ----------------------------------------------------------------


def test_productive_wealth_two_routes_agree(partial_two_mark, tol):
    """Quadrature of q_P against the algebraic productive-wealth equation."""
    lo, hi = partial_two_mark.certified_domain
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        K = lo + frac * (hi - lo)
        assert partial_two_mark.H_P(K) == pytest.approx(partial_two_mark.H_P_algebraic(K), abs=1e-5)


def test_wealth_derivative_identity(partial_two_mark):
    """H_P'(K) = q_P(K), the identity that makes the two wealth routes consistent."""
    lo, hi = partial_two_mark.certified_domain
    for frac in (0.2, 0.5, 0.8):
        K = lo + frac * (hi - lo)
        assert partial_two_mark.H_P_derivative(K) == pytest.approx(
            partial_two_mark.q_P(K), abs=1e-5
        )


def test_anchor_normalisation(partial_two_mark, two_mark_fixture):
    """At the rest point g = 0, so H_P collapses to [Y_P - iota_delta K]/r_P."""
    from ak_partial_ramsey.primitives import task_output

    p = two_mark_fixture.params
    pt = partial_two_mark.point
    expected = (
        task_output(pt.K_star, p.partial_technology) - pt.iota_delta * pt.K_star
    ) / p.rates.rP_bar
    assert partial_two_mark.H_anchor == pytest.approx(expected, rel=1e-15)
    assert partial_two_mark.H_P(pt.K_star) == pytest.approx(expected, abs=1e-9)


def test_present_value_route_agrees(partial_two_mark, two_mark_fixture, tol):
    """A third route: the discounted product-flow integral along the manifold."""
    from ak_partial_ramsey.evaluator import check_partial_present_value

    p = two_mark_fixture.params
    lo, hi = partial_two_mark.certified_domain
    for frac in (0.25, 0.5, 0.75):
        K = lo + frac * (hi - lo)
        res = check_partial_present_value(
            K_0=K,
            price_graph=partial_two_mark.q_P,
            H_reported=partial_two_mark.H_P(K),
            rP_bar=p.rates.rP_bar,
            Z=p.partial_technology.Z,
            I_P=p.partial_technology.I,
            varphi=p.installation.varphi,
            delta=p.installation.delta,
            K_star=partial_two_mark.point.K_star,
            iota_delta=partial_two_mark.point.iota_delta,
            tolerance=tol.cross_route_atol,
        )
        assert res.passed, res.as_dict()


# --- IVP versus BVP --------------------------------------------------------------------


def test_ivp_and_bvp_routes_agree(partial_two_mark, tol):
    """Mandatory: the integrated manifold and an independent collocation solve agree.

    The collocation route is seeded from the *linear* stable solution, not from the
    integrated manifold, so it shares no numerical state with the IVP construction.
    """
    d = partial_two_mark.diagnostics
    checks = d["collocation_checks"]
    assert len(checks) >= 3
    for c in checks:
        assert c["success"], c
        assert abs(c["ivp_bvp_difference"]) < tol.cross_route_atol
    assert d["max_ivp_bvp_difference"] < tol.cross_route_atol


# --- certified domain ------------------------------------------------------------------


def test_interpolants_refuse_to_extrapolate(partial_two_mark):
    """Outside the certified domain the interpolants carry no evidence, so they raise."""
    lo, hi = partial_two_mark.certified_domain
    for K in (lo * 0.5, lo - 1.0, hi + 1.0, hi * 2.0):
        with pytest.raises(DomainError):
            partial_two_mark.q_P(K)
        with pytest.raises(DomainError):
            partial_two_mark.H_P(K)


def test_declared_interval_must_contain_the_rest_point(two_mark_fixture, tol):
    from ak_partial_ramsey.params import PartialCapitalInterval
    from ak_partial_ramsey.successors.partial import solve_partial_successor

    p = two_mark_fixture.params
    K_star = partial_stationary_point(p).K_star
    with pytest.raises(DomainError) as exc:
        solve_partial_successor(
            p, PartialCapitalInterval(K_lo=K_star * 2.0, K_hi=K_star * 3.0), tol
        )
    assert "rest point" in exc.value.message


# --- successor values and envelopes ----------------------------------------------------


def test_partial_value_and_envelope_identities(partial_two_mark, two_mark_fixture):
    p = two_mark_fixture.params
    rho = p.preferences.rho
    lo, hi = partial_two_mark.certified_domain
    K = 0.5 * (lo + hi)
    e = 100.0
    X = partial_two_mark.X_P(K, e)
    closed = math.log(rho * X) / rho + (p.rates.rP_bar - rho) / rho**2
    assert partial_two_mark.V_P(K, e) == pytest.approx(closed, rel=1e-15)
    assert partial_two_mark.V_P_e(K, e) == pytest.approx(1.0 / (rho * X), rel=1e-15)
    assert partial_two_mark.V_P_K(K, e) == pytest.approx(
        partial_two_mark.q_P(K) * partial_two_mark.V_P_e(K, e), rel=1e-15
    )
    assert partial_two_mark.C_P(K, e) == pytest.approx(rho * X, rel=1e-15)


def test_nonpositive_successor_wealth_is_refused(partial_two_mark):
    lo, hi = partial_two_mark.certified_domain
    K = 0.5 * (lo + hi)
    with pytest.raises(DomainError):
        partial_two_mark.V_P(K, -partial_two_mark.H_P(K) - 1.0)


def test_certified_domain_message_names_the_domain(partial_two_mark):
    """Outside the certified domain the refusal says so explicitly."""
    _lo, hi = partial_two_mark.certified_domain
    with pytest.raises(DomainError) as exc:
        partial_two_mark.q_P(hi * 1.01)
    assert "certified" in exc.value.message


def test_frozen_manufactured_interval_still_brackets_the_rest_point(two_mark_fixture):
    """The frozen fixture window must still contain the rest point it was derived from.

    fixtures.py freezes the manufactured capital window as literals so that it does not
    recompute itself from the code under test. This re-derives K_P* from the parameters
    and checks the freeze has not gone stale.
    """
    p = two_mark_fixture.params
    K_star = partial_stationary_point(p).K_star
    iv = two_mark_fixture.partial_capital_interval
    assert iv.contains(K_star)
    assert iv.K_lo == pytest.approx(K_star / 1.5, rel=1e-12)
    assert iv.K_hi == pytest.approx(K_star * 1.5, rel=1e-12)
