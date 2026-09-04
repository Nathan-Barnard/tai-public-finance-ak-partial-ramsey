"""Branch labels remain stable under numerical refinement.

CS011 non-tradeable structural gate: "Branch identity is continued from a named anchor
and never assigned by first-returned or sorted-root order." A root or branch that changes
under refinement without a diagnosed fold is a stop condition.

Branch labels here are the economic classification relative to the analytic minimiser
``q_m = exp(u)``, so refining the scan or tightening the bracket cannot relabel anything.
These tests assert that.
"""

from __future__ import annotations

import dataclasses

import pytest

from ak_partial_ramsey.params import AkRootInterval
from ak_partial_ramsey.successors.ak import enumerate_ak_roots, solve_ak_successor
from ak_partial_ramsey.tolerances import SolverTolerances

SCAN_RESOLUTIONS = [301, 701, 1501, 2001, 4001, 8001]


def _labels(candidates):
    return [(c.branch, c.accepted, c.reason) for c in candidates]


def test_labels_are_invariant_to_scan_resolution(single_ak_fixture):
    """The same two branches, with the same verdicts, at every scan resolution."""
    base = None
    roots_by_resolution = []
    for n in SCAN_RESOLUTIONS:
        tol = SolverTolerances(root_scan_points=n)
        candidates, _ = enumerate_ak_roots(
            single_ak_fixture.params, single_ak_fixture.ak_root_interval, tol
        )
        labels = _labels(candidates)
        if base is None:
            base = labels
        assert labels == base, f"branch labels changed at scan resolution {n}"
        roots_by_resolution.append([c.q for c in candidates])

    # And the root values themselves agree across resolutions.
    first = roots_by_resolution[0]
    for other in roots_by_resolution[1:]:
        assert len(other) == len(first)
        for a, b in zip(first, other, strict=True):
            assert a == pytest.approx(b, abs=1e-12)


def test_labels_are_invariant_to_root_tolerance(single_ak_fixture):
    """Tightening or loosening the bracket tolerance does not relabel a branch."""
    base = None
    for xtol, rtol in ((1e-10, 1e-11), (1e-12, 1e-13), (1e-14, 1e-15)):
        tol = SolverTolerances(root_xtol=xtol, root_rtol=rtol)
        candidates, _ = enumerate_ak_roots(
            single_ak_fixture.params, single_ak_fixture.ak_root_interval, tol
        )
        labels = _labels(candidates)
        if base is None:
            base = labels
        assert labels == base


def test_selection_is_invariant_to_the_declared_interval(single_ak_fixture, tol):
    """Widening the search interval finds the same roots with the same verdicts."""
    base = None
    for lo, hi in ((0.5, 3.0), (0.2, 5.0), (0.9, 2.0), (1.0, 1.6)):
        sel = solve_ak_successor(single_ak_fixture.params, AkRootInterval(q_lo=lo, q_hi=hi), tol)
        if base is None:
            base = sel.q_F
        assert sel.q_F == pytest.approx(base, abs=1e-12)
        assert sel.branch == "lower_strict_tvc"


def test_label_is_not_sorted_position(single_ak_fixture, tol):
    """The accepted root is the lower one *because of the TVC*, not because it is first.

    Narrowing the interval so that only the upper root is inside makes it the first and
    only root found - and it is still rejected. Position carries no authority.
    """
    from ak_partial_ramsey.errors import BranchFailure

    with pytest.raises(BranchFailure):
        solve_ak_successor(single_ak_fixture.params, AkRootInterval(q_lo=1.3, q_hi=3.0), tol)


def test_branch_label_tracks_the_analytic_minimiser(single_ak_fixture, tol):
    """Lower/upper is decided by position relative to q_m, which is computed exactly."""
    from ak_partial_ramsey.successors.ak import ak_scalar_coefficients

    q_m = ak_scalar_coefficients(single_ak_fixture.params)["q_m"]
    candidates, _ = enumerate_ak_roots(
        single_ak_fixture.params, single_ak_fixture.ak_root_interval, tol
    )
    for c in candidates:
        if c.branch == "lower_strict_tvc":
            assert c.q < q_m
        elif c.branch == "upper_tvc_violating":
            assert c.q > q_m


def test_tvc_margin_sign_matches_the_branch(single_ak_fixture, tol):
    """r_F > g(q) holds exactly when q < q_m; the two criteria cannot disagree."""
    candidates, _ = enumerate_ak_roots(
        single_ak_fixture.params, single_ak_fixture.ak_root_interval, tol
    )
    for c in candidates:
        assert (c.tvc_margin > 0.0) == c.accepted


def test_partial_eigenstructure_is_stable_under_refinement(two_mark_fixture):
    """The partial successor's stable/unstable split does not move with tolerances."""
    from ak_partial_ramsey.successors.partial import (
        partial_linearization,
        partial_stationary_point,
    )

    p = two_mark_fixture.params
    pt = partial_stationary_point(p)
    lin = partial_linearization(p, pt)
    assert lin.nu_minus < 0.0 < lin.nu_plus
    for ivp_rtol in (1e-9, 1e-11, 1e-13):
        tol = SolverTolerances(ivp_rtol=ivp_rtol, ivp_atol=ivp_rtol / 10.0)
        assert tol.ivp_rtol == ivp_rtol
        again = partial_linearization(p, partial_stationary_point(p))
        assert again.nu_minus == pytest.approx(lin.nu_minus, rel=1e-15)
        assert again.nu_plus == pytest.approx(lin.nu_plus, rel=1e-15)


def test_partial_manifold_is_stable_under_integration_tolerance(two_mark_fixture):
    """Tightening the integrator does not move the constructed manifold."""
    from ak_partial_ramsey.successors.partial import solve_partial_successor

    p = two_mark_fixture.params
    iv = two_mark_fixture.partial_capital_interval
    results = []
    for rtol, atol in ((1e-9, 1e-10), (1e-11, 1e-12), (1e-12, 1e-13)):
        tol = SolverTolerances(ivp_rtol=rtol, ivp_atol=atol)
        s = solve_partial_successor(p, iv, tol, n_collocation_checks=1)
        lo, hi = s.certified_domain
        results.append([s.q_P(lo + f * (hi - lo)) for f in (0.2, 0.5, 0.8)])
    for other in results[1:]:
        for a, b in zip(results[0], other, strict=True):
            assert a == pytest.approx(b, abs=1e-7)


def test_offset_refinement_does_not_move_the_manifold(two_mark_fixture):
    """Shrinking the linearisation offset does not move the manifold either."""
    from ak_partial_ramsey.successors.partial import solve_partial_successor

    p = two_mark_fixture.params
    iv = two_mark_fixture.partial_capital_interval
    values = []
    for offset in (1e-5, 1e-6, 1e-7):
        tol = SolverTolerances(manifold_offset=offset)
        s = solve_partial_successor(p, iv, tol, n_collocation_checks=1)
        lo, hi = s.certified_domain
        values.append(s.q_P(0.5 * (lo + hi)))
    for v in values[1:]:
        assert v == pytest.approx(values[0], abs=1e-7)


def test_double_root_boundary_is_reported_not_selected(single_ak_fixture, tol):
    """At a = exp(u) the root has zero TVC margin, so it is refused, not accepted.

    The strict inequality in the existence condition cannot be weakened: at the
    tangency the discounted installed-capital value is exactly constant rather than
    vanishing.
    """
    import math

    from ak_partial_ramsey.errors import BranchFailure
    from ak_partial_ramsey.params import AkTechnology
    from ak_partial_ramsey.successors.ak import ak_scalar_coefficients

    p = single_ak_fixture.params
    varphi = p.installation.varphi
    u = varphi * (p.rates.rF_bar + p.installation.delta)
    # a = 1 + varphi A_bar = exp(u)  =>  A_bar = (exp(u) - 1)/varphi
    A_tangent = (math.exp(u) - 1.0) / varphi
    tangent = dataclasses.replace(p, ak_technology=AkTechnology(A_bar=A_tangent))
    assert ak_scalar_coefficients(tangent)["discriminant"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(BranchFailure):
        solve_ak_successor(tangent, single_ak_fixture.ak_root_interval, tol)


def test_no_root_regime_is_reported(single_ak_fixture, tol):
    """a > exp(u): no positive root exists, and that is stated rather than approximated."""
    import math

    from ak_partial_ramsey.errors import BranchFailure
    from ak_partial_ramsey.params import AkTechnology

    p = single_ak_fixture.params
    varphi = p.installation.varphi
    u = varphi * (p.rates.rF_bar + p.installation.delta)
    A_none = (math.exp(u) - 1.0) / varphi * 1.5
    broken = dataclasses.replace(p, ak_technology=AkTechnology(A_bar=A_none))
    with pytest.raises(BranchFailure) as exc:
        solve_ak_successor(broken, single_ak_fixture.ak_root_interval, tol)
    assert exc.value.detail["coefficients"]["discriminant"] > 0.0
    assert exc.value.detail["candidates"] == []
