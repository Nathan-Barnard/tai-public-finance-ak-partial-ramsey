"""G0: reproduce the theory packet's two-root AK fixture, selection and rejection alike.

CS011: "G0 passes only if the implementation reproduces the selected and rejected roots,
values, derivatives, event maps, recovery equations, and separate TVC margins to declared
numerical precision."

The packet presents this fixture as a falsification of "any scalar root is admissible".
Reproducing the *rejection* is therefore as much a part of G0 as reproducing the
selection, and both are asserted here.
"""

from __future__ import annotations

import math

import pytest

from ak_partial_ramsey.evaluator import check_ak_successor
from ak_partial_ramsey.successors.ak import (
    ak_lambert_roots,
    ak_price_residual_level,
    ak_scalar_coefficients,
)

# The packet quotes nine decimal places; compare to a little inside that.
PACKET_TOL = 5e-10


def test_scalar_coefficients(single_ak_fixture, ak_single):
    expected = single_ak_fixture.expected
    c = ak_scalar_coefficients(single_ak_fixture.params)
    assert c["a"] == pytest.approx(expected["a"], abs=1e-15)
    assert c["q_m"] == pytest.approx(expected["q_m"], abs=PACKET_TOL)
    # a < exp(u) is exactly the two-root condition (EU.3).
    assert c["discriminant"] < 0.0


def test_both_positive_roots_are_enumerated(single_ak_fixture, ak_single):
    assert len(ak_single.candidates) == single_ak_fixture.expected["n_positive_roots"]
    assert ak_single.diagnostics["n_roots_found"] == 2
    assert ak_single.diagnostics["n_roots_expected_from_discriminant"] == 2


def test_accepted_root_matches_packet(single_ak_fixture, ak_single):
    exp = single_ak_fixture.expected["accepted"]
    K, e = single_ak_fixture.state["K"], single_ak_fixture.state["e"]
    assert ak_single.q_F == pytest.approx(exp["q_F"], abs=PACKET_TOL)
    assert ak_single.iota_F == pytest.approx(exp["iota_F"], abs=PACKET_TOL)
    assert ak_single.g_F == pytest.approx(exp["g_F"], abs=PACKET_TOL)
    assert ak_single.installation_margin == pytest.approx(
        exp["installation_margin"], abs=PACKET_TOL
    )
    assert ak_single.tvc_margin == pytest.approx(exp["tvc_margin"], abs=PACKET_TOL)
    assert ak_single.full_bgp_residual == pytest.approx(exp["full_bgp_residual"], abs=PACKET_TOL)
    assert ak_single.X_F(K, e) == pytest.approx(exp["X_F"], abs=PACKET_TOL)
    assert ak_single.C_F(K, e) == pytest.approx(exp["C_F_W"], abs=PACKET_TOL)
    assert ak_single.recovered_tau_F == pytest.approx(exp["recovered_tau_F"], abs=1e-12)
    assert ak_single.branch == "lower_strict_tvc"


def test_rejected_root_matches_packet_and_is_retained(single_ak_fixture, ak_single):
    """The upper root must be found, kept, and rejected for the stated reason."""
    exp = single_ak_fixture.expected["rejected"]
    params = single_ak_fixture.params
    K, e = single_ak_fixture.state["K"], single_ak_fixture.state["e"]

    rejected = [c for c in ak_single.candidates if not c.accepted]
    assert len(rejected) == 1
    r = rejected[0]
    assert r.q == pytest.approx(exp["q_F"], abs=PACKET_TOL)
    assert r.iota == pytest.approx(exp["iota_F"], abs=PACKET_TOL)
    assert r.g == pytest.approx(exp["g_F"], abs=PACKET_TOL)
    assert r.installation_margin == pytest.approx(exp["installation_margin"], abs=PACKET_TOL)
    assert r.tvc_margin == pytest.approx(exp["tvc_margin"], abs=PACKET_TOL)
    assert r.reason == exp["rejection_reason"]
    assert r.branch == "upper_tvc_violating"
    # The packet's point: the rejected root passes every *current* positivity and
    # balance-sheet check. It fails only the forward-looking TVC.
    assert e + r.q * K == pytest.approx(exp["X_F"], abs=PACKET_TOL)
    assert params.preferences.rho * (e + r.q * K) == pytest.approx(exp["C_F_W"], abs=PACKET_TOL)
    assert r.installation_margin > 0.0
    assert r.tvc_margin < 0.0


def test_both_roots_solve_the_original_level_equation(single_ak_fixture, ak_single):
    """Both roots satisfy AK.1 itself, not merely the multiplied polynomial."""
    for c in ak_single.candidates:
        assert abs(ak_price_residual_level(c.q, single_ak_fixture.params)) < 1e-14


def test_closed_form_lambert_roots_agree(single_ak_fixture, ak_single):
    """A third independent route: the Lambert-W closed form."""
    lam = ak_lambert_roots(single_ak_fixture.params)
    found = sorted(c.q for c in ak_single.candidates)
    assert lam["q_lower"] == pytest.approx(found[0], abs=1e-12)
    assert lam["q_upper"] == pytest.approx(found[1], abs=1e-12)


def test_balance_sheet_round_trip_at_either_root(single_ak_fixture, ak_single):
    """The packet's balance sheet closes at both roots (its stated purpose)."""
    from ak_partial_ramsey.events import normalized_from_level

    exp = single_ak_fixture.expected["balance_sheet_at_either_root"]
    K = single_ak_fixture.state["K"]
    e = single_ak_fixture.state["e"]
    B = single_ak_fixture.state["B"]
    A_O = single_ak_fixture.state["A_O"]
    for c in ak_single.candidates:
        q_F = c.q
        F = e + q_F * K
        Theta = F + B
        norm = normalized_from_level(F, Theta, q_F, K)
        assert norm["psi"] == pytest.approx(exp["psi"], abs=1e-12)
        assert norm["B"] == pytest.approx(B, abs=1e-12)
        # Owner holds no post-arrival equity in the packet's fixture.
        foreign_equity = q_F * K - Theta - 0.0
        foreign_safe = B - A_O
        assert foreign_equity == pytest.approx(exp["foreign_equity"], abs=1e-12)
        assert foreign_safe == pytest.approx(exp["foreign_safe"], abs=1e-12)
        assert foreign_equity + foreign_safe == pytest.approx(exp["foreign_net_claim"], abs=1e-12)
        # The foreign net claim is exactly minus domestic net foreign assets.
        assert foreign_equity + foreign_safe == pytest.approx(-(A_O + e), abs=1e-12)


def test_successor_value_and_envelope_identities(single_ak_fixture, ak_single):
    """Value, envelope, and TVC identities at the selected root (mandatory test 10)."""
    params = single_ak_fixture.params
    rho = params.preferences.rho
    K, e = single_ak_fixture.state["K"], single_ak_fixture.state["e"]

    X = ak_single.X_F(K, e)
    closed_form = math.log(rho * X) / rho + (params.rates.rF_bar - rho) / rho**2
    assert ak_single.V_F(K, e) == pytest.approx(closed_form, rel=1e-15)
    assert ak_single.V_F_e(K, e) == pytest.approx(1.0 / (rho * X), rel=1e-15)
    assert ak_single.V_F_K(K, e) == pytest.approx(ak_single.q_F * ak_single.V_F_e(K, e), rel=1e-15)
    assert ak_single.H_F(K) == pytest.approx(ak_single.q_F * K, rel=1e-15)
    # Strict productive-value TVC on the selected branch, violated on the other.
    assert ak_single.tvc_margin > 0.0


def test_independent_evaluator_passes_on_the_fixture(single_ak_fixture, ak_single, tol):
    """The separate evaluator confirms the package by its own routes."""
    params = single_ak_fixture.params
    K, e = single_ak_fixture.state["K"], single_ak_fixture.state["e"]
    report = check_ak_successor(
        q_F=ak_single.q_F,
        K=K,
        e=e,
        V_F=ak_single.V_F(K, e),
        V_F_e=ak_single.V_F_e(K, e),
        V_F_K=ak_single.V_F_K(K, e),
        H_F=ak_single.H_F(K),
        rho=params.preferences.rho,
        rF_bar=params.rates.rF_bar,
        A_bar=params.ak_technology.A_bar,
        varphi=params.installation.varphi,
        delta=params.installation.delta,
        tolerance=tol.identity_atol,
        fd_step=tol.fd_step,
    )
    report.raise_if_failed()
    assert report.passed


def test_no_root_is_selected_when_none_passes_the_tvc(single_ak_fixture, tol):
    """With no admissible root the service refuses rather than returning the nearest."""
    import dataclasses

    from ak_partial_ramsey.errors import BranchFailure
    from ak_partial_ramsey.params import AkRootInterval
    from ak_partial_ramsey.successors.ak import solve_ak_successor

    # Restrict the search interval to the upper, TVC-violating root only.
    with pytest.raises(BranchFailure) as excinfo:
        solve_ak_successor(single_ak_fixture.params, AkRootInterval(q_lo=1.30, q_hi=3.0), tol)
    assert "transversality" in excinfo.value.message
    # The rejected candidate is still reported, with its reason.
    candidates = excinfo.value.detail["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["reason"] == "productive_value_tvc_violated"
    del dataclasses
