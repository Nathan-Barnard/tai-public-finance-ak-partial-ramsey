"""The independent evaluator is structurally independent, and it catches corruption.

Two claims are tested here, and they are different claims:

1. **Structural independence.** The evaluator's import graph does not reach the equation
   core or the successor services. This is checked by parsing the module, not by
   convention, so the independence cannot rot silently.

2. **Detection power.** An evaluator that shares its algebra with the code under test
   will pass everything, independence notwithstanding. So each check is also run against
   a *deliberately corrupted* equation, event map, and value, and is required to fail.
   A check that cannot fail is not evidence.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ak_partial_ramsey import evaluator
from ak_partial_ramsey.evaluator import (
    check_ak_successor,
    check_event_map_and_exposure,
    check_level_budget_round_trip,
    check_partial_successor,
    check_single_ak_reduction,
)

FORBIDDEN = {"exposure", "canonical", "recovery", "successors", "assembly", "primitives"}


def test_evaluator_imports_no_solver_module():
    """Parse the evaluator and assert its first-party imports stay inside the safe set.

    ``primitives`` is on the forbidden list too. The evaluator rewrites the production,
    installation, and price functions itself, so that an algebra or sign error in the
    equation core cannot be reproduced identically on both sides of a comparison and
    cancel out.
    """
    source = pathlib.Path(evaluator.__file__).read_text()
    tree = ast.parse(source)
    first_party: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            first_party.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ak_partial_ramsey"):
                    first_party.add(alias.name.split(".")[1])
    offending = first_party & FORBIDDEN
    assert not offending, f"evaluator imports solver modules: {sorted(offending)}"
    assert first_party <= {"errors", "validation"}, sorted(first_party)


def test_evaluator_defines_its_own_primitives():
    """The duplicated primitives really are present, so the independence has substance."""
    for name in ("_omega", "_Y", "_R", "_W", "_iota", "_g"):
        assert hasattr(evaluator, name), f"evaluator is missing its own {name}"


def test_evaluator_primitives_agree_with_the_core(two_mark_fixture):
    """Independent rewrite, same answer: agreement is the point, sharing is not."""
    from ak_partial_ramsey.primitives import (
        capital_growth,
        installation_rate,
        omega_Z,
        task_output,
        task_rental,
        task_wage,
    )

    p = two_mark_fixture.params
    tech = p.pre_arrival_technology
    Z, I_0 = tech.Z, tech.I
    varphi, delta = p.installation.varphi, p.installation.delta
    assert evaluator._omega(Z, I_0) == pytest.approx(omega_Z(tech), rel=1e-14)
    for K in (1.0, 10.0, 320.0):
        assert evaluator._Y(K, Z, I_0) == pytest.approx(task_output(K, tech), rel=1e-14)
        assert evaluator._R(K, Z, I_0) == pytest.approx(task_rental(K, tech), rel=1e-14)
        assert evaluator._W(K, Z, I_0) == pytest.approx(task_wage(K, tech), rel=1e-13)
    for q in (0.9, 1.06, 1.5):
        assert evaluator._iota(q, varphi) == pytest.approx(
            installation_rate(q, p.installation), rel=1e-14
        )
        assert evaluator._g(q, varphi, delta) == pytest.approx(
            capital_growth(q, p.installation), rel=1e-14
        )


# --- detection power: corrupted equations must be caught -----------------------------


def _ak_kwargs(fixture, ak, tol):
    K, e = fixture.state["K"], fixture.state["e"]
    p = fixture.params
    return {
        "q_F": ak.q_F,
        "K": K,
        "e": e,
        "V_F": ak.V_F(K, e),
        "V_F_e": ak.V_F_e(K, e),
        "V_F_K": ak.V_F_K(K, e),
        "H_F": ak.H_F(K),
        "rho": p.preferences.rho,
        "rF_bar": p.rates.rF_bar,
        "A_bar": p.ak_technology.A_bar,
        "varphi": p.installation.varphi,
        "delta": p.installation.delta,
        "tolerance": tol.identity_atol,
        "fd_step": tol.fd_step,
    }


def test_uncorrupted_ak_package_passes(single_ak_fixture, ak_single, tol):
    """The negative control: without corruption, everything passes."""
    assert check_ak_successor(**_ak_kwargs(single_ak_fixture, ak_single, tol)).passed


@pytest.mark.parametrize("relative_error", [1e-6, 1e-4, 1e-2])
def test_corrupted_ak_price_is_detected(single_ak_fixture, ak_single, tol, relative_error):
    """A wrong q_F fails the level-form price equation."""
    kwargs = _ak_kwargs(single_ak_fixture, ak_single, tol)
    kwargs["q_F"] = ak_single.q_F * (1.0 + relative_error)
    report = check_ak_successor(**kwargs)
    assert not report.passed
    failed = {r.name for r in report.failures}
    assert "ak_price_equation_level_form" in failed


def test_corrupted_productive_wealth_is_detected(single_ak_fixture, ak_single, tol):
    """H_F != q_F K is caught."""
    kwargs = _ak_kwargs(single_ak_fixture, ak_single, tol)
    kwargs["H_F"] = kwargs["H_F"] * 1.001
    report = check_ak_successor(**kwargs)
    assert not report.passed
    assert "ak_productive_wealth" in {r.name for r in report.failures}


def test_corrupted_value_constant_is_detected(single_ak_fixture, ak_single, tol):
    """A wrong additive constant in V_F is caught by the discounted quadrature.

    This is the check that the closed form cannot make for itself: the quadrature
    reconstructs the integral rather than restating the formula.
    """
    kwargs = _ak_kwargs(single_ak_fixture, ak_single, tol)
    kwargs["V_F"] = kwargs["V_F"] + 1e-6
    report = check_ak_successor(**kwargs)
    assert not report.passed
    assert "ak_value_by_quadrature" in {r.name for r in report.failures}


def test_corrupted_envelope_is_detected(single_ak_fixture, ak_single, tol):
    """V_{F,K} != q_F V_{F,e} is caught."""
    kwargs = _ak_kwargs(single_ak_fixture, ak_single, tol)
    kwargs["V_F_K"] = kwargs["V_F_K"] * 1.0001
    report = check_ak_successor(**kwargs)
    assert not report.passed
    assert {"ak_envelope_condition", "ak_value_derivative_K"} & {r.name for r in report.failures}


def test_raise_if_failed_raises_and_carries_evidence(single_ak_fixture, ak_single, tol):
    from ak_partial_ramsey.errors import EvaluatorMismatch

    kwargs = _ak_kwargs(single_ak_fixture, ak_single, tol)
    kwargs["q_F"] = ak_single.q_F * 1.01
    with pytest.raises(EvaluatorMismatch) as exc:
        check_ak_successor(**kwargs).raise_if_failed()
    assert exc.value.detail["failures"]


# --- corrupted event map ---------------------------------------------------------------


def _marks_payload(H_P: float, H_F: float, q_P: float, q_F: float):
    return [
        {
            "label": "P",
            "lambda_physical": 0.012,
            "lambda_star": 0.010,
            "q_successor": q_P,
            "H": H_P,
        },
        {
            "label": "F",
            "lambda_physical": 0.008,
            "lambda_star": 0.015,
            "q_successor": q_F,
            "H": H_F,
        },
    ]


def _correct_wealth(e, psi, q, q_j, H):
    """The event map as the theory states it: e_j^+ = e + psi J_j, X_j = e_j^+ + H_j."""
    J = (q_j - q) / q
    return e + psi * J + H


def test_uncorrupted_event_map_passes(tol):
    e, psi, q, K, C, rho = -6.0, -4.94654, 1.10, 10.0, 0.4, 0.04
    q_P, q_F, H_P, H_F = 1.21, 1.65, 16.0, 14.0
    marks = _marks_payload(H_P, H_F, q_P, q_F)
    for m, H in zip(marks, (H_P, H_F), strict=True):
        m["X_reported"] = _correct_wealth(e, psi, q, m["q_successor"], H)
    report = check_event_map_and_exposure(
        e=e, psi=psi, q=q, K=K, C=C, rho=rho, marks=marks, tolerance=tol.identity_atol
    )
    assert report.passed


@pytest.mark.parametrize(
    ("name", "corrupt"),
    [
        # A sign flip in the event map: e - psi J instead of e + psi J.
        ("sign_flip", lambda e, psi, q, q_j, H: e - psi * ((q_j - q) / q) + H),
        # Gain measured against the successor price instead of the pre-arrival one.
        ("wrong_denominator", lambda e, psi, q, q_j, H: e + psi * ((q_j - q) / q_j) + H),
        # Productive wealth omitted from successor wealth.
        ("omitted_wealth", lambda e, psi, q, q_j, H: e + psi * ((q_j - q) / q)),
        # Gain applied after rebalancing rather than before: J computed at q_j.
        ("wrong_timing", lambda e, psi, q, q_j, H: e + psi * ((q_j - q) / q) * 0.999 + H),
    ],
)
def test_corrupted_event_map_is_detected(tol, name, corrupt):
    """Each corruption of the event map must be caught, not absorbed."""
    e, psi, q, K, C, rho = -6.0, -4.94654, 1.10, 10.0, 0.4, 0.04
    q_P, q_F, H_P, H_F = 1.21, 1.65, 16.0, 14.0
    marks = _marks_payload(H_P, H_F, q_P, q_F)
    for m, H in zip(marks, (H_P, H_F), strict=True):
        m["X_reported"] = corrupt(e, psi, q, m["q_successor"], H)
    report = check_event_map_and_exposure(
        e=e, psi=psi, q=q, K=K, C=C, rho=rho, marks=marks, tolerance=tol.identity_atol
    )
    assert not report.passed, f"corruption {name!r} went undetected"
    assert "successor_wealth_against_solver" in {r.name for r in report.failures}


def test_corrupted_exposure_root_is_detected(tol):
    """A psi that does not solve the unmultiplied FOC leaves a nonzero residual."""
    from ak_partial_ramsey.exposure import Mark, solve_public_exposure

    m = (Mark("P", 0.012, 0.010, 0.1, 10.0), Mark("F", 0.008, 0.015, 0.5, 8.0))
    good = solve_public_exposure(m, 0.4, 0.04, tol)
    q = 1.10
    marks = [
        {
            "label": mk.label,
            "lambda_physical": mk.lambda_physical,
            "lambda_star": mk.lambda_star,
            "q_successor": q * (1.0 + mk.J),
            "H": mk.h - (-6.0),
        }
        for mk in m
    ]
    bad_psi = good.psi * 1.05
    report = check_event_map_and_exposure(
        e=-6.0,
        psi=bad_psi,
        q=q,
        K=10.0,
        C=0.4,
        rho=0.04,
        marks=marks,
        tolerance=tol.identity_atol,
    )
    assert not report.passed
    assert "public_exposure_condition" in {r.name for r in report.failures}


# --- corrupted budget and reduction ----------------------------------------------------


def test_corrupted_budget_law_is_detected(two_mark_fixture, tol):
    from ak_partial_ramsey.canonical import state_e_dot

    p = two_mark_fixture.params
    args = dict(K=320.0, e=5.0, q=1.15, C=10.0, psi=-20.0, Lambda=0.0004)
    correct = state_e_dot(
        args["K"], args["e"], args["q"], args["C"], args["psi"], args["Lambda"], p
    )
    common = dict(
        e=args["e"],
        psi=args["psi"],
        q=args["q"],
        K=args["K"],
        C_W=args["C"],
        Lambda=args["Lambda"],
        Z=p.pre_arrival_technology.Z,
        I_0=p.pre_arrival_technology.I,
        varphi=p.installation.varphi,
        r0_bar=p.rates.r0_bar,
        tolerance=tol.identity_atol,
    )
    assert check_level_budget_round_trip(e_dot_reported=correct, **common).passed
    bad = check_level_budget_round_trip(e_dot_reported=correct + 1e-6, **common)
    assert not bad.passed
    assert "external_wealth_law" in {r.name for r in bad.failures}


def test_corrupted_reduction_is_detected(tol):
    """A nonzero U or a nonzero recovered tax on the support branch must be caught."""
    base = dict(
        u_P=0.0,
        u_F=0.0,
        U=0.0,
        D=0.0,
        consumption_growth_reported=0.03 + 0.015 - 0.02 - 0.04,
        tau_reported=0.0,
        rho=0.04,
        lambda_total=0.02,
        lambda_F_star=0.015,
        r0_bar=0.03,
        tolerance=tol.identity_atol,
    )
    assert check_single_ak_reduction(**base).passed
    for field, value, expected in (
        ("U", 1e-6, "single_ak_total_residual_U"),
        ("D", 1e-6, "single_ak_investment_wedge_D"),
        ("tau_reported", 1e-6, "single_ak_zero_source_tax"),
        ("u_F", 1e-6, "single_ak_u_F_vanishes"),
    ):
        report = check_single_ak_reduction(**{**base, field: value})
        assert not report.passed, field
        assert expected in {r.name for r in report.failures}


def test_corrupted_partial_manifold_is_detected(partial_two_mark, two_mark_fixture, tol):
    """A price graph that is not flow-invariant fails the invariance residual."""
    p = two_mark_fixture.params
    lo, hi = partial_two_mark.certified_domain
    nodes = [lo + f * (hi - lo) for f in (0.2, 0.4, 0.6, 0.8)]
    common = dict(
        rho=p.preferences.rho,
        rP_bar=p.rates.rP_bar,
        Z=p.partial_technology.Z,
        I_P=p.partial_technology.I,
        varphi=p.installation.varphi,
        delta=p.installation.delta,
        tolerance=tol.cross_route_atol,
    )
    good = check_partial_successor(
        K_nodes=nodes,
        q_P=[partial_two_mark.q_P(K) for K in nodes],
        q_P_prime=[partial_two_mark.q_P_derivative(K) for K in nodes],
        H_P=[partial_two_mark.H_P(K) for K in nodes],
        **common,
    )
    assert good.passed

    bad = check_partial_successor(
        K_nodes=nodes,
        q_P=[partial_two_mark.q_P(K) * 1.001 for K in nodes],
        q_P_prime=[partial_two_mark.q_P_derivative(K) for K in nodes],
        H_P=[partial_two_mark.H_P(K) for K in nodes],
        **common,
    )
    assert not bad.passed
    assert {"partial_manifold_invariance", "partial_productive_wealth_equation"} & {
        r.name for r in bad.failures
    }
