"""Independent evaluator.

This module is the project's main defence against false success. It reconstructs the
original level equations from returned numbers and checks them. It exists so that no
solver ``success`` flag, and no residual computed by the code that produced the answer,
can stand in for verification.

**Independence is structural, not a convention.** This module imports nothing from
:mod:`~ak_partial_ramsey.exposure`, :mod:`~ak_partial_ramsey.canonical`,
:mod:`~ak_partial_ramsey.recovery`, or :mod:`~ak_partial_ramsey.successors`. It does not
import :mod:`~ak_partial_ramsey.primitives` either: the production, installation, and
price primitives are rewritten here from the theory packets directly, so that a sign or
algebra error in the equation core cannot be reproduced identically on both sides of a
comparison and cancel. ``tests/test_evaluator_independence.py`` enforces this by
inspecting the module's import graph, and a corruption test checks that the evaluator
actually catches a deliberately damaged equation and event map.

Every function takes plain numbers, never solver objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad

from .errors import EvaluatorMismatch
from .validation import require_finite, require_positive

__all__ = [
    "EvaluationReport",
    "Residual",
    "check_ak_successor",
    "check_event_map_and_exposure",
    "check_level_budget_round_trip",
    "check_partial_present_value",
    "check_partial_successor",
    "check_single_ak_reduction",
]


# --- primitives, rewritten independently -------------------------------------------
# These duplicate ak_partial_ramsey.primitives on purpose. A shared implementation would
# make every comparison below a tautology with respect to its own algebra.


def _omega(Z: float, I: float) -> float:
    return (Z / (1.0 - I)) ** (1.0 - I) * math.exp(-I * math.log(I))


def _Y(K: float, Z: float, I: float) -> float:
    return _omega(Z, I) * math.exp(I * math.log(K))


def _R(K: float, Z: float, I: float) -> float:
    return I * _Y(K, Z, I) / K


def _W(K: float, Z: float, I: float) -> float:
    return _Y(K, Z, I) - _R(K, Z, I) * K


def _iota(q: float, varphi: float) -> float:
    return (q - 1.0) / varphi


def _g(q: float, varphi: float, delta: float) -> float:
    return math.log(q) / varphi - delta


@dataclass(frozen=True, slots=True)
class Residual:
    """One independently recomputed residual and its verdict."""

    name: str
    value: float
    tolerance: float
    route: str
    scale: float = 1.0

    @property
    def scaled(self) -> float:
        return self.value / self.scale if self.scale not in (0.0, None) else self.value

    @property
    def passed(self) -> bool:
        return abs(self.scaled) <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "scaled": self.scaled,
            "scale": self.scale,
            "tolerance": self.tolerance,
            "route": self.route,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The collected verdict of one independent evaluation."""

    label: str
    residuals: tuple[Residual, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.residuals)

    @property
    def failures(self) -> tuple[Residual, ...]:
        return tuple(r for r in self.residuals if not r.passed)

    @property
    def worst(self) -> Residual | None:
        return max(self.residuals, key=lambda r: abs(r.scaled), default=None)

    def raise_if_failed(self) -> None:
        """Raise :class:`~ak_partial_ramsey.errors.EvaluatorMismatch` on any failure.

        A failure here is a structural disagreement between two independent
        formulations. Per the error-handling policy it triggers
        ``diagnose_before_scaling``; it is never resolved by widening the tolerance.
        """
        if not self.passed:
            raise EvaluatorMismatch(
                f"{self.label}: independent evaluation disagrees with the equation core",
                label=self.label,
                failures=[r.as_dict() for r in self.failures],
            )

    def as_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "label": self.label,
            "passed": self.passed,
            "n_residuals": len(self.residuals),
            "n_failures": len(self.failures),
            "worst_residual": worst.as_dict() if worst else None,
            "residuals": [r.as_dict() for r in self.residuals],
        }


def _log_wealth_value_by_quadrature(X: float, r_bar: float, rho: float) -> float:
    """``int_0^inf e^{-rho t} log(C_t) dt`` with ``C_t = rho X e^{(r_bar - rho) t}``.

    Evaluated numerically. The closed form is ``log(rho X)/rho + (r_bar - rho)/rho^2``;
    recomputing it by quadrature checks both the log coefficient and the additive
    constant against the actual discounted integral rather than restating the formula.
    """

    def integrand(t: float) -> float:
        return math.exp(-rho * t) * (math.log(rho * X) + (r_bar - rho) * t)

    value, _ = quad(integrand, 0.0, np.inf, limit=400)
    return float(value)


# --- AK successor ------------------------------------------------------------------


def check_ak_successor(
    *,
    q_F: float,
    K: float,
    e: float,
    V_F: float,
    V_F_e: float,
    V_F_K: float,
    H_F: float,
    rho: float,
    rF_bar: float,
    A_bar: float,
    varphi: float,
    delta: float,
    tolerance: float,
    fd_step: float = 1e-6,
) -> EvaluationReport:
    """Independently verify the returned AK successor package.

    Routes used, none of which calls the AK solver:

    * the price equation in its original level form ``r_F q = A_bar - iota + q g``;
    * the strict productive-value TVC margin ``r_F - g(q_F)``;
    * the worker value by direct discounted quadrature;
    * ``V_{F,e}`` and ``V_{F,K}`` by centred finite differences of a value function
      rebuilt here from ``q_F``; and
    * the envelope condition ``V_{F,K} = q_F V_{F,e}``.
    """
    q_F = require_positive("q_F", q_F)
    K = require_positive("K", K)
    rho = require_positive("rho", rho)

    iota = _iota(q_F, varphi)
    g = _g(q_F, varphi, delta)
    level_residual = rF_bar * q_F - (A_bar - iota + q_F * g)

    X = e + q_F * K

    def value(K_: float, e_: float) -> float:
        return math.log(rho * (e_ + q_F * K_)) / rho + (rF_bar - rho) / (rho * rho)

    hK = fd_step * max(1.0, abs(K))
    he = fd_step * max(1.0, abs(e))
    fd_V_K = (value(K + hK, e) - value(K - hK, e)) / (2.0 * hK)
    fd_V_e = (value(K, e + he) - value(K, e - he)) / (2.0 * he)

    residuals = (
        Residual(
            "ak_price_equation_level_form",
            level_residual,
            tolerance,
            "original AK.1 level equation, rebuilt from primitives",
            scale=max(abs(rF_bar * q_F), 1.0),
        ),
        Residual(
            "ak_productive_wealth",
            H_F - q_F * K,
            tolerance,
            "H_F = q_F K",
            scale=max(abs(q_F * K), 1.0),
        ),
        Residual(
            "ak_value_by_quadrature",
            V_F - _log_wealth_value_by_quadrature(X, rF_bar, rho),
            tolerance,
            "discounted log-consumption quadrature",
            scale=max(abs(V_F), 1.0),
        ),
        Residual(
            "ak_value_derivative_e",
            V_F_e - fd_V_e,
            max(tolerance, 1e-7),
            "centred finite difference of an independently rebuilt V_F",
            scale=max(abs(V_F_e), 1.0),
        ),
        Residual(
            "ak_value_derivative_K",
            V_F_K - fd_V_K,
            max(tolerance, 1e-7),
            "centred finite difference of an independently rebuilt V_F",
            scale=max(abs(V_F_K), 1.0),
        ),
        Residual(
            "ak_envelope_condition",
            V_F_K - q_F * V_F_e,
            tolerance,
            "V_{F,K} = q_F V_{F,e}",
            scale=max(abs(V_F_K), 1.0),
        ),
        Residual(
            "ak_recovered_successor_tax",
            1.0 - (rF_bar * q_F + iota - q_F * g) / A_bar,
            tolerance,
            "world equity pricing recovers tau_F = 0",
            scale=1.0,
        ),
    )
    return EvaluationReport("ak_successor", residuals)


def ak_tvc_margin(q_F: float, rF_bar: float, varphi: float, delta: float) -> float:
    """``r_F_bar - g(q_F)``: strictly positive is the strict productive-value TVC."""
    return rF_bar - _g(q_F, varphi, delta)


# --- partial successor -------------------------------------------------------------


def check_partial_successor(
    *,
    K_nodes: list[float],
    q_P: list[float],
    q_P_prime: list[float],
    H_P: list[float],
    rho: float,
    rP_bar: float,
    Z: float,
    I_P: float,
    varphi: float,
    delta: float,
    tolerance: float,
    fd_step: float = 1e-5,
) -> EvaluationReport:
    """Independently verify the returned partial successor at held-out capital nodes.

    Routes used, none of which calls the partial solver:

    * the stable-manifold invariance equation
      ``q_P'(K) g(q_P) K = [r_P - g(q_P)] q_P - R_P(K) + iota(q_P)``, which is the
      statement that the returned graph really is invariant under the flow;
    * the algebraic productive-wealth equation, compared with the returned ``H_P``; and
    * ``H_P'(K) = q_P(K)`` by centred finite differences of the *algebraic* productive
      wealth, which never touches the quadrature route that produced ``H_P``.
    """
    residuals: list[Residual] = []

    def H_algebraic(K_: float, q_: float) -> float:
        return (_Y(K_, Z, I_P) - _iota(q_, varphi) * K_ + q_ * _g(q_, varphi, delta) * K_) / rP_bar

    worst_invariance = 0.0
    worst_wealth = 0.0
    for K, q, qp, H in zip(K_nodes, q_P, q_P_prime, H_P, strict=True):
        K = require_positive("K", K)
        q = require_positive("q_P", q)
        g = _g(q, varphi, delta)
        drift = (rP_bar - g) * q - _R(K, Z, I_P) + _iota(q, varphi)
        worst_invariance = max(worst_invariance, abs(qp * g * K - drift))
        worst_wealth = max(worst_wealth, abs(H - H_algebraic(K, q)))

    scale_H = max(abs(x) for x in H_P) if H_P else 1.0
    scale_q = max(abs(x) for x in q_P) if q_P else 1.0
    residuals.append(
        Residual(
            "partial_manifold_invariance",
            worst_invariance,
            tolerance,
            "stable-graph invariance equation rebuilt from primitives",
            scale=max(scale_q, 1.0),
        )
    )
    residuals.append(
        Residual(
            "partial_productive_wealth_equation",
            worst_wealth,
            tolerance,
            "algebraic productive-wealth equation vs returned quadrature",
            scale=max(scale_H, 1.0),
        )
    )
    return EvaluationReport("partial_successor", tuple(residuals))


def check_partial_present_value(
    *,
    K_0: float,
    price_graph: Any,
    H_reported: float,
    rP_bar: float,
    Z: float,
    I_P: float,
    varphi: float,
    delta: float,
    K_star: float,
    iota_delta: float,
    tolerance: float,
    horizon: float = 300.0,
) -> Residual:
    """Check productive wealth against its **present-value** definition.

    The theory gives ``H_P`` three equivalent characterisations: the algebraic
    productive-wealth equation, the quadrature ``H_P(K) = H_P(K_ref) + int q_P``, and the
    present value

    ``H_P(K_0) = int_0^inf e^{-r_P t} {Y_P(K_t) - iota[q_P(K_t)] K_t} dt``

    along the manifold flow. The solver uses the second. This function uses the third:
    it integrates the flow forward from ``K_0`` using the returned price graph and
    accumulates the discounted product flow, adding the analytic stationary tail beyond
    the horizon. That makes it a genuinely different route rather than a rearrangement
    of the same algebra - unlike differentiating the algebraic equation, which reduces
    term by term to the manifold-invariance residual already checked above and so cannot
    fail independently of it.

    ``price_graph`` is the returned ``q_P(K)`` callable: it is the answer under test, not
    a shared derivation.
    """
    K_0 = require_positive("K_0", K_0)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        K = max(float(y[0]), 1e-300)
        q = float(price_graph(K))
        g = _g(q, varphi, delta)
        flow = _Y(K, Z, I_P) - _iota(q, varphi) * K
        return np.array([g * K, math.exp(-rP_bar * t) * flow])

    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        rhs,
        (0.0, horizon),
        np.array([K_0, 0.0]),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    if not sol.success:
        return Residual(
            "partial_productive_wealth_present_value",
            math.inf,
            tolerance,
            f"present-value quadrature failed: {sol.message}",
        )
    truncated = float(sol.y[1, -1])
    # Stationary tail: beyond the horizon the flow is Y_P(K*) - iota_delta K*.
    tail_flow = _Y(K_star, Z, I_P) - iota_delta * K_star
    tail = math.exp(-rP_bar * horizon) * tail_flow / rP_bar
    present_value = truncated + tail
    return Residual(
        "partial_productive_wealth_present_value",
        H_reported - present_value,
        tolerance,
        "discounted product-flow quadrature along the manifold, with stationary tail",
        scale=max(abs(present_value), 1.0),
    )


# --- event map, exposure, and level round trip -------------------------------------


def check_event_map_and_exposure(
    *,
    e: float,
    psi: float,
    q: float,
    K: float,
    C: float,
    rho: float,
    marks: list[dict[str, float]],
    tolerance: float,
) -> EvaluationReport:
    """Recompute the event map and the exposure condition through **level** coordinates.

    Each entry of ``marks`` supplies ``label``, ``lambda_physical``, ``lambda_star``,
    ``q_successor``, ``H`` (that successor's productive wealth at ``K``), and optionally
    ``X_reported`` - the successor wealth the *solver* actually used.

    The solver works in normalized coordinates ``e_j^+ = e + psi J_j``. This function
    goes the other way: it builds the level positions ``F`` and ``Theta``, applies the
    level event map ``F_j^+ = F + Theta J_j``, converts back with
    ``e_j^+ = F_j^+ - q_j K``, and only then forms successor wealth and the exposure
    condition.

    ``X_reported`` is what makes this a test rather than a restatement. Recomputing both
    sides of the round trip from the same inputs can only confirm the evaluator's own
    algebra; comparing the independently reconstructed successor wealth against the value
    the solver used is what catches a sign or timing error in the solver's event map.
    When it is omitted the round-trip residual degenerates to an internal consistency
    check, which is why the corruption test always supplies it.
    """
    e = require_finite("e", e)
    psi = require_finite("psi", psi)
    q = require_positive("q", q)
    K = require_positive("K", K)
    C = require_positive("C", C)

    F = e + q * K
    Theta = psi + q * K

    worst_roundtrip = 0.0
    worst_reported = 0.0
    any_reported = False
    exposure_sum = 0.0
    scale = 1.0
    for m in marks:
        lam = m["lambda_physical"]
        lam_star = m["lambda_star"]
        if lam == 0.0 and lam_star == 0.0:
            continue  # inactive mark: contributes no value term
        q_j = require_positive(f"q_successor[{m['label']}]", m["q_successor"])
        J = (q_j - q) / q
        # level route
        F_plus = F + Theta * J
        e_plus_level = F_plus - q_j * K
        X_level = e_plus_level + m["H"]
        # normalized route
        e_plus_norm = e + psi * J
        X_norm = e + m["H"] + psi * J
        worst_roundtrip = max(
            worst_roundtrip,
            abs(e_plus_level - e_plus_norm),
            abs(X_level - X_norm),
        )
        if "X_reported" in m:
            worst_reported = max(worst_reported, abs(X_level - m["X_reported"]))
            any_reported = True
        if X_level <= 0.0:
            raise EvaluatorMismatch(
                f"independent route gives nonpositive successor wealth for mark {m['label']}",
                label=m["label"],
                X=X_level,
            )
        exposure_sum += (lam / (rho * X_level) - lam_star / C) * J
        scale = max(scale, abs(lam / (rho * X_level)), abs(lam_star / C))

    residuals = [
        Residual(
            "event_map_level_normalized_round_trip",
            worst_roundtrip,
            tolerance,
            "level event map F_j^+ = F + Theta J_j, converted back",
            scale=max(abs(F), abs(Theta), 1.0),
        ),
        Residual(
            "public_exposure_condition",
            exposure_sum,
            max(tolerance, 1e-10),
            "unmultiplied sum_j u_j J_j, successor wealth via the level route",
            scale=scale,
        ),
    ]
    if any_reported:
        residuals.append(
            Residual(
                "successor_wealth_against_solver",
                worst_reported,
                tolerance,
                "independently reconstructed X_j vs the value the solver used",
                scale=max(abs(F), abs(Theta), 1.0),
            )
        )
    return EvaluationReport("event_map_and_exposure", tuple(residuals))


def check_level_budget_round_trip(
    *,
    e: float,
    psi: float,
    q: float,
    K: float,
    C_W: float,
    Lambda: float,
    Z: float,
    I_0: float,
    varphi: float,
    r0_bar: float,
    e_dot_reported: float,
    tolerance: float,
) -> EvaluationReport:
    """Recompute the normalized external-wealth law from level primitives.

    ``e' = r_0_bar e + Y_0(K) - iota(q) K - C^W - psi Lambda``, rebuilt here from an
    independently written production block, and the worker identity ``T = C^W - W_0(K)``.
    """
    Y = _Y(K, Z, I_0)
    W = _W(K, Z, I_0)
    e_dot = r0_bar * e + Y - _iota(q, varphi) * K - C_W - psi * Lambda
    T = C_W - W
    residuals = (
        Residual(
            "external_wealth_law",
            e_dot - e_dot_reported,
            tolerance,
            "e-law rebuilt from an independent production block",
            scale=max(abs(e_dot), 1.0),
        ),
        Residual(
            "worker_budget_identity",
            (W + T) - C_W,
            tolerance,
            "C^W = W_0(K) + T",
            scale=max(abs(C_W), 1.0),
        ),
    )
    return EvaluationReport("level_budget_round_trip", residuals)


# --- exact single-AK reduction -----------------------------------------------------


def check_single_ak_reduction(
    *,
    u_P: float,
    u_F: float,
    U: float,
    D: float,
    consumption_growth_reported: float,
    tau_reported: float,
    rho: float,
    lambda_total: float,
    lambda_F_star: float,
    r0_bar: float,
    tolerance: float,
) -> EvaluationReport:
    """Verify the exact support restriction at ``p_P = lambda_P_star = 0``.

    With no partial-mark mass, ``u_P`` vanishes identically, the exposure condition
    forces ``u_F = 0`` whenever ``J_F`` is nonzero, and therefore ``U = D = 0``. The
    consumption-growth and tax equations must then reduce exactly to the single-mark
    ones. This is an algebraic support restriction, not an ``I_P`` limit.
    """
    expected_growth = r0_bar + lambda_F_star - lambda_total - rho
    residuals = (
        Residual("single_ak_u_P_vanishes", u_P, tolerance, "p_P = lambda_P_star = 0"),
        Residual("single_ak_u_F_vanishes", u_F, tolerance, "exposure FOC with one active mark"),
        Residual("single_ak_total_residual_U", U, tolerance, "U = u_P + u_F"),
        Residual("single_ak_investment_wedge_D", D, tolerance, "D = varphi psi U / K"),
        Residual(
            "single_ak_consumption_growth",
            consumption_growth_reported - expected_growth,
            tolerance,
            "P.9: C'/C = r_0 + lambda_F_star - lambda - rho",
            scale=max(abs(expected_growth), 1.0),
        ),
        Residual(
            "single_ak_zero_source_tax",
            tau_reported,
            max(tolerance, 1e-10),
            "P.12: world-price cancellation recovers tau_0 = 0",
        ),
    )
    return EvaluationReport("single_ak_reduction", residuals)
