"""Absorbing fixed-``I_P`` partial-automation successor.

This is the finite-task-share model with ``0 < I_0 < I_P < 1``. Its productive subsystem
is

``K' = g(q) K``,   ``q' = [r_P_bar - g(q)] q - R_P(K) + iota(q)``

with rest point ``q* = q_delta = exp(varphi delta)`` and ``K_P* = [I_P Omega_Z(I_P)/U_P]
** (1/(1 - I_P))``, where ``U_P = r_P_bar q_delta + iota_delta``. The linearisation has
characteristic polynomial ``nu^2 - r_P_bar nu - (1 - I_P) U_P/(varphi q_delta)``, hence
one stable and one unstable root, and a one-dimensional stable manifold ``q_P(K)`` with
slope ``varphi q_delta nu_- / K_P*`` at the rest point.

Two independent routes to the manifold are kept, as CS011 requires: high-accuracy
backward integration from a linearised offset, and a two-point collocation solve seeded
from the *linear* stable solution rather than from the integrated one. Productive wealth
likewise has two routes - quadrature of ``q_P`` and the algebraic productive-wealth
equation - whose disagreement is reported, never averaged.

The AK successor is **not** obtained from this module by sending ``I_P`` to one. See
:func:`refuse_ak_by_task_share_limit`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import PchipInterpolator

from ..errors import BranchFailure, DomainError, NestingError
from ..params import ModelParameters, PartialCapitalInterval
from ..primitives import (
    capital_growth,
    installation_rate,
    omega_Z,
    task_output,
    task_rental,
    task_rental_derivative,
)
from ..tolerances import SolverTolerances
from ..validation import require_finite, require_positive

__all__ = [
    "PartialLinearization",
    "PartialStationaryPoint",
    "PartialSuccessor",
    "partial_linearization",
    "partial_stationary_point",
    "refuse_ak_by_task_share_limit",
    "solve_partial_successor",
]


def refuse_ak_by_task_share_limit(I_P: float) -> None:
    """Refuse any attempt to construct the AK successor as ``I_P -> 1``.

    The AK block is a separately defined technology: direct ``Y_F = A_bar K`` production
    with ``W_F = 0``. The finite-task-share formula does not define it, does not define
    its wage boundary, and does not define its value normalisation. Sending ``I_P`` to
    one is a technology substitution error, so it raises here rather than returning a
    plausible-looking number.
    """
    I_P = require_finite("I_P", I_P)
    raise NestingError(
        "the AK successor is never obtained by sending I_P to one; it is a separately "
        "defined technology with direct A_bar*K production and a zero wage. Use "
        "successors.ak.solve_ak_successor instead.",
        I_P=I_P,
        requested="I_P -> 1 as an AK construction",
    )


# --- stationary point -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartialStationaryPoint:
    """The interior production rest point of the partial successor."""

    K_star: float
    q_delta: float
    iota_delta: float
    U_P: float
    residual_K: float
    residual_q: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "K_star": self.K_star,
            "q_delta": self.q_delta,
            "iota_delta": self.iota_delta,
            "U_P": self.U_P,
            "residual_K": self.residual_K,
            "residual_q": self.residual_q,
        }


def _partial_technology(params: ModelParameters):
    if params.partial_technology is None:
        raise DomainError(
            "these parameters carry no partial successor technology",
            has_partial_successor=False,
        )
    return params.partial_technology


def partial_field(K: float, q: float, params: ModelParameters) -> tuple[float, float]:
    """The productive vector field ``(K', q')`` of the absorbing partial successor."""
    tech = _partial_technology(params)
    rP = params.rates.rP_bar
    g = capital_growth(q, params.installation)
    return (
        g * K,
        (rP - g) * q - task_rental(K, tech) + installation_rate(q, params.installation),
    )


def partial_stationary_point(params: ModelParameters) -> PartialStationaryPoint:
    """Solve ``(K_P*, q_delta)``.

    ``K' = 0`` forces ``g(q) = 0``, whose only positive solution is
    ``q_delta = exp(varphi delta)``. ``q' = 0`` then requires ``R_P(K) = U_P``, and
    ``R_P`` is strictly decreasing from infinity to zero, so a positive ``K_P*`` exists
    exactly when ``U_P > 0`` and is then unique.
    """
    tech = _partial_technology(params)
    q_delta = params.installation.q_delta
    iota_delta = installation_rate(q_delta, params.installation)
    U_P = params.rates.rP_bar * q_delta + iota_delta
    if U_P <= 0.0:
        raise BranchFailure(
            "the partial successor has no positive production rest point: its scalar "
            "marginal-product target U_P is nonpositive",
            U_P=U_P,
            q_delta=q_delta,
            iota_delta=iota_delta,
            r_P_bar=params.rates.rP_bar,
        )
    I_P = tech.I
    K_star = (I_P * omega_Z(tech) / U_P) ** (1.0 / (1.0 - I_P))
    dK, dq = partial_field(K_star, q_delta, params)
    return PartialStationaryPoint(
        K_star=K_star,
        q_delta=q_delta,
        iota_delta=iota_delta,
        U_P=U_P,
        residual_K=dK,
        residual_q=dq,
    )


# --- linearisation ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartialLinearization:
    """Linearisation at the rest point: eigenvalues, eigenvectors, and the slope."""

    jacobian: tuple[tuple[float, float], tuple[float, float]]
    nu_minus: float
    nu_plus: float
    stable_eigenvector: tuple[float, float]
    unstable_left_eigenvector: tuple[float, float]
    manifold_slope: float
    characteristic_residual: float
    spectral_gap: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "jacobian": [list(r) for r in self.jacobian],
            "nu_minus": self.nu_minus,
            "nu_plus": self.nu_plus,
            "stable_eigenvector": list(self.stable_eigenvector),
            "unstable_left_eigenvector": list(self.unstable_left_eigenvector),
            "manifold_slope": self.manifold_slope,
            "characteristic_residual": self.characteristic_residual,
            "spectral_gap": self.spectral_gap,
        }


def partial_linearization(
    params: ModelParameters, point: PartialStationaryPoint
) -> PartialLinearization:
    """Linearise the productive subsystem and verify one stable, one unstable root."""
    tech = _partial_technology(params)
    varphi = params.installation.varphi
    a12 = point.K_star / (varphi * point.q_delta)
    a21 = -task_rental_derivative(point.K_star, tech)
    a22 = params.rates.rP_bar
    A = np.array([[0.0, a12], [a21, a22]])

    eigvals, eigvecs = np.linalg.eig(A)
    if np.any(np.abs(eigvals.imag) > 1e-12):
        raise BranchFailure(
            "the partial rest point has complex eigenvalues; it is not a saddle",
            eigenvalues=[complex(v) for v in eigvals],
        )
    real = eigvals.real
    stable_idx = int(np.argmin(real))
    unstable_idx = int(np.argmax(real))
    nu_minus, nu_plus = float(real[stable_idx]), float(real[unstable_idx])
    if not (nu_minus < 0.0 < nu_plus):
        raise BranchFailure(
            "the partial rest point does not have exactly one stable and one unstable root",
            nu_minus=nu_minus,
            nu_plus=nu_plus,
        )

    v = eigvecs[:, stable_idx].real
    if v[0] == 0.0:
        raise BranchFailure(
            "the stable eigenvector has no capital component, so the manifold is not a "
            "graph over K",
            stable_eigenvector=[float(v[0]), float(v[1])],
        )
    v = v / v[0]  # normalise so the K component is one

    # Left eigenvector for the unstable root: used as the terminal projection in the
    # independent collocation route.
    w_vals, w_vecs = np.linalg.eig(A.T)
    w_idx = int(np.argmax(w_vals.real))
    w = w_vecs[:, w_idx].real
    w = w / np.linalg.norm(w)

    analytic_slope = varphi * point.q_delta * nu_minus / point.K_star
    char_residual = (
        nu_minus**2 - a22 * nu_minus - (1.0 - tech.I) * point.U_P / (varphi * point.q_delta)
    )
    return PartialLinearization(
        jacobian=((0.0, a12), (a21, a22)),
        nu_minus=nu_minus,
        nu_plus=nu_plus,
        stable_eigenvector=(float(v[0]), float(v[1])),
        unstable_left_eigenvector=(float(w[0]), float(w[1])),
        manifold_slope=analytic_slope,
        characteristic_residual=char_residual,
        spectral_gap=min(abs(nu_minus), abs(nu_plus)),
    )


# --- stable manifold by backward integration --------------------------------------


def _integrate_branch(
    params: ModelParameters,
    point: PartialStationaryPoint,
    lin: PartialLinearization,
    interval: PartialCapitalInterval,
    tolerances: SolverTolerances,
    *,
    offset: float,
    sign: float,
    n_dense: int = 1501,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Trace one side of the stable manifold by integrating the reversed field.

    Forward time contracts onto the rest point, so the manifold is swept out by
    integrating ``-f``. ``sign`` selects the branch above or below ``K_P*``.
    """
    v = np.array(lin.stable_eigenvector, dtype=float)
    v = v / np.linalg.norm(v)
    z0 = np.array([point.K_star, point.q_delta]) + sign * offset * point.K_star * v

    target = interval.K_hi if sign > 0 else interval.K_lo

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        K, q = float(z[0]), float(z[1])
        if K <= 0.0 or q <= 0.0:
            return np.array([0.0, 0.0])
        dK, dq = partial_field(K, q, params)
        return np.array([-dK, -dq])  # reversed field: away from the rest point

    def hit_boundary(_t: float, z: np.ndarray) -> float:
        return float(z[0]) - target

    hit_boundary.terminal = True
    hit_boundary.direction = 1.0 if sign > 0 else -1.0

    def leaves_domain(_t: float, z: np.ndarray) -> float:
        return min(float(z[0]), float(z[1])) - 1e-12

    leaves_domain.terminal = True

    # Horizon needed for the reversed stable mode to grow the *displacement* from its
    # linearised starting size out to the declared boundary. The displacement grows like
    # exp(|nu_minus| t), so the requirement is a log-amplification, not a log capital
    # ratio. The terminal event stops the integration when the boundary is actually
    # reached; the factor and additive slack cover the nonlinear part of the trace,
    # where the true rate departs from the linear one.
    initial_displacement = offset * point.K_star * abs(v[0])
    target_displacement = abs(target - point.K_star)
    if target_displacement <= initial_displacement:
        raise DomainError(
            "the declared capital boundary lies inside the linearisation offset, so "
            "there is no manifold to trace on this side",
            branch="upper" if sign > 0 else "lower",
            target=target,
            K_star=point.K_star,
            initial_displacement=initial_displacement,
        )
    t_end = math.log(target_displacement / initial_displacement) / abs(lin.nu_minus) * 1.5 + 1.0

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        z0,
        method="DOP853",
        rtol=tolerances.ivp_rtol,
        atol=tolerances.ivp_atol,
        dense_output=True,
        events=(hit_boundary, leaves_domain),
        max_step=t_end / 200.0,
    )
    if not sol.success:
        raise BranchFailure(
            "stable-manifold integration failed on the partial successor",
            branch="upper" if sign > 0 else "lower",
            message=sol.message,
        )
    # Sample the dense output rather than the integrator's own step points. The step
    # points are chosen for local error control and are far too sparse to carry the
    # manifold's accuracy into a downstream interpolant; the dense solution is accurate
    # to the same tolerance everywhere between them.
    t_dense = np.linspace(0.0, float(sol.t[-1]), n_dense)
    z_dense = sol.sol(t_dense)
    K = z_dense[0]
    q = z_dense[1]
    diagnostics = {
        "branch": "upper" if sign > 0 else "lower",
        "offset": offset,
        "n_steps": int(sol.t.size),
        "n_dense_samples": int(n_dense),
        "n_rhs_evaluations": int(sol.nfev),
        "t_end_used": float(sol.t[-1]),
        "reached_declared_boundary": bool(sol.t_events[0].size > 0),
        "left_positive_domain": bool(sol.t_events[1].size > 0),
        "K_range": [float(np.min(K)), float(np.max(K))],
    }
    return K, q, diagnostics


def _direction_check(
    params: ModelParameters,
    point: PartialStationaryPoint,
    lin: PartialLinearization,
    K0: float,
    q0: float,
    tolerances: SolverTolerances,
) -> dict[str, Any]:
    """Verify that forward time on the constructed manifold contracts to the rest point.

    This is the integration-direction check: a branch traced with the wrong sign of the
    reversed field would diverge under forward time instead of contracting.

    The horizon has to be chosen with care. A point that is only *numerically* on the
    stable manifold carries a round-off-sized unstable component that grows like
    ``exp(nu_plus t)``, so a long horizon eventually reports divergence for a perfectly
    correct branch. The horizon here is capped so that the unstable mode can amplify by
    at most a factor of one hundred, which leaves the genuine stable contraction
    ``exp(nu_minus T)`` clearly visible while keeping the test sharp: a wrongly directed
    branch diverges immediately, not asymptotically.
    """

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        K, q = float(z[0]), float(z[1])
        if K <= 0.0 or q <= 0.0:
            return np.array([0.0, 0.0])
        return np.array(partial_field(K, q, params))

    horizon = min(math.log(100.0) / lin.nu_plus, 10.0 / abs(lin.nu_minus))
    d0 = math.hypot(K0 - point.K_star, q0 - point.q_delta)
    sol = solve_ivp(
        rhs,
        (0.0, horizon),
        np.array([K0, q0]),
        method="DOP853",
        rtol=tolerances.ivp_rtol,
        atol=tolerances.ivp_atol,
    )
    if not sol.success:
        return {
            "contracts": False,
            "reason": sol.message,
            "initial_distance": d0,
            "horizon": horizon,
        }
    K1, q1 = float(sol.y[0, -1]), float(sol.y[1, -1])
    d1 = math.hypot(K1 - point.K_star, q1 - point.q_delta)
    ratio = d1 / d0 if d0 > 0 else math.nan
    return {
        "contracts": d1 < d0,
        "initial_distance": d0,
        "final_distance": d1,
        "contraction_ratio": ratio,
        "horizon": horizon,
        # What pure stable-mode decay would give over the same horizon. A ratio far
        # above this signals unstable-mode contamination even when the test passes.
        "linear_predicted_ratio": math.exp(lin.nu_minus * horizon),
    }


# --- independent collocation route ------------------------------------------------


def _collocation_price_at(
    params: ModelParameters,
    point: PartialStationaryPoint,
    lin: PartialLinearization,
    K_node: float,
    tolerances: SolverTolerances,
) -> dict[str, Any]:
    """Recover ``q_P(K_node)`` by two-point collocation, independently of the IVP route.

    The boundary conditions are the inherited capital ``K(0) = K_node`` and a terminal
    stable-manifold condition expressed as the vanishing of the unstable component,
    ``w_u . (y(T) - y*) = 0``.

    The initial guess is the *linear* stable solution, not the integrated manifold, so
    this route shares no numerical state with :func:`_integrate_branch`. Agreement
    between the two is therefore evidence, not a tautology.
    """
    v = np.array(lin.stable_eigenvector, dtype=float)
    w = np.array(lin.unstable_left_eigenvector, dtype=float)
    y_star = np.array([point.K_star, point.q_delta])
    eps = (K_node - point.K_star) / v[0]
    T = 12.0 / abs(lin.nu_minus)

    def fun(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        K = np.clip(y[0], 1e-300, None)
        q = np.clip(y[1], 1e-300, None)
        g = np.log(q) / params.installation.varphi - params.installation.delta
        tech = _partial_technology(params)
        R = tech.I * omega_Z(tech) * K ** (tech.I - 1.0)
        iota = (q - 1.0) / params.installation.varphi
        return np.vstack([g * K, (params.rates.rP_bar - g) * q - R + iota])

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([ya[0] - K_node, float(w @ (yb - y_star))])

    t = np.linspace(0.0, T, 61)
    guess = y_star[:, None] + eps * np.exp(lin.nu_minus * t)[None, :] * v[:, None]
    sol = solve_bvp(fun, bc, t, guess, tol=tolerances.bvp_tol, max_nodes=20000)
    terminal = np.array([sol.sol(T)[0], sol.sol(T)[1]]) - y_star
    return {
        "K_node": K_node,
        "q_node": float(sol.sol(0.0)[1]),
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": sol.message,
        "n_nodes": int(sol.x.size),
        "max_residual": float(np.max(sol.rms_residuals)) if sol.rms_residuals.size else 0.0,
        "horizon": T,
        "terminal_distance_to_rest_point": float(np.linalg.norm(terminal)),
        "terminal_unstable_component": float(w @ terminal),
    }


# --- assembled successor ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartialSuccessor:
    """The partial successor package: manifold, productive wealth, values, diagnostics.

    Interpolants are valid only on ``certified_domain``. Evaluating outside it raises
    rather than extrapolating, because an interpolant beyond the integrated range
    carries no evidence.
    """

    params: ModelParameters
    point: PartialStationaryPoint
    linearization: PartialLinearization
    certified_domain: tuple[float, float]
    H_anchor: float
    diagnostics: dict[str, Any]
    _q_interp: PchipInterpolator
    _H_interp: PchipInterpolator

    def _check_domain(self, K: float) -> float:
        K = require_positive("K", K)
        lo, hi = self.certified_domain
        if not lo <= K <= hi:
            raise DomainError(
                "K lies outside the certified partial-successor domain; the "
                "interpolants carry no evidence there",
                K=K,
                certified_domain=[lo, hi],
            )
        return K

    def q_P(self, K: float) -> float:
        """Selected stable-manifold price ``q_P(K)``."""
        return float(self._q_interp(self._check_domain(K)))

    def q_P_derivative(self, K: float) -> float:
        """``q_P'(K)``, used for the partial mark's payoff slope ``J_{P,K}``."""
        return float(self._q_interp.derivative()(self._check_domain(K)))

    def H_P(self, K: float) -> float:
        """Productive wealth by quadrature of ``q_P`` from the anchor."""
        return float(self._H_interp(self._check_domain(K)))

    def H_P_derivative(self, K: float) -> float:
        """``H_P'(K)``. The theory makes this identically ``q_P(K)``."""
        return float(self._H_interp.derivative()(self._check_domain(K)))

    def H_P_algebraic(self, K: float) -> float:
        """Independent route: ``r_P_bar H_P = Y_P - iota(q_P) K + q_P g(q_P) K``."""
        K = self._check_domain(K)
        q = self.q_P(K)
        tech = _partial_technology(self.params)
        return (
            task_output(K, tech)
            - installation_rate(q, self.params.installation) * K
            + q * capital_growth(q, self.params.installation) * K
        ) / self.params.rates.rP_bar

    def X_P(self, K: float, e: float) -> float:
        """``X_P = e + H_P(K)``."""
        X = require_finite("e", e) + self.H_P(K)
        if X <= 0.0:
            raise DomainError(
                "partial successor worker wealth X_P is nonpositive, so the log value is undefined",
                X_P=X,
                K=K,
                e=e,
            )
        return X

    def V_P(self, K: float, e: float) -> float:
        """``V_P = (1/rho) log(rho X_P) + (r_P_bar - rho)/rho^2``."""
        rho = self.params.preferences.rho
        return math.log(rho * self.X_P(K, e)) / rho + (self.params.rates.rP_bar - rho) / (rho * rho)

    def V_P_e(self, K: float, e: float) -> float:
        """``V_{P,e} = 1/(rho X_P)``."""
        return 1.0 / (self.params.preferences.rho * self.X_P(K, e))

    def V_P_K(self, K: float, e: float) -> float:
        """``V_{P,K} = q_P(K) V_{P,e}``, the successor envelope condition."""
        return self.q_P(K) * self.V_P_e(K, e)

    def C_P(self, K: float, e: float) -> float:
        """``C_P^W = rho X_P``."""
        return self.params.preferences.rho * self.X_P(K, e)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stationary_point": self.point.as_dict(),
            "linearization": self.linearization.as_dict(),
            "certified_domain": list(self.certified_domain),
            "H_anchor": self.H_anchor,
            "diagnostics": self.diagnostics,
        }


def solve_partial_successor(
    params: ModelParameters,
    interval: PartialCapitalInterval,
    tolerances: SolverTolerances,
    *,
    n_collocation_checks: int = 3,
) -> PartialSuccessor:
    """Build the partial successor over the declared capital interval.

    Steps, in order, each of which can refuse:

    1. solve the stationary endpoint and verify ``U_P > 0``;
    2. linearise and verify exactly one stable and one unstable root;
    3. trace both manifold branches by backward integration, and repeat at a smaller
       linearisation offset for the offset-size check;
    4. verify by forward integration that the traced branch really contracts to the
       rest point (the integration-direction check);
    5. build shape-preserving interpolants on the certified domain only;
    6. compute productive wealth by quadrature, and compare with the algebraic
       productive-wealth equation; and
    7. compare selected nodes with an independent two-point collocation solve.
    """
    tech = _partial_technology(params)
    point = partial_stationary_point(params)
    if not interval.contains(point.K_star):
        raise DomainError(
            "the declared capital interval does not contain the partial rest point, so "
            "the stable manifold cannot be anchored inside it",
            K_star=point.K_star,
            declared_interval=interval.as_dict(),
        )
    lin = partial_linearization(params, point)

    offset = tolerances.manifold_offset
    small = offset / tolerances.manifold_offset_ratio
    branches: dict[str, Any] = {}
    Ks: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    for sign, name in ((1.0, "upper"), (-1.0, "lower")):
        K_b, q_b, diag = _integrate_branch(
            params, point, lin, interval, tolerances, offset=offset, sign=sign
        )
        K_s, q_s, diag_small = _integrate_branch(
            params, point, lin, interval, tolerances, offset=small, sign=sign
        )
        # Offset-size check: the two traces must agree where their capital ranges overlap.
        lo = max(min(K_b.min(), K_b.max()), min(K_s.min(), K_s.max()))
        hi = min(max(K_b.min(), K_b.max()), max(K_s.min(), K_s.max()))
        if hi > lo:
            probe = np.linspace(lo, hi, 25)
            f_b = PchipInterpolator(*_sorted(K_b, q_b))
            f_s = PchipInterpolator(*_sorted(K_s, q_s))
            offset_spread = float(np.max(np.abs(f_b(probe) - f_s(probe))))
        else:
            offset_spread = math.nan
        direction = _direction_check(params, point, lin, float(K_b[-1]), float(q_b[-1]), tolerances)
        if not direction["contracts"]:
            raise BranchFailure(
                "the constructed branch does not contract to the rest point under "
                "forward time; the integration direction is wrong",
                branch=name,
                direction_check=direction,
            )
        branches[name] = {
            "integration": diag,
            "integration_small_offset": diag_small,
            "offset_size_spread": offset_spread,
            "direction_check": direction,
        }
        Ks.append(K_b)
        qs.append(q_b)

    K_all = np.concatenate(Ks)
    q_all = np.concatenate(qs)
    K_sorted, q_sorted = _sorted(K_all, q_all)
    K_min, K_max = float(K_sorted[0]), float(K_sorted[-1])
    certified = (max(K_min, interval.K_lo), min(K_max, interval.K_hi))
    if not certified[0] < certified[1]:
        raise BranchFailure(
            "the traced manifold does not cover a nonempty part of the declared capital interval",
            traced_range=[K_min, K_max],
            declared_interval=interval.as_dict(),
        )

    # Interpolate through the traced points themselves. Resampling onto a uniform grid
    # first would discard accuracy for no benefit, and the traced distribution - dense
    # near the rest point, where the trace moves slowly - is well suited to a
    # shape-preserving interpolant.
    inside = (K_sorted >= certified[0]) & (K_sorted <= certified[1])
    nodes, q_nodes = K_sorted[inside], q_sorted[inside]
    if nodes.size < 8:
        raise BranchFailure(
            "too few traced manifold points inside the certified domain to interpolate",
            n_points=int(nodes.size),
            certified_domain=list(certified),
        )
    q_interp = PchipInterpolator(nodes, q_nodes)

    # Productive wealth by quadrature from the anchor value at the rest point, where
    # g(q_delta) = 0 so the algebraic equation collapses to Y_P - iota_delta K.
    H_anchor = (
        task_output(point.K_star, tech) - point.iota_delta * point.K_star
    ) / params.rates.rP_bar
    antider = q_interp.antiderivative()
    H_nodes = H_anchor + (antider(nodes) - antider(point.K_star))
    H_interp = PchipInterpolator(nodes, H_nodes)
    n_interp_nodes = int(nodes.size)

    successor = PartialSuccessor(
        params=params,
        point=point,
        linearization=lin,
        certified_domain=certified,
        H_anchor=H_anchor,
        diagnostics={},
        _q_interp=q_interp,
        _H_interp=H_interp,
    )

    # --- cross-route diagnostics ---
    probe = np.linspace(
        certified[0] + 0.02 * (certified[1] - certified[0]),
        certified[1] - 0.02 * (certified[1] - certified[0]),
        21,
    )
    wealth_route_gap = max(
        abs(successor.H_P(float(K)) - successor.H_P_algebraic(float(K))) for K in probe
    )
    derivative_identity_gap = max(
        abs(successor.H_P_derivative(float(K)) - successor.q_P(float(K))) for K in probe
    )
    manifold_invariance = max(
        abs(
            successor.q_P_derivative(float(K))
            * capital_growth(successor.q_P(float(K)), params.installation)
            * float(K)
            - partial_field(float(K), successor.q_P(float(K)), params)[1]
        )
        for K in probe
    )
    slope_gap = abs(successor.q_P_derivative(point.K_star) - lin.manifold_slope)

    collocation = []
    if n_collocation_checks > 0:
        span = certified[1] - certified[0]
        for frac in np.linspace(0.2, 0.8, n_collocation_checks):
            K_node = float(certified[0] + frac * span)
            rec = _collocation_price_at(params, point, lin, K_node, tolerances)
            rec["q_node_ivp"] = successor.q_P(K_node)
            rec["ivp_bvp_difference"] = rec["q_node"] - rec["q_node_ivp"]
            collocation.append(rec)
    max_ivp_bvp = max(abs(r["ivp_bvp_difference"]) for r in collocation) if collocation else None

    diagnostics: dict[str, Any] = {
        "branches": branches,
        "certified_domain": list(certified),
        "declared_interval": interval.as_dict(),
        "n_interpolation_nodes": n_interp_nodes,
        "H_anchor": H_anchor,
        "wealth_route_max_gap": wealth_route_gap,
        "derivative_identity_max_gap": derivative_identity_gap,
        "manifold_invariance_max_residual": manifold_invariance,
        "rest_point_slope_gap": slope_gap,
        "collocation_checks": collocation,
        "max_ivp_bvp_difference": max_ivp_bvp,
        "max_offset_size_spread": max(
            (
                b["offset_size_spread"]
                for b in branches.values()
                if not math.isnan(b["offset_size_spread"])
            ),
            default=math.nan,
        ),
    }
    return PartialSuccessor(
        params=params,
        point=point,
        linearization=lin,
        certified_domain=certified,
        H_anchor=H_anchor,
        diagnostics=diagnostics,
        _q_interp=q_interp,
        _H_interp=H_interp,
    )


def _sorted(K: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort a traced branch by capital and drop duplicate abscissae."""
    order = np.argsort(K)
    K_s, q_s = K[order], q[order]
    keep = np.concatenate(([True], np.diff(K_s) > 0))
    return K_s[keep], q_s[keep]
