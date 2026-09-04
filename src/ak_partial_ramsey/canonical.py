"""Pre-arrival state, costate, and control equations.

These are the equations of the four-dimensional canonical system for
``z = (K, e, mu_K, mu_e)`` with algebraic controls ``v = (C, psi, q)``. This module
supplies them as *equations*; solving the transition boundary-value problem is block N4
and is deliberately not implemented here.

The two-mark system is the general one. The exact single-AK system is its algebraic
support restriction at ``p_P = lambda_P_star = 0``, not a separate model and not an
``I_P`` limit: with those intensities zero the ``P`` mark is inactive, ``u_P = 0``
identically, the exposure condition forces ``u_F = 0``, and therefore ``U = D = 0``. The
saving, investment, capital-costate, tax-recovery, and portfolio equations then collapse
term by term onto the single-mark system. :func:`single_ak_*` below writes that system
out separately so the collapse can be tested rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import RankFailure
from .exposure import Mark, fiscal_valuation_residual
from .params import ModelParameters
from .primitives import (
    capital_growth,
    capital_growth_derivative,
    installation_rate,
    installation_rate_derivative,
    task_rental,
)
from .validation import require_finite, require_positive

__all__ = [
    "MarkGeometry",
    "consumption_growth",
    "control_map_residual",
    "costate_K_dot",
    "costate_e_dot",
    "investment_wedge",
    "mark_geometry",
    "risk_neutral_drag",
    "single_ak_consumption_growth",
    "single_ak_exposure_closed_form",
    "single_ak_productive_wealth_rhs",
    "single_ak_q_dot",
    "state_K_dot",
    "state_e_dot",
    "tax_identity_residual",
    "total_valuation_residual",
]


@dataclass(frozen=True, slots=True)
class MarkGeometry:
    """Payoff-vector geometry at a fixed pre-arrival state."""

    J: dict[str, float]
    J_q: dict[str, float]
    J_K: dict[str, float]
    Lambda: float

    def as_dict(self) -> dict[str, Any]:
        return {"J": self.J, "J_q": self.J_q, "J_K": self.J_K, "Lambda": self.Lambda}


def mark_geometry(
    marks: tuple[Mark, ...], q: float, successor_price_slopes: dict[str, float]
) -> MarkGeometry:
    """Assemble ``J_j``, ``J_{j,q} = -(1+J_j)/q``, ``J_{j,K} = q_j'(K)/q`` and ``Lambda``.

    ``Lambda = sum_j lambda_j_star J_j`` is the risk-neutral drag on the external-wealth
    law. ``q_j'(K)`` is zero for the AK successor and generally nonzero for the partial
    successor.
    """
    q = require_positive("q", q)
    active = tuple(m for m in marks if m.is_active)
    return MarkGeometry(
        J={m.label: m.J for m in active},
        J_q={m.label: -(1.0 + m.J) / q for m in active},
        J_K={m.label: successor_price_slopes.get(m.label, 0.0) / q for m in active},
        Lambda=sum(m.lambda_star * m.J for m in active),
    )


def risk_neutral_drag(marks: tuple[Mark, ...]) -> float:
    """``Lambda(K, q) = sum_j lambda_j_star J_j``."""
    return sum(m.lambda_star * m.J for m in marks if m.is_active)


def total_valuation_residual(marks: tuple[Mark, ...], psi: float, C: float, rho: float) -> float:
    """``U = sum_j u_j``.

    Distinct from the exposure condition ``sum_j u_j J_j = 0``: the condition sets the
    residual vector orthogonal to the payoff line, which does not make its sum vanish
    unless the marks coincide or one is absent.
    """
    return sum(fiscal_valuation_residual(m, psi, C, rho) for m in marks if m.is_active)


def investment_wedge(psi: float, U: float, K: float, varphi: float) -> float:
    """``D = mu_K - q mu_e = varphi psi U / K``.

    Zero exactly when the unspanned valuation residual ``U`` is zero or the exposure is,
    which is what makes the single-mark tax cancellation reappear under the support
    restriction.
    """
    K = require_positive("K", K)
    return varphi * require_finite("psi", psi) * require_finite("U", U) / K


def consumption_growth(params: ModelParameters, U: float, mu_e: float) -> float:
    """``C'/C = r_0_bar + lambda_Sigma_star - rho - lambda + U/mu_e``."""
    mu_e = require_positive("mu_e", mu_e)
    i = params.intensities
    return (
        params.rates.r0_bar
        + i.lambda_star_sum
        - params.preferences.rho
        - i.lambda_total
        + require_finite("U", U) / mu_e
    )


def state_K_dot(K: float, q: float, params: ModelParameters) -> float:
    """``K' = g(q) K``."""
    return capital_growth(q, params.installation) * require_positive("K", K)


def state_e_dot(
    K: float,
    e: float,
    q: float,
    C: float,
    psi: float,
    Lambda: float,
    params: ModelParameters,
) -> float:
    """``e' = r_0_bar e + Y_0(K) - iota(q) K - C - psi Lambda``."""
    from .primitives import task_output

    return (
        params.rates.r0_bar * require_finite("e", e)
        + task_output(K, params.pre_arrival_technology)
        - installation_rate(q, params.installation) * K
        - require_positive("C", C)
        - require_finite("psi", psi) * require_finite("Lambda", Lambda)
    )


def costate_e_dot(
    mu_e: float,
    successor_value_e: dict[str, float],
    marks: tuple[Mark, ...],
    params: ModelParameters,
) -> float:
    """``mu_e' = (rho + lambda - r_0_bar) mu_e - sum_j lambda_j V_{j,e}``."""
    mu_e = require_positive("mu_e", mu_e)
    total = sum(m.lambda_physical * successor_value_e[m.label] for m in marks if m.is_active)
    return (
        params.preferences.rho + params.intensities.lambda_total - params.rates.r0_bar
    ) * mu_e - total


def costate_K_dot(
    K: float,
    q: float,
    mu_K: float,
    mu_e: float,
    psi: float,
    successor_value_K: dict[str, float],
    valuation_residual: dict[str, float],
    J_K: dict[str, float],
    marks: tuple[Mark, ...],
    params: ModelParameters,
) -> float:
    """``mu_K' = (rho + lambda - g) mu_K - mu_e [R_0 - iota] - sum_j lambda_j V_{j,K}
    - psi sum_j u_j J_{j,K}``."""
    g = capital_growth(q, params.installation)
    R_0 = task_rental(K, params.pre_arrival_technology)
    iota = installation_rate(q, params.installation)
    envelope = sum(m.lambda_physical * successor_value_K[m.label] for m in marks if m.is_active)
    slope = sum(valuation_residual[m.label] * J_K[m.label] for m in marks if m.is_active)
    return (
        (params.preferences.rho + params.intensities.lambda_total - g) * mu_K
        - mu_e * (R_0 - iota)
        - envelope
        - require_finite("psi", psi) * slope
    )


def control_map_residual(
    C: float,
    psi: float,
    q: float,
    mu_K: float,
    mu_e: float,
    K: float,
    marks: tuple[Mark, ...],
    geometry: MarkGeometry,
    params: ModelParameters,
) -> tuple[float, float, float]:
    """``Gamma(z, v)``: the three interior control first-order conditions.

    Components, in order:

    1. ``1/C - mu_e``;
    2. ``sum_j u_j J_j``, the unmultiplied exposure condition; and
    3. ``mu_K g_q K - mu_e iota_q K + psi sum_j u_j J_{j,q}``, the investment condition.
    """
    rho = params.preferences.rho
    u = {m.label: fiscal_valuation_residual(m, psi, C, rho) for m in marks if m.is_active}
    exposure = sum(u[label] * geometry.J[label] for label in u)
    investment = (
        mu_K * capital_growth_derivative(q, params.installation) * K
        - mu_e * installation_rate_derivative(params.installation) * K
        + psi * sum(u[label] * geometry.J_q[label] for label in u)
    )
    return (1.0 / require_positive("C", C) - mu_e, exposure, investment)


def tax_identity_residual(
    tau_0: float,
    mu_e: float,
    R_0: float,
    D: float,
    D_dot: float,
    g: float,
    psi: float,
    valuation_residual: dict[str, float],
    J_K: dict[str, float],
    params: ModelParameters,
) -> float:
    """Residual of ``mu_e tau_0 R_0 = (rho + lambda - g) D - D' - psi sum_j u_j J_{j,K}``.

    The conditional exact identity that locates the unspanned fiscal wedge. It carries
    no sign result: ``D``, ``D'``, and the payoff-slope term have ambiguous sign, so a
    nonzero tax cannot be inferred from rank deficiency alone.
    """
    slope = sum(valuation_residual[k] * J_K[k] for k in valuation_residual)
    rhs = (params.preferences.rho + params.intensities.lambda_total - g) * D - D_dot - psi * slope
    return mu_e * tau_0 * R_0 - rhs


# --- exact single-AK support restriction ------------------------------------------


def single_ak_consumption_growth(params: ModelParameters) -> float:
    """``C'/C = r_0_bar + lambda_F_star - lambda - rho``  (P.9)."""
    return (
        params.rates.r0_bar
        + params.intensities.lambda_F_star
        - params.intensities.lambda_total
        - params.preferences.rho
    )


def single_ak_q_dot(K: float, q: float, q_F: float, params: ModelParameters) -> float:
    """``q' = [r_0_bar + lambda_F_star - g(q)] q - R_0(K) + iota(q) - lambda_F_star q_F``
    (P.10)."""
    return (
        (
            params.rates.r0_bar
            + params.intensities.lambda_F_star
            - capital_growth(q, params.installation)
        )
        * q
        - task_rental(K, params.pre_arrival_technology)
        + installation_rate(q, params.installation)
        - params.intensities.lambda_F_star * q_F
    )


def single_ak_productive_wealth_rhs(
    K: float, q: float, q_F: float, params: ModelParameters
) -> float:
    """``(r_0_bar + lambda_F_star) H_0 = Y_0 - iota(q) K + q g(q) K + lambda_F_star q_F K``
    (P.16), returned as ``H_0``."""
    from .primitives import task_output

    return (
        task_output(K, params.pre_arrival_technology)
        - installation_rate(q, params.installation) * K
        + q * capital_growth(q, params.installation) * K
        + params.intensities.lambda_F_star * q_F * K
    ) / (params.rates.r0_bar + params.intensities.lambda_F_star)


def single_ak_exposure_closed_form(
    X_0: float, e: float, K: float, q_F: float, J_F: float, params: ModelParameters
) -> float:
    """``psi = [(lambda/lambda_F_star) X_0 - e - q_F K] / J_F``  (P.20).

    This is the single-mark theory's own closed form and it *does* divide by ``J_F``, so
    it is guarded: a zero payoff is a rank failure, refused rather than regularised. It
    exists to cross-check the general unmultiplied root, which needs no such division;
    the two must agree wherever both are defined.
    """
    if J_F == 0.0:
        raise RankFailure(
            "the single-AK closed-form exposure divides by J_F, which is zero; the "
            "exposure is not identified",
            J_F=J_F,
        )
    i = params.intensities
    return ((i.lambda_total / i.lambda_F_star) * X_0 - e - q_F * K) / J_F
