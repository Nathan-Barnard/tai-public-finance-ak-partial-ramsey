"""Government, private-owner, and foreign-residual recovery.

Given an accepted reduced path, every decentralized object is recovered algebraically:

``iota = (q-1)/varphi``,  ``F = e + qK``,  ``Theta = psi + qK``,  ``B = psi - e``,
``T = C^W - W_0(K)``,

the source tax from world equity pricing, the owner block from the private portfolio
condition, and the foreign positions as the residual that clears both markets.

Every recovered inequality is returned as a **margin**, never enforced. A negative
margin means the candidate fails implementation and must be rejected or re-solved on the
corresponding constrained branch; it is never clipped back into the interior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .params import ModelParameters
from .primitives import (
    capital_growth,
    installation_domain_margin,
    installation_rate,
    task_rental,
    task_wage,
)
from .validation import require_finite, require_positive

__all__ = [
    "ImplementationMargins",
    "OwnerBlock",
    "implementation_margins",
    "recover_ak_source_tax",
    "recover_foreign_residual",
    "recover_owner",
    "recover_positions",
    "recover_source_tax",
    "recover_transfer",
]


def recover_positions(e: float, psi: float, q: float, K: float) -> dict[str, float]:
    """``F = e + qK``, ``Theta = psi + qK``, ``B = psi - e``."""
    e = require_finite("e", e)
    psi = require_finite("psi", psi)
    q = require_positive("q", q)
    K = require_positive("K", K)
    return {"F": e + q * K, "Theta": psi + q * K, "B": psi - e}


def recover_transfer(C_W: float, K: float, params: ModelParameters) -> float:
    """``T = C^W - W_0(K)``. Workers hold no assets, so ``C^W = W_0 + T``."""
    return require_positive("C_W", C_W) - task_wage(K, params.pre_arrival_technology)


def recover_source_tax(
    K: float, q: float, q_dot: float, Lambda: float, params: ModelParameters
) -> float:
    """``tau_0 = 1 - [(r_0_bar - g) q + iota - q' - q Lambda] / R_0(K)``.

    The two-mark form. ``R_0(K) > 0`` strictly for ``0 < I_0 < 1`` and ``K > 0``, so
    this division is safe; it is not a division by an event payoff.
    """
    g = capital_growth(q, params.installation)
    iota = installation_rate(q, params.installation)
    R_0 = task_rental(K, params.pre_arrival_technology)
    numerator = (
        (params.rates.r0_bar - g) * q
        + iota
        - require_finite("q_dot", q_dot)
        - q * require_finite("Lambda", Lambda)
    )
    return 1.0 - numerator / R_0


def recover_ak_source_tax(
    K: float, q: float, q_dot: float, q_F: float, params: ModelParameters
) -> float:
    """``tau_0 = 1 - [r_0_bar q + iota - q' - q g - lambda_F_star (q_F - q)] / R_0(K)``.

    The single-AK packet's form (G0.15). Algebraically identical to
    :func:`recover_source_tax` once ``Lambda`` collapses to ``lambda_F_star J_F``, but
    assembled independently; the two are compared in the evaluator.
    """
    g = capital_growth(q, params.installation)
    iota = installation_rate(q, params.installation)
    R_0 = task_rental(K, params.pre_arrival_technology)
    numerator = (
        params.rates.r0_bar * q
        + iota
        - require_finite("q_dot", q_dot)
        - q * g
        - params.intensities.lambda_F_star * (q_F - q)
    )
    return 1.0 - numerator / R_0


def recover_ak_successor_tax(q_F: float, params: ModelParameters) -> float:
    """``tau_F = 1 - [r_F_bar q_F + iota(q_F) - q_F g(q_F)] / A_bar``  (G0.16)."""
    return (
        1.0
        - (
            params.rates.rF_bar * q_F
            + installation_rate(q_F, params.installation)
            - q_F * capital_growth(q_F, params.installation)
        )
        / params.ak_technology.A_bar
    )


@dataclass(frozen=True, slots=True)
class OwnerBlock:
    """The domestic-owner block recovered from the private portfolio condition."""

    a: float
    pi: float
    C_O: float
    a_dot: float
    vartheta: float
    d_O: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "pi": self.pi,
            "C_O": self.C_O,
            "a_dot": self.a_dot,
            "vartheta": self.vartheta,
            "d_O": self.d_O,
        }


def recover_owner(a: float, pi: float, Lambda: float, params: ModelParameters) -> OwnerBlock:
    """``C^O = rho a``, ``a' = (r_0_bar - rho - pi Lambda) a``, ``vartheta = pi a``."""
    a = require_positive("a", a)
    pi = require_finite("pi", pi)
    Lambda = require_finite("Lambda", Lambda)
    rho = params.preferences.rho
    vartheta = pi * a
    return OwnerBlock(
        a=a,
        pi=pi,
        C_O=rho * a,
        a_dot=(params.rates.r0_bar - rho - pi * Lambda) * a,
        vartheta=vartheta,
        d_O=a - vartheta,
    )


def recover_foreign_residual(
    q: float, K: float, Theta: float, vartheta: float, B: float, d_O: float
) -> dict[str, float]:
    """``Theta_for = qK - Theta - vartheta``, ``D_for = B - d_O``.

    These clear both traded markets exactly by construction; the clearing residuals are
    returned so that the identity is checked rather than assumed.
    """
    q = require_positive("q", q)
    K = require_positive("K", K)
    Theta_for = q * K - Theta - vartheta
    D_for = B - d_O
    return {
        "Theta_for": Theta_for,
        "D_for": D_for,
        "equity_clearing_residual": Theta + vartheta + Theta_for - q * K,
        "safe_clearing_residual": -B + d_O + D_for,
        "foreign_net_claim": Theta_for + D_for,
    }


@dataclass(frozen=True, slots=True)
class ImplementationMargins:
    """Distances to every maintained smooth-branch inequality.

    A margin is a distance, not a constraint. Nothing here clips: a nonpositive margin
    means the candidate is not implementable on the smooth branch and must be rejected
    or re-solved with the corresponding multiplier retained.
    """

    debt_B: float
    transfer_T: float
    tax_lower: float
    tax_upper: float
    owner_wealth: float
    worker_consumption: float
    installation_domain: float
    successor_wealth: dict[str, float]
    owner_event_solvency: dict[str, float]

    @property
    def smallest(self) -> tuple[str, float]:
        """The binding margin: its name and value."""
        items: list[tuple[str, float]] = [
            ("debt_B", self.debt_B),
            ("transfer_T", self.transfer_T),
            ("tax_lower", self.tax_lower),
            ("tax_upper", self.tax_upper),
            ("owner_wealth", self.owner_wealth),
            ("worker_consumption", self.worker_consumption),
            ("installation_domain", self.installation_domain),
        ]
        items += [(f"successor_wealth[{k}]", v) for k, v in self.successor_wealth.items()]
        items += [(f"owner_event_solvency[{k}]", v) for k, v in self.owner_event_solvency.items()]
        return min(items, key=lambda kv: kv[1])

    @property
    def all_strict(self) -> bool:
        return self.smallest[1] > 0.0

    def as_dict(self) -> dict[str, Any]:
        name, value = self.smallest
        return {
            "debt_B": self.debt_B,
            "transfer_T": self.transfer_T,
            "tax_lower": self.tax_lower,
            "tax_upper": self.tax_upper,
            "owner_wealth": self.owner_wealth,
            "worker_consumption": self.worker_consumption,
            "installation_domain": self.installation_domain,
            "successor_wealth": self.successor_wealth,
            "owner_event_solvency": self.owner_event_solvency,
            "smallest_margin_name": name,
            "smallest_margin_value": value,
            "all_strict": self.all_strict,
        }


def implementation_margins(
    *,
    B: float,
    T: float,
    tau: float,
    a: float,
    C_W: float,
    q: float,
    successor_wealth: dict[str, float],
    owner_event_solvency: dict[str, float],
    params: ModelParameters,
) -> ImplementationMargins:
    """Assemble every maintained inequality as a signed distance from its boundary."""
    return ImplementationMargins(
        debt_B=require_finite("B", B),
        transfer_T=require_finite("T", T),
        tax_lower=require_finite("tau", tau),
        tax_upper=1.0 - tau,
        owner_wealth=require_finite("a", a),
        worker_consumption=require_finite("C_W", C_W),
        installation_domain=installation_domain_margin(q, params.installation),
        successor_wealth=dict(successor_wealth),
        owner_event_solvency=dict(owner_event_solvency),
    )
