"""Assemble the successor services and the equation core at one pre-arrival state.

This is the thin layer that ties the pieces together for the CLI and the tests. It owns
no equations of its own: everything here is a call into
:mod:`~ak_partial_ramsey.successors`, :mod:`~ak_partial_ramsey.exposure`,
:mod:`~ak_partial_ramsey.canonical`, or :mod:`~ak_partial_ramsey.recovery`.

The independent evaluator deliberately does not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import (
    investment_wedge,
    mark_geometry,
    risk_neutral_drag,
    total_valuation_residual,
)
from .exposure import (
    Mark,
    PrivatePortfolioSolution,
    PublicExposureSolution,
    solve_private_portfolio,
    solve_public_exposure,
)
from .params import ModelParameters
from .recovery import (
    ImplementationMargins,
    OwnerBlock,
    implementation_margins,
    recover_foreign_residual,
    recover_owner,
    recover_positions,
    recover_source_tax,
    recover_transfer,
)
from .successors.ak import AkSuccessor
from .successors.partial import PartialSuccessor
from .tolerances import SolverTolerances
from .validation import require_positive

__all__ = ["MarkSet", "StateEvaluation", "build_marks", "evaluate_state"]


@dataclass(frozen=True, slots=True)
class MarkSet:
    """The event marks at one pre-arrival state, with their successor price slopes."""

    marks: tuple[Mark, ...]
    successor_prices: dict[str, float]
    successor_price_slopes: dict[str, float]
    successor_wealth_at_zero_exposure: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "marks": [m.as_dict() for m in self.marks],
            "successor_prices": self.successor_prices,
            "successor_price_slopes": self.successor_price_slopes,
            "successor_wealth_at_zero_exposure": self.successor_wealth_at_zero_exposure,
        }


def build_marks(
    params: ModelParameters,
    ak: AkSuccessor,
    partial: PartialSuccessor | None,
    K: float,
    e: float,
    q: float,
) -> MarkSet:
    """Assemble the ``P`` and ``F`` marks at ``(K, e, q)``.

    The ``P`` mark is included whenever a partial successor is present, even when its
    intensities are zero. Under the single-AK support restriction it is then an
    *inactive* mark: it contributes nothing and constrains nothing, and no quantity is
    divided by ``p_P`` or ``lambda_P_star`` to discover that.
    """
    K = require_positive("K", K)
    q = require_positive("q", q)
    i = params.intensities

    marks: list[Mark] = []
    prices: dict[str, float] = {}
    slopes: dict[str, float] = {}
    h: dict[str, float] = {}

    if partial is not None:
        q_P = partial.q_P(K)
        H_P = partial.H_P(K)
        prices["P"] = q_P
        slopes["P"] = partial.q_P_derivative(K)
        h["P"] = e + H_P
        marks.append(
            Mark(
                label="P",
                lambda_physical=i.lambda_P,
                lambda_star=i.lambda_P_star,
                J=(q_P - q) / q,
                h=h["P"],
            )
        )

    q_F = ak.q_F
    prices["F"] = q_F
    slopes["F"] = 0.0  # the selected AK price is constant, so J_{F,K} = 0
    h["F"] = e + ak.H_F(K)
    marks.append(
        Mark(
            label="F",
            lambda_physical=i.lambda_F,
            lambda_star=i.lambda_F_star,
            J=(q_F - q) / q,
            h=h["F"],
        )
    )
    return MarkSet(tuple(marks), prices, slopes, h)


@dataclass(frozen=True, slots=True)
class StateEvaluation:
    """Every equation-core object at one pre-arrival state."""

    K: float
    e: float
    q: float
    C: float
    mark_set: MarkSet
    exposure: PublicExposureSolution
    portfolio: PrivatePortfolioSolution
    Lambda: float
    U: float
    D: float
    geometry: Any
    positions: dict[str, float]
    transfer: float
    tau_0: float
    owner: OwnerBlock
    foreign: dict[str, float]
    margins: ImplementationMargins

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": {"K": self.K, "e": self.e, "q": self.q, "C": self.C},
            "marks": self.mark_set.as_dict(),
            "exposure": self.exposure.as_dict(),
            "private_portfolio": self.portfolio.as_dict(),
            "geometry": self.geometry.as_dict(),
            "Lambda": self.Lambda,
            "total_valuation_residual_U": self.U,
            "investment_wedge_D": self.D,
            "positions": self.positions,
            "transfer_T": self.transfer,
            "recovered_tau_0": self.tau_0,
            "owner": self.owner.as_dict(),
            "foreign": self.foreign,
            "implementation_margins": self.margins.as_dict(),
        }


def evaluate_state(
    params: ModelParameters,
    ak: AkSuccessor,
    partial: PartialSuccessor | None,
    *,
    K: float,
    e: float,
    q: float,
    C: float,
    a: float,
    q_dot: float,
    tolerances: SolverTolerances,
) -> StateEvaluation:
    """Solve the algebraic control block and recover every decentralized object.

    ``q_dot`` is supplied by the caller rather than derived: the pre-arrival price path
    is determined by the transition solve, which is block N4 and is out of scope here.
    The recovered tax is therefore conditional on the ``q_dot`` supplied, and is
    reported as such.
    """
    mark_set = build_marks(params, ak, partial, K, e, q)
    marks = mark_set.marks

    exposure = solve_public_exposure(marks, C, params.preferences.rho, tolerances)
    portfolio = solve_private_portfolio(marks, tolerances)
    psi = exposure.psi

    Lambda = risk_neutral_drag(marks)
    U = total_valuation_residual(marks, psi, C, params.preferences.rho)
    D = investment_wedge(psi, U, K, params.installation.varphi)
    geometry = mark_geometry(marks, q, mark_set.successor_price_slopes)

    positions = recover_positions(e, psi, q, K)
    transfer = recover_transfer(C, K, params)
    tau_0 = recover_source_tax(K, q, q_dot, Lambda, params)
    owner = recover_owner(a, portfolio.pi, Lambda, params)
    foreign = recover_foreign_residual(
        q, K, positions["Theta"], owner.vartheta, positions["B"], owner.d_O
    )
    margins = implementation_margins(
        B=positions["B"],
        T=transfer,
        tau=tau_0,
        a=a,
        C_W=C,
        q=q,
        successor_wealth=exposure.successor_wealth,
        owner_event_solvency=portfolio.event_solvency,
        params=params,
    )
    return StateEvaluation(
        K=K,
        e=e,
        q=q,
        C=C,
        mark_set=mark_set,
        exposure=exposure,
        portfolio=portfolio,
        Lambda=Lambda,
        U=U,
        D=D,
        geometry=geometry,
        positions=positions,
        transfer=transfer,
        tau_0=tau_0,
        owner=owner,
        foreign=foreign,
        margins=margins,
    )
