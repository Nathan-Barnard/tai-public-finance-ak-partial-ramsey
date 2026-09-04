"""Unmultiplied public and private portfolio equations, and their root services.

The public exposure first-order condition is the projection condition

``sum_j u_j J_j = 0``,   ``u_j = lambda_j V_{j,e} - lambda_j_star mu_e
                              = lambda_j/(rho X_j) - lambda_j_star/C``,

with successor wealth ``X_j = h_j + psi J_j`` and ``h_j = e + H_j(K)``.

This module works exclusively with that **unmultiplied** form. Clearing the denominators
produces a quadratic in ``psi`` whose leading coefficient ``rho mu_e Lambda J_P J_F``
vanishes whenever either payoff or ``Lambda`` vanishes, so the quadratic route degenerates
in exactly the exceptional cases that matter. Dividing the condition mark-by-mark by
``J_j`` is worse still: it manufactures ``u_j = 0`` per mark, which is false whenever the
payoff vector has two distinct components.

Consequences carried through deliberately:

* a zero payoff vector is refused (rank failure), not regularised;
* a single zero payoff component is retained, and its mark stays unspanned with
  ``u_j != 0``;
* marks with zero physical and zero risk-neutral intensity - the single-AK support
  restriction ``p_P = lambda_P_star = 0`` - are inactive and constrain nothing, with no
  division by either quantity anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import BranchFailure, ConfigurationError, DomainError
from .rootfinding import BracketedRoot, march_to_bracket, solve_in_bracket
from .tolerances import SolverTolerances
from .validation import require_finite, require_nonzero_payoff_vector, require_positive

__all__ = [
    "Mark",
    "PrivatePortfolioSolution",
    "PublicExposureSolution",
    "fiscal_valuation_residual",
    "private_portfolio_residual",
    "public_exposure_residual",
    "public_exposure_residual_derivative",
    "solve_private_portfolio",
    "solve_public_exposure",
    "successor_wealth",
    "successor_wealth_interval",
]


@dataclass(frozen=True, slots=True)
class Mark:
    """One event mark's data at a fixed pre-arrival state.

    ``lambda_physical`` and ``lambda_star`` are the physical and risk-neutral
    intensities. They are distinct quantities and are never interchanged.
    """

    label: str
    lambda_physical: float
    lambda_star: float
    #: Endogenous installed-equity total-gain jump ``J_j``.
    J: float
    #: ``h_j = e + H_j(K)``: successor wealth at zero exposure.
    h: float

    def __post_init__(self) -> None:
        require_finite(f"lambda_physical[{self.label}]", self.lambda_physical)
        require_finite(f"lambda_star[{self.label}]", self.lambda_star)
        require_finite(f"J[{self.label}]", self.J)
        require_finite(f"h[{self.label}]", self.h)
        if self.lambda_physical < 0.0 or self.lambda_star < 0.0:
            raise DomainError(
                f"intensities for mark {self.label} must be nonnegative",
                label=self.label,
                lambda_physical=self.lambda_physical,
                lambda_star=self.lambda_star,
            )

    @property
    def is_active(self) -> bool:
        """False only when both intensities vanish, i.e. the mark is absent.

        An inactive mark contributes no value term to the Hamiltonian, so it imposes no
        positive-wealth domain restriction and no term in the exposure condition.
        """
        return self.lambda_physical > 0.0 or self.lambda_star > 0.0

    @property
    def is_spanned_direction(self) -> bool:
        """True when this mark's payoff is nonzero, so exposure can move its wealth."""
        return self.J != 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "lambda_physical": self.lambda_physical,
            "lambda_star": self.lambda_star,
            "J": self.J,
            "h": self.h,
            "is_active": self.is_active,
            "is_spanned_direction": self.is_spanned_direction,
        }


def successor_wealth(mark: Mark, psi: float) -> float:
    """``X_j(psi) = h_j + psi * J_j``."""
    return mark.h + require_finite("psi", psi) * mark.J


def _active(marks: tuple[Mark, ...]) -> tuple[Mark, ...]:
    active = tuple(m for m in marks if m.is_active)
    if not active:
        raise ConfigurationError(
            "no active mark: every mark has zero physical and zero risk-neutral intensity",
            marks=[m.as_dict() for m in marks],
        )
    return active


def successor_wealth_interval(marks: tuple[Mark, ...]) -> tuple[float, float]:
    """Return the open interval of ``psi`` on which every active mark has ``X_j > 0``.

    Marks with ``J_j = 0`` have constant successor wealth and so do not bound ``psi``;
    their positivity is instead a standing domain requirement, checked here because a
    nonpositive constant ``h_j`` makes the log value undefined at every exposure.

    Raises :class:`~ak_partial_ramsey.errors.DomainError` when the interval is empty.
    """
    active = _active(marks)
    lo, hi = -math.inf, math.inf
    for m in active:
        if m.J == 0.0:
            if m.h <= 0.0:
                raise DomainError(
                    f"mark {m.label} has zero payoff and nonpositive successor wealth, "
                    "so its log value is undefined at every exposure",
                    label=m.label,
                    h=m.h,
                    J=m.J,
                )
            continue
        bound = -m.h / m.J
        if m.J > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    if not lo < hi:
        raise DomainError(
            "the positive-successor-wealth interval is empty; no exposure keeps every "
            "active mark's successor wealth positive",
            interval=[lo, hi],
            marks=[m.as_dict() for m in active],
        )
    return lo, hi


def fiscal_valuation_residual(mark: Mark, psi: float, C: float, rho: float) -> float:
    """``u_j = lambda_j/(rho X_j) - lambda_j_star/C``.

    The fiscal-to-market state-price residual for one mark. Both terms are evaluated
    directly; neither intensity is ever divided by the other, and the payoff ``J_j``
    does not appear.

    An **inactive** mark - both intensities exactly zero, as under the single-AK support
    restriction ``p_P = lambda_P_star = 0`` - returns zero immediately, without forming
    its successor wealth. That is not a shortcut: such a mark contributes no term to the
    Hamiltonian, so its successor wealth is not a domain requirement and may legitimately
    be nonpositive at the exposure that the remaining active marks select. Forming
    ``X_j`` first and refusing on its sign would impose a constraint the model does not
    contain, and would make the support restriction unreachable.
    """
    C = require_positive("C", C)
    rho = require_positive("rho", rho)
    if not mark.is_active:
        return 0.0
    X = successor_wealth(mark, psi)
    if X <= 0.0:
        raise DomainError(
            f"successor wealth for mark {mark.label} is nonpositive",
            label=mark.label,
            X=X,
            psi=psi,
        )
    return mark.lambda_physical / (rho * X) - mark.lambda_star / C


def public_exposure_residual(psi: float, marks: tuple[Mark, ...], C: float, rho: float) -> float:
    """``sum_j u_j J_j``: the unmultiplied public exposure first-order condition.

    This equals the Hamiltonian derivative ``H_psi`` exactly. Inactive marks contribute
    nothing. No denominator involving a payoff appears.
    """
    return sum(fiscal_valuation_residual(m, psi, C, rho) * m.J for m in _active(marks))


def public_exposure_residual_derivative(
    psi: float, marks: tuple[Mark, ...], C: float, rho: float
) -> float:
    """``H_{psi psi} = -sum_j lambda_j J_j^2 / (rho X_j^2)``, strictly negative.

    Strict negativity on the positive-wealth interval is what makes the interior root
    unique; it is reported rather than assumed.
    """
    rho = require_positive("rho", rho)
    require_positive("C", C)
    total = 0.0
    for m in _active(marks):
        if m.J == 0.0:
            continue
        X = successor_wealth(m, psi)
        if X <= 0.0:
            raise DomainError(
                f"successor wealth for mark {m.label} is nonpositive",
                label=m.label,
                X=X,
                psi=psi,
            )
        total -= m.lambda_physical * m.J * m.J / (rho * X * X)
    return total


@dataclass(frozen=True, slots=True)
class PublicExposureSolution:
    """The interior public exposure root and the evidence for it."""

    psi: float
    root: BracketedRoot
    interval: tuple[float, float]
    successor_wealth: dict[str, float]
    fiscal_valuation_residual: dict[str, float]
    orthogonality_residual: float
    curvature: float
    #: Labels of active marks whose payoff is zero: unspanned, and deliberately retained.
    unspanned_marks: tuple[str, ...]
    payoff_vector: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "psi": self.psi,
            "root": self.root.as_dict(),
            "interval": list(self.interval),
            "successor_wealth": self.successor_wealth,
            "fiscal_valuation_residual": self.fiscal_valuation_residual,
            "orthogonality_residual": self.orthogonality_residual,
            "curvature": self.curvature,
            "unspanned_marks": list(self.unspanned_marks),
            "payoff_vector": self.payoff_vector,
        }


def solve_public_exposure(
    marks: tuple[Mark, ...],
    C: float,
    rho: float,
    tolerances: SolverTolerances,
) -> PublicExposureSolution:
    """Solve the unmultiplied exposure condition on the positive-wealth interval.

    Refuses a zero payoff vector outright: with ``J = 0`` the condition reads ``0 = 0``
    and identifies nothing, which is a rank failure and not a conditioning problem.

    Retains any single zero-payoff mark. Its ``u_j`` is generally nonzero at the root,
    which is the exact statement that the mark is unspanned.
    """
    active = _active(marks)
    require_nonzero_payoff_vector("public event payoff vector", tuple(m.J for m in active))
    lo, hi = successor_wealth_interval(marks)

    def residual(psi: float) -> float:
        return public_exposure_residual(psi, marks, C, rho)

    if math.isfinite(lo) and math.isfinite(hi):
        anchor = 0.5 * (lo + hi)
    elif math.isfinite(lo):
        anchor = lo + max(1.0, abs(lo))
    elif math.isfinite(hi):
        anchor = hi - max(1.0, abs(hi))
    else:
        anchor = 0.0

    bracket = march_to_bracket(
        residual,
        anchor=anchor,
        lower=lo,
        upper=hi,
        label="public exposure FOC",
    )
    if bracket.a == bracket.b:
        psi = bracket.a
        root = BracketedRoot(psi, 0.0, bracket, 0.0)
    else:
        root = solve_in_bracket(residual, bracket, tolerances, label="public exposure FOC")
        psi = root.x

    return PublicExposureSolution(
        psi=psi,
        root=root,
        interval=(lo, hi),
        successor_wealth={m.label: successor_wealth(m, psi) for m in active},
        fiscal_valuation_residual={
            m.label: fiscal_valuation_residual(m, psi, C, rho) for m in active
        },
        orthogonality_residual=residual(psi),
        curvature=public_exposure_residual_derivative(psi, marks, C, rho),
        unspanned_marks=tuple(m.label for m in active if m.J == 0.0),
        payoff_vector={m.label: m.J for m in active},
    )


# --- private portfolio ------------------------------------------------------------


def private_portfolio_residual(pi: float, marks: tuple[Mark, ...]) -> float:
    """``sum_j lambda_j J_j/(1 + pi J_j) - Lambda``, with ``Lambda = sum_j lambda_j_star J_j``.

    The unmultiplied domestic-owner log-portfolio condition. As with the public
    condition, the payoffs are never divided out.
    """
    pi = require_finite("pi", pi)
    total = 0.0
    for m in _active(marks):
        solvency = 1.0 + pi * m.J
        if solvency <= 0.0:
            raise DomainError(
                f"owner event solvency 1 + pi J fails for mark {m.label}",
                label=m.label,
                pi=pi,
                J=m.J,
                solvency=solvency,
            )
        total += m.lambda_physical * m.J / solvency - m.lambda_star * m.J
    return total


def _private_solvency_interval(marks: tuple[Mark, ...]) -> tuple[float, float]:
    """Open interval of ``pi`` on which ``1 + pi J_j > 0`` for every active mark.

    Always contains ``pi = 0``.
    """
    lo, hi = -math.inf, math.inf
    for m in _active(marks):
        if m.J == 0.0:
            continue
        bound = -1.0 / m.J
        if m.J > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    return lo, hi


@dataclass(frozen=True, slots=True)
class PrivatePortfolioSolution:
    """The interior owner equity share and the evidence for it."""

    pi: float
    root: BracketedRoot
    interval: tuple[float, float]
    event_solvency: dict[str, float]
    residual: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pi": self.pi,
            "root": self.root.as_dict(),
            "interval": list(self.interval),
            "event_solvency": self.event_solvency,
            "residual": self.residual,
        }


def solve_private_portfolio(
    marks: tuple[Mark, ...], tolerances: SolverTolerances
) -> PrivatePortfolioSolution:
    """Solve the owner's unmultiplied log-portfolio condition on its solvency interval."""
    active = _active(marks)
    require_nonzero_payoff_vector("private event payoff vector", tuple(m.J for m in active))
    lo, hi = _private_solvency_interval(marks)

    def residual(pi: float) -> float:
        return private_portfolio_residual(pi, marks)

    bracket = march_to_bracket(
        residual, anchor=0.0, lower=lo, upper=hi, label="private portfolio FOC"
    )
    if bracket.a == bracket.b:
        pi = bracket.a
        root = BracketedRoot(pi, 0.0, bracket, 0.0)
    else:
        root = solve_in_bracket(residual, bracket, tolerances, label="private portfolio FOC")
        pi = root.x

    solvency = {m.label: 1.0 + pi * m.J for m in active}
    if any(v <= 0.0 for v in solvency.values()):
        raise BranchFailure(
            "the interior owner portfolio root violates event solvency",
            pi=pi,
            event_solvency=solvency,
        )
    return PrivatePortfolioSolution(
        pi=pi,
        root=root,
        interval=(lo, hi),
        event_solvency=solvency,
        residual=residual(pi),
    )
