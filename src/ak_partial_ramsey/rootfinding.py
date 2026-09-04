"""Scanning, bracketing, and bracketed scalar root solving.

CS011 forbids selecting the first root returned by a nonlinear solver. Everything here
is therefore bracket-based: a root is found only inside an interval whose endpoints have
opposite signs, and enumeration scans a declared interval rather than relying on a
starting guess.

The bracket width at convergence is retained as the root's own numerical uncertainty and
is reported alongside it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy.optimize import brentq

from .errors import BranchFailure, RootCoverageError
from .tolerances import SolverTolerances

__all__ = ["Bracket", "BracketedRoot", "march_to_bracket", "scan_brackets", "solve_in_bracket"]


@dataclass(frozen=True, slots=True)
class Bracket:
    """A sign-change interval for a continuous scalar function."""

    a: float
    b: float
    f_a: float
    f_b: float

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "f_a": self.f_a, "f_b": self.f_b}


@dataclass(frozen=True, slots=True)
class BracketedRoot:
    """A root together with the evidence that located it."""

    x: float
    residual: float
    bracket: Bracket
    #: Width of the bracket at convergence: this root's own numerical uncertainty.
    bracket_width: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "residual": self.residual,
            "bracket": self.bracket.as_dict(),
            "bracket_width": self.bracket_width,
        }


def scan_brackets(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    n_points: int,
    label: str = "function",
) -> tuple[list[Bracket], list[float], dict[str, Any]]:
    """Scan ``[lo, hi]`` on a uniform grid and return every sign-change bracket.

    Returns ``(brackets, exact_zeros, scan_diagnostics)``. Points where the scan itself
    lands exactly on a zero are reported separately so they are never lost between two
    same-signed neighbours.

    The scan diagnostics record the minimum of ``|f|`` over the grid and where it
    occurred. A small minimum with no sign change is the signature of a double root or a
    near-tangency: enumeration reports it rather than silently returning nothing, because
    "no root found" and "a tangency the scan could not bracket" are different results.
    """
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("scan interval must be finite")
    step = (hi - lo) / (n_points - 1)
    xs = [lo + i * step for i in range(n_points)]
    vals: list[float] = []
    for x in xs:
        try:
            vals.append(float(f(x)))
        except Exception:
            vals.append(math.nan)

    brackets: list[Bracket] = []
    exact_zeros: list[float] = []
    for i in range(n_points - 1):
        fa, fb = vals[i], vals[i + 1]
        if math.isnan(fa) or math.isnan(fb):
            continue
        if fa == 0.0:
            exact_zeros.append(xs[i])
            continue
        if fb == 0.0:
            continue  # picked up as fa on the next iteration, or appended below
        if (fa < 0.0) != (fb < 0.0):
            brackets.append(Bracket(xs[i], xs[i + 1], fa, fb))
    if vals and vals[-1] == 0.0:
        exact_zeros.append(xs[-1])

    finite = [(abs(v), x) for v, x in zip(vals, xs, strict=True) if not math.isnan(v)]
    min_abs, min_at = min(finite) if finite else (math.nan, math.nan)

    diagnostics = {
        "label": label,
        "interval": [lo, hi],
        "n_points": n_points,
        "step": step,
        "n_sign_change_brackets": len(brackets),
        "n_exact_zeros_on_grid": len(exact_zeros),
        "n_non_evaluable_points": sum(1 for v in vals if math.isnan(v)),
        "min_abs_value": min_abs,
        "min_abs_at": min_at,
    }
    return brackets, exact_zeros, diagnostics


def solve_in_bracket(
    f: Callable[[float], float],
    bracket: Bracket,
    tolerances: SolverTolerances,
    *,
    label: str = "function",
) -> BracketedRoot:
    """Solve ``f(x) = 0`` inside a sign-change bracket with Brent's method."""
    if (bracket.f_a < 0.0) == (bracket.f_b < 0.0):
        raise RootCoverageError(
            f"{label}: bracket endpoints do not straddle zero",
            bracket=bracket.as_dict(),
        )
    x, result = brentq(
        f,
        bracket.a,
        bracket.b,
        xtol=tolerances.root_xtol,
        rtol=tolerances.root_rtol,
        full_output=True,
    )
    if not result.converged:
        raise BranchFailure(
            f"{label}: bracketed solve did not converge",
            bracket=bracket.as_dict(),
            iterations=result.iterations,
        )
    # The converged bracket width bounds this root's numerical uncertainty.
    width = max(tolerances.root_xtol, abs(x) * tolerances.root_rtol)
    return BracketedRoot(x=x, residual=float(f(x)), bracket=bracket, bracket_width=width)


def march_to_bracket(
    f: Callable[[float], float],
    *,
    anchor: float,
    lower: float,
    upper: float,
    label: str = "function",
    max_steps: int = 200,
) -> Bracket:
    """Find a sign-change bracket for a strictly monotone decreasing ``f``.

    ``(lower, upper)`` is the open domain of ``f`` and may be infinite at either end.
    Because ``f`` is decreasing, the sign at ``anchor`` says which way the root lies, so
    this marches in exactly one direction: geometrically toward an infinite end, and
    geometrically closer to a finite open boundary.

    Raises :class:`~ak_partial_ramsey.errors.BranchFailure` if no sign change is reached
    within ``max_steps``, rather than returning a best guess.
    """
    f_anchor = float(f(anchor))
    if f_anchor == 0.0:
        return Bracket(anchor, anchor, 0.0, 0.0)

    go_right = f_anchor > 0.0  # decreasing: positive value means the root is to the right
    boundary = upper if go_right else lower
    x_prev, f_prev = anchor, f_anchor

    if math.isinf(boundary):
        step = max(1.0, abs(anchor))
        for _ in range(max_steps):
            x = x_prev + step if go_right else x_prev - step
            try:
                fx = float(f(x))
            except Exception:
                break
            if math.isfinite(fx) and (fx < 0.0) != (f_prev < 0.0):
                return (
                    Bracket(x_prev, x, f_prev, fx) if go_right else Bracket(x, x_prev, fx, f_prev)
                )
            if math.isfinite(fx):
                x_prev, f_prev = x, fx
            step *= 2.0
    else:
        gap = boundary - x_prev if go_right else x_prev - boundary
        for _ in range(max_steps):
            gap *= 0.5
            x = boundary - gap if go_right else boundary + gap
            try:
                fx = float(f(x))
            except Exception:
                continue
            if math.isfinite(fx) and (fx < 0.0) != (f_prev < 0.0):
                return (
                    Bracket(x_prev, x, f_prev, fx) if go_right else Bracket(x, x_prev, fx, f_prev)
                )
            if math.isfinite(fx):
                x_prev, f_prev = x, fx

    raise BranchFailure(
        f"{label}: no sign change found on the admissible domain; "
        "the first-order condition has no interior root there",
        anchor=anchor,
        value_at_anchor=f_anchor,
        domain=[lower, upper],
        searched_toward="upper" if go_right else "lower",
        max_steps=max_steps,
    )
