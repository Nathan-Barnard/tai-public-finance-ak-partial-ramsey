"""Finiteness and domain guards.

Every public numerical entry point in this package passes its inputs through these
guards first. They raise rather than return a sentinel, so a violated domain cannot be
propagated as a NaN or silently replaced by a clipped value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .errors import DomainError, NonFiniteInputError, RankFailure

__all__ = [
    "require_finite",
    "require_finite_array",
    "require_finite_mapping",
    "require_nonnegative",
    "require_nonzero_payoff_vector",
    "require_ordered",
    "require_positive",
    "require_strictly_between",
]


def require_finite(name: str, value: Any) -> float:
    """Return ``value`` as a float, refusing NaN, infinity, and non-numeric input."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise NonFiniteInputError(
            f"{name} is not a real number", name=name, value=repr(value)
        ) from exc
    if not math.isfinite(out):
        raise NonFiniteInputError(f"{name} is not finite (got {out!r})", name=name, value=out)
    return out


def require_finite_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    """Apply :func:`require_finite` to every entry of a mapping."""
    return {key: require_finite(key, val) for key, val in values.items()}


def require_finite_array(name: str, values: Any) -> np.ndarray:
    """Return ``values`` as a float array, refusing any non-finite entry."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        raise NonFiniteInputError(
            f"{name} contains {len(bad)} non-finite entr{'y' if len(bad) == 1 else 'ies'}",
            name=name,
            first_bad_index=[int(i) for i in bad[0]] if len(bad) else None,
            size=int(arr.size),
        )
    return arr


def require_positive(name: str, value: Any) -> float:
    """Return ``value`` after refusing non-finite input and any value ``<= 0``."""
    out = require_finite(name, value)
    if out <= 0.0:
        raise DomainError(
            f"{name} must be strictly positive (got {out!r})",
            name=name,
            value=out,
            required="> 0",
        )
    return out


def require_nonnegative(name: str, value: Any) -> float:
    """Return ``value`` after refusing non-finite input and any value ``< 0``."""
    out = require_finite(name, value)
    if out < 0.0:
        raise DomainError(
            f"{name} must be nonnegative (got {out!r})",
            name=name,
            value=out,
            required=">= 0",
        )
    return out


def require_strictly_between(name: str, value: Any, lo: float, hi: float) -> float:
    """Return ``value`` after refusing anything outside the open interval ``(lo, hi)``."""
    out = require_finite(name, value)
    if not (lo < out < hi):
        raise DomainError(
            f"{name} must lie strictly inside ({lo}, {hi}) (got {out!r})",
            name=name,
            value=out,
            lower=lo,
            upper=hi,
            required=f"in ({lo}, {hi})",
        )
    return out


def require_ordered(lower_name: str, lower: float, upper_name: str, upper: float) -> None:
    """Refuse a pair that is not strictly increasing."""
    if not lower < upper:
        raise DomainError(
            f"{lower_name} must be strictly less than {upper_name} (got {lower!r} and {upper!r})",
            lower_name=lower_name,
            lower=lower,
            upper_name=upper_name,
            upper=upper,
        )


def require_nonzero_payoff_vector(name: str, payoffs: Iterable[float]) -> tuple[float, ...]:
    """Refuse identification when the entire event payoff vector vanishes.

    A zero payoff vector is a rank failure, not a conditioning problem: no finite
    exposure changes successor wealth under any mark, so the portfolio first-order
    condition carries no information. Refused rather than regularised.
    """
    values = tuple(require_finite(f"{name}[{i}]", p) for i, p in enumerate(payoffs))
    if all(p == 0.0 for p in values):
        raise RankFailure(
            f"{name} is the zero vector; the event exposure is not identified",
            name=name,
            payoffs=list(values),
            rank=0,
        )
    return values
