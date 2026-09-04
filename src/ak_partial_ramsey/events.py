"""Event maps and normalized-to-level balance-sheet transformations.

Coordinates, reconciled across both theory packets:

``e = F - q K``     public external-wealth coordinate
``psi = Theta - q K``  public installed-equity exposure coordinate
``B = psi - e``     gross public safe debt

The event is totally inaccessible. Physical capital is continuous across it, and the
predictable pre-event equity position realises its gain at the selected successor price
*before* any rebalancing. In level coordinates that is ``F_j^+ = F + Theta * J_j``; in
normalized coordinates it is ``e_j^+ = e + psi * J_j``. These are the same statement,
and :func:`round_trip_diagnostics` checks that they agree numerically rather than
assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .validation import require_finite, require_positive

__all__ = [
    "RoundTrip",
    "event_level",
    "event_normalized",
    "jump_payoff",
    "jump_payoff_dK",
    "jump_payoff_dq",
    "level_from_normalized",
    "national_event_jump",
    "national_net_foreign_assets",
    "normalized_from_level",
    "normalized_successor_wealth_coordinate",
    "owner_event",
    "round_trip_diagnostics",
    "successor_price_from_jump",
]


# --- endogenous event payoffs ----------------------------------------------------


def jump_payoff(q_successor: float, q: float) -> float:
    """``J_j(K, q) = (q_j(K) - q)/q``, the installed-equity total-gain jump.

    ``q`` is the pre-arrival price and is strictly positive, so this division is always
    safe. Division by ``J_j`` itself is what this package never does.
    """
    q = require_positive("q", q)
    q_successor = require_positive("q_successor", q_successor)
    return (q_successor - q) / q


def jump_payoff_dq(J: float, q: float) -> float:
    """``J_{j,q} = -(1 + J_j)/q``."""
    q = require_positive("q", q)
    J = require_finite("J", J)
    return -(1.0 + J) / q


def jump_payoff_dK(q_successor_derivative: float, q: float) -> float:
    """``J_{j,K} = q_j'(K)/q``.

    Zero for the AK successor, whose selected price is a constant. Generally nonzero
    for the partial successor, whose price varies along its stable manifold.
    """
    q = require_positive("q", q)
    return require_finite("q_successor_derivative", q_successor_derivative) / q


def successor_price_from_jump(J: float, q: float) -> float:
    """Invert :func:`jump_payoff`: ``q_j = q * (1 + J_j)``."""
    q = require_positive("q", q)
    return q * (1.0 + require_finite("J", J))


# --- normalized <-> level ---------------------------------------------------------


def normalized_from_level(F: float, Theta: float, q: float, K: float) -> dict[str, float]:
    """Map level positions ``(F, Theta)`` to normalized ``(e, psi, B)``."""
    F = require_finite("F", F)
    Theta = require_finite("Theta", Theta)
    q = require_positive("q", q)
    K = require_positive("K", K)
    e = F - q * K
    psi = Theta - q * K
    return {"e": e, "psi": psi, "B": psi - e}


def level_from_normalized(e: float, psi: float, q: float, K: float) -> dict[str, float]:
    """Map normalized ``(e, psi)`` to level positions ``(F, Theta, B)``."""
    e = require_finite("e", e)
    psi = require_finite("psi", psi)
    q = require_positive("q", q)
    K = require_positive("K", K)
    return {"F": e + q * K, "Theta": psi + q * K, "B": psi - e}


# --- event maps -------------------------------------------------------------------


def event_normalized(e: float, psi: float, J: float) -> float:
    """``e_j^+ = e + psi * J_j``."""
    e = require_finite("e", e)
    psi = require_finite("psi", psi)
    J = require_finite("J", J)
    return e + psi * J


def event_level(F: float, Theta: float, J: float) -> float:
    """``F_j^+ = F + Theta * J_j``, the gain realised before rebalancing."""
    F = require_finite("F", F)
    Theta = require_finite("Theta", Theta)
    J = require_finite("J", J)
    return F + Theta * J


def normalized_successor_wealth_coordinate(
    F_successor: float, q_successor: float, K: float
) -> float:
    """``e_j^+ = F_j^+ - q_j K``: convert the post-event level position back."""
    F_successor = require_finite("F_successor", F_successor)
    q_successor = require_positive("q_successor", q_successor)
    K = require_positive("K", K)
    return F_successor - q_successor * K


def owner_event(a: float, pi: float, J: float) -> float:
    """``a_j^+ = a * (1 + pi * J_j)``, domestic-owner wealth across the event."""
    a = require_finite("a", a)
    pi = require_finite("pi", pi)
    J = require_finite("J", J)
    return a * (1.0 + pi * J)


def national_net_foreign_assets(a: float, e: float) -> float:
    """``n = a + e``, domestic net foreign assets."""
    return require_finite("a", a) + require_finite("e", e)


def national_event_jump(psi: float, pi: float, a: float, J: float) -> float:
    """``n_j^+ - n = (psi + pi * a) * J_j``."""
    psi = require_finite("psi", psi)
    pi = require_finite("pi", pi)
    a = require_finite("a", a)
    J = require_finite("J", J)
    return (psi + pi * a) * J


# --- round trip -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """Residuals of the normalized/level coordinate and event round trips."""

    coordinate_residual_e: float
    coordinate_residual_psi: float
    event_residual: float
    national_residual: float

    @property
    def max_abs(self) -> float:
        return max(
            abs(self.coordinate_residual_e),
            abs(self.coordinate_residual_psi),
            abs(self.event_residual),
            abs(self.national_residual),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate_residual_e": self.coordinate_residual_e,
            "coordinate_residual_psi": self.coordinate_residual_psi,
            "event_residual": self.event_residual,
            "national_residual": self.national_residual,
            "max_abs": self.max_abs,
        }


def round_trip_diagnostics(
    *,
    e: float,
    psi: float,
    q: float,
    K: float,
    J: float,
    q_successor: float,
    a: float = 0.0,
    pi: float = 0.0,
) -> RoundTrip:
    """Check that the normalized and level representations agree.

    Three independent round trips:

    1. ``(e, psi) -> (F, Theta) -> (e, psi)`` reproduces the coordinates;
    2. the level event map ``F_j^+ = F + Theta J_j`` followed by
       ``e_j^+ = F_j^+ - q_j K`` reproduces the normalized event map
       ``e_j^+ = e + psi J_j``; and
    3. the national identity ``n = a + e`` with jump ``(psi + pi a) J_j`` is consistent
       with the public and owner event maps summed.

    A nonzero residual here is a coordinate or timing defect, which CS011 classes as a
    structural failure rather than a tolerance question.
    """
    level = level_from_normalized(e, psi, q, K)
    back = normalized_from_level(level["F"], level["Theta"], q, K)

    F_plus = event_level(level["F"], level["Theta"], J)
    e_plus_via_level = normalized_successor_wealth_coordinate(F_plus, q_successor, K)
    e_plus_direct = event_normalized(e, psi, J)

    a_plus = owner_event(a, pi, J)
    n = national_net_foreign_assets(a, e)
    n_plus_via_parts = national_net_foreign_assets(a_plus, e_plus_direct)
    n_plus_via_identity = n + national_event_jump(psi, pi, a, J)

    return RoundTrip(
        coordinate_residual_e=back["e"] - e,
        coordinate_residual_psi=back["psi"] - psi,
        event_residual=e_plus_via_level - e_plus_direct,
        national_residual=n_plus_via_parts - n_plus_via_identity,
    )
