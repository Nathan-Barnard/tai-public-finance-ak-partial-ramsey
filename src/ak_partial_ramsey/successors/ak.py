"""Absorbing AK successor: root enumeration, branch selection, values and envelopes.

The stationary zero-tax world-user-cost equation is

``r_F_bar q_F = A_bar - iota(q_F) + q_F g(q_F)``    (AK.1)

which, multiplied through by ``varphi``, is the scalar equation

``f(q) = q log q - (1 + u) q + a = 0``,   ``a = 1 + varphi A_bar``,
``u = varphi (r_F_bar + delta)``.          (AK.2)

``f`` is strictly convex with its minimum at ``q_m = exp(u)``. So there are two positive
roots when ``a < exp(u)``, one double root when ``a = exp(u)``, and none when
``a > exp(u)``.

Selection is by the **strict productive-value transversality condition**
``r_F_bar > g(q_F)``, which holds exactly when ``q < q_m``. Only the lower root passes.
The upper root fails it - discounted installed-capital value grows rather than vanishing -
and the double root has a zero TVC margin. Both are retained with their rejection
reasons; neither is discarded silently, and the first root any solver happens to return
is never selected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from scipy.special import lambertw

from ..errors import BranchFailure, DomainError
from ..params import AkRootInterval, ModelParameters
from ..primitives import (
    capital_growth,
    installation_domain_margin,
    installation_rate,
)
from ..rootfinding import scan_brackets, solve_in_bracket
from ..tolerances import SolverTolerances
from ..validation import require_finite, require_positive

__all__ = [
    "AkRootCandidate",
    "AkSuccessor",
    "ak_lambert_roots",
    "ak_price_residual_level",
    "ak_price_residual_polynomial",
    "ak_scalar_coefficients",
    "enumerate_ak_roots",
    "solve_ak_successor",
]

#: Branch labels. These name the economic branch, not a position in a sorted list, so
#: they are stable under numerical refinement.
BRANCH_LOWER = "lower_strict_tvc"
BRANCH_UPPER = "upper_tvc_violating"
BRANCH_DOUBLE = "double_root_zero_tvc_margin"

REJECT_TVC = "productive_value_tvc_violated"
REJECT_TANGENCY = "zero_tvc_margin_at_double_root"
ACCEPT = "accepted_strict_productive_value_tvc"


def ak_scalar_coefficients(params: ModelParameters) -> dict[str, float]:
    """Return ``a``, ``u``, the minimiser ``q_m = exp(u)``, and the discriminant.

    The discriminant ``a - exp(u)`` decides the root count analytically: negative gives
    two roots, zero a double root, positive none. It is computed exactly rather than
    inferred from a scan, and the scan is then checked against it.
    """
    varphi = params.installation.varphi
    a = 1.0 + varphi * params.ak_technology.A_bar
    u = varphi * (params.rates.rF_bar + params.installation.delta)
    q_m = math.exp(u)
    return {"a": a, "u": u, "q_m": q_m, "discriminant": a - q_m}


def ak_price_residual_polynomial(q: float, params: ModelParameters) -> float:
    """``f(q) = q log q - (1 + u) q + a``: the ``varphi``-multiplied form (AK.2)."""
    q = require_positive("q", q)
    c = ak_scalar_coefficients(params)
    return q * math.log(q) - (1.0 + c["u"]) * q + c["a"]


def ak_price_residual_level(q: float, params: ModelParameters) -> float:
    """``r_F_bar q - [A_bar - iota(q) + q g(q)]``: the original level form (AK.1).

    Algebraically the same equation as :func:`ak_price_residual_polynomial` divided by
    ``varphi``, but assembled from the level primitives rather than from the collected
    coefficients. Used as the independent route in the evaluator.
    """
    q = require_positive("q", q)
    return params.rates.rF_bar * q - (
        params.ak_technology.A_bar
        - installation_rate(q, params.installation)
        + q * capital_growth(q, params.installation)
    )


def ak_lambert_roots(params: ModelParameters) -> dict[str, float | None]:
    """Closed-form roots via the Lambert ``W`` function.

    ``q_L = -a / W_{-1}(-a e^{-(1+u)})`` and ``q_H = -a / W_0(-a e^{-(1+u)})``.

    A third independent route, used to cross-check the scanned enumeration. Returns
    ``None`` for a branch that does not exist at these parameters.
    """
    c = ak_scalar_coefficients(params)
    a, u = c["a"], c["u"]
    arg = -a * math.exp(-(1.0 + u))
    out: dict[str, float | None] = {"q_lower": None, "q_upper": None}
    if arg < -math.exp(-1.0):
        return out  # below the branch point: no real root
    for key, branch in (("q_lower", -1), ("q_upper", 0)):
        w = complex(lambertw(arg, branch))
        if abs(w.imag) < 1e-12 and w.real != 0.0:
            out[key] = -a / w.real
    return out


@dataclass(frozen=True, slots=True)
class AkRootCandidate:
    """One enumerated positive root of the AK price equation, accepted or rejected."""

    q: float
    branch: str
    accepted: bool
    reason: str
    #: ``r_F_bar - g(q)``. Strictly positive is the strict productive-value TVC.
    tvc_margin: float
    residual_polynomial: float
    residual_level: float
    bracket_width: float
    iota: float
    g: float
    installation_margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "branch": self.branch,
            "accepted": self.accepted,
            "reason": self.reason,
            "tvc_margin": self.tvc_margin,
            "residual_polynomial": self.residual_polynomial,
            "residual_level": self.residual_level,
            "bracket_width": self.bracket_width,
            "iota": self.iota,
            "g": self.g,
            "installation_margin": self.installation_margin,
        }


def _classify(q: float, params: ModelParameters, bracket_width: float) -> AkRootCandidate:
    g = capital_growth(q, params.installation)
    tvc_margin = params.rates.rF_bar - g
    c = ak_scalar_coefficients(params)
    # Branch identity is the economic classification relative to the analytic minimiser
    # q_m, never a position in a sorted list. This keeps the label stable under
    # refinement of the scan or the bracket.
    rel = (q - c["q_m"]) / c["q_m"]
    if abs(rel) <= 1e-10:
        branch, accepted, reason = BRANCH_DOUBLE, False, REJECT_TANGENCY
    elif q < c["q_m"]:
        branch = BRANCH_LOWER
        accepted = tvc_margin > 0.0
        reason = ACCEPT if accepted else REJECT_TVC
    else:
        branch, accepted, reason = BRANCH_UPPER, False, REJECT_TVC
    return AkRootCandidate(
        q=q,
        branch=branch,
        accepted=accepted,
        reason=reason,
        tvc_margin=tvc_margin,
        residual_polynomial=ak_price_residual_polynomial(q, params),
        residual_level=ak_price_residual_level(q, params),
        bracket_width=bracket_width,
        iota=installation_rate(q, params.installation),
        g=g,
        installation_margin=installation_domain_margin(q, params.installation),
    )


def enumerate_ak_roots(
    params: ModelParameters,
    interval: AkRootInterval,
    tolerances: SolverTolerances,
) -> tuple[list[AkRootCandidate], dict[str, Any]]:
    """Enumerate **every** positive root in the declared interval, with accept/reject reasons.

    Scans the declared interval on a uniform grid, solves inside each sign-change
    bracket, and cross-checks the resulting count against both the analytic discriminant
    and the closed-form Lambert-``W`` roots. A disagreement between the scan and the
    discriminant is reported in the diagnostics rather than resolved silently.
    """

    def residual(q: float) -> float:
        return ak_price_residual_polynomial(q, params)

    brackets, exact_zeros, scan = scan_brackets(
        residual,
        interval.q_lo,
        interval.q_hi,
        n_points=tolerances.root_scan_points,
        label="AK price equation",
    )
    candidates: list[AkRootCandidate] = []
    for bracket in brackets:
        root = solve_in_bracket(residual, bracket, tolerances, label="AK price equation")
        candidates.append(_classify(root.x, params, root.bracket_width))
    for q in exact_zeros:
        candidates.append(_classify(q, params, 0.0))
    candidates.sort(key=lambda c: c.q)

    coeffs = ak_scalar_coefficients(params)
    lambert = ak_lambert_roots(params)
    expected = 2 if coeffs["discriminant"] < 0.0 else (1 if coeffs["discriminant"] == 0.0 else 0)
    in_interval = [q for q in lambert.values() if q is not None and interval.contains(q)]
    diagnostics: dict[str, Any] = {
        "scan": scan,
        "coefficients": coeffs,
        "lambert_roots": lambert,
        "n_roots_found": len(candidates),
        "n_roots_expected_from_discriminant": expected,
        "n_lambert_roots_inside_declared_interval": len(in_interval),
        "scan_agrees_with_discriminant": len(candidates) == min(expected, len(in_interval))
        or len(candidates) == expected,
        "max_lambert_disagreement": (
            max(
                (min(abs(c.q - q) for q in in_interval) for c in candidates),
                default=0.0,
            )
            if in_interval and candidates
            else None
        ),
        "declared_interval": interval.as_dict(),
    }
    return candidates, diagnostics


@dataclass(frozen=True, slots=True)
class AkSuccessor:
    """The selected AK successor package: price, values, envelopes, domains, diagnostics."""

    q_F: float
    branch: str
    iota_F: float
    g_F: float
    tvc_margin: float
    full_bgp_residual: float
    installation_margin: float
    recovered_tau_F: float
    residual_polynomial: float
    residual_level: float
    bracket_width: float
    candidates: tuple[AkRootCandidate, ...]
    diagnostics: dict[str, Any]
    params: ModelParameters

    # --- productive wealth, worker value, envelopes ---

    def H_F(self, K: float) -> float:
        """``H_F(K) = q_F K``: productive wealth on the selected AK branch."""
        return self.q_F * require_positive("K", K)

    def X_F(self, K: float, e: float) -> float:
        """``X_F = e + q_F K``: total worker wealth after the AK event."""
        X = require_finite("e", e) + self.H_F(K)
        if X <= 0.0:
            raise DomainError(
                "AK successor worker wealth X_F is nonpositive, so the log value is undefined",
                X_F=X,
                K=K,
                e=e,
                q_F=self.q_F,
            )
        return X

    def V_F(self, K: float, e: float) -> float:
        """``V_F = (1/rho) log(rho X_F) + (r_F_bar - rho)/rho^2``."""
        rho = self.params.preferences.rho
        X = self.X_F(K, e)
        return math.log(rho * X) / rho + (self.params.rates.rF_bar - rho) / (rho * rho)

    def V_F_e(self, K: float, e: float) -> float:
        """``V_{F,e} = 1/(rho X_F)``."""
        return 1.0 / (self.params.preferences.rho * self.X_F(K, e))

    def V_F_K(self, K: float, e: float) -> float:
        """``V_{F,K} = q_F V_{F,e}``, the successor envelope condition."""
        return self.q_F * self.V_F_e(K, e)

    def C_F(self, K: float, e: float) -> float:
        """``C_F^W = rho X_F``, the log optimum after arrival."""
        return self.params.preferences.rho * self.X_F(K, e)

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_F": self.q_F,
            "branch": self.branch,
            "iota_F": self.iota_F,
            "g_F": self.g_F,
            "tvc_margin": self.tvc_margin,
            "full_bgp_residual": self.full_bgp_residual,
            "installation_margin": self.installation_margin,
            "recovered_tau_F": self.recovered_tau_F,
            "residual_polynomial": self.residual_polynomial,
            "residual_level": self.residual_level,
            "bracket_width": self.bracket_width,
            "candidates": [c.as_dict() for c in self.candidates],
            "diagnostics": self.diagnostics,
        }


def solve_ak_successor(
    params: ModelParameters,
    interval: AkRootInterval,
    tolerances: SolverTolerances,
) -> AkSuccessor:
    """Enumerate, classify, and select the AK successor branch.

    Raises :class:`~ak_partial_ramsey.errors.BranchFailure` when no root passes the
    strict productive-value TVC, and equally when more than one does - an ambiguous
    selection is a stop condition, not a case for picking one.
    """
    candidates, diagnostics = enumerate_ak_roots(params, interval, tolerances)
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        raise BranchFailure(
            "no AK root in the declared interval satisfies the strict "
            "productive-value transversality condition",
            declared_interval=interval.as_dict(),
            candidates=[c.as_dict() for c in candidates],
            coefficients=ak_scalar_coefficients(params),
        )
    if len(accepted) > 1:
        raise BranchFailure(
            "more than one AK root satisfies the strict productive-value "
            "transversality condition; selection is ambiguous",
            accepted=[c.as_dict() for c in accepted],
        )
    sel = accepted[0]
    # World equity pricing recovers tau_F = 0 on the production-efficient branch.
    recovered_tau_F = (
        1.0 - (params.rates.rF_bar * sel.q + sel.iota - sel.q * sel.g) / params.ak_technology.A_bar
    )
    return AkSuccessor(
        q_F=sel.q,
        branch=sel.branch,
        iota_F=sel.iota,
        g_F=sel.g,
        tvc_margin=sel.tvc_margin,
        full_bgp_residual=params.rates.rF_bar - params.preferences.rho - sel.g,
        installation_margin=sel.installation_margin,
        recovered_tau_F=recovered_tau_F,
        residual_polynomial=sel.residual_polynomial,
        residual_level=sel.residual_level,
        bracket_width=sel.bracket_width,
        candidates=tuple(candidates),
        diagnostics=diagnostics,
        params=params,
    )
