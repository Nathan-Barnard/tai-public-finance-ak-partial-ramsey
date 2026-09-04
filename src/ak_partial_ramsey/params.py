"""Immutable, validated, hashable parameter and domain schemas.

Every field here is an economic primitive drawn from the frozen derivation contracts of
the two theory packets. Solver tolerances live in :mod:`ak_partial_ramsey.tolerances`
and are deliberately not reachable from these objects.

Domains that a CS011 activation gate has left open - the partial successor's ``K``
interval and the admissible AK root interval - are modelled as required inputs with no
default. Constructing a run without supplying them raises
:class:`~ak_partial_ramsey.errors.ActivationGateError`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .errors import ConfigurationError, DomainError
from .hashing import digest_mapping
from .validation import (
    require_finite,
    require_nonnegative,
    require_ordered,
    require_positive,
    require_strictly_between,
)

__all__ = [
    "AkRootInterval",
    "AkTechnology",
    "Installation",
    "MarkIntensities",
    "ModelParameters",
    "PartialCapitalInterval",
    "Preferences",
    "TaskTechnology",
    "WorldRates",
]


@dataclass(frozen=True, slots=True)
class Preferences:
    """Worker preferences. Log utility with discount rate ``rho > 0``."""

    rho: float

    def __post_init__(self) -> None:
        require_positive("rho", self.rho)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Installation:
    """Log installation technology, common to every regime.

    ``Phi(iota) = log(1 + varphi * iota) / varphi`` with ``iota > -1/varphi``, giving
    ``iota(q) = (q - 1)/varphi`` and ``g(q) = log(q)/varphi - delta`` for ``q > 0``.
    """

    varphi: float
    delta: float

    def __post_init__(self) -> None:
        require_positive("varphi", self.varphi)
        require_nonnegative("delta", self.delta)

    @property
    def iota_lower_bound(self) -> float:
        """The installation-domain boundary ``-1/varphi``."""
        return -1.0 / self.varphi

    @property
    def q_delta(self) -> float:
        """The zero-growth price ``q_delta = exp(varphi * delta)``, where ``g(q) = 0``."""
        import math

        return math.exp(self.varphi * self.delta)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskTechnology:
    """Fixed-task-share production ``Y(K) = Omega_Z(I) * K**I``.

    ``Omega_Z(I) = (Z/(1-I))**(1-I) * I**(-I)``, per section 1 of the two-mark packet.
    The single-AK packet writes the same object as ``Y_0(K) = A_0 * K**I_0``; the two
    are the same function with ``A_0 = Omega_Z(I_0)``. See ``docs/equation_crosswalk.md``.
    """

    Z: float
    I: float

    def __post_init__(self) -> None:
        require_positive("Z", self.Z)
        require_strictly_between("I", self.I, 0.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AkTechnology:
    """Absorbing labour-free AK production ``Y_F(K) = A_bar * K``, ``W_F = 0``."""

    A_bar: float

    def __post_init__(self) -> None:
        require_positive("A_bar", self.A_bar)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorldRates:
    """Exogenous world safe rates, one per regime.

    ``r_P_bar`` is required only when a partial successor is present.
    """

    r0_bar: float
    rF_bar: float
    rP_bar: float | None = None

    def __post_init__(self) -> None:
        require_finite("r0_bar", self.r0_bar)
        require_finite("rF_bar", self.rF_bar)
        if self.rP_bar is not None:
            require_finite("rP_bar", self.rP_bar)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MarkIntensities:
    """Physical and risk-neutral event intensities, kept strictly distinct.

    The physical intensities are ``lambda_j = lambda_total * p_j``. The risk-neutral
    intensities ``lambda_j_star`` come from the exogenous world stochastic discount
    factor and are never substituted for the physical ones.

    The single-AK support restriction ``p_P = lambda_P_star = 0`` is a first-class
    branch, not a degenerate input: G0 requires it to be retained with no division by
    either quantity.
    """

    lambda_total: float
    p_P: float
    p_F: float
    lambda_P_star: float
    lambda_F_star: float

    def __post_init__(self) -> None:
        require_positive("lambda_total", self.lambda_total)
        require_nonnegative("p_P", self.p_P)
        require_positive("p_F", self.p_F)
        require_nonnegative("lambda_P_star", self.lambda_P_star)
        require_positive("lambda_F_star", self.lambda_F_star)
        if abs(self.p_P + self.p_F - 1.0) > 1e-12:
            raise DomainError(
                "mark probabilities must sum to one",
                p_P=self.p_P,
                p_F=self.p_F,
                sum=self.p_P + self.p_F,
            )
        if self.p_P == 0.0 and self.lambda_P_star != 0.0:
            raise DomainError(
                "a mark with zero physical mass cannot carry a positive risk-neutral "
                "intensity; set lambda_P_star to zero for the single-AK support "
                "restriction",
                p_P=self.p_P,
                lambda_P_star=self.lambda_P_star,
            )

    @property
    def lambda_P(self) -> float:
        """Physical intensity of the partial-automation mark."""
        return self.lambda_total * self.p_P

    @property
    def lambda_F(self) -> float:
        """Physical intensity of the AK mark."""
        return self.lambda_total * self.p_F

    @property
    def lambda_star_sum(self) -> float:
        """``lambda_Sigma_star = lambda_P_star + lambda_F_star``."""
        return self.lambda_P_star + self.lambda_F_star

    @property
    def is_single_ak_support(self) -> bool:
        """True on the exact single-AK support restriction ``p_P = lambda_P_star = 0``."""
        return self.p_P == 0.0 and self.lambda_P_star == 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AkRootInterval:
    """Declared search interval for the AK installed-capital price ``q_F``.

    CS011 activation gate 3 leaves the admissible AK root interval open for a
    substantive run. This object therefore has no default: the caller supplies it and
    the enumeration scans exactly it, reporting every root found inside.
    """

    q_lo: float
    q_hi: float

    def __post_init__(self) -> None:
        require_positive("q_lo", self.q_lo)
        require_positive("q_hi", self.q_hi)
        require_ordered("q_lo", self.q_lo, "q_hi", self.q_hi)

    def contains(self, q: float) -> bool:
        return self.q_lo <= q <= self.q_hi

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PartialCapitalInterval:
    """Declared capital interval over which the partial stable manifold is certified.

    CS011 activation gate 3 leaves this open for a substantive run; it is a required
    input here. Interpolants built on it are valid only inside it, and evaluating them
    outside raises rather than extrapolating.
    """

    K_lo: float
    K_hi: float

    def __post_init__(self) -> None:
        require_positive("K_lo", self.K_lo)
        require_positive("K_hi", self.K_hi)
        require_ordered("K_lo", self.K_lo, "K_hi", self.K_hi)

    def contains(self, K: float) -> bool:
        return self.K_lo <= K <= self.K_hi

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelParameters:
    """The full economic primitive set for the pre-arrival economy and both successors.

    ``partial_technology`` is ``None`` for a configuration that carries only the AK
    successor - the exact single-AK branch. When it is present the task shares must
    satisfy ``0 < I_0 < I_P < 1``, which is the finite-task-share condition of the
    two-mark packet.
    """

    preferences: Preferences
    installation: Installation
    pre_arrival_technology: TaskTechnology
    ak_technology: AkTechnology
    rates: WorldRates
    intensities: MarkIntensities
    partial_technology: TaskTechnology | None = None

    def __post_init__(self) -> None:
        if self.partial_technology is not None:
            I_0 = self.pre_arrival_technology.I
            I_P = self.partial_technology.I
            if not (0.0 < I_0 < I_P < 1.0):
                raise DomainError(
                    "the partial successor requires 0 < I_0 < I_P < 1",
                    I_0=I_0,
                    I_P=I_P,
                )
            if self.rates.rP_bar is None:
                raise ConfigurationError(
                    "a partial successor requires the world safe rate r_P_bar",
                    partial_task_share=I_P,
                )
        elif self.intensities.p_P != 0.0:
            raise ConfigurationError(
                "positive partial-mark probability requires a partial successor technology",
                p_P=self.intensities.p_P,
            )

    @property
    def has_partial_successor(self) -> bool:
        return self.partial_technology is not None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "preferences": self.preferences.as_dict(),
            "installation": self.installation.as_dict(),
            "pre_arrival_technology": self.pre_arrival_technology.as_dict(),
            "ak_technology": self.ak_technology.as_dict(),
            "rates": self.rates.as_dict(),
            "intensities": self.intensities.as_dict(),
            "partial_technology": (
                self.partial_technology.as_dict() if self.partial_technology else None
            ),
        }
        return out

    def digest(self) -> str:
        """SHA-256 over the canonical serialisation of these economic primitives."""
        return digest_mapping(self.as_dict())
