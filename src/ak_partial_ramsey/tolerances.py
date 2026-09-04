"""Solver tolerances, held separately from economic primitives.

CS011 requires that economic primitives and solver tolerances stay separate objects, so
that a tolerance can never be mistaken for a model quantity and a tolerance change can
never masquerade as a specification change. They are hashed separately for the same
reason.

Nothing here is an economic threshold. Economic materiality thresholds sit behind an
open CS011 activation gate and are not defined by this package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .hashing import digest_mapping
from .validation import require_positive


@dataclass(frozen=True, slots=True)
class SolverTolerances:
    """Numerical tolerances for root finding, integration, and cross-route comparison."""

    #: Absolute tolerance for a converged scalar root bracket.
    root_xtol: float = 1e-14
    #: Relative tolerance for a converged scalar root bracket.
    root_rtol: float = 1e-15
    #: Number of scan points used to bracket roots across a declared interval.
    root_scan_points: int = 2001
    #: Absolute and relative tolerances handed to the initial-value integrator.
    ivp_atol: float = 1e-12
    ivp_rtol: float = 1e-11
    #: Residual tolerance handed to the boundary-value collocation solver.
    bvp_tol: float = 1e-10
    #: Relative offset from a stationary point used to seed a stable-manifold branch.
    manifold_offset: float = 1e-6
    #: Ratio by which the offset is shrunk for the offset-size sensitivity check.
    manifold_offset_ratio: float = 4.0
    #: Step used by centred finite differences in derivative cross-checks.
    fd_step: float = 1e-6
    #: Tolerance on identities the theory makes exact (envelopes, round trips, HJB).
    identity_atol: float = 1e-9
    #: Tolerance for agreement between two independently formulated numerical routes.
    cross_route_atol: float = 1e-7

    def __post_init__(self) -> None:
        for name in (
            "root_xtol",
            "root_rtol",
            "ivp_atol",
            "ivp_rtol",
            "bvp_tol",
            "manifold_offset",
            "fd_step",
            "identity_atol",
            "cross_route_atol",
        ):
            require_positive(name, getattr(self, name))
        if self.root_scan_points < 3:
            raise ValueError("root_scan_points must be at least 3")
        if self.manifold_offset_ratio <= 1.0:
            raise ValueError("manifold_offset_ratio must exceed 1")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def digest(self) -> str:
        """SHA-256 over the canonical serialisation of these tolerances."""
        return digest_mapping(self.as_dict())


#: Tolerances used by the CLI smoke stages. These are numerical settings only; they
#: encode no economic judgement and no materiality threshold.
DEFAULT_TOLERANCES = SolverTolerances()
