"""Typed refusals.

CS011 requires that every public numerical function reject non-finite inputs and return
an explicit domain or branch failure rather than a NaN or a silently clipped value. The
exceptions here are that contract. Each carries a machine-readable ``detail`` mapping so
a refusal can be serialised into a diagnostic report without re-parsing a message string.
"""

from __future__ import annotations

from typing import Any


class AkPartialRamseyError(Exception):
    """Base class for every refusal raised by this package."""

    #: Short stable slug used in machine-readable reports.
    kind: str = "error"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "type": type(self).__name__,
            "message": self.message,
            "detail": self.detail,
        }


class NonFiniteInputError(AkPartialRamseyError):
    """A NaN or infinity reached a public numerical entry point.

    Per the error-handling policy a non-finite value on the claimed domain is a
    hard stop, never a value to propagate or replace with a default.
    """

    kind = "non_finite_input"


class DomainError(AkPartialRamseyError):
    """An input or an intermediate quantity left its declared economic domain.

    Raised rather than clipping. A true boundary is never replaced by a computational
    box or a conservative guardrail.
    """

    kind = "domain_failure"


class BranchFailure(AkPartialRamseyError):
    """No admissible branch exists, or branch identity could not be established."""

    kind = "branch_failure"


class RootCoverageError(AkPartialRamseyError):
    """Root enumeration could not certify that it found every root in the interval."""

    kind = "root_coverage_failure"


class RankFailure(AkPartialRamseyError):
    """The event payoff vector is zero, so the exposure is not identified.

    This is a change in rank, not poor numerical conditioning, and is refused rather
    than regularised.
    """

    kind = "rank_failure"


class NestingError(AkPartialRamseyError):
    """A false nesting relation was requested.

    Specifically: constructing the AK successor as a limit of the finite-task-share
    partial successor. The AK block is a separately defined technology; ``I_P -> 1``
    does not define its direct ``A_bar * K`` production or its wage and value boundary.
    """

    kind = "nesting_error"


class EvaluatorMismatch(AkPartialRamseyError):
    """The independent evaluator disagrees with the equation core beyond tolerance.

    Per CS011 this is a diagnostic failure requiring ``diagnose_before_scaling``; it is
    never repaired by loosening the tolerance.
    """

    kind = "evaluator_mismatch"


class RefinementInstability(AkPartialRamseyError):
    """A root, branch label, or manifold changed under numerical refinement.

    Reported without a diagnosed fold, this is a stop condition.
    """

    kind = "refinement_instability"


class ConfigurationError(AkPartialRamseyError):
    """A parameter, domain, or tolerance object is internally inconsistent or absent.

    Also raised when a caller asks this package to supply an object that a CS011
    activation gate has deliberately left open.
    """

    kind = "configuration_error"


class ActivationGateError(ConfigurationError):
    """A caller requested an object frozen behind an open CS011 activation gate.

    The five open gates are the canonical inherited state ``(S_0, M_0)``, the economic
    parameter scenario, the final synthetic anchor, successor domains for a substantive
    run, and economic materiality thresholds. This package accepts each as a validated
    input and refuses to invent a default for any of them.
    """

    kind = "activation_gate"
