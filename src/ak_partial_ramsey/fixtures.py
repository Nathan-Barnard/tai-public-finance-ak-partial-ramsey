"""Frozen test fixtures (block G0).

Two kinds of fixture live here, and the distinction is enforced by the ``provenance``
field rather than left to a reader's judgement:

``theory_supplied``
    Numbers written down verbatim in a theory packet, together with the values that
    packet states they must produce. Reproducing these is what G0 means.

``manufactured``
    Numbers chosen here solely to exercise code paths. They carry **no economic
    interpretation**, are not a calibration, and are not an anchor. Their names all
    begin with ``MANUFACTURED_``.

Nothing in this module supplies an object behind an open CS011 activation gate. In
particular there is no canonical inherited ``(S_0, M_0)``, no economic parameter
scenario, no final synthetic anchor, and no materiality threshold. Where a fixture needs
a value that the theory packet does not fix - a pre-arrival task share for a fixture that
only exercises the AK block, say - that value is listed in ``manufactured_fields`` and is
irrelevant to the quantities being checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .params import (
    AkRootInterval,
    AkTechnology,
    Installation,
    MarkIntensities,
    ModelParameters,
    PartialCapitalInterval,
    Preferences,
    TaskTechnology,
    WorldRates,
)

__all__ = ["FIXTURES", "Fixture", "get_fixture"]


@dataclass(frozen=True, slots=True)
class Fixture:
    """A named, immutable test configuration with its provenance recorded."""

    name: str
    provenance: str  # "theory_supplied" or "manufactured"
    source_locator: str
    economic_interpretation: str
    params: ModelParameters
    ak_root_interval: AkRootInterval
    state: dict[str, float]
    expected: dict[str, Any] = field(default_factory=dict)
    manufactured_fields: tuple[str, ...] = ()
    partial_capital_interval: PartialCapitalInterval | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "source_locator": self.source_locator,
            "economic_interpretation": self.economic_interpretation,
            "manufactured_fields": list(self.manufactured_fields),
            "params": self.params.as_dict(),
            "params_digest": self.params.digest(),
            "ak_root_interval": self.ak_root_interval.as_dict(),
            "partial_capital_interval": (
                self.partial_capital_interval.as_dict() if self.partial_capital_interval else None
            ),
            "state": self.state,
            "expected": self.expected,
            "notes": self.notes,
        }


# --- theory-supplied ---------------------------------------------------------------

#: The single-AK packet's admissible/rejected scalar-root fixture, verbatim.
#:
#: Source: research-notes/single-ak-small-open-full-automation-solution.md, section
#: "4. Admissible and rejected scalar-root fixtures". The packet fixes
#: ``(rho, r_F_bar, A_bar, varphi, delta) = (0.04, 0.035, 0.10, 2.5, 0.06)`` and the
#: current balance sheet ``K = 1``, ``e = 0.2``, ``A^O = 0.5``, ``B = 0.1``, and states
#: the full two-root table reproduced in ``expected`` below.
SINGLE_AK_TWO_ROOT = Fixture(
    name="single-ak-two-root",
    provenance="theory_supplied",
    source_locator=(
        "research-notes/single-ak-small-open-full-automation-solution.md "
        "section '4. Admissible and rejected scalar-root fixtures'"
    ),
    economic_interpretation=(
        "None. The packet presents this as a falsification of 'any scalar root is "
        "admissible', not as numerical evidence of global existence and not as a "
        "calibration."
    ),
    params=ModelParameters(
        preferences=Preferences(rho=0.04),
        installation=Installation(varphi=2.5, delta=0.06),
        # The AK block depends only on (rho, r_F_bar, A_bar, varphi, delta). The
        # pre-arrival technology, r_0_bar and the intensities below are manufactured
        # padding required to construct a complete parameter object; no checked
        # quantity depends on them.
        pre_arrival_technology=TaskTechnology(Z=1.0, I=0.6),
        ak_technology=AkTechnology(A_bar=0.10),
        rates=WorldRates(r0_bar=0.03, rF_bar=0.035),
        intensities=MarkIntensities(
            lambda_total=0.02,
            p_P=0.0,
            p_F=1.0,
            lambda_P_star=0.0,
            lambda_F_star=0.015,
        ),
    ),
    ak_root_interval=AkRootInterval(q_lo=0.5, q_hi=3.0),
    state={"K": 1.0, "e": 0.2, "A_O": 0.5, "B": 0.1},
    manufactured_fields=(
        "pre_arrival_technology.Z",
        "pre_arrival_technology.I",
        "rates.r0_bar",
        "intensities.lambda_total",
        "intensities.lambda_F_star",
        "ak_root_interval.q_lo",
        "ak_root_interval.q_hi",
    ),
    expected={
        "a": 1.25,
        "q_m": 1.268074997,
        "n_positive_roots": 2,
        "accepted": {
            "q_F": 1.060083947,
            "iota_F": 0.024033579,
            "g_F": -0.036660760,
            "installation_margin": 0.424033579,
            "X_F": 1.260083947,
            "C_F_W": 0.050403358,
            "B": 0.1,
            "A_O": 0.5,
            "tvc_margin": 0.071660760,
            "full_bgp_residual": 0.031660760,
            "recovered_tau_F": 0.0,
        },
        "rejected": {
            "q_F": 1.488123711,
            "iota_F": 0.195249484,
            "g_F": 0.099006429,
            "installation_margin": 0.595249484,
            "X_F": 1.688123711,
            "C_F_W": 0.067524948,
            "B": 0.1,
            "A_O": 0.5,
            "tvc_margin": -0.064006429,
            "full_bgp_residual": -0.104006429,
            "rejection_reason": "productive_value_tvc_violated",
        },
        "balance_sheet_at_either_root": {
            "psi": 0.3,
            "foreign_equity": -0.3,
            "foreign_safe": -0.4,
            "foreign_net_claim": -0.7,
        },
    },
    notes=(
        "Both roots pass the scalar equation, current positivity, balance-sheet "
        "solvency, finite-log-value, and the gross-position checks. The upper root is "
        "rejected only because discounted installed-capital value grows instead of "
        "vanishing, so reproducing the rejection is as much a part of G0 as "
        "reproducing the selection."
    ),
)


# --- manufactured -------------------------------------------------------------------

#: Manufactured two-mark configuration. Exercises both successor services and the
#: two-mark exposure geometry. No economic interpretation.
_MANUFACTURED_TWO_MARK_PARAMS = ModelParameters(
    preferences=Preferences(rho=0.04),
    installation=Installation(varphi=2.5, delta=0.06),
    pre_arrival_technology=TaskTechnology(Z=1.0, I=0.40),
    partial_technology=TaskTechnology(Z=1.0, I=0.60),
    ak_technology=AkTechnology(A_bar=0.10),
    rates=WorldRates(r0_bar=0.030, rF_bar=0.035, rP_bar=0.045),
    intensities=MarkIntensities(
        lambda_total=0.02,
        p_P=0.6,
        p_F=0.4,
        lambda_P_star=0.010,
        lambda_F_star=0.015,
    ),
)


#: A capital window bracketing the manufactured partial rest point.
#:
#: Frozen as literals rather than recomputed at import, for two reasons. A fixture that
#: recomputes itself from the code under test is not frozen, and it would make this
#: module depend on the successor service it exists to supply inputs for.
#:
#: Derivation, recorded so the numbers are reproducible rather than arbitrary: the
#: manufactured partial rest point is K_P* = 320.2402599858991, and the window is
#: [K_P*/1.5, K_P*x1.5]. ``tests/test_partial_successor.py`` re-derives K_P* from the
#: parameters and asserts these bounds still bracket it, so the freeze cannot go stale
#: silently.
#:
#: This is a code-path window, not a frozen successor domain. Successor domains for a
#: substantive run sit behind CS011 activation gate 3 and are not defined here.
_MANUFACTURED_PARTIAL_INTERVAL = PartialCapitalInterval(
    K_lo=213.49350665726607,
    K_hi=480.3603899788487,
)


MANUFACTURED_TWO_MARK = Fixture(
    name="manufactured-two-mark",
    provenance="manufactured",
    source_locator="ak_partial_ramsey.fixtures (this module)",
    economic_interpretation=(
        "None. Chosen to exercise both successor services and the two-mark exposure "
        "geometry. Not a calibration, not an anchor, not a scenario."
    ),
    params=_MANUFACTURED_TWO_MARK_PARAMS,
    ak_root_interval=AkRootInterval(q_lo=0.5, q_hi=3.0),
    state={"K": 300.0, "e": 5.0, "q": 1.15, "a": 20.0},
    manufactured_fields=("all",),
    partial_capital_interval=_MANUFACTURED_PARTIAL_INTERVAL,
    notes=(
        "Every number here exists to exercise a code path. The state in particular is "
        "an arbitrary probe, not an equilibrium point, and its implementation margins "
        "are not expected to be strict: a strictly interior pre-arrival state is an "
        "equilibrium object and locating one requires blocks N2 and N4. Searching for "
        "one and freezing it here would amount to inventing the economic scenario that "
        "CS011 activation gate 2 leaves open."
    ),
)


#: Manufactured single-AK support restriction of the two-mark configuration:
#: ``p_P = lambda_P_star = 0`` with the partial technology still present. Used to test
#: that the two-mark equations collapse exactly onto the single-mark system.
MANUFACTURED_SINGLE_AK_SUPPORT = Fixture(
    name="manufactured-single-ak-support",
    provenance="manufactured",
    source_locator="ak_partial_ramsey.fixtures (this module)",
    economic_interpretation=(
        "None. The exact support restriction p_P = lambda_P_star = 0 of the "
        "manufactured two-mark configuration, retained as a first-class branch."
    ),
    params=ModelParameters(
        preferences=Preferences(rho=0.04),
        installation=Installation(varphi=2.5, delta=0.06),
        pre_arrival_technology=TaskTechnology(Z=1.0, I=0.40),
        partial_technology=TaskTechnology(Z=1.0, I=0.60),
        ak_technology=AkTechnology(A_bar=0.10),
        rates=WorldRates(r0_bar=0.030, rF_bar=0.035, rP_bar=0.045),
        intensities=MarkIntensities(
            lambda_total=0.02,
            p_P=0.0,
            p_F=1.0,
            lambda_P_star=0.0,
            lambda_F_star=0.015,
        ),
    ),
    ak_root_interval=AkRootInterval(q_lo=0.5, q_hi=3.0),
    state={"K": 300.0, "e": 5.0, "q": 1.15, "a": 20.0},
    manufactured_fields=("all",),
    partial_capital_interval=_MANUFACTURED_PARTIAL_INTERVAL,
    notes=(
        "The partial technology stays present so that the collapse is exercised through "
        "the two-mark code path with a genuinely inactive mark, rather than by removing "
        "the mark from the configuration."
    ),
)


FIXTURES: dict[str, Fixture] = {
    f.name: f
    for f in (
        SINGLE_AK_TWO_ROOT,
        MANUFACTURED_TWO_MARK,
        MANUFACTURED_SINGLE_AK_SUPPORT,
    )
}


def get_fixture(name: str) -> Fixture:
    """Look up a fixture by name, listing the available names on a miss."""
    try:
        return FIXTURES[name]
    except KeyError:
        raise KeyError(f"unknown fixture {name!r}; available: {sorted(FIXTURES)}") from None
