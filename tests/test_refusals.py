"""Non-finite input and missing-domain refusals.

CS011: every public numerical function must reject non-finite inputs and return an
explicit domain or branch failure rather than a NaN or a silently clipped value. These
tests assert the refusals actually happen, with the right type, at every public entry
point that takes a number.
"""

from __future__ import annotations

import math

import pytest

from ak_partial_ramsey.errors import (
    ActivationGateError,
    ConfigurationError,
    DomainError,
    NonFiniteInputError,
)
from ak_partial_ramsey.events import (
    event_normalized,
    jump_payoff,
    level_from_normalized,
    normalized_from_level,
)
from ak_partial_ramsey.exposure import Mark, solve_public_exposure
from ak_partial_ramsey.params import (
    AkRootInterval,
    AkTechnology,
    Installation,
    MarkIntensities,
    PartialCapitalInterval,
    Preferences,
    TaskTechnology,
    WorldRates,
)
from ak_partial_ramsey.primitives import (
    capital_growth,
    installation_function,
    installation_rate,
    task_output,
    task_rental,
    task_wage,
)
from ak_partial_ramsey.validation import (
    require_finite,
    require_finite_array,
    require_nonzero_payoff_vector,
    require_positive,
)

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_require_finite_refuses(bad):
    with pytest.raises(NonFiniteInputError):
        require_finite("x", bad)


def test_require_finite_refuses_non_numeric():
    with pytest.raises(NonFiniteInputError):
        require_finite("x", "not a number")


@pytest.mark.parametrize("bad", NON_FINITE)
def test_require_finite_array_refuses(bad):
    with pytest.raises(NonFiniteInputError) as exc:
        require_finite_array("v", [1.0, 2.0, bad, 4.0])
    assert exc.value.detail["first_bad_index"] == [2]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_primitives_refuse_non_finite_prices(bad):
    inst = Installation(varphi=2.5, delta=0.06)
    for fn in (capital_growth, installation_rate):
        with pytest.raises(NonFiniteInputError):
            fn(bad, inst)
    with pytest.raises(NonFiniteInputError):
        installation_function(bad, inst)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_production_refuses_non_finite_capital(bad):
    tech = TaskTechnology(Z=1.0, I=0.6)
    for fn in (task_output, task_wage, task_rental):
        with pytest.raises(NonFiniteInputError):
            fn(bad, tech)


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-300])
def test_primitives_refuse_nonpositive_prices(bad):
    inst = Installation(varphi=2.5, delta=0.06)
    with pytest.raises(DomainError):
        capital_growth(bad, inst)
    with pytest.raises(DomainError):
        installation_rate(bad, inst)


def test_installation_domain_boundary_is_refused_not_clipped():
    """iota <= -1/varphi is outside the log installation domain."""
    inst = Installation(varphi=2.5, delta=0.06)
    with pytest.raises(DomainError) as exc:
        installation_function(-1.0 / 2.5, inst)
    assert exc.value.detail["lower_bound"] == pytest.approx(-0.4)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_event_maps_refuse_non_finite(bad):
    with pytest.raises(NonFiniteInputError):
        event_normalized(bad, 1.0, 0.5)
    with pytest.raises(NonFiniteInputError):
        event_normalized(1.0, bad, 0.5)
    with pytest.raises(NonFiniteInputError):
        jump_payoff(bad, 1.0)
    with pytest.raises(NonFiniteInputError):
        normalized_from_level(bad, 1.0, 1.0, 1.0)
    with pytest.raises(NonFiniteInputError):
        level_from_normalized(bad, 1.0, 1.0, 1.0)


def test_jump_payoff_refuses_nonpositive_prices():
    with pytest.raises(DomainError):
        jump_payoff(1.0, 0.0)
    with pytest.raises(DomainError):
        jump_payoff(0.0, 1.0)


# --- parameter schema validation ------------------------------------------------------


@pytest.mark.parametrize("bad", [*NON_FINITE, 0.0, -0.04])
def test_preferences_refuse_bad_discount_rate(bad):
    with pytest.raises((NonFiniteInputError, DomainError)):
        Preferences(rho=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_installation_refuses_nonpositive_varphi(bad):
    with pytest.raises(DomainError):
        Installation(varphi=bad, delta=0.06)


def test_installation_refuses_negative_depreciation():
    with pytest.raises(DomainError):
        Installation(varphi=2.5, delta=-0.01)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
def test_task_share_domain(bad):
    with pytest.raises(DomainError):
        TaskTechnology(Z=1.0, I=bad)


def test_ak_technology_refuses_nonpositive_productivity():
    with pytest.raises(DomainError):
        AkTechnology(A_bar=0.0)


def test_mark_probabilities_must_sum_to_one():
    with pytest.raises(DomainError) as exc:
        MarkIntensities(
            lambda_total=0.02,
            p_P=0.5,
            p_F=0.4,
            lambda_P_star=0.01,
            lambda_F_star=0.015,
        )
    assert "sum to one" in exc.value.message


def test_massless_mark_may_not_be_priced():
    """p_P = 0 with lambda_P_star > 0 is incoherent, and is refused explicitly."""
    with pytest.raises(DomainError) as exc:
        MarkIntensities(
            lambda_total=0.02,
            p_P=0.0,
            p_F=1.0,
            lambda_P_star=0.01,
            lambda_F_star=0.015,
        )
    assert "zero physical mass" in exc.value.message


def test_single_ak_support_restriction_is_accepted():
    """The support restriction itself is a first-class configuration, not an error."""
    i = MarkIntensities(lambda_total=0.02, p_P=0.0, p_F=1.0, lambda_P_star=0.0, lambda_F_star=0.015)
    assert i.is_single_ak_support
    assert i.lambda_P == 0.0
    assert i.lambda_F == pytest.approx(0.02)


def test_intervals_must_be_ordered_and_positive():
    with pytest.raises(DomainError):
        AkRootInterval(q_lo=3.0, q_hi=0.5)
    with pytest.raises(DomainError):
        AkRootInterval(q_lo=-1.0, q_hi=3.0)
    with pytest.raises(DomainError):
        PartialCapitalInterval(K_lo=10.0, K_hi=1.0)


def test_world_rates_refuse_non_finite():
    with pytest.raises(NonFiniteInputError):
        WorldRates(r0_bar=float("nan"), rF_bar=0.035)


# --- missing domain -------------------------------------------------------------------


def test_empty_successor_wealth_interval_is_refused(tol):
    """Opposite-signed payoffs whose bounds cross leave no admissible exposure."""
    from ak_partial_ramsey.exposure import successor_wealth_interval

    # J_P > 0 forces psi > -h_P/J_P = 10; J_F < 0 forces psi < -h_F/J_F = -8.
    m = (Mark("P", 0.012, 0.010, 0.5, -5.0), Mark("F", 0.008, 0.015, -0.5, -4.0))
    with pytest.raises(DomainError) as exc:
        successor_wealth_interval(m)
    assert "empty" in exc.value.message
    with pytest.raises(DomainError):
        solve_public_exposure(m, 0.4, 0.04, tol)


def test_zero_payoff_vector_helper():
    with pytest.raises(Exception) as exc:
        require_nonzero_payoff_vector("J", [0.0, 0.0])
    assert exc.value.detail["rank"] == 0
    assert require_nonzero_payoff_vector("J", [0.0, 0.5]) == (0.0, 0.5)


def test_require_positive_message_names_the_field():
    with pytest.raises(DomainError) as exc:
        require_positive("K", -1.0)
    assert exc.value.detail["name"] == "K"
    assert exc.value.detail["required"] == "> 0"


# --- activation gates -----------------------------------------------------------------


def test_activation_gate_error_is_a_configuration_error():
    """A caller asking for a gated object gets a distinct, catchable refusal."""
    assert issubclass(ActivationGateError, ConfigurationError)
    exc = ActivationGateError("gated", gate="inherited state")
    assert exc.kind == "activation_gate"
    assert exc.as_dict()["detail"]["gate"] == "inherited state"


def test_no_module_supplies_a_default_inherited_state():
    """There is no S_0/M_0 anywhere in the package: the gate is open, so nothing fills it."""
    import ak_partial_ramsey

    names = dir(ak_partial_ramsey)
    for forbidden in ("S_0", "M_0", "INHERITED_STATE", "ECONOMIC_SCENARIO", "ANCHOR"):
        assert forbidden not in names


def test_errors_serialise_for_machine_reports():
    exc = DomainError("boundary", name="B", value=-1.0)
    d = exc.as_dict()
    assert d["kind"] == "domain_failure"
    assert d["type"] == "DomainError"
    assert d["detail"]["value"] == -1.0
    assert math.isfinite(d["detail"]["value"])
