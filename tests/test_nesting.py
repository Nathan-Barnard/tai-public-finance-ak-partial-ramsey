"""Nesting claims, true and false.

Two CS011 benchmark rows:

* "Partial technology nesting": ``I_P`` returning to ``I_0`` nests only when the
  remaining successor primitives are also aligned. Otherwise the claim is false.
* "AK non-limit": ``I_P`` approaching one is never used as the AK solver. The AK block is
  a separately defined technology.
"""

from __future__ import annotations

import dataclasses

import pytest

from ak_partial_ramsey.errors import DomainError, NestingError
from ak_partial_ramsey.params import AkRootInterval, AkTechnology, TaskTechnology
from ak_partial_ramsey.successors.ak import solve_ak_successor
from ak_partial_ramsey.successors.partial import (
    partial_stationary_point,
    refuse_ak_by_task_share_limit,
    solve_partial_successor,
)

# --- the AK block is not a task-share limit -----------------------------------------


def test_ak_by_task_share_limit_is_refused():
    with pytest.raises(NestingError) as exc:
        refuse_ak_by_task_share_limit(0.999999)
    assert "separately defined technology" in exc.value.message


def test_task_share_of_one_is_outside_the_domain():
    with pytest.raises(DomainError):
        TaskTechnology(Z=1.0, I=1.0)


def test_partial_successor_cannot_reproduce_the_ak_block_at_any_task_share(two_mark_fixture, tol):
    """A structural argument, independent of the particular parameter values.

    The partial rest-point price is ``q_delta = exp(varphi delta)``, which does not
    depend on ``A_bar`` at all. The AK price solves ``r_F q = A_bar - iota + q g`` and
    does depend on ``A_bar``. So varying ``A_bar`` alone moves the AK successor while
    leaving every partial successor untouched, at any ``I_P`` whatsoever. No task-share
    limit can therefore deliver the AK block.
    """
    base = two_mark_fixture.params
    # 0.08 rather than a larger value: the AK existence condition a < exp(u) must
    # still hold, or the contrast would be between a root and a refusal rather than
    # between two roots.
    other = dataclasses.replace(base, ak_technology=AkTechnology(A_bar=0.08))

    ak_a = solve_ak_successor(base, AkRootInterval(0.5, 3.0), tol)
    ak_b = solve_ak_successor(other, AkRootInterval(0.5, 3.0), tol)
    assert abs(ak_a.q_F - ak_b.q_F) > 1e-3  # the AK price moved

    # Every partial rest point is identical across the two, for any task share.
    for I_P in (0.5, 0.7, 0.9, 0.99):
        pa = dataclasses.replace(base, partial_technology=TaskTechnology(Z=1.0, I=I_P))
        pb = dataclasses.replace(other, partial_technology=TaskTechnology(Z=1.0, I=I_P))
        sa = partial_stationary_point(pa)
        sb = partial_stationary_point(pb)
        assert sa.q_delta == sb.q_delta
        assert sa.K_star == pytest.approx(sb.K_star, rel=1e-15)
        # And the partial price is q_delta, never the AK price.
        assert abs(sa.q_delta - ak_a.q_F) > 1e-3


# --- partial technology nesting ------------------------------------------------------


def test_task_share_ordering_is_enforced(two_mark_fixture):
    """The finite-task-share condition 0 < I_0 < I_P < 1 is a domain requirement."""
    base = two_mark_fixture.params
    I_0 = base.pre_arrival_technology.I
    with pytest.raises(DomainError) as exc:
        dataclasses.replace(base, partial_technology=TaskTechnology(Z=1.0, I=I_0))
    assert "0 < I_0 < I_P < 1" in exc.value.message
    with pytest.raises(DomainError):
        dataclasses.replace(base, partial_technology=TaskTechnology(Z=1.0, I=I_0 / 2))


def test_equal_task_share_alone_does_not_nest(two_mark_fixture, tol):
    """Same I_P, different r_P_bar: the successors genuinely differ."""
    from ak_partial_ramsey.params import WorldRates

    base = two_mark_fixture.params
    other = dataclasses.replace(base, rates=WorldRates(r0_bar=0.030, rF_bar=0.035, rP_bar=0.055))
    a = partial_stationary_point(base)
    b = partial_stationary_point(other)
    assert base.partial_technology.I == other.partial_technology.I
    assert abs(a.K_star - b.K_star) / a.K_star > 1e-3
    assert a.U_P != b.U_P


def test_full_primitive_alignment_does_nest(two_mark_fixture, tol):
    """When every successor primitive coincides, the successors coincide exactly."""
    base = two_mark_fixture.params
    twin = dataclasses.replace(base)
    a = solve_partial_successor(base, two_mark_fixture.partial_capital_interval, tol)
    b = solve_partial_successor(twin, two_mark_fixture.partial_capital_interval, tol)
    assert a.point.K_star == pytest.approx(b.point.K_star, rel=1e-15)
    assert a.certified_domain == pytest.approx(b.certified_domain, rel=1e-15)
    lo, hi = a.certified_domain
    for frac in (0.2, 0.5, 0.8):
        K = lo + frac * (hi - lo)
        assert a.q_P(K) == pytest.approx(b.q_P(K), rel=1e-14)
        assert a.H_P(K) == pytest.approx(b.H_P(K), rel=1e-14)


def test_partial_successor_requires_its_own_world_rate(two_mark_fixture):
    """A partial successor without r_P_bar is a configuration error, not a default."""
    from ak_partial_ramsey.errors import ConfigurationError
    from ak_partial_ramsey.params import WorldRates

    base = two_mark_fixture.params
    with pytest.raises(ConfigurationError):
        dataclasses.replace(base, rates=WorldRates(r0_bar=0.03, rF_bar=0.035))


def test_positive_partial_mass_requires_a_partial_technology(two_mark_fixture):
    from ak_partial_ramsey.errors import ConfigurationError

    base = two_mark_fixture.params
    with pytest.raises(ConfigurationError):
        dataclasses.replace(base, partial_technology=None)
