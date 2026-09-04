"""Shared test fixtures.

Only two kinds of numbers appear anywhere in this suite: values supplied verbatim by a
theory packet, and clearly named manufactured values that carry no economic
interpretation. Nothing here closes a CS011 activation gate.
"""

from __future__ import annotations

import pytest

from ak_partial_ramsey.fixtures import get_fixture
from ak_partial_ramsey.tolerances import DEFAULT_TOLERANCES


@pytest.fixture(scope="session")
def tol():
    return DEFAULT_TOLERANCES


@pytest.fixture(scope="session")
def single_ak_fixture():
    """The theory-supplied two-root AK fixture."""
    return get_fixture("single-ak-two-root")


@pytest.fixture(scope="session")
def two_mark_fixture():
    """Manufactured two-mark configuration. No economic interpretation."""
    return get_fixture("manufactured-two-mark")


@pytest.fixture(scope="session")
def support_fixture():
    """Manufactured single-AK support restriction p_P = lambda_P_star = 0."""
    return get_fixture("manufactured-single-ak-support")


@pytest.fixture(scope="session")
def ak_single(single_ak_fixture, tol):
    from ak_partial_ramsey.successors.ak import solve_ak_successor

    return solve_ak_successor(single_ak_fixture.params, single_ak_fixture.ak_root_interval, tol)


@pytest.fixture(scope="session")
def ak_two_mark(two_mark_fixture, tol):
    from ak_partial_ramsey.successors.ak import solve_ak_successor

    return solve_ak_successor(two_mark_fixture.params, two_mark_fixture.ak_root_interval, tol)


@pytest.fixture(scope="session")
def partial_two_mark(two_mark_fixture, tol):
    from ak_partial_ramsey.successors.partial import solve_partial_successor

    return solve_partial_successor(
        two_mark_fixture.params, two_mark_fixture.partial_capital_interval, tol
    )
