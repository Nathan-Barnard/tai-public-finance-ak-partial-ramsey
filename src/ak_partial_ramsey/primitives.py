"""Production, installation, price, and factor-payment primitives.

Equation locators refer to ``docs/equation_crosswalk.md``, which maps each function here
to its numbered equation in the two theory packets.

Every function validates its inputs and refuses non-finite values and domain
violations. None of them clips.
"""

from __future__ import annotations

import math

from .errors import DomainError
from .params import AkTechnology, Installation, TaskTechnology
from .validation import require_finite, require_positive

__all__ = [
    "ak_output",
    "ak_rental",
    "ak_wage",
    "capital_growth",
    "capital_growth_derivative",
    "installation_domain_margin",
    "installation_function",
    "installation_rate",
    "installation_rate_derivative",
    "omega_Z",
    "task_output",
    "task_rental",
    "task_rental_derivative",
    "task_wage",
    "zero_growth_price",
]


# --- fixed-task-share production -------------------------------------------------


def omega_Z(technology: TaskTechnology) -> float:
    """``Omega_Z(I) = (Z/(1-I))**(1-I) * I**(-I)``.

    Two-mark packet, section 1. The single-AK packet's ``A_0`` is this quantity at
    ``I = I_0``.
    """
    Z, I = technology.Z, technology.I
    return (Z / (1.0 - I)) ** (1.0 - I) * I ** (-I)


def task_output(K: float, technology: TaskTechnology) -> float:
    """``Y_s(K) = Omega_Z(I_s) * K**I_s``."""
    K = require_positive("K", K)
    return omega_Z(technology) * K**technology.I


def task_wage(K: float, technology: TaskTechnology) -> float:
    """``W_s(K) = (1 - I_s) * Y_s(K)``."""
    return (1.0 - technology.I) * task_output(K, technology)


def task_rental(K: float, technology: TaskTechnology) -> float:
    """``R_s(K) = I_s * Y_s(K) / K``, the marginal product of capital."""
    K = require_positive("K", K)
    return technology.I * task_output(K, technology) / K


def task_rental_derivative(K: float, technology: TaskTechnology) -> float:
    """``R_s'(K) = -(1 - I_s) * R_s(K) / K``, strictly negative for ``0 < I_s < 1``."""
    K = require_positive("K", K)
    return -(1.0 - technology.I) * task_rental(K, technology) / K


# --- absorbing AK production -----------------------------------------------------


def ak_output(K: float, technology: AkTechnology) -> float:
    """``Y_F(K) = A_bar * K``."""
    K = require_positive("K", K)
    return technology.A_bar * K


def ak_wage(K: float, technology: AkTechnology) -> float:
    """``W_F(K) = 0``: the AK state is labour free.

    ``K`` and ``technology`` are accepted so that this shares the signature of
    :func:`task_wage` and can stand in for it in regime-generic code.
    """
    require_positive("K", K)
    _ = technology
    return 0.0


def ak_rental(K: float, technology: AkTechnology) -> float:
    """``R_F(K) = A_bar``, independent of ``K``."""
    require_positive("K", K)
    return technology.A_bar


# --- log installation technology -------------------------------------------------


def installation_function(z: float, installation: Installation) -> float:
    """``Phi(z) = log(1 + varphi * z) / varphi`` on ``z > -1/varphi``."""
    z = require_finite("z", z)
    varphi = installation.varphi
    arg = 1.0 + varphi * z
    if arg <= 0.0:
        raise DomainError(
            "installation rate is at or below the domain boundary -1/varphi",
            z=z,
            varphi=varphi,
            lower_bound=installation.iota_lower_bound,
        )
    return math.log(arg) / varphi


def installation_rate(q: float, installation: Installation) -> float:
    """``iota(q) = (q - 1)/varphi``, the interior firm investment branch."""
    q = require_positive("q", q)
    return (q - 1.0) / installation.varphi


def installation_rate_derivative(installation: Installation) -> float:
    """``iota_q = 1/varphi``, constant."""
    return 1.0 / installation.varphi


def capital_growth(q: float, installation: Installation) -> float:
    """``g(q) = log(q)/varphi - delta`` for ``q > 0``."""
    q = require_positive("q", q)
    return math.log(q) / installation.varphi - installation.delta


def capital_growth_derivative(q: float, installation: Installation) -> float:
    """``g_q(q) = 1/(varphi * q)``.

    The identity ``q * g_q = iota_q = 1/varphi`` is what makes the productive-wealth
    derivative identity ``H'(K) = q(K)`` hold exactly; it is checked in the tests.
    """
    q = require_positive("q", q)
    return 1.0 / (installation.varphi * q)


def installation_domain_margin(q: float, installation: Installation) -> float:
    """``iota(q) + 1/varphi``: distance to the installation-domain boundary.

    Strictly positive for every ``q > 0``; reported as a slack, never enforced by
    clipping.
    """
    return installation_rate(q, installation) + 1.0 / installation.varphi


def zero_growth_price(installation: Installation) -> float:
    """``q_delta = exp(varphi * delta)``, the unique positive price with ``g(q) = 0``."""
    return installation.q_delta
