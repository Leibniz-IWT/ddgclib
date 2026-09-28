"""Pluggable contact force models for DEM.

Force models follow the :class:`~ddgclib.operators._registry.MethodRegistry`
pattern. Each model is registered by name and can be selected at runtime.

Models compute normal force ``F_n`` and tangential force ``F_t`` from a
:class:`~ddgclib.dem._contact.Contact`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ddgclib.dem._contact import Contact
from ddgclib.operators._registry import MethodRegistry


@dataclass
class ContactForceResult:
    """Result of a contact force computation.

    Attributes
    ----------
    F_n : np.ndarray
        Normal force on particle i (repulsive, away from j).
    F_t : np.ndarray
        Tangential friction force on particle i.
    torque_i : np.ndarray
        Torque on particle i from this contact.
    torque_j : np.ndarray
        Torque on particle j from this contact.
    """

    F_n: np.ndarray
    F_t: np.ndarray
    torque_i: np.ndarray
    torque_j: np.ndarray

    @property
    def total_force_on_i(self) -> np.ndarray:
        """Total force vector acting on particle i."""
        return self.F_n + self.F_t

    @property
    def total_force_on_j(self) -> np.ndarray:
        """Total force vector acting on particle j (Newton III)."""
        return -(self.F_n + self.F_t)


class ContactForceModel(ABC):
    """Abstract base for DEM contact force models."""

    @abstractmethod
    def compute(self, contact: Contact) -> ContactForceResult:
        """Compute contact forces from a detected contact."""

    @abstractmethod
    def name(self) -> str:
        """Return the model name for registry lookup."""


class HertzContact(ContactForceModel):
    """Hertzian contact model (non-linear elastic + viscous damping).

    Normal force::

        F_n = -k_n * delta_n^(3/2) - gamma_n * v_n

    where ``k_n = (4/3) * E_star * sqrt(R_star)``, with::

        1/E_star = (1-nu_i^2)/E_i + (1-nu_j^2)/E_j  (same material)
        1/R_star = 1/R_i + 1/R_j

    Tangential force (simplified Coulomb)::

        F_t = min(mu * |F_n|, gamma_t * |v_t|) * t_hat

    Parameters
    ----------
    E : float
        Young's modulus [Pa]. Default: 70e9 (silica glass).
    nu : float
        Poisson's ratio. Default: 0.22 (silica glass).
    gamma_n : float
        Normal damping coefficient [N s/m].
    mu_friction : float
        Coulomb friction coefficient.
    gamma_t : float
        Tangential damping coefficient [N s/m].
    """

    def __init__(
        self,
        E: float = 70e9,
        nu: float = 0.22,
        gamma_n: float = 1e-4,
        mu_friction: float = 0.3,
        gamma_t: float = 1e-5,
    ):
        self.E = E
        self.nu = nu
        self.gamma_n = gamma_n
        self.mu_friction = mu_friction
        self.gamma_t = gamma_t

    def name(self) -> str:
        return "hertz"

    def compute(self, contact: Contact) -> ContactForceResult:
        R_i, R_j = contact.p_i.radius, contact.p_j.radius
        R_star = (R_i * R_j) / (R_i + R_j)
        E_star = self.E / (2.0 * (1.0 - self.nu**2))

        dim = len(contact.n_ij)

        # Normal force (Hertz elastic + viscous damping)
        k_n = (4.0 / 3.0) * E_star * np.sqrt(R_star)
        F_n_mag = k_n * contact.delta_n**1.5 + self.gamma_n * contact.v_n
        F_n = -F_n_mag * contact.n_ij  # repulsive: pushes i away from j

        # Tangential force (Coulomb friction limit)
        v_t_mag = float(np.linalg.norm(contact.v_t))
        if v_t_mag > 1e-30:
            t_hat = contact.v_t / v_t_mag
            F_t_mag = min(
                self.mu_friction * abs(F_n_mag), self.gamma_t * v_t_mag
            )
            F_t = -F_t_mag * t_hat  # opposes relative sliding
        else:
            F_t = np.zeros(dim)

        # Torques from tangential force
        torque_i, torque_j = _compute_torques(
            contact, F_t, dim
        )

        return ContactForceResult(
            F_n=F_n, F_t=F_t, torque_i=torque_i, torque_j=torque_j
        )


class LinearSpringDashpot(ContactForceModel):
    """Linear spring-dashpot contact model (Cundall & Strack).

    Normal force::

        F_n = -k_n * delta_n - c_n * v_n

    Tangential force::

        F_t = -min(mu * |F_n|, c_t * |v_t|) * t_hat

    Parameters
    ----------
    k_n : float
        Normal spring stiffness [N/m].
    c_n : float
        Normal damping coefficient [N s/m].
    k_t : float
        Tangential spring stiffness [N/m] (unused in current dashpot-only
        tangential model; reserved for incremental spring extension).
    c_t : float
        Tangential damping coefficient [N s/m].
    mu_friction : float
        Coulomb friction coefficient.
    """

    def __init__(
        self,
        k_n: float = 1e5,
        c_n: float = 1e2,
        k_t: float = 5e4,
        c_t: float = 1e1,
        mu_friction: float = 0.3,
    ):
        self.k_n = k_n
        self.c_n = c_n
        self.k_t = k_t
        self.c_t = c_t
        self.mu_friction = mu_friction

    def name(self) -> str:
        return "linear_spring_dashpot"

    def compute(self, contact: Contact) -> ContactForceResult:
        dim = len(contact.n_ij)

        # Normal: linear spring + dashpot
        F_n_mag = self.k_n * contact.delta_n + self.c_n * contact.v_n
        F_n = -F_n_mag * contact.n_ij

        # Tangential: dashpot limited by Coulomb
        v_t_mag = float(np.linalg.norm(contact.v_t))
        if v_t_mag > 1e-30:
            t_hat = contact.v_t / v_t_mag
            F_t_mag = min(
                self.mu_friction * abs(F_n_mag), self.c_t * v_t_mag
            )
            F_t = -F_t_mag * t_hat
        else:
            F_t = np.zeros(dim)

        torque_i, torque_j = _compute_torques(
            contact, F_t, dim
        )

        return ContactForceResult(
            F_n=F_n, F_t=F_t, torque_i=torque_i, torque_j=torque_j
        )


def _compute_torques(
    contact: Contact, F_t: np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute torques on both particles from tangential contact force."""
    r_i = contact.x_contact - contact.p_i.x_a[:dim]
    r_j = contact.x_contact - contact.p_j.x_a[:dim]
    if dim == 3:
        torque_i = np.cross(r_i, F_t)
        torque_j = np.cross(r_j, -F_t)
    else:
        # 2D: torque is scalar (z-component of cross product)
        torque_i = np.array([r_i[0] * F_t[1] - r_i[1] * F_t[0]])
        torque_j = np.array([r_j[0] * (-F_t[1]) - r_j[1] * (-F_t[0])])
    return torque_i, torque_j


# Registry (follows ddgclib.operators._registry.MethodRegistry)
contact_force_registry = MethodRegistry("contact_force")
contact_force_registry.register("hertz", HertzContact)
contact_force_registry.register("linear_spring_dashpot", LinearSpringDashpot)
