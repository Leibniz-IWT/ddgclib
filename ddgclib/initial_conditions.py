"""
Abstracted initial conditions for dynamic continuum simulations.

Each IC is applied to a simplicial Complex (HC) by setting vertex attributes
(v.u, v.p, v.m, etc.) in-place. ICs can be composed via CompositeIC.

Usage
-----
    from ddgclib.initial_conditions import (
        CompositeIC, ZeroVelocity, HydrostaticPressure, UniformMass
    )

    ic = CompositeIC(
        ZeroVelocity(dim=3),
        HydrostaticPressure(rho=1000.0, g=9.81, axis=2, h_ref=1.0),
        UniformMass(total_volume=1.0, rho=1000.0),
    )
    ic.apply(HC, bV)
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np


class InitialCondition(ABC):
    """Abstract base for initial conditions on a simplicial complex."""

    @abstractmethod
    def apply(self, HC, bV: set) -> None:
        """Apply initial condition to all vertices in HC.

        Sets vertex attributes (v.u, v.p, v.m, etc.) in-place.
        Boundary vertices in bV may be treated differently by subclasses.
        """


class CompositeIC(InitialCondition):
    """Apply multiple ICs in sequence (e.g., velocity IC then pressure IC)."""

    def __init__(self, *ics: InitialCondition):
        self.ics = ics

    def apply(self, HC, bV: set) -> None:
        for ic in self.ics:
            ic.apply(HC, bV)


# Scalar field ICs (pressure)

class UniformPressure(InitialCondition):
    """Set P = P0 (scalar) on all vertices."""

    def __init__(self, P0: float = 0.0):
        self.P0 = P0

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.p = self.P0


class HydrostaticPressure(InitialCondition):
    """Set P = P_ref + rho * g * (h_ref - x[axis]).

    Parameters
    ----------
    rho : float
        Fluid density [kg/m^3].
    g : float
        Gravitational acceleration magnitude [m/s^2].
    axis : int
        Coordinate axis aligned with gravity (0=x, 1=y, 2=z).
    h_ref : float
        Reference height where P = P_ref.
    P_ref : float
        Reference pressure at h_ref.
    """

    def __init__(self, rho: float, g: float = 9.81,
                 axis: int = 2, h_ref: float = 0.0,
                 P_ref: float = 0.0):
        self.rho = rho
        self.g = g
        self.axis = axis
        self.h_ref = h_ref
        self.P_ref = P_ref

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.p = self.P_ref + self.rho * self.g * (self.h_ref - v.x_a[self.axis])


class LinearPressureGradient(InitialCondition):
    """Set P = P_ref - G * x[axis].

    Models a constant pressure gradient along the flow axis,
    as in fully-developed Poiseuille flow.

    Parameters
    ----------
    G : float
        Pressure gradient magnitude (positive = pressure decreases along axis).
    axis : int
        Flow axis (0=x, 1=y, 2=z).
    P_ref : float
        Reference pressure at x[axis] = 0.
    """

    def __init__(self, G: float, axis: int = 0, P_ref: float = 0.0):
        self.G = G
        self.axis = axis
        self.P_ref = P_ref

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.p = self.P_ref - self.G * v.x_a[self.axis]


# Vector field ICs (velocity)

class ZeroVelocity(InitialCondition):
    """Set u = 0 on all vertices."""

    def __init__(self, dim: int = 3):
        self.dim = dim

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.u = np.zeros(self.dim)


class UniformVelocity(InitialCondition):
    """Set u = u_vec on all vertices."""

    def __init__(self, u_vec):
        self.u_vec = np.asarray(u_vec, dtype=float)

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.u = self.u_vec.copy()


class PoiseuillePlanar(InitialCondition):
    """Planar Poiseuille: u_flow(y) = (G / 2mu) * (y - y_lb) * (y_ub - y).

    Analytical velocity profile for 2D channel flow between parallel plates.

    Parameters
    ----------
    G : float
        Pressure gradient magnitude.
    mu : float
        Dynamic viscosity.
    y_lb, y_ub : float
        Lower and upper plate positions along normal_axis.
    flow_axis : int
        Axis of the flow velocity component.
    normal_axis : int
        Axis perpendicular to the plates.
    dim : int
        Spatial dimension.
    """

    def __init__(self, G: float, mu: float,
                 y_lb: float = 0.0, y_ub: float = 1.0,
                 flow_axis: int = 0, normal_axis: int = 1,
                 dim: int = 2):
        self.G = G
        self.mu = mu
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.flow_axis = flow_axis
        self.normal_axis = normal_axis
        self.dim = dim

    def analytical_velocity(self, x_a: np.ndarray) -> float:
        """Return the analytical flow-axis velocity at position x_a."""
        y = x_a[self.normal_axis]
        return (self.G / (2 * self.mu)) * (y - self.y_lb) * (self.y_ub - y)

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.u = np.zeros(self.dim)
            v.u[self.flow_axis] = self.analytical_velocity(v.x_a)


class HagenPoiseuille3D(InitialCondition):
    """3D tube Poiseuille: u_z(r) = U_max * (1 - (r/R)^2).

    Analytical velocity profile for fully-developed pipe flow.

    Parameters
    ----------
    U_max : float
        Centerline velocity.
    R : float
        Tube radius.
    flow_axis : int
        Axis of flow (default 2 = z).
    dim : int
        Spatial dimension (default 3).
    """

    def __init__(self, U_max: float, R: float,
                 flow_axis: int = 2, dim: int = 3):
        self.U_max = U_max
        self.R = R
        self.flow_axis = flow_axis
        self.dim = dim

    def analytical_velocity(self, x_a: np.ndarray) -> float:
        """Return the analytical axial velocity at position x_a."""
        radial_axes = [i for i in range(self.dim) if i != self.flow_axis]
        r = np.linalg.norm([x_a[ax] for ax in radial_axes])
        return self.U_max * max(0.0, 1.0 - (r / self.R) ** 2)

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            v.u = np.zeros(self.dim)
            v.u[self.flow_axis] = self.analytical_velocity(v.x_a)


# Custom / generic ICs

class CustomFieldIC(InitialCondition):
    """User-defined IC via callable fn(x_a) -> value.

    Parameters
    ----------
    fn : callable
        Function mapping vertex position (numpy array) to field value.
    field_name : str
        Vertex attribute to set (e.g., 'u', 'p', 'm').
    """

    def __init__(self, fn: Callable[[np.ndarray], Union[float, np.ndarray]],
                 field_name: str = 'u'):
        self.fn = fn
        self.field_name = field_name

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            setattr(v, self.field_name, self.fn(v.x_a))


class UniformMass(InitialCondition):
    """Distribute total mass uniformly across all vertices.

    Parameters
    ----------
    total_volume : float
        Total domain volume (used to compute total mass = rho * V).
    rho : float
        Fluid density.
    """

    def __init__(self, total_volume: float, rho: float = 1.0):
        self.total_volume = total_volume
        self.rho = rho

    def apply(self, HC, bV: set) -> None:
        n_verts = sum(1 for _ in HC.V)
        if n_verts == 0:
            return
        mass_per_vert = self.rho * self.total_volume / n_verts
        for v in HC.V:
            v.m = mass_per_vert


class DualVolumeMass(InitialCondition):
    """Set mass proportional to dual cell volume: m_i = rho * dual_vol_i.

    Unlike :class:`UniformMass`, this ensures that the local density
    ``rho_i = m_i / dual_vol_i`` equals *rho* exactly for every vertex,
    including boundary vertices with half-cells.  This is essential for
    EOS-coupled simulations where pressure is derived from density.

    Requires ``compute_vd`` and ``cache_dual_volumes`` to have been
    called before applying this IC.

    Parameters
    ----------
    rho : float
        Target fluid density [kg/m^3].
    """

    def __init__(self, rho: float = 1000.0):
        self.rho = rho

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            dual_vol = getattr(v, 'dual_vol', None)
            if dual_vol is None or dual_vol < 1e-30:
                v.m = self.rho * 1e-30
            else:
                v.m = self.rho * dual_vol


class HydrostaticEOSMass(InitialCondition):
    """Set mass so that EOS pressure matches the **compressible** hydrostatic profile.

    Solves the compressible hydrostatic ODE ``dP/dz = -rho(P) * g``
    downward from the free surface at ``h_ref`` where ``P = P_ref``.
    The density at each depth is ``rho(P) = eos.density(P)``, so the
    pressure and density are thermodynamically consistent.

    For a linear EOS (n=1, ``P = P0 + K*(rho/rho0 - 1)``), the
    closed-form solution is:

        P(z) = P_ref + K * (exp(rho0 * g * (h_ref - z) / K) - 1)

    For general EOS, the ODE is integrated numerically via
    ``scipy.integrate.solve_ivp``.

    Requires ``compute_vd`` and ``cache_dual_volumes`` to have been
    called before applying this IC.

    Parameters
    ----------
    eos : EquationOfState
        Equation of state with ``pressure(rho)`` and ``density(P)`` methods.
    rho0 : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    gravity_axis : int
        Coordinate axis aligned with gravity.
    h_ref : float
        Height where P = P_ref (free surface).
    P_ref : float
        Reference pressure at h_ref.
    """

    def __init__(self, eos, rho0: float = 1000.0, g: float = 9.81,
                 gravity_axis: int = 1, h_ref: float = 1.0,
                 P_ref: float = 0.0):
        self.eos = eos
        self.rho0 = rho0
        self.g = g
        self.gravity_axis = gravity_axis
        self.h_ref = h_ref
        self.P_ref = P_ref

    def _compressible_pressure(self, z: float) -> float:
        """Compressible hydrostatic pressure at height z.

        Uses closed-form for linear EOS (n=1), ODE for general EOS.
        """
        depth = self.h_ref - z  # depth below free surface
        if depth <= 0:
            return self.P_ref

        # Check for linear EOS (n=1): closed-form exponential solution
        n = getattr(self.eos, 'n', None)
        K = getattr(self.eos, 'K', None)
        if n is not None and K is not None and abs(n - 1.0) < 1e-12:
            alpha = self.rho0 * self.g / K
            return self.P_ref + K * (np.exp(alpha * depth) - 1.0)

        # General EOS: integrate dP/dz = -rho(P)*g from h_ref downward
        from scipy.integrate import solve_ivp

        def dP_dz(z_coord, P_arr):
            rho_local = float(self.eos.density(P_arr[0]))
            return [-rho_local * self.g]  # dP/dz = -rho*g (z decreasing)

        sol = solve_ivp(dP_dz, [self.h_ref, z], [self.P_ref],
                        rtol=1e-12, atol=1e-12)
        return float(sol.y[0, -1])

    def apply(self, HC, bV: set) -> None:
        for v in HC.V:
            y = v.x_a[self.gravity_axis]
            P_target = self._compressible_pressure(y)
            rho_eq = float(self.eos.density(P_target))
            dual_vol = getattr(v, 'dual_vol', None)
            if dual_vol is None or dual_vol < 1e-30:
                v.m = rho_eq * 1e-30
            else:
                v.m = rho_eq * dual_vol
            v.rho = rho_eq
            v.p = P_target
