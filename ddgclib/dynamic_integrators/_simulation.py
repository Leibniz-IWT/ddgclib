"""
DynamicSimulation: optional convenience runner for dynamic simulations.

Bundles mesh, boundary conditions, initial conditions, integrator, and
parameters into a single object.  Cases can use this OR call integrators
directly — it is purely a convenience layer.

Usage
-----
    from ddgclib.dynamic_integrators._simulation import DynamicSimulation, SimulationParams
    from ddgclib.operators.gradient import acceleration

    sim = DynamicSimulation(HC, bV, params=SimulationParams(dt=1e-4, dim=2, mu=1e-3))
    sim.set_initial_conditions(CompositeIC(ZeroVelocity(2), UniformPressure(0.0)))
    sim.set_boundary_conditions(bc_set)
    sim.set_integrator(euler_velocity_only)
    sim.set_acceleration_fn(acceleration)
    t_final = sim.run(callback=my_callback)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ddgclib.dynamic_integrators._integrators_dynamic import euler_velocity_only


@dataclass
class SimulationParams:
    """Parameters for a dynamic simulation.

    Attributes
    ----------
    dt : float
        Time step size.
    n_steps : int
        Number of time steps (used by fixed-step integrators).
    t_end : float or None
        End time (used by adaptive integrators). Overrides n_steps if set.
    dim : int
        Spatial dimension.
    mu : float
        Dynamic viscosity.
    rho : float
        Density.
    extra : dict
        Additional keyword arguments forwarded to the acceleration function.
    """
    dt: float = 1e-4
    n_steps: int = 100
    t_end: Optional[float] = None
    dim: int = 3
    mu: float = 8.9e-4
    rho: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def dudt_kwargs(self) -> dict:
        """Build keyword arguments for dudt_fn from params."""
        kw = {'dim': self.dim, 'mu': self.mu}
        kw.update(self.extra)
        return kw


class DynamicSimulation:
    """Convenience runner that bundles all simulation components.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (mesh).
    bV : set
        Boundary vertex set.
    params : SimulationParams
        Simulation parameters.
    """

    def __init__(self, HC, bV: set, params: Optional[SimulationParams] = None):
        self.HC = HC
        self.bV = bV
        self.params = params if params is not None else SimulationParams()
        self._ic = None
        self._bc_set = None
        self._integrator = euler_velocity_only
        self._dudt_fn = None
        self.t_final = 0.0

    def set_initial_conditions(self, ic) -> 'DynamicSimulation':
        """Set initial conditions (an InitialCondition object).

        Returns self for chaining.
        """
        self._ic = ic
        return self

    def set_boundary_conditions(self, bc_set) -> 'DynamicSimulation':
        """Set boundary condition set (a BoundaryConditionSet object).

        Returns self for chaining.
        """
        self._bc_set = bc_set
        return self

    def set_integrator(self, integrator_fn: Callable) -> 'DynamicSimulation':
        """Set the time integration function.

        The function should have the same signature as
        :func:`euler_velocity_only` (or :func:`euler`, etc.).

        Returns self for chaining.
        """
        self._integrator = integrator_fn
        return self

    def set_acceleration_fn(self, dudt_fn: Callable) -> 'DynamicSimulation':
        """Set the acceleration (du/dt) function.

        Returns self for chaining.
        """
        self._dudt_fn = dudt_fn
        return self

    def run(self, callback: Optional[Callable] = None) -> float:
        """Run the simulation.

        1. Apply initial conditions (if set)
        2. Run integrator with BC enforcement
        3. Return final time

        Parameters
        ----------
        callback : callable or None
            Passed to the integrator.

        Returns
        -------
        float
            Final simulation time.

        Raises
        ------
        ValueError
            If no acceleration function has been set.
        """
        if self._dudt_fn is None:
            raise ValueError(
                "No acceleration function set. Call sim.set_acceleration_fn() "
                "before running."
            )

        # Step 1: Apply initial conditions
        if self._ic is not None:
            self._ic.apply(self.HC, self.bV)

        # Step 2: Determine integrator arguments
        p = self.params
        integrator_kwargs = {
            'HC': self.HC,
            'bV': self.bV,
            'dudt_fn': self._dudt_fn,
            'dt': p.dt,
            'dim': p.dim,
            'callback': callback,
            'bc_set': self._bc_set,
        }
        integrator_kwargs.update(p.dudt_kwargs)

        # Use n_steps or t_end depending on integrator
        from ddgclib.dynamic_integrators._integrators_dynamic import euler_adaptive
        if self._integrator is euler_adaptive:
            # Adaptive integrator uses dt_initial + t_end
            integrator_kwargs['dt_initial'] = p.dt
            del integrator_kwargs['dt']
            integrator_kwargs['t_end'] = p.t_end if p.t_end else p.n_steps * p.dt
        else:
            integrator_kwargs['n_steps'] = p.n_steps

        self.t_final = self._integrator(**integrator_kwargs)
        return self.t_final
