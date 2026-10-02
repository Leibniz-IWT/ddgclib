"""``SolverMethods``: one explicit, validated, serialisable record of every
method choice a dynamic run makes, plus builders that turn it into the
objects the integrators consume.

Design rule
-----------
This is a NAMING + PLUMBING + RECORDING layer.  The builders produce
exactly the ``functools.partial`` objects and integrator kwargs the case
runners built by hand before (see ``ddgclib/tests/test_methods.py`` for
the bit-identity proofs), so no pinned baseline moves.  Choices that the
operator layer still makes implicitly (by dimension, by cache presence)
are not changed here; :func:`ddgclib.methods.effective_methods` resolves
and records them.

Methods vs. geometry / physics
------------------------------
Fields of :class:`SolverMethods` are METHOD choices only.  Geometry and
physics objects (``HC``, ``mps``, ``domain_bounds``, ``mu``, the EOS, a
custom retopology callable, a ``boundary_filter``) are passed to the
builders at build time.

Typical use in a case runner::

    from ddgclib.methods import PRESETS, record_methods

    methods = PRESETS['oscillating_droplet_2D']
    HC, bV, mps, bc_set, dudt_fn, _, params = setup_oscillating_droplet(
        dim=2, ..., split_method=methods.split_method,
        redistribute_mass=methods.redistribute_mass)
    t_final = methods.integrate(HC, bV, dudt_fn, dt=dt, n_steps=n_steps,
                                bc_set=bc_set, callback=callback, mps=mps)
    record_methods('results/methods.json', methods, HC,
                   extra={'dt': dt, 'n_steps': n_steps})
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ddgclib.methods._axes import AXES, MethodOption

__all__ = ['SolverMethods']

# Fields that are only meaningful for the multiphase pipeline.  For a
# single-phase config they must stay at their defaults.
_MULTI_ONLY = ('projection_every', 'split_method', 'curvature_path')

# Connectivity values whose retopology function runs the multiphase
# redistribution block (so remap / projection_every can apply).
_RECONNECTING = ('delaunay', 'adaptive', 'delaunay_material')

# Connectivity values for which frozen_set='membership' is implemented:
# bV is rebuilt from the hull of a NEW connectivity inside _retopologize
# and nothing else creates, moves or removes wall vertices.  'adaptive'
# also rebuilds the hull but is excluded: hyperct.remesh splits wall edges
# into vertices that are not members and collapses / smooths members that
# are off the hull (measured, laneL fix round 1).
_HULL_REBUILDING = ('delaunay',)


@dataclass(frozen=True)
class SolverMethods:
    """Every explicit method choice of a dynamic run.

    One field per explicit axis in :data:`ddgclib.methods.AXES`.  See
    that registry (or ``METHODS.md``) for the allowed values, their
    status and evidence.  Construction validates the combination and
    raises ``ValueError`` for combinations the code would otherwise
    silently ignore (e.g. a conservative remap under ``dual_only``).
    """

    dim: int
    phases: str = 'single'
    integrator: str = 'symplectic_euler'
    connectivity: str = 'delaunay'
    remap: str | None = None
    frozen_set: str = 'hull'
    projection_every: int = 1
    redistribute_mass: bool = False
    split_method: str = 'neighbour_count'
    curvature_path: str = 'integrated'
    pressure_flux: str = 'centred'
    viscous_flux: str = 'two_point'
    density_diffusion: float | None = None
    displacement_eps: float | None = None
    merge_cdist: float | None = None
    remesh_kwargs: dict[str, Any] | None = None
    periodic_axes: tuple[int, ...] | None = None
    backend: str | None = None
    workers: int | None = None
    label: str = ''
    notes: str = ''

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {self.dim!r}")
        self._check_choice('phases', self.phases)
        self._check_choice('integrator', self.integrator)
        self._check_choice('connectivity', self.connectivity)
        self._check_choice('remap', self.remap)
        self._check_choice('frozen_set', self.frozen_set)
        self._check_choice('split_method', self.split_method)
        self._check_choice('curvature_path', self.curvature_path)
        self._check_choice('pressure_flux', self.pressure_flux)
        self._check_choice('viscous_flux', self.viscous_flux)
        self._check_choice('backend', self.backend)

        if self.density_diffusion is not None and not self.density_diffusion > 0:
            raise ValueError("density_diffusion must be None or > 0")
        if self.phases == 'multi':
            if self.pressure_flux != AXES['pressure_flux'].default:
                raise ValueError("pressure_flux applies to phases='single' only "
                                 "(the multiphase force hard-codes the centred flux)")
            if self.viscous_flux != AXES['viscous_flux'].default:
                raise ValueError("viscous_flux applies to phases='single' only "
                                 "(the multiphase force hard-codes the two-point flux)")
            if self.density_diffusion is not None:
                raise ValueError("density_diffusion applies to phases='single' only")
        if self.connectivity == 'periodic' and 'simplex_gradient' in (
                self.pressure_flux, self.viscous_flux):
            raise ValueError(
                "the 'simplex_gradient' fluxes are not available under "
                "connectivity='periodic' (the seam simplices are cached with "
                "raw coordinates, laneP)")
        if self.density_diffusion is not None and self.integrator not in (
                'euler', 'symplectic_euler'):
            raise ValueError("density_diffusion is only implemented on the "
                             "euler / symplectic_euler integrators")

        if not isinstance(self.projection_every, int) or isinstance(
                self.projection_every, bool) or self.projection_every < 1:
            raise ValueError(
                f"projection_every must be an int >= 1, got "
                f"{self.projection_every!r}")
        if self.displacement_eps is not None and not self.displacement_eps > 0:
            raise ValueError("displacement_eps must be None or > 0")
        if self.merge_cdist is not None and not self.merge_cdist > 0:
            raise ValueError("merge_cdist must be None or > 0")
        if self.workers is not None and (
                not isinstance(self.workers, int) or self.workers < 1):
            raise ValueError("workers must be None or an int >= 1")
        if self.remesh_kwargs is not None and self.connectivity != 'adaptive':
            raise ValueError(
                "remesh_kwargs only applies to connectivity='adaptive'")

        multi = self.phases == 'multi'
        if not multi:
            for name in _MULTI_ONLY:
                default = AXES[name].default
                if getattr(self, name) != default:
                    raise ValueError(
                        f"{name}={getattr(self, name)!r} applies to "
                        f"phases='multi' only (single-phase keeps "
                        f"{default!r})")

        if self.connectivity == 'adaptive' and self.dim != 2:
            raise ValueError("connectivity='adaptive' is 2D only "
                             "(hyperct.remesh has no 3D operations)")
        if self.connectivity == 'periodic':
            if not self.periodic_axes:
                raise ValueError("connectivity='periodic' needs periodic_axes "
                                 "(domain_bounds is passed at build time)")
            if any(a not in range(self.dim) for a in self.periodic_axes):
                raise ValueError(f"periodic_axes {self.periodic_axes!r} "
                                 f"outside range({self.dim})")
        elif self.periodic_axes is not None:
            raise ValueError("periodic_axes requires connectivity='periodic'")

        if self.connectivity == 'dual_only_bare':
            # The bare refresh never touches masses, pressures or the
            # split policy: make every such request an error, not a no-op.
            if self.redistribute_mass:
                raise ValueError("connectivity='dual_only_bare' never "
                                 "redistributes mass (redistribute_mass must "
                                 "be False)")
            if self.split_method != AXES['split_method'].default:
                raise ValueError("connectivity='dual_only_bare' re-splits with "
                                 "the default split_method only")

        if self.connectivity == 'delaunay_material':
            # One retopology function, single phase only; it has no merge
            # step and offers redistribution only as part of the remap.
            if multi:
                raise ValueError("connectivity='delaunay_material' is "
                                 "single-phase only")
            if self.merge_cdist is not None:
                raise ValueError("merge_cdist is not applied by "
                                 "connectivity='delaunay_material'")
            if self.backend is not None:
                raise ValueError("backend is not applied by "
                                 "connectivity='delaunay_material' (its "
                                 "dual refresh is numpy only)")
            if self.redistribute_mass and self.remap != 'conservative':
                raise ValueError(
                    "connectivity='delaunay_material' redistributes mass "
                    "only as part of remap='conservative' (redistribution "
                    "against the stale v.p is a measured DO-NOT, laneK)")

        if (self.frozen_set == 'membership'
                and self.connectivity not in _HULL_REBUILDING):
            raise ValueError(
                f"frozen_set='membership' is not applied under "
                f"connectivity={self.connectivity!r}: it is implemented for "
                f"{', '.join(_HULL_REBUILDING)} only. 'dual_only', "
                f"'dual_only_bare' and 'frozen' never rebuild the hull, so "
                f"their bV is persistent already; 'adaptive' creates wall "
                f"vertices that are not members and remeshes members that "
                f"are off the hull")

        if self.remap == 'conservative':
            if not self.redistribute_mass:
                raise ValueError("remap='conservative' requires "
                                 "redistribute_mass=True")
            if self.connectivity not in _RECONNECTING:
                raise ValueError(
                    f"remap='conservative' is a silent no-op under "
                    f"connectivity={self.connectivity!r}; use 'delaunay', "
                    f"'adaptive' or 'delaunay_material', or set remap=None")
        if self.projection_every > 1:
            if not self.redistribute_mass:
                raise ValueError("projection_every > 1 requires "
                                 "redistribute_mass=True")
            if self.connectivity in _RECONNECTING and self.remap != 'conservative':
                raise ValueError(
                    "projection_every > 1 under bare reconnection re-opens "
                    "the lane-5 KE pump; it needs remap='conservative' or "
                    "connectivity='dual_only'")
            if self.connectivity not in _RECONNECTING + ('dual_only',):
                raise ValueError("projection_every > 1 needs a retopology "
                                 "path that runs the redistribution block "
                                 "(delaunay, adaptive or dual_only)")

        self._warn_broken()

    def _check_choice(self, axis: str, value: Any) -> None:
        ax = AXES[axis]
        try:
            opt = ax.option(value)
        except KeyError as e:
            raise ValueError(str(e)) from None
        if ax.applies_to == 'multi' and self.phases != 'multi':
            return  # inert default on a single-phase config
        if self.dim not in opt.dims:
            raise ValueError(
                f"{axis}={value!r} is not available in {self.dim}D "
                f"(dims: {opt.dims})")

    def _warn_broken(self) -> None:
        for name, value in self.explicit_items():
            opt = self._option(name, value)
            if opt is not None and opt.status == 'broken':
                warnings.warn(
                    f"SolverMethods: {name}={value!r} has status 'broken': "
                    f"{opt.evidence}", UserWarning, stacklevel=3)

    def _warn_single_phase_eos(self) -> None:
        """Measured-unstable single-phase + EOS combinations (lane K,
        docs_temp/debug_session/laneK-single-phase-eos-instability.md).
        Warnings, not errors: a lane may re-try them on purpose."""
        # dim 1: the 'delaunay' rebuild is the sorted chain, which cannot
        # flip (laneP: equal to a loop that never reconnects up to the
        # summation order of the force, 3e-17 in u after 128 steps).  Until
        # laneP this warning was also raised in 1D.
        if (self.dim > 1 and self.connectivity in _RECONNECTING
                and self.remap != 'conservative'):
            warnings.warn(
                f"SolverMethods: single-phase connectivity="
                f"{self.connectivity!r} with an EOS in dudt_fn and no remap is "
                "measured UNSTABLE (laneK: a Delaunay flip changes a dual "
                "volume by 33-100 %, read as 3e4-5e4 Pa of compression; blows "
                "up at any CFL / c_s / n, with or without redistribute_mass). "
                "Use remap='conservative' with redistribute_mass=True, or "
                "connectivity='dual_only'.", UserWarning, stacklevel=3)
        if self.integrator == 'euler':
            warnings.warn(
                "SolverMethods: integrator='euler' with an EOS grows at "
                "CFL 0.25 even on fixed connectivity (laneK); use "
                "'symplectic_euler'.", UserWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def explicit_items(self) -> list[tuple[str, Any]]:
        """(field, value) for every explicit axis, in registry order."""
        return [(a.name, getattr(self, a.name)) for a in AXES.values()
                if a.explicit and hasattr(self, a.name)]

    @staticmethod
    def _option(axis: str, value: Any) -> MethodOption | None:
        ax = AXES[axis]
        try:
            return ax.option(value)
        except KeyError:
            pass
        # numeric axes: pick the "N>1"/"eps>0" style option
        if ax.kind in ('int', 'float') and value is not None:
            for o in ax.options:
                if isinstance(o.key, str):
                    return o
        return None

    def status_of(self, axis: str) -> str:
        """Registry status of this config's value on *axis*."""
        opt = self._option(axis, getattr(self, axis))
        return opt.status if opt is not None else 'unknown'

    def describe(self) -> str:
        """Human-readable summary (one line per explicit axis)."""
        lines = [f"SolverMethods {self.label or ''}".rstrip()]
        lines.append(f"  {'dim':<18} = {self.dim!r}")
        for name, value in self.explicit_items():
            ax = AXES[name]
            if ax.applies_to == 'multi' and self.phases != 'multi':
                continue
            opt = self._option(name, value)
            status = opt.status if opt else ''
            summary = opt.summary if opt else ''
            lines.append(f"  {name:<18} = {value!r:<18} [{status}] {summary}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                v = list(v)
            d[f.name] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'SolverMethods':
        d = dict(d)
        if d.get('periodic_axes') is not None:
            d['periodic_axes'] = tuple(d['periodic_axes'])
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def from_json(cls, path: str | Path) -> 'SolverMethods':
        return cls.from_dict(json.loads(Path(path).read_text()))

    def replace(self, **changes: Any) -> 'SolverMethods':
        """Copy with some fields changed (re-validated)."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # builders (must stay bit-identical to the hand-written case code)
    # ------------------------------------------------------------------
    def retopologize_fn(self, mps=None, custom: Callable | None = None,
                        domain_bounds=None):
        """Value for the integrators' ``retopologize_fn`` kwarg.

        - ``'frozen'``          -> ``False``
        - ``'custom'``          -> *custom* (required)
        - ``'dual_only_bare'``  -> ``partial(bare_dual_refresh, mps=mps)``
        - ``'delaunay_material'``-> ``retopologize_material_delaunay`` (a
          partial binding ``retopo_remap`` when the remap is on)
        - ``'periodic'`` + multi-> ``partial(retopologize_multiphase_periodic, ...)``
          (*domain_bounds* required)
        - single-phase + remap  -> ``partial(_retopologize, retopo_remap=...)``
          (the integrator forwards its retopology kwargs to it by name)
        - other single-phase    -> ``None`` (library default ``_retopologize``;
          the policy is carried by :meth:`integrator_kwargs`)
        - other multiphase      -> ``partial(_retopologize_multiphase, ...)``
          with exactly the keywords the case runners bind by hand.

        ``frozen_set='membership'`` adds ``frozen_set=`` to the single-phase
        or multiphase partial (a single-phase config then always gets a
        ``partial(_retopologize, ...)``); the default ``'hull'`` binds
        nothing, so the objects above are unchanged.
        """
        if self.connectivity == 'frozen':
            return False
        if self.connectivity == 'custom':
            if custom is None:
                raise ValueError("connectivity='custom' needs a callable "
                                 "(pass custom=...)")
            return custom
        if custom is not None:
            raise ValueError("custom retopologize_fn given but connectivity "
                             f"is {self.connectivity!r}, not 'custom'")
        if self.connectivity == 'dual_only_bare':
            from ddgclib.methods._retopo import bare_dual_refresh
            return partial(bare_dual_refresh, mps=mps)
        if self.connectivity == 'delaunay_material':
            from ddgclib.methods._retopo import retopologize_material_delaunay
            if self.remap is None:
                return retopologize_material_delaunay
            return partial(retopologize_material_delaunay,
                           retopo_remap=self.remap)
        membership = self.frozen_set != AXES['frozen_set'].default
        if self.phases == 'single':
            if self.remap is None and not membership:
                return None
            from ddgclib.dynamic_integrators._integrators_dynamic import (
                _retopologize,
            )
            kw_s: dict[str, Any] = {}
            if self.remap is not None:
                kw_s['retopo_remap'] = self.remap
            if membership:
                kw_s['frozen_set'] = self.frozen_set
            return partial(_retopologize, **kw_s)
        if mps is None:
            raise ValueError("multiphase configs need mps=MultiphaseSystem")
        if self.connectivity == 'periodic':
            if domain_bounds is None:
                raise ValueError("periodic multiphase needs domain_bounds=")
            from ddgclib.methods._retopo import retopologize_multiphase_periodic
            return partial(
                retopologize_multiphase_periodic, mps=mps,
                periodic_axes=list(self.periodic_axes),
                domain_bounds=[tuple(b) for b in domain_bounds],
                split_method=self.split_method,
                redistribute_mass=self.redistribute_mass,
            )
        from ddgclib.dynamic_integrators._integrators_dynamic import (
            _retopologize_multiphase,
        )
        kw: dict[str, Any] = dict(
            mps=mps,
            split_method=self.split_method,
            redistribute_mass=self.redistribute_mass,
        )
        if self.connectivity == 'dual_only':
            kw['skip_triangulation'] = True
        if self.remap is not None:
            kw['retopo_remap'] = self.remap
        if self.projection_every != 1:
            kw['projection_every'] = self.projection_every
        if membership:
            kw['frozen_set'] = self.frozen_set
        return partial(_retopologize_multiphase, **kw)

    def integrator_kwargs(self, mps=None, custom: Callable | None = None,
                          boundary_filter: Callable | None = None,
                          pressure_model=None,
                          domain_bounds=None) -> dict[str, Any]:
        """Retopology / execution kwargs for the integrator call.

        *pressure_model* is only consumed by single-phase mass
        redistribution (the force operator gets its own copy through
        :meth:`dudt_fn`).  Multiphase redistribution reads the per-phase
        EOS from *mps*.  *domain_bounds* is required for
        ``connectivity='periodic'``.
        """
        kw: dict[str, Any] = {
            'retopologize_fn': self.retopologize_fn(
                mps=mps, custom=custom, domain_bounds=domain_bounds),
            'skip_triangulation': self.connectivity == 'dual_only',
            'remesh_mode': 'adaptive' if self.connectivity == 'adaptive'
                           else 'delaunay',
            'remesh_kwargs': self.remesh_kwargs,
            'merge_cdist': self.merge_cdist,
            'displacement_eps': self.displacement_eps,
            'backend': self.backend,
            'workers': self.workers,
            'boundary_filter': boundary_filter,
        }
        if self.connectivity == 'periodic' and self.phases == 'single':
            if domain_bounds is None:
                raise ValueError("connectivity='periodic' needs domain_bounds=")
            kw['periodic_axes'] = list(self.periodic_axes)
            kw['domain_bounds'] = [tuple(b) for b in domain_bounds]
        if self.phases == 'single':
            if self.redistribute_mass and pressure_model is None:
                raise ValueError("single-phase redistribute_mass=True needs "
                                 "pressure_model=EOS")
            kw['redistribute_mass'] = self.redistribute_mass
            kw['pressure_model'] = pressure_model
            if self.density_diffusion is not None:
                if not hasattr(pressure_model, 'sound_speed'):
                    raise ValueError("density_diffusion needs pressure_model=EOS")
                kw['density_diffusion'] = self.density_diffusion
        return kw

    def dudt_fn(self, HC, *, mu: float | None = None, mps=None,
                pressure_model=None,
                body_force: Any = None) -> Callable:
        """Acceleration function bound the canonical way.

        single-phase: ``partial(dudt_i, dim, mu, HC, pressure_model
        [, pressure_flux][, viscous_flux])``
        multiphase:   ``partial(multiphase_dudt_i, dim, mps, HC,
        pressure_model[, curvature_path])``

        *body_force* (per unit mass, e.g. ``[0, -9.81]``) wraps the
        result as ``a + g`` exactly like the dam-break setup does.
        """
        if self.phases == 'single':
            if mu is None:
                raise ValueError("single-phase dudt_fn needs mu=")
            if pressure_model is not None:
                self._warn_single_phase_eos()
            from ddgclib.operators.stress import dudt_i
            kw_s: dict[str, Any] = dict(dim=self.dim, mu=mu, HC=HC,
                                        pressure_model=pressure_model)
            if self.pressure_flux != AXES['pressure_flux'].default:
                if (self.pressure_flux == 'acoustic-riemann'
                        and not hasattr(pressure_model, 'sound_speed')):
                    raise ValueError(f"pressure_flux={self.pressure_flux!r} "
                                     "needs pressure_model=EOS")
                kw_s['pressure_flux'] = self.pressure_flux
            if self.viscous_flux != AXES['viscous_flux'].default:
                kw_s['viscous_flux'] = self.viscous_flux
            fn: Callable = partial(dudt_i, **kw_s)
        else:
            if mps is None:
                raise ValueError("multiphase dudt_fn needs mps=")
            from ddgclib.operators.multiphase_stress import multiphase_dudt_i
            kw: dict[str, Any] = dict(dim=self.dim, mps=mps, HC=HC,
                                      pressure_model=pressure_model)
            if self.curvature_path != 'integrated':
                kw['curvature_path'] = self.curvature_path
            fn = partial(multiphase_dudt_i, **kw)
        if body_force is None:
            return fn
        g_vec = np.asarray(body_force, dtype=float)
        if g_vec.shape != (self.dim,):
            raise ValueError(f"body_force must have exactly {self.dim} "
                             f"components, got shape {g_vec.shape}")
        _stress_fn = fn

        def dudt_with_body_force(v):
            return _stress_fn(v) + g_vec

        dudt_with_body_force.stress_fn = _stress_fn  # type: ignore[attr-defined]
        dudt_with_body_force.body_force = g_vec     # type: ignore[attr-defined]
        return dudt_with_body_force

    def integrate(self, HC, bV, dudt_fn, *, dt: float,
                  n_steps: int | None = None, t_end: float | None = None,
                  bc_set=None, callback: Callable | None = None,
                  mps=None, custom: Callable | None = None,
                  boundary_filter: Callable | None = None,
                  pressure_model=None, domain_bounds=None,
                  **extra: Any) -> float:
        """Run the configured integrator; returns the final time.

        *extra* is forwarded verbatim (``save_every``, ``rtol``,
        ``velocity_only``, extra ``dudt_fn`` kwargs, ...).  Fixed-step
        integrators need *n_steps*; ``euler_adaptive`` needs *t_end*
        (its *dt* is ``dt_initial``).
        """
        import ddgclib.dynamic_integrators as _di
        fn = getattr(_di, self.integrator)
        kw = self.integrator_kwargs(mps=mps, custom=custom,
                                    boundary_filter=boundary_filter,
                                    pressure_model=pressure_model,
                                    domain_bounds=domain_bounds)
        kw.update(extra)
        if self.integrator == 'euler_adaptive':
            if t_end is None:
                raise ValueError("euler_adaptive needs t_end=")
            return fn(HC, bV, dudt_fn, dt_initial=dt, t_end=t_end,
                      dim=self.dim, callback=callback, bc_set=bc_set, **kw)
        if n_steps is None:
            raise ValueError(f"{self.integrator} needs n_steps=")
        return fn(HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=self.dim,
                  callback=callback, bc_set=bc_set, **kw)
