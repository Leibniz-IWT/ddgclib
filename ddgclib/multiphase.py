"""Multiphase data model for discrete Lagrangian FVM simulations.

Provides phase assignment, interface identification, per-phase field
storage, and dual volume splitting for multiphase fluid simulations
on simplicial complexes.

Per-vertex data model
---------------------
Every vertex ``v`` in ``HC.V`` carries:

- ``v.phase`` — integer phase ID (0, 1, ..., n_phases-1).
  For bulk vertices this is the single phase; for interface vertices
  this is the **inner** (droplet) phase by convention.
- ``v.is_interface`` — ``True`` only for sharp-interface vertices
  (droplet-phase vertices on the circle/sphere that neighbour the
  outer phase).
- ``v.interface_phases`` — frozenset of phase IDs present at this vertex.

Per-phase arrays (length ``n_phases``, set by ``init_phase_fields``):

- ``v.m_phase[k]``        — mass of phase *k* in this dual cell
- ``v.p_phase[k]``        — pressure of phase *k*
- ``v.rho_phase[k]``      — density of phase *k*
- ``v.dual_vol_phase[k]`` — dual cell sub-volume belonging to phase *k*

For bulk vertices only ``v.*_phase[v.phase]`` is non-zero.
For interface vertices multiple entries are non-zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ddgclib.eos._base import EquationOfState


@dataclass
class PhaseProperties:
    """Material properties for a single phase.

    Parameters
    ----------
    eos : EquationOfState
        Equation of state mapping density to pressure.
    mu : float
        Dynamic viscosity [Pa.s].
    rho0 : float
        Reference density [kg/m^3].
    name : str
        Human-readable phase name.
    """
    eos: EquationOfState
    mu: float
    rho0: float
    name: str = ""


class MultiphaseSystem:
    """Container for multiphase configuration and vertex-level phase state.

    Parameters
    ----------
    phases : list[PhaseProperties]
        Phase properties indexed by phase ID (0..n-1).
    gamma : dict[tuple[int, int], float]
        Interfacial tension [N/m] for each phase pair.  Keys are
        ``(phase_i, phase_j)`` with ``phase_i < phase_j``.
    """

    def __init__(
        self,
        phases: list[PhaseProperties],
        gamma: dict[tuple[int, int], float] | None = None,
    ):
        self.phases = phases
        self.n_phases = len(phases)
        self._gamma: dict[tuple[int, int], float] = {}
        if gamma is not None:
            for (i, j), g in gamma.items():
                key = (min(i, j), max(i, j))
                self._gamma[key] = g

    # -- Phase assignment ----------------------------------------------------

    def assign_phases(
        self, HC, criterion_fn: Callable[[np.ndarray], int],
    ) -> None:
        """Set ``v.phase`` on all vertices based on spatial position."""
        for v in HC.V:
            v.phase = int(criterion_fn(v.x_a))

    # -- Interface identification -------------------------------------------

    def identify_interface(self, HC, interface_phase: int = 1) -> set:
        """Find **sharp** interface vertices.

        The sharp interface consists of vertices that:
        1. Belong to ``interface_phase`` (the droplet / inner phase), AND
        2. Have at least one neighbour in a different phase.

        Only these vertices carry interface physics (surface tension,
        dual-phase mass/pressure).  Neighbouring outer-phase vertices
        are pure bulk — never ``is_interface``.

        Parameters
        ----------
        HC : Complex
        interface_phase : int
            Phase ID whose boundary IS the interface (typically 1).

        Returns
        -------
        set
            The set of sharp interface vertex objects.
        """
        interface = set()
        for v in HC.V:
            neighbor_phases = {nb.phase for nb in v.nn}
            cross_phase = neighbor_phases - {v.phase}
            if v.phase == interface_phase and cross_phase:
                v.is_interface = True
                v.interface_phases = frozenset(neighbor_phases | {v.phase})
                interface.add(v)
            else:
                v.is_interface = False
                v.interface_phases = frozenset({v.phase})
        return interface

    # -- Per-phase field initialisation -------------------------------------

    def init_phase_fields(self, HC) -> None:
        """Initialise per-phase arrays on every vertex.

        Sets ``v.m_phase``, ``v.p_phase``, ``v.rho_phase``, and
        ``v.dual_vol_phase`` to zero arrays of length ``n_phases``.
        Must be called before ``split_dual_volumes`` or
        ``compute_phase_pressures``.
        """
        n = self.n_phases
        for v in HC.V:
            v.m_phase = np.zeros(n)
            v.p_phase = np.zeros(n)
            v.rho_phase = np.zeros(n)
            v.dual_vol_phase = np.zeros(n)

    # -- Dual volume splitting ----------------------------------------------

    def split_dual_volumes(self, HC, dim: int) -> None:
        """Split each vertex's dual cell volume among phases.

        **Bulk vertices**: the entire dual volume belongs to the
        vertex's own phase.

        **Interface vertices**: the dual cell straddles two (or more)
        phases.  The split is approximated by the fraction of 1-ring
        neighbours in each phase (including the vertex itself).

        Sets ``v.dual_vol_phase[k]`` for every vertex.

        .. note::
           This is an approximation.  A future improvement will compute
           the exact geometric split of the dual polygon/polyhedron by
           the interface curve/surface.
        """
        n = self.n_phases
        for v in HC.V:
            vol = getattr(v, 'dual_vol', 0.0)
            if not getattr(v, 'is_interface', False):
                # Bulk: entire volume is one phase
                v.dual_vol_phase = np.zeros(n)
                v.dual_vol_phase[v.phase] = vol
            else:
                # Interface: approximate split from neighbour counts
                counts = np.zeros(n)
                counts[v.phase] += 1.0
                for nb in v.nn:
                    counts[nb.phase] += 1.0
                total = counts.sum()
                fracs = counts / total if total > 0 else np.zeros(n)
                v.dual_vol_phase = fracs * vol

    # -- Per-phase mass -----------------------------------------------------

    def compute_phase_masses(self, HC) -> None:
        """Set per-phase mass from per-phase density and sub-volume.

        For each phase *k*:
            ``v.m_phase[k] = rho0_k * v.dual_vol_phase[k]``

        The total mass ``v.m`` is set to the sum.

        Requires ``split_dual_volumes`` to have been called.
        """
        for v in HC.V:
            for k in range(self.n_phases):
                vol_k = v.dual_vol_phase[k]
                if vol_k > 1e-30:
                    v.m_phase[k] = self.phases[k].rho0 * vol_k
                else:
                    v.m_phase[k] = 0.0
            v.m = float(np.sum(v.m_phase))

    # -- Per-phase pressure -------------------------------------------------

    def compute_phase_pressures(self, HC) -> None:
        """Compute per-phase pressures from per-phase density.

        For each phase *k* present at vertex *v*:
            ``rho_k = m_k / dual_vol_phase_k``
            ``p_k   = eos_k.pressure(rho_k)``

        Sets ``v.p_phase[k]``, ``v.rho_phase[k]``, and ``v.p``
        (which stores the **inner-phase** pressure for interface
        vertices, or the single-phase pressure for bulk vertices).
        """
        for v in HC.V:
            for k in range(self.n_phases):
                vol_k = v.dual_vol_phase[k]
                m_k = v.m_phase[k]
                if vol_k > 1e-30 and m_k > 1e-30:
                    rho_k = m_k / vol_k
                    v.rho_phase[k] = rho_k
                    v.p_phase[k] = float(self.phases[k].eos.pressure(rho_k))
                else:
                    v.rho_phase[k] = 0.0
                    v.p_phase[k] = 0.0

            # v.p stores the pressure of the vertex's own phase
            # (inner/droplet for interface vertices)
            v.p = v.p_phase[v.phase]

    # -- Viscosity lookup ---------------------------------------------------

    def get_mu(self, phase_id: int) -> float:
        """Return viscosity for a single phase."""
        return self.phases[phase_id].mu

    # -- Surface tension lookup ---------------------------------------------

    def get_gamma(self, v_i, v_j) -> float:
        """Interfacial tension coefficient for the phase pair.

        Returns 0 if both vertices are in the same phase.
        """
        if v_i.phase == v_j.phase:
            return 0.0
        key = (min(v_i.phase, v_j.phase), max(v_i.phase, v_j.phase))
        return self._gamma.get(key, 0.0)

    def get_gamma_pair(self, phase_a: int, phase_b: int) -> float:
        """Interfacial tension between two phase IDs."""
        if phase_a == phase_b:
            return 0.0
        key = (min(phase_a, phase_b), max(phase_a, phase_b))
        return self._gamma.get(key, 0.0)

    # -- Full refresh -------------------------------------------------------

    def refresh(self, HC, dim: int, reset_mass: bool = True) -> None:
        """One-call refresh of all multiphase state.

        Calls ``identify_interface``, ``init_phase_fields``,
        ``split_dual_volumes``, optionally ``compute_phase_masses``,
        and ``compute_phase_pressures`` in the correct order.

        Parameters
        ----------
        HC : Complex
        dim : int
        reset_mass : bool
            If ``True`` (default), recompute per-phase mass from
            ``rho0 * dual_vol_phase``.  Set to ``False`` during
            simulation (after initialisation) to preserve Lagrangian
            mass.  When ``False``, ``init_phase_fields`` preserves
            existing ``m_phase`` values and only resets geometry /
            pressure arrays.
        """
        self.identify_interface(HC)
        if reset_mass:
            self.init_phase_fields(HC)
        else:
            self._reinit_geometry_fields(HC)
        self.split_dual_volumes(HC, dim)
        if reset_mass:
            self.compute_phase_masses(HC)
        self.compute_phase_pressures(HC)

    def _reinit_geometry_fields(self, HC) -> None:
        """Re-initialise geometry/pressure arrays, preserving mass.

        Called by ``refresh(reset_mass=False)`` during simulation.
        Resets ``dual_vol_phase``, ``p_phase``, ``rho_phase`` but
        keeps ``m_phase`` and ``v.m`` unchanged.
        """
        n = self.n_phases
        for v in HC.V:
            v.p_phase = np.zeros(n)
            v.rho_phase = np.zeros(n)
            v.dual_vol_phase = np.zeros(n)
            # Preserve m_phase — if it doesn't exist yet, init to zero
            if not hasattr(v, 'm_phase') or v.m_phase is None:
                v.m_phase = np.zeros(n)

    # -- Convenience --------------------------------------------------------

    def phase_vertices(self, HC, phase_id: int) -> set:
        """Return all vertices belonging to a given phase."""
        return {v for v in HC.V if v.phase == phase_id}

    def interface_vertices(self, HC) -> set:
        """Return the set of interface vertices."""
        return {v for v in HC.V if getattr(v, 'is_interface', False)}

    def __repr__(self) -> str:
        names = [p.name or f"phase_{i}" for i, p in enumerate(self.phases)]
        return f"MultiphaseSystem(phases={names}, gamma={self._gamma})"


# ---------------------------------------------------------------------------
# Utility: mass-conserving vertex merge
# ---------------------------------------------------------------------------

def mass_conserving_merge(HC, cdist: float = 1e-10) -> int:
    """Merge near-duplicate vertices while conserving total mass.

    For each pair of vertices within *cdist*, the surviving vertex
    receives the sum of both masses.  Other scalar fields (``p``) are
    averaged; vector fields (``u``) are mass-weighted averaged.

    Parameters
    ----------
    HC : Complex
        Simplicial complex (modified in-place).
    cdist : float
        Merge tolerance.

    Returns
    -------
    int
        Number of vertices removed.
    """
    verts = list(HC.V)
    coords = np.array([v.x_a for v in verts])
    dim = coords.shape[1]

    from scipy.spatial import cKDTree
    tree = cKDTree(coords[:, :dim])
    pairs = tree.query_pairs(cdist)
    if not pairs:
        return 0

    # Union-find
    parent = list(range(len(verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(verts)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    n_removed = 0
    for _root, members in groups.items():
        if len(members) < 2:
            continue
        survivor = verts[members[0]]
        to_remove = [verts[m] for m in members[1:]]

        # Lump mass
        total_mass = sum(verts[m].m for m in members)
        total_momentum = sum(
            verts[m].m * verts[m].u for m in members
            if hasattr(verts[m], 'u')
        )
        avg_p = np.mean([getattr(verts[m], 'p', 0.0) for m in members])

        survivor.m = total_mass
        if total_mass > 0 and hasattr(survivor, 'u'):
            survivor.u = total_momentum / total_mass
        survivor.p = avg_p

        for v in to_remove:
            for nb in list(v.nn):
                if nb is not survivor:
                    if nb not in survivor.nn:
                        survivor.connect(nb)
                v.disconnect(nb)
            HC.V.remove(v)
            n_removed += 1

    return n_removed
