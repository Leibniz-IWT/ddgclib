"""Multiphase data model for discrete Lagrangian FVM simulations.

Provides phase assignment, interface identification, per-phase field
storage, and dual volume splitting for multiphase fluid simulations
on simplicial complexes.

Per-vertex data model
---------------------
Every vertex ``v`` in ``HC.V`` carries:

- ``v.phase`` — integer phase ID (0, 1, ..., n_phases-1) for bulk
  vertices, or ``INTERFACE_PHASE`` (``-1``) for vertices on the sharp
  interface.  Interface vertices are geometrically ON the phase
  boundary but still represent **integrated bulk fluid volume** —
  their dual cells straddle two (or more) phases, and the per-phase
  arrays below store the split contributions from each phase.
- ``v.is_interface`` — ``True`` only for sharp-interface vertices
  (vertices of the primal interface subcomplex).
- ``v.interface_phases`` — frozenset of phase IDs present at this vertex
  (for a bulk vertex this is ``{v.phase}``; for an interface vertex it
  is the set of phases of its incident top-simplices).

Per-phase arrays (length ``n_phases``, set by ``init_phase_fields``):

- ``v.m_phase[k]``        — mass of phase *k* in this dual cell
- ``v.p_phase[k]``        — pressure of phase *k*
- ``v.rho_phase[k]``      — density of phase *k*
- ``v.dual_vol_phase[k]`` — dual cell sub-volume belonging to phase *k*

For bulk vertices only ``v.*_phase[v.phase]`` is non-zero.
For interface vertices multiple entries are non-zero: the dual cell
is split between phases, and each sub-cell has its own mass, pressure,
and density from the respective phase's EOS and viscosity.

Interface model
---------------
The sharp interface is a **primal subcomplex** of the mesh: a closed
polyline of primal edges in 2D, a closed triangulated 2-manifold of
primal faces in 3D.  Each primal top-simplex (triangle in 2D, tet in
3D) carries a phase label stored in
:attr:`MultiphaseSystem.simplex_phase`.  The interface is the set of
primal simplices shared between two top-simplices of different phases.

Interface topology is stored as plain set attributes on ``HC``:
``HC.interface_vertices``, ``HC.interface_edges``,
``HC.interface_triangles``.  See
:mod:`ddgclib.geometry._interface_subcomplex`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ddgclib.eos._base import EquationOfState


# Sentinel phase ID for interface vertices.  Chosen as -1 so that any
# v.phase-indexed array access (e.g. v.m_phase[v.phase]) either wraps
# around (numpy negative indexing) in a visibly wrong way or trips a
# bounds check — in either case it flags a missing
# ``not v.is_interface`` gate in the caller.  Callers MUST check
# ``v.is_interface`` (or equivalently ``v.phase >= 0``) before indexing
# per-phase arrays by ``v.phase`` directly.
INTERFACE_PHASE: int = -1


def iter_top_simplices(HC, dim: int):
    """Yield each top-simplex of the primal complex exactly once.

    - 2D: yields tuples ``(v0, v1, v2)`` of three mutually connected
      vertices (a triangle).  Delegates to
      :func:`hyperct.remesh._quality.iter_triangles_2d`.
    - 3D: yields tuples ``(v0, v1, v2, v3)`` from ``HC._simplices``
      (populated by
      :func:`ddgclib.geometry.connect_and_cache_simplices`).

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3, or if ``dim == 3`` and
        ``HC._simplices`` is missing/empty.
    """
    if dim == 2:
        from hyperct.remesh._quality import iter_triangles_2d
        yield from iter_triangles_2d(HC)
        return
    if dim == 3:
        simplices = getattr(HC, '_simplices', None)
        if not simplices:
            raise ValueError(
                "iter_top_simplices(dim=3) requires HC._simplices; "
                "call ddgclib.geometry.connect_and_cache_simplices first."
            )
        yield from simplices
        return
    raise ValueError(f"iter_top_simplices supports dim ∈ {{2, 3}}; got {dim}")


def _simplex_key(simplex) -> frozenset:
    """Canonical key for a top-simplex: frozenset of vertex ``v.x`` tuples.

    Order-independent and stable across retriangulation — each primal
    simplex is identified by the set of its vertex coordinate keys.
    """
    return frozenset(v.x for v in simplex)


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
        # Per-top-simplex phase label, keyed by frozenset of vertex
        # coordinate tuples (order-independent, stable across
        # retriangulation).  Authoritative source of truth: the
        # interface subcomplex is derived from this dict, and
        # ``v.phase`` / ``v.is_interface`` are derived caches.
        self.simplex_phase: dict[frozenset, int] = {}
        # Cached centroid-criterion function used to re-label
        # top-simplices after retriangulation.  Set by
        # ``assign_simplex_phases``; reused by ``refresh`` when no
        # explicit ``criterion_fn`` is passed.
        self._simplex_criterion_fn: Callable[[np.ndarray], int] | None = None

    # -- Phase assignment ----------------------------------------------------

    def assign_phases(
        self, HC, criterion_fn: Callable[[np.ndarray], int],
    ) -> None:
        """Set ``v.phase`` on all vertices based on spatial position.

        Legacy phase-field entry point.  For the primal-subcomplex
        model, prefer :meth:`assign_simplex_phases` followed by
        :meth:`assign_vertex_phases_from_simplices`.
        """
        for v in HC.V:
            v.phase = int(criterion_fn(v.x_a))

    # -- Top-simplex phase labelling (primal subcomplex model) --------------

    def assign_simplex_phases(
        self,
        HC,
        dim: int,
        criterion_fn: Callable[[np.ndarray], int],
    ) -> None:
        """Label every top-simplex with a phase ID.

        For each triangle (2D) or tet (3D) in ``HC``, evaluates
        ``criterion_fn`` at the simplex centroid and stores the result
        in :attr:`simplex_phase` keyed by the frozenset of the
        simplex's vertex coordinate tuples.

        This is the **authoritative** source of truth for the
        primal-subcomplex interface model.  Vertex labels and the
        interface subcomplex are derived from it.

        Parameters
        ----------
        HC : Complex
            Simplicial complex.  In 3D ``HC._simplices`` must be
            populated (call
            :func:`ddgclib.geometry.connect_and_cache_simplices` after
            any retriangulation).
        dim : int
            Spatial dimension (2 or 3).
        criterion_fn : callable
            ``criterion_fn(centroid: np.ndarray) -> int`` returning the
            phase ID for a simplex whose centroid is at ``centroid``.
        """
        # Auto-populate ``HC._simplices`` in 3D if the mesh was built
        # without it (e.g. via hyperct's ``triangulate`` +
        # ``refine_all``, or via domain builders that skip
        # ``connect_and_cache_simplices``).  Re-running Delaunay on
        # the current vertex cloud is idempotent — edges are re-added
        # via ``v.connect`` with set semantics — and populates the
        # simplex cache consumed by :func:`iter_top_simplices`.
        if dim == 3 and not getattr(HC, '_simplices', None):
            from ddgclib.geometry._retriangulation import (
                connect_and_cache_simplices,
            )
            verts = list(HC.V)
            coords = np.array([v.x_a[:3] for v in verts], dtype=float)
            connect_and_cache_simplices(HC, verts, 3, coords=coords)

        self.simplex_phase = {}
        self._simplex_criterion_fn = criterion_fn
        for simplex in iter_top_simplices(HC, dim):
            coords_arr = np.array([v.x_a[:dim] for v in simplex], dtype=float)
            centroid = coords_arr.mean(axis=0)
            self.simplex_phase[_simplex_key(simplex)] = int(
                criterion_fn(centroid)
            )

    def assign_simplex_phases_from_vertices(self, HC, dim: int) -> None:
        """Label each top-simplex by majority vote of its vertex phases.

        For runtime retopologization: vertex phases (``v.phase``)
        survive Delaunay reconnection as Lagrangian attributes. This
        method derives the new simplex labels from the existing vertex
        labels via majority rule among the bulk (non-interface) vertices
        of each simplex.

        Rules per simplex:
        - All bulk vertices in same phase → simplex gets that phase.
        - Mixed bulk phases → simplex gets the majority; if tied, the
          simplex is a cross-phase boundary → assigned to the lower
          phase ID (arbitrary but deterministic).
        - All vertices are interface (phase = -1) → use the first
          ``interface_phases`` entry as fallback.

        Must be called AFTER retopologization (so that
        ``iter_top_simplices`` reflects the new connectivity) but
        BEFORE ``identify_interface_from_subcomplex`` (which reads
        ``simplex_phase``).
        """
        if dim == 3 and not getattr(HC, '_simplices', None):
            from ddgclib.geometry._retriangulation import (
                connect_and_cache_simplices,
            )
            verts = list(HC.V)
            coords = np.array([v.x_a[:3] for v in verts], dtype=float)
            connect_and_cache_simplices(HC, verts, 3, coords=coords)

        self.simplex_phase = {}
        for simplex in iter_top_simplices(HC, dim):
            bulk_phases = [
                int(v.phase) for v in simplex if v.phase >= 0
            ]
            if bulk_phases:
                # Majority vote among bulk vertices.
                from collections import Counter
                counts = Counter(bulk_phases)
                winner = counts.most_common(1)[0][0]
            else:
                # All vertices are interface — use interface_phases
                # from any vertex as fallback.
                iph = set()
                for v in simplex:
                    iph.update(
                        k for k in getattr(v, 'interface_phases', frozenset())
                        if k >= 0
                    )
                winner = min(iph) if iph else 0
            self.simplex_phase[_simplex_key(simplex)] = winner

    def assign_vertex_phases_from_simplices(self, HC, dim: int) -> None:
        """Derive ``v.phase`` from the authoritative simplex labels.

        For each vertex, collects the phases of its incident
        top-simplices:

        - If all incident simplices share one phase, ``v.phase`` is
          set to that phase.
        - Otherwise the vertex sits on the interface subcomplex and
          ``v.phase`` is set to :data:`INTERFACE_PHASE`.

        :attr:`interface_phases` is populated for every vertex (a
        frozenset of its incident-simplex phases); callers can use
        this to decide which per-phase arrays are active.

        Must be preceded by :meth:`assign_simplex_phases`.
        """
        # Build incidence: for each vertex, the set of incident-simplex phases.
        incidence: dict = {id(v): set() for v in HC.V}
        for simplex in iter_top_simplices(HC, dim):
            key = _simplex_key(simplex)
            phase = self.simplex_phase.get(key)
            if phase is None:
                continue
            for v in simplex:
                incidence[id(v)].add(phase)

        for v in HC.V:
            phases = incidence[id(v)]
            if not phases:
                # Isolated vertex (no incident simplex); leave phase
                # untouched if already set, otherwise default to 0.
                if not hasattr(v, 'phase'):
                    v.phase = 0
                v.interface_phases = frozenset({v.phase})
                continue
            if len(phases) == 1:
                v.phase = int(next(iter(phases)))
            else:
                v.phase = INTERFACE_PHASE
            v.interface_phases = frozenset(phases)

    # -- Interface identification -------------------------------------------

    def identify_interface_from_subcomplex(
        self, HC, dim: int, strict_closure: bool = True,
    ) -> set:
        """Find interface vertices from the primal-subcomplex model.

        Extracts the interface subcomplex from :attr:`simplex_phase`,
        optionally validates its closure, and marks every vertex:

        - ``v.is_interface = True`` if ``v`` is a vertex of the
          subcomplex; ``v.phase = INTERFACE_PHASE``; ``v.interface_phases
          = {phases of incident top-simplices}``.
        - Otherwise ``v.is_interface = False`` and ``v.phase`` /
          ``v.interface_phases`` are set by
          :meth:`assign_vertex_phases_from_simplices`.

        Parameters
        ----------
        strict_closure : bool
            If ``True`` (default), raise ``ValueError`` when the
            interface is not a closed manifold (appropriate for initial
            mesh construction where conformity is guaranteed).  If
            ``False``, log a warning instead (appropriate during
            runtime retopologization where Delaunay may temporarily
            break closure).

        Returns
        -------
        set
            The set of interface vertex objects.
        """
        from ddgclib.geometry._interface_subcomplex import (
            extract_interface,
            validate_closure,
        )
        if not self.simplex_phase:
            raise ValueError(
                "identify_interface_from_subcomplex requires "
                "simplex_phase to be populated — call "
                "assign_simplex_phases first."
            )

        extract_interface(HC, self.simplex_phase, dim)
        bvs = {v for v in HC.V if getattr(v, 'boundary', False)}
        if strict_closure:
            validate_closure(HC, dim, boundary_vertices=bvs)
        else:
            try:
                validate_closure(HC, dim, boundary_vertices=bvs)
            except ValueError:
                pass  # non-conforming mesh after retopologization

        # Derive bulk phases from simplex incidence.
        self.assign_vertex_phases_from_simplices(HC, dim)

        # Tag is_interface from the sets now on HC.
        interface_vs = getattr(HC, 'interface_vertices', set())
        for v in HC.V:
            if v in interface_vs:
                v.is_interface = True
                v.phase = INTERFACE_PHASE
            else:
                v.is_interface = False

        return interface_vs

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

    def split_dual_volumes(
        self, HC, dim: int, method: str = 'neighbour_count',
    ) -> None:
        """Split each vertex's dual cell volume among phases.

        **Bulk vertices**: the entire dual volume belongs to the
        vertex's own phase.

        **Interface vertices**: the dual cell straddles two (or more)
        phases.  The split depends on ``method``:

        - ``'neighbour_count'`` (default): approximated by the fraction
          of 1-ring neighbours in each phase (including the vertex
          itself).  Fast and robust; backward-compatible.
        - ``'exact'``: exact geometric split of the dual cell by the
          interface.  In 2D, clips the barycentric dual polygon against
          the piecewise-linear interface curve (see
          :func:`ddgclib.geometry._dual_split_2d.split_dual_polygon_2d`).
          In 3D, splits the DEC ``p_ij`` dual polyhedron by the local
          interface tangent plane through ``v`` (see
          :func:`ddgclib.geometry._dual_split_3d.split_dual_polyhedron_3d`) —
          exact on locally planar interfaces, ``O(h^2)`` on curved ones.

        Sets ``v.dual_vol_phase[k]`` for every vertex.
        """
        n = self.n_phases
        if method == 'exact':
            if dim == 2:
                from ddgclib.geometry._dual_split_2d import (
                    split_dual_polygon_2d,
                )
                iface = HC  # interface sets live on HC
                for v in HC.V:
                    areas = split_dual_polygon_2d(
                        v, n_phases=n, interface=iface,
                    )
                    v.dual_vol_phase = np.array(
                        [float(areas.get(k, 0.0)) for k in range(n)]
                    )
                return
            if dim == 3:
                from ddgclib.geometry._dual_split_2d import (
                    split_dual_polyhedron_3d,
                )
                for v in HC.V:
                    vols = split_dual_polyhedron_3d(v, HC, n_phases=n)
                    v.dual_vol_phase = np.array(
                        [float(vols.get(k, 0.0)) for k in range(n)]
                    )
                return

        for v in HC.V:
            vol = getattr(v, 'dual_vol', 0.0)
            if not getattr(v, 'is_interface', False):
                # Bulk: entire volume is one phase.  ``v.phase`` is a
                # valid in-range index here (interface vertices are
                # filtered by ``is_interface``).
                v.dual_vol_phase = np.zeros(n)
                if 0 <= v.phase < n:
                    v.dual_vol_phase[v.phase] = vol
            else:
                # Interface: approximate split from neighbour counts.
                # Only count neighbours in the bulk phases actually
                # present at this interface vertex (via
                # ``v.interface_phases``).  Interface-to-interface
                # edges carry no bulk phase info and are excluded.
                active = frozenset(
                    k for k in getattr(v, 'interface_phases', frozenset())
                    if 0 <= k < n
                )
                counts = np.zeros(n)
                for nb in v.nn:
                    if 0 <= nb.phase < n and nb.phase in active:
                        counts[nb.phase] += 1.0
                total = counts.sum()
                if total > 0:
                    fracs = counts / total
                elif active:
                    # All neighbours are themselves interface vertices
                    # — split equally across the phases present at v.
                    fracs = np.zeros(n)
                    share = 1.0 / len(active)
                    for k in active:
                        fracs[k] = share
                else:
                    fracs = np.zeros(n)
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

            # v.p stores a representative pressure for the vertex.
            # For bulk vertices this is the single-phase pressure; for
            # interface vertices (v.phase == INTERFACE_PHASE) it is the
            # average of the phase pressures actually present at v.
            if getattr(v, 'is_interface', False) or v.phase < 0:
                active = [
                    v.p_phase[k]
                    for k in v.interface_phases
                    if 0 <= k < self.n_phases and v.p_phase[k] != 0.0
                ]
                v.p = float(np.mean(active)) if active else 0.0
            else:
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

    def refresh(
        self,
        HC,
        dim: int,
        reset_mass: bool = True,
        split_method: str = 'neighbour_count',
        criterion_fn: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        """One-call refresh of all multiphase state.

        Pipeline:

        1. If ``criterion_fn`` is passed (or a cached one is available
           from a previous :meth:`assign_simplex_phases` call),
           re-label every top-simplex.  This re-derivation is the
           key retriangulation-safety invariant: after a Delaunay or
           adaptive remesh, the simplex identities change, so the old
           :attr:`simplex_phase` keys are stale — always refresh.
        2. :meth:`identify_interface_from_subcomplex` extracts the
           primal interface subcomplex, validates closure, and tags
           vertices (``v.is_interface``, ``v.phase``).
        3. Reset per-phase arrays (optionally preserving mass).
        4. :meth:`split_dual_volumes` assigns each vertex's dual
           volume to its active phases.
        5. Compute per-phase masses (if ``reset_mass``) and pressures.

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
        split_method : {'neighbour_count', 'exact'}
            Passed through to :meth:`split_dual_volumes`.
        criterion_fn : callable, optional
            Centroid-to-phase-ID mapping used to re-label top-simplices.
            If ``None``, the cached function from the most recent
            :meth:`assign_simplex_phases` call is used.

        Raises
        ------
        ValueError
            If neither ``criterion_fn`` is supplied nor is
            :attr:`simplex_phase` already populated.
        """
        fn = criterion_fn or self._simplex_criterion_fn
        if reset_mass and fn is not None:
            # Initial setup: use the spatial criterion to label simplices.
            self.assign_simplex_phases(HC, dim, fn)
        else:
            # Runtime retopologization: vertex phases (v.phase) survived
            # the Delaunay reconnection as Lagrangian attributes.  Derive
            # the new simplex labels from vertex majority vote — this
            # tracks the moving interface rather than applying a stale
            # fixed-radius spatial test.
            self.assign_simplex_phases_from_vertices(HC, dim)
        if not self.simplex_phase:
            raise ValueError(
                "MultiphaseSystem.refresh requires simplex_phase to be "
                "populated.  Either call assign_simplex_phases(HC, "
                "dim, criterion_fn) before refresh, or pass "
                "criterion_fn= directly to refresh."
            )

        # strict_closure=True on initial setup (reset_mass=True);
        # relaxed during runtime retopologization (reset_mass=False)
        # where Delaunay may temporarily break interface conformity.
        self.identify_interface_from_subcomplex(
            HC, dim, strict_closure=reset_mass,
        )
        if reset_mass:
            self.init_phase_fields(HC)
        else:
            self._reinit_geometry_fields(HC)
        self.split_dual_volumes(HC, dim, method=split_method)
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
