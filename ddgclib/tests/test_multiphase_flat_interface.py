"""Tier 2A gatekeeper test: flat-interface zero-force invariant.

Two variants, each exercised in both 2D and 3D:

- **2A.i**  ``gamma = 0`` (no surface tension).  Two phases with
  uniform per-phase pressure across a horizontal interface (``y = 0``
  in 2D, ``z = 0`` in 3D).  ``u = 0`` and the interface is exactly
  straight, so ``kappa = 0``.  ``multiphase_stress_force`` must be
  machine-precision zero on every vertex including the interface —
  this isolates the per-phase summed pressure-flux cancellation.
- **2A.ii** ``gamma > 0`` with geometrically-exact ``kappa = 0``.  Same
  flat interface, uniform pressure per side, non-zero surface
  tension.  Must still give ``F ≈ 0`` to machine precision — catches
  whether the surface-tension code path leaks noise even when the
  curvature stencil should evaluate to exactly zero.

If any variant fails, D4 (residual interface-vertex force, see
``docs/3d_multiphase_interface_pressure_fix.md``) is real and the
multiphase attack lane is fully justified.  If every variant passes,
the per-phase summed formula is exact on flat interfaces and the
remaining multiphase instability lives elsewhere (curvature stencil
on *curved* interfaces or retopology).

The 3D setup uses a Kuhn-decomposed structured cube mesh rather than
the default hyperct triangulation.  Default ``Complex(3).triangulate()``
produces tetrahedra that cross ``z = 0`` even after ``refine_all()``,
so the extracted interface is a jagged surface with vertices at
``z = -h/4`` (curvature ≠ 0).  The Kuhn decomposition of each cube
into 6 tets contains the cube's diagonal (0,0,0)–(1,1,1); every cube
face is split by its own diagonal, so adjacent cubes share triangles
conformally.  With ``z = 0`` placed on a cube-face layer, no tet
crosses it and the extracted interface is exactly planar.
"""
from __future__ import annotations

import numpy as np
import pytest

from hyperct import Complex
from hyperct.ddg import compute_vd

from ddgclib.eos import TaitMurnaghan
from ddgclib.multiphase import MultiphaseSystem, PhaseProperties
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.geometry._retriangulation import connect_and_cache_simplices


ATOL = 1e-12
P_UNIFORM = 10.0  # uniform pressure (both phases) for the equilibrium check
GAMMA_NZ = 0.05   # non-zero surface tension for the 2A.ii variants


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------


def _build_flat_interface_2d(n_refine: int = 2, L: float = 1.0):
    """2D rectangle ``[-L, L]^2`` with a mesh-aligned horizontal interface
    at ``y = 0``.

    After ``triangulate + refine_all`` the vertex y-coordinates include
    ``y = 0``, and the vertices at ``y = 0`` are linked by primal edges,
    so the centroid criterion ``c[1] < 0`` produces an interface that
    lies exactly on the ``y = 0`` line (a mesh-aligned polyline from
    the left wall to the right wall).
    """
    HC = Complex(2, domain=[(-L, L), (-L, L)])
    HC.triangulate()
    for _ in range(n_refine):
        HC.refine_all()
    bV = set()
    for v in HC.V:
        x, y = float(v.x_a[0]), float(v.x_a[1])
        on_bnd = (abs(x) >= L - 1e-14 or abs(y) >= L - 1e-14)
        v.boundary = on_bnd
        if on_bnd:
            bV.add(v)
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, 2)
    return HC, bV


def _build_flat_interface_3d(n_xy: int = 3, n_half: int = 2, L: float = 1.0):
    """3D cube ``[-L, L]^3`` with a mesh-aligned horizontal interface
    at ``z = 0`` via Kuhn-decomposed cubes.

    Vertices lie on a regular ``(n_xy+1) x (n_xy+1) x (2*n_half+1)``
    grid.  The z-axis has ``2*n_half+1`` layers so that ``z = 0`` is a
    cube-face layer (not inside any cube cell).  Each cube is split
    into 6 tets via Kuhn's decomposition (monotone paths from
    (0,0,0) to (1,1,1) in each of the 6 axis-orderings); adjacent
    cubes share triangulated faces conformally because every cube
    face is split by its own diagonal.  No tet crosses ``z = 0``.
    """
    xs = np.linspace(-L, L, n_xy + 1)
    ys = np.linspace(-L, L, n_xy + 1)
    zs = np.linspace(-L, L, 2 * n_half + 1)

    positions = []
    idx: dict[tuple[int, int, int], int] = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                idx[(i, j, k)] = len(positions)
                positions.append([float(x), float(y), float(z)])
    positions = np.array(positions)

    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2],
             [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    simplices: list[tuple[int, int, int, int]] = []
    for k in range(len(zs) - 1):
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                for perm in perms:
                    path = [(0, 0, 0)]
                    cur = [0, 0, 0]
                    for axis in perm:
                        cur = list(cur)
                        cur[axis] = 1
                        path.append(tuple(cur))
                    tet = tuple(
                        idx[(i + dx, j + dy, k + dz)] for (dx, dy, dz) in path
                    )
                    simplices.append(tet)
    simplices = np.array(simplices)

    eps = 1e-12
    domain = [(-L - eps, L + eps)] * 3
    HC = Complex(3, domain=domain)
    verts = []
    for pos in positions:
        v = HC.V[tuple(pos)]
        verts.append(v)
    connect_and_cache_simplices(HC, verts, 3, simplices=simplices)

    bV = set()
    for v in HC.V:
        on_bnd = any(abs(v.x_a[i]) >= L - 1e-14 for i in range(3))
        v.boundary = on_bnd
        if on_bnd:
            bV.add(v)
    compute_vd(HC, method='barycentric')
    cache_dual_volumes(HC, 3)
    return HC, bV


# ---------------------------------------------------------------------------
# Shared setup + evaluation helpers
# ---------------------------------------------------------------------------


def _configure_flat_two_phase(HC, dim: int, gamma: float) -> MultiphaseSystem:
    """Assign simplex phases by the ``centroid[-1] < 0`` criterion and
    refresh so ``v.is_interface``, ``v.phase``, ``v.interface_phases``,
    and ``v.dual_vol_phase`` are all populated.
    """
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=TaitMurnaghan(rho0=1000.0),
                            mu=0.1, rho0=1000.0, name='phase0'),
            PhaseProperties(eos=TaitMurnaghan(rho0=800.0),
                            mu=0.1, rho0=800.0, name='phase1'),
        ],
        gamma={(0, 1): float(gamma)},
    )
    normal_axis = dim - 1  # y in 2D, z in 3D
    mps.assign_simplex_phases(
        HC, dim, criterion_fn=lambda c: 0 if c[normal_axis] < 0 else 1,
    )
    mps.refresh(HC, dim=dim)
    return mps


def _set_uniform_pressure(HC, dim: int, P: float) -> None:
    """Set ``v.u = 0`` and a uniform pressure ``P`` in both phase slots.

    Uniform per-phase pressure is the *physical* equilibrium for a
    flat interface (no Young-Laplace jump when ``kappa = 0``), so for
    both the ``gamma = 0`` and ``gamma > 0`` variants the net force
    must be exactly zero.
    """
    for v in HC.V:
        v.u = np.zeros(dim)
        if getattr(v, 'is_interface', False):
            v.p_phase[0] = P
            v.p_phase[1] = P
        else:
            v.p_phase[:] = 0.0
            v.p_phase[int(v.phase)] = P
        v.p = P


def _report_max_force(HC, dim: int, mps: MultiphaseSystem, bV: set):
    """Compute ``max |F|`` over interior vertices, split by bulk vs interface.

    Boundary vertices are excluded — their truncated dual cells carry
    an unrelated boundary-flux closure term that is not what this test
    is probing.
    """
    max_bulk = 0.0
    max_iface = 0.0
    worst_v = None
    for v in HC.V:
        if v in bV:
            continue
        F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC)
        nf = float(np.linalg.norm(F))
        if getattr(v, 'is_interface', False):
            if nf > max_iface:
                max_iface = nf
                worst_v = v
        else:
            if nf > max_bulk:
                max_bulk = nf
                worst_v = v if nf > max_iface else worst_v
    return max_bulk, max_iface, worst_v


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------


class TestFlatInterface2D:
    """Tier 2A in 2D: horizontal interface at ``y = 0`` in ``[-1, 1]^2``."""

    def test_gamma_zero(self, capsys):
        """2A.i: gamma = 0.  Per-phase summed pressure flux must cancel."""
        HC, bV = _build_flat_interface_2d(n_refine=2)
        mps = _configure_flat_two_phase(HC, dim=2, gamma=0.0)
        _set_uniform_pressure(HC, dim=2, P=P_UNIFORM)

        n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        max_bulk, max_iface, _ = _report_max_force(HC, 2, mps, bV)
        with capsys.disabled():
            print(f"\n[2A.i 2D] gamma=0, n_iface={n_iface}  "
                  f"max|F| bulk={max_bulk:.3e}  iface={max_iface:.3e}")
        assert max_bulk < ATOL, f"bulk max|F|={max_bulk}"
        assert max_iface < ATOL, f"interface max|F|={max_iface}"

    def test_gamma_nonzero_flat_curvature(self, capsys):
        """2A.ii: gamma > 0 with kappa = 0.  Surface-tension code path
        must return exactly zero on a straight interface.
        """
        HC, bV = _build_flat_interface_2d(n_refine=2)
        mps = _configure_flat_two_phase(HC, dim=2, gamma=GAMMA_NZ)
        _set_uniform_pressure(HC, dim=2, P=P_UNIFORM)

        max_bulk, max_iface, _ = _report_max_force(HC, 2, mps, bV)
        with capsys.disabled():
            print(f"[2A.ii 2D] gamma={GAMMA_NZ}, "
                  f"max|F| bulk={max_bulk:.3e}  iface={max_iface:.3e}")
        assert max_bulk < ATOL, f"bulk max|F|={max_bulk}"
        assert max_iface < ATOL, f"interface max|F|={max_iface}"


# ---------------------------------------------------------------------------
# 3D tests
# ---------------------------------------------------------------------------


class TestFlatInterface3D:
    """Tier 2A in 3D: horizontal interface at ``z = 0`` in ``[-1, 1]^3``,
    built on a Kuhn-decomposed structured cube mesh so the interface
    is exactly planar (no tets cross ``z = 0``)."""

    def test_gamma_zero(self, capsys):
        HC, bV = _build_flat_interface_3d(n_xy=3, n_half=2)
        mps = _configure_flat_two_phase(HC, dim=3, gamma=0.0)
        _set_uniform_pressure(HC, dim=3, P=P_UNIFORM)

        n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
        iface_z = sorted(
            set(round(float(v.x_a[2]), 12)
                for v in HC.V if getattr(v, 'is_interface', False))
        )
        max_bulk, max_iface, _ = _report_max_force(HC, 3, mps, bV)
        with capsys.disabled():
            print(f"\n[2A.i 3D] gamma=0, n_iface={n_iface}, "
                  f"iface z-values={iface_z}  "
                  f"max|F| bulk={max_bulk:.3e}  iface={max_iface:.3e}")
        assert iface_z == [0.0], f"interface not planar: z-values={iface_z}"
        assert max_bulk < ATOL, f"bulk max|F|={max_bulk}"
        assert max_iface < ATOL, f"interface max|F|={max_iface}"

    def test_gamma_nonzero_flat_curvature(self, capsys):
        HC, bV = _build_flat_interface_3d(n_xy=3, n_half=2)
        mps = _configure_flat_two_phase(HC, dim=3, gamma=GAMMA_NZ)
        _set_uniform_pressure(HC, dim=3, P=P_UNIFORM)

        max_bulk, max_iface, _ = _report_max_force(HC, 3, mps, bV)
        with capsys.disabled():
            print(f"[2A.ii 3D] gamma={GAMMA_NZ}, "
                  f"max|F| bulk={max_bulk:.3e}  iface={max_iface:.3e}")
        assert max_bulk < ATOL, f"bulk max|F|={max_bulk}"
        assert max_iface < ATOL, f"interface max|F|={max_iface}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
