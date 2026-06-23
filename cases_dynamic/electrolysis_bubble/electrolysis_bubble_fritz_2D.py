#!/usr/bin/env python3
"""2D electrolysis hydrogen bubble with Fritz equilibrium initial geometry.

The previous ``electrolysis_bubble_2D.py`` starts from a *fully-immersed
circular* gas bubble offset toward the electrode.  That is the wrong
initial state for an electrolysis bubble: on a flat electrode with
pinned three-phase contact, the equilibrium shape under gravity +
surface tension is the **Fritz bubble** — the solution of the
axisymmetric Young-Laplace equation

    gamma * kappa = P_0 + rho_diff * g * z                      (1)

for a sessile gas bubble with a pinned contact line (usually at 90 deg
for a hydrophobic electrode).  The resulting profile is NOT a sphere:
it flattens at the top and elongates at the foot as the Bond number
grows, which is exactly what Fritz's detachment criterion relies on.

This script focuses on getting the initial geometry right:

1.  Solve the Young-Laplace ODE (same Adams-Bashforth integration used
    in the mean-flow surface-mesh case ``cases_mean_flow/bubble/``) to
    obtain the equilibrium profile ``r(s), z(s)``.
2.  Build the two-phase *volumetric* mesh for the dynamic pipeline by
    (a) warping a rectangle into the Fritz interior (gas phase), and
    (b) combining it with an outer rectangular box (liquid phase) via
    the same ``_build_combined_mesh`` helper used by
    ``droplet_in_box_2d`` — no hand-rolled triangulation, refinement,
    edge-flipping, etc.
3.  Plot the initial mesh using the standard ``plot_fluid`` /
    ``plot_primal`` interface so the phase colouring and interface
    markers are picked up for free.

No time integration is performed here — this is a geometry-only smoke
test.  A follow-up script can plug the resulting ``HC, bV, mps`` into
``setup_electrolysis_bubble`` (or a stripped-down dudt/retopo pair) to
run the dynamic simulation from the Fritz equilibrium.

Usage
-----
    python cases_dynamic/electrolysis_bubble/electrolysis_bubble_fritz_2D.py
"""
from __future__ import annotations

import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hyperct.ddg import compute_vd

from ddgclib.geometry.domains._rectangles import rectangle
from ddgclib.geometry.domains._multiphase_droplet import _build_combined_mesh
from ddgclib.multiphase import (
    MultiphaseSystem, PhaseProperties,
)
from ddgclib.eos import TaitMurnaghan, MultiphaseEOS
from ddgclib.multiphase import mass_conserving_merge
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.operators.multiphase_stress import multiphase_dudt_i
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import (
    _retopologize_multiphase,
)
from ddgclib._boundary_conditions import (
    BoundaryConditionSet, NoSlipWallBC,
)
from ddgclib.visualization import plot_fluid

from cases_dynamic.electrolysis_bubble.src._setup import WallClampBC

from cases_dynamic.electrolysis_bubble.src._params import (
    R0, L_domain, rho_liq, rho_gas, mu_liq, mu_gas,
    gamma, K_liq, K_gas, g, P0,
    n_refine_drop_2d, cfl_safety,
)
from cases_dynamic.electrolysis_bubble.src._analytical import (
    capillary_length, bond_number, young_laplace_jump,
)

# Override the electrolysis_bubble default (2) — the Fritz case wants a
# finer outer water mesh so the edge length in the liquid is comparable
# to the edge length inside the bubble.  At refinement=4 the outer box
# has ~17 vertices per side (256 triangles) which matches the ~2^3 rows
# of gas interior.
n_refine_outer_2d = 4


_CASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG = os.path.join(_CASE_DIR, 'fig')


# =====================================================================
# Fritz (Young-Laplace) profile
# =====================================================================

def fritz_profile(
    Bo: float,
    R_top: float,
    contact_angle: float = 0.5 * np.pi,
    ds: float | None = None,
    max_steps: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fritz bubble interface profile (r, z).

    Solves the dimensionless axisymmetric Young-Laplace ODE from
    Demirkir 2024 / Chesters 1977::

        dr/ds   = cos(psi)
        dz/ds   = sin(psi)
        dpsi/ds = 2 - Bo * z - sin(psi) / r

    starting from the bubble apex ``(r, z, psi) = (0, 0, 0)`` and
    marching in arc length ``s`` until the tangent angle ``psi``
    reaches ``contact_angle`` (pinned foot).  All returned coordinates
    are in *physical* units with::

        r[0]  = 0                 (apex, on the symmetry axis)
        z[0]  = 0                 (apex, measured downward along s)
        r[-1] = r_foot            (pinned contact radius)
        z[-1] = -height           (foot is below the apex → positive z up)

    Parameters
    ----------
    Bo
        Bond number ``rho_diff * g * R_top**2 / gamma``.
    R_top
        Radius of curvature at the bubble apex, in metres.  Sets the
        physical length scale of the profile.
    contact_angle
        Interior angle between the interface and the electrode at the
        pinned foot (radians).  Default ``pi/2`` = hemispherical foot.
    ds
        Arc-length step in dimensionless units.  Defaults to
        ``0.001 * min(1/Bo, 1)`` so that small-Bond profiles (which
        march a long way) still have reasonable resolution.
    max_steps
        Hard cap on integration steps.

    Returns
    -------
    r, z : (N,) arrays in metres
    """
    if Bo <= 0:
        raise ValueError("Bo must be > 0 for a Fritz profile under gravity")
    if ds is None:
        ds = 0.001 * min(1.0 / Bo, 1.0)

    psi = 0.0
    r = 0.0
    z = 0.0
    r_list = [r]
    z_list = [z]
    for _ in range(max_steps):
        r += ds * np.cos(psi)
        dz = ds * np.sin(psi)
        z += dz
        r_list.append(r)
        z_list.append(z)
        if r <= 0:
            break
        psi += ds * (2.0 - Bo * z - np.sin(psi) / r)
        # stopping criterion: pinned contact angle reached
        if psi >= contact_angle:
            break
        # fallback: equation singular
        if 2.0 - Bo * z - np.sin(psi) / r < 0 and psi >= contact_angle:
            break

    r_arr = np.asarray(r_list) * R_top
    # Flip z so positive = up (apex at z=0, foot at z=-height).
    z_arr = -np.asarray(z_list) * R_top
    return r_arr, z_arr


# =====================================================================
# Sampled positions for the two-phase mesh
# =====================================================================

def _sample_interface_positions(
    r_profile: np.ndarray,
    z_profile: np.ndarray,
    n_interface: int,
    reflect: bool = True,
) -> np.ndarray:
    """Resample (r, z) profile to ``n_interface`` vertices per side.

    Produces a dense, evenly arc-length–spaced discretisation of the
    Fritz interface suitable for Delaunay triangulation.  If
    ``reflect`` is True the profile is mirrored in x=0 (so both
    left and right halves of the closed bubble boundary are returned).
    The apex and foot vertices appear once each.
    """
    # Cumulative arc length along the profile.
    dr = np.diff(r_profile)
    dz = np.diff(z_profile)
    ds = np.hypot(dr, dz)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_target = np.linspace(0.0, s[-1], n_interface)
    r_samp = np.interp(s_target, s, r_profile)
    z_samp = np.interp(s_target, s, z_profile)

    right = np.column_stack([r_samp, z_samp])
    if not reflect:
        return right
    # mirror, excluding the apex (r=0) and foot (on the symmetry axis
    # if the foot is at r=0 — which is not the case here for a Fritz
    # profile, so include both endpoints on the left side as well,
    # except the apex itself which is exactly on x=0).
    left = np.column_stack([-r_samp[1:], z_samp[1:]])
    return np.vstack([right, left])


def _sample_gas_interior(
    r_profile: np.ndarray,
    z_profile: np.ndarray,
    n_rows: int,
    safety: float = 0.88,
) -> np.ndarray:
    """Sample interior points inside the Fritz profile.

    Places a variable-density row of vertices at each z slice — the
    number of columns scales with the local bubble width ``r(z)``
    relative to the widest ``r_foot`` so the interior edge length is
    roughly uniform.  Rows are shrunk radially by ``safety`` so
    interior vertices never collide with interface vertices during
    Delaunay triangulation.
    """
    z_top = float(z_profile[0])    # apex (= 0 in raw profile coords)
    z_foot = float(z_profile[-1])  # negative in raw profile coords
    r_foot = float(r_profile[-1])

    z_grid = np.linspace(z_foot, z_top, n_rows + 2)[1:-1]  # strict interior
    z_sorted = z_profile[::-1]
    r_sorted = r_profile[::-1]

    pts: list[np.ndarray] = []
    for z in z_grid:
        r_max = float(np.interp(z, z_sorted, r_sorted)) * safety
        if r_max <= 1e-12:
            continue
        # Keep the per-row vertex spacing similar to the widest row's
        # spacing (≈ r_foot / n_rows) so triangles stay roughly
        # equilateral throughout the bubble.
        n_cols = max(1, int(round(n_rows * r_max / r_foot)))
        if n_cols == 1:
            pts.append(np.array([0.0, z]))
            continue
        for j in range(n_cols):
            x_frac = -1.0 + 2.0 * (j + 0.5) / n_cols
            pts.append(np.array([x_frac * r_max, z]))
    return np.asarray(pts)


# =====================================================================
# Mesh builder
# =====================================================================

def build_fritz_bubble_in_box_2d(
    R_top: float,
    L_domain: float,
    Bo: float,
    electrode_z: float,
    refinement_outer: int = 2,
    refinement_droplet: int = 3,
    contact_angle: float = 0.5 * np.pi,
):
    """Assemble the two-phase 2D mesh with a Fritz-shaped gas bubble.

    Parameters
    ----------
    R_top
        Apex radius of curvature [m] — sets the bubble scale.
    L_domain
        Half-side of the outer rectangular liquid domain (box is
        ``[-L, L] x [electrode_z, electrode_z + 2 L]``).
    Bo
        Bond number ``(rho_liq - rho_gas) * g * R_top**2 / gamma``.
    electrode_z
        y-coordinate of the flat electrode (the bubble foot sits on
        this line).
    refinement_outer, refinement_droplet
        Mesh refinement levels for the liquid box and the gas
        interior, respectively.  Larger → finer.
    contact_angle
        Interior angle between the Fritz interface and the electrode
        at the pinned foot (radians).

    Returns
    -------
    HC, bV_walls, mps, meta
        ``HC``           combined simplicial complex, barycentric duals
                         already cached.
        ``bV_walls``     set of outer-box wall vertices (no-slip).
        ``mps``          :class:`MultiphaseSystem` with ``simplex_phase``
                         populated (phase 0 = liquid, phase 1 = gas).
        ``meta``         dict of geometry metadata (profile, r_foot,
                         height, interface positions, ...).
    """
    dim = 2

    # ---------- 1. Fritz equilibrium profile ------------------------
    r_profile, z_profile = fritz_profile(
        Bo=Bo, R_top=R_top, contact_angle=contact_angle,
    )
    r_foot = float(r_profile[-1])
    height = float(z_profile[0] - z_profile[-1])  # apex - foot > 0

    # ---------- 2. Outer liquid box ---------------------------------
    # [-L, L] in x, [electrode_z, electrode_z + 2 L] in y.  The
    # electrode is the bottom wall.  Build the mesh directly at the
    # target origin — never translate after the fact with raw
    # ``HC.V.move()`` in a loop, because the tuple-keyed vertex cache
    # silently merges colliding coordinates (e.g. translating a
    # ``[0, 2L]^2`` mesh by ``(-L, -L)`` maps the ``(2L, 2L)`` corner
    # onto the existing centre vertex ``(L, L)`` — the corner
    # disappears).  When a translate / stretch / shrink really is
    # unavoidable, use ``ddgclib.geometry._complex_operations.translate``
    # (with ``jitter=1e-12`` if needed) — it documents this failure
    # mode and provides the workaround.
    outer = rectangle(
        L=2.0 * L_domain, h=2.0 * L_domain,
        refinement=refinement_outer, flow_axis=0,
        origin=(-L_domain, electrode_z),
    )
    HC_outer = outer.HC

    # ---------- 3. Collect positions by phase -----------------------
    # Liquid: outer-box vertices that are NOT inside the Fritz profile.
    # The cut region is grown by a small guard band (sized by the gas
    # mesh spacing, not the outer mesh) so outer vertices don't land
    # on top of interface vertices in Delaunay.
    outer_h = _estimate_outer_edge_length(HC_outer, dim)
    gas_h = r_foot / max(1, 2 ** refinement_droplet)
    guard_band = 0.5 * gas_h

    positions: list[np.ndarray] = []
    phases: list[int] = []

    for v in HC_outer.V:
        pos2 = np.round(v.x_a[:dim], decimals=10)
        if not _inside_fritz(pos2[0], pos2[1], r_profile, z_profile,
                             electrode_z=electrode_z,
                             inflate=guard_band):
            positions.append(pos2)
            phases.append(0)

    # Gas interior: variable-density structured sampling inside the
    # Fritz profile.  ``refinement_droplet`` sets the radial vertex
    # count along the widest row; other rows are scaled proportionally
    # so triangles stay roughly equilateral.
    gas_n_rows = 2 ** refinement_droplet
    gas_interior = _sample_gas_interior(
        r_profile=r_profile,
        z_profile=z_profile,
        n_rows=gas_n_rows,
        safety=0.88,
    )
    for pt in gas_interior:
        positions.append(np.round(
            np.array([pt[0], pt[1] + electrode_z + height]),
            decimals=10,
        ))
        phases.append(1)

    # Interface: sampled Fritz profile (both sides), snapped to the
    # electrode at the foot.  These vertices belong to the gas side
    # of the interface for labelling purposes; the primal-subcomplex
    # model derives the actual interface from simplex phase labels.
    n_interface = max(16, 2 * (2 ** refinement_droplet))
    iface = _sample_interface_positions(
        r_profile=r_profile, z_profile=z_profile,
        n_interface=n_interface, reflect=True,
    )
    for pt in iface:
        x_world = pt[0]
        z_world = pt[1] + electrode_z + height  # place foot at electrode_z
        # Snap points that land exactly on the electrode onto it.
        if abs(z_world - electrode_z) < 1e-10:
            z_world = electrode_z
        positions.append(np.round(np.array([x_world, z_world]), decimals=10))
        phases.append(1)

    # ---------- 4. Build unified complex ----------------------------
    HC = _build_combined_mesh(positions, phases, dim)

    # ---------- 5. Tag boundaries, compute duals --------------------
    dV = HC.boundary()
    tol = 1e-9
    bV_walls: set = set()
    for v in dV:
        x, y = float(v.x_a[0]), float(v.x_a[1])
        on_left = abs(x + L_domain) < tol
        on_right = abs(x - L_domain) < tol
        on_bottom = abs(y - electrode_z) < tol
        on_top = abs(y - (electrode_z + 2.0 * L_domain)) < tol
        if on_left or on_right or on_bottom or on_top:
            bV_walls.add(v)

    for v in HC.V:
        v.boundary = v in dV

    compute_vd(HC, method="barycentric")
    cache_dual_volumes(HC, dim)

    # ---------- 6. Multiphase system + simplex labelling ------------
    eos_liq = TaitMurnaghan(rho0=rho_liq, P0=P0, K=K_liq, n=1.0,
                            rho_clip=(0.5, 2.0))
    eos_gas = TaitMurnaghan(rho0=rho_gas, P0=P0, K=K_gas, n=1.0,
                            rho_clip=(0.5, 2.0))
    mps = MultiphaseSystem(
        phases=[
            PhaseProperties(eos=eos_liq, mu=mu_liq, rho0=rho_liq,
                            name="liquid"),
            PhaseProperties(eos=eos_gas, mu=mu_gas, rho0=rho_gas,
                            name="gas"),
        ],
        gamma={(0, 1): gamma},
    )

    def _phase_criterion(centroid):
        x = float(centroid[0])
        y = float(centroid[1])
        return 1 if _inside_fritz(
            x, y, r_profile, z_profile,
            electrode_z=electrode_z, inflate=0.0,
        ) else 0

    mps._simplex_criterion_fn = _phase_criterion
    mps.refresh(HC, dim, reset_mass=True, split_method='neighbour_count',
                criterion_fn=_phase_criterion)

    meta = {
        'r_profile': r_profile,
        'z_profile': z_profile,
        'r_foot': r_foot,
        'height': height,
        'electrode_z': electrode_z,
        'L_domain': L_domain,
        'Bo': Bo,
        'R_top': R_top,
        'contact_angle': contact_angle,
        'outer_edge_length': outer_h,
    }
    return HC, bV_walls, mps, meta


# =====================================================================
# Helpers
# =====================================================================

def _estimate_outer_edge_length(HC, dim: int) -> float:
    lengths = []
    for v in HC.V:
        for nb in v.nn:
            d = float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
            if d > 1e-15:
                lengths.append(d)
        if len(lengths) > 200:
            break
    return float(np.median(lengths)) if lengths else 0.0


def _inside_fritz(
    x: float,
    y: float,
    r_profile: np.ndarray,
    z_profile: np.ndarray,
    electrode_z: float,
    inflate: float = 0.0,
) -> bool:
    """Return True if the point ``(x, y)`` lies inside the Fritz bubble.

    The bubble occupies the region bounded below by the electrode at
    ``y = electrode_z``, above by the Fritz profile ``r(z)`` (mirrored
    in x = 0).  ``inflate`` enlarges the region isotropically (used to
    carve out a guard band for the outer-box vertex cull).
    """
    z_top = float(z_profile[0])     # = 0 in raw coords
    z_foot = float(z_profile[-1])   # = -height in raw coords
    height = z_top - z_foot

    # Translate world coordinate to raw profile coordinate.
    z_local = y - (electrode_z + height)
    if z_local > 0.0 + inflate:
        return False
    if z_local < z_foot - inflate:
        return False

    # Interpolate r_max(z).  z_profile is monotonically decreasing
    # from 0 (apex) to z_foot (foot), so reverse for np.interp.
    z_sorted = z_profile[::-1]
    r_sorted = r_profile[::-1]
    r_max = float(np.interp(z_local, z_sorted, r_sorted))
    return abs(x) <= r_max + inflate


# =====================================================================
# Main: build + plot
# =====================================================================

def main():
    os.makedirs(_FIG, exist_ok=True)

    dim = 2
    electrode_z = -L_domain

    rho_diff = rho_liq - rho_gas
    lam = capillary_length(gamma, rho_diff, g)
    Bo0 = bond_number(R0, gamma, rho_diff, g)
    dP_lap = young_laplace_jump(gamma, R0, dim=dim)

    print("=" * 64)
    print("2D electrolysis bubble (Fritz equilibrium geometry)")
    print("=" * 64)
    print(f"Capillary length  lambda = {lam * 1e3:.3f} mm")
    print(f"Bond number       Bo(R0) = {Bo0:.3e}")
    print(f"Laplace jump   DeltaP    = {dP_lap:.2f} Pa")
    print(f"Electrode at  y = {electrode_z * 1e3:.2f} mm,"
          f"  domain  [-L, L] = [{-L_domain * 1e3:.1f}, "
          f"{L_domain * 1e3:.1f}] mm")

    # The Fritz profile is parameterised by the *apex* radius of
    # curvature, not the equivalent spherical radius.  For small
    # Bond numbers R_top ≈ R0; for larger Bo a nonlinear mapping is
    # needed — we take R_top = R0 as the proof-of-concept starting
    # point.  The reported r_foot / height below show the actual
    # departure from a circle.
    R_top = R0

    print("\nBuilding Fritz-shaped two-phase mesh...")
    HC, bV, mps, meta = build_fritz_bubble_in_box_2d(
        R_top=R_top,
        L_domain=L_domain,
        Bo=Bo0,
        electrode_z=electrode_z,
        refinement_outer=n_refine_outer_2d,
        refinement_droplet=n_refine_drop_2d,
        contact_angle=0.5 * np.pi,
    )

    n_verts = sum(1 for _ in HC.V)
    n_iface = sum(1 for v in HC.V if getattr(v, 'is_interface', False))
    n_gas = sum(1 for v in HC.V
                if getattr(v, 'phase', None) == 1
                and not getattr(v, 'is_interface', False))
    n_liq = sum(1 for v in HC.V
                if getattr(v, 'phase', None) == 0
                and not getattr(v, 'is_interface', False))
    print(f"Fritz profile:  r_foot = {meta['r_foot'] * 1e3:.4f} mm,"
          f"  height = {meta['height'] * 1e3:.4f} mm")
    r_apex_ratio = meta['r_foot'] / R_top
    height_ratio = meta['height'] / R_top
    print(f"  r_foot / R_top = {r_apex_ratio:.4f},"
          f"  height / R_top = {height_ratio:.4f}"
          f"  (sphere: 1.0 / 2.0)")
    print(f"Mesh: {n_verts} vertices "
          f"(liquid={n_liq}, gas={n_gas}, interface={n_iface}, "
          f"walls={len(bV)})")

    # Full IC: mass / pressure / velocity (load-bearing for the dynamic
    # run — masses set the EOS-derived per-phase pressures every step).
    _apply_fritz_ic(HC, mps, meta, electrode_z=electrode_z,
                    P0=P0, rho_liq=rho_liq, rho_gas=rho_gas,
                    g=g, gamma=gamma, R_top=R_top)
    # Pin masses against any pre-existing duplicate vertices, then
    # re-derive phase fields (volumes, pressures) from the new state.
    mass_conserving_merge(HC, cdist=1e-12)
    mps.refresh(HC, dim=dim, reset_mass=False,
                split_method='neighbour_count')

    # -- Static plots: initial state ------------------------------------
    _save_fritz_plot(
        HC, bV, meta,
        electrode_z=electrode_z, L_domain=L_domain,
        xlim=(-L_domain * 1.05, L_domain * 1.05),
        ylim=(electrode_z - 0.05 * L_domain,
              electrode_z + 2.0 * L_domain + 0.05 * L_domain),
        save_path=os.path.join(_FIG,
                               'electrolysis_bubble_fritz_2D_init.png'),
        title_suffix=' (full domain)',
    )
    zoom = 1.5 * max(meta['r_foot'], meta['height'])
    _save_fritz_plot(
        HC, bV, meta,
        electrode_z=electrode_z, L_domain=L_domain,
        xlim=(-zoom, zoom),
        ylim=(electrode_z - 0.1 * zoom, electrode_z + zoom),
        save_path=os.path.join(_FIG,
                               'electrolysis_bubble_fritz_2D_init_zoom.png'),
        title_suffix=' (zoom on bubble)',
    )

    # -- Short dynamic run ----------------------------------------------
    print("\nRunning short dynamics smoke test...")
    info = run_short_dynamics(HC, bV, mps, meta, n_steps=80)
    print(f"  finished at t = {info['t_final']:.3e} s")

    _save_fritz_plot(
        HC, bV, meta,
        electrode_z=electrode_z, L_domain=L_domain,
        xlim=(-L_domain * 1.05, L_domain * 1.05),
        ylim=(electrode_z - 0.05 * L_domain,
              electrode_z + 2.0 * L_domain + 0.05 * L_domain),
        save_path=os.path.join(_FIG,
                               'electrolysis_bubble_fritz_2D_final.png'),
        title_suffix=f" (t = {info['t_final']:.2e} s)",
    )
    _save_fritz_plot(
        HC, bV, meta,
        electrode_z=electrode_z, L_domain=L_domain,
        xlim=(-zoom, zoom),
        ylim=(electrode_z - 0.1 * zoom, electrode_z + zoom),
        save_path=os.path.join(_FIG,
                               'electrolysis_bubble_fritz_2D_final_zoom.png'),
        title_suffix=f" zoom (t = {info['t_final']:.2e} s)",
    )

    print("\nDone.")


def _save_fritz_plot(HC, bV, meta, *, electrode_z, L_domain,
                      xlim, ylim, save_path, title_suffix=""):
    """Two-panel plot (pressure + velocity) with Fritz overlay."""
    import matplotlib.pyplot as plt
    fig, axes = plot_fluid(
        HC, bV=bV, t=0.0,
        scalar_field='p', vector_field='u',
        scalar_label='Pressure [Pa]' + title_suffix,
        vector_label='Velocity [m/s]  (zero at t=0)' + title_suffix,
        xlim=xlim, ylim=ylim,
        save_path=None,
        face_alpha=0.5,
    )
    z_world = meta['z_profile'] + electrode_z + meta['height']
    for ax in axes:
        ax.plot(meta['r_profile'], z_world, 'r-', lw=1.2,
                label='Fritz profile')
        ax.plot(-meta['r_profile'], z_world, 'r-', lw=1.2)
        ax.axhline(electrode_z, color='k', ls='--', lw=1.0,
                   label='electrode')
        ax.set_aspect('equal', adjustable='box')
    axes[0].legend(loc='upper right', fontsize=7)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {save_path}")


def _apply_fritz_ic(HC, mps, meta, *,
                     electrode_z: float, P0: float,
                     rho_liq: float, rho_gas: float,
                     g: float, gamma: float,
                     R_top: float) -> None:
    """Apply hydrostatic-liquid + Young-Laplace-gas IC.

    Same recipe as ``setup_electrolysis_bubble`` but applied to the
    pre-built Fritz mesh.  Sets:

    - ``v.u``           — zero velocity
    - ``v.m_phase[k]``  — ``rho_target_k * dual_vol_phase[k]`` where
                           ``rho_target_k = eos.density(P_target_k)``
    - ``v.p_phase[k]``  — diagnostic, recomputed by ``mps.compute_phase_pressures``
    - ``v.p``           — phase-volume-weighted aggregate for the
                          scalar plot

    The masses are the load-bearing IC: per-vertex pressures are
    recovered by the EOS from rho = m / vol on every step.
    """
    dim = 2
    axis = dim - 1
    L_domain = meta['L_domain']
    wall_top = electrode_z + 2.0 * L_domain
    z_top_of_bubble = electrode_z + meta['height']
    P_gas_apex = (P0 + rho_liq * g * (wall_top - z_top_of_bubble)
                  + gamma / R_top)

    eos_liq = mps.phases[0].eos
    eos_gas = mps.phases[1].eos

    for v in HC.V:
        v.u = np.zeros(dim)
        y = float(v.x_a[axis])

        vol_l = float(v.dual_vol_phase[0])
        if np.isfinite(vol_l) and vol_l > 1e-30:
            P_l = P0 + rho_liq * g * (wall_top - y)
            rho_l = float(eos_liq.density(P_l))
            v.m_phase[0] = rho_l * vol_l
            v.p_phase[0] = P_l

        vol_g = float(v.dual_vol_phase[1])
        if np.isfinite(vol_g) and vol_g > 1e-30:
            P_g = P_gas_apex + rho_gas * g * (z_top_of_bubble - y)
            rho_g = float(eos_gas.density(P_g))
            v.m_phase[1] = rho_g * vol_g
            v.p_phase[1] = P_g

        # NaN sweep
        if not np.all(np.isfinite(v.m_phase)):
            v.m_phase = np.nan_to_num(v.m_phase, nan=0.0,
                                      posinf=0.0, neginf=0.0)
        v.m = float(np.sum(v.m_phase))

        num = 0.0
        den = 0.0
        if vol_l > 1e-30:
            num += float(v.p_phase[0]) * vol_l
            den += vol_l
        if vol_g > 1e-30:
            num += float(v.p_phase[1]) * vol_g
            den += vol_g
        v.p = num / den if den > 0 else 0.0


# =====================================================================
# Dynamic-run wiring (BCs, dudt, retopology)
# =====================================================================

def setup_fritz_dynamics(HC, bV, mps, meta):
    """Build ``bc_set, dudt_fn, retopo_fn`` for the Fritz mesh.

    Mirrors the wall-clamp + multiphase-stress + gravity recipe from
    ``setup_electrolysis_bubble``, parameterised by the Fritz mesh
    metadata.  The returned callables plug straight into
    ``symplectic_euler(HC, bV, dudt_fn, ..., bc_set=bc_set,
    retopologize_fn=retopo_fn)``.
    """
    dim = 2
    axis = dim - 1
    L_dom = meta['L_domain']
    wall_bottom = meta['electrode_z']
    wall_top = wall_bottom + 2.0 * L_dom
    R_top = meta['R_top']

    bc_set = BoundaryConditionSet()
    bc_set.add(NoSlipWallBC(dim=dim), bV)
    bc_set.add(
        WallClampBC(axis=axis, level=wall_bottom, direction=+1,
                    min_gap=0.02 * R_top, exclude=bV),
        None,
    )
    bc_set.add(
        WallClampBC(axis=axis, level=wall_top, direction=-1,
                    min_gap=0.02 * R_top, exclude=bV),
        None,
    )

    eos_liq = mps.phases[0].eos
    eos_gas = mps.phases[1].eos
    meos = MultiphaseEOS([eos_liq, eos_gas])

    _base_dudt = partial(
        multiphase_dudt_i,
        dim=dim, mps=mps, HC=HC, pressure_model=meos,
    )
    gravity_vec = np.zeros(dim)
    gravity_vec[axis] = -g

    def dudt_fn(v, **_kw):
        if not np.isfinite(v.m) or v.m < 1e-30:
            return np.zeros(dim)
        a = _base_dudt(v)
        if not np.all(np.isfinite(a)):
            return np.zeros(dim)
        return a + gravity_vec

    retopo_fn = partial(
        _retopologize_multiphase, mps=mps,
        split_method='neighbour_count',
    )
    return bc_set, dudt_fn, retopo_fn


def run_short_dynamics(HC, bV, mps, meta, *,
                        n_steps: int = 80,
                        cfl: float = 0.05) -> dict:
    """Run a short symplectic-Euler integration to confirm stability.

    Returns ``{'t_final', 'dt', 'dx_min'}``.  The mesh ``HC`` is updated
    in place (positions, velocities, pressures, dual volumes).
    """
    dim = 2
    bc_set, dudt_fn, retopo_fn = setup_fritz_dynamics(HC, bV, mps, meta)

    c_s_liq = float(np.sqrt(K_liq / rho_liq))
    c_s_gas = float(np.sqrt(K_gas / rho_gas))
    c_s = max(c_s_liq, c_s_gas)
    dx_min = min(
        float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim]))
        for v in HC.V for nb in v.nn
        if float(np.linalg.norm(v.x_a[:dim] - nb.x_a[:dim])) > 1e-15
    )
    dt_cfl = cfl * dx_min / c_s
    dt_st = 0.4 * float(np.sqrt(rho_liq * dx_min**3 / gamma))
    dt = min(dt_cfl, dt_st)

    print(f"  c_s = {c_s:.2f} m/s, dx_min = {dx_min:.3e} m,"
          f" dt = {dt:.3e} s, n_steps = {n_steps}"
          f"  → t_window = {dt * n_steps:.3e} s")

    t_final = symplectic_euler(
        HC, bV, dudt_fn, dt=dt, n_steps=n_steps, dim=dim,
        bc_set=bc_set, retopologize_fn=retopo_fn,
        remesh_mode='delaunay', remesh_kwargs=None,
    )
    return {'t_final': float(t_final), 'dt': float(dt), 'dx_min': float(dx_min)}


if __name__ == '__main__':
    main()
