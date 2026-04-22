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

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from hyperct.ddg import compute_vd

from ddgclib.geometry.domains._rectangles import rectangle
from ddgclib.geometry.domains._multiphase_droplet import _build_combined_mesh
from ddgclib.multiphase import (
    MultiphaseSystem, PhaseProperties,
)
from ddgclib.eos import TaitMurnaghan
from ddgclib.operators.stress import cache_dual_volumes
from ddgclib.visualization import plot_fluid

from cases_dynamic.electrolysis_bubble.src._params import (
    R0, L_domain, rho_liq, rho_gas, mu_liq, mu_gas,
    gamma, K_liq, K_gas, g, P0,
    n_refine_outer_2d, n_refine_drop_2d,
)
from cases_dynamic.electrolysis_bubble.src._analytical import (
    capillary_length, bond_number, young_laplace_jump,
)


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
    n_cols: int,
    safety: float = 0.92,
) -> np.ndarray:
    """Sample interior points inside the Fritz profile.

    Uses a structured grid over ``[-r(z), +r(z)] x [z_foot, z_top]``,
    scaled inward by ``safety`` so interior vertices never land on the
    interface (which would collide with the interface vertices during
    Delaunay triangulation).
    """
    z_top = float(z_profile[0])   # apex (= 0 in raw profile coords)
    z_foot = float(z_profile[-1])  # negative in raw profile coords

    z_grid = np.linspace(z_foot, z_top, n_rows + 2)[1:-1]  # strict interior
    pts = [np.array([0.0, 0.5 * (z_top + z_foot)])]
    # sort profile by z for 1-D interpolation (z is monotonically
    # decreasing from apex=0 down to foot=-height, so reverse)
    z_sorted = z_profile[::-1]
    r_sorted = r_profile[::-1]
    for z in z_grid:
        r_max = float(np.interp(z, z_sorted, r_sorted)) * safety
        if r_max <= 1e-12:
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
    #  [-L, L] in x, [electrode_z, electrode_z + 2 L] in y.  The
    #  electrode is the bottom wall.
    Lx = 2.0 * L_domain
    Ly = 2.0 * L_domain
    outer = rectangle(
        L=Lx, h=Ly, refinement=refinement_outer, flow_axis=0,
    )
    HC_outer = outer.HC
    for v in list(HC_outer.V):
        pos = v.x_a.copy()
        pos[0] -= L_domain          # shift x: [0, 2L] → [-L, L]
        pos[1] += electrode_z        # shift y: [0, 2L] → [electrode_z, electrode_z + 2L]
        HC_outer.V.move(v, tuple(pos))

    # ---------- 3. Collect positions by phase -----------------------
    # Liquid: outer-box vertices that are NOT inside the Fritz profile
    # (we grow the cut region by one edge length to give Delaunay room
    # to draw quality triangles between the interface and the bulk).
    outer_h = _estimate_outer_edge_length(HC_outer, dim)
    guard_band = outer_h

    positions: list[np.ndarray] = []
    phases: list[int] = []

    for v in HC_outer.V:
        pos2 = np.round(v.x_a[:dim], decimals=10)
        if not _inside_fritz(pos2[0], pos2[1], r_profile, z_profile,
                             electrode_z=electrode_z,
                             inflate=guard_band):
            positions.append(pos2)
            phases.append(0)

    # Gas interior: a structured sampling inside the Fritz profile.
    gas_n_rows = 2 * (2 ** refinement_droplet)
    gas_n_cols = 2 * (2 ** refinement_droplet)
    gas_interior = _sample_gas_interior(
        r_profile=r_profile,
        z_profile=z_profile,
        n_rows=gas_n_rows, n_cols=gas_n_cols,
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
    n_interface = max(32, 4 * (2 ** refinement_droplet))
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

    # Simple pressure IC just for the plot: hydrostatic in the liquid
    # + Young-Laplace in the gas, so the pressure panel shows the
    # expected step across the interface.  This is NOT the full IC
    # used by the dynamic setup — it's just enough to make the
    # visualisation informative.
    _apply_geometry_only_ic(HC, mps, meta, electrode_z=electrode_z,
                            P0=P0, rho_liq=rho_liq, rho_gas=rho_gas,
                            g=g, gamma=gamma, R_top=R_top)

    # -- Static plot -----------------------------------------------------
    out_png = os.path.join(_FIG, 'electrolysis_bubble_fritz_2D_init.png')
    fig, axes = plot_fluid(
        HC, bV=bV, t=0.0,
        scalar_field='p', vector_field='u',
        scalar_label='Pressure [Pa]',
        vector_label='Velocity [m/s]  (zero at t=0)',
        xlim=(-L_domain * 1.05, L_domain * 1.05),
        ylim=(electrode_z - 0.05 * L_domain,
              electrode_z + 2.0 * L_domain + 0.05 * L_domain),
        save_path=out_png,
        face_alpha=0.5,
    )

    # Overlay the analytical Fritz profile and the electrode line on
    # the pressure panel so the discretised interface can be eyeballed
    # against the ODE solution.
    import matplotlib.pyplot as plt
    try:
        ax = axes[0]
        z_world = meta['z_profile'] + electrode_z + meta['height']
        ax.plot(meta['r_profile'], z_world, 'r-', lw=1.2,
                label='Fritz profile')
        ax.plot(-meta['r_profile'], z_world, 'r-', lw=1.2)
        ax.axhline(electrode_z, color='k', ls='--', lw=1.0,
                   label='electrode')
        ax.legend(loc='upper right', fontsize=8)
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Saved initial-geometry plot: {out_png}")
    except Exception as exc:   # plotting should never block the run
        print(f"(overlay failed: {exc})")
    finally:
        plt.close(fig)

    print("\nGeometry-only check complete.")
    print("To turn this into a full dynamic run, the next step is to"
          " feed ``HC, bV, mps`` into the same integrator + retopology"
          " used by ``electrolysis_bubble_2D.py``.")


def _apply_geometry_only_ic(HC, mps, meta, *,
                             electrode_z: float, P0: float,
                             rho_liq: float, rho_gas: float,
                             g: float, gamma: float,
                             R_top: float) -> None:
    """Minimal hydrostatic + Young-Laplace IC for the initial plot."""
    dim = 2
    axis = dim - 1
    L_domain = meta['L_domain']
    wall_top = electrode_z + 2.0 * L_domain
    z_top_of_bubble = electrode_z + meta['height']
    P_gas_apex = (P0 + rho_liq * g * (wall_top - z_top_of_bubble)
                  + gamma / R_top)

    for v in HC.V:
        v.u = np.zeros(dim)
        y = float(v.x_a[axis])
        vol_l = float(v.dual_vol_phase[0])
        vol_g = float(v.dual_vol_phase[1])
        if vol_l > 1e-30:
            v.p_phase[0] = P0 + rho_liq * g * (wall_top - y)
        if vol_g > 1e-30:
            v.p_phase[1] = P_gas_apex + rho_gas * g * (z_top_of_bubble - y)
        # Phase-volume-weighted aggregate for the colour panel.
        num = 0.0
        den = 0.0
        if vol_l > 1e-30:
            num += float(v.p_phase[0]) * vol_l
            den += vol_l
        if vol_g > 1e-30:
            num += float(v.p_phase[1]) * vol_g
            den += vol_g
        v.p = num / den if den > 0 else 0.0


if __name__ == '__main__':
    main()
