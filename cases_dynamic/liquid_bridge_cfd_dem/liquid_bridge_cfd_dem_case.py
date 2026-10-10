"""CFD-DEM two-particle liquid bridge case study.

Dynamic fluid film evolution with surface tension from discrete curvature
(Heron's formula).  Each particle is surrounded by a hemispherical fluid
shell that moves under surface tension via ddgclib's ``symplectic_euler``
integrator.  When the two shells approach, they connect and form a
capillary bridge dynamically.

The operator-splitting loop is:

    1. Fluid film integration (surface tension via symplectic_euler)
    2. Stokes integral (capillary force on particles from film curvature)
    3. DEM step (contact + bridge + capillary)
    4. Sync film to particles (no-slip on particle surface)
    5. Bridge detection (connect close rim vertices)

The film runs on the ``liquid_bridge_cfd_dem_3D`` preset (``phases='film'``:
Heron surface-tension force with velocity damping; ``connectivity='custom'``:
the case-local ``retopologize_surface``), recorded in ``results/methods.json``
(laneX, 2026-10-06).

Usage::

    python liquid_bridge_cfd_dem_case.py [--n-steps N]
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ddgclib.dem import dem_step, save_particles
from ddgclib.methods import PRESETS, record_methods
from cases_dynamic.liquid_bridge_cfd_dem.src._params import (
    dt, n_steps as N_STEPS_DEFAULT, record_every, gamma, print_params,
    rho_f, film_thickness, film_min_edge, film_max_edge,
    film_damping, bridge_threshold, n_fluid_sub, dt_fluid,
)
from cases_dynamic.liquid_bridge_cfd_dem.src._setup import (
    setup_liquid_bridge_cfd_dem,
)
from cases_dynamic.liquid_bridge_cfd_dem.src._fluid_film import (
    retopologize_surface,
    stokes_integral,
    detect_and_form_bridge,
    sync_film_to_particles,
    snapshot as film_snapshot,
)


def run(save_fig: bool = True, save_results: bool = True, verbose: bool = True,
        n_steps: int = N_STEPS_DEFAULT):
    """Run the CFD-DEM two-particle liquid bridge case.

    Returns
    -------
    history : list[dict]
    """
    methods = PRESETS['liquid_bridge_cfd_dem_3D']
    # ── 1. Setup ──────────────────────────────────────────────────────
    if verbose:
        print_params()
        print()

    ps, detector, contact_model, bridge_mgr, HC_film, bV_film = (
        setup_liquid_bridge_cfd_dem()
    )

    p1, p2 = ps.particles[0], ps.particles[1]
    dim = ps.dim

    # Film force of the configuration (surface tension + damping)
    dudt_fn = methods.dudt_fn(HC_film, gamma=gamma, damping=film_damping)

    # Bind surface retopologize function
    retopo_fn = partial(
        retopologize_surface,
        min_edge=film_min_edge,
        max_edge=film_max_edge,
        particle_centers=[p1.x_a[:3], p2.x_a[:3]],
        particle_radii=[p1.radius, p2.radius],
        rho_f=rho_f,
        film_thickness=film_thickness,
    )

    n_film_verts = sum(1 for _ in HC_film.V)
    n_frozen = len(bV_film)
    if verbose:
        print(f"  Film mesh: {n_film_verts} vertices "
              f"({n_frozen} frozen, {n_film_verts - n_frozen} free)")
        print(f"  Fluid sub-steps: {n_fluid_sub} x dt={dt_fluid:.2e} s")
        print()

    # ── 2. Simulation loop (operator-splitting) ───────────────────────
    history: list[dict] = []
    bridge_formed_step: int | None = None
    film_bridge_formed: bool = False

    for step in range(n_steps):
        t = step * dt

        # 2a. Fluid film integration (surface tension) on the configured
        # integrator with the surface-aware custom retopology
        methods.integrate(
            HC_film, bV_film, dudt_fn,
            dt=dt_fluid, n_steps=n_fluid_sub, custom=retopo_fn,
        )

        # 2b. Stokes integral: capillary force on particles from film
        F_cap_1 = stokes_integral(HC_film, particle_id=p1.id, gamma=gamma)
        F_cap_2 = stokes_integral(HC_film, particle_id=p2.id, gamma=gamma)

        # 2c. DEM bridge check
        n_new = bridge_mgr.check_formation(ps)
        if n_new > 0 and bridge_formed_step is None:
            bridge_formed_step = step
            if verbose:
                print(
                    f"  DEM bridge formed at step {step}, t = {t*1e3:.3f} ms, "
                    f"sep = {_sep(p1, p2, dim)*1e6:.1f} um"
                )

        # 2d. Apply capillary force from film as external force
        # (dem_step calls external_forces_fn(ps); it adds to p.force)
        def _film_forces_fn(ps_):
            for p in ps_.particles:
                if p.id == p1.id:
                    p.force[:dim] += F_cap_1[:dim]
                elif p.id == p2.id:
                    p.force[:dim] += F_cap_2[:dim]

        # DEM step with film capillary force
        dem_step(
            ps, detector, contact_model, dt, dim=dim,
            bridge_manager=bridge_mgr,
            external_forces_fn=_film_forces_fn,
        )

        # 2e. Sync film to particle positions (no-slip)
        sync_film_to_particles(
            HC_film, ps.particles,
            film_radius_factor=1.0 + film_thickness / p1.radius,
        )

        # Update retopo_fn with new particle positions
        retopo_fn = partial(
            retopologize_surface,
            min_edge=film_min_edge,
            max_edge=film_max_edge,
            particle_centers=[p1.x_a[:3].copy(), p2.x_a[:3].copy()],
            particle_radii=[p1.radius, p2.radius],
            rho_f=rho_f,
            film_thickness=film_thickness,
        )

        # 2f. Bridge detection (connect close rim vertices)
        if not film_bridge_formed:
            n_connected = detect_and_form_bridge(
                HC_film, threshold=bridge_threshold,
            )
            if n_connected > 0:
                film_bridge_formed = True
                if verbose:
                    print(
                        f"  Film bridge formed at step {step}, t = {t*1e3:.3f} ms, "
                        f"{n_connected} connections"
                    )

        # Record history
        if step % record_every == 0:
            sep = _sep(p1, p2, dim)
            history.append({
                "step": step,
                "t": t,
                "x1": p1.x_a[:dim].tolist(),
                "x2": p2.x_a[:dim].tolist(),
                "v1": p1.u[:dim].tolist(),
                "v2": p2.u[:dim].tolist(),
                "sep": sep,
                "n_bridges": bridge_mgr.active_count,
                "capillary_force_1": F_cap_1.tolist(),
                "capillary_force_2": F_cap_2.tolist(),
                "capillary_force_mag": float(np.linalg.norm(F_cap_1)),
                "film": film_snapshot(HC_film),
                "film_bridge_formed": film_bridge_formed,
            })

    # Final record
    t_final = n_steps * dt
    history.append({
        "step": n_steps,
        "t": t_final,
        "x1": p1.x_a[:dim].tolist(),
        "x2": p2.x_a[:dim].tolist(),
        "v1": p1.u[:dim].tolist(),
        "v2": p2.u[:dim].tolist(),
        "sep": _sep(p1, p2, dim),
        "n_bridges": bridge_mgr.active_count,
        "capillary_force_1": stokes_integral(HC_film, p1.id, gamma).tolist(),
        "capillary_force_2": stokes_integral(HC_film, p2.id, gamma).tolist(),
        "capillary_force_mag": float(np.linalg.norm(
            stokes_integral(HC_film, p1.id, gamma)
        )),
        "film": film_snapshot(HC_film),
        "film_bridge_formed": film_bridge_formed,
    })

    # ── 3. Post-processing ────────────────────────────────────────────
    if verbose:
        _print_summary(history, bridge_formed_step, dt)

    # ── 4. Save ───────────────────────────────────────────────────────
    if save_results:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        record_methods(results_dir / "methods.json", methods, HC_film,
                       extra={"dt": dt, "n_steps": n_steps,
                              "n_fluid_sub": n_fluid_sub, "dt_fluid": dt_fluid,
                              "gamma": gamma, "film_damping": film_damping,
                              "n_film_vertices_final": sum(1 for _ in HC_film.V)})
        save_particles(ps, t_final, results_dir / "final_state.json")
        (results_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )
        if verbose:
            print(f"\n  Results saved to {results_dir}/")

    # ── 5. Plot ───────────────────────────────────────────────────────
    if save_fig:
        fig_dir = Path(__file__).parent / "fig"
        fig_dir.mkdir(exist_ok=True)
        _make_plots(history, bridge_formed_step, dt, fig_dir, verbose)

    return history


# ── Helpers ───────────────────────────────────────────────────────────


def _sep(p1, p2, dim: int) -> float:
    return float(np.linalg.norm(p2.x_a[:dim] - p1.x_a[:dim])) - p1.radius - p2.radius


def _print_summary(history, bridge_formed_step, dt):
    print("\n" + "=" * 60)
    print("  CFD-DEM Simulation Summary (Dynamic Film)")
    print("=" * 60)
    if bridge_formed_step is not None:
        t_form = bridge_formed_step * dt
        print(f"  DEM bridge formed at step {bridge_formed_step} "
              f"(t = {t_form*1e3:.3f} ms)")
    else:
        print("  No DEM bridge formed during simulation.")

    # Film bridge
    film_bridge_steps = [h for h in history if h.get("film_bridge_formed")]
    if film_bridge_steps:
        print(f"  Film bridge formed (detected at recording)")
    else:
        print("  No film bridge formed during simulation.")

    final = history[-1]
    print(f"  Final separation  = {final['sep']*1e6:.2f} um")
    print(f"  Final time        = {final['t']*1e3:.2f} ms")
    print(f"  Active DEM bridges = {final['n_bridges']}")
    print(f"  Film capillary |F|= {final['capillary_force_mag']*1e6:.4f} uN")

    film = final.get("film")
    if film:
        print(f"  Film mesh         = {film['n_vertices']} vertices, "
              f"{film['n_edges']} edges")

    x1 = np.array(final["x1"])
    x2 = np.array(final["x2"])
    com = 0.5 * (x1 + x2)
    print(f"  CoM drift         = {np.linalg.norm(com):.2e} m")
    print("=" * 60)


def _make_plots(history, bridge_formed_step, dt, fig_dir, verbose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([h["t"] for h in history]) * 1e3
    seps = np.array([h["sep"] for h in history]) * 1e6
    v1x = np.array([h["v1"][0] for h in history]) * 1e3
    v2x = np.array([h["v2"][0] for h in history]) * 1e3
    cap_force = np.array([h["capillary_force_mag"] for h in history]) * 1e6

    t_form = bridge_formed_step * dt * 1e3 if bridge_formed_step is not None else None

    # ── Separation ────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times, seps, "b-", linewidth=1.5, label="CFD-DEM (dynamic film)")

    # Overlay pure DEM if available
    dem_hist = _load_dem_history()
    if dem_hist is not None:
        dem_t = np.array([h["t"] for h in dem_hist]) * 1e3
        dem_s = np.array([h["sep"] for h in dem_hist]) * 1e6
        ax1.plot(dem_t, dem_s, "k--", linewidth=1.0, alpha=0.6, label="DEM only")
        ax1.legend()

    if t_form is not None:
        ax1.axvline(t_form, color="r", linestyle="--", alpha=0.7,
                     label=f"Bridge (t={t_form:.1f} ms)")
        ax1.legend()
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Surface separation [um]")
    ax1.set_title("CFD-DEM: Particle Separation vs Time")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(fig_dir / "separation.png", dpi=150)
    plt.close(fig1)

    # ── Forces ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(times, cap_force, "g-", linewidth=1.5, label="Film capillary")
    if t_form is not None:
        ax2.axvline(t_form, color="r", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Force [uN]")
    ax2.set_title("CFD-DEM: Capillary Force from Film Curvature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "forces.png", dpi=150)
    plt.close(fig2)

    # ── Combined summary ──────────────────────────────────────────────
    fig3, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, seps, "b-", linewidth=1.5, label="CFD-DEM")
    if dem_hist is not None:
        axes[0].plot(dem_t, dem_s, "k--", linewidth=1.0, alpha=0.6,
                     label="DEM only")
        axes[0].legend(loc="best")
    axes[0].set_ylabel("Separation [um]")
    axes[0].set_title("CFD-DEM vs DEM --- Liquid Bridge (Dynamic Film)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, v1x, "b-", label="P1 v_x")
    axes[1].plot(times, v2x, "r--", label="P2 v_x")
    axes[1].set_ylabel("Velocity [mm/s]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, cap_force, "g-", label="Film capillary")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Force [uN]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    if t_form is not None:
        for ax in axes:
            ax.axvline(t_form, color="r", linestyle="--", alpha=0.4)

    fig3.tight_layout()
    fig3.savefig(fig_dir / "summary.png", dpi=150)
    plt.close(fig3)

    if verbose:
        print(f"  Figures saved to {fig_dir}/")


def _load_dem_history():
    """Try to load the pure-DEM history for comparison overlay."""
    dem_path = (
        Path(__file__).parent.parent / "liquid_bridge_dem" / "results" / "history.json"
    )
    if dem_path.exists():
        return json.loads(dem_path.read_text())
    return None


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="CFD-DEM liquid bridge")
    _ap.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT)
    run(n_steps=_ap.parse_args().n_steps)
