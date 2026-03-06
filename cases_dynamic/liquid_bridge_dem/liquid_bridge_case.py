"""Two-particle liquid bridge case study.

Demonstrates capillary bridge formation, agglomeration, and Hertz contact
between two wetted silica glass spheres approaching each other in vacuum.

Five-step pattern:
    1. Setup — create particles, contact model, bridge manager
    2. Simulation loop — advance with dem_step, check bridge formation
    3. Post-processing — print summary statistics
    4. Save — write history and final state
    5. Plot — matplotlib figures saved to fig/

Usage::

    python liquid_bridge_case.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Allow running from the case directory or project root
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ddgclib.dem import dem_step, save_particles
from cases_dynamic.liquid_bridge_dem.src._params import (
    dt, n_steps, record_every, print_params,
)
from cases_dynamic.liquid_bridge_dem.src._setup import setup_liquid_bridge


def run(save_fig: bool = True, save_results: bool = True, verbose: bool = True):
    """Run the two-particle liquid bridge case study.

    Returns
    -------
    history : list[dict]
        Time-series records from the simulation.
    """
    # ── 1. Setup ──────────────────────────────────────────────────────────
    if verbose:
        print_params()
        print()

    ps, detector, contact_model, bridge_mgr = setup_liquid_bridge()

    p1, p2 = ps.particles[0], ps.particles[1]
    dim = ps.dim

    # ── 2. Simulation loop ────────────────────────────────────────────────
    history: list[dict] = []
    bridge_formed_step: int | None = None

    for step in range(n_steps):
        t = step * dt

        # Check for new bridge formation
        n_new = bridge_mgr.check_formation(ps)
        if n_new > 0 and bridge_formed_step is None:
            bridge_formed_step = step
            if verbose:
                print(
                    f"  Bridge formed at step {step}, t = {t*1e3:.3f} ms, "
                    f"sep = {_sep(p1, p2, dim)*1e6:.1f} um"
                )

        # Advance DEM
        dem_step(
            ps, detector, contact_model, dt, dim=dim,
            bridge_manager=bridge_mgr,
        )

        # Record history at regular intervals
        if step % record_every == 0:
            sep = _sep(p1, p2, dim)
            cap_force = _capillary_force_mag(bridge_mgr, dim)
            history.append({
                "step": step,
                "t": t,
                "x1": p1.x_a[:dim].tolist(),
                "x2": p2.x_a[:dim].tolist(),
                "v1": p1.u[:dim].tolist(),
                "v2": p2.u[:dim].tolist(),
                "sep": sep,
                "n_bridges": bridge_mgr.active_count,
                "capillary_force_mag": cap_force,
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
        "capillary_force_mag": _capillary_force_mag(bridge_mgr, dim),
    })

    # ── 3. Post-processing ────────────────────────────────────────────────
    if verbose:
        _print_summary(history, bridge_formed_step, dt)

    # ── 4. Save ───────────────────────────────────────────────────────────
    if save_results:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        save_particles(ps, t_final, results_dir / "final_state.json")
        (results_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )
        if verbose:
            print(f"\n  Results saved to {results_dir}/")

    # ── 5. Plot ───────────────────────────────────────────────────────────
    if save_fig:
        fig_dir = Path(__file__).parent / "fig"
        fig_dir.mkdir(exist_ok=True)
        _make_plots(history, bridge_formed_step, dt, fig_dir, verbose)

    return history


# ── Helpers ───────────────────────────────────────────────────────────────


def _sep(p1, p2, dim: int) -> float:
    """Surface-to-surface separation."""
    return float(np.linalg.norm(p2.x_a[:dim] - p1.x_a[:dim])) - p1.radius - p2.radius


def _capillary_force_mag(bridge_mgr, dim: int) -> float:
    """Magnitude of capillary force on the active bridge (if any)."""
    for b in bridge_mgr.bridges:
        if b.active:
            F_i, _ = b.capillary_force(dim)
            return float(np.linalg.norm(F_i))
    return 0.0


def _print_summary(history, bridge_formed_step, dt):
    """Print post-processing summary."""
    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)
    if bridge_formed_step is not None:
        t_form = bridge_formed_step * dt
        print(f"  Bridge formed at step {bridge_formed_step} "
              f"(t = {t_form*1e3:.3f} ms)")
    else:
        print("  No bridge formed during simulation.")

    final = history[-1]
    print(f"  Final separation  = {final['sep']*1e6:.2f} um")
    print(f"  Final time        = {final['t']*1e3:.2f} ms")
    print(f"  Active bridges    = {final['n_bridges']}")

    # Check centre-of-mass drift (should be ~0 by symmetry)
    x1 = np.array(final["x1"])
    x2 = np.array(final["x2"])
    com = 0.5 * (x1 + x2)
    print(f"  CoM drift         = {np.linalg.norm(com):.2e} m")
    print("=" * 60)


def _make_plots(history, bridge_formed_step, dt, fig_dir, verbose):
    """Generate and save matplotlib figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([h["t"] for h in history]) * 1e3  # ms
    seps = np.array([h["sep"] for h in history]) * 1e6  # um
    v1x = np.array([h["v1"][0] for h in history]) * 1e3  # mm/s
    v2x = np.array([h["v2"][0] for h in history]) * 1e3  # mm/s
    cap_force = np.array([h["capillary_force_mag"] for h in history]) * 1e6  # uN

    t_form = bridge_formed_step * dt * 1e3 if bridge_formed_step is not None else None

    # ── Separation vs time ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times, seps, "b-", linewidth=1.5)
    if t_form is not None:
        ax1.axvline(t_form, color="r", linestyle="--", alpha=0.7,
                     label=f"Bridge formed (t={t_form:.1f} ms)")
        ax1.legend()
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Surface separation [um]")
    ax1.set_title("Particle Separation vs Time")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(fig_dir / "separation.png", dpi=150)
    plt.close(fig1)

    # ── Velocity vs time ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(times, v1x, "b-", linewidth=1.5, label="Particle 1 (v_x)")
    ax2.plot(times, v2x, "r--", linewidth=1.5, label="Particle 2 (v_x)")
    if t_form is not None:
        ax2.axvline(t_form, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Velocity [mm/s]")
    ax2.set_title("Particle Velocities vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "velocity.png", dpi=150)
    plt.close(fig2)

    # ── Capillary force vs time ───────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(times, cap_force, "g-", linewidth=1.5)
    if t_form is not None:
        ax3.axvline(t_form, color="r", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Time [ms]")
    ax3.set_ylabel("Capillary force [uN]")
    ax3.set_title("Capillary Bridge Force vs Time")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "capillary_force.png", dpi=150)
    plt.close(fig3)

    # ── Combined summary ──────────────────────────────────────────────
    fig4, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, seps, "b-", linewidth=1.5)
    axes[0].set_ylabel("Separation [um]")
    axes[0].set_title("Two-Particle Liquid Bridge — Summary")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, v1x, "b-", label="P1 v_x")
    axes[1].plot(times, v2x, "r--", label="P2 v_x")
    axes[1].set_ylabel("Velocity [mm/s]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, cap_force, "g-", linewidth=1.5)
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel("Cap. force [uN]")
    axes[2].grid(True, alpha=0.3)

    if t_form is not None:
        for ax in axes:
            ax.axvline(t_form, color="r", linestyle="--", alpha=0.4)

    fig4.tight_layout()
    fig4.savefig(fig_dir / "summary.png", dpi=150)
    plt.close(fig4)

    if verbose:
        print(f"  Figures saved to {fig_dir}/")


if __name__ == "__main__":
    run()
