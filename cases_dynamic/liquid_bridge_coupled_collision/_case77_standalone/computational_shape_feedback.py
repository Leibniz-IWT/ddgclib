"""Mode-aware computational capillary-shape feedback for Case77.

The Case74 transport mesh is retained because it is the fast, conservative
PR33/PR35/PR37 state. Its outer-film V is already written into that live mesh.
In the low-inertia branch this module additionally records the zero-angle
Young--Laplace bridge selected by the computed bridge volume and uses its
pressure through Case77's existing junction/supply closure. That pressure
therefore participates in the next PR35 contact-supply active set; the
Young--Laplace curve is not merely a plotting curve.

The overhanging Young--Laplace meridian is an embedded capillary submodel. It
does not replace the material contact-line coordinate, so the accepted PR37
displacement remains exactly the PR35 displacement. During transient mode no
equilibrium shape is installed: the bridge remains velocity transported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import partitioned_core as model


STATE_DIR_NAME = "computational_capillary_states"
AUDIT_NAME = "case77_computational_shape_feedback_audit.json"

_INSTALLED = False
_ORIGINAL_REPROJECTION = None
_ORIGINAL_TRY_ACCEPT = None
_ORIGINAL_SAVE_SNAPSHOT = None


def _record(state: Any, values: dict[str, float]) -> None:
    trial = dict(getattr(state, "case62_trial_diag", {}))
    trial.update(values)
    state.case62_trial_diag = trial
    legacy = dict(getattr(state, "case29_last_diag", {}))
    legacy.update(values)
    state.case29_last_diag = legacy


def _manifold_state(
    state: Any,
    ring_region: np.ndarray,
    config: Any,
) -> dict[str, float] | None:
    """Return the instantaneous Young--Laplace state from computed volume."""

    surface = np.asarray(state.surface_points(), dtype=float)
    pressure_pa = model._young_laplace_bridge_junction_pressure_pa(
        surface,
        state.rings,
        ring_region,
        config,
    )
    manifold = model._EQUILIBRIUM_BRIDGE_MANIFOLD
    if pressure_pa is None or manifold is None:
        return None
    volume_ul = float(
        model.parent.bridge_inventory_volume_ul(
            surface,
            state.rings,
            ring_region,
            config,
        )
    )
    volume_grid_ul = (
        np.asarray(manifold.bridge_volume_m3, dtype=float) * 1.0e9
    )
    if (
        not math.isfinite(volume_ul)
        or volume_ul < float(volume_grid_ul[0])
        or volume_ul > float(volume_grid_ul[-1])
    ):
        return None
    contact_radius_m = float(
        np.interp(
            volume_ul,
            volume_grid_ul,
            np.asarray(manifold.contact_radius_m, dtype=float),
        )
    )
    footprint_radius_m = float(
        np.interp(
            volume_ul,
            volume_grid_ul,
            np.asarray(manifold.footprint_radius_m, dtype=float),
        )
    )
    return {
        "bridge_volume_ul": volume_ul,
        "liquid_pressure_pa": float(pressure_pa),
        "contact_radius_m": contact_radius_m,
        "footprint_radius_m": footprint_radius_m,
    }


def apply_feedback_state(
    state: Any,
    ring_region: np.ndarray,
    config: Any,
) -> None:
    """Build a retry-safe trial capillary state only in Stokes mode."""

    mode = str(getattr(state, "case77_momentum_mode", "dynamic"))
    values = {
        "case77_computational_v_mesh_active": 1.0,
        "case77_computational_young_laplace_active": 0.0,
        "case77_young_laplace_pressure_feedback_active": 0.0,
        "case77_dynamic_geometry_overwrite_active": 0.0,
        "case77_experimental_shape_input_active": 0.0,
    }
    state.case77_trial_capillary_state = None
    if mode == "stokes":
        capillary_state = _manifold_state(state, ring_region, config)
        if capillary_state is not None:
            state.case77_trial_capillary_state = dict(capillary_state)
            values.update(
                {
                    "case77_computational_young_laplace_active": 1.0,
                    "case77_young_laplace_pressure_feedback_active": 1.0,
                    "case77_computational_young_laplace_pressure_pa": float(
                        capillary_state["liquid_pressure_pa"]
                    ),
                    "case77_computational_young_laplace_contact_radius_mm": (
                        float(capillary_state["contact_radius_m"]) * 1.0e3
                    ),
                    "case77_computational_young_laplace_footprint_radius_mm": (
                        float(capillary_state["footprint_radius_m"]) * 1.0e3
                    ),
                }
            )
    _record(state, values)


def commit_feedback_state(state: Any) -> None:
    trial = getattr(state, "case77_trial_capillary_state", None)
    state.case77_accepted_capillary_state = (
        dict(trial) if trial is not None else None
    )


def _save_capillary_state(
    state: Any,
    step: int,
    time_s: float,
    out_dir: Path,
) -> None:
    accepted = getattr(state, "case77_accepted_capillary_state", None)
    if accepted is None:
        return
    state_dir = Path(out_dir) / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    time_label = f"{float(time_s):.4f}".replace(".", "p")
    np.savez_compressed(
        state_dir
        / f"case77_capillary_state_step{int(step):07d}_t{time_label}s.npz",
        time_s=np.asarray(float(time_s)),
        step=np.asarray(int(step)),
        bridge_volume_ul=np.asarray(float(accepted["bridge_volume_ul"])),
        liquid_pressure_pa=np.asarray(float(accepted["liquid_pressure_pa"])),
        contact_radius_m=np.asarray(float(accepted["contact_radius_m"])),
        footprint_radius_m=np.asarray(float(accepted["footprint_radius_m"])),
        source=np.asarray(
            "computed bridge volume -> zero-angle Young-Laplace manifold"
        ),
        used_by_next_pr35_supply_active_set=np.asarray(True),
        experimental_data_used=np.asarray(False),
    )


def install() -> None:
    """Install feedback after Case77's ordinary Case74-derived operators."""

    global _INSTALLED
    global _ORIGINAL_REPROJECTION, _ORIGINAL_TRY_ACCEPT, _ORIGINAL_SAVE_SNAPSHOT
    if _INSTALLED:
        return
    _ORIGINAL_REPROJECTION = model.case28.adaptive_radial_reprojection
    _ORIGINAL_TRY_ACCEPT = model.case28.try_accept_velocity_step
    _ORIGINAL_SAVE_SNAPSHOT = model.case28.save_snapshot

    def reprojection(state, ring_region, config, time_s=None, dt_s=0.0):
        _ORIGINAL_REPROJECTION(
            state,
            ring_region,
            config,
            time_s=time_s,
            dt_s=dt_s,
        )
        apply_feedback_state(state, ring_region, config)
        if float(dt_s) <= 0.0:
            commit_feedback_state(state)

    def try_accept(*args, **kwargs):
        accepted, scale, quality = _ORIGINAL_TRY_ACCEPT(*args, **kwargs)
        if bool(accepted):
            commit_feedback_state(args[0])
        return accepted, scale, quality

    def save_snapshot(
        state,
        ring_region,
        step,
        time_s,
        bridge_volume_ul,
        config,
        out_dir,
    ):
        _ORIGINAL_SAVE_SNAPSHOT(
            state,
            ring_region,
            step,
            time_s,
            bridge_volume_ul,
            config,
            out_dir,
        )
        _save_capillary_state(state, step, time_s, Path(out_dir))

    model.case28.adaptive_radial_reprojection = reprojection
    model.case28.try_accept_velocity_step = try_accept
    model.case28.save_snapshot = save_snapshot
    _INSTALLED = True


def write_audit(out_dir: Path) -> Path:
    """Write a compact machine-readable implementation audit."""

    output = Path(out_dir).expanduser().resolve()
    states = sorted((output / STATE_DIR_NAME).glob("*.npz"))
    audit = {
        "case": "Case 77",
        "best_treatment": {
            "film_v_shape": "accepted live tetra height field",
            "quasi_static_bridge": (
                "computed Young-Laplace manifold pressure feeds the next "
                "PR35 conservative contact-supply active set"
            ),
            "dynamic_bridge": "PR35 velocity transport; no YL overwrite",
            "contact_line": "unchanged PR37/PR35 accepted displacement",
            "volume": "conservative bridge/film inventory",
        },
        "computational_capillary_state_count": len(states),
        "global_velocity_clipping": False,
        "K_lub": 0.0,
        "experimental_data_used": False,
    }
    path = output / AUDIT_NAME
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return path
