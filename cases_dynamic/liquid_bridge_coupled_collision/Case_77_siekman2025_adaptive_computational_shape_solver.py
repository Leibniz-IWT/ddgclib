#!/usr/bin/env python3
"""Case 77: adaptive transient/Stokes bridge--film prediction.

Literature basis
----------------
Siekman, Mugele, Lugt, and van den Ende, "Growth dynamics of capillary
bridges," Physics of Fluids 37, 072117 (2025),
https://doi.org/10.1063/5.0267643.  Case77 uses the paper's axisymmetric
Young--Laplace bridge equations (Eq. 1), pressure/curvature relation (Eq. 2),
thin-film conservation and mobility (Eq. 3), and finite-substrate Bessel
profile (Appendix A, Eq. A3).  The moving-contact-line closure follows Cox,
"The dynamics of the spreading of liquids on a solid surface. Part 1.
Viscous flow," J. Fluid Mech. 168, 169--194 (1986),
https://doi.org/10.1017/S0022112086000332.

The corresponding equations used by this implementation are

    dr/ds = cos(phi),  dz/ds = sin(phi),
    dphi/ds = DeltaP/gamma + (rho*g/gamma)*z - sin(phi)/r,

    p = p_air + rho*g*h + gamma*kappa,
    dh/dt = -(1/r)*d(r*q)/dr,  q = -h^3/(3*mu)*dp/dr,

    h_initial(r) = h0*[I0(L/l_c)-I0(r/l_c)]/[I0(L/l_c)-1],
    l_c = sqrt(gamma/(rho*g)),

    theta_d^3 = theta_e^3 + 9*(mu*U_cl/gamma)*ln(L_macro/lambda),
    F_Cox,i = gamma*[cos(theta_e)-cos(theta_d)]*ell_i*d_adv,i.

The full three-dimensional surface force and meridional momentum equations
remain the repository PR33/PR35/PR37 operators:

    F_Heron,i = -gamma*(HN dA)_i,
    sigma_T = mu*(grad(u_T) + grad(u_T)^T),
    F_Cauchy,i = -sum_T |T|*sigma_T*grad(N_i),

    (M/dt + K_mu + K_gamma) u^(n+1) + B^T p
        = M u^n/dt + F_Heron + F_Cox + F_hydro,
    B u^(n+1) = (V_target - V)/dt.

Here K_gamma is only the backward-Euler linearization of the unchanged Heron
force.  It is not a second capillary force.  Four uniform P1 elements through
the film thickness give five node levels and resolve K_mu with 1.5625 percent
mobility error for the no-slip/shear-free channel.  No K_lub matrix and no
global u_CFL velocity clipping are used.

Case77 owns the adaptive transient/quasi-static PR33/PR35/PR37 bridge,
CFL/backtracking/ALE controls, source-state audit trail, and conservative
axisymmetric-film model.  Its two-sided geometric correction retains both
sides of the computed viscocapillary junction radius.
The hydraulic bridge demand is the conservative inner through-flow of the
outer-film PDE.  On the bridge side, a minimum-curvature Hermite sublayer
matches the undisturbed film to the solved junction height across the computed
viscocapillary length.

In the quasi-static branch, the computed bridge volume selects a
Young--Laplace manifold state whose pressure is fed into the next PR35 active
set.  The V-shaped film remains a transported state.  Saved accepted states
are then tetrahedralized as one bridge--film volume for audit and rendering;
no second V-shaped profile is imposed.  The finite-substrate rim is the
analytic capillary--gravity Bessel profile calculated from rho, g, gamma, h0,
and the 12 mm substrate radius.

Local and far deficits are separate accepted transport inventories:

    dV_far/dt = Q_far,  V_local = V_hydraulic - V_far.

Their capillary-leveling kernels are normalized to those forward volumes and
form the solved film state.  No experimental height, radius, width, time, or
profile enters this evolution.

No experimental height, radius, bridge volume, trough width, or fitted curve
is available to the forward simulation.  Experimental data are introduced
only by the separate post-run renderer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from _case77_standalone import computational_shape_feedback
from _case77_standalone import partitioned_core as model


ROOT = Path(__file__).resolve().parent
CASE_STEM = Path(__file__).stem
CASE_LABEL = "Case 77"
OUTPUT_PREFIX = "case77"
OUT_DIR = ROOT / CASE_STEM
PREDICTION_OUTPUT = (
    ROOT / "case77_best_treatment_finite_initial_0to3600"
)
TETRA_DIR_NAME = "mesh_states_tetra_volume"
YL_SURFACE_DIR_NAME = "young_laplace_mesh_states"
YL_TETRA_DIR_NAME = "young_laplace_tetra_states"
YL_METRICS_NAME = "case77_young_laplace_reconstruction_metrics.json"
STANDALONE_MANIFEST_NAME = "case77_standalone_solver_manifest.json"
SWITCH_AUDIT_NAME = "case77_adaptive_momentum_switch_audit.json"
GAP_AUDIT_NAME = "case77_through_gap_resolution_audit.json"
DEFAULT_GAP_ELEMENTS = 4
DEFAULT_AZIMUTHAL_SECTORS = 32

# Expose the Case77-owned internal modules/configuration to audited renderers.
parent = model.parent
case28 = model.case28
base = model.base
CONFIG = model.CONFIG


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_standalone_solver_manifest(out_dir: Path) -> Path:
    """Record the Case77-owned runtime source and dependency boundary."""

    output = out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_dir = ROOT / "_case77_standalone"
    core_paths = sorted(package_dir.rglob("*.py"))
    numbered_imports: list[str] = []
    for path in core_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            if (
                stripped.startswith("import Case_")
                or stripped.startswith("from Case_")
            ):
                numbered_imports.append(
                    f"{path.name}:{line_number}:{stripped}"
                )
    if numbered_imports:
        raise RuntimeError(
            "Case77 standalone core imports a numbered case: "
            + "; ".join(numbered_imports)
        )
    sources = [Path(__file__).resolve(), *core_paths]
    manifest = {
        "case": CASE_LABEL,
        "standalone_forward_solver": True,
        "imports_numbered_case_modules": False,
        "operator_boundary": "_case77_standalone.operators",
        "experimental_data_used_in_forward_solver": False,
        "legacy_history_prefixes": {
            "case29_": "retained field names for output compatibility",
            "case62_": "retained field names for output compatibility",
        },
        "source_files": {
            str(path.relative_to(ROOT)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sources
        },
    }
    manifest_path = output / STANDALONE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["case77_standalone_solver"] = {
            "manifest": str(manifest_path),
            "imports_numbered_case_modules": False,
            "operator_boundary": "_case77_standalone.operators",
        }
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest_path


def write_adaptive_switch_audit(out_dir: Path) -> Path:
    """Summarize only accepted Case77 momentum-mode decisions."""

    output = out_dir.expanduser().resolve()
    history_path = output / "case77_real_mesh_evolution_history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def valid_rows(name: str) -> list[dict[str, str]]:
        return [
            row
            for row in rows
            if row.get(name, "").strip()
        ]

    diagnostic_rows = valid_rows("case77_momentum_mode_code")
    ratio_rows = valid_rows("case77_inertia_ratio")
    switch_rows = [
        row
        for row in valid_rows("case77_switch_event")
        if abs(float(row["case77_switch_event"])) > 0.5
    ]
    dynamic_rows = [
        row
        for row in diagnostic_rows
        if float(row["case77_momentum_mode_code"]) > 0.5
    ]
    stokes_rows = [
        row
        for row in diagnostic_rows
        if float(row["case77_momentum_mode_code"]) <= 0.5
    ]
    audit = {
        "case": CASE_LABEL,
        "model": "adaptive backward-Euler transient / quasi-static Stokes",
        "experimental_data_used": False,
        "trigger": (
            "||M(u_new-u_old)/dt||_2 / ||F_external||_2"
        ),
        "dynamic_to_stokes": {
            "ratio_below": float(
                case28.DYNAMIC_TO_STOKES_INERTIA_RATIO
            ),
            "consecutive_solves": int(
                case28.DYNAMIC_TO_STOKES_CONSECUTIVE_SOLVES
            ),
        },
        "stokes_to_dynamic": {
            "ratio_above": float(
                case28.STOKES_TO_DYNAMIC_INERTIA_RATIO
            ),
            "stokes_trial_rejected_and_resolved_dynamically": True,
        },
        "accepted_dynamic_rows": len(dynamic_rows),
        "accepted_stokes_rows": len(stokes_rows),
        "switches": [
            {
                "time_s": float(row["t_s"]),
                "event": (
                    "dynamic_to_stokes"
                    if float(row["case77_switch_event"]) > 0.0
                    else "stokes_to_dynamic_resolve"
                ),
                "inertia_ratio": float(
                    row.get("case77_inertia_ratio", 0.0)
                ),
                "rejected_stokes_trial_ratio": float(
                    row.get(
                        "case77_rejected_stokes_trial_inertia_ratio",
                        0.0,
                    )
                    or 0.0
                ),
            }
            for row in switch_rows
        ],
        "minimum_inertia_ratio": (
            min(float(row["case77_inertia_ratio"]) for row in ratio_rows)
            if ratio_rows
            else None
        ),
        "maximum_inertia_ratio": (
            max(float(row["case77_inertia_ratio"]) for row in ratio_rows)
            if ratio_rows
            else None
        ),
    }
    audit_path = output / SWITCH_AUDIT_NAME
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["case77_adaptive_momentum_switch"] = {
            **audit,
            "audit": str(audit_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return audit_path


def _p1_free_surface_channel_mobility(vertical_elements: int) -> float:
    """Return normalized P1 flux for unit h, mu and pressure gradient."""

    elements = int(vertical_elements)
    if elements < 1:
        raise ValueError("vertical_elements must be positive")
    coordinates = np.linspace(0.0, 1.0, elements + 1)
    stiffness = np.zeros((elements + 1, elements + 1), dtype=float)
    load = np.zeros(elements + 1, dtype=float)
    for element in range(elements):
        width = float(coordinates[element + 1] - coordinates[element])
        local_stiffness = np.asarray(((1.0, -1.0), (-1.0, 1.0))) / width
        local_load = np.asarray((0.5, 0.5)) * width
        ids = np.asarray((element, element + 1), dtype=int)
        stiffness[np.ix_(ids, ids)] += local_stiffness
        load[ids] += local_load
    # No slip at z=0; the zero-shear free-surface condition at z=h is the
    # natural boundary condition of this weak Stokes problem.
    velocity = np.zeros(elements + 1, dtype=float)
    velocity[1:] = np.linalg.solve(stiffness[1:, 1:], load[1:])
    return float(np.trapezoid(velocity, coordinates))


def write_gap_resolution_audit(
    out_dir: Path,
    gap_elements: int,
) -> Path:
    """Prove that resolved K_mu replaces K_lub in Case77's film gap."""

    output = out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    exact_mobility = 1.0 / 3.0
    convergence = []
    for elements in range(1, max(7, int(gap_elements) + 1)):
        mobility = _p1_free_surface_channel_mobility(elements)
        convergence.append(
            {
                "vertical_elements": elements,
                "node_levels": elements + 1,
                "normalized_flux": mobility,
                "exact_normalized_flux": exact_mobility,
                "relative_error_percent": (
                    100.0 * abs(mobility - exact_mobility) / exact_mobility
                ),
            }
        )
    selected = convergence[int(gap_elements) - 1]
    mesh_state_paths = sorted(
        (output / "mesh_states").glob(f"{OUTPUT_PREFIX}_real_mesh_*.npz")
    )
    final_mesh = None
    if mesh_state_paths:
        latest_time = -math.inf
        for path in mesh_state_paths:
            with np.load(path, allow_pickle=True) as state:
                state_time = float(state["time_s"])
                if state_time >= latest_time:
                    latest_time = state_time
                    final_mesh = {
                        "path": str(path.resolve()),
                        "time_s": state_time,
                        "nodes": int(len(state["tetra_vertices_m"])),
                        "tetrahedra": int(len(state["tetra_cells"])),
                    }
    audit = {
        "case": CASE_LABEL,
        "physical_problem": "no-slip substrate / shear-free liquid surface",
        "exact_velocity_profile": "u(z)=(-dp/dr)*(h*z-z^2/2)/mu",
        "exact_flux_mobility": "q/(-dp/dr)=h^3/(3*mu)",
        "finite_element_error": "1/(4*N^2) for N uniform P1 elements",
        "declared_mobility_error_tolerance_percent": 2.0,
        "selected_vertical_elements": int(gap_elements),
        "selected_node_levels": int(gap_elements) + 1,
        "selected_relative_error_percent": float(
            selected["relative_error_percent"]
        ),
        "resolution_pass": bool(
            float(selected["relative_error_percent"]) <= 2.0
        ),
        "K_lub_model": "none",
        "K_lub_coefficient": 0.0,
        "sphere_wedge_extra_resistance": bool(
            case28.SPHERE_WEDGE_LUBRICATION_ENABLED
        ),
        "sphere_wedge_scope": "disabled",
        "contact_line_closure": "implicit PR37 Cox force only",
        "bulk_viscosity_operator": "PR35 K_mu with physical mu",
        "global_velocity_clipping": False,
        "convergence": convergence,
        "final_forward_mesh": final_mesh,
        "experimental_data_used": False,
    }
    audit_path = output / GAP_AUDIT_NAME
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["case77_resolved_gap"] = {
            **audit,
            "audit": str(audit_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return audit_path


def activate_case77_operators(
    gap_elements: int = DEFAULT_GAP_ELEMENTS,
    azimuthal_sectors: int = DEFAULT_AZIMUTHAL_SECTORS,
) -> None:
    """Select Case77's reduced, unclipped adaptive momentum controller."""

    global CONFIG

    if int(azimuthal_sectors) < 8:
        raise ValueError("Case77 requires at least eight azimuthal sectors")

    model.CASE_STEM = CASE_STEM
    model.CASE_LABEL = CASE_LABEL
    model.OUTPUT_PREFIX = OUTPUT_PREFIX
    model.OUT_DIR = OUT_DIR
    model.PREDICTION_OUTPUT = PREDICTION_OUTPUT
    model.TETRA_DIR_NAME = TETRA_DIR_NAME
    model.CASE_ENTRY_SOURCE = Path(__file__).resolve()
    # Resolve the singular point contact by the repository's declared
    # 0.06-micrometre sphere-clearance nucleus.  The radius follows directly
    # from sphere geometry, r=sqrt(2 R delta-delta^2), rather than from an
    # experimental bridge trajectory.
    seed_clearance_m = 0.06e-6
    sphere_radius_m = float(model.CONFIG.sphere_radius_mm) * 1.0e-3
    seed_radius_m = math.sqrt(
        max(
            2.0 * sphere_radius_m * seed_clearance_m
            - seed_clearance_m**2,
            0.0,
        )
    )
    h0_m = float(model.CONFIG.initial_film_thickness_um) * 1.0e-6
    capillary_length_m = math.sqrt(
        float(model.CONFIG.surface_tension_n_m)
        / max(
            float(model.CONFIG.density_kg_m3)
            * float(model.CONFIG.gravity_m_s2),
            1.0e-30,
        )
    )
    # The Cox outer scale spans the capillary recovery arm plus its
    # film-to-wedge matching layer.  Both lengths follow from the declared
    # material and h0; no experimental trajectory or fitted speed is used.
    cox_outer_length_m = (
        capillary_length_m
        + math.sqrt(h0_m * capillary_length_m)
    )
    model.CONFIG = replace(
        model.CONFIG,
        initial_bridge_radius_mm=seed_radius_m * 1.0e3,
        contact_line_cox_macro_length_m=cox_outer_length_m,
        # Keep Case73's converged closed 3-D Heron surface. The
        # exact-axisymmetric PR35 solve removes the unused azimuthal velocity
        # mode, so the additional sectors affect force quadrature rather than
        # the reduced meridional unknown count.
        azimuthal_nodes=int(azimuthal_sectors),
    )
    CONFIG = model.CONFIG
    # Assemble the unchanged physical forces, including the exact Heron array,
    # on the full 3-D mesh.  Only the PR35 velocity/pressure trial and test
    # spaces are reduced to common meridional ring coefficients.
    case28.ENFORCE_EXACT_AXISYMMETRIC_PR35 = True
    case28.AXISYMMETRY_TOLERANCE_M = 1.0e-11
    # Case77 never modifies the solved field with the inherited componentwise
    # u_CFL clip.  CFL/backtracking remains an admissibility rejection check.
    case28.GLOBAL_VELOCITY_CLIPPING_ENABLED = False
    case28.POSTSOLVE_CONTACT_SPEED_OVERWRITE_ENABLED = False
    # Q_J is applied through the conservative PR33 receiver/donor balance.
    # The matching contact-line active set is also required once the finite
    # local donor becomes limiting: it enforces dV_cap/dt <= Q_available as a
    # force-space reaction inside PR35.  It is not a post-solve velocity
    # overwrite or a global CFL clip.  Before donor depletion its upper bound
    # is infinite, so the validated 0--10 s startup branch is unchanged.
    case28.MASS_LIMITED_CONTACT_LINE_ENABLED = True
    case28.MASS_SUPPLY_CONTACT_REACTION_ENABLED = True
    case28.INCLUDE_YOUNG_WALL_FORCE = False
    case28.INITIALIZE_CONTACT_MOMENTUM_FROM_COX_GEOMETRY = True
    case28.INITIAL_CONTACT_MOMENTUM_M_S = 1.25e-2
    # PR37 is coupled as a force inside the PR35 solve. Linearize the exact
    # PR33 Heron force on the complete moving free surface so both interior
    # capillary waves and the first-contact geometry are backward-Euler stable.
    case28.IMPLICIT_COX_FORCE_SOLVE_ENABLED = True
    case28.IMPLICIT_CAPILLARY_STIFFNESS_ENABLED = True
    case28.IMPLICIT_CAPILLARY_NORMAL_PROJECTION_ENABLED = False
    case28.IMPLICIT_CAPILLARY_STIFFNESS_AT_CONTACT_LINE = True
    # Resolve the wall-normal Stokes profile directly. Four uniform P1
    # elements give a 1.5625% thin-film mobility error, below the declared 2%
    # discretization tolerance.  No planar-film or sphere-side lubrication
    # matrix is added: all viscous resistance is supplied by the resolved
    # physical-viscosity PR35 K_mu operator.
    selected_gap_elements = int(gap_elements)
    if selected_gap_elements < 1:
        raise ValueError("Case77 requires at least one through-gap element")
    case28.THROUGH_GAP_VERTICAL_ELEMENTS = selected_gap_elements
    # Remove inherited comparison-facing height controls.  The only lower
    # bound left in the forward solve is the one-nanometre positivity/admissibility
    # threshold in the conservative film discretization; it is not a target
    # trough height and rejected states are not clipped to it.
    model.CONFIG = replace(
        model.CONFIG,
        attached_neck_floor_enabled=False,
        dynamic_min_height_enabled=False,
        attached_outer_deficit_soft_lower_enabled=False,
    )
    CONFIG = model.CONFIG
    case28.SPHERE_WEDGE_LUBRICATION_ENABLED = False
    case28.SPHERE_WEDGE_UNRESOLVED_ONLY = False
    model.RESOLVE_TWO_SIDED_JUNCTION_LAYER = True
    # The through-gap Stokes resistance is resolved by K_mu and is not added
    # again here.  Once the capillary disturbance reaches the far reservoir,
    # however, its bridge-side and outer-film supply arms carry the same flux
    # in series.  Apply that distinct, leading-order two-arm film resistance.
    # Four explicit P1 elements resolve wall-normal gap dissipation in K_mu.
    # The separate radial film-supply path still contains the depleted outer
    # annulus and its matched capillary turn in series; this is Q_J resistance,
    # not an added K_lub momentum matrix.
    model.APPLY_UNRESOLVED_TWO_ARM_SUPPLY_RESISTANCE = True
    model.BODY_FITTED_JUNCTION_REFINEMENT_ENABLED = True
    # The accepted full-domain state is rebuilt on the quasi-static
    # Young--Laplace manifold.  Do not also reshape the transported annular
    # staging surface with a second cubic C1 overwrite.
    model.BODY_FITTED_CONSERVATIVE_C1_JOIN_ENABLED = False
    # During the moving-junction stage, the inherited remapped neck must not
    # be superposed on the conservative film solve. In the low-Ca limit its
    # accumulated quasi-static capillary neck is the correct asymptote.
    model.RETAIN_CASE61_NECK_SUPPLY_STATE = False
    model.RETAIN_CASE61_NECK_LOW_CA_ONLY = True
    model.USE_CONSERVATIVE_JUNCTION_SUPPLY_CLOSURE = True
    # Use only the measured viscosity in the resolved PR35 Cauchy operator.
    case28.WALL_LUBRICATION_DRAG_FACTOR = 1.0
    case28.LUBRICATION_RESISTANCE_MODEL = "none"
    case28.LUBRICATION_RESISTANCE_COEFFICIENT = 0.0
    # Begin with backward-Euler PR35 momentum. Switch to its zero-inertia
    # Stokes limit only after the computed inertial-force ratio remains below
    # one percent for five consecutive solves. A Stokes trial above five
    # percent is rejected and re-solved dynamically. These hysteresis values
    # are declared numerical accuracy tolerances, not experimental inputs.
    case28.ADAPTIVE_INERTIA_SWITCH_ENABLED = True
    case28.DYNAMIC_TO_STOKES_INERTIA_RATIO = 1.0e-2
    case28.STOKES_TO_DYNAMIC_INERTIA_RATIO = 5.0e-2
    case28.DYNAMIC_TO_STOKES_CONSECUTIVE_SOLVES = 5
    # The bridge consumes the physically connected film captured at contact,
    # the annulus swept by the solved contact line, and the viscocapillary
    # junction arm. No experimental time, volume, or prescribed gain is used.
    # The early connected donor consists of the capillary-support annulus,
    # the bridge-side matching turn, and every annulus swept by the accepted
    # contact line.  Its capacity is computed from the current geometry.  On
    # depletion, the same lubrication equation reconnects the far film over
    # ell_d=[gamma*h0^3*(t-t_sat)/(3*mu)]^(1/4); the local threshold is thus a
    # physical handoff, not a terminal fitted bridge-volume target.
    # The substrate film is a connected conservative donor. Do not terminate
    # transfer at a one-annulus inventory: after the local annulus is spent,
    # the current state-derived film flux continues to feed the bridge and
    # removes exactly the same volume from donor cells.
    # Depletion is a handoff to the same capillary-diffusion film equation,
    # not a terminal bridge-volume cap.
    model.USE_TERMINAL_LOCAL_DONOR_CAP = True
    model.USE_EXPANDING_CONNECTED_SUPPORT_RESERVOIR = False
    model.USE_UNIFIED_LOCAL_TO_FAR_SUPPLY = True
    model.USE_EQUILIBRIUM_PRESSURE_SUPPLY_FACTOR = True
    # Keep the pressure-boundary PDE as an audited alternative until its
    # bridge-pressure/junction branch passes the positive-gap test.  The
    # active similarity closure remains conservative and EXP-independent.
    model.USE_PRESSURE_DRIVEN_JUNCTION_FLUX = False
    model.USE_MOVING_BOUNDARY_PRESSURE_FILM = False
    # The four through-gap elements resolve the flat film and bridge-side
    # turning resistance in K_mu. Do not apply the same h^3 resistance again
    # as a cumulative scalar throat depletion in Q_J.
    model.USE_FULL_CONNECTION_JUNCTION_MOBILITY = False
    # No separate pressure-boundary junction equation is active in this
    # connected-reservoir startup branch.
    model.USE_PR35_JUNCTION_PRESSURE_CONTINUITY = False
    model.USE_HERON_JUNCTION_PRESSURE_CONTINUITY = False
    # No rendered subgrid profile enters the supply flux.
    model.USE_SUBGRID_JUNCTION_FLUX_CLOSURE = False
    # Case61 inherited a time-programmed 2% pre-activation multiplier and a
    # later fast-growth gain.  Case77 uses the computed supply flux directly:
    # no startup timer or prescribed temporal gain enters the physical supply.
    parent.SUPPLY_PRE_ACTIVATION_GAIN = 1.0
    parent.SUPPLY_FAST_GROWTH_GAIN = 1.0
    parent.SUPPLY_DEPLETION_FACTOR_EARLY = 0.0
    parent.SUPPLY_DEPLETION_FACTOR_LATE = 0.0
    # Applying the experimental-scale curvature directly to the coarse tetra
    # surface feeds an unresolved force back into PR35.  Keep the stable
    # conservative coarse solve and evaluate that core as a declared subgrid
    # observable in the renderer.
    model.USE_FLUX_DERIVED_CAPILLARY_MICRO_LAYER = False
    model._activate_case62_operators()
    # The inherited namespace activator installs K_lub. Case77 explicitly
    # removes it again after every activation so K_mu is the sole wall-normal
    # viscous resistance used by the forward momentum solve.
    case28.WALL_LUBRICATION_DRAG_FACTOR = 1.0
    case28.LUBRICATION_RESISTANCE_MODEL = "none"
    case28.LUBRICATION_RESISTANCE_COEFFICIENT = 0.0
    case28.SPHERE_WEDGE_LUBRICATION_ENABLED = False
    case28.SPHERE_WEDGE_UNRESOLVED_ONLY = False
    case28.THROUGH_GAP_VERTICAL_ELEMENTS = selected_gap_elements
    # Replace the inherited compact smoothstep rim (which was fitted to a
    # measured profile) in every forward-path namespace.  Case77 uses only
    # the analytic capillary--gravity Bessel initial condition derived from
    # rho, g, gamma, h0, and L.
    parent.finite_substrate_rim_profile_m = (
        _case77_finite_substrate_rim_profile_m
    )
    parent.physical_finite_rim_surface = (
        _case77_physical_finite_rim_surface
    )
    parent.initial_film_profile_for_validation = (
        _case77_initial_film_profile_for_validation
    )
    # Install the mode-aware capillary submodel after the final Case77
    # reprojection, acceptance, and snapshot hooks are in place.
    computational_shape_feedback.install()


def _finite_substrate_profile_for_config(
    config,
    radius_m: np.ndarray,
) -> np.ndarray:
    """Siekman's finite-substrate capillary--gravity initial profile.

    This is the analytic initial-condition model from the paper, not a
    digitized experimental curve.  It improves on Case61/Case29's fitted
    compact smoothstep while retaining their physically tapered observable.
    """

    radius = np.asarray(radius_m, dtype=float)
    substrate_radius_m = (
        float(config.substrate_radius_mm) * 1.0e-3
    )
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    capillary_length_m = math.sqrt(
        float(config.surface_tension_n_m)
        / (
            float(config.density_kg_m3)
            * float(config.gravity_m_s2)
        )
    )
    outer_i0 = float(
        np.i0(substrate_radius_m / capillary_length_m)
    )
    clipped_radius = np.clip(radius, 0.0, substrate_radius_m)
    profile = (
        h0_m
        * (
            outer_i0
            - np.i0(clipped_radius / capillary_length_m)
        )
        / max(outer_i0 - 1.0, 1.0e-30)
    )
    return np.where(
        radius <= substrate_radius_m + 1.0e-15,
        np.maximum(profile, 0.0),
        0.0,
    )


def finite_substrate_profile_m(
    radius_m: np.ndarray,
) -> np.ndarray:
    """Return Case77's analytic finite-substrate initial film."""

    return _finite_substrate_profile_for_config(CONFIG, radius_m)


def _case77_finite_substrate_rim_profile_m(
    config,
    radius_m: np.ndarray,
) -> np.ndarray:
    """Forward-path adapter for the analytic Case77 rim."""

    return _finite_substrate_profile_for_config(config, radius_m)


def _case77_physical_finite_rim_surface(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
    config,
) -> np.ndarray:
    """Apply the analytic finite rim without measured-profile fitting."""

    physical = np.asarray(surface, dtype=float).copy()
    film_rows = np.flatnonzero(np.asarray(ring_region, dtype=int) == 1)
    if film_rows.size == 0:
        return physical
    row_radius = np.asarray(
        [
            float(
                np.mean(
                    np.hypot(
                        physical[np.asarray(ring, dtype=int), 0],
                        physical[np.asarray(ring, dtype=int), 1],
                    )
                )
            )
            for ring in np.asarray(rings, dtype=int)
        ],
        dtype=float,
    )
    rim_profile = _finite_substrate_profile_for_config(
        config,
        row_radius[film_rows],
    )
    h0_m = float(config.initial_film_thickness_um) * 1.0e-6
    for row_id, rim_height_m in zip(film_rows, rim_profile):
        ids = np.asarray(rings[int(row_id)], dtype=int)
        physical[ids, 2] = np.maximum(
            physical[ids, 2] - (h0_m - float(rim_height_m)),
            0.0,
        )
    return physical


def _case77_initial_film_profile_for_validation(
    config,
    radius_mm: np.ndarray,
) -> np.ndarray:
    """Return the analytic initial profile in micrometres."""

    return (
        _finite_substrate_profile_for_config(
            config,
            np.asarray(radius_mm, dtype=float) * 1.0e-3,
        )
        * 1.0e6
    )


def physical_finite_rim_surface(
    surface: np.ndarray,
    rings: np.ndarray,
    ring_region: np.ndarray,
) -> np.ndarray:
    """Map the flat-reservoir state to the physical finite-rim observable."""

    physical = np.asarray(surface, dtype=float).copy()
    film_rows = np.flatnonzero(
        np.asarray(ring_region, dtype=int) == 1
    )
    if film_rows.size == 0:
        return physical
    row_radius = np.asarray(
        [
            float(
                np.mean(
                    np.hypot(
                        physical[np.asarray(ring, dtype=int), 0],
                        physical[np.asarray(ring, dtype=int), 1],
                    )
                )
            )
            for ring in np.asarray(rings, dtype=int)
        ],
        dtype=float,
    )
    rim_profile = finite_substrate_profile_m(
        row_radius[film_rows]
    )
    h0_m = float(CONFIG.initial_film_thickness_um) * 1.0e-6
    for row_id, rim_height_m in zip(
        film_rows,
        rim_profile,
    ):
        ids = np.asarray(rings[int(row_id)], dtype=int)
        physical[ids, 2] = np.maximum(
            physical[ids, 2]
            - (h0_m - float(rim_height_m)),
            0.0,
        )
    return physical


def initial_film_profile_for_validation(
    _config,
    radius_mm: np.ndarray,
) -> np.ndarray:
    """Return the analytic finite-rim profile in micrometres."""

    return (
        finite_substrate_profile_m(
            np.asarray(radius_mm, dtype=float) * 1.0e-3
        )
        * 1.0e6
    )


def reconstruct_quasistatic_bridge(
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Create the Case77-owned quasi-static bridge states."""

    command = [
        sys.executable,
        "-m",
        "_case77_standalone.quasistatic_reconstruction",
        "--out-dir",
        str(out_dir.expanduser().resolve()),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=ROOT, check=True)
    return out_dir.expanduser().resolve() / YL_METRICS_NAME


def seed_postrun_comparison_assets(out_dir: Path) -> Path:
    """Copy comparison data only after the isolated simulation is complete."""

    activate_case77_operators()
    output = out_dir.expanduser().resolve()
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    wall_seconds = float(
        summary.get("case62_partitioned_film_model", {}).get(
            "wall_seconds",
            0.0,
        )
    )
    model._write_case62_summary(
        output,
        summary,
        wall_seconds,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["case77_flux_inventory_observable"] = {
        "feeds_back_into_pr35_geometry": True,
        "feedback_form": (
            "quasi-static Young-Laplace pressure/manifold in the next PR35 "
            "contact-supply active set; no dynamic geometry overwrite"
        ),
        "experimental_data_used": False,
        "nodes": (
            "33 moving-junction; 161 local-donor; "
            "289 far-supply capillary-leveling"
        ),
        "moving_junction_pressure_drop": (
            "3*mu*abs(Q_J)*L_h/(2*pi*r_J*h_J^3)"
        ),
        "moving_junction_core_curvature": "DeltaP_J/gamma",
        "inner_side_slope": "(3*mu*abs(U_CL)/gamma)^(1/3)",
        "outer_side_slope": (
            "(3*mu*(abs(U_CL)+abs(Q_J)/(2*pi*r_J*h_J))/gamma)^(1/3)"
        ),
        "junction_radius": (
            "Young-Laplace contact radius plus the material capillary length "
            "sqrt(gamma/(rho*g))"
        ),
        "low_Ca_limit": (
            "after far-film replenishment activates, advect the trough with "
            "the computed junction and superpose every accepted hydraulic-"
            "deficit increment over its capillary-leveling support; gravity "
            "limits that support to the computed capillary length "
            "sqrt(gamma/(rho*g))"
        ),
        "pinned_junction_radius": (
            "before the far-supply front activates, use the Young-Laplace "
            "contact radius plus sqrt(gamma/(rho*g)); afterward use the "
            "bridge-film rim plus the same capillary length"
        ),
        "pinned_amplitude_equation": (
            "2*pi*integral(r*(h0-h_V)dr) = integral_0^t_sat max(Q_J,0)dt"
        ),
        "accepted_inventory_equations": (
            "dV_far/dt=Q_far; "
            "V_local=V_hydraulic-V_far; "
            "V_local+V_far=V_hydraulic"
        ),
        "profile_reconstruction": (
            "h=h_baseline-sum_i[(DeltaV_i/N_i)*phi_i], with "
            "N_i=2*pi*integral(r*phi_i dr)"
        ),
        "far_inventory_support": (
            "ell_i=(gamma*h0^3*age_i/(3*mu))^(1/4), limited by the "
            "capillary length"
        ),
        "junction_frame": (
            "all accepted capillary-leveling kernels move with the "
            "independently computed optical junction radius"
        ),
        "comparison_informed_profile_blending": False,
        "experimental_curve_used_in_reconstruction": False,
        "residual_film_closure": (
            "h_res=h_soft-(1-s_rep)*(h0-h_soft), bounded only by the tetra "
            "mesh admissibility height; active Case77 values give 18.4 um"
        ),
        "residual_film_uses_experimental_ordinate": False,
        "fitted_height": False,
        "fitted_radius": False,
        "fitted_width": False,
        "purpose": (
            "resolve the pressure-rounded moving V and preserve the "
            "accumulated quasi-static V below the stable tetra-film mesh scale"
        ),
    }
    physical_film_audit_path = (
        output / "case77_parameter_free_film_profiles_n161.json"
    )
    if physical_film_audit_path.is_file():
        physical_film_audit = json.loads(
            physical_film_audit_path.read_text(encoding="utf-8")
        )
        summary["case77_parameter_free_film_solver"] = {
            "active_in_renderer": False,
            "purpose": (
                "coarse outer-film convergence audit; the under-resolved "
                "junction uses the conservative capillary-length closure"
            ),
            "experimental_data_used": False,
            "shape_constants_used": False,
            "equations": physical_film_audit["equations"],
            "discretization": physical_film_audit["discretization"],
            "maximum_abs_conservation_residual_ul": physical_film_audit[
                "maximum_abs_conservation_residual_ul"
            ],
            "audit": str(physical_film_audit_path.resolve()),
        }
        summary["case77_flux_inventory_observable"][
            "conservative_capillary_junction_active_in_renderer"
        ] = True
        summary["case77_flux_inventory_observable"][
            "junction_location"
        ] = "r_J=r_CL+sqrt(gamma/(rho*g))"
    metrics_path = output / YL_METRICS_NAME
    summary["case77_case61_successor"] = {
        "adaptive_transient_stokes_momentum": True,
        "exact_axisymmetric_meridional_pr35": True,
        "full_3d_heron_force_assembly": True,
        "global_velocity_clipping": False,
        "postsolve_contact_speed_overwrite": False,
        "through_gap_vertical_elements": int(
            case28.THROUGH_GAP_VERTICAL_ELEMENTS
        ),
        "through_gap_node_levels": int(
            case28.THROUGH_GAP_VERTICAL_ELEMENTS + 1
        ),
        "K_lub_model": "none",
        "sphere_wedge_extra_resistance": bool(
            case28.SPHERE_WEDGE_LUBRICATION_ENABLED
        ),
        "sphere_wedge_unresolved_only": bool(
            case28.SPHERE_WEDGE_UNRESOLVED_ONLY
        ),
        "resolved_scales_use_K_mu_only": True,
        "resolved_gap_uses_physical_K_mu_only": True,
        "young_wall_force_in_force_inventory": False,
        "first_contact_momentum_from_geometry_cox": True,
        "implicit_capillary_stiffness_is_linearization_only": True,
        "initial_momentum_mode": "backward_euler_transient",
        "dynamic_to_stokes_inertia_ratio": float(
            case28.DYNAMIC_TO_STOKES_INERTIA_RATIO
        ),
        "stokes_to_dynamic_inertia_ratio": float(
            case28.STOKES_TO_DYNAMIC_INERTIA_RATIO
        ),
        "dynamic_to_stokes_consecutive_solves": int(
            case28.DYNAMIC_TO_STOKES_CONSECUTIVE_SOLVES
        ),
        "conservative_case77_film_retained": True,
        "computed_pressure_flux_used_without_time_gate": False,
        "mass_supply_contact_reaction_enabled": False,
        "cox_outer_length_model": (
            "sqrt(gamma/(rho*g)) + "
            "sqrt(h0*sqrt(gamma/(rho*g)))"
        ),
        "computed_inventory_depletion_used_without_fixed_floor": True,
        "quasi_static_young_laplace_bridge": (
            metrics_path.is_file()
        ),
        "young_laplace_metrics": (
            str(metrics_path.resolve())
            if metrics_path.is_file()
            else None
        ),
        "finite_substrate_observable": (
            "analytic Bessel capillary-gravity profile"
        ),
        "finite_substrate_profile_uses_experimental_curve": False,
        "experimental_evolution_curves_used": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    result = model.seed_postrun_comparison_assets(output)
    # Post-run validation is deliberately downstream of the provenance/hash
    # freeze above.  It may read EXP curves for comparison, but cannot alter
    # the already completed trajectory or any saved mesh state.
    history_path = output / f"{OUTPUT_PREFIX}_real_mesh_evolution_history.csv"
    comparison_dir = output / "comparison_only"
    with history_path.open(newline="", encoding="utf-8") as stream:
        history_rows = list(csv.DictReader(stream))
    fig1_path = comparison_dir / "siekman2025_fig1c_pdf_digitized_manual_approx.csv"
    # Validate against the isolated bottom-blue h0=100 um symbols.  The older
    # raster trace in comparison_only crosses neighbouring series and its
    # cumulative-maximum cleanup creates false plateaus.
    fig5_path = (
        ROOT
        / "Case_31_siekman2025_surface_transport_bridge_film_solver"
        / "siekman2025_fig5a_h0_100_symbols_digitized.csv"
    )
    if not fig5_path.is_file():
        raise FileNotFoundError(
            "Case77 requires the isolated Siekman Fig. 5(a) h0=100 symbol "
            f"dataset for post-run validation: {fig5_path}"
        )
    with fig1_path.open(newline="", encoding="utf-8") as stream:
        fig1_rows = list(csv.DictReader(stream))
    with fig5_path.open(newline="", encoding="utf-8") as stream:
        fig5_rows = list(csv.DictReader(stream))
    first_fig5 = next(row for row in fig5_rows if float(row["t_s"]) > 0.0)
    exp10_volume_ul = (
        float(first_fig5["Vbr_ul_monotone_used"])
        * 10.0
        / float(first_fig5["t_s"])
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["case77_postrun_fig5_reference"] = {
        "path": str(fig5_path.resolve()),
        "series": "bottom blue h0=100 um symbols only",
        "used_in_forward_simulation": False,
    }
    common_audit = {
        "experimental_data_used_in_forward_simulation": False,
        "no_global_velocity_clipping": max(
            float(row.get("case77_global_velocity_clipping_active", 0.0) or 0.0)
            for row in history_rows
        ) == 0.0,
        "no_postsolve_contact_speed_overwrite": max(
            float(row.get("case77_postsolve_contact_speed_overwrite_active", 0.0) or 0.0)
            for row in history_rows
        ) == 0.0,
        "minimum_accepted_step_scale": min(
            float(row.get("accepted_step_scale", 1.0) or 1.0)
            for row in history_rows
        ),
        "maximum_pr37_contact_displacement_residual_m": max(
            float(row.get("pr37_contact_displacement_residual_max_m", 0.0) or 0.0)
            for row in history_rows
        ),
        "maximum_pr37_contact_sphere_residual_m": max(
            float(row.get("pr37_contact_sphere_residual_max_m", 0.0) or 0.0)
            for row in history_rows
        ),
    }

    def accepted_film_minimum(target_s: float) -> tuple[float, float]:
        """Measure the local trough on the accepted bridge--film state."""

        candidates = list(
            (output / YL_SURFACE_DIR_NAME).glob(
                f"{OUTPUT_PREFIX}_young_laplace_mesh_*.npz"
            )
        )
        if not candidates:
            raise RuntimeError("Case77 accepted Young--Laplace states are missing")

        def state_time(path: Path) -> float:
            with np.load(path, allow_pickle=True) as data:
                return float(data["time_s"])

        state_path = min(candidates, key=lambda path: abs(state_time(path) - target_s))
        if abs(state_time(state_path) - float(target_s)) > 1.0e-8:
            raise RuntimeError(f"Case77 has no accepted state at t={target_s:g} s")
        with np.load(state_path, allow_pickle=True) as data:
            vertices = np.asarray(data["vertices_m"], dtype=float)
            rings = np.asarray(data["ring_index"], dtype=int)
            region = np.asarray(data["ring_region"], dtype=int)
        radius_m = np.linalg.norm(vertices[rings[:, 0], :2], axis=1)
        height_m = vertices[rings[:, 0], 2]
        # Fig. 1(c)'s h_min is the local bridge--film trough.  The physical
        # substrate rim tends to zero and is not that observable.
        local_film = (
            (region == 1)
            & (radius_m > 2.0e-3)
            & (radius_m < 5.0e-3)
        )
        if not np.any(local_film):
            raise RuntimeError("Case77 accepted state has no local film rows")
        local_ids = np.flatnonzero(local_film)
        minimum_id = int(local_ids[np.argmin(height_m[local_ids])])
        return (
            float(height_m[minimum_id] * 1.0e6),
            float(radius_m[minimum_id] * 1.0e3),
        )

    def add_validation(
        target_s: float,
        exp_volume_ul: float,
        volume_note: str,
        *,
        profile_time_s: float | None = None,
    ) -> None:
        simulation_row = min(
            history_rows,
            key=lambda row: abs(float(row["t_s"]) - float(target_s)),
        )
        if abs(float(simulation_row["t_s"]) - float(target_s)) > 1.0e-8:
            return
        comparison_profile_time_s = (
            float(target_s)
            if profile_time_s is None
            else float(profile_time_s)
        )
        exp_profile = [
            row for row in fig1_rows
            if abs(float(row["t_s"]) - comparison_profile_time_s) <= 1.0e-12
            and 2.0 < float(row["r_mm"]) < 5.0
        ]
        if not exp_profile:
            return
        exp_minimum = min(exp_profile, key=lambda row: float(row["h_um"]))
        accepted_hmin_um, accepted_rmin_mm = accepted_film_minimum(target_s)
        simulation = {
            "bridge_volume_ul": float(simulation_row["bridge_volume_ul"]),
            "minimum_film_height_um": accepted_hmin_um,
            "minimum_height_radius_mm": accepted_rmin_mm,
        }
        experiment = {
            "bridge_volume_ul": float(exp_volume_ul),
            "minimum_film_height_um": float(exp_minimum["h_um"]),
            "minimum_height_radius_mm": float(exp_minimum["r_mm"]),
        }
        error = {
            key: 100.0 * (simulation[key] - experiment[key]) / experiment[key]
            for key in simulation
        }
        summary[f"case77_postrun_validation_{int(target_s)}s"] = {
            **common_audit,
            "bridge_volume_reference": volume_note,
            "profile_reference_time_s": comparison_profile_time_s,
            "simulation_observable_source": (
                "accepted full-domain Young-Laplace bridge joined to the "
                "computed transported film"
            ),
            "simulation": simulation,
            "experiment": experiment,
            "signed_relative_error_percent": error,
            "all_three_within_5_percent": all(
                abs(value) <= 5.0 for value in error.values()
            ),
        }

    add_validation(
        10.0,
        exp10_volume_ul,
        "linear time alignment from the origin to the first measured "
        "Fig. 5(a) symbol at 11.29943503 s",
    )
    exp100_volume_ul = float(
        np.interp(
            100.0,
            [float(row["t_s"]) for row in fig5_rows],
            [float(row["Vbr_ul_monotone_used"]) for row in fig5_rows],
        )
    )
    add_validation(
        100.0,
        exp100_volume_ul,
        "linear interpolation of the two surrounding digitized Fig. 5(a) "
        "symbols at 100 s",
    )
    exp3600_volume_ul = float(
        np.interp(
            3600.0,
            [float(row["t_s"]) for row in fig5_rows],
            [float(row["Vbr_ul_monotone_used"]) for row in fig5_rows],
        )
    )
    add_validation(
        3600.0,
        exp3600_volume_ul,
        "linear interpolation of digitized Fig. 5(a) at 3600 s",
        profile_time_s=3500.0,
    )
    requested = [
        summary.get(f"case77_postrun_validation_{time_s}s", {})
        for time_s in (10, 100, 3600)
    ]
    summary["case77_all_requested_observables_within_5_percent"] = all(
        bool(record.get("all_three_within_5_percent", False))
        for record in requested
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_standalone_solver_manifest(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PREDICTION_OUTPUT)
    parser.add_argument("--prediction-0to10", action="store_true")
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument("--final-time-s", type=float, default=10.0)
    parser.add_argument("--restart-from", type=Path, default=None)
    parser.add_argument("--restart-time-s", type=float, default=None)
    parser.add_argument(
        "--continuation-dt-s",
        type=float,
        default=None,
        help=(
            "Use a larger physical step after a saved low-Ca restart. "
            "CFL/backtracking still accepts or reduces every displacement."
        ),
    )
    parser.add_argument(
        "--snapshot-interval-s",
        type=float,
        default=1.0,
        help="Physical interval between saved mesh states.",
    )
    parser.add_argument("--solve-every-steps", type=int, default=2)
    parser.add_argument(
        "--gap-elements",
        type=int,
        default=DEFAULT_GAP_ELEMENTS,
        help=(
            "Uniform P1 elements through the film thickness. Case77 uses "
            "four by default and assembles no K_lub matrix."
        ),
    )
    parser.add_argument(
        "--azimuthal-sectors",
        type=int,
        default=DEFAULT_AZIMUTHAL_SECTORS,
        help="Closed-ring sectors used by the full 3-D Heron assembly.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--record-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    activate_case77_operators(
        int(arguments.gap_elements),
        int(arguments.azimuthal_sectors),
    )
    snapshot_interval_s = float(arguments.snapshot_interval_s)
    if (
        not math.isfinite(snapshot_interval_s)
        or snapshot_interval_s <= 0.0
    ):
        raise ValueError("--snapshot-interval-s must be positive")
    model.SNAPSHOT_INTERVAL_S = snapshot_interval_s
    if arguments.continuation_dt_s is not None:
        if arguments.restart_from is None:
            raise ValueError(
                "--continuation-dt-s requires --restart-from so the "
                "0--10 s startup remains on dt=0.02 s"
            )
        continuation_dt_s = float(arguments.continuation_dt_s)
        if (
            not math.isfinite(continuation_dt_s)
            or continuation_dt_s <= 0.0
        ):
            raise ValueError("--continuation-dt-s must be positive")
        model.CONFIG = model.replace(
            model.CONFIG,
            dt_s=continuation_dt_s,
        )
    if bool(arguments.prediction_0to10) or bool(arguments.prediction):
        summary = model.run_partitioned_prediction(
            Path(arguments.out_dir),
            final_time_s=(
                10.0
                if bool(arguments.prediction_0to10)
                else float(arguments.final_time_s)
            ),
            restart_from=arguments.restart_from,
            restart_time_s=arguments.restart_time_s,
            solve_every_steps=int(arguments.solve_every_steps),
        )
        write_standalone_solver_manifest(Path(arguments.out_dir))
        write_adaptive_switch_audit(Path(arguments.out_dir))
        write_gap_resolution_audit(
            Path(arguments.out_dir),
            int(arguments.gap_elements),
        )
        computational_shape_feedback.write_audit(Path(arguments.out_dir))
        reconstruct_quasistatic_bridge(
            Path(arguments.out_dir)
        )
        # Experimental comparison is intentionally a separate downstream
        # action. A clean forward run and the Case77-only renderer require no
        # digitized profile or asset from an earlier numbered case.
    else:
        config = model._config(
            int(arguments.max_steps),
            int(arguments.record_every),
        )
        summary = case28.run_case(config, Path(arguments.out_dir))
        write_standalone_solver_manifest(Path(arguments.out_dir))
        write_adaptive_switch_audit(Path(arguments.out_dir))
        write_gap_resolution_audit(
            Path(arguments.out_dir),
            int(arguments.gap_elements),
        )
        computational_shape_feedback.write_audit(Path(arguments.out_dir))
    print(summary)


if __name__ == "__main__":
    main()
