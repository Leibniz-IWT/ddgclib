"""Case 12 axisym: #9 Pitois-2000 separation solver with #3 initial mesh.

This case keeps the Pitois separation solver path from
``Case_2b_axisym_pitois2000_separation.py`` but initializes the compute mesh
directly from the force-calibrated #3 Gmsh ``mesh_initial.msh`` file.

1. Load the Gmsh volume/surface mesh as the compute mesh.
2. Move only the top and bottom cap interiors with prescribed velocities.
3. Leave the contact-line rings free so they can slide on the spheres.
4. Compute axial bridge force through the volumetric stress path and write the
   same separation history, CSV, MSH, and PNG outputs as the base case.

Important limitation:
    This case now uses a sparse tetrahedral velocity-pressure projection for
    incompressibility. The capillary pressure driving the force balance is
    still an axisymmetric Laplace-pressure surrogate built from the evolving
    outer bridge profile.

Contact-angle note:
    The Pitois 2000 Fig. 5 setup in this script uses a baseline modeling
    assumption of ``theta = 0 deg``. This is not a directly reported Pitois
    measurement in the current workflow. It is a literature-informed
    approximation motivated by silicone oils spreading essentially completely
    in air on clean solid substrates, including oxide-like high-energy solids;
    see Svitova et al., Langmuir 2002, DOI: 10.1021/la020006x.

Gravity note:
    The Fig. 5 experiment is not gravity-free, so this case now includes a
    hydrostatic pressure contribution ``-rho g (z - z_ref)`` along the
    physical vertical ``z`` axis. This is still a surrogate model, but it is
    closer to the experiment than the old gravity-free version.

Dimensionless-computation note:
    The user-facing controls remain in SI units, but the active compute state
    is nondimensionalized with ``L0 = R``, ``T0 = sqrt(R^3 / gamma)``,
    ``P0 = gamma / R``, and ``F0 = gamma R``.  This VolFlux0_v3 variant preserves
    the old volume-mass acceleration scale while using dimensionless geometry.
    This Vol_v3 path uses a final Gmsh geometric volume restore after each
    incompressible substep instead of relying on the disabled volume-flux path
    or the expensive post-edit continuity solve for global volume control. The
    FHeron variant leaves the
    plot/history sphere force on the pressure-balance path, but replaces the
    solver-used per-vertex pressure force in ``_Ftot`` with the Heron surface
    force.

Equation/reference map used by this solver:
    Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231 (2000):
        target Fig. 5 force data, separation D/R, bridge volume, and Eq. [6]
        contact-radius estimate used by the baseline rCL = 1.540 mm cases.
    Young-Laplace / capillary pressure:
        the gorge-pressure model uses an axisymmetric liquid-bridge curvature
        surrogate and capillary scaling p ~ gamma kappa.
    ddgclib FHeron force:
        surface-tension force is computed through ddgclib Heron/dual-area
        geometry.  The DDG geometric background follows Meyer et al. (2003),
        "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds".
    Newtonian viscosity:
        the tet-Cauchy option uses sigma = -p I + mu (grad u + grad u^T);
        see standard continuum mechanics, e.g. Batchelor (1967).
    Incompressible projection:
        the volume/pressure projection follows the Chorin projection idea
        adapted to Gmsh tetrahedra with the PR33 volume-gradient operator.
    PR33 pressure force:
        F_p = B^T p is an internal PR33/ddgclib operator path.  It is active
        for cases 9-12 in _ddgclib_case_runner.py.

Case-selection caution:
    The optional bulk-viscosity term zeta div(u) I exists as a CLI option, but
    it is not selected by the canonical cases 1-12 in _ddgclib_case_runner.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import warnings
from collections import defaultdict

sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPT_STEM = SCRIPT_PATH.stem
# Default output folder: a sibling directory with the same basename as this .py file.
OUT_ROOT = SCRIPT_DIR / SCRIPT_STEM
# Keep Matplotlib's generated cache/config files inside the same output folder.
os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

try:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    coo_matrix = None
    diags = None
    spsolve = None


def _resolve_repo_root(script_dir: Path) -> Path:
    """Find the repository root when this script is copied to another folder.

    The original case file assumed it lived at least four directories below the
    repository root.  This version instead walks upward and picks the first
    parent containing the required local packages.  If those packages are not
    present, the import below will raise the normal ModuleNotFoundError with a
    clear missing-package name rather than an IndexError from path handling.
    """
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "ddgclib").is_dir() and (candidate / "cases_dynamic").is_dir():
            return candidate
    return script_dir


REPO_ROOT = _resolve_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_equilibrium.Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark import (
    HEX_TETS,
    PRISM_TETS,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import _move, _recompute_duals
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.operators.surface_tension import dual_area_heron, surface_tension_force
from ddgclib.operators.stress import (
    cauchy_stress,
    dual_area_vector,
    dual_volume,
    velocity_difference_tensor_pointwise,
)
# PR33 operator dependency used by cases 9-12.  The PR archive keeps a local
# copy beside this file, with a fallback to the original ddgclib package path
# when this solver is run from the full repo.
#   B = dV/dx is the tetrahedral volume-gradient operator.
#   F_p = B^T p maps tet pressure to nodal force.
#   B M^-1 B^T p = residual is the pressure/volume projection form.
try:
    from pr33_operators import (
        heron_forces_for_points,
        multiphase_sparse_compressible_eos_pressure_correction,
        multiphase_tet_volume_lumped_masses,
        pressure_equivalent_from_forces,
    )
except ImportError:
    from cases_dynamic.oscillating_droplet_p_ref.scripts.pr33_operators import (
        heron_forces_for_points,
        multiphase_sparse_compressible_eos_pressure_correction,
        multiphase_tet_volume_lumped_masses,
        pressure_equivalent_from_forces,
    )


class _DisplayVertex:
    def __init__(self, coords):
        self.x_a = np.asarray(coords, dtype=float)
        self.nn: set[object] = set()
        self.boundary = False

    def connect(self, other) -> None:
        self.nn.add(other)
        other.nn.add(self)


class _DisplayVertexStore(dict):
    def __init__(self):
        super().__init__()
        self._key_by_vertex_id: dict[int, tuple[float, float, float]] = {}

    def __setitem__(self, key, vertex):
        old_vertex = self.get(key)
        if old_vertex is not None and old_vertex is not vertex:
            self._key_by_vertex_id.pop(id(old_vertex), None)
        super().__setitem__(key, vertex)
        self._key_by_vertex_id[id(vertex)] = key

    def __delitem__(self, key):
        old_vertex = self.get(key)
        if old_vertex is not None:
            self._key_by_vertex_id.pop(id(old_vertex), None)
        super().__delitem__(key)

    def clear(self):
        self._key_by_vertex_id.clear()
        super().clear()

    def __missing__(self, key):
        vertex = _DisplayVertex(key)
        self[key] = vertex
        return vertex

    def __iter__(self):
        return iter(self.values())

    def move(self, vertex, pos):
        old_key = self._key_by_vertex_id.get(id(vertex))
        if old_key is not None and self.get(old_key) is vertex:
            del self[old_key]
        else:
            for key, value in list(self.items()):
                if value is vertex:
                    del self[key]
                    break
        vertex.x_a = np.asarray(pos, dtype=float)
        self[tuple(float(x) for x in vertex.x_a[:3])] = vertex


class Complex:
    def __init__(self, _dim: int, domain=None):
        self.dim = int(_dim)
        self.domain = domain
        self.V = _DisplayVertexStore()


# OUT_ROOT is defined near the imports so Matplotlib's generated files also stay
# inside the script-name output folder.
FIG5_COMPARE_PNG_NAME = "1_pitois2000_volumetric_separation_digitized_compare.png"
CL_FTOT_RADIAL_REALUSED_PNG_NAME = "cl_ftot_radial_realused.png"
HARDCODED_INITIAL_MSH_PATH = (
    Path(__file__).resolve().parent
    / "out"
    / "Case_2b_initialmesh_t0_force_calibrated"
    / "mesh_initial.msh"
)

# Case #12 reproduction notes
# ---------------------------
# These defaults intentionally preserve the old-good #9 solver setup while
# changing only the initial mesh to the force-calibrated #3 hourglass mesh:
#   out/Case_2b_initialmesh_t0_force_calibrated/mesh_initial.msh
# Intended 5000-step run:
#   case12_force_calibrated_mesh_pr33bt_kinfgate_prefzero_hydrostatic_topcl_5000
# The match used the PR#33/ddgclib pressure path with a compressible label but
# the incompressible-limit projection enabled, zero PR33 pressure reference, tet
# Cauchy viscous force, hydrostatic force referenced at the top contact line,
# and PR33 contact-line mobility enabled. This runner intentionally preserves
# that reproducibility setup and changes only the starting mesh from #9 to #3.


def _required_initial_msh_path(config: "VolumetricPitoisConfig | None" = None) -> Path:
    configured = ""
    if config is not None:
        configured = str(getattr(config, "initial_msh_path", "") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
    else:
        path = HARDCODED_INITIAL_MSH_PATH
    if path.suffix.lower() != ".msh":
        raise RuntimeError(f"Initial mesh must be a .msh file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required initial Gmsh mesh not found: {path}")
    return path


class _TimingBlock:
    def __init__(self, timing: dict[str, float] | None, name: str):
        self.timing = timing
        self.name = name
        self.start = 0.0

    def __enter__(self):
        if self.timing is not None:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timing is not None:
            self.timing[self.name] += time.perf_counter() - self.start
        return False


def _time_module(timing: dict[str, float] | None, name: str) -> _TimingBlock:
    return _TimingBlock(timing, name)


def _print_step_timing(timing: dict[str, float], *, completed_step: int, substeps: int) -> None:
    if not timing:
        return
    preferred_order = (
        "dt_select",
        "cap_move_axisym",
        "mass_dual",
        "pressure_scalar",
        "pressure_projection",
        "force_accel",
        "velocity_project",
        "volume_flux",
        "mesh_move",
        "contact_line",
        "post_continuity",
        "final_mass_pressure",
        "record",
        "history_png",
        "mesh_png",
    )
    parts = []
    for name in preferred_order:
        value = float(timing.get(name, 0.0))
        if value > 0.0:
            parts.append(f"{name} {value:.4f}s")
    extra_names = sorted(
        name for name, value in timing.items()
        if name not in preferred_order and name != "step_total" and float(value) > 0.0
    )
    for name in extra_names:
        parts.append(f"{name} {float(timing[name]):.4f}s")
    total = float(timing.get("step_total", sum(float(v) for v in timing.values())))
    print(
        "Compute time modules      = "
        + " | ".join(parts)
        + f" | total {total:.4f}s | substeps {int(substeps)} | step {int(completed_step)}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# USER CONTROLS
# Edit these values at the top of the file when you want to change the
# separation run or open an interactive viewer without touching the code below.
# ---------------------------------------------------------------------------
USER_REFINEMENT = 1
# Exact number of radial rings used near the contact line.
USER_CONTACT_LINE_RADIAL_RINGS = 2
# Additional outer-band rings near the contact line. Keep this consistent with
# the initialshape case so imported profiles retain the refined CL topology.
USER_EXTRA_CL_RADIAL_RINGS = 4
# 1 keeps all sidewall axial rings.  The neck has high curvature, so keep the
# full axial stack for the physical axisymmetric case.
USER_SIDEWALL_AXIAL_RING_STRIDE = 1
USER_NECK_EXTRA_AXIAL_LAYERS = 12
USER_CL_EXTRA_AXIAL_LAYERS = 6
# Radial gap ratio. Larger r is finer; adjacent gap ratio is 1.1.
USER_CL_RADIAL_BIAS_RATIO = 1.1
# Max separation iterations. Keep at 1 for the requested smoke check; increase
# here when you want a longer Fig. 5 sweep.
USER_TOTAL_STEPS = 5000
# Time-step rule:
#   USER_DT_S > 0: fixed user step.
#   USER_DT_S < 0: physical step, dt = min(dt_CL, dt_capillary, dt_mesh).
# This keeps the automatic step computed from numerical/physical limits only.
USER_DT_S = 0.01
USER_TARGET_SWEEP_TIME_S = abs(USER_DT_S) * max(1, USER_TOTAL_STEPS)
USER_ENABLE_ADAPTIVE_DT = USER_DT_S < 0.0
# Negative means no artificial ceiling; the selected value is the right_dt
# physical minimum. Positive USER_DT_S remains the fixed step.
USER_ADAPTIVE_DT_MAX_S = -1.0 if USER_DT_S < 0.0 else USER_DT_S
# Do not let a user floor override the physical capillary/mesh/contact-line
# limit. The selector only keeps a tiny numerical guard against exactly zero.
USER_ADAPTIVE_DT_MIN_S = 0.0
USER_ADAPTIVE_DT_CAPILLARY_SAFETY = 0.25
USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC = 0.15

USER_RECORD_EVERY_STEPS = 100
USER_MESH_SNAPSHOT_EVERY_STEPS = 100
USER_RENDER_RAW_MESH_SNAPSHOT_PNG = True
USER_HISTORY_PNG_EVERY_STEPS = 1
# The imported t=0 geometry is the Fig. 5 starting setup.  Do not use the first
# near-start row as independent dynamic validation in the comparison plot.
USER_FIG5_COMPARE_SKIP_INITIAL_STEPS = 0
# Preferred minimum axis-window width in mm. The renderer will automatically
# expand beyond this if the actual mesh/sphere geometry is larger.
USER_X_AXIS_MIN_MM = -1
USER_X_AXIS_MAX_MM = 1
USER_Z_AXIS_MIN_MM = -1.7
USER_Z_AXIS_MAX_MM = 1.7
USER_INITIAL_NECK_RADIUS_RATIO = 0.44
USER_ENABLE_INITIAL_RELAXATION = True
USER_INITIAL_RELAX_STEPS = 120
USER_INITIAL_RELAX_DT_S = 2.0e-4
USER_INITIAL_RELAX_DAMPING = 8.0
USER_INITIAL_RELAX_TOL_SPEED = 1.0e-7
USER_SHOW_MESH_FACES = False
USER_SHOW_MESH_VERTICES = True
USER_SHOW_REAL_COMPUTE_TRIANGLES = True
USER_SHOW_SIDE_TRIANGLE_DIAGONALS = True
# "full" shows all real compute-mesh edges/vertices in PNGs and the
# interactive window. Switch to "meridian_strip" only when you explicitly
# want a Fig. 1-style section view instead of the full 3D graph.
USER_REAL_TRIANGLE_VIEW = "full"
USER_SHOW_CONTACT_RING_OVERLAY = False
USER_SHOW_SURFACE_OVERLAY = False
USER_SURFACE_OVERLAY_ALPHA = 0.58
USER_MESH_VERTEX_SIZE = 5.0
USER_WIREFRAME_MERIDIANS = 24
USER_MESH_ALPHA = 0.99
USER_CAP_EDGE_ALPHA = 0.0
USER_FILL_PARTICLE_CAP_SURFACES = False
USER_INCLUDE_GRAVITY = True
USER_GRAVITY_MPS2 = 9.81
USER_ENABLE_SOLVER_HYDROSTATIC_FORCE = True
USER_SOLVER_HYDROSTATIC_ZREF_MODE = "top_cl"
USER_ENABLE_SOLVER_PR33_PRESSURE_FORCE = True
USER_SOLVER_PR33_PRESSURE_INCLUDE_HYDROSTATIC = False
USER_SOLVER_PR33_PRESSURE_CONTACT_LINE_MOBILITY = True
USER_SOLVER_PR33_USE_RETURNED_VELOCITY = False
USER_SOLVER_BULK_VISCOSITY_PA_S = 0.0
USER_INTEGRATION_SUBSTEPS = 1
USER_CONTACT_RADIUS_SAMPLES = 128
USER_PRESSURE_CLOSURE = "compressible"
USER_COMPRESSIBLE_BULK_MODULUS_PA = 9.7e8#1.0e5
USER_COMPRESSIBLE_USE_INCOMPRESSIBLE_LIMIT_PROJECTION = False
USER_COMPRESSIBLE_EOS_POSITION_CORRECTION_RELAXATION = 0.0
USER_COMPRESSIBLE_EOS_PRESSURE_DELTA_LIMIT_FRACTION = 1.0
USER_COMPRESSIBLE_EOS_REFERENCE_VOLUME_RELAXATION = 0.0
USER_ENABLE_INCOMPRESSIBLE_VOLUME_ALPHA_SEARCH = False
# Old good #9 keeps the incompressible-limit projection on even though the
# pressure-closure label is compressible.
USER_ENABLE_INCOMPRESSIBLE_PROJECTION = False
USER_INCOMPRESSIBLE_PROJECTION_REGULARIZATION = 1.0e-12
USER_ENABLE_VOLUME_PROJECTION = False
USER_ENABLE_PRESSURE_TRIAL_REFINEMENT = True
USER_VOLUME_PROJECTION_MAX_ITERS = 28
USER_VOLUME_PROJECTION_REL_TOL = 5.0e-10
USER_MAX_ACCELERATION = 5.0e-3
USER_ABORT_ON_NONFINITE_STATE = True
USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION = True
USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL = 5.0e-10
USER_GMSH_GLOBAL_VOLUME_RESTORE_TRIGGER_REL = 1.0e-6
USER_MAX_VOLUME_REL_ERROR_FOR_ABORT = 0.25
USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE = False
USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL = 5.0e-5
USER_CONTACT_LINE_VOLUME_SLIDE_MIN_RADIUS_FRACTION = 0.15
USER_CONTACT_LINE_VOLUME_SLIDE_MAX_RADIUS_FRACTION = 1.05
USER_ENABLE_POSITION_VOLUME_CONSTRAINT = False
USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL = 1.0e-5
USER_POSITION_VOLUME_CONSTRAINT_MAX_ITERS = 8
USER_ENABLE_KINEMATIC_VOLUME_CONSTRAINT = False
USER_KINEMATIC_VOLUME_CONSTRAINT_REL_TOL = 1.0e-8
USER_KINEMATIC_VOLUME_CONSTRAINT_MAX_ITERS = 40
USER_ENABLE_VOLUME_FLUX_CORRECTION = False
USER_VOLUME_FLUX_INCLUDE_CAP_BOUNDARY_FLUX = True
USER_VOLUME_FLUX_RELAXATION = 1.0
USER_VOLUME_FLUX_VOLUME_ERROR_GAIN = 0.0
USER_VOLUME_FLUX_MAX_NORMAL_DISP_M = 2.0e-5
USER_ENABLE_POST_EDIT_CONTINUITY_PROJECTION = False
USER_POST_EDIT_CONTINUITY_REL_TOL = 1.0e-6
USER_POST_EDIT_CONTINUITY_MAX_ITERS = 80
USER_POST_EDIT_CONTINUITY_MAX_ALPHA = 2.0
USER_POST_EDIT_CONTINUITY_EDGE_FRACTION = 2.0
USER_POST_EDIT_CONTINUITY_MAX_CORRECTION_M = 2.0e-5
USER_POST_EDIT_CONTINUITY_MAX_COORD_M = 2.0e-2
USER_GMSH_MAX_FREE_MESH_DISP_M = 2.0e-6
# Eq. 3 pressure is evaluated from a local quadratic neck fit.  The single
# waist ring is excluded because its discrete cusp otherwise dominates d2r/dz2.
# The Gmsh neck has enough refined axial layers that a wider local fit is more
# stable than the two-ring cusp-sensitive stencil.
USER_PRESSURE_NECK_FIT_SIDE_RINGS = 6
USER_ALLOW_CONTACT_LINE_GROWTH = False
USER_ENABLE_DYNAMIC_CONTACT_ANGLE = True
USER_ENABLE_GMSH_CONTACT_ANGLE_KINEMATICS = True
USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG = 25.0
USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG = 0.0
USER_CONTACT_LINE_MAX_SLIDE_UM = 5.0
USER_CONTACT_LINE_CONTINUATION_WEIGHT = 0.02
USER_CONTACT_LINE_FIT_RINGS = 4
USER_CONTACT_LINE_MIN_ANGLE_IMPROVEMENT_DEG = 0.0
USER_CONTACT_LINE_MIN_RELATIVE_IMPROVEMENT = 0.0
USER_CONTACT_LINE_LOCAL_PROFILE_FLOOR_FRACTION = 0.92
USER_CONTACT_LINE_COX_MACRO_LENGTH_M = 1.0e-3
USER_CONTACT_LINE_COX_SLIP_LENGTH_M = 2.0e-9
USER_USE_COX_CONTACT_LINE_FORCE = False
# Reported particle/sphere forces for the Pitois Fig. 5 comparison use the
# pressure-balance/gorge force.  A 1000-step validation showed that activating
# the wall-cap pressure force overuses the loaded 1.54 mm contact radius and
# overpredicts the initial Fig. 5 force by more than 2x.
USER_ENABLE_GMSH_WALL_STRESS_SPHERE_FORCE = False
USER_RECORD_GMSH_WALL_STRESS_HISTORY = False
USER_GMSH_WALL_STRESS_INCLUDE_PRESSURE = True
USER_GMSH_WALL_STRESS_INCLUDE_VISCOUS = True
USER_GMSH_WALL_STRESS_PROJECT_AXISYM = True
USER_REPORTED_GMSH_CONTACT_LINE_FORCE = False
# Report the axial capillary line reaction on the wetted spherical solid.  The
# default uses the raw simulated contact radius; Eq. [6] remains available as an
# effective-radius sensitivity check for the Pitois Fig. 5 setup.
USER_REPORTED_CAPILLARY_LINE_FORCE = True
USER_REPORTED_CAPILLARY_LINE_RADIUS_MODE = "current"  # current | pitois_eq6 | loaded | min_current_pitois_eq6
USER_INITIAL_CONTACT_RADIUS_MODE = "loaded"  # loaded | pitois_eq6 | pitois_eq6_bulged
USER_INITIAL_GEOMETRY_RADIAL_BLEND_POWER = 4.0
USER_INITIAL_BULGED_NECK_RADIUS_M = 1.10e-3
USER_INITIAL_BULGED_CONTACT_SLOPE_SCALE = -0.85
USER_GORGE_PRESSURE_MODEL = "axisym_fit"  # axisym_fit | fheron_neck | axisym_fheron_floor | axisym_initial_fheron_neck
USER_FHERON_NECK_PRESSURE_FACTOR = 1.0
USER_REPORTED_FORCE_SCALE = 1.0
USER_REPORTED_FORCE_MONOTONIC_ENVELOPE = False
# The run loops honor this value now. Keep the default serial because this
# operator is Python/object-graph heavy; 8 ThreadPool workers tested slower on
# the current axisymmetric force-cache path. Raise manually after benchmarking.
USER_ACCEL_WORKERS = 1
USER_ENFORCE_NO_SWIRL = True
USER_ENFORCE_FULL_AXISYMMETRY = True
USER_COMPUTE_DIMENSIONLESS = True

USER_OPEN_INTERACTIVE_WINDOW = False
USER_INTERACTIVE_STEP = USER_TOTAL_STEPS

USER_INTERACTIVE_ELEV_DEG =  25.0 #0#
USER_INTERACTIVE_AZIM_DEG =  45.0 #-87#


def _pitois_eq6_wetted_radius(
    *,
    particle_radius_m: float,
    bridge_volume_m3: float,
    initial_d_over_r: float,
) -> float:
    """Compute the wetted/contact radius b from Pitois et al. Eq. [6].

    Source: Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231
    (2000), Fig. 5 and Eq. [6].  Fig. 5 gives the experiment as a bridge of
    liquid 2 with volume V = 1.1 mm^3 between ruby spheres of radius R = 4 mm.
    The paper does not give a separate fixed contact radius.  Eq. [6] gives
    the cylindrical/flat-profile bridge volume

        V = (pi R / 2) * (H(b)^2 - D^2),  H(b) = D + b^2 / R,

    so the contact/wetted radius b is computed from R, V, and the starting
    separation D = (D/R)R.  With this Case_2b starting D/R this gives
    b = 1.207424315572 mm; using the first digitized Fig. 5 point gives
    b = 1.207887812431 mm.
    """
    radius = float(particle_radius_m)
    volume = float(bridge_volume_m3)
    gap = float(initial_d_over_r) * radius
    if radius <= 0.0 or volume <= 0.0:
        raise ValueError("Pitois Eq. [6] needs positive particle radius and bridge volume.")
    height_at_contact = math.sqrt(gap * gap + 2.0 * volume / (math.pi * radius))
    return math.sqrt(max(radius * (height_at_contact - gap), 0.0))


def _positive_scaled(value: float, scale: float) -> float:
    value = float(value)
    return value / float(scale) if value > 0.0 else value


def _is_dimensionless(config) -> bool:
    return bool(getattr(config, "compute_dimensionless", False))


def _scale_to_compute(config, value, scale_attr: str):
    if not _is_dimensionless(config):
        return value
    scale = max(float(getattr(config, scale_attr, 1.0)), 1.0e-300)
    arr = np.asarray(value, dtype=float) / scale
    return float(arr) if arr.ndim == 0 else arr


def _scale_from_compute(config, value, scale_attr: str):
    if not _is_dimensionless(config):
        return value
    scale = float(getattr(config, scale_attr, 1.0))
    arr = np.asarray(value, dtype=float) * scale
    return float(arr) if arr.ndim == 0 else arr


def _length_to_compute(config, value):
    return _scale_to_compute(config, value, "length_scale_m")


def _length_from_compute(config, value):
    return _scale_from_compute(config, value, "length_scale_m")


def _time_from_compute(config, value):
    return _scale_from_compute(config, value, "time_scale_s")


def _rate_from_compute(config, value):
    if not _is_dimensionless(config):
        return value
    arr = np.asarray(value, dtype=float) / max(float(getattr(config, "time_scale_s", 1.0)), 1.0e-300)
    return float(arr) if arr.ndim == 0 else arr


def _velocity_to_compute(config, value):
    return _scale_to_compute(config, value, "velocity_scale_mps")


def _velocity_from_compute(config, value):
    return _scale_from_compute(config, value, "velocity_scale_mps")


def _acceleration_to_compute(config, value):
    return _scale_to_compute(config, value, "acceleration_scale_mps2")


def _acceleration_from_compute(config, value):
    return _scale_from_compute(config, value, "acceleration_scale_mps2")


def _pressure_from_compute(config, value):
    return _scale_from_compute(config, value, "pressure_scale_pa")


def _force_from_compute(config, value):
    return _scale_from_compute(config, value, "force_scale_n")


def _volume_from_compute(config, value):
    return _scale_from_compute(config, value, "volume_scale_m3")


def _area_from_compute(config, value):
    if not _is_dimensionless(config):
        return value
    scale = float(getattr(config, "length_scale_m", 1.0)) ** 2
    arr = np.asarray(value, dtype=float) * scale
    return float(arr) if arr.ndim == 0 else arr


def _volume_flux_from_compute(config, value):
    if not _is_dimensionless(config):
        return value
    scale = float(getattr(config, "volume_scale_m3", 1.0)) / max(
        float(getattr(config, "time_scale_s", 1.0)),
        1.0e-300,
    )
    arr = np.asarray(value, dtype=float) * scale
    return float(arr) if arr.ndim == 0 else arr


def _surface_tension_from_compute(config, value):
    return _scale_from_compute(config, value, "surface_tension_scale_npm")


def _viscosity_from_compute(config, value):
    return _scale_from_compute(config, value, "viscosity_scale_pas")


def _density_from_compute(config, value):
    return _scale_from_compute(config, value, "density_scale_kgpm3")


def _contact_line_slide_limit_compute(config) -> float:
    slide_m = max(float(config.contact_line_max_slide_um) * 1.0e-6, 1.0e-9)
    return float(_length_to_compute(config, slide_m))


def _solver_mass_from_lumped_volume(config, volume):
    mass = np.asarray(volume, dtype=float)
    if (
        bool(getattr(config, "compute_dimensionless", False))
        and bool(getattr(config, "dimensionless_legacy_volume_mass", False))
    ):
        mass = mass / max(float(getattr(config, "density_scale_kgpm3", 1.0)), 1.0e-300)
    return float(mass) if mass.ndim == 0 else mass


@dataclass(frozen=True)
class VolumetricPitoisConfig:
    name: str
    title: str
    refinement: int = USER_REFINEMENT
    contact_line_radial_rings: int = USER_CONTACT_LINE_RADIAL_RINGS
    extra_cl_radial_rings: int = USER_EXTRA_CL_RADIAL_RINGS
    axial_ring_stride: int = USER_SIDEWALL_AXIAL_RING_STRIDE
    neck_extra_axial_layers: int = USER_NECK_EXTRA_AXIAL_LAYERS
    cl_extra_axial_layers: int = USER_CL_EXTRA_AXIAL_LAYERS
    contact_line_radial_bias_ratio: float = USER_CL_RADIAL_BIAS_RATIO
    particle_radius: float = 4.0e-3
    # Starting normalized surface gap. This is the existing Case_2b start value,
    # close to the first digitized Fig. 5 point D/R = 0.014403481854782.
    initial_d_over_r: float = 0.014484537138212631
    initial_msh_path: str = ""
    # Pitois Fig. 5 bridge volume: V = 1.1 mm^3 = 1.10e-9 m^3.
    initial_bridge_volume_m3: float = 1.10e-9
    # Pitois Fig. 5 gives R, V, and D/R, not a contact radius directly.
    # Therefore the default target_cap_radius is not a hard-coded 1.54 mm value;
    # it is computed as the wetted radius b from Pitois Eq. [6].
    use_pitois_eq6_contact_radius: bool = True
    target_cap_radius: float = 0.0
    initial_contact_radius_mode: str = USER_INITIAL_CONTACT_RADIUS_MODE
    initial_bulged_neck_radius: float = USER_INITIAL_BULGED_NECK_RADIUS_M
    initial_bulged_contact_slope_scale: float = USER_INITIAL_BULGED_CONTACT_SLOPE_SCALE
    initial_neck_radius_ratio: float = USER_INITIAL_NECK_RADIUS_RATIO
    enable_initial_relaxation: bool = USER_ENABLE_INITIAL_RELAXATION
    initial_relax_steps: int = USER_INITIAL_RELAX_STEPS
    initial_relax_dt: float = USER_INITIAL_RELAX_DT_S
    initial_relax_damping: float = USER_INITIAL_RELAX_DAMPING
    initial_relax_tol_speed: float = USER_INITIAL_RELAX_TOL_SPEED
    gamma: float = 2.10e-2
    mu_f: float = 1.0e-1
    rho_f: float = 965.0
    dt: float = USER_DT_S
    enable_adaptive_dt: bool = USER_ENABLE_ADAPTIVE_DT
    adaptive_dt_max_s: float = USER_ADAPTIVE_DT_MAX_S
    adaptive_dt_min_s: float = USER_ADAPTIVE_DT_MIN_S
    adaptive_dt_capillary_safety: float = USER_ADAPTIVE_DT_CAPILLARY_SAFETY
    adaptive_dt_mesh_displacement_frac: float = USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC
    n_steps: int = USER_TOTAL_STEPS
    cap_speed: float = 5.0e-6
    damping: float = 0# 2.0e-1
    max_acceleration: float = USER_MAX_ACCELERATION
    integration_substeps: int = USER_INTEGRATION_SUBSTEPS
    record_every: int = USER_RECORD_EVERY_STEPS
    mesh_snapshot_every: int = USER_MESH_SNAPSHOT_EVERY_STEPS
    render_raw_mesh_snapshot_png: bool = USER_RENDER_RAW_MESH_SNAPSHOT_PNG
    display_motion_scale: float = 4000.0
    use_axisymmetric_laplace_pressure: bool = True
    pressure_scale: float = 1.0
    enable_heron_surface_tension: bool = True
    solver_viscous_force_model: str = "tet_cauchy"
    solver_bulk_viscosity_pa_s: float = USER_SOLVER_BULK_VISCOSITY_PA_S
    include_gravity: bool = USER_INCLUDE_GRAVITY
    gravity_mps2: float = USER_GRAVITY_MPS2
    enable_solver_hydrostatic_force: bool = USER_ENABLE_SOLVER_HYDROSTATIC_FORCE
    solver_hydrostatic_zref_mode: str = USER_SOLVER_HYDROSTATIC_ZREF_MODE
    enable_solver_pr33_pressure_force: bool = USER_ENABLE_SOLVER_PR33_PRESSURE_FORCE
    solver_pr33_pressure_include_hydrostatic: bool = USER_SOLVER_PR33_PRESSURE_INCLUDE_HYDROSTATIC
    solver_pr33_pressure_contact_line_mobility: bool = USER_SOLVER_PR33_PRESSURE_CONTACT_LINE_MOBILITY
    solver_pr33_use_returned_velocity: bool = USER_SOLVER_PR33_USE_RETURNED_VELOCITY
    solver_pr33_pressure_reference_mode: str = "zero"
    contact_radius_samples: int = USER_CONTACT_RADIUS_SAMPLES
    pressure_closure: str = USER_PRESSURE_CLOSURE
    compressible_bulk_modulus_pa: float = USER_COMPRESSIBLE_BULK_MODULUS_PA
    compressible_bulk_modulus: float = USER_COMPRESSIBLE_BULK_MODULUS_PA
    compressible_use_incompressible_limit_projection: bool = (
        USER_COMPRESSIBLE_USE_INCOMPRESSIBLE_LIMIT_PROJECTION
    )
    compressible_eos_position_correction_relaxation: float = (
        USER_COMPRESSIBLE_EOS_POSITION_CORRECTION_RELAXATION
    )
    compressible_eos_pressure_delta_limit_fraction: float = (
        USER_COMPRESSIBLE_EOS_PRESSURE_DELTA_LIMIT_FRACTION
    )
    compressible_eos_reference_volume_relaxation: float = (
        USER_COMPRESSIBLE_EOS_REFERENCE_VOLUME_RELAXATION
    )
    enable_incompressible_projection: bool = USER_ENABLE_INCOMPRESSIBLE_PROJECTION
    incompressible_projection_regularization: float = USER_INCOMPRESSIBLE_PROJECTION_REGULARIZATION
    enable_volume_projection: bool = USER_ENABLE_VOLUME_PROJECTION
    volume_projection_max_iters: int = USER_VOLUME_PROJECTION_MAX_ITERS
    volume_projection_rel_tol: float = USER_VOLUME_PROJECTION_REL_TOL
    enable_gmsh_geometric_volume_correction: bool = USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION
    gmsh_geometric_volume_correction_trigger_rel: float = USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL
    max_volume_rel_error_for_abort: float = USER_MAX_VOLUME_REL_ERROR_FOR_ABORT
    pressure_neck_fit_side_rings: int = USER_PRESSURE_NECK_FIT_SIDE_RINGS
    gorge_pressure_model: str = USER_GORGE_PRESSURE_MODEL
    fheron_neck_pressure_factor: float = USER_FHERON_NECK_PRESSURE_FACTOR
    allow_contact_line_growth: bool = USER_ALLOW_CONTACT_LINE_GROWTH
    enable_dynamic_contact_angle: bool = USER_ENABLE_DYNAMIC_CONTACT_ANGLE
    enable_gmsh_contact_angle_kinematics: bool = USER_ENABLE_GMSH_CONTACT_ANGLE_KINEMATICS
    dynamic_contact_angle_max_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG
    dynamic_contact_angle_min_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG
    contact_line_max_slide_um: float = USER_CONTACT_LINE_MAX_SLIDE_UM
    contact_line_continuation_weight: float = USER_CONTACT_LINE_CONTINUATION_WEIGHT
    contact_line_fit_rings: int = USER_CONTACT_LINE_FIT_RINGS
    contact_line_cox_macro_length_m: float = USER_CONTACT_LINE_COX_MACRO_LENGTH_M
    contact_line_cox_slip_length_m: float = USER_CONTACT_LINE_COX_SLIP_LENGTH_M
    use_cox_contact_line_force: bool = USER_USE_COX_CONTACT_LINE_FORCE
    enable_gmsh_wall_stress_sphere_force: bool = USER_ENABLE_GMSH_WALL_STRESS_SPHERE_FORCE
    record_gmsh_wall_stress_history: bool = USER_RECORD_GMSH_WALL_STRESS_HISTORY
    gmsh_wall_stress_include_pressure: bool = USER_GMSH_WALL_STRESS_INCLUDE_PRESSURE
    gmsh_wall_stress_include_viscous: bool = USER_GMSH_WALL_STRESS_INCLUDE_VISCOUS
    gmsh_wall_stress_project_axisym: bool = USER_GMSH_WALL_STRESS_PROJECT_AXISYM
    reported_gmsh_contact_line_force: bool = USER_REPORTED_GMSH_CONTACT_LINE_FORCE
    reported_capillary_line_force: bool = USER_REPORTED_CAPILLARY_LINE_FORCE
    reported_capillary_line_radius_mode: str = USER_REPORTED_CAPILLARY_LINE_RADIUS_MODE
    reported_force_scale: float = USER_REPORTED_FORCE_SCALE
    reported_force_monotonic_envelope: bool = USER_REPORTED_FORCE_MONOTONIC_ENVELOPE
    accel_workers: int = USER_ACCEL_WORKERS
    enforce_no_swirl: bool = USER_ENFORCE_NO_SWIRL
    compute_dimensionless: bool = USER_COMPUTE_DIMENSIONLESS
    length_scale_m: float = 1.0
    time_scale_s: float = 1.0
    velocity_scale_mps: float = 1.0
    acceleration_scale_mps2: float = 1.0
    pressure_scale_pa: float = 1.0
    force_scale_n: float = 1.0
    volume_scale_m3: float = 1.0
    density_scale_kgpm3: float = 1.0
    viscosity_scale_pas: float = 1.0
    surface_tension_scale_npm: float = 1.0
    dimensionless_active: bool = False
    dimensionless_legacy_volume_mass: bool = True
    # Literature-informed baseline for silicone oil on clean oxide-like solids
    # in air; this is stored explicitly for the Fig. 5 separation setup.
    contact_angle_deg: float = 0.0

    @property
    def relative_speed(self) -> float:
        return self.cap_speed

    def __post_init__(self) -> None:
        pressure_closure = str(self.pressure_closure).strip().lower()
        if pressure_closure not in {"legacy", "compressible", "incompressible"}:
            raise ValueError("pressure_closure must be legacy, compressible, or incompressible.")
        object.__setattr__(self, "pressure_closure", pressure_closure)
        initial_contact_radius_mode = str(self.initial_contact_radius_mode).strip().lower()
        if initial_contact_radius_mode not in {"loaded", "pitois_eq6", "pitois_eq6_bulged"}:
            raise ValueError("initial_contact_radius_mode must be loaded, pitois_eq6, or pitois_eq6_bulged.")
        object.__setattr__(self, "initial_contact_radius_mode", initial_contact_radius_mode)
        gorge_pressure_model = str(self.gorge_pressure_model).strip().lower()
        if gorge_pressure_model not in {
            "axisym_fit",
            "fheron_neck",
            "axisym_fheron_floor",
            "axisym_initial_fheron_neck",
        }:
            raise ValueError(
                "gorge_pressure_model must be axisym_fit, fheron_neck, "
                "axisym_fheron_floor, or axisym_initial_fheron_neck."
            )
        object.__setattr__(self, "gorge_pressure_model", gorge_pressure_model)
        solver_viscous_force_model = str(self.solver_viscous_force_model).strip().lower()
        if solver_viscous_force_model not in {"edge_flux", "tet_cauchy"}:
            raise ValueError("solver_viscous_force_model must be edge_flux or tet_cauchy.")
        object.__setattr__(self, "solver_viscous_force_model", solver_viscous_force_model)
        solver_pr33_pressure_reference_mode = str(self.solver_pr33_pressure_reference_mode).strip().lower()
        if solver_pr33_pressure_reference_mode not in {"heron", "zero"}:
            raise ValueError("solver_pr33_pressure_reference_mode must be heron or zero.")
        object.__setattr__(
            self,
            "solver_pr33_pressure_reference_mode",
            solver_pr33_pressure_reference_mode,
        )
        solver_hydrostatic_zref_mode = str(self.solver_hydrostatic_zref_mode).strip().lower()
        if solver_hydrostatic_zref_mode not in {"midpoint", "top_cl"}:
            raise ValueError("solver_hydrostatic_zref_mode must be midpoint or top_cl.")
        object.__setattr__(self, "solver_hydrostatic_zref_mode", solver_hydrostatic_zref_mode)
        if (not np.isfinite(float(self.fheron_neck_pressure_factor))) or float(self.fheron_neck_pressure_factor) <= 0.0:
            raise ValueError("fheron_neck_pressure_factor must be positive and finite.")
        capillary_line_radius_mode = str(self.reported_capillary_line_radius_mode).strip().lower()
        valid_radius_modes = {"pitois_eq6", "current", "loaded", "min_current_pitois_eq6"}
        if capillary_line_radius_mode not in valid_radius_modes:
            raise ValueError(
                "reported_capillary_line_radius_mode must be pitois_eq6, current, loaded, or min_current_pitois_eq6."
            )
        object.__setattr__(self, "reported_capillary_line_radius_mode", capillary_line_radius_mode)
        object.__setattr__(
            self,
            "enable_incompressible_projection",
            bool(
                self.enable_incompressible_projection
                or pressure_closure == "incompressible"
                or (
                    pressure_closure == "compressible"
                    and bool(self.compressible_use_incompressible_limit_projection)
                )
            ),
        )
        if bool(self.dimensionless_active):
            if float(self.target_cap_radius) <= 0.0:
                raise ValueError("target_cap_radius must be positive.")
            return

        particle_radius_m = float(self.particle_radius)
        bridge_volume_m3 = float(self.initial_bridge_volume_m3)
        gamma_npm = float(self.gamma)
        rho_kgpm3 = float(self.rho_f)
        mu_pas = float(self.mu_f)
        target_cap_radius_m = float(self.target_cap_radius)
        if bool(self.use_pitois_eq6_contact_radius):
            target_cap_radius_m = _pitois_eq6_wetted_radius(
                particle_radius_m=particle_radius_m,
                bridge_volume_m3=bridge_volume_m3,
                initial_d_over_r=self.initial_d_over_r,
            )
        object.__setattr__(self, "target_cap_radius", target_cap_radius_m)
        if target_cap_radius_m <= 0.0:
            raise ValueError("target_cap_radius must be positive.")
        if not bool(self.compute_dimensionless):
            object.__setattr__(self, "compressible_bulk_modulus", float(self.compressible_bulk_modulus_pa))
            return

        if particle_radius_m <= 0.0 or gamma_npm <= 0.0 or rho_kgpm3 <= 0.0 or mu_pas <= 0.0:
            raise ValueError("Dimensionless computation needs positive R, gamma, rho, and mu.")

        length_scale = particle_radius_m
        time_scale = math.sqrt(length_scale**3 / gamma_npm)
        velocity_scale = length_scale / time_scale
        acceleration_scale = length_scale / (time_scale * time_scale)
        pressure_scale = gamma_npm / length_scale
        force_scale = gamma_npm * length_scale
        volume_scale = length_scale**3
        viscosity_scale = gamma_npm / velocity_scale

        object.__setattr__(self, "length_scale_m", length_scale)
        object.__setattr__(self, "time_scale_s", time_scale)
        object.__setattr__(self, "velocity_scale_mps", velocity_scale)
        object.__setattr__(self, "acceleration_scale_mps2", acceleration_scale)
        object.__setattr__(self, "pressure_scale_pa", pressure_scale)
        object.__setattr__(self, "force_scale_n", force_scale)
        object.__setattr__(self, "volume_scale_m3", volume_scale)
        object.__setattr__(self, "density_scale_kgpm3", 1.0)
        object.__setattr__(self, "viscosity_scale_pas", viscosity_scale)
        object.__setattr__(self, "surface_tension_scale_npm", gamma_npm)

        object.__setattr__(self, "particle_radius", 1.0)
        object.__setattr__(self, "initial_bridge_volume_m3", bridge_volume_m3 / volume_scale)
        object.__setattr__(self, "target_cap_radius", target_cap_radius_m / length_scale)
        object.__setattr__(self, "initial_bulged_neck_radius", self.initial_bulged_neck_radius / length_scale)
        object.__setattr__(self, "initial_relax_dt", _positive_scaled(self.initial_relax_dt, time_scale))
        object.__setattr__(self, "initial_relax_tol_speed", self.initial_relax_tol_speed / velocity_scale)
        object.__setattr__(self, "gamma", 1.0)
        object.__setattr__(self, "rho_f", rho_kgpm3)
        object.__setattr__(self, "mu_f", mu_pas / viscosity_scale)
        object.__setattr__(
            self,
            "solver_bulk_viscosity_pa_s",
            float(self.solver_bulk_viscosity_pa_s) / viscosity_scale,
        )
        object.__setattr__(self, "dt", _positive_scaled(self.dt, time_scale))
        object.__setattr__(self, "adaptive_dt_max_s", _positive_scaled(self.adaptive_dt_max_s, time_scale))
        object.__setattr__(self, "adaptive_dt_min_s", _positive_scaled(self.adaptive_dt_min_s, time_scale))
        object.__setattr__(self, "cap_speed", self.cap_speed / velocity_scale)
        object.__setattr__(self, "damping", self.damping * velocity_scale / force_scale)
        object.__setattr__(self, "max_acceleration", self.max_acceleration / acceleration_scale)
        object.__setattr__(self, "gravity_mps2", self.gravity_mps2 / acceleration_scale)
        object.__setattr__(self, "initial_relax_damping", self.initial_relax_damping * velocity_scale / force_scale)
        object.__setattr__(self, "contact_line_cox_macro_length_m", self.contact_line_cox_macro_length_m / length_scale)
        object.__setattr__(self, "contact_line_cox_slip_length_m", self.contact_line_cox_slip_length_m / length_scale)
        object.__setattr__(self, "compressible_bulk_modulus", float(self.compressible_bulk_modulus_pa) / pressure_scale)
        object.__setattr__(self, "dimensionless_active", True)


@dataclass
class VolumetricPitoisState:
    config: VolumetricPitoisConfig
    HC: object
    bV_caps: set
    cap_bottom: list
    cap_top: list
    cap_bottom_interior: list
    cap_top_interior: list
    bottom_contact_ring: list
    top_contact_ring: list
    cap_bottom_center: object
    cap_top_center: object
    layer_centers: list
    layer_fractions: tuple[float, ...]
    outer_rings: list
    layer_rings: list
    surface_edges: list
    surface_boundary_indices: set
    mps: object
    radial_scale: float
    axial_scale: float
    initial_gap: float
    cap_ring_factors: tuple[float, ...]
    bottom_sphere_center: np.ndarray
    top_sphere_center: np.ndarray
    target_volume_m3: float
    target_snapshot_volume_m3: float
    pressure_projection_scalar: float = 0.0
    pressure_projection_area_map: dict[int, float] = field(default_factory=dict)
    pressure_projection_normal_map: dict[int, np.ndarray] = field(default_factory=dict)
    incompressible_projection_pressure: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    incompressible_divergence_before_l2: float = 0.0
    incompressible_divergence_after_l2: float = 0.0
    elapsed_time_s: float = 0.0
    last_step_dt: float = USER_DT_S
    last_bottom_contact_line_speed: float = 0.0
    last_top_contact_line_speed: float = 0.0
    last_dt_limit_cl: float = USER_DT_S
    last_dt_limit_capillary: float = USER_DT_S
    last_dt_limit_mesh: float = USER_DT_S
    last_dt_limiter: str = "fixed"
    last_contact_line_volume_delta_m3: float = 0.0
    last_gmsh_cl_volume_slide_status: str = "not_run"
    last_gmsh_cl_volume_slide_scale: float = 1.0
    last_gmsh_cl_volume_slide_rel_before: float = 0.0
    last_gmsh_cl_volume_slide_rel_after: float = 0.0
    last_position_volume_constraint_status: str = "not_run"
    last_position_volume_constraint_rel_before: float = 0.0
    last_position_volume_constraint_rel_after: float = 0.0
    last_final_noncl_volume_projection_status: str = "not_run"
    last_final_noncl_volume_projection_rel_before: float = 0.0
    last_final_noncl_volume_projection_rel_after: float = 0.0
    last_final_noncl_volume_projection_iters: int = 0
    last_final_noncl_volume_projection_max_disp_m: float = 0.0
    last_volume_flux_lambda_mps: float = 0.0
    last_volume_flux_before_m3ps: float = 0.0
    last_volume_flux_target_m3ps: float = 0.0
    last_volume_flux_after_m3ps: float = 0.0
    last_volume_flux_corrected_area_m2: float = 0.0
    ddgclib_reference_tet_volumes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ddgclib_pressure_closure: str = "not_run"
    ddgclib_closure_pressure: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    ddgclib_closure_volume_rel_l2: float = 0.0
    ddgclib_closure_volume_rel_linf: float = 0.0
    ddgclib_closure_alpha: float = 1.0
    surface_export_vertices: list = field(default_factory=list)
    surface_export_node_ids: dict[int, int] = field(default_factory=dict)
    surface_export_side_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    surface_export_bottom_cap_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    surface_export_top_cap_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    volume_export_vertices: list = field(default_factory=list)
    volume_export_node_ids: dict[int, int] = field(default_factory=dict)
    volume_export_tets: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))
    topology_frozen: bool = False
    frozen_layer_ring_id_structure: tuple = field(default_factory=tuple)
    frozen_surface_vertex_id_order: tuple = field(default_factory=tuple)
    frozen_volume_vertex_id_order: tuple = field(default_factory=tuple)


def _refresh_cap_boundary_sets(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    state.bottom_contact_ring = list(state.layer_rings[0][-1]) if state.layer_rings and state.layer_rings[0] else []
    state.top_contact_ring = list(state.layer_rings[-1][-1]) if state.layer_rings and state.layer_rings[-1] else []
    bottom_contact_ids = {id(v) for v in state.bottom_contact_ring}
    top_contact_ids = {id(v) for v in state.top_contact_ring}
    state.cap_bottom_interior = [v for v in state.cap_bottom if id(v) not in bottom_contact_ids]
    state.cap_top_interior = [v for v in state.cap_top if id(v) not in top_contact_ids]
    state.bV_caps = set(state.cap_bottom_interior + state.cap_top_interior)


def _surface_export_vertex_order(state: VolumetricPitoisState) -> list:
    ordered: list = []
    seen: set[int] = set()

    def add(vertex) -> None:
        key = id(vertex)
        if key in seen:
            return
        seen.add(key)
        ordered.append(vertex)

    add(state.cap_bottom_center)
    for rings in state.layer_rings:
        for ring in rings:
            for vertex in ring:
                add(vertex)
    add(state.cap_top_center)
    return ordered


def _layer_ring_id_structure(state: VolumetricPitoisState) -> tuple:
    return tuple(
        tuple(tuple(id(vertex) for vertex in ring) for ring in rings)
        for rings in state.layer_rings
    )


def _surface_vertex_id_order(state: VolumetricPitoisState) -> tuple:
    return tuple(id(vertex) for vertex in _surface_export_vertex_order(state))


def _volume_export_vertex_order(state: VolumetricPitoisState) -> list:
    return list(state.HC.V)


def _assert_fixed_topology(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state, "topology_frozen", False)):
        return
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        current_volume_export_ids = tuple(id(vertex) for vertex in state.volume_export_vertices)
        if current_volume_export_ids != state.frozen_volume_vertex_id_order:
            raise RuntimeError("Fixed topology violated: Gmsh volume vertex IDs changed during time stepping")
        return
    if _layer_ring_id_structure(state) != state.frozen_layer_ring_id_structure:
        raise RuntimeError("Fixed topology violated: layer ring order/connectivity changed during time stepping")
    if _surface_vertex_id_order(state) != state.frozen_surface_vertex_id_order:
        raise RuntimeError("Fixed topology violated: surface vertex ID order changed during time stepping")
    current_export_ids = tuple(id(vertex) for vertex in state.surface_export_vertices)
    if current_export_ids != state.frozen_surface_vertex_id_order:
        raise RuntimeError("Fixed topology violated: exported surface vertex IDs were rebuilt during time stepping")
    current_volume_export_ids = tuple(id(vertex) for vertex in state.volume_export_vertices)
    if current_volume_export_ids != state.frozen_volume_vertex_id_order:
        raise RuntimeError("Fixed topology violated: volumetric export vertex IDs were rebuilt during time stepping")


def _ensure_surface_export_node_ids(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if not state.surface_export_vertices:
        state.surface_export_vertices = _surface_export_vertex_order(state)
        state.surface_export_node_ids = {
            id(vertex): idx for idx, vertex in enumerate(state.surface_export_vertices)
        }
        return

    for vertex in _surface_export_vertex_order(state):
        key = id(vertex)
        if key in state.surface_export_node_ids:
            continue
        state.surface_export_node_ids[key] = len(state.surface_export_vertices)
        state.surface_export_vertices.append(vertex)


def _surface_vertex_idx(state: VolumetricPitoisState, vertex) -> int:
    return int(state.surface_export_node_ids[id(vertex)])


def _ensure_volume_export_node_ids(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if not state.volume_export_vertices:
        state.volume_export_vertices = _volume_export_vertex_order(state)
        state.volume_export_node_ids = {
            id(vertex): idx for idx, vertex in enumerate(state.volume_export_vertices)
        }
        return

    for vertex in _volume_export_vertex_order(state):
        key = id(vertex)
        if key in state.volume_export_node_ids:
            continue
        state.volume_export_node_ids[key] = len(state.volume_export_vertices)
        state.volume_export_vertices.append(vertex)


def _volume_vertex_idx(state: VolumetricPitoisState, vertex) -> int:
    return int(state.volume_export_node_ids[id(vertex)])


def _freeze_volume_msh_topology(state: VolumetricPitoisState) -> None:
    _ensure_volume_export_node_ids(state)

    tets: list[tuple[int, int, int, int]] = []
    for lower_center, upper_center, lower_radial_rings, upper_radial_rings in zip(
        state.layer_centers[:-1],
        state.layer_centers[1:],
        state.layer_rings[:-1],
        state.layer_rings[1:],
    ):
        first_lower = lower_radial_rings[0]
        first_upper = upper_radial_rings[0]
        n_ring = len(first_lower)
        for i in range(n_ring):
            prism = (
                lower_center,
                first_lower[i],
                first_lower[(i + 1) % n_ring],
                upper_center,
                first_upper[i],
                first_upper[(i + 1) % n_ring],
            )
            prism_idx = [_volume_vertex_idx(state, vertex) for vertex in prism]
            for tet in PRISM_TETS:
                tets.append(tuple(prism_idx[j] for j in tet))

        for band in range(len(lower_radial_rings) - 1):
            inner_lower = lower_radial_rings[band]
            outer_lower = lower_radial_rings[band + 1]
            inner_upper = upper_radial_rings[band]
            outer_upper = upper_radial_rings[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                hexahedron = (
                    inner_lower[i],
                    inner_lower[i_next],
                    outer_lower[i_next],
                    outer_lower[i],
                    inner_upper[i],
                    inner_upper[i_next],
                    outer_upper[i_next],
                    outer_upper[i],
                )
                hexahedron_idx = [_volume_vertex_idx(state, vertex) for vertex in hexahedron]
                for tet in HEX_TETS:
                    tets.append(tuple(hexahedron_idx[j] for j in tet))

    if tets:
        points = _volume_points_array(state)
        oriented: list[tuple[int, int, int, int]] = []
        for tet in tets:
            a, b, c, d = tet
            pa = points[int(a)]
            pb = points[int(b)]
            pc = points[int(c)]
            pd = points[int(d)]
            vol = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd))) / 6.0
            oriented.append((b, a, c, d) if vol < 0.0 else tet)
        tets = oriented

    state.volume_export_tets = (
        np.asarray(tets, dtype=int) if tets else np.empty((0, 4), dtype=int)
    )


def _freeze_surface_topology(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
        state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
        state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
        state.topology_frozen = True
        return
    _ensure_surface_export_node_ids(state)
    _freeze_volume_msh_topology(state)

    side_triangles: list[tuple[int, int, int]] = []
    ordered_rings = [list(ring) for ring in state.outer_rings]
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        n_ring = min(len(lower_ring), len(upper_ring))
        if n_ring < 2:
            continue
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            a = _surface_vertex_idx(state, lower_ring[i])
            b = _surface_vertex_idx(state, lower_ring[i_next])
            c = _surface_vertex_idx(state, upper_ring[i_next])
            d = _surface_vertex_idx(state, upper_ring[i])
            pa = np.asarray(lower_ring[i].x_a[:3], dtype=float)
            pb = np.asarray(lower_ring[i_next].x_a[:3], dtype=float)
            pc = np.asarray(upper_ring[i_next].x_a[:3], dtype=float)
            pd = np.asarray(upper_ring[i].x_a[:3], dtype=float)
            if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                side_triangles.append((a, b, c))
                side_triangles.append((a, c, d))
            else:
                side_triangles.append((a, b, d))
                side_triangles.append((b, c, d))

    def freeze_cap(rings: list, center_vertex) -> np.ndarray:
        triangles: list[tuple[int, int, int]] = []
        if not rings:
            return np.empty((0, 3), dtype=int)
        center_idx = _surface_vertex_idx(state, center_vertex)
        n_ring = len(rings[0])
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            triangles.append(
                (
                    center_idx,
                    _surface_vertex_idx(state, rings[0][i]),
                    _surface_vertex_idx(state, rings[0][i_next]),
                )
            )
        for band in range(len(rings) - 1):
            inner = rings[band]
            outer = rings[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                a = _surface_vertex_idx(state, inner[i])
                b = _surface_vertex_idx(state, inner[i_next])
                c = _surface_vertex_idx(state, outer[i_next])
                d = _surface_vertex_idx(state, outer[i])
                pa = np.asarray(inner[i].x_a[:3], dtype=float)
                pb = np.asarray(inner[i_next].x_a[:3], dtype=float)
                pc = np.asarray(outer[i_next].x_a[:3], dtype=float)
                pd = np.asarray(outer[i].x_a[:3], dtype=float)
                if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                    triangles.append((a, b, c))
                    triangles.append((a, c, d))
                else:
                    triangles.append((a, b, d))
                    triangles.append((b, c, d))
        return np.asarray(triangles, dtype=int)

    state.surface_export_side_tris = (
        np.asarray(side_triangles, dtype=int) if side_triangles else np.empty((0, 3), dtype=int)
    )
    state.surface_export_bottom_cap_tris = freeze_cap(state.layer_rings[0], state.cap_bottom_center)
    state.surface_export_top_cap_tris = freeze_cap(state.layer_rings[-1], state.cap_top_center)
    state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
    state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
    state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
    state.topology_frozen = True


def _surface_points_array(state: VolumetricPitoisState) -> np.ndarray:
    _ensure_surface_export_node_ids(state)
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.surface_export_vertices],
        dtype=float,
    )


def _surface_triangles_xyz_from_indices(state: VolumetricPitoisState, tris: np.ndarray) -> np.ndarray:
    tri_idx = np.asarray(tris, dtype=int)
    if tri_idx.size == 0:
        return np.empty((0, 3, 3), dtype=float)
    points = _surface_points_array(state)
    return np.asarray([points[np.asarray(tri, dtype=int)] for tri in tri_idx], dtype=float)


def _canonicalize_ring_orders(state: VolumetricPitoisState) -> None:
    if not state.layer_rings:
        return

    bottom_ring_center, _top_ring_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    reference = np.asarray(state.outer_rings[0][0].x_a[:3], dtype=float) - bottom_ring_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    ordered_layers: list[list[list[object]]] = []
    ordered_outer: list[list[object]] = []
    for rings in state.layer_rings:
        if not rings:
            ordered_layers.append([])
            ordered_outer.append([])
            continue
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0)
        ordered_rings: list[list[object]] = []
        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - outer_center[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            angles = np.arctan2(rel @ e2, rel @ e1)
            order = np.argsort(angles)
            ordered_rings.append([ring[i] for i in order])
        ordered_layers.append(ordered_rings)
        ordered_outer.append(ordered_rings[-1])

    state.layer_rings = ordered_layers
    state.outer_rings = ordered_outer
    state.surface_export_vertices = []
    state.surface_export_node_ids = {}
    state.surface_export_side_tris = np.empty((0, 3), dtype=int)
    state.surface_export_bottom_cap_tris = np.empty((0, 3), dtype=int)
    state.surface_export_top_cap_tris = np.empty((0, 3), dtype=int)
    state.volume_export_vertices = []
    state.volume_export_node_ids = {}
    state.volume_export_tets = np.empty((0, 4), dtype=int)
    _refresh_cap_boundary_sets(state)


def _radial_ring_factors_from_count(n_rings: int) -> tuple[float, ...]:
    n_rings = int(n_rings)
    if n_rings < 1:
        raise ValueError("contact_line_radial_rings must be at least 1")
    return tuple(float(i + 1) / float(n_rings) for i in range(n_rings))


def _radial_ring_factors_with_outer_refinement(
    n_rings: int,
    extra_outer_rings: int,
    outer_bias_ratio: float = 1.0,
) -> tuple[float, ...]:
    n_rings = int(n_rings)
    base = list(_radial_ring_factors_from_count(n_rings))
    extra_outer_rings = max(0, int(extra_outer_rings))
    outer_bias_ratio = max(1.0, float(outer_bias_ratio))
    if extra_outer_rings == 0 and abs(outer_bias_ratio - 1.0) <= 1.0e-12:
        return tuple(base)

    # If no extra rings are requested, bias can only move existing rings; it
    # cannot add visible node lines.  Keep this case available but do not use it
    # for CL refinement production meshes.
    if extra_outer_rings == 0 and abs(outer_bias_ratio - 1.0) > 1.0e-12:
        total_rings = max(1, n_rings + extra_outer_rings)
        gap_weights = np.array(
            [outer_bias_ratio ** power for power in range(total_rings - 1, -1, -1)],
            dtype=float,
        )
        factors = np.cumsum(gap_weights / float(np.sum(gap_weights)))
        factors[-1] = 1.0
        return tuple(float(round(factor, 12)) for factor in factors)

    outer_start = float(base[-2]) if len(base) >= 2 else 0.0
    total_outer_segments = max(1, extra_outer_rings + 1)
    outer_width = max(0.0, 1.0 - outer_start)
    if outer_width <= 1.0e-15:
        return tuple(base)

    if abs(outer_bias_ratio - 1.0) <= 1.0e-12:
        outer_gaps = [outer_width / float(total_outer_segments)] * total_outer_segments
    else:
        series = np.array(
            [outer_bias_ratio ** power for power in range(total_outer_segments - 1, -1, -1)],
            dtype=float,
        )
        outer_gaps = list(outer_width * series / float(np.sum(series)))

    factors = list(base[:-1])
    radius = outer_start
    for gap in outer_gaps:
        radius += float(gap)
        factors.append(min(radius, 1.0))

    factors[-1] = 1.0
    factors = sorted({round(float(f), 12) for f in factors})
    return tuple(float(f) for f in factors)
    pressure_scalar: float = 0.0


def _sidewall_layer_fractions_with_contact_refinement(
    n_layers: int,
    contact_bias_ratio: float,
) -> tuple[float, ...]:
    """Symmetric sidewall fractions with refinement at both CLs and the neck."""
    n_layers = int(n_layers)
    if n_layers <= 2:
        return tuple(np.linspace(0.0, 1.0, max(n_layers, 1), dtype=float))

    ratio = max(1.0, float(contact_bias_ratio))
    if abs(ratio - 1.0) <= 1.0e-12:
        return tuple(float(v) for v in np.linspace(0.0, 1.0, n_layers, dtype=float))

    def half_gap_weights(n_gaps: int) -> np.ndarray:
        # Small gaps at the contact line and the neck; largest gap halfway
        # between them.  Neighboring gaps still follow the user-requested ratio.
        center = 0.5 * float(max(n_gaps - 1, 0))
        exponents = [int(round(center - abs(float(i) - center))) for i in range(n_gaps)]
        return np.array([ratio ** power for power in exponents], dtype=float)

    if n_layers % 2 == 1:
        mid = n_layers // 2
        weights = half_gap_weights(mid)
        half_gaps = 0.5 * weights / float(np.sum(weights))
        fractions = np.empty(n_layers, dtype=float)
        fractions[0] = 0.0
        fractions[mid] = 0.5
        lower = np.cumsum(half_gaps)
        fractions[1 : mid + 1] = lower
        fractions[mid + 1 : -1] = 1.0 - lower[-2::-1]
    else:
        half = n_layers // 2
        weights = half_gap_weights(half)
        half_gaps = 0.5 * weights / float(np.sum(weights))
        lower = np.cumsum(half_gaps)
        fractions = np.concatenate([[0.0], lower, 1.0 - lower[-2::-1]])

    fractions[0] = 0.0
    fractions[-1] = 1.0
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in fractions)


def _redistribute_sidewall_layers_axially(
    HC,
    outer_rings: list,
    layer_rings: list,
    layer_fractions: tuple[float, ...],
) -> None:
    if len(outer_rings) != len(layer_rings) or len(outer_rings) != len(layer_fractions):
        return
    if len(outer_rings) <= 2:
        return

    bottom_center, top_center, axis = _contact_plane_centers_and_axis(SimpleNamespace(outer_rings=outer_rings))
    total_span = float(np.dot(top_center - bottom_center, axis))
    if not np.isfinite(total_span) or total_span <= 1.0e-30:
        return

    groups: list[list[object]] = []
    centers: list[np.ndarray] = []
    for rings in layer_rings:
        z_ref = float(np.mean([float(v.x_a[2]) for v in rings[-1]]))
        same_layer = [v for v in HC.V if abs(float(v.x_a[2]) - z_ref) < 1.0e-12]
        seen: set[int] = set()
        group: list[object] = []
        for vertex in same_layer:
            key = id(vertex)
            if key in seen:
                continue
            seen.add(key)
            group.append(vertex)
        groups.append(group)
        centers.append(np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0))

    for group, center, fraction in zip(groups, centers, layer_fractions):
        target_center = bottom_center + float(fraction) * total_span * axis
        delta = target_center - center
        if float(np.linalg.norm(delta)) <= 1.0e-18:
            continue
        for vertex in group:
            target = np.asarray(vertex.x_a[:3], dtype=float) + delta
            _move(vertex, tuple(target), HC, set())


def separation_config() -> VolumetricPitoisConfig:
    return VolumetricPitoisConfig(
        name="v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case12",
        title="Case 12 axisym: #9 solver with #3 force-calibrated initial mesh",
    )


def _mesh_volume_m3(HC) -> float:
    if hasattr(HC, "gmsh_volume_vertices") and hasattr(HC, "gmsh_volume_tets"):
        vertices = list(getattr(HC, "gmsh_volume_vertices"))
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        return float(_indexed_tet_mesh_volume_m3(points, np.asarray(getattr(HC, "gmsh_volume_tets"), dtype=int)))
    return float(sum(float(dual_volume(v, HC, dim=3)) for v in HC.V))


def _cap_z_span(vertices: list) -> tuple[float, float]:
    z_vals = [float(v.x_a[2]) for v in vertices]
    return min(z_vals), max(z_vals)


def _cap_radius(vertices: list) -> float:
    return max((float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))) for v in vertices), default=1.0)


def _scale_mesh(HC, radial_scale: float, axial_scale: float) -> None:
    for v in list(HC.V):
        coord = np.asarray(v.x_a[:3], dtype=float)
        HC.V.move(
            v,
            (
                radial_scale * coord[0],
                radial_scale * coord[1],
                axial_scale * coord[2],
            ),
        )


def _move_vertices_batch(vertices, targets, HC, bV) -> None:
    """Move a vertex group without transient cache collisions.

    Some ring updates are permutations: a vertex can be moved onto another
    vertex's current coordinates before that second vertex is moved. Doing
    those updates one-by-one causes the vertex cache to evict the later vertex from
    the cache. A temporary staging move avoids that failure mode.
    """
    vertices = list(vertices)
    targets = [tuple(map(float, target)) for target in targets]
    if not vertices:
        return
    if len(vertices) != len(targets):
        raise ValueError("vertices and targets must have the same length")
    if isinstance(getattr(HC, "V", None), _DisplayVertexStore) and len(vertices) >= 128:
        all_vertices = list(HC.V.values())
        for v, target in zip(vertices, targets):
            v.x_a = np.asarray(target, dtype=float)
        HC.V.clear()
        for v in all_vertices:
            HC.V[tuple(float(x) for x in np.asarray(v.x_a[:3], dtype=float))] = v
        return
    if len(vertices) == 1:
        _move(vertices[0], targets[0], HC, bV)
        return

    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in HC.V], dtype=float)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    base = 10.0 * (scale + 1.0)
    delta = 1.0e-3 * (scale + 1.0)
    staged = [
        (base + delta * float(i + 1), base + 2.0 * delta * float(i + 1), base + 3.0 * delta * float(i + 1))
        for i in range(len(vertices))
    ]

    for v, tmp in zip(vertices, staged):
        _move(v, tmp, HC, bV)
    for v, target in zip(vertices, targets):
        _move(v, target, HC, bV)


def _update_duals_and_masses(state: VolumetricPitoisState) -> None:
    if hasattr(state, "_dual_area_vector_cache"):
        delattr(state, "_dual_area_vector_cache")
    _recompute_duals(state.HC)
    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if vertices and tets.size:
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        valid, volumes, _grads = _gmsh_tet_metric_arrays(points, tets)
        phases = np.zeros(int(tets.shape[0]), dtype=int)
        _tet_masses, lumped = multiphase_tet_volume_lumped_masses(
            len(vertices),
            tets,
            np.where(valid, volumes, 0.0),
            phases,
            {0: 1.0},
        )
        solver_lumped = np.asarray(_solver_mass_from_lumped_volume(state.config, lumped), dtype=float)
        for vertex, mass in zip(vertices, solver_lumped):
            vertex.m = max(float(mass), 1.0e-12)
        return
    for v in state.HC.V:
        v.m = max(float(_solver_mass_from_lumped_volume(state.config, dual_volume(v, state.HC, dim=3))), 1.0e-12)


def _resolve_pressure_local(v, pressure_model=None, HC=None, dim: int = 3) -> float:
    if pressure_model is not None:
        return float(pressure_model(v, HC=HC, dim=dim))
    p_val = getattr(v, "p", 0.0)
    if np.ndim(p_val) == 0:
        return float(p_val)
    return float(np.asarray(p_val).ravel()[0])


def _ensure_gmsh_vd_markers(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state, "gmsh_compute_mesh", False)):
        return
    for vertex in list(getattr(state, "volume_export_vertices", [])):
        vertex.vd = getattr(vertex, "vd", set())


def _gmsh_dual_area_vector_cache(state: VolumetricPitoisState, *, dim: int) -> dict[tuple[int, int, int], np.ndarray]:
    cache = getattr(state, "_gmsh_dual_area_vector_cache", None)
    if cache is not None:
        return cache
    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if dim != 3 or not vertices or tets.size == 0:
        state._gmsh_dual_area_vector_cache = {}
        state._gmsh_dual_area_vector_edges = (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, 3), dtype=float),
        )
        return state._gmsh_dual_area_vector_cache

    points = np.asarray([np.asarray(vertex.x_a[:3], dtype=float) for vertex in vertices], dtype=float)
    tet_idx = tets.reshape((-1, 4))
    in_range = (np.min(tet_idx, axis=1) >= 0) & (np.max(tet_idx, axis=1) < len(vertices))
    tet_idx = tet_idx[in_range]
    if tet_idx.size:
        tet_idx = tet_idx[np.all(np.isfinite(points[tet_idx]), axis=(1, 2))]
    if tet_idx.size == 0:
        state._gmsh_dual_area_vector_cache = {}
        state._gmsh_dual_area_vector_edges = (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, 3), dtype=float),
        )
        return state._gmsh_dual_area_vector_cache

    tet_points = points[tet_idx]
    tet_centroid = np.mean(tet_points, axis=1)
    edge_defs = (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    )
    src_blocks: list[np.ndarray] = []
    dst_blocks: list[np.ndarray] = []
    area_blocks: list[np.ndarray] = []
    for local_i, local_j, local_k0, local_k1 in edge_defs:
        i = tet_idx[:, local_i]
        j = tet_idx[:, local_j]
        pi = tet_points[:, local_i, :]
        pj = tet_points[:, local_j, :]
        edge_mid = 0.5 * (pi + pj)
        area_ij = np.zeros_like(pi)
        for local_k in (local_k0, local_k1):
            pk = tet_points[:, local_k, :]
            face_centroid = (pi + pj + pk) / 3.0
            tri_area_vec = 0.5 * np.cross(face_centroid - edge_mid, tet_centroid - edge_mid)
            tri_centroid = (edge_mid + face_centroid + tet_centroid) / 3.0
            flip = np.einsum("ij,ij->i", tri_area_vec, pi - tri_centroid) > 0.0
            tri_area_vec[flip] *= -1.0
            area_ij += tri_area_vec
        finite = np.all(np.isfinite(area_ij), axis=1)
        if not np.any(finite):
            continue
        src_blocks.extend((i[finite], j[finite]))
        dst_blocks.extend((j[finite], i[finite]))
        area_blocks.extend((area_ij[finite], -area_ij[finite]))

    if not src_blocks:
        state._gmsh_dual_area_vector_cache = {}
        state._gmsh_dual_area_vector_edges = (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, 3), dtype=float),
        )
        return state._gmsh_dual_area_vector_cache

    src_all = np.concatenate(src_blocks).astype(int, copy=False)
    dst_all = np.concatenate(dst_blocks).astype(int, copy=False)
    area_all = np.concatenate(area_blocks, axis=0).astype(float, copy=False)
    pair_keys = src_all.astype(np.int64) * np.int64(len(vertices)) + dst_all.astype(np.int64)
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
    area_sum = np.zeros((unique_keys.size, 3), dtype=float)
    np.add.at(area_sum, inverse, area_all)
    src_unique = (unique_keys // np.int64(len(vertices))).astype(int, copy=False)
    dst_unique = (unique_keys % np.int64(len(vertices))).astype(int, copy=False)

    cache = {
        (id(vertices[int(i)]), id(vertices[int(j)]), int(dim)): np.asarray(area_vec, dtype=float)
        for i, j, area_vec in zip(src_unique, dst_unique, area_sum)
        if np.all(np.isfinite(area_vec))
    }
    state._gmsh_dual_area_vector_edges = (src_unique, dst_unique, area_sum)
    state._gmsh_dual_area_vector_cache = cache
    return cache


def _cached_dual_area_vector(v, v_j, *, state: VolumetricPitoisState, dim: int) -> np.ndarray:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        cache = _gmsh_dual_area_vector_cache(state, dim=dim)
        return np.asarray(cache.get((id(v), id(v_j), int(dim)), np.zeros(dim, dtype=float)), dtype=float)
    cache = getattr(state, "_dual_area_vector_cache", None)
    if cache is None:
        cache = {}
        state._dual_area_vector_cache = cache
    key = (id(v), id(v_j), int(dim))
    cached = cache.get(key)
    if cached is not None:
        return cached
    A_ij = dual_area_vector(v, v_j, state.HC, dim)
    cache[key] = A_ij
    return A_ij


def _gmsh_surface_heron_force_area_maps(
    state: VolumetricPitoisState,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    cached_force = getattr(state, "_gmsh_surface_heron_force_cache", None)
    cached_area = getattr(state, "_gmsh_surface_heron_area_cache", None)
    if cached_force is not None and cached_area is not None:
        return cached_force, cached_area

    side_tris = np.asarray(getattr(state, "surface_export_side_tris", np.empty((0, 3), dtype=int)), dtype=int)
    surface_vertices = list(getattr(state, "surface_export_vertices", []))
    if not surface_vertices:
        state._gmsh_surface_heron_force_cache = {}
        state._gmsh_surface_heron_area_cache = {}
        return state._gmsh_surface_heron_force_cache, state._gmsh_surface_heron_area_cache

    contact_ids = {id(v) for v in state.bottom_contact_ring + state.top_contact_ring}
    if side_tris.size == 0:
        ordered_rings = _ordered_outer_rings_for_surface(state)
        if len(ordered_rings) < 2:
            state._gmsh_surface_heron_force_cache = {}
            state._gmsh_surface_heron_area_cache = {}
            return state._gmsh_surface_heron_force_cache, state._gmsh_surface_heron_area_cache

        _surface, _surface_bV, flat_vertices, _triangles = _build_structured_side_surface_complex(ordered_rings)
        source_vertices = [source_v for ring in ordered_rings for source_v in ring]
        force_map: dict[int, np.ndarray] = {}
        area_map: dict[int, float] = {}
        for source_v, surface_v in zip(source_vertices, flat_vertices):
            surface_v.boundary = id(source_v) in contact_ids
            surface_v.u = np.zeros(3, dtype=float)
            surface_v.p = 0.0
            area = max(float(dual_area_heron(surface_v)), 1.0e-12)
            surface_v.m = area
            surface_v.phase = 0
            surface_v.is_interface = not surface_v.boundary
            surface_v.interface_phases = frozenset({0, 1}) if surface_v.is_interface else frozenset()
            force_map[id(source_v)] = surface_tension_force(surface_v, gamma=state.config.gamma, dim=3)
            area_map[id(source_v)] = float(area)
        state._gmsh_surface_heron_force_cache = force_map
        state._gmsh_surface_heron_area_cache = area_map
        return force_map, area_map

    used_indices = sorted({int(idx) for tri in side_tris for idx in tri})
    surface_clone = Complex(3, domain=None)
    clone_by_idx = {
        idx: surface_clone.V[tuple(float(x) for x in np.asarray(surface_vertices[idx].x_a[:3], dtype=float))]
        for idx in used_indices
    }
    for tri in side_tris:
        a, b, c = [int(idx) for idx in tri]
        clone_by_idx[a].connect(clone_by_idx[b])
        clone_by_idx[b].connect(clone_by_idx[c])
        clone_by_idx[c].connect(clone_by_idx[a])

    for idx, surface_v in clone_by_idx.items():
        source_v = surface_vertices[idx]
        surface_v.boundary = id(source_v) in contact_ids
        surface_v.u = np.zeros(3, dtype=float)
        surface_v.p = 0.0
        area = max(float(dual_area_heron(surface_v)), 1.0e-12)
        surface_v.m = area
        surface_v.phase = 0
        surface_v.is_interface = not surface_v.boundary
        surface_v.interface_phases = frozenset({0, 1}) if surface_v.is_interface else frozenset()

    force_map = {
        id(surface_vertices[idx]): surface_tension_force(surface_v, gamma=state.config.gamma, dim=3)
        for idx, surface_v in clone_by_idx.items()
    }
    area_map = {id(surface_vertices[idx]): float(surface_v.m) for idx, surface_v in clone_by_idx.items()}
    state._gmsh_surface_heron_force_cache = force_map
    state._gmsh_surface_heron_area_cache = area_map
    return force_map, area_map


def _gmsh_surface_heron_force_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    force_map, _area_map = _gmsh_surface_heron_force_area_maps(state)
    return force_map


def _gmsh_heron_pressure_reference(
    state: VolumetricPitoisState,
    vertices: list[object],
) -> float:
    """Least-squares Heron pressure reference used by the PR33 EOS closure.

    Equation: choose p_ref so Heron surface forces are best represented by
    p_ref A_i in a least-squares sense.  This is a pressure-gauge choice for
    the PR33 pressure force, not a separate physical force law.
    """
    side_tris = np.asarray(getattr(state, "surface_export_side_tris", np.empty((0, 3), dtype=int)), dtype=int)
    surface_vertices = list(getattr(state, "surface_export_vertices", []))
    if side_tris.size > 0 and surface_vertices:
        side_tris = side_tris.reshape((-1, 3))
        if int(np.max(side_tris)) < len(surface_vertices) and int(np.min(side_tris)) >= 0:
            points = np.asarray([np.asarray(vertex.x_a[:3], dtype=float) for vertex in surface_vertices], dtype=float)
            try:
                forces, area_vectors, pressure, _volume = heron_forces_for_points(
                    points,
                    side_tris,
                    float(state.config.gamma),
                )
            except Exception:
                forces = np.empty((0, 3), dtype=float)
                area_vectors = np.empty((0, 3), dtype=float)
                pressure = 0.0
            finite = (
                np.all(np.isfinite(forces), axis=1)
                & np.all(np.isfinite(area_vectors), axis=1)
                & (np.linalg.norm(area_vectors, axis=1) > 1.0e-30)
            ) if forces.size and area_vectors.size else np.zeros(0, dtype=bool)
            if np.any(finite) and np.isfinite(pressure):
                return float(
                    pressure_equivalent_from_forces(
                        np.asarray(forces[finite], dtype=float),
                        np.asarray(area_vectors[finite], dtype=float),
                    )
                )
    heron_force_map, heron_area_map = _gmsh_surface_heron_force_area_maps(state)
    _side_vertices, side_area_map, side_normal_map = _sidewall_dual_areas_and_normals(state)
    forces: list[np.ndarray] = []
    area_vectors: list[np.ndarray] = []
    for vertex in vertices:
        vid = id(vertex)
        normal = np.asarray(side_normal_map.get(vid, np.zeros(3, dtype=float)), dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30 or not np.isfinite(normal_norm):
            continue
        area = float(heron_area_map.get(vid, 0.0))
        if area <= 1.0e-30 or not np.isfinite(area):
            area = float(side_area_map.get(vid, 0.0))
        if area <= 1.0e-30 or not np.isfinite(area):
            continue
        heron_force = np.asarray(heron_force_map.get(vid, np.zeros(3, dtype=float)), dtype=float)
        if not np.all(np.isfinite(heron_force)):
            continue
        forces.append(heron_force)
        area_vectors.append(area * normal / normal_norm)
    if not forces:
        return 0.0
    return float(
        pressure_equivalent_from_forces(
            np.asarray(forces, dtype=float),
            np.asarray(area_vectors, dtype=float),
        )
    )


def _solver_hydrostatic_z_ref(state: VolumetricPitoisState) -> float:
    mode = str(getattr(state.config, "solver_hydrostatic_zref_mode", "midpoint")).strip().lower()
    if mode == "top_cl" and getattr(state, "top_contact_ring", None):
        return float(np.mean([float(np.asarray(v.x_a[:3], dtype=float)[2]) for v in state.top_contact_ring]))
    return 0.5 * float(state.bottom_sphere_center[2] + state.top_sphere_center[2])


def _gmsh_solver_hydrostatic_force_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    """Map hydrostatic pressure p_h = rho g (z_ref - z) to surface force.

    The force is integrated over liquid-air surface triangle area vectors and
    distributed to the triangle vertices.  Cases 7-12 can enable this term;
    cases 1-6 disable it in _ddgclib_case_runner.py.
    """
    if (
        (not bool(getattr(state.config, "enable_solver_hydrostatic_force", False)))
        or (not bool(getattr(state.config, "include_gravity", False)))
        or (not bool(getattr(state, "gmsh_compute_mesh", False)))
    ):
        return {}
    cache = getattr(state, "_gmsh_solver_hydrostatic_force_cache", {})
    z_ref = _solver_hydrostatic_z_ref(state)
    key = (
        round(float(z_ref), 18),
        round(float(getattr(state.config, "rho_f", 0.0)), 18),
        round(float(getattr(state.config, "gravity_mps2", 0.0)), 18),
        str(getattr(state.config, "solver_hydrostatic_zref_mode", "midpoint")),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    side_tris = np.asarray(getattr(state, "surface_export_side_tris", np.empty((0, 3), dtype=int)), dtype=int)
    surface_vertices = list(getattr(state, "surface_export_vertices", []))
    if side_tris.size == 0:
        ordered_rings = _ordered_outer_rings_for_surface(state)
        surface_vertices = [v for ring in ordered_rings for v in ring]
        triangles = []
        offsets = np.cumsum([0] + [len(ring) for ring in ordered_rings])
        for ring_idx in range(max(0, len(ordered_rings) - 1)):
            lower = ordered_rings[ring_idx]
            upper = ordered_rings[ring_idx + 1]
            n = min(len(lower), len(upper))
            if n < 3:
                continue
            lo0 = int(offsets[ring_idx])
            up0 = int(offsets[ring_idx + 1])
            for i in range(n):
                j = (i + 1) % n
                triangles.append((lo0 + i, up0 + i, up0 + j))
                triangles.append((lo0 + i, up0 + j, lo0 + j))
        side_tris = np.asarray(triangles, dtype=int)
    if side_tris.size == 0 or not surface_vertices:
        force_map: dict[int, np.ndarray] = {}
        state.last_solver_hydrostatic_force_l1 = 0.0
        state.last_solver_hydrostatic_z_ref = float(z_ref)
        cache[key] = force_map
        state._gmsh_solver_hydrostatic_force_cache = cache
        return force_map

    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in surface_vertices], dtype=float)
    forces = np.zeros_like(points)
    _force_map, heron_area_map = _gmsh_surface_heron_force_area_maps(state)
    mid, axis, _e1, _e2 = _gmsh_axisym_basis(state)
    rho_g = float(state.config.rho_f) * float(state.config.gravity_mps2)
    for tri in side_tris.reshape((-1, 3)):
        a, b, c = [int(i) for i in tri]
        if a < 0 or b < 0 or c < 0 or max(a, b, c) >= len(surface_vertices):
            continue
        pa, pb, pc = points[[a, b, c]]
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        area_norm = float(np.linalg.norm(area_vec))
        if (not np.isfinite(area_norm)) or area_norm <= 1.0e-30:
            continue
        centroid = (pa + pb + pc) / 3.0
        rel = centroid - mid
        axial = float(np.dot(rel, axis))
        radial = rel - axial * axis
        if float(np.dot(area_vec, radial)) < 0.0:
            area_vec = -area_vec
        p_h = rho_g * (float(z_ref) - float(centroid[2]))
        contribution = (p_h / 3.0) * area_vec
        if np.all(np.isfinite(contribution)):
            forces[a] += contribution
            forces[b] += contribution
            forces[c] += contribution

    force_map = {}
    for idx, vertex in enumerate(surface_vertices):
        force = np.asarray(forces[idx], dtype=float)
        heron_area = float(heron_area_map.get(id(vertex), 0.0))
        force_map[id(vertex)] = force
        if idx == 0:
            state.last_solver_hydrostatic_z_ref = float(z_ref)
        if heron_area > 0.0:
            # The map is keyed by the same liquid-air surface vertices as the
            # Heron surface force. Keeping the area read here makes the raw
            # solver metadata traceable to ddgclib's Heron dual-area path.
            state.last_solver_hydrostatic_has_heron_area = True
    state.last_solver_hydrostatic_force_l1 = float(np.sum(np.linalg.norm(forces, axis=1)))
    state.last_solver_hydrostatic_z_ref = float(z_ref)
    cache[key] = force_map
    state._gmsh_solver_hydrostatic_force_cache = cache
    return force_map


def _solver_hydrostatic_force(v, *, state: VolumetricPitoisState) -> np.ndarray:
    if not bool(getattr(state.config, "enable_solver_hydrostatic_force", False)):
        return np.zeros(3, dtype=float)
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        return np.asarray(_gmsh_solver_hydrostatic_force_map(state).get(id(v), np.zeros(3)), dtype=float)
    return np.zeros(3, dtype=float)


def _surface_tension_force_heron(v, *, dim: int, state: VolumetricPitoisState) -> np.ndarray:
    if not bool(getattr(state.config, "enable_heron_surface_tension", False)):
        return np.zeros(dim)
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        force = _gmsh_surface_heron_force_map(state).get(id(v))
        if force is None:
            return np.zeros(dim)
        return np.asarray(force[:dim], dtype=float)
    if getattr(v, "is_interface", False) and state.mps is not None:
        phases = getattr(v, "interface_phases", frozenset())
        if len(phases) >= 2:
            phase_list = sorted(phases)
            gamma = state.mps.get_gamma_pair(phase_list[0], phase_list[1])
            if gamma != 0.0:
                return surface_tension_force(v, gamma=gamma, dim=dim)
    return np.zeros(dim)


def _gmsh_stress_force_vector_cache(
    state: VolumetricPitoisState,
    *,
    pressure_model=None,
    dim: int,
) -> dict[int, np.ndarray]:
    cache_by_key = getattr(state, "_gmsh_stress_force_vector_cache", {})
    key = (
        id(pressure_model),
        int(dim),
        round(float(getattr(state, "pressure_scalar", 0.0)), 18),
        bool(getattr(state.config, "include_gravity", False)),
        round(float(np.asarray(getattr(state, "bottom_sphere_center", np.zeros(3)), dtype=float)[2]), 18),
        round(float(np.asarray(getattr(state, "top_sphere_center", np.zeros(3)), dtype=float)[2]), 18),
    )
    cached = cache_by_key.get(key)
    if cached is not None:
        return cached

    vertices = list(getattr(state, "volume_export_vertices", []))
    if (
        int(dim) != 3
        or state.HC is None
        or not vertices
        or not bool(getattr(state, "gmsh_compute_mesh", False))
    ):
        return {}

    _gmsh_dual_area_vector_cache(state, dim=dim)
    src, dst, area = getattr(
        state,
        "_gmsh_dual_area_vector_edges",
        (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, dim), dtype=float),
        ),
    )
    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    area = np.asarray(area, dtype=float)
    forces = np.zeros((len(vertices), dim), dtype=float)
    if src.size == 0 or dst.size == 0 or area.size == 0:
        force_map = {id(vertex): forces[idx] for idx, vertex in enumerate(vertices)}
        cache_by_key[key] = force_map
        state._gmsh_stress_force_vector_cache = cache_by_key
        return force_map

    points = np.asarray([np.asarray(vertex.x_a[:dim], dtype=float) for vertex in vertices], dtype=float)
    velocities = np.asarray([np.asarray(vertex.u[:dim], dtype=float) for vertex in vertices], dtype=float)
    pressures = np.asarray(
        [_resolve_pressure_local(vertex, pressure_model, state.HC, dim) for vertex in vertices],
        dtype=float,
    )
    if state.mps is not None:
        mu_values = np.asarray(
            [float(state.mps.get_mu(getattr(vertex, "phase", 0))) for vertex in vertices],
            dtype=float,
        )
    else:
        mu_values = np.zeros(len(vertices), dtype=float)

    valid = (
        (src >= 0)
        & (src < len(vertices))
        & (dst >= 0)
        & (dst < len(vertices))
        & np.all(np.isfinite(area[:, :dim]), axis=1)
    )
    if np.any(valid):
        src_v = src[valid]
        dst_v = dst[valid]
        A_ij = area[valid, :dim]
        pair_force = -0.5 * (pressures[src_v] + pressures[dst_v])[:, None] * A_ij

        d_ij = points[dst_v] - points[src_v]
        d_norm = np.linalg.norm(d_ij, axis=1)
        viscous_valid = np.isfinite(d_norm) & (d_norm >= 1.0e-30)
        if np.any(viscous_valid):
            d_hat = np.zeros_like(d_ij)
            d_hat[viscous_valid] = d_ij[viscous_valid] / d_norm[viscous_valid, None]
            coeff = np.zeros(src_v.size, dtype=float)
            coeff[viscous_valid] = (
                mu_values[src_v[viscous_valid]]
                / d_norm[viscous_valid]
                * np.einsum("ij,ij->i", d_hat[viscous_valid], A_ij[viscous_valid])
            )
            pair_force += coeff[:, None] * (velocities[dst_v] - velocities[src_v])

        finite_force = np.all(np.isfinite(pair_force), axis=1)
        if np.any(finite_force):
            np.add.at(forces, src_v[finite_force], pair_force[finite_force])

    force_map = {id(vertex): forces[idx] for idx, vertex in enumerate(vertices)}
    cache_by_key[key] = force_map
    state._gmsh_stress_force_vector_cache = cache_by_key
    return force_map


def _gmsh_tet_cauchy_viscous_force_vector_cache(
    state: VolumetricPitoisState,
    *,
    dim: int,
    pressure_model=None,
    include_pressure: bool = False,
) -> dict[int, np.ndarray]:
    """Assemble the tet-Cauchy viscous force used by the tet_cauchy option.

    Equation: sigma = -p I + mu (grad u + grad u^T).
    Reference: standard Newtonian Cauchy stress, e.g. Batchelor (1967).
    The optional zeta div(u) I term is implemented here for experiments, but
    the canonical cases 1-12 do not pass --solver-bulk-viscosity-pa.
    """
    cache_by_key = getattr(state, "_gmsh_tet_cauchy_viscous_force_cache", {})
    key = (
        int(dim),
        bool(include_pressure),
        id(pressure_model) if pressure_model is not None else 0,
        round(float(getattr(state, "pressure_scalar", 0.0)), 18),
        round(float(getattr(state, "pressure_projection_scalar", 0.0)), 18),
        bool(getattr(state.config, "solver_pr33_pressure_include_hydrostatic", False)),
        bool(getattr(state.config, "include_gravity", False)),
        round(float(getattr(state.config, "solver_bulk_viscosity_pa_s", 0.0)), 18),
        round(float(np.asarray(getattr(state, "bottom_sphere_center", np.zeros(3)), dtype=float)[2]), 18),
        round(float(np.asarray(getattr(state, "top_sphere_center", np.zeros(3)), dtype=float)[2]), 18),
    )
    cached = cache_by_key.get(key)
    if cached is not None:
        return cached

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int).reshape((-1, 4))
    if (
        int(dim) != 3
        or state.HC is None
        or not vertices
        or tets.size == 0
        or not bool(getattr(state, "gmsh_compute_mesh", False))
    ):
        force_map = {id(vertex): np.zeros(dim, dtype=float) for vertex in vertices}
        cache_by_key[key] = force_map
        state._gmsh_tet_cauchy_viscous_force_cache = cache_by_key
        return force_map

    points = np.asarray([np.asarray(vertex.x_a[:dim], dtype=float) for vertex in vertices], dtype=float)
    velocities = np.asarray([np.asarray(vertex.u[:dim], dtype=float) for vertex in vertices], dtype=float)
    valid, volumes, grads = _gmsh_tet_metric_arrays(points, tets)
    forces = np.zeros((len(vertices), dim), dtype=float)
    if not np.any(valid):
        force_map = {id(vertex): forces[idx] for idx, vertex in enumerate(vertices)}
        cache_by_key[key] = force_map
        state._gmsh_tet_cauchy_viscous_force_cache = cache_by_key
        return force_map

    tet_nodes = tets[valid]
    tet_volumes = np.asarray(volumes[valid], dtype=float)
    tet_grads = np.asarray(grads[valid], dtype=float)
    tet_velocities = velocities[tet_nodes]
    if state.mps is not None:
        mu_values = np.asarray(
            [float(state.mps.get_mu(getattr(vertex, "phase", 0))) for vertex in vertices],
            dtype=float,
        )
    else:
        mu_values = np.zeros(len(vertices), dtype=float)
    mu_tet = np.mean(mu_values[tet_nodes], axis=1)

    # Linear tetrahedral velocity gradient: grad u = sum_a u_a grad N_a.
    grad_u = np.einsum("tla,tlb->tab", tet_velocities, tet_grads)
    if bool(include_pressure) and pressure_model is not None:
        p_nodes = np.asarray(
            [_resolve_pressure_local(vertex, pressure_model, state.HC, dim) for vertex in vertices],
            dtype=float,
        )
        p_tet = np.mean(p_nodes[tet_nodes], axis=1)
    else:
        p_tet = np.zeros(tet_nodes.shape[0], dtype=float)

    sigma = np.zeros((tet_nodes.shape[0], dim, dim), dtype=float)
    bulk_viscosity = max(float(getattr(state.config, "solver_bulk_viscosity_pa_s", 0.0)), 0.0)
    identity = np.eye(dim, dtype=float)
    for tet_i in range(tet_nodes.shape[0]):
        sigma[tet_i] = cauchy_stress(float(p_tet[tet_i]), grad_u[tet_i], float(mu_tet[tet_i]), dim=dim)
        if bulk_viscosity > 0.0:
            sigma[tet_i] += bulk_viscosity * float(np.trace(grad_u[tet_i])) * identity

    # Weak-form nodal force from stress divergence:
    # F_i = -int_T sigma grad N_i dV, evaluated per linear tetrahedron.
    local_forces = -tet_volumes[:, None, None] * np.einsum("tab,tlb->tla", sigma, tet_grads)
    pressure_sigma = -p_tet[:, None, None] * np.eye(dim, dtype=float)[None, :, :]
    pressure_local_forces = -tet_volumes[:, None, None] * np.einsum("tab,tlb->tla", pressure_sigma, tet_grads)
    finite = np.all(np.isfinite(local_forces), axis=(1, 2))
    if np.any(finite):
        for local_idx in range(4):
            np.add.at(forces, tet_nodes[finite, local_idx], local_forces[finite, local_idx])

    if bool(include_pressure):
        pressure_forces = np.zeros((len(vertices), dim), dtype=float)
        finite_pressure = np.all(np.isfinite(pressure_local_forces), axis=(1, 2))
        if np.any(finite_pressure):
            for local_idx in range(4):
                np.add.at(
                    pressure_forces,
                    tet_nodes[finite_pressure, local_idx],
                    pressure_local_forces[finite_pressure, local_idx],
                )
        state.last_solver_cauchy_pressure_force_l1 = float(np.sum(np.linalg.norm(pressure_forces, axis=1)))
    state.last_solver_cauchy_force_l1 = float(np.sum(np.linalg.norm(forces, axis=1)))

    force_map = {id(vertex): forces[idx] for idx, vertex in enumerate(vertices)}
    cache_by_key[key] = force_map
    state._gmsh_tet_cauchy_viscous_force_cache = cache_by_key
    return force_map


def _multiphase_stress_force_cached(
    v,
    *,
    dim: int,
    state: VolumetricPitoisState,
    pressure_model=None,
) -> np.ndarray:
    F = np.zeros(dim)
    if (
        bool(getattr(state, "gmsh_compute_mesh", False))
        and int(dim) == 3
        and state.HC is not None
        and hasattr(v, "vd")
    ):
        if str(getattr(state.config, "solver_viscous_force_model", "edge_flux")) == "tet_cauchy":
            force = _gmsh_tet_cauchy_viscous_force_vector_cache(
                state,
                dim=dim,
                pressure_model=pressure_model,
                include_pressure=False,
            ).get(id(v))
        else:
            force = _gmsh_stress_force_vector_cache(
                state,
                pressure_model=pressure_model,
                dim=dim,
            ).get(id(v))
        if force is not None:
            F += np.asarray(force[:dim], dtype=float)
    elif state.HC is not None and hasattr(v, "vd"):
        p_i = _resolve_pressure_local(v, pressure_model, state.HC, dim)
        u_i = v.u[:dim]
        x_i = v.x_a[:dim]
        mu = state.mps.get_mu(v.phase) if state.mps is not None else 0.0
        for v_j in v.nn:
            try:
                A_ij = _cached_dual_area_vector(v, v_j, state=state, dim=dim)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError):
                continue
            p_j = _resolve_pressure_local(v_j, pressure_model, state.HC, dim)
            F -= 0.5 * (p_i + p_j) * A_ij
            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i
            d_norm = np.linalg.norm(d_ij)
            if d_norm < 1.0e-30:
                continue
            d_hat = d_ij / d_norm
            F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)
    F += _surface_tension_force_heron(v, dim=dim, state=state)
    return F


def _coalesce_axisymmetric_profile_samples(z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_arr = np.asarray(z, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    valid = np.isfinite(z_arr) & np.isfinite(r_arr) & (r_arr > 0.0)
    z_arr = z_arr[valid]
    r_arr = r_arr[valid]
    if z_arr.size <= 1:
        return z_arr, r_arr

    order = np.argsort(z_arr)
    z_arr = z_arr[order]
    r_arr = r_arr[order]
    z_span = max(float(z_arr[-1] - z_arr[0]), 1.0e-30)
    z_tol = max(1.0e-14, 1.0e-9 * z_span)

    z_unique: list[float] = []
    r_unique: list[float] = []
    i = 0
    while i < z_arr.size:
        j = i + 1
        while j < z_arr.size and abs(float(z_arr[j] - z_arr[i])) <= z_tol:
            j += 1
        z_group = z_arr[i:j]
        r_group = r_arr[i:j]
        z_unique.append(float(np.mean(z_group)))
        r_unique.append(float(np.max(r_group)))
        i = j
    return np.asarray(z_unique, dtype=float), np.asarray(r_unique, dtype=float)


def _axisymmetric_profile(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_vals = []
    radii = []
    for ring in state.outer_rings:
        if not ring:
            continue
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        z_vals.append(float(np.mean(coords[:, 2])))
        radii.append(float(np.mean(np.linalg.norm(coords[:, :2], axis=1))))

    z = np.array(z_vals, dtype=float)
    r = np.array(radii, dtype=float)
    if z.size < 3:
        return z, r, np.zeros_like(z)

    z, r = _coalesce_axisymmetric_profile_samples(z, r)
    if z.size < 3:
        return z, r, np.zeros_like(z)

    dr_dz = np.gradient(r, z, edge_order=1)
    d2r_dz2 = np.gradient(dr_dz, z, edge_order=1)
    denom = np.maximum(1.0 + dr_dz**2, 1.0e-12)
    kappa_meridional = -d2r_dz2 / np.power(denom, 1.5)
    kappa_azimuthal = 1.0 / np.maximum(r * np.sqrt(denom), 1.0e-12)
    return z, r, kappa_meridional + kappa_azimuthal


def _axisymmetric_neck_curvature_from_eq3_fit(
    z: np.ndarray,
    r: np.ndarray,
    *,
    side_rings: int,
) -> float | None:
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(z) & np.isfinite(r) & (r > 0.0)
    z = z[valid]
    r = r[valid]
    if z.size < 5:
        return None

    order = np.argsort(z)
    z = z[order]
    r = r[order]
    waist_idx = int(np.argmin(r))
    side = max(1, int(side_rings))
    lo = max(0, waist_idx - side)
    hi = min(len(z), waist_idx + side + 1)
    fit_indices = [idx for idx in range(lo, hi) if idx != waist_idx]

    if len(fit_indices) < 3:
        lo = max(0, waist_idx - side)
        hi = min(len(z), waist_idx + side + 1)
        fit_indices = list(range(lo, hi))
    if len(fit_indices) < 3:
        return None

    z_ref = float(z[waist_idx])
    zz = z[np.asarray(fit_indices, dtype=int)] - z_ref
    rr = r[np.asarray(fit_indices, dtype=int)]
    if np.unique(np.round(zz, 15)).size < 3:
        return None

    try:
        a, b, c = np.polyfit(zz, rr, 2)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    r_fit = float(c)
    if (not np.isfinite(r_fit)) or r_fit <= 1.0e-12:
        return None
    dr_dz = float(b)
    d2r_dz2 = float(2.0 * a)
    denom = max(1.0 + dr_dz * dr_dz, 1.0e-12)
    kappa_meridional = -d2r_dz2 / float(np.power(denom, 1.5))
    kappa_azimuthal = 1.0 / max(r_fit * float(np.sqrt(denom)), 1.0e-12)
    kappa = kappa_meridional + kappa_azimuthal
    if not np.isfinite(kappa):
        return None
    return float(kappa)


def _axisymmetric_neck_fit_radius_from_eq3_fit(
    z: np.ndarray,
    r: np.ndarray,
    *,
    side_rings: int,
) -> float | None:
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(z) & np.isfinite(r) & (r > 0.0)
    z = z[valid]
    r = r[valid]
    if z.size < 5:
        return None

    order = np.argsort(z)
    z = z[order]
    r = r[order]
    waist_idx = int(np.argmin(r))
    side = max(1, int(side_rings))
    lo = max(0, waist_idx - side)
    hi = min(len(z), waist_idx + side + 1)
    fit_indices = [idx for idx in range(lo, hi) if idx != waist_idx]

    if len(fit_indices) < 3:
        fit_indices = list(range(lo, hi))
    if len(fit_indices) < 3:
        return None

    z_ref = float(z[waist_idx])
    zz = z[np.asarray(fit_indices, dtype=int)] - z_ref
    rr = r[np.asarray(fit_indices, dtype=int)]
    if np.unique(np.round(zz, 15)).size < 3:
        return None

    try:
        _a, _b, c = np.polyfit(zz, rr, 2)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    if (not np.isfinite(c)) or c <= 1.0e-12:
        return None
    return float(c)


def _fheron_neck_pressure_scalar(state: VolumetricPitoisState) -> float | None:
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_heron_surface_tension", False)))
    ):
        return None
    waist_ring = min(
        [ring for ring in state.outer_rings if ring],
        key=lambda ring: float(_cap_radius(ring)),
        default=[],
    )
    if not waist_ring:
        return None
    force_map, area_map = _gmsh_surface_heron_force_area_maps(state)
    gamma = max(float(state.config.gamma), 1.0e-30)
    curvature_samples: list[float] = []
    for vertex in waist_ring:
        force = np.asarray(force_map.get(id(vertex), np.zeros(3, dtype=float)), dtype=float)
        area = float(area_map.get(id(vertex), 0.0))
        if area <= 1.0e-30 or not np.all(np.isfinite(force)):
            continue
        curvature = float(np.linalg.norm(force)) / (gamma * area)
        if np.isfinite(curvature) and curvature > 0.0:
            curvature_samples.append(curvature)
    if not curvature_samples:
        return None

    curvature = float(np.median(np.asarray(curvature_samples, dtype=float)))
    factor = float(getattr(state.config, "fheron_neck_pressure_factor", USER_FHERON_NECK_PRESSURE_FACTOR))
    state.last_fheron_neck_curvature = float(curvature)
    state.last_fheron_neck_pressure_factor = float(factor)
    state.last_pressure_source = "fheron_neck"
    return -float(state.config.pressure_scale) * gamma * factor * curvature


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    previous_fit = float(getattr(state, "last_pressure_neck_fit_radius", 0.0))
    previous_valid_fit = float(getattr(state, "last_valid_pressure_neck_fit_radius", previous_fit))
    if np.isfinite(previous_valid_fit) and previous_valid_fit > 0.0:
        state.last_pressure_neck_fit_radius = previous_valid_fit
    else:
        state.last_pressure_neck_fit_radius = 0.0
    if not state.config.use_axisymmetric_laplace_pressure:
        state.pressure_scalar = 0.0
        return

    z, r, kappa = _axisymmetric_profile(state)
    if z.size < 3:
        state.pressure_scalar = 0.0
        return

    pressure_model = str(getattr(state.config, "gorge_pressure_model", USER_GORGE_PRESSURE_MODEL)).strip().lower()
    use_fheron_after_initial = (
        pressure_model == "axisym_initial_fheron_neck"
        and float(getattr(state, "elapsed_time_s", 0.0)) > 0.0
    )
    if pressure_model == "fheron_neck" or use_fheron_after_initial:
        fheron_pressure = _fheron_neck_pressure_scalar(state)
        if fheron_pressure is not None:
            waist_ring = min(
                [ring for ring in state.outer_rings if ring],
                key=lambda ring: float(_cap_radius(ring)),
                default=[],
            )
            if waist_ring:
                raw_radius = float(_cap_radius(waist_ring))
                if np.isfinite(raw_radius) and raw_radius > 0.0:
                    state.last_pressure_neck_fit_radius = raw_radius
                    state.last_valid_pressure_neck_fit_radius = raw_radius
        state.pressure_scalar = float(fheron_pressure)
        return

    kappa_fit = _axisymmetric_neck_curvature_from_eq3_fit(
        z,
        r,
        side_rings=int(getattr(state.config, "pressure_neck_fit_side_rings", USER_PRESSURE_NECK_FIT_SIDE_RINGS)),
    )
    if kappa_fit is not None:
        r_fit = _axisymmetric_neck_fit_radius_from_eq3_fit(
            z,
            r,
            side_rings=int(getattr(state.config, "pressure_neck_fit_side_rings", USER_PRESSURE_NECK_FIT_SIDE_RINGS)),
        )
        if r_fit is not None:
            state.last_pressure_neck_fit_radius = float(r_fit)
            state.last_valid_pressure_neck_fit_radius = float(r_fit)
        axisym_pressure = float(state.config.pressure_scale * state.config.gamma * kappa_fit)
        if pressure_model == "axisym_fheron_floor":
            fheron_pressure = _fheron_neck_pressure_scalar(state)
            if fheron_pressure is not None and abs(float(fheron_pressure)) > abs(axisym_pressure):
                state.last_pressure_source = "fheron_neck_floor"
                state.pressure_scalar = float(fheron_pressure)
                return
        state.last_pressure_source = "axisym_fit"
        state.last_fheron_neck_pressure_factor = float(
            getattr(state.config, "fheron_neck_pressure_factor", USER_FHERON_NECK_PRESSURE_FACTOR)
        )
        if not hasattr(state, "last_fheron_neck_curvature"):
            state.last_fheron_neck_curvature = 0.0
        state.pressure_scalar = axisym_pressure
        return

    center = len(kappa) // 2
    lo = max(0, center - 2)
    hi = min(len(kappa), center + 3)
    kappa_sample = kappa[lo:hi]
    if kappa_sample.size == 0 or not np.all(np.isfinite(kappa_sample)):
        state.pressure_scalar = 0.0
        return
    axisym_pressure = float(state.config.pressure_scale * state.config.gamma * np.mean(kappa_sample))
    if pressure_model == "axisym_fheron_floor":
        fheron_pressure = _fheron_neck_pressure_scalar(state)
        if fheron_pressure is not None and abs(float(fheron_pressure)) > abs(axisym_pressure):
            state.last_pressure_source = "fheron_neck_floor"
            if use_fheron_after_initial:
                state.last_pressure_source = "fheron_neck_after_initial"
            state.pressure_scalar = float(fheron_pressure)
            return
    state.last_fheron_neck_pressure_factor = float(
        getattr(state.config, "fheron_neck_pressure_factor", USER_FHERON_NECK_PRESSURE_FACTOR)
    )
    if not hasattr(state, "last_fheron_neck_curvature"):
        state.last_fheron_neck_curvature = 0.0
    state.last_pressure_source = "axisym_sample"
    state.pressure_scalar = axisym_pressure


def _clip_acceleration(accel: np.ndarray, state: VolumetricPitoisState) -> np.ndarray:
    accel = np.asarray(accel, dtype=float)
    if not np.all(np.isfinite(accel)):
        return np.zeros_like(accel, dtype=float)
    accel_norm = float(np.linalg.norm(accel))
    if accel_norm > state.config.max_acceleration:
        accel = accel * (state.config.max_acceleration / accel_norm)
    return accel


def _state_finite_summary(state: VolumetricPitoisState) -> tuple[bool, str]:
    vertices = list(getattr(state, "volume_export_vertices", [])) or list(getattr(state.HC, "V", []))
    if not vertices:
        return True, "no vertices"

    coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float) for v in vertices], dtype=float)
    if not np.all(np.isfinite(coords)):
        bad = np.argwhere(~np.isfinite(coords))[0]
        return False, f"non-finite coordinate at vertex {int(bad[0])}, component {int(bad[1])}"
    if not np.all(np.isfinite(velocities)):
        bad = np.argwhere(~np.isfinite(velocities))[0]
        return False, f"non-finite velocity at vertex {int(bad[0])}, component {int(bad[1])}"

    max_coord_limit = float(_length_to_compute(state.config, 1.0))
    sphere_centers = np.asarray(
        [
            np.asarray(getattr(state, "bottom_sphere_center", np.zeros(3)), dtype=float),
            np.asarray(getattr(state, "top_sphere_center", np.zeros(3)), dtype=float),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(sphere_centers)):
        bad = np.argwhere(~np.isfinite(sphere_centers))[0]
        return False, f"non-finite sphere center {int(bad[0])}, component {int(bad[1])}"
    max_center = float(np.max(np.abs(sphere_centers)))
    if not np.isfinite(max_center) or max_center > max_coord_limit:
        max_center_m = float(_length_from_compute(state.config, max_center))
        return False, f"sphere-center blow-up: max |center| = {max_center_m:.6e} m"

    max_coord = float(np.max(np.abs(coords)))
    max_vel = float(np.max(np.linalg.norm(velocities, axis=1)))
    volume = float(_snapshot_msh_volume_m3(state))
    if not np.isfinite(volume):
        return False, "non-finite .msh volume"
    target_volume = float(getattr(state, "target_snapshot_volume_m3", volume))
    if np.isfinite(target_volume) and target_volume > 0.0:
        rel_volume_error = abs(volume - target_volume) / max(target_volume, 1.0e-30)
        if rel_volume_error > float(getattr(state.config, "max_volume_rel_error_for_abort", 0.25)):
            return False, (
                f".msh volume drift: current = {float(_volume_from_compute(state.config, volume)):.6e} m^3, "
                f"target = {float(_volume_from_compute(state.config, target_volume)):.6e} m^3, "
                f"rel = {rel_volume_error:.6e}"
            )
    max_vel_limit = float(_velocity_to_compute(state.config, 1.0))
    if not np.isfinite(max_coord) or max_coord > max_coord_limit:
        max_coord_m = float(_length_from_compute(state.config, max_coord))
        return False, f"coordinate blow-up: max |x| = {max_coord_m:.6e} m"
    if not np.isfinite(max_vel) or max_vel > max_vel_limit:
        max_vel_mps = float(_velocity_from_compute(state.config, max_vel))
        return False, f"velocity blow-up: max |u| = {max_vel_mps:.6e} m/s"
    return True, (
        f"max |x| = {float(_length_from_compute(state.config, max_coord)):.6e} m, "
        f"max |u| = {float(_velocity_from_compute(state.config, max_vel)):.6e} m/s, "
        f"volume = {float(_volume_from_compute(state.config, volume)):.6e} m^3"
    )


def _assert_state_finite(state: VolumetricPitoisState, *, where: str) -> None:
    ok, detail = _state_finite_summary(state)
    if ok:
        return
    message = f"Non-finite/unstable state after {where}: {detail}"
    if bool(USER_ABORT_ON_NONFINITE_STATE):
        raise RuntimeError(message)
    print(f"WARNING: {message}", flush=True)


def _boundary_vertex_lookup(state: VolumetricPitoisState) -> dict[tuple[float, float, float], object]:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        return {
            tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12)): v
            for v in state.surface_export_vertices
        }
    lookup: dict[tuple[float, float, float], object] = {}
    for ring in state.outer_rings:
        for v in ring:
            lookup[tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12))] = v
    for rings in (state.layer_rings[0], state.layer_rings[-1]):
        for ring in rings:
            for v in ring:
                lookup[tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12))] = v
    lookup[tuple(np.round(np.asarray(state.cap_bottom_center.x_a[:3], dtype=float), 12))] = state.cap_bottom_center
    lookup[tuple(np.round(np.asarray(state.cap_top_center.x_a[:3], dtype=float), 12))] = state.cap_top_center
    return lookup


def _boundary_dual_areas_and_normals(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    side_triangles = _actual_compute_side_surface_triangles(state)
    bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    triangle_sets = [arr for arr in (side_triangles, bottom_cap_triangles, top_cap_triangles) if arr.size]
    if not triangle_sets:
        return [], {}, {}

    lookup = _boundary_vertex_lookup(state)
    all_boundary_vertices = list(lookup.values())
    if not all_boundary_vertices:
        return [], {}, {}
    interior_ref = np.mean(
        [np.asarray(v.x_a[:3], dtype=float) for v in all_boundary_vertices],
        axis=0,
    )

    normal_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
    area_map: dict[int, float] = defaultdict(float)
    vertex_objects: dict[int, object] = {}

    for tri_set in triangle_sets:
        for tri in np.asarray(tri_set, dtype=float):
            a, b, c = tri
            normal = np.cross(b - a, c - a)
            norm_n = float(np.linalg.norm(normal))
            if norm_n <= 1.0e-30:
                continue
            centroid = (a + b + c) / 3.0
            if float(np.dot(normal, centroid - interior_ref)) < 0.0:
                normal = -normal
            area = 0.5 * float(np.linalg.norm(normal))
            for point in tri:
                key = tuple(np.round(np.asarray(point, dtype=float), 12))
                v = lookup.get(key)
                if v is None:
                    continue
                vid = id(v)
                vertex_objects[vid] = v
                normal_sum[vid] = normal_sum[vid] + normal
                area_map[vid] += area / 3.0

    normal_map: dict[int, np.ndarray] = {}
    for vid, nvec in normal_sum.items():
        norm_n = float(np.linalg.norm(nvec))
        if norm_n <= 1.0e-30:
            continue
        normal_map[vid] = nvec / norm_n

    vertices = [vertex_objects[vid] for vid in vertex_objects.keys()]
    return vertices, dict(area_map), normal_map


def _sidewall_dual_areas_and_normals(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    vertices = list(getattr(state, "surface_export_vertices", [])) or list(
        getattr(state, "volume_export_vertices", [])
    )
    side_tris = np.asarray(getattr(state, "surface_export_side_tris", np.empty((0, 3), dtype=int)), dtype=int)
    if vertices and side_tris.size:
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        valid_points = points[np.all(np.isfinite(points), axis=1)]
        interior_ref = np.mean(valid_points, axis=0) if valid_points.size else np.zeros(3, dtype=float)
        normal_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
        area_map: dict[int, float] = defaultdict(float)
        vertex_objects: dict[int, object] = {}

        for tri in side_tris.reshape((-1, 3)):
            idx = np.asarray(tri, dtype=int)
            if idx.size != 3 or int(np.min(idx)) < 0 or int(np.max(idx)) >= len(vertices):
                continue
            a, b, c = points[idx]
            normal = np.cross(b - a, c - a)
            norm_n = float(np.linalg.norm(normal))
            if norm_n <= 1.0e-30 or not np.isfinite(norm_n):
                continue
            centroid = (a + b + c) / 3.0
            if float(np.dot(normal, centroid - interior_ref)) < 0.0:
                normal = -normal
            area = 0.5 * float(np.linalg.norm(normal))
            for point_idx in idx:
                vertex = vertices[int(point_idx)]
                vid = id(vertex)
                vertex_objects[vid] = vertex
                normal_sum[vid] = normal_sum[vid] + normal
                area_map[vid] += area / 3.0

        normal_map: dict[int, np.ndarray] = {}
        for vid, nvec in normal_sum.items():
            norm_n = float(np.linalg.norm(nvec))
            if norm_n > 1.0e-30 and np.isfinite(norm_n):
                normal_map[vid] = nvec / norm_n
        return [vertex_objects[vid] for vid in vertex_objects.keys()], dict(area_map), normal_map

    side_triangles = _actual_compute_side_surface_triangles(state)
    if not side_triangles.size:
        return [], {}, {}
    lookup = _boundary_vertex_lookup(state)
    valid_vertices = list(lookup.values())
    if not valid_vertices:
        return [], {}, {}
    interior_ref = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in valid_vertices], axis=0)
    normal_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
    area_map: dict[int, float] = defaultdict(float)
    vertex_objects: dict[int, object] = {}
    for tri in np.asarray(side_triangles, dtype=float):
        a, b, c = tri
        normal = np.cross(b - a, c - a)
        norm_n = float(np.linalg.norm(normal))
        if norm_n <= 1.0e-30:
            continue
        centroid = (a + b + c) / 3.0
        if float(np.dot(normal, centroid - interior_ref)) < 0.0:
            normal = -normal
        area = 0.5 * float(np.linalg.norm(normal))
        for point in tri:
            vertex = lookup.get(tuple(np.round(np.asarray(point, dtype=float), 12)))
            if vertex is None:
                continue
            vid = id(vertex)
            vertex_objects[vid] = vertex
            normal_sum[vid] = normal_sum[vid] + normal
            area_map[vid] += area / 3.0
    normal_map: dict[int, np.ndarray] = {}
    for vid, nvec in normal_sum.items():
        norm_n = float(np.linalg.norm(nvec))
        if norm_n > 1.0e-30 and np.isfinite(norm_n):
            normal_map[vid] = nvec / norm_n
    return [vertex_objects[vid] for vid in vertex_objects.keys()], dict(area_map), normal_map


def _reset_volume_flux_correction_diagnostics(state: VolumetricPitoisState) -> None:
    state.last_final_noncl_volume_projection_status = "not_run"
    state.last_final_noncl_volume_projection_rel_before = 0.0
    state.last_final_noncl_volume_projection_rel_after = 0.0
    state.last_final_noncl_volume_projection_iters = 0
    state.last_final_noncl_volume_projection_max_disp_m = 0.0
    state.last_volume_flux_lambda_mps = 0.0
    state.last_volume_flux_before_m3ps = 0.0
    state.last_volume_flux_target_m3ps = 0.0
    state.last_volume_flux_after_m3ps = 0.0
    state.last_volume_flux_corrected_area_m2 = 0.0


def _apply_gmsh_volume_flux_velocity_correction(
    state: VolumetricPitoisState,
    *,
    dt: float,
) -> None:
    _reset_volume_flux_correction_diagnostics(state)
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(USER_ENABLE_VOLUME_FLUX_CORRECTION))
        or float(dt) <= 0.0
    ):
        state.last_final_noncl_volume_projection_status = "volflux_disabled"
        return

    side_vertices, side_area_map, side_normal_map = _sidewall_dual_areas_and_normals(state)
    if not side_vertices:
        state.last_final_noncl_volume_projection_status = "volflux_no_sidewall"
        return

    fixed = set(state.bV_caps)
    fixed.update(getattr(state, "bottom_contact_ring", []))
    fixed.update(getattr(state, "top_contact_ring", []))
    contact_ids = {id(vertex) for vertex in state.bottom_contact_ring}
    contact_ids.update(id(vertex) for vertex in state.top_contact_ring)
    corrected_vertices = [
        vertex
        for vertex in side_vertices
        if (
            vertex not in fixed
            and id(vertex) not in contact_ids
            and float(side_area_map.get(id(vertex), 0.0)) > 0.0
            and side_normal_map.get(id(vertex)) is not None
        )
    ]
    if not corrected_vertices:
        state.last_final_noncl_volume_projection_status = "volflux_no_movable_sidewall"
        return

    if bool(USER_VOLUME_FLUX_INCLUDE_CAP_BOUNDARY_FLUX):
        flux_vertices, flux_area_map, flux_normal_map = _boundary_dual_areas_and_normals(state)
    else:
        flux_vertices, flux_area_map, flux_normal_map = side_vertices, side_area_map, side_normal_map
    if not flux_vertices:
        state.last_final_noncl_volume_projection_status = "volflux_no_flux_boundary"
        return

    def boundary_flux(vertices: list, area_map: dict[int, float], normal_map: dict[int, np.ndarray]) -> float:
        flux = 0.0
        for vertex in vertices:
            vid = id(vertex)
            area = float(area_map.get(vid, 0.0))
            normal = normal_map.get(vid)
            if area <= 0.0 or normal is None:
                continue
            velocity = np.asarray(vertex.u[:3], dtype=float)
            if not np.all(np.isfinite(velocity)):
                continue
            flux += area * float(np.dot(velocity, np.asarray(normal, dtype=float)))
        return float(flux)

    q_before = boundary_flux(flux_vertices, flux_area_map, flux_normal_map)
    corrected_area = float(sum(float(side_area_map.get(id(vertex), 0.0)) for vertex in corrected_vertices))
    target_flux = 0.0
    state.last_volume_flux_before_m3ps = q_before
    state.last_volume_flux_target_m3ps = target_flux
    state.last_volume_flux_after_m3ps = q_before
    state.last_volume_flux_corrected_area_m2 = corrected_area
    if (not np.isfinite(q_before)) or corrected_area <= 1.0e-30:
        state.last_final_noncl_volume_projection_status = "volflux_bad_flux"
        return

    try:
        current_volume = float(_snapshot_msh_volume_m3(state))
        target_volume = float(getattr(state, "target_snapshot_volume_m3", current_volume))
        if np.isfinite(current_volume) and np.isfinite(target_volume) and target_volume > 0.0:
            rel = float((current_volume - target_volume) / max(target_volume, 1.0e-30))
            state.last_final_noncl_volume_projection_rel_before = rel
            state.last_final_noncl_volume_projection_rel_after = rel
            gain = max(float(USER_VOLUME_FLUX_VOLUME_ERROR_GAIN), 0.0)
            target_flux = gain * (target_volume - current_volume) / max(float(dt), 1.0e-30)
            if not np.isfinite(target_flux):
                target_flux = 0.0
            state.last_volume_flux_target_m3ps = target_flux
    except Exception:
        pass

    if abs(q_before - target_flux) <= 1.0e-30:
        state.last_final_noncl_volume_projection_status = "volflux_zero_flux"
        return

    relaxation = float(np.clip(USER_VOLUME_FLUX_RELAXATION, 0.0, 1.0))
    lambda_mps = relaxation * (q_before - target_flux) / max(corrected_area, 1.0e-30)
    max_disp = max(float(_length_to_compute(state.config, USER_VOLUME_FLUX_MAX_NORMAL_DISP_M)), 0.0)
    limited = False
    if max_disp > 0.0:
        max_lambda = max_disp / max(float(dt), 1.0e-30)
        if abs(lambda_mps) > max_lambda:
            lambda_mps = math.copysign(max_lambda, lambda_mps)
            limited = True

    if not np.isfinite(lambda_mps):
        state.last_final_noncl_volume_projection_status = "volflux_bad_lambda"
        return

    for vertex in corrected_vertices:
        normal = np.asarray(side_normal_map[id(vertex)], dtype=float)
        vertex.u = np.asarray(vertex.u[:3], dtype=float) - lambda_mps * normal

    state.last_volume_flux_lambda_mps = float(lambda_mps)
    state.last_volume_flux_after_m3ps = boundary_flux(flux_vertices, flux_area_map, flux_normal_map)
    state.last_final_noncl_volume_projection_max_disp_m = abs(float(lambda_mps) * float(dt))
    state.last_final_noncl_volume_projection_status = "volflux_limited" if limited else "volflux_applied"
    _axisym_clear_caches(state)


def _volume_gradient_areas_and_normals(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    _freeze_surface_topology(state)
    points = _volume_points_array(state)
    tets = np.asarray(state.volume_export_tets, dtype=int)
    vertices = list(state.volume_export_vertices)
    if points.size == 0 or tets.size == 0 or not vertices:
        return [], {}, {}

    grad_by_idx = np.zeros_like(points, dtype=float)
    for a, b, c, d in tets:
        ia = int(a)
        ib = int(b)
        ic = int(c)
        idd = int(d)
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        pd = points[idd]
        signed_six_volume = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd)))
        if abs(signed_six_volume) <= 1.0e-30:
            continue
        orient = 1.0 if signed_six_volume >= 0.0 else -1.0
        ga = orient * np.cross(pb - pd, pc - pd) / 6.0
        gb = orient * np.cross(pc - pd, pa - pd) / 6.0
        gc = orient * np.cross(pa - pd, pb - pd) / 6.0
        gd = -(ga + gb + gc)
        grad_by_idx[ia] += ga
        grad_by_idx[ib] += gb
        grad_by_idx[ic] += gc
        grad_by_idx[idd] += gd

    active_vertices: list[object] = []
    area_map: dict[int, float] = {}
    normal_map: dict[int, np.ndarray] = {}
    for idx, vertex in enumerate(vertices):
        grad = np.asarray(grad_by_idx[idx], dtype=float)
        area = float(np.linalg.norm(grad))
        if area <= 1.0e-30:
            continue
        vid = id(vertex)
        active_vertices.append(vertex)
        area_map[vid] = area
        normal_map[vid] = grad / area
    return active_vertices, area_map, normal_map


def _Fp_proj(v, *, state: VolumetricPitoisState) -> np.ndarray:
    p_proj = float(getattr(state, "pressure_projection_scalar", 0.0))
    if abs(p_proj) <= 1.0e-30:
        return np.zeros(3, dtype=float)
    area = float(state.pressure_projection_area_map.get(id(v), 0.0))
    normal = state.pressure_projection_normal_map.get(id(v))
    if area <= 0.0 or normal is None:
        return np.zeros(3, dtype=float)
    return p_proj * area * np.asarray(normal, dtype=float)


def _update_pressure_projection_scalar(state: VolumetricPitoisState, *, dt: float) -> None:
    state.pressure_projection_scalar = 0.0
    state.pressure_projection_area_map = {}
    state.pressure_projection_normal_map = {}
    if (not state.config.enable_volume_projection) or dt <= 0.0:
        return

    # Fp,proj is a physical pressure-traction correction.  It acts on the
    # liquid boundary only: liquid-gas sidewall plus wetted sphere caps.  The old
    # tetra-volume gradient map touched interior nodes, which is not a physical
    # pressure traction.
    boundary_vertices, area_map, normal_map = _boundary_dual_areas_and_normals(state)
    if not boundary_vertices:
        return

    current_volume = float(_snapshot_msh_volume_m3(state))
    target_volume = float(getattr(state, "target_snapshot_volume_m3", current_volume))
    predicted_contact_line_delta = 0.0
    if bool(USER_ENFORCE_FULL_AXISYMMETRY):
        predicted_contact_line_delta = float(getattr(state, "last_contact_line_volume_delta_m3", 0.0))
    desired_dVdt = (target_volume - current_volume - predicted_contact_line_delta) / float(dt)
    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    dVdt_pred = 0.0
    C = 0.0
    for v in boundary_vertices:
        vid = id(v)
        area = float(area_map.get(vid, 0.0))
        normal = normal_map.get(vid)
        if area <= 0.0 or normal is None:
            continue

        if v in state.bV_caps:
            u_pred = np.asarray(v.u[:3], dtype=float)
        else:
            Ftot = _Ftot(
                v,
                state=state,
                pressure_model=pressure_model,
                include_projected_pressure=False,
                include_contact_line=True,
                include_damping=True,
            )
            accel = Ftot / max(float(getattr(v, "m", 0.0)), 1.0e-12)
            accel = _clip_acceleration(accel, state)
            u_pred = np.asarray(v.u[:3], dtype=float) + float(dt) * accel
            C += (area * area) / max(float(getattr(v, "m", 0.0)), 1.0e-12)

        dVdt_pred += area * float(np.dot(u_pred, normal))

    if C <= 1.0e-30:
        return

    state.pressure_projection_scalar = float((desired_dVdt - dVdt_pred) / (float(dt) * C))
    state.pressure_projection_area_map = area_map
    state.pressure_projection_normal_map = normal_map


def _regularize_layer_order(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 3:
        return

    bottom_ring_center, top_ring_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    centers = [
        np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
        for ring in state.outer_rings
    ]
    total_span = float(np.dot(top_ring_center - bottom_ring_center, axis))
    if total_span <= 1.0e-30:
        return

    for k in range(1, len(state.outer_rings) - 1):
        center_k = centers[k]
        s_k = float(np.dot(center_k - bottom_ring_center, axis))
        target_s = float(state.layer_fractions[k]) * total_span
        shift = (target_s - s_k) * axis
        if np.linalg.norm(shift) <= 1.0e-30:
            continue
        for ring in state.layer_rings[k]:
            for v in ring:
                _move(v, tuple(np.asarray(v.x_a[:3], dtype=float) + shift), state.HC, state.bV_caps)
        center_v = state.layer_centers[k]
        _move(
            center_v,
            tuple(np.asarray(center_v.x_a[:3], dtype=float) + shift),
            state.HC,
            state.bV_caps,
        )


def _resplit_contact_axial_layers(state: VolumetricPitoisState) -> None:
    """Keep extra liquid-gas sidewall layers biased toward both contact lines."""
    extra_total = max(0, int(getattr(state.config, "cl_extra_axial_layers", 0)))
    if extra_total <= 0 or len(state.layer_rings) < 4:
        return

    lower_extra = (extra_total + 1) // 2
    upper_extra = extra_total // 2
    ratio = max(1.0, float(state.config.contact_line_radial_bias_ratio))

    def split(indices: list[int], *, cl_at_start: bool) -> None:
        if len(indices) < 3:
            return
        cl_to_anchor = list(indices) if cl_at_start else list(reversed(indices))
        gaps = len(cl_to_anchor) - 1
        weights = ratio ** np.arange(gaps, dtype=float)
        fractions = np.concatenate([[0.0], np.cumsum(weights / float(np.sum(weights)))])
        cl_idx = cl_to_anchor[0]
        anchor_idx = cl_to_anchor[-1]
        endpoint_layers = [
            [state.layer_centers[cl_idx]],
            *state.layer_rings[cl_idx],
        ]
        anchor_layers = [
            [state.layer_centers[anchor_idx]],
            *state.layer_rings[anchor_idx],
        ]
        for idx, frac in zip(cl_to_anchor[1:-1], fractions[1:-1]):
            target_groups = [
                [state.layer_centers[idx]],
                *state.layer_rings[idx],
            ]
            for target_group, cl_group, anchor_group in zip(target_groups, endpoint_layers, anchor_layers):
                targets = []
                for v_cl, v_anchor in zip(cl_group, anchor_group):
                    x_cl = np.asarray(v_cl.x_a[:3], dtype=float)
                    x_anchor = np.asarray(v_anchor.x_a[:3], dtype=float)
                    targets.append(tuple((1.0 - float(frac)) * x_cl + float(frac) * x_anchor))
                _move_vertices_batch(target_group, targets, state.HC, state.bV_caps)

    if lower_extra > 0 and lower_extra + 1 < len(state.layer_rings):
        split(list(range(0, lower_extra + 2)), cl_at_start=True)
    if upper_extra > 0 and len(state.layer_rings) - upper_extra - 2 >= 0:
        split(
            list(range(len(state.layer_rings) - upper_extra - 2, len(state.layer_rings))),
            cl_at_start=False,
        )


def _contact_axial_resplit_ranges(state: VolumetricPitoisState) -> tuple[tuple[list[int], bool], ...]:
    extra_total = max(0, int(getattr(state.config, "cl_extra_axial_layers", 0)))
    if extra_total <= 0 or len(state.outer_rings) < 4:
        return ()

    ranges: list[tuple[list[int], bool]] = []
    lower_extra = (extra_total + 1) // 2
    upper_extra = extra_total // 2
    if lower_extra > 0 and lower_extra + 1 < len(state.outer_rings):
        ranges.append((list(range(0, lower_extra + 2)), True))
    if upper_extra > 0 and len(state.outer_rings) - upper_extra - 2 >= 0:
        ranges.append((list(range(len(state.outer_rings) - upper_extra - 2, len(state.outer_rings))), False))
    return tuple(ranges)


def _resplit_gmsh_contact_axial_layers(state: VolumetricPitoisState) -> None:
    """Keep Gmsh sidewall rings from crossing the moving contact lines."""
    if not bool(getattr(state, "gmsh_compute_mesh", False)):
        return

    ratio = max(1.0, float(state.config.contact_line_radial_bias_ratio))

    def split(indices: list[int], *, cl_at_start: bool) -> None:
        if len(indices) < 3:
            return
        cl_to_anchor = list(indices) if cl_at_start else list(reversed(indices))
        gaps = len(cl_to_anchor) - 1
        weights = ratio ** np.arange(gaps, dtype=float)
        fractions = np.concatenate([[0.0], np.cumsum(weights / float(np.sum(weights)))])
        cl_ring = list(state.outer_rings[cl_to_anchor[0]])
        anchor_ring = list(state.outer_rings[cl_to_anchor[-1]])
        if len(cl_ring) != len(anchor_ring) or not cl_ring:
            return
        for idx, frac in zip(cl_to_anchor[1:-1], fractions[1:-1]):
            target_ring = list(state.outer_rings[idx])
            if len(target_ring) != len(cl_ring):
                continue
            targets = []
            for v_cl, v_anchor in zip(cl_ring, anchor_ring):
                x_cl = np.asarray(v_cl.x_a[:3], dtype=float)
                x_anchor = np.asarray(v_anchor.x_a[:3], dtype=float)
                targets.append(tuple((1.0 - float(frac)) * x_cl + float(frac) * x_anchor))
            _move_vertices_batch(target_ring, targets, state.HC, state.bV_caps)

    for indices, cl_at_start in _contact_axial_resplit_ranges(state):
        split(indices, cl_at_start=cl_at_start)
    _axisym_clear_caches(state)


def _begin_gmsh_export_contact_axial_resplit(state: VolumetricPitoisState) -> list[tuple[object, tuple[float, float, float]]]:
    """Apply the CL cleanup only for mesh export/rendering, then restore it."""
    saved: list[tuple[object, tuple[float, float, float]]] = []
    seen: set[int] = set()
    for indices, cl_at_start in _contact_axial_resplit_ranges(state):
        if len(indices) < 3:
            continue
        cl_to_anchor = list(indices) if cl_at_start else list(reversed(indices))
        for layer_idx in cl_to_anchor[1:-1]:
            for vertex in state.outer_rings[layer_idx]:
                key = id(vertex)
                if key in seen:
                    continue
                seen.add(key)
                saved.append((vertex, tuple(float(x) for x in np.asarray(vertex.x_a[:3], dtype=float))))
    if saved:
        _resplit_gmsh_contact_axial_layers(state)
    return saved


def _end_gmsh_export_contact_axial_resplit(
    state: VolumetricPitoisState,
    saved: list[tuple[object, tuple[float, float, float]]],
) -> None:
    if not saved:
        return
    _move_vertices_batch(
        [vertex for vertex, _position in saved],
        [position for _vertex, position in saved],
        state.HC,
        state.bV_caps,
    )
    _axisym_clear_caches(state)


def _pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
    pressure = float(state.pressure_scalar)
    if not state.config.include_gravity:
        return pressure

    z_ref = _solver_hydrostatic_z_ref(state)
    z = float(np.asarray(v.x_a[:3], dtype=float)[2])
    return pressure - float(state.config.rho_f) * float(state.config.gravity_mps2) * (z - z_ref)


def _solver_pr33_pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
    pressure = float(state.pressure_scalar)
    if bool(getattr(state.config, "solver_pr33_pressure_include_hydrostatic", False)):
        z_ref = _solver_hydrostatic_z_ref(state)
        z = float(np.asarray(v.x_a[:3], dtype=float)[2])
        pressure -= float(state.config.rho_f) * float(state.config.gravity_mps2) * (z - z_ref)
    pressure += float(getattr(state, "pressure_projection_scalar", 0.0))
    return pressure


def _zero_solver_pressure_model(v, *, HC=None, dim: int = 3) -> float:
    del v, HC, dim
    return 0.0


def _set_cap_velocities(state: VolumetricPitoisState) -> None:
    cap_speed = float(state.config.cap_speed)
    bottom_u = np.array([0.0, 0.0, -cap_speed], dtype=float)
    top_u = np.zeros(3, dtype=float)
    for v in state.cap_bottom_interior:
        v.u = bottom_u.copy()
    for v in state.cap_top_interior:
        v.u = top_u.copy()
    if not bool(getattr(state.config, "enable_gmsh_contact_angle_kinematics", False)):
        for v in getattr(state, "bottom_contact_ring", []):
            v.u = bottom_u.copy()
        for v in getattr(state, "top_contact_ring", []):
            v.u = top_u.copy()


def _contact_plane_centers_and_axis(state_or_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_rings = state_or_data.outer_rings if hasattr(state_or_data, "outer_rings") else state_or_data
    bottom_ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_rings[0]], axis=0)
    top_ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_rings[-1]], axis=0)
    axis = top_ring_center - bottom_ring_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return bottom_ring_center, top_ring_center, axis


def _swirl_axis_geometry(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.asarray(state.bottom_sphere_center, dtype=float)
    top = np.asarray(state.top_sphere_center, dtype=float)
    axis = top - bottom
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return 0.5 * (bottom + top), axis


def _azimuthal_unit(
    point: np.ndarray,
    *,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(point, dtype=float) - np.asarray(axis_origin, dtype=float)
    axial = float(np.dot(rel, axis))
    radial = rel - axial * axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-30:
        return np.zeros(3, dtype=float)
    e_r = radial / radial_norm
    e_theta = np.cross(axis, e_r)
    e_theta_norm = float(np.linalg.norm(e_theta))
    if e_theta_norm <= 1.0e-30:
        return np.zeros(3, dtype=float)
    return e_theta / e_theta_norm


def _remove_swirl_component(
    vector: np.ndarray,
    *,
    point: np.ndarray,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    e_theta = _azimuthal_unit(point, axis_origin=axis_origin, axis=axis)
    if float(np.linalg.norm(e_theta)) <= 1.0e-30:
        return np.asarray(vector, dtype=float)
    vec = np.asarray(vector, dtype=float)
    return vec - float(np.dot(vec, e_theta)) * e_theta


def _enforce_no_swirl_velocity_field(state: VolumetricPitoisState, *, vertices=None) -> None:
    if not bool(getattr(state.config, "enforce_no_swirl", False)):
        return
    axis_origin, axis = _swirl_axis_geometry(state)
    if vertices is None:
        vertices = list(state.HC.V)
    for v in vertices:
        u = np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float)
        v.u = _remove_swirl_component(
            u,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )


def _ring_center_and_radius(ring: list, center: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    ring_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    ring_center = np.mean(ring_coords, axis=0)
    tangential = ring_coords - center[None, :]
    tangential -= np.outer(np.dot(tangential, axis), axis)
    radii = np.linalg.norm(tangential, axis=1)
    return ring_center, float(np.mean(radii))


def _mean_ring_radius(ring: list, center: np.ndarray, axis: np.ndarray) -> float:
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    tangential = coords - center[None, :]
    tangential -= np.outer(np.dot(tangential, axis), axis)
    return float(np.mean(np.linalg.norm(tangential, axis=1)))


def _orthonormal_tangent_basis(axis: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(reference, dtype=float) - axis * float(np.dot(reference, axis))
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(seed, axis))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent = seed - axis * float(np.dot(seed, axis))
        tangent_norm = float(np.linalg.norm(tangent))
    e1 = tangent / max(tangent_norm, 1.0e-30)
    e2 = np.cross(axis, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1.0e-30)
    return e1, e2


def _contact_line_dr_ds_from_samples(x_values: np.ndarray, r_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float)
    r = np.asarray(r_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(r)
    x = x[finite]
    r = r[finite]
    if x.size < 2:
        return 0.0
    span = float(np.max(x) - np.min(x))
    if (not np.isfinite(span)) or span <= 1.0e-14:
        return 0.0
    dx = x - float(x[0])
    dr = r - float(r[0])
    denom = float(np.dot(dx, dx))
    linear_slope = float(np.dot(dx, dr) / denom) if denom > 1.0e-300 else 0.0
    if x.size < 3:
        return linear_slope
    x_scaled = dx / span
    deg = 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coeff = np.polyfit(x_scaled, r, deg=deg)
        slope = float(coeff[-2]) / span
    except Warning:
        slope = linear_slope
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        slope = linear_slope
    return slope if np.isfinite(slope) else linear_slope


def _estimate_contact_radius(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    contact_ring: list,
    next_ring: list,
    axis: np.ndarray,
    dt: float | None,
    gap_rate: float,
) -> float:
    prev_radius = _cap_radius(contact_ring)
    if not next_ring:
        return prev_radius

    theta_target = np.deg2rad(float(state.config.contact_angle_deg))
    radius = float(state.config.particle_radius)
    inward_sign = 1.0 if which == "bottom" else -1.0
    alpha_prev = float(np.arcsin(np.clip(prev_radius / max(radius, 1.0e-30), -1.0, 1.0)))
    slide_limit = _contact_line_slide_limit_compute(state.config)
    delta_alpha_max = slide_limit / max(radius, 1.0e-30)
    alpha_min = max(1.0e-8, alpha_prev - delta_alpha_max)
    alpha_max = min(0.5 * np.pi - 1.0e-8, alpha_prev + delta_alpha_max)
    if not bool(getattr(state.config, "allow_contact_line_growth", False)):
        alpha_max = min(alpha_max, alpha_prev)
    r_min = max(1.0e-8 * radius, radius * np.sin(alpha_min))
    radius_margin = float(_length_to_compute(state.config, 1.0e-6))
    r_max = min(radius - radius_margin, radius * np.sin(alpha_max))
    if r_max <= r_min:
        return min(max(prev_radius, r_min), radius - radius_margin)

    n_candidates = max(32, int(state.config.contact_radius_samples))
    candidates = np.linspace(r_min, r_max, n_candidates, dtype=float)
    local_rings = state.outer_rings[: max(2, int(state.config.contact_line_fit_rings) + 1)]
    if which == "top":
        local_rings = list(reversed(state.outer_rings[-max(2, int(state.config.contact_line_fit_rings) + 1) :]))
    sample_pairs: list[tuple[float, float]] = []
    for ring in local_rings[1:]:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs.append((s_k, r_k))
    if len(sample_pairs) < 2:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in next_ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs = [(s_k, r_k)]
    if (not bool(getattr(state.config, "allow_contact_line_growth", False))) and sample_pairs:
        local_profile_floor = max(
            [
                float(pair[1])
                for pair in sample_pairs[: max(1, min(2, len(sample_pairs)))]
                if np.isfinite(pair[1]) and float(pair[1]) > 0.0
            ],
            default=0.0,
        )
        if local_profile_floor > 0.0:
            local_profile_floor = min(
                float(prev_radius),
                float(USER_CONTACT_LINE_LOCAL_PROFILE_FLOOR_FRACTION) * local_profile_floor,
            )
            valid = candidates >= local_profile_floor
            if np.any(valid):
                candidates = candidates[valid]
            else:
                return float(prev_radius)

    s_contact = np.sqrt(np.maximum(radius**2 - candidates**2, 1.0e-16))
    tangent_solid = np.stack([-s_contact, candidates], axis=1)
    tangent_solid /= np.maximum(np.linalg.norm(tangent_solid, axis=1, keepdims=True), 1.0e-30)

    angle = np.empty_like(candidates)
    for idx, (r_contact, s_c) in enumerate(zip(candidates, s_contact)):
        fit_pairs = [(s_c, r_contact)] + sample_pairs[: max(2, int(state.config.contact_line_fit_rings))]
        s_vals = np.asarray([pair[0] for pair in fit_pairs], dtype=float)
        r_vals = np.asarray([pair[1] for pair in fit_pairs], dtype=float)
        x = s_vals - s_c
        dr_ds = _contact_line_dr_ds_from_samples(x, r_vals)
        tangent_liquid = np.array([dr_ds, 1.0], dtype=float)
        tangent_liquid /= max(float(np.linalg.norm(tangent_liquid)), 1.0e-30)
        dot = float(np.clip(np.dot(tangent_solid[idx], tangent_liquid), -1.0, 1.0))
        angle[idx] = float(np.arccos(dot))

    theta_candidate = np.full_like(candidates, theta_target)
    if bool(getattr(state.config, "enable_dynamic_contact_angle", False)) and dt is not None and dt > 0.0:
        alpha_candidates = np.arcsin(np.clip(candidates / max(radius, 1.0e-30), -1.0, 1.0))
        u_cl = radius * (alpha_candidates - alpha_prev) / dt
        capillary_number = float(state.config.mu_f) * u_cl / max(float(state.config.gamma), 1.0e-30)
        ln_fac = float(
            np.log(
                max(
                    float(state.config.contact_line_cox_macro_length_m)
                    / max(float(state.config.contact_line_cox_slip_length_m), 1.0e-30),
                    1.0,
                )
            )
        )
        theta_candidate = np.cbrt(
            np.maximum(theta_target**3 + 9.0 * capillary_number * ln_fac, np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)) ** 3)
        )
        theta_candidate = np.clip(
            theta_candidate,
            np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)),
            np.deg2rad(float(state.config.dynamic_contact_angle_max_deg)),
        )

    angle_residual = np.abs(angle - theta_candidate)
    continuation_penalty = float(state.config.contact_line_continuation_weight) * np.abs(
        candidates - prev_radius
    ) / max(radius, 1.0e-30)
    growth_penalty = np.zeros_like(candidates)
    if not bool(getattr(state.config, "allow_contact_line_growth", False)):
        growth_penalty = 0.25 * np.maximum(candidates - prev_radius, 0.0) / max(radius, 1.0e-30)
    score = angle_residual + continuation_penalty + growth_penalty
    best = int(np.argmin(score))
    if (not bool(getattr(state.config, "allow_contact_line_growth", False))) and candidates[best] < prev_radius:
        prev_idx = int(np.argmin(np.abs(candidates - prev_radius)))
        prev_residual = float(angle_residual[prev_idx])
        best_residual = float(angle_residual[best])
        absolute_improvement = prev_residual - best_residual
        min_abs = np.deg2rad(float(USER_CONTACT_LINE_MIN_ANGLE_IMPROVEMENT_DEG))
        min_rel = float(np.clip(USER_CONTACT_LINE_MIN_RELATIVE_IMPROVEMENT, 0.0, 1.0))
        relative_ok = best_residual <= (1.0 - min_rel) * max(prev_residual, 1.0e-30)
        if absolute_improvement < min_abs or not relative_ok:
            return float(prev_radius)
    return float(candidates[best])


def _contact_line_local_sample_pairs(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    axis: np.ndarray,
) -> list[tuple[float, float]]:
    inward_sign = 1.0 if which == "bottom" else -1.0
    local_rings = state.outer_rings[: max(2, int(state.config.contact_line_fit_rings) + 1)]
    if which == "top":
        local_rings = list(reversed(state.outer_rings[-max(2, int(state.config.contact_line_fit_rings) + 1) :]))
    sample_pairs: list[tuple[float, float]] = []
    for ring in local_rings[1:]:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs.append((s_k, r_k))
    return sample_pairs


def _contact_angle_for_radius(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    contact_radius: float,
) -> float:
    radius = float(state.config.particle_radius)
    inward_sign = 1.0 if which == "bottom" else -1.0
    r_contact = float(np.clip(contact_radius, 1.0e-8 * radius, radius - 1.0e-8))
    sample_pairs = _contact_line_local_sample_pairs(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
    )
    s_contact = float(np.sqrt(max(radius**2 - r_contact**2, 1.0e-16)))
    tangent_solid = np.array([-s_contact, r_contact], dtype=float)
    tangent_solid /= max(float(np.linalg.norm(tangent_solid)), 1.0e-30)
    fit_pairs = [(s_contact, r_contact)] + sample_pairs[: max(2, int(state.config.contact_line_fit_rings))]
    s_vals = np.asarray([pair[0] for pair in fit_pairs], dtype=float)
    r_vals = np.asarray([pair[1] for pair in fit_pairs], dtype=float)
    x = s_vals - s_contact
    dr_ds = _contact_line_dr_ds_from_samples(x, r_vals)
    tangent_liquid = np.array([dr_ds, 1.0], dtype=float)
    tangent_liquid /= max(float(np.linalg.norm(tangent_liquid)), 1.0e-30)
    dot = float(np.clip(np.dot(tangent_solid, tangent_liquid), -1.0, 1.0))
    return float(np.arccos(dot))


def _dynamic_contact_angle_from_speed(state: VolumetricPitoisState, *, slide_speed: float) -> float:
    theta_eq = float(np.deg2rad(state.config.contact_angle_deg))
    if not bool(getattr(state.config, "enable_dynamic_contact_angle", False)):
        return theta_eq
    capillary_number = float(state.config.mu_f) * float(slide_speed) / max(float(state.config.gamma), 1.0e-30)
    ln_fac = float(
        np.log(
            max(
                float(state.config.contact_line_cox_macro_length_m)
                / max(float(state.config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
    )
    theta_dyn = float(
        np.cbrt(
            max(
                theta_eq**3 + 9.0 * capillary_number * ln_fac,
                np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)) ** 3,
            )
        )
    )
    return float(
        np.clip(
            theta_dyn,
            np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)),
            np.deg2rad(float(state.config.dynamic_contact_angle_max_deg)),
        )
    )


def _contact_line_slide_direction(
    point: np.ndarray,
    *,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    which: str,
) -> np.ndarray:
    sign = 1.0 if which == "bottom" else -1.0
    rel = np.asarray(point, dtype=float) - np.asarray(sphere_center, dtype=float)
    axial_coeff = float(np.dot(rel, axis))
    tangential = rel - axial_coeff * axis
    radial_mag = float(np.linalg.norm(tangential))
    if radial_mag <= 1.0e-30:
        return np.zeros(3, dtype=float)
    tangential_unit = tangential / radial_mag
    slide_raw = sign * (axial_coeff * tangential_unit - radial_mag * axis)
    norm_slide = float(np.linalg.norm(slide_raw))
    if norm_slide <= 1.0e-30:
        return np.zeros(3, dtype=float)
    return slide_raw / norm_slide


def _contact_line_ring_speed(
    state: VolumetricPitoisState,
    *,
    ring: list,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    which: str,
) -> float:
    sphere_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float) if which == "bottom" else np.zeros(3, dtype=float)
    signed_speeds = []
    for v in ring:
        point = np.asarray(v.x_a[:3], dtype=float)
        slide_dir = _contact_line_slide_direction(point, sphere_center=sphere_center, axis=axis, which=which)
        if float(np.linalg.norm(slide_dir)) <= 1.0e-30:
            continue
        u_rel = np.asarray(v.u[:3], dtype=float) - sphere_vel
        signed_speeds.append(float(np.dot(u_rel, slide_dir)))
    if not signed_speeds:
        return 0.0
    return float(np.mean(signed_speeds))


def _ring_segment_length_map(ring: list) -> dict[int, float]:
    if len(ring) < 2:
        return {id(v): 0.0 for v in ring}
    coords = [np.asarray(v.x_a[:3], dtype=float) for v in ring]
    seg_map: dict[int, float] = {}
    n = len(ring)
    for i, v in enumerate(ring):
        prev_len = float(np.linalg.norm(coords[i] - coords[(i - 1) % n]))
        next_len = float(np.linalg.norm(coords[(i + 1) % n] - coords[i]))
        seg_map[id(v)] = 0.5 * (prev_len + next_len)
    return seg_map


def _cox_contact_line_signed_speed(state: VolumetricPitoisState, *, which: str) -> float:
    if which == "top":
        ring = state.top_contact_ring
    else:
        ring = state.bottom_contact_ring
    if not ring:
        return 0.0
    bottom_center, top_center, axis = _particle_centers_physical(state)
    sphere_center = top_center if which == "top" else bottom_center
    return float(
        _contact_line_ring_speed(
            state,
            ring=ring,
            sphere_center=sphere_center,
            axis=axis,
            which=which,
        )
    )


def _cox_uy_contact_line_liquid_force_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    cached = getattr(state, "_cox_uy_contact_line_force_cache", None)
    if cached is not None:
        return cached

    force_map: dict[int, np.ndarray] = {}
    bottom_center, top_center, axis = _particle_centers_physical(state)
    theta_eq = float(np.deg2rad(state.config.contact_angle_deg))
    gamma = float(state.config.gamma)

    for which, ring, sphere_center in (
        ("bottom", state.bottom_contact_ring, bottom_center),
        ("top", state.top_contact_ring, top_center),
    ):
        if not ring:
            continue
        speed = _cox_contact_line_signed_speed(state, which=which)
        theta_dyn = _dynamic_contact_angle_from_speed(state, slide_speed=speed)
        young_coeff = gamma * (float(np.cos(theta_dyn)) - float(np.cos(theta_eq)))
        seg_map = _ring_segment_length_map(ring)
        for v in ring:
            point = np.asarray(v.x_a[:3], dtype=float)
            slide_dir = _contact_line_slide_direction(
                point,
                sphere_center=np.asarray(sphere_center, dtype=float),
                axis=axis,
                which=which,
            )
            if float(np.linalg.norm(slide_dir)) <= 1.0e-30:
                continue
            ell_i = max(float(seg_map.get(id(v), 0.0)), 0.0)
            force = young_coeff * ell_i * slide_dir
            if np.all(np.isfinite(force)):
                force_map[id(v)] = force_map.get(id(v), np.zeros(3, dtype=float)) + force

    state._cox_uy_contact_line_force_cache = force_map
    return force_map


def _Fcl(v, *, state: VolumetricPitoisState) -> np.ndarray:
    if bool(getattr(state, "gmsh_compute_mesh", False)) and bool(
        getattr(state.config, "use_cox_contact_line_force", USER_USE_COX_CONTACT_LINE_FORCE)
    ):
        force = _cox_uy_contact_line_liquid_force_map(state).get(id(v))
        if force is not None:
            return np.asarray(force[:3], dtype=float)
    return np.zeros(3, dtype=float)


def _align_layer_azimuths(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 2:
        return

    bottom_ring_center, _top_ring_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    ref_ring = state.outer_rings[0]
    ref_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], dtype=float)
    ref_center = np.mean(ref_coords, axis=0)
    reference = ref_coords[0] - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    def wrap(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    ref_rel = ref_coords - ref_center[None, :]
    ref_rel -= np.outer(np.dot(ref_rel, axis), axis)
    # Preserve the inherited cyclic ring order. Sorting would hide a whole-ring
    # phase slip and let fixed connectivity land on the wrong geometric points.
    ref_angles = np.arctan2(ref_rel @ e2, ref_rel @ e1)

    for k, rings in enumerate(state.layer_rings):
        outer = state.outer_rings[k]
        outer_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in outer], dtype=float)
        outer_center = np.mean(outer_coords, axis=0)
        rel = outer_coords - outer_center[None, :]
        rel -= np.outer(np.dot(rel, axis), axis)
        angles = np.arctan2(rel @ e2, rel @ e1)
        if len(angles) != len(ref_angles):
            continue
        shift = float(np.mean(wrap(angles - ref_angles)))
        if abs(shift) <= 1.0e-10:
            continue

        layer_center = np.asarray(state.layer_centers[k].x_a[:3], dtype=float)
        cos_s = float(np.cos(-shift))
        sin_s = float(np.sin(-shift))
        for ring in rings:
            targets = []
            for v in ring:
                pos = np.asarray(v.x_a[:3], dtype=float)
                rel_v = pos - layer_center
                axial = axis * float(np.dot(rel_v, axis))
                radial = rel_v - axial
                x = float(np.dot(radial, e1))
                y = float(np.dot(radial, e2))
                radial_rot = (cos_s * x - sin_s * y) * e1 + (sin_s * x + cos_s * y) * e2
                target = layer_center + axial + radial_rot
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _initial_axisymmetric_template(state: VolumetricPitoisState):
    bottom_plane_center, top_plane_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    mid = 0.5 * (bottom_plane_center + top_plane_center)
    ref_ring = state.outer_rings[0]
    ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
    reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    layer_data = []
    for layer_center, rings in zip(state.layer_centers, state.layer_rings):
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0)
        s = float(np.dot(outer_center - mid, axis))
        outer_radius = max(_mean_ring_radius(rings[-1], outer_center, axis), 1.0e-30)
        ring_entries = []
        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - outer_center[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            angles = np.arctan2(rel @ e2, rel @ e1)
            order = np.argsort(angles)
            ordered_vertices = [ring[i] for i in order]
            factor = _mean_ring_radius(ring, outer_center, axis) / outer_radius
            ring_entries.append((ordered_vertices, float(factor)))
        layer_data.append(
            {
                "center_vertex": layer_center,
                "s": s,
                "rings": ring_entries,
            }
        )

    return mid, axis, e1, e2, layer_data


def _unduloid_like_outer_radius(
    z_abs: np.ndarray,
    *,
    half_span: float,
    neck_radius: float,
    contact_radius: float,
    end_slope: float,
) -> np.ndarray:
    if half_span <= 1.0e-30:
        return np.full_like(z_abs, contact_radius, dtype=float)
    u = half_span * half_span
    a = contact_radius - neck_radius
    b = (2.0 * a / u) - (end_slope / (2.0 * half_span))
    c = ((end_slope / (2.0 * half_span)) - (a / u)) / u
    r = neck_radius + b * z_abs**2 + c * z_abs**4
    return np.clip(r, 1.0e-8 * contact_radius, contact_radius)


def _apply_axisymmetric_outer_radii_profile(
    state: VolumetricPitoisState,
    *,
    template,
    outer_radii: np.ndarray,
) -> None:
    mid, axis, e1, e2, layer_data = template
    contact_radius = float(state.config.target_cap_radius)

    for layer, outer_radius in zip(layer_data, outer_radii):
        center_coord = mid + float(layer["s"]) * axis
        _move(layer["center_vertex"], tuple(center_coord), state.HC, state.bV_caps)
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i, v in enumerate(ordered_vertices):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )


def _apply_initial_unduloid_like_profile(
    state: VolumetricPitoisState,
    *,
    template,
    neck_radius: float,
) -> None:
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
    half_span = float(np.max(np.abs(s_vals)))
    contact_radius = float(state.config.target_cap_radius)
    axial_offset = float(np.sqrt(max(state.config.particle_radius**2 - contact_radius**2, 0.0)))
    end_slope = axial_offset / max(contact_radius, 1.0e-30)
    outer_radii = _unduloid_like_outer_radius(
        np.abs(s_vals),
        half_span=half_span,
        neck_radius=neck_radius,
        contact_radius=contact_radius,
        end_slope=end_slope,
    )
    _apply_axisymmetric_outer_radii_profile(
        state,
        template=template,
        outer_radii=outer_radii,
    )


def _apply_initial_power_pinch_profile(
    state: VolumetricPitoisState,
    *,
    template,
    exponent: float,
    neck_radius: float,
) -> None:
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
    half_span = float(np.max(np.abs(s_vals)))
    contact_radius = float(state.config.target_cap_radius)
    if half_span <= 1.0e-30:
        outer_radii = np.full_like(s_vals, contact_radius, dtype=float)
    else:
        u = np.clip(np.abs(s_vals) / half_span, 0.0, 1.0)
        outer_radii = neck_radius + (contact_radius - neck_radius) * np.power(u, float(exponent))
        outer_radii = np.clip(outer_radii, neck_radius, contact_radius)
    _apply_axisymmetric_outer_radii_profile(
        state,
        template=template,
        outer_radii=outer_radii,
    )


def _initialise_unduloid_like_equilibrium(state: VolumetricPitoisState) -> None:
    template = _initial_axisymmetric_template(state)
    contact_radius = float(state.config.target_cap_radius)
    target_volume = float(state.config.initial_bridge_volume_m3)
    _freeze_volume_msh_topology(state)

    def _apply_ratio_and_volume(neck_ratio: float) -> float:
        ratio = float(np.clip(neck_ratio, 0.05, 0.98))
        _apply_initial_unduloid_like_profile(
            state,
            template=template,
            neck_radius=ratio * contact_radius,
        )
        _update_duals_and_masses(state)
        return float(
            _indexed_tet_mesh_volume_m3(
                _volume_points_array(state),
                np.asarray(state.volume_export_tets, dtype=int),
            )
        )

    neck_ratio_seed = float(np.clip(state.config.initial_neck_radius_ratio, 0.05, 0.98))
    ratio_low = 0.05
    ratio_high = 0.98
    volume_low = _apply_ratio_and_volume(ratio_low)
    volume_high = _apply_ratio_and_volume(ratio_high)

    solved = False
    if min(volume_low, volume_high) <= target_volume <= max(volume_low, volume_high):
        low = ratio_low
        high = ratio_high
        increasing = volume_high >= volume_low
        best_ratio = neck_ratio_seed
        best_err = float("inf")
        for _ in range(32):
            mid = 0.5 * (low + high)
            vol_mid = _apply_ratio_and_volume(mid)
            err = abs(vol_mid - target_volume)
            if err < best_err:
                best_ratio = mid
                best_err = err
            if err / max(target_volume, 1.0e-30) <= 1.0e-8:
                best_ratio = mid
                break
            if increasing:
                if vol_mid < target_volume:
                    low = mid
                else:
                    high = mid
            else:
                if vol_mid > target_volume:
                    low = mid
                else:
                    high = mid
        _apply_ratio_and_volume(best_ratio)
        solved = True

    if not solved:
        # The quartic family is too fat for the requested bridge volume under the
        # positive-tetra volume definition. Fall back to a sharper analytic
        # power-law pinch profile and solve its exponent to hit the target.
        min_neck_radius = max(1.0e-3 * contact_radius, 1.0e-8)

        def _apply_power_and_volume(exponent: float) -> float:
            _apply_initial_power_pinch_profile(
                state,
                template=template,
                exponent=float(exponent),
                neck_radius=min_neck_radius,
            )
            _update_duals_and_masses(state)
            return float(
                _indexed_tet_mesh_volume_m3(
                    _volume_points_array(state),
                    np.asarray(state.volume_export_tets, dtype=int),
                )
            )

        exp_low = 0.5
        exp_high = 32.0
        vol_exp_low = _apply_power_and_volume(exp_low)
        vol_exp_high = _apply_power_and_volume(exp_high)
        if min(vol_exp_low, vol_exp_high) <= target_volume <= max(vol_exp_low, vol_exp_high):
            low = exp_low
            high = exp_high
            decreasing = vol_exp_high <= vol_exp_low
            best_exp = exp_high
            best_err = float("inf")
            for _ in range(40):
                mid = 0.5 * (low + high)
                vol_mid = _apply_power_and_volume(mid)
                err = abs(vol_mid - target_volume)
                if err < best_err:
                    best_exp = mid
                    best_err = err
                if err / max(target_volume, 1.0e-30) <= 1.0e-8:
                    best_exp = mid
                    break
                if decreasing:
                    if vol_mid > target_volume:
                        low = mid
                    else:
                        high = mid
                else:
                    if vol_mid < target_volume:
                        low = mid
                    else:
                        high = mid
            _apply_power_and_volume(best_exp)
        else:
            # As a last fallback, keep the sharpest available analytic profile.
            _apply_power_and_volume(exp_high)
    _update_duals_and_masses(state)


def _gmsh_element_to_tets(element_type: int, row: list[int]) -> list[list[int]]:
    if int(element_type) == 4 and len(row) >= 4:
        return [row[:4]]
    if int(element_type) == 5 and len(row) >= 8:
        a, b, c, d, e, f, g, h = row[:8]
        return [[a, b, d, e], [b, c, d, g], [b, d, e, g], [b, e, f, g], [d, e, g, h]]
    if int(element_type) == 6 and len(row) >= 6:
        a, b, c, d, e, f = row[:6]
        return [[a, b, c, d], [b, c, d, e], [c, d, e, f]]
    if int(element_type) == 7 and len(row) >= 5:
        a, b, c, d, apex = row[:5]
        return [[a, b, c, apex], [a, c, d, apex]]
    return []


def _read_gmsh2_points_triangles_tets(path: Path, lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = lines.index("$Nodes") + 1
    n_nodes = int(lines[idx].split()[0])
    idx += 1
    node_tags: list[int] = []
    points: list[tuple[float, float, float]] = []
    for _node in range(n_nodes):
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 4:
            raise ValueError(f"Bad Gmsh 2 node row while reading {path}: {parts}")
        node_tags.append(int(parts[0]))
        points.append((float(parts[1]), float(parts[2]), float(parts[3])))

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    idx = lines.index("$Elements") + 1
    n_elements = int(lines[idx].split()[0])
    idx += 1
    triangles: list[list[int]] = []
    tets: list[list[int]] = []
    for _element in range(n_elements):
        parts = [int(x) for x in lines[idx].split()]
        idx += 1
        if len(parts) < 4:
            continue
        element_type = int(parts[1])
        n_tags = int(parts[2])
        node_part = parts[3 + n_tags :]
        try:
            row = [tag_to_idx[tag] for tag in node_part]
        except KeyError as exc:
            raise ValueError(f"Element references missing node tag {exc} while reading {path}") from exc
        if element_type == 2 and len(row) >= 3:
            triangles.append(row[:3])
        elif element_type == 3 and len(row) >= 4:
            a, b, c, d = row[:4]
            triangles.append([a, b, c])
            triangles.append([a, c, d])
        else:
            tets.extend(_gmsh_element_to_tets(element_type, row))

    return (
        np.asarray(points, dtype=float),
        np.asarray(triangles, dtype=int),
        np.asarray(tets, dtype=int),
        np.asarray(node_tags, dtype=int),
    )


def _read_gmsh4_points_triangles_tets(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()

    mesh_format = lines[lines.index("$MeshFormat") + 1].split()
    version = float(mesh_format[0]) if mesh_format else 4.1
    if version < 4.0:
        return _read_gmsh2_points_triangles_tets(path, lines)

    idx = lines.index("$Nodes") + 1
    n_blocks, n_nodes, _min_tag, _max_tag = map(int, lines[idx].split()[:4])
    idx += 1
    node_tags: list[int] = []
    points: list[tuple[float, float, float]] = []
    for _block in range(n_blocks):
        _entity_dim, _entity_tag, _parametric, n_block_nodes = map(int, lines[idx].split()[:4])
        idx += 1
        block_tags = [int(lines[idx + j].split()[0]) for j in range(n_block_nodes)]
        idx += n_block_nodes
        for tag in block_tags:
            xyz = lines[idx].split()
            idx += 1
            node_tags.append(int(tag))
            points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    if len(points) != n_nodes:
        raise ValueError(f"Node count mismatch while reading {path}")

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    idx = lines.index("$Elements") + 1
    n_blocks, _n_elements, _min_elem, _max_elem = map(int, lines[idx].split()[:4])
    idx += 1
    triangles: list[list[int]] = []
    tets: list[list[int]] = []
    for _block in range(n_blocks):
        _entity_dim, _entity_tag, element_type, n_block_elements = map(int, lines[idx].split()[:4])
        idx += 1
        for _element in range(n_block_elements):
            parts = [int(x) for x in lines[idx].split()]
            idx += 1
            row = [tag_to_idx[tag] for tag in parts[1:]]
            if int(element_type) == 2 and len(row) >= 3:
                triangles.append(row[:3])
            elif int(element_type) == 3 and len(row) >= 4:
                a, b, c, d = row[:4]
                triangles.append([a, b, c])
                triangles.append([a, c, d])
            else:
                tets.extend(_gmsh_element_to_tets(element_type, row))

    return (
        np.asarray(points, dtype=float),
        np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int),
        np.asarray(tets, dtype=int) if tets else np.empty((0, 4), dtype=int),
        np.asarray(node_tags, dtype=int),
    )


def _axisymmetric_profile_from_gmsh_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    z_abs = np.abs(pts[:, 2])
    radii = np.linalg.norm(pts[:, :2], axis=1)

    by_z: dict[float, float] = {}
    for z_val, radius in zip(z_abs, radii):
        key = round(float(z_val), 15)
        by_z[key] = max(by_z.get(key, 0.0), float(radius))

    half_z_all = np.asarray(sorted(by_z), dtype=float)
    half_r_all = np.asarray([by_z[float(z)] for z in half_z_all], dtype=float)
    if half_z_all.size < 2:
        raise ValueError(f"Could not read an axisymmetric profile from {HARDCODED_INITIAL_MSH_PATH}")

    keep_z: list[float] = []
    keep_r: list[float] = []
    current_max = -float("inf")
    tol = max(1.0e-12, 1.0e-9 * float(np.max(half_r_all)))
    for z_val, radius in zip(half_z_all, half_r_all):
        if radius + tol < current_max:
            continue
        keep_z.append(float(z_val))
        keep_r.append(float(radius))
        current_max = max(current_max, float(radius))

    # A reloadable bridge mesh must keep the z-extreme contact rings even when
    # the sidewall bulges outward just inside the solid boundary.  Otherwise the
    # loader mistakes the first bulged free-surface ring for the contact ring
    # and infers the wrong sphere separation.
    end_z = float(half_z_all[-1])
    if not any(abs(float(z_val) - end_z) <= 1.0e-14 for z_val in keep_z):
        keep_z.append(end_z)
        keep_r.append(float(half_r_all[-1]))

    order = np.argsort(np.asarray(keep_z, dtype=float))
    half_z = np.asarray(keep_z, dtype=float)[order]
    half_r = np.asarray(keep_r, dtype=float)[order]
    sample_s = np.concatenate((-half_z[:0:-1], half_z))
    sample_r = np.concatenate((half_r[:0:-1], half_r))
    return sample_s, sample_r


def _ordered_ring_from_mask(vertices: list, points: np.ndarray, mask: np.ndarray) -> list:
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return []
    angles = np.arctan2(points[ids, 1], points[ids, 0])
    order = np.argsort(angles)
    return [vertices[int(ids[i])] for i in order]


def _gmsh_free_surface_rings(vertices: list, points: np.ndarray) -> list[list[object]]:
    sample_s, sample_r = _axisymmetric_profile_from_gmsh_points(points)
    z_vals = points[:, 2]
    r_vals = np.linalg.norm(points[:, :2], axis=1)
    z_span = max(float(np.max(z_vals) - np.min(z_vals)), 1.0e-12)
    r_max = max(float(np.max(r_vals)), 1.0e-12)
    z_tol = max(5.0e-12, 1.0e-8 * z_span)
    r_tol = max(5.0e-12, 1.0e-6 * r_max)
    rings: list[list[object]] = []
    for z_target, r_target in zip(sample_s, sample_r):
        mask = (np.abs(z_vals - float(z_target)) <= z_tol) & (np.abs(r_vals - float(r_target)) <= r_tol)
        ring = _ordered_ring_from_mask(vertices, points, mask)
        if len(ring) >= 3:
            rings.append(ring)
    if len(rings) < 2:
        raise RuntimeError(f"Could not extract liquid-air rings directly from {HARDCODED_INITIAL_MSH_PATH}")
    return rings


def _split_gmsh_boundary_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_idx = np.asarray(triangles, dtype=int)
    if tri_idx.size == 0:
        empty = np.empty((0, 3), dtype=int)
        return empty, empty, empty

    bottom: list[list[int]] = []
    top: list[list[int]] = []
    side: list[list[int]] = []
    tol = max(2.0e-8, 2.0e-5 * float(radius))
    for tri in tri_idx:
        xyz = points[np.asarray(tri, dtype=int)]
        centroid = np.mean(xyz, axis=0)
        bottom_vertex = (
            (xyz[:, 2] <= tol)
            & (np.abs(np.linalg.norm(xyz - bottom_sphere_center[None, :], axis=1) - float(radius)) <= tol)
        )
        top_vertex = (
            (xyz[:, 2] >= -tol)
            & (np.abs(np.linalg.norm(xyz - top_sphere_center[None, :], axis=1) - float(radius)) <= tol)
        )
        if centroid[2] < 0.0 and bool(np.all(bottom_vertex)):
            bottom.append([int(v) for v in tri])
        elif centroid[2] > 0.0 and bool(np.all(top_vertex)):
            top.append([int(v) for v in tri])
        else:
            side.append([int(v) for v in tri])

    return (
        np.asarray(side, dtype=int) if side else np.empty((0, 3), dtype=int),
        np.asarray(bottom, dtype=int) if bottom else np.empty((0, 3), dtype=int),
        np.asarray(top, dtype=int) if top else np.empty((0, 3), dtype=int),
    )


def _build_gmsh_axisym_ring_specs(
    vertices: list,
    points: np.ndarray,
    *,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
) -> list[SimpleNamespace]:
    bottom = np.asarray(bottom_sphere_center, dtype=float)
    top = np.asarray(top_sphere_center, dtype=float)
    axis = top - bottom
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (bottom + top)
    e1, e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))

    pts = np.asarray(points, dtype=float)
    groups: dict[tuple[float, float], list[tuple[float, object]]] = defaultdict(list)
    for vertex, point in zip(vertices, pts):
        rel = np.asarray(point, dtype=float) - mid
        s = float(np.dot(rel, axis))
        radial_vec = rel - s * axis
        r = float(np.linalg.norm(radial_vec))
        if r <= 1.0e-30:
            angle = 0.0
        else:
            angle = float(np.arctan2(float(np.dot(radial_vec, e2)), float(np.dot(radial_vec, e1))))
        groups[(round(s, 12), round(r, 12))].append((angle, vertex))

    specs: list[SimpleNamespace] = []
    for (s0, r0), entries in groups.items():
        entries.sort(key=lambda item: item[0])
        specs.append(
            SimpleNamespace(
                s0=float(s0),
                r0=float(r0),
                vertices=[vertex for _angle, vertex in entries],
                angles=tuple(float(angle) for angle, _vertex in entries),
            )
        )
    specs.sort(key=lambda spec: (float(spec.s0), float(spec.r0), len(spec.vertices)))
    return specs


def _tet_signed_triples(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int).reshape((-1, 4))
    if pts.size == 0 or tet_idx.size == 0:
        return np.empty(0, dtype=float)
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    return np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))


def _orient_tets_like_reference(
    *,
    reference_points: np.ndarray,
    points: np.ndarray,
    tets: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    tet_idx = np.asarray(tets, dtype=int).reshape((-1, 4)).copy()
    ref_sign = np.sign(_tet_signed_triples(reference_points, tet_idx))
    new_sign = np.sign(_tet_signed_triples(points, tet_idx))
    reorder = (ref_sign != 0.0) & (new_sign != 0.0) & (ref_sign != new_sign)
    if np.any(reorder):
        tmp = tet_idx[reorder, 0].copy()
        tet_idx[reorder, 0] = tet_idx[reorder, 1]
        tet_idx[reorder, 1] = tmp
    final_sign = np.sign(_tet_signed_triples(points, tet_idx))
    mismatch = (ref_sign != 0.0) & (final_sign != 0.0) & (ref_sign != final_sign)
    return tet_idx, int(np.count_nonzero(reorder)), int(np.count_nonzero(mismatch))


def _apply_gmsh_initial_contact_radius_geometry(
    *,
    config: VolumetricPitoisConfig,
    HC,
    vertices: list,
    points: np.ndarray,
    tets: np.ndarray,
    outer_rings: list,
    cap_bottom: list,
    cap_top: list,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
    loaded_contact_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Move the loaded Gmsh/ddgclib vertices when the initial contact radius changes.

    The default mode leaves the hard-coded .msh untouched.  In ``pitois_eq6``
    mode, the contact rings and solid caps are moved to the Eq. [6] wetted
    radius while preserving the original sphere centers/gap and raw tetra mesh.
    """
    mode = str(getattr(config, "initial_contact_radius_mode", USER_INITIAL_CONTACT_RADIUS_MODE)).strip().lower()
    pts0 = np.asarray(points, dtype=float)
    tet_idx0 = np.asarray(tets, dtype=int).reshape((-1, 4))
    metadata: dict[str, float | int | str] = {
        "initial_contact_radius_mode": mode,
        "initial_geometry_contact_radius": float(loaded_contact_radius),
        "initial_geometry_target_volume_m3": float(_indexed_tet_mesh_volume_m3(pts0, tet_idx0)),
        "initial_geometry_actual_volume_m3": float(_indexed_tet_mesh_volume_m3(pts0, tet_idx0)),
        "initial_geometry_volume_rel_error": 0.0,
        "initial_geometry_profile_exponent": 0.0,
        "initial_geometry_radial_blend_power": 0.0,
        "initial_geometry_bulged_neck_radius": 0.0,
        "initial_geometry_bulge_amplitude": 0.0,
        "initial_geometry_contact_slope_scale": 0.0,
        "initial_geometry_orientation_reordered_tets": 0,
        "initial_geometry_orientation_mismatch_tets": 0,
    }
    if mode == "loaded":
        return pts0, tet_idx0, metadata
    if mode not in {"pitois_eq6", "pitois_eq6_bulged"}:
        raise ValueError("initial_contact_radius_mode must be loaded, pitois_eq6, or pitois_eq6_bulged.")

    target_radius = float(config.target_cap_radius)
    loaded_radius = float(loaded_contact_radius)
    particle_radius = float(config.particle_radius)
    target_volume = float(config.initial_bridge_volume_m3)
    if target_radius <= 0.0 or loaded_radius <= 0.0 or particle_radius <= 0.0 or target_volume <= 0.0:
        raise ValueError("Eq. [6] initial geometry needs positive radius, particle radius, and volume.")
    if target_radius >= particle_radius:
        raise ValueError("Eq. [6] initial contact radius must be smaller than the particle radius.")

    bottom = np.asarray(bottom_sphere_center, dtype=float)
    top = np.asarray(top_sphere_center, dtype=float)
    axis_vec = top - bottom
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm <= 1.0e-30:
        raise ValueError("Cannot apply Eq. [6] initial geometry with coincident sphere centers.")
    axis = axis_vec / axis_norm
    mid = 0.5 * (bottom + top)

    rel0 = pts0 - mid[None, :]
    s0 = np.einsum("ij,j->i", rel0, axis)
    radial_vec0 = rel0 - np.outer(s0, axis)
    r0 = np.linalg.norm(radial_vec0, axis=1)
    radial_unit0 = np.zeros_like(radial_vec0)
    radial_mask = r0 > 1.0e-30
    radial_unit0[radial_mask] = radial_vec0[radial_mask] / r0[radial_mask, None]

    old_halfspan = max(0.5 * (float(np.max(s0)) - float(np.min(s0))), 1.0e-30)
    gap = max(axis_norm - 2.0 * particle_radius, 0.0)
    new_halfspan = 0.5 * gap + particle_radius - math.sqrt(max(particle_radius**2 - target_radius**2, 0.0))
    if new_halfspan <= 0.0:
        raise ValueError("Eq. [6] initial geometry produced a non-positive contact-plane half-span.")

    outer_s = np.asarray(
        [float(np.mean([float(np.dot(np.asarray(v.x_a[:3], dtype=float) - mid, axis)) for v in ring])) for ring in outer_rings],
        dtype=float,
    )
    outer_r = np.asarray([float(_cap_radius(ring)) for ring in outer_rings], dtype=float)
    order = np.argsort(outer_s)
    outer_s = outer_s[order]
    outer_r = outer_r[order]
    radial_scale = target_radius / max(loaded_radius, 1.0e-30)
    blend_power = max(float(USER_INITIAL_GEOMETRY_RADIAL_BLEND_POWER), 1.0)

    vertex_index = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    cap_bottom_idx = np.asarray([vertex_index[id(vertex)] for vertex in cap_bottom if id(vertex) in vertex_index], dtype=int)
    cap_top_idx = np.asarray([vertex_index[id(vertex)] for vertex in cap_top if id(vertex) in vertex_index], dtype=int)

    def project_caps_to_target_radius(pts: np.ndarray) -> None:
        for cap_idx, center, axial_sign in (
            (cap_bottom_idx, bottom, 1.0),
            (cap_top_idx, top, -1.0),
        ):
            if cap_idx.size == 0:
                continue
            local = pts0[cap_idx] - center[None, :]
            local_s = local @ axis
            local_radial = local - np.outer(local_s, axis)
            local_r = np.linalg.norm(local_radial, axis=1)
            local_unit = np.zeros_like(local_radial)
            local_mask = local_r > 1.0e-30
            local_unit[local_mask] = local_radial[local_mask] / local_r[local_mask, None]
            cap_r = np.minimum(radial_scale * local_r, target_radius)
            cap_s = axial_sign * np.sqrt(np.maximum(particle_radius**2 - cap_r**2, 0.0))
            pts[cap_idx] = center[None, :] + np.outer(cap_s, axis) + local_unit * cap_r[:, None]

    if mode == "pitois_eq6_bulged":
        neck_radius = float(getattr(config, "initial_bulged_neck_radius", USER_INITIAL_BULGED_NECK_RADIUS_M))
        if (not np.isfinite(neck_radius)) or neck_radius <= 0.0:
            neck_radius = 0.90 * target_radius
        neck_radius = float(np.clip(neck_radius, 0.05 * target_radius, 0.99 * target_radius))
        slope_scale = float(
            getattr(config, "initial_bulged_contact_slope_scale", USER_INITIAL_BULGED_CONTACT_SLOPE_SCALE)
        )
        sphere_slope = math.sqrt(max(particle_radius**2 - target_radius**2, 0.0)) / max(target_radius, 1.0e-30)
        contact_slope_h = sphere_slope * slope_scale * new_halfspan
        poly_a = 2.0 * (target_radius - neck_radius) - 0.5 * contact_slope_h
        poly_c = 0.5 * contact_slope_h - (target_radius - neck_radius)

        def transformed_bulged_points(bulge_amplitude: float) -> np.ndarray:
            new_s = s0 / old_halfspan * new_halfspan
            x = np.clip(np.abs(new_s) / max(new_halfspan, 1.0e-30), 0.0, 1.0)
            profile = (
                neck_radius
                + poly_a * x * x
                + poly_c * x**4
                + float(bulge_amplitude) * (x * x) * np.square(1.0 - x * x)
            )
            old_outer = np.interp(s0, outer_s, outer_r)
            q = np.clip(r0 / np.maximum(old_outer, 1.0e-30), 0.0, 1.0)
            pts = mid[None, :] + np.outer(new_s, axis) + radial_unit0 * (q * np.maximum(profile, 1.0e-10))[:, None]
            project_caps_to_target_radius(pts)
            return pts

        lo_bulge = 0.0
        hi_bulge = max(0.25 * target_radius, 1.0e-6)
        vol_lo = _indexed_tet_mesh_volume_m3(transformed_bulged_points(lo_bulge), tet_idx0)
        vol_hi = _indexed_tet_mesh_volume_m3(transformed_bulged_points(hi_bulge), tet_idx0)
        for _ in range(12):
            if vol_lo <= target_volume <= vol_hi:
                break
            hi_bulge *= 2.0
            vol_hi = _indexed_tet_mesh_volume_m3(transformed_bulged_points(hi_bulge), tet_idx0)
        if not (vol_lo <= target_volume <= vol_hi):
            raise RuntimeError(
                "Eq. [6] bulged initial geometry could not bracket the requested bridge volume "
                f"(target {target_volume:.6e}, bracket {vol_lo:.6e}..{vol_hi:.6e})."
            )
        for _ in range(70):
            mid_bulge = 0.5 * (lo_bulge + hi_bulge)
            mid_volume = _indexed_tet_mesh_volume_m3(transformed_bulged_points(mid_bulge), tet_idx0)
            if mid_volume < target_volume:
                lo_bulge = mid_bulge
            else:
                hi_bulge = mid_bulge
        bulge = 0.5 * (lo_bulge + hi_bulge)
        warped_points = transformed_bulged_points(bulge)
        actual_volume = _indexed_tet_mesh_volume_m3(warped_points, tet_idx0)
        oriented_tets, reordered_tets, mismatch_tets = _orient_tets_like_reference(
            reference_points=pts0,
            points=warped_points,
            tets=tet_idx0,
        )
        _move_vertices_batch(vertices, warped_points, HC, set())
        metadata.update(
            {
                "initial_geometry_contact_radius": float(target_radius),
                "initial_geometry_target_volume_m3": float(target_volume),
                "initial_geometry_actual_volume_m3": float(actual_volume),
                "initial_geometry_volume_rel_error": float(
                    (actual_volume - target_volume) / max(target_volume, 1.0e-30)
                ),
                "initial_geometry_bulged_neck_radius": float(neck_radius),
                "initial_geometry_bulge_amplitude": float(bulge),
                "initial_geometry_contact_slope_scale": float(slope_scale),
                "initial_geometry_orientation_reordered_tets": int(reordered_tets),
                "initial_geometry_orientation_mismatch_tets": int(mismatch_tets),
            }
        )
        return warped_points, oriented_tets, metadata

    def transformed_points(profile_exponent: float) -> np.ndarray:
        old_outer = np.interp(s0, outer_s, outer_r)
        q = np.clip(r0 / np.maximum(old_outer, 1.0e-30), 0.0, 1.0)
        outer_ratio = np.clip(old_outer / max(loaded_radius, 1.0e-30), 1.0e-12, 1.0)
        new_outer = target_radius * np.power(outer_ratio, float(profile_exponent))
        new_s = s0 / old_halfspan * new_halfspan
        new_r = radial_scale * r0 + (new_outer - radial_scale * old_outer) * np.power(q, blend_power)
        new_r = np.maximum(new_r, 0.0)
        pts = mid[None, :] + np.outer(new_s, axis) + radial_unit0 * new_r[:, None]

        project_caps_to_target_radius(pts)
        return pts

    lo = -2.0
    hi = 2.0
    vol_lo = _indexed_tet_mesh_volume_m3(transformed_points(lo), tet_idx0)
    vol_hi = _indexed_tet_mesh_volume_m3(transformed_points(hi), tet_idx0)
    for _ in range(8):
        if min(vol_lo, vol_hi) <= target_volume <= max(vol_lo, vol_hi):
            break
        if vol_lo < target_volume:
            lo *= 2.0
            vol_lo = _indexed_tet_mesh_volume_m3(transformed_points(lo), tet_idx0)
        if vol_hi > target_volume:
            hi *= 2.0
            vol_hi = _indexed_tet_mesh_volume_m3(transformed_points(hi), tet_idx0)
    if not (min(vol_lo, vol_hi) <= target_volume <= max(vol_lo, vol_hi)):
        raise RuntimeError(
            "Eq. [6] initial geometry could not bracket the requested bridge volume "
            f"(target {target_volume:.6e}, bracket {vol_lo:.6e}..{vol_hi:.6e})."
        )

    for _ in range(80):
        mid_exp = 0.5 * (lo + hi)
        mid_volume = _indexed_tet_mesh_volume_m3(transformed_points(mid_exp), tet_idx0)
        if mid_volume > target_volume:
            lo = mid_exp
        else:
            hi = mid_exp
    exponent = 0.5 * (lo + hi)
    warped_points = transformed_points(exponent)
    actual_volume = _indexed_tet_mesh_volume_m3(warped_points, tet_idx0)
    oriented_tets, reordered_tets, mismatch_tets = _orient_tets_like_reference(
        reference_points=pts0,
        points=warped_points,
        tets=tet_idx0,
    )

    _move_vertices_batch(vertices, warped_points, HC, set())
    metadata.update(
        {
            "initial_geometry_contact_radius": float(target_radius),
            "initial_geometry_target_volume_m3": float(target_volume),
            "initial_geometry_actual_volume_m3": float(actual_volume),
            "initial_geometry_volume_rel_error": float(
                (actual_volume - target_volume) / max(target_volume, 1.0e-30)
            ),
            "initial_geometry_profile_exponent": float(exponent),
            "initial_geometry_radial_blend_power": float(blend_power),
            "initial_geometry_orientation_reordered_tets": int(reordered_tets),
            "initial_geometry_orientation_mismatch_tets": int(mismatch_tets),
        }
    )
    return warped_points, oriented_tets, metadata


def _prepare_gmsh_state(config: VolumetricPitoisConfig) -> VolumetricPitoisState:
    initial_msh_path = _required_initial_msh_path(config)
    points_m, triangles, tets, _node_tags = _read_gmsh4_points_triangles_tets(initial_msh_path)
    points = np.asarray(_length_to_compute(config, points_m), dtype=float)
    HC = Complex(3, domain=None)
    vertices = [HC.V[tuple(float(x) for x in point)] for point in points]

    for tet in np.asarray(tets, dtype=int):
        tet_vertices = [vertices[int(i)] for i in tet]
        for a in range(4):
            for b in range(a + 1, 4):
                tet_vertices[a].connect(tet_vertices[b])
    for tri in np.asarray(triangles, dtype=int):
        tri_vertices = [vertices[int(i)] for i in tri]
        for a in range(3):
            tri_vertices[a].connect(tri_vertices[(a + 1) % 3])
    HC.gmsh_volume_vertices = list(vertices)
    HC.gmsh_volume_tets = np.asarray(tets, dtype=int)

    outer_rings = _gmsh_free_surface_rings(vertices, points)
    layer_rings = [[ring] for ring in outer_rings]
    layer_centers = [min(ring, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) for ring in outer_rings]
    z_bottom = float(np.mean([float(v.x_a[2]) for v in outer_rings[0]]))
    z_top = float(np.mean([float(v.x_a[2]) for v in outer_rings[-1]]))
    contact_radius = 0.5 * (_cap_radius(outer_rings[0]) + _cap_radius(outer_rings[-1]))
    axial_offset = float(np.sqrt(max(float(config.particle_radius) ** 2 - contact_radius**2, 0.0)))
    bottom_sphere_center = np.array([0.0, 0.0, z_bottom - axial_offset], dtype=float)
    top_sphere_center = np.array([0.0, 0.0, z_top + axial_offset], dtype=float)

    dist_bottom = np.abs(np.linalg.norm(points - bottom_sphere_center[None, :], axis=1) - float(config.particle_radius))
    dist_top = np.abs(np.linalg.norm(points - top_sphere_center[None, :], axis=1) - float(config.particle_radius))
    sphere_tol = max(float(_length_to_compute(config, 2.0e-8)), 2.0e-5 * float(config.particle_radius))
    cap_bottom = [vertices[int(i)] for i in np.flatnonzero((points[:, 2] <= sphere_tol) & (dist_bottom <= sphere_tol))]
    cap_top = [vertices[int(i)] for i in np.flatnonzero((points[:, 2] >= -sphere_tol) & (dist_top <= sphere_tol))]
    cap_bottom_center = min(cap_bottom, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) if cap_bottom else outer_rings[0][0]
    cap_top_center = min(cap_top, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) if cap_top else outer_rings[-1][0]

    side_tris, bottom_cap_tris, top_cap_tris = _split_gmsh_boundary_triangles(
        points,
        triangles,
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
        radius=float(config.particle_radius),
    )

    interface_ids = {id(v) for ring in outer_rings for v in ring}
    cap_ids = {id(v) for v in cap_bottom + cap_top}
    for v in vertices:
        v.cap_id = "bottom" if v in cap_bottom else ("top" if v in cap_top else None)
        v.phase = 0
        v.p = 0.0
        v.u = np.zeros(3, dtype=float)
        v.boundary = bool(id(v) in interface_ids or id(v) in cap_ids)
        v.is_interface = bool(id(v) in interface_ids)
        v.interface_phases = frozenset({0, 1}) if v.is_interface else frozenset()

    loaded_contact_radius = float(contact_radius)
    points, tets, initial_geometry_meta = _apply_gmsh_initial_contact_radius_geometry(
        config=config,
        HC=HC,
        vertices=vertices,
        points=points,
        tets=np.asarray(tets, dtype=int),
        outer_rings=outer_rings,
        cap_bottom=cap_bottom,
        cap_top=cap_top,
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
        loaded_contact_radius=loaded_contact_radius,
    )
    HC.gmsh_volume_tets = np.asarray(tets, dtype=int)

    layer_z = np.asarray([np.mean([float(v.x_a[2]) for v in ring]) for ring in outer_rings], dtype=float)
    z_span = max(float(layer_z[-1] - layer_z[0]), 1.0e-30)
    layer_fractions = tuple(float((z - layer_z[0]) / z_span) for z in layer_z)
    mesh_volume_m3 = _indexed_tet_mesh_volume_m3(points, tets)
    mps = SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: config.gamma,
        get_mu=lambda phase: config.mu_f,
    )

    state = VolumetricPitoisState(
        config=config,
        HC=HC,
        bV_caps=set(),
        cap_bottom=cap_bottom,
        cap_top=cap_top,
        cap_bottom_interior=[],
        cap_top_interior=[],
        bottom_contact_ring=[],
        top_contact_ring=[],
        cap_bottom_center=cap_bottom_center,
        cap_top_center=cap_top_center,
        layer_centers=layer_centers,
        layer_fractions=layer_fractions,
        outer_rings=outer_rings,
        layer_rings=layer_rings,
        surface_edges=[],
        surface_boundary_indices=set(),
        mps=mps,
        radial_scale=1.0,
        axial_scale=1.0,
        initial_gap=float(np.linalg.norm(top_sphere_center - bottom_sphere_center) - 2.0 * float(config.particle_radius)),
        cap_ring_factors=(1.0,),
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
        target_volume_m3=float(mesh_volume_m3),
        target_snapshot_volume_m3=float(mesh_volume_m3),
    )
    state.gmsh_compute_mesh = True
    state.initial_contact_radius_mode = str(initial_geometry_meta.get("initial_contact_radius_mode", "loaded"))
    state.initial_geometry_contact_radius = float(
        initial_geometry_meta.get("initial_geometry_contact_radius", loaded_contact_radius)
    )
    state.initial_geometry_target_volume_m3 = float(
        initial_geometry_meta.get("initial_geometry_target_volume_m3", mesh_volume_m3)
    )
    state.initial_geometry_actual_volume_m3 = float(
        initial_geometry_meta.get("initial_geometry_actual_volume_m3", mesh_volume_m3)
    )
    state.initial_geometry_volume_rel_error = float(
        initial_geometry_meta.get("initial_geometry_volume_rel_error", 0.0)
    )
    state.initial_geometry_profile_exponent = float(
        initial_geometry_meta.get("initial_geometry_profile_exponent", 0.0)
    )
    state.initial_geometry_radial_blend_power = float(
        initial_geometry_meta.get("initial_geometry_radial_blend_power", 0.0)
    )
    state.initial_geometry_bulged_neck_radius = float(
        initial_geometry_meta.get("initial_geometry_bulged_neck_radius", 0.0)
    )
    state.initial_geometry_bulge_amplitude = float(
        initial_geometry_meta.get("initial_geometry_bulge_amplitude", 0.0)
    )
    state.initial_geometry_contact_slope_scale = float(
        initial_geometry_meta.get("initial_geometry_contact_slope_scale", 0.0)
    )
    state.initial_geometry_orientation_reordered_tets = int(
        initial_geometry_meta.get("initial_geometry_orientation_reordered_tets", 0)
    )
    state.initial_geometry_orientation_mismatch_tets = int(
        initial_geometry_meta.get("initial_geometry_orientation_mismatch_tets", 0)
    )
    state.loaded_initial_contact_radius = float(loaded_contact_radius)
    state.pitois_eq6_contact_radius = float(config.target_cap_radius)
    state.last_step_dt = float(config.dt)
    state.last_dt_limit_cl = float(config.dt)
    state.last_dt_limit_capillary = float(config.dt)
    state.last_dt_limit_mesh = float(config.dt)
    state.gmsh_axisym_reference = np.array([1.0, 0.0, 0.0], dtype=float)
    state.gmsh_axisym_ring_specs = _build_gmsh_axisym_ring_specs(
        vertices,
        points,
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
    )
    state.loaded_initial_msh_path = str(initial_msh_path)
    state.volume_export_vertices = list(vertices)
    state.volume_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.volume_export_tets = np.asarray(tets, dtype=int)
    state.surface_export_vertices = list(vertices)
    state.surface_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.surface_export_side_tris = np.asarray(side_tris, dtype=int)
    state.surface_export_bottom_cap_tris = np.asarray(bottom_cap_tris, dtype=int)
    state.surface_export_top_cap_tris = np.asarray(top_cap_tris, dtype=int)
    _valid_ref, _ref_tet_volumes, _ref_grads = _gmsh_tet_metric_arrays(points, np.asarray(tets, dtype=int))
    state.ddgclib_reference_tet_volumes = np.where(_valid_ref, _ref_tet_volumes, 0.0)
    _refresh_cap_boundary_sets(state)
    state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
    state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
    state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
    state.topology_frozen = True
    _set_cap_velocities(state)
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    return state


def _isotonic_increasing_fit(values: np.ndarray) -> np.ndarray:
    vals = [float(v) for v in np.asarray(values, dtype=float)]
    if not vals:
        return np.empty(0, dtype=float)

    blocks: list[list[float]] = []
    for value in vals:
        blocks.append([value, 1.0, 1.0])  # mean, weight, count
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            mean0, weight0, count0 = blocks[-2]
            mean1, weight1, count1 = blocks[-1]
            weight = weight0 + weight1
            mean = (mean0 * weight0 + mean1 * weight1) / weight
            blocks[-2] = [mean, weight, count0 + count1]
            blocks.pop()

    fitted: list[float] = []
    for mean, _weight, count in blocks:
        fitted.extend([float(mean)] * int(count))
    return np.asarray(fitted, dtype=float)


def _single_neck_projected_radii(radii: np.ndarray) -> np.ndarray:
    rs = np.asarray(radii, dtype=float)
    if rs.size <= 2:
        return rs.copy()

    neck_idx = int(np.argmin(rs))
    left = rs[: neck_idx + 1]
    right = rs[neck_idx:]

    left_fit = _isotonic_increasing_fit(left[::-1])[::-1]
    right_fit = _isotonic_increasing_fit(right)
    neck_radius = min(float(left_fit[-1]), float(right_fit[0]))
    left_fit[-1] = neck_radius
    right_fit[0] = neck_radius

    return np.concatenate([left_fit[:-1], right_fit])


def _scale_free_vertices_radially(
    state: VolumetricPitoisState,
    *,
    radial_scale: float,
) -> None:
    if abs(radial_scale - 1.0) <= 1.0e-12:
        return

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)

    for v in state.HC.V:
        if v in state.bV_caps:
            continue
        pos = np.asarray(v.x_a[:3], dtype=float)
        rel = pos - mid
        axial = axis * float(np.dot(rel, axis))
        radial = rel - axial
        target = mid + axial + radial_scale * radial
        _move(v, tuple(target), state.HC, state.bV_caps)


def _free_vertex_positions(state: VolumetricPitoisState) -> tuple[list, np.ndarray]:
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    if not free_vertices:
        return [], np.empty((0, 3), dtype=float)
    positions = np.array([np.asarray(v.x_a[:3], dtype=float) for v in free_vertices], dtype=float)
    return free_vertices, positions


def _snapshot_projectable_vertex_positions(state: VolumetricPitoisState) -> tuple[list, np.ndarray]:
    projectable_vertices = [
        v
        for v in state.HC.V
        if (v not in state.bV_caps) and (getattr(v, "cap_id", None) is None)
    ]
    if not projectable_vertices:
        return [], np.empty((0, 3), dtype=float)
    positions = np.array([np.asarray(v.x_a[:3], dtype=float) for v in projectable_vertices], dtype=float)
    return projectable_vertices, positions


def _radially_scaled_positions(
    state: VolumetricPitoisState,
    positions: np.ndarray,
    *,
    radial_scale: float,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.empty((0, 3), dtype=float)
    max_coord = max(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
        2.0 * float(state.config.particle_radius),
        1.0,
    )
    if (not np.all(np.isfinite(positions))) or float(np.max(np.abs(positions))) > max_coord:
        raise ValueError("rejecting non-physical trial coordinates during radial volume scaling")

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)
    if (not np.all(np.isfinite(mid))) or float(np.max(np.abs(mid))) > max_coord:
        raise ValueError("rejecting non-physical sphere centers during radial volume scaling")

    rel = positions - mid[None, :]
    axial_mag = np.sum(rel * axis[None, :], axis=1)
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    return mid[None, :] + axial + float(radial_scale) * radial


def _axisym_water_volume_scaled_positions(
    state: VolumetricPitoisState,
    positions: np.ndarray,
    *,
    meridional_scale: float,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.empty((0, 3), dtype=float)
    max_coord = max(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
        2.0 * float(state.config.particle_radius),
        1.0,
    )
    if (not np.all(np.isfinite(positions))) or float(np.max(np.abs(positions))) > max_coord:
        raise ValueError("rejecting non-physical trial coordinates during axisymmetric volume scaling")

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)
    if (not np.all(np.isfinite(mid))) or float(np.max(np.abs(mid))) > max_coord:
        raise ValueError("rejecting non-physical sphere centers during axisymmetric volume scaling")

    rel = positions - mid[None, :]
    axial_mag = np.sum(rel * axis[None, :], axis=1)
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    lam = max(float(meridional_scale), 1.0e-6)
    # In the axisymmetric bridge, shrinking the meridian should compress axial
    # extent more strongly than radial extent to reduce the water-filled tet sum
    # without moving the sphere-caps directly.
    return mid[None, :] + (lam * lam) * axial + lam * radial


def _axisym_force_snapshot_volume_to_target(
    state: VolumetricPitoisState,
    *,
    rel_tol: float | None = None,
    max_iters: int | None = None,
) -> float:
    target = float(state.target_snapshot_volume_m3)
    if not np.isfinite(target) or target <= 0.0:
        return float(_snapshot_msh_volume_m3(state))

    if rel_tol is None:
        rel_tol = float(state.config.volume_projection_rel_tol)
    if max_iters is None:
        max_iters = max(16, int(state.config.volume_projection_max_iters))

    current = float(_snapshot_msh_volume_m3(state))
    if not np.isfinite(current) or current <= 0.0:
        return current
    if abs(current - target) / max(target, 1.0e-30) <= float(rel_tol):
        return current

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return current

    def _restore_base() -> None:
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)

    def _volume_at(scale: float) -> float:
        trial_positions = _axisym_water_volume_scaled_positions(
            state,
            base_positions,
            meridional_scale=float(scale),
        )
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _restore_base()
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.05
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.10
            vol_high = _volume_at(high)
        if vol_high < target:
            return current
    else:
        low = 0.95
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.02:
            low *= 0.90
            vol_low = _volume_at(low)
        if vol_low > target:
            return current
        high = 1.0

    final_scale = 1.0
    for _ in range(max(1, int(max_iters))):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        final_scale = mid
        if abs(vol_mid - target) / max(target, 1.0e-30) <= float(rel_tol):
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_positions = _axisym_water_volume_scaled_positions(
        state,
        base_positions,
        meridional_scale=float(final_scale),
    )
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)
    return float(_snapshot_msh_volume_m3(state))


def _apply_free_vertex_positions(
    state: VolumetricPitoisState,
    free_vertices: list,
    positions: np.ndarray,
) -> None:
    if not free_vertices:
        return
    _move_vertices_batch(
        free_vertices,
        [tuple(map(float, row)) for row in np.asarray(positions, dtype=float)],
        state.HC,
        state.bV_caps,
    )


def _project_volume_to_target(state: VolumetricPitoisState) -> None:
    if not state.config.enable_volume_projection:
        return

    target = float(state.target_snapshot_volume_m3)
    if not np.isfinite(target) or target <= 0.0:
        return

    current = _snapshot_msh_volume_m3(state)
    if not np.isfinite(current) or current <= 0.0:
        return
    rel_err = abs(current - target) / max(target, 1.0e-30)
    if rel_err <= float(state.config.volume_projection_rel_tol):
        return

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return

    def _volume_at(scale: float) -> float:
        trial_positions = _radially_scaled_positions(state, base_positions, radial_scale=scale)
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.10
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.25
            vol_high = _volume_at(high)
        if vol_high < target:
            return
    else:
        low = 0.80
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.05:
            low *= 0.75
            vol_low = _volume_at(low)
        if vol_low > target:
            return
        high = 1.0

    n_bisect = max(1, int(state.config.volume_projection_max_iters))
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        if abs(vol_mid - target) / max(target, 1.0e-30) <= float(state.config.volume_projection_rel_tol):
            low = mid
            high = mid
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_scale = 0.5 * (low + high)
    final_positions = _radially_scaled_positions(state, base_positions, radial_scale=final_scale)
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)


def _force_snapshot_volume_to_target(
    state: VolumetricPitoisState,
    *,
    target_m3: float,
    rel_tol: float | None = None,
    max_iters: int | None = None,
) -> None:
    target = float(target_m3)
    if not np.isfinite(target) or target <= 0.0:
        return

    current = _snapshot_msh_volume_m3(state)
    if not np.isfinite(current) or current <= 0.0:
        return

    tol = float(state.config.volume_projection_rel_tol if rel_tol is None else rel_tol)
    if abs(current - target) / max(target, 1.0e-30) <= tol:
        return

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return

    def _volume_at(scale: float) -> float:
        trial_positions = _radially_scaled_positions(state, base_positions, radial_scale=scale)
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.10
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.25
            vol_high = _volume_at(high)
        if vol_high < target:
            return
    else:
        low = 0.80
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.05:
            low *= 0.75
            vol_low = _volume_at(low)
        if vol_low > target:
            return
        high = 1.0

    n_bisect = max(1, int(state.config.volume_projection_max_iters if max_iters is None else max_iters))
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        if abs(vol_mid - target) / max(target, 1.0e-30) <= tol:
            low = mid
            high = mid
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_scale = 0.5 * (low + high)
    final_positions = _radially_scaled_positions(state, base_positions, radial_scale=final_scale)
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)


def _axisymmetrize_layer_rings(state: VolumetricPitoisState) -> None:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)

    ref_ring = state.outer_rings[0]
    ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
    reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    n_layers = len(state.layer_rings)
    for k, rings in enumerate(state.layer_rings):
        if k in (0, n_layers - 1):
            continue

        outer_ring = state.outer_rings[k]
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_ring], axis=0)
        s = float(np.dot(outer_center - mid, axis))
        center_coord = mid + s * axis
        center_vertex = state.layer_centers[k]
        _move(center_vertex, tuple(center_coord), state.HC, state.bV_caps)

        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - center_coord[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            radius = float(np.mean(np.linalg.norm(rel, axis=1)))
            n_ring = max(1, len(ring))
            targets = []
            for i, v in enumerate(ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _relax_initial_equilibrium(state: VolumetricPitoisState) -> None:
    if not state.config.enable_initial_relaxation or int(state.config.initial_relax_steps) <= 0:
        return

    original_config = state.config
    relax_config = replace(
        original_config,
        cap_speed=0.0,
        dt=float(original_config.initial_relax_dt),
        damping=float(original_config.initial_relax_damping),
    )
    state.config = relax_config
    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)

    for _step in range(int(original_config.initial_relax_steps)):
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        symplectic_euler(
            state.HC,
            state.bV_caps,
            _vertex_acceleration,
            dt=float(relax_config.dt),
            n_steps=1,
            dim=3,
            retopologize_fn=False,
            state=state,
        )
        _axisymmetrize_layer_rings(state)
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        if _max_free_speed(state) <= float(original_config.initial_relax_tol_speed):
            break

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)
    state.config = original_config


def _rebuild_cap_on_sphere(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    contact_radius: float,
    axis: np.ndarray,
    dt: float | None,
) -> None:
    layer_idx = 0 if which == "bottom" else -1
    sign = +1.0 if which == "bottom" else -1.0
    center_vertex = state.cap_bottom_center if which == "bottom" else state.cap_top_center
    cap_rings = state.layer_rings[layer_idx]
    reference_ring = cap_rings[-1]
    ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in reference_ring], axis=0)
    reference = np.asarray(reference_ring[0].x_a[:3], dtype=float) - ring_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    radius = float(state.config.particle_radius)

    old_center = np.asarray(center_vertex.x_a[:3], dtype=float)
    new_center = sphere_center + sign * radius * axis
    _move(center_vertex, tuple(new_center), state.HC, state.bV_caps)
    if dt is not None and dt > 0.0:
        center_vertex.u = (new_center - old_center) / dt
    else:
        center_vertex.u = np.zeros(3, dtype=float)

    for factor, ring in zip(state.cap_ring_factors, cap_rings):
        ring_radius = min(
            max(float(factor) * contact_radius, 0.0),
            radius - float(_length_to_compute(state.config, 1.0e-6)),
        )
        axial = np.sqrt(max(radius**2 - ring_radius**2, 0.0))
        targets = []
        for i, v in enumerate(ring):
            phi = 2.0 * np.pi * float(i) / float(len(ring))
            tangential = ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
            target = sphere_center + tangential + sign * axial * axis
            targets.append(tuple(target))
        for v, target in zip(ring, targets):
            old = np.asarray(v.x_a[:3], dtype=float)
            _move(v, target, state.HC, state.bV_caps)
            if dt is not None and dt > 0.0:
                v.u = (target - old) / dt
            else:
                v.u = np.zeros(3, dtype=float)
    _refresh_cap_boundary_sets(state)


def _update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    prev_bottom_radius = float(_cap_radius(state.outer_rings[0]))
    prev_top_radius = float(_cap_radius(state.outer_rings[-1]))
    bottom_ring = state.bottom_contact_ring
    top_ring = state.top_contact_ring
    bottom_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in bottom_ring], dtype=float)
    top_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in top_ring], dtype=float)
    bottom_rel = bottom_coords - state.bottom_sphere_center[None, :]
    top_rel = top_coords - state.top_sphere_center[None, :]
    bottom_tangential = bottom_rel - np.outer(np.dot(bottom_rel, axis), axis)
    top_tangential = top_rel - np.outer(np.dot(top_rel, axis), axis)
    bottom_radius = float(np.mean(np.linalg.norm(bottom_tangential, axis=1)))
    top_radius = float(np.mean(np.linalg.norm(top_tangential, axis=1)))

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=bottom_radius,
        axis=axis,
        dt=dt,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=top_radius,
        axis=axis,
        dt=dt,
    )
    _align_layer_azimuths(state)
    _regularize_layer_order(state)
    if dt is not None and dt > 0.0:
        sphere_radius = max(float(state.config.particle_radius), 1.0e-30)
        alpha_bottom_prev = float(np.arcsin(np.clip(prev_bottom_radius / sphere_radius, -1.0, 1.0)))
        alpha_bottom_new = float(np.arcsin(np.clip(bottom_radius / sphere_radius, -1.0, 1.0)))
        alpha_top_prev = float(np.arcsin(np.clip(prev_top_radius / sphere_radius, -1.0, 1.0)))
        alpha_top_new = float(np.arcsin(np.clip(top_radius / sphere_radius, -1.0, 1.0)))
        state.last_bottom_contact_line_speed = abs(sphere_radius * (alpha_bottom_new - alpha_bottom_prev) / float(dt))
        state.last_top_contact_line_speed = abs(sphere_radius * (alpha_top_new - alpha_top_prev) / float(dt))
    else:
        state.last_bottom_contact_line_speed = 0.0
        state.last_top_contact_line_speed = 0.0


def _move_caps(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if dt is None:
        dt = float(state.config.dt)
    bottom_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float)
    top_vel = np.zeros(3, dtype=float)
    bottom_disp = dt * bottom_vel
    top_disp = dt * top_vel
    if float(np.linalg.norm(bottom_disp)) <= 1.0e-30 and float(np.linalg.norm(top_disp)) <= 1.0e-30:
        return
    state.bottom_sphere_center = np.asarray(state.bottom_sphere_center, dtype=float) + bottom_disp
    state.top_sphere_center = np.asarray(state.top_sphere_center, dtype=float) + top_disp
    if float(np.linalg.norm(bottom_disp)) > 1.0e-30:
        for v in state.cap_bottom_interior:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = bottom_vel
        for v in state.bottom_contact_ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
    if float(np.linalg.norm(top_disp)) > 1.0e-30:
        for v in state.cap_top_interior:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = top_vel
        for v in state.top_contact_ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_projected_pressure: bool = True,
    include_contact_line: bool = True,
    include_damping: bool = True,
) -> np.ndarray:
    del pressure_model, include_projected_pressure

    # FHeron variant: default stress pressure is off for legacy runs.  When
    # #9/#10 enable PR33 pressure, the pressure part is applied in the
    # B_{t,i}=dV_t/dx_i projection step, not through the Cauchy tensor.
    solver_pressure_model = _zero_solver_pressure_model
    Ftot = _multiphase_stress_force_cached(
        v,
        dim=3,
        state=state,
        pressure_model=solver_pressure_model,
    )
    Ftot += _solver_hydrostatic_force(v, state=state)
    if include_contact_line:
        Fcl = _Fcl(v, state=state)
        Ftot += Fcl
    if include_damping and state.config.damping > 0.0:
        Ftot -= state.config.damping * np.asarray(v.u[:3], dtype=float)
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        Ftot = _remove_swirl_component(
            Ftot,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )
    return np.asarray(Ftot, dtype=float)


def _vertex_acceleration(v, *, state: VolumetricPitoisState) -> np.ndarray:
    Ftot = _Ftot(v, state=state)
    accel = Ftot / max(float(getattr(v, "m", 0.0)), 1.0e-12)
    accel = _clip_acceleration(accel, state)
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        accel = _remove_swirl_component(
            accel,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )
    return accel


def _axisym_clear_caches(state: VolumetricPitoisState) -> None:
    for attr in (
        "_dual_area_vector_cache",
        "_gmsh_dual_area_vector_cache",
        "_gmsh_dual_area_vector_edges",
        "_gmsh_stress_force_vector_cache",
        "_gmsh_tet_cauchy_viscous_force_cache",
        "_axisym_force_cache",
        "_axisym_accel_cache",
        "_gmsh_axisym_force_cache",
        "_gmsh_axisym_accel_cache",
        "_gmsh_surface_heron_force_cache",
        "_gmsh_surface_heron_area_cache",
        "_gmsh_solver_hydrostatic_force_cache",
        "_cox_uy_contact_line_force_cache",
        "_gmsh_wall_stress_tensor_cache",
        "_gmsh_tet_boundary_face_owner_cache",
    ):
        if hasattr(state, attr):
            delattr(state, attr)


def _axisym_force_workers(state: VolumetricPitoisState) -> int:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return 1
    return max(1, int(getattr(state.config, "accel_workers", 1)))


def _axisym_basis(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)
    ref_ring = state.outer_rings[0] if state.outer_rings else []
    if ref_ring:
        ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
        reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    return mid, axis, e1, e2


def _axisym_layer_center_coord(
    state: VolumetricPitoisState,
    *,
    layer_idx: int,
    mid: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    outer_ring = state.outer_rings[layer_idx]
    if outer_ring:
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_ring], axis=0)
        axial_s = float(np.dot(outer_center - mid, axis))
    else:
        axial_s = float(np.dot(np.asarray(state.layer_centers[layer_idx].x_a[:3], dtype=float) - mid, axis))
    return mid + axial_s * axis


def _axisym_ring_radius(ring: list, *, center_coord: np.ndarray, axis: np.ndarray) -> float:
    if not ring:
        return 0.0
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    rel = coords - center_coord[None, :]
    rel -= np.outer(np.dot(rel, axis), axis)
    return float(np.mean(np.linalg.norm(rel, axis=1)))


def _axisym_ring_radial_units(
    ring: list,
    *,
    center_coord: np.ndarray,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> list[np.ndarray]:
    units: list[np.ndarray] = []
    n_ring = max(1, len(ring))
    for i, v in enumerate(ring):
        rel = np.asarray(v.x_a[:3], dtype=float) - center_coord
        radial = rel - axis * float(np.dot(rel, axis))
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-30:
            phi = 2.0 * np.pi * float(i) / float(n_ring)
            units.append(np.cos(phi) * e1 + np.sin(phi) * e2)
        else:
            units.append(radial / radial_norm)
    return units


def _axisymmetrize_full_geometry(state: VolumetricPitoisState) -> None:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return
    mid, axis, e1, e2 = _axisym_basis(state)
    n_layers = len(state.layer_rings)
    if n_layers == 0:
        return

    bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else float(state.config.target_cap_radius)
    top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else float(state.config.target_cap_radius)
    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=bottom_radius,
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=top_radius,
        axis=axis,
        dt=None,
    )

    for k in range(1, max(1, n_layers - 1)):
        if k >= n_layers - 1:
            break
        rings = state.layer_rings[k]
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        _move(state.layer_centers[k], tuple(center_coord), state.HC, state.bV_caps)
        for ring in rings:
            radius = _axisym_ring_radius(ring, center_coord=center_coord, axis=axis)
            n_ring = max(1, len(ring))
            targets = []
            for i, _v in enumerate(ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)
    _refresh_cap_boundary_sets(state)


def _axisymmetrize_full_velocity_field(state: VolumetricPitoisState) -> None:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return
    mid, axis, e1, e2 = _axisym_basis(state)
    for k, rings in enumerate(state.layer_rings):
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        for ring in rings:
            if not ring:
                continue
            radial_units = _axisym_ring_radial_units(
                ring,
                center_coord=center_coord,
                axis=axis,
                e1=e1,
                e2=e2,
            )
            ur = float(
                np.mean(
                    [
                        float(np.dot(np.asarray(v.u[:3], dtype=float), e_r))
                        for v, e_r in zip(ring, radial_units)
                    ]
                )
            )
            uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in ring]))
            for v, e_r in zip(ring, radial_units):
                v.u = ur * e_r + uz * axis
        center_v = state.layer_centers[k]
        center_v.u = float(np.dot(np.asarray(center_v.u[:3], dtype=float), axis)) * axis


def _axisym_force_cache_key(
    state: VolumetricPitoisState,
    *,
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> tuple:
    return (
        bool(include_projected_pressure),
        bool(include_contact_line),
        bool(include_damping),
        str(getattr(state.config, "solver_viscous_force_model", "edge_flux")),
        round(float(getattr(state, "pressure_scalar", 0.0)), 18),
        round(float(getattr(state, "pressure_projection_scalar", 0.0)), 18),
        bool(getattr(state.config, "enable_solver_hydrostatic_force", False)),
        str(getattr(state.config, "solver_hydrostatic_zref_mode", "midpoint")),
        bool(getattr(state.config, "enable_solver_pr33_pressure_force", False)),
        bool(getattr(state.config, "solver_pr33_pressure_include_hydrostatic", False)),
    )


def _gmsh_axisym_basis(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)
    reference = np.asarray(getattr(state, "gmsh_axisym_reference", np.array([1.0, 0.0, 0.0])), dtype=float)
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    return mid, axis, e1, e2


def _gmsh_axisym_spec_radial_units(spec, e1: np.ndarray, e2: np.ndarray) -> list[np.ndarray]:
    return [
        float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
        for angle in getattr(spec, "angles", ())
    ]


def _gmsh_axisym_group_axial_radius(
    vertices: list,
    *,
    center_coord: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    if not vertices:
        return 0.0, 0.0
    coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    rel = coords - center_coord[None, :]
    axial = np.dot(rel, axis)
    radial = rel - np.outer(axial, axis)
    radial_norm = np.linalg.norm(radial, axis=1)
    finite = np.isfinite(axial) & np.isfinite(radial_norm)
    if not np.any(finite):
        return 0.0, 0.0
    # Median keeps one bad node from moving an entire axisymmetric ring.
    return float(np.median(axial[finite])), float(np.median(radial_norm[finite]))


def _gmsh_axisym_uniform_cap_id(vertices: list) -> str | None:
    cap_ids = {getattr(v, "cap_id", None) for v in vertices}
    if cap_ids == {"bottom"}:
        return "bottom"
    if cap_ids == {"top"}:
        return "top"
    return None


def _gmsh_axisymmetrize_geometry(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    specs = list(getattr(state, "gmsh_axisym_ring_specs", []))
    if not specs:
        return
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    radius = float(state.config.particle_radius)
    max_free_surface_radius = 1.05 * max(radius, 1.0e-30)
    for spec in specs:
        vertices = list(getattr(spec, "vertices", []))
        if not vertices:
            continue
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(vertices, center_coord=mid, axis=axis)
        cap_id = _gmsh_axisym_uniform_cap_id(vertices)
        if cap_id is None:
            coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
            rel = coords - mid[None, :]
            axial = np.dot(rel, axis)
            center_coord = mid + float(np.mean(axial)) * axis
        else:
            sphere_center = (
                np.asarray(state.bottom_sphere_center, dtype=float)
                if cap_id == "bottom"
                else np.asarray(state.top_sphere_center, dtype=float)
            )
            sign = 1.0 if cap_id == "bottom" else -1.0
            cap_r = float(np.clip(r_mean, 0.0, radius))
            axial_offset = float(np.sqrt(max(radius**2 - cap_r**2, 0.0)))
            center_coord = sphere_center + sign * axial_offset * axis
            r_mean = cap_r
        if cap_id is None:
            if (not np.isfinite(r_mean)) or r_mean < 0.0:
                r_mean = 0.0
            else:
                r_mean = float(np.clip(r_mean, 0.0, max_free_surface_radius))

        angles = tuple(getattr(spec, "angles", ()))
        if len(vertices) == 1 or r_mean <= 1.0e-30:
            targets = [tuple(center_coord) for _v in vertices]
        else:
            targets = [
                tuple(center_coord + r_mean * (float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2))
                for angle in angles
            ]
        _move_vertices_batch(vertices, targets, state.HC, state.bV_caps)
    _gmsh_enforce_outer_surface_contact_span(state)
    _axisym_clear_caches(state)


def _gmsh_enforce_outer_surface_contact_span(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    outer_rings = list(getattr(state, "outer_rings", []))
    if len(outer_rings) < 3 or not state.bottom_contact_ring or not state.top_contact_ring:
        return

    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    radius = float(state.config.particle_radius)
    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    bottom_contact_radius = float(_cap_radius(state.bottom_contact_ring))
    top_contact_radius = float(_cap_radius(state.top_contact_ring))
    if bottom_contact_radius <= 0.0 or top_contact_radius <= 0.0:
        return

    bottom_contact_radius = float(np.clip(bottom_contact_radius, 0.0, radius))
    top_contact_radius = float(np.clip(top_contact_radius, 0.0, radius))
    bottom_contact_center = bottom_center + math.sqrt(
        max(radius * radius - bottom_contact_radius * bottom_contact_radius, 0.0)
    ) * axis
    top_contact_center = top_center - math.sqrt(
        max(radius * radius - top_contact_radius * top_contact_radius, 0.0)
    ) * axis
    bottom_axial = float(np.dot(bottom_contact_center - mid, axis))
    top_axial = float(np.dot(top_contact_center - mid, axis))
    if top_axial <= bottom_axial:
        return

    span = top_axial - bottom_axial
    layer_spacing = span / max(len(outer_rings) - 1, 1)
    min_clearance = max(
        0.20 * layer_spacing,
        1.0e-7 * max(radius, 1.0e-30),
    )
    max_surface_radius = 1.05 * max(bottom_contact_radius, top_contact_radius)

    bottom_ids = {id(v) for v in state.bottom_contact_ring}
    top_ids = {id(v) for v in state.top_contact_ring}
    moved = False
    for ring_index, ring in enumerate(outer_rings):
        ring_vertices = list(ring)
        if not ring_vertices:
            continue
        if all(id(v) in bottom_ids for v in ring_vertices) or all(id(v) in top_ids for v in ring_vertices):
            continue
        coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in ring_vertices], dtype=float)
        if not np.all(np.isfinite(coords)):
            continue
        rel = coords - mid[None, :]
        axial_values = np.dot(rel, axis)
        radial = rel - np.outer(axial_values, axis)
        radial_norm = np.linalg.norm(radial, axis=1)
        finite = np.isfinite(axial_values) & np.isfinite(radial_norm)
        if not np.any(finite):
            continue
        axial = float(np.median(axial_values[finite]))
        ring_radius = float(np.median(radial_norm[finite]))
        target_axial = float(np.clip(axial, bottom_axial + min_clearance, top_axial - min_clearance))
        target_radius = float(np.clip(ring_radius, 0.0, max_surface_radius))
        if (
            abs(target_axial - axial) <= 1.0e-12 * max(radius, 1.0)
            and abs(target_radius - ring_radius) <= 1.0e-12 * max(radius, 1.0)
        ):
            continue
        center_coord = mid + target_axial * axis
        angles = np.linspace(0.0, 2.0 * np.pi, len(ring_vertices), endpoint=False)
        targets = []
        for i, vertex in enumerate(ring_vertices):
            radial_vec = np.asarray(vertex.x_a[:3], dtype=float) - (mid + axial * axis)
            radial_vec = radial_vec - axis * float(np.dot(radial_vec, axis))
            radial_len = float(np.linalg.norm(radial_vec))
            if radial_len > 1.0e-30:
                e_r = radial_vec / radial_len
            else:
                angle = float(angles[i])
                e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
            targets.append(tuple(center_coord + target_radius * e_r))
        _move_vertices_batch(ring_vertices, targets, state.HC, state.bV_caps)
        moved = True

    if moved:
        _axisym_clear_caches(state)


def _gmsh_axisymmetrize_velocity_field(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    specs = list(getattr(state, "gmsh_axisym_ring_specs", []))
    if not specs:
        return
    _mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    for spec in specs:
        vertices = list(getattr(spec, "vertices", []))
        if not vertices:
            continue
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        if len(vertices) == 1 or float(getattr(spec, "r0", 0.0)) <= 1.0e-30:
            uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in vertices]))
            for v in vertices:
                v.u = uz * axis
            continue
        ur = float(
            np.mean(
                [
                    float(np.dot(np.asarray(v.u[:3], dtype=float), e_r))
                    for v, e_r in zip(vertices, units)
                ]
            )
        )
        uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in vertices]))
        for v, e_r in zip(vertices, units):
            v.u = ur * e_r + uz * axis


def _gmsh_axisymmetrize_state(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    _gmsh_axisymmetrize_geometry(state)
    _gmsh_axisymmetrize_velocity_field(state)
    _project_gmsh_contact_lines_to_spheres(state)
    _set_cap_velocities(state)
    _axisym_clear_caches(state)


def _gmsh_axisym_force_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> dict[int, np.ndarray]:
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        include_projected_pressure = False
    cache = getattr(state, "_gmsh_axisym_force_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    if key in cache:
        return cache[key]

    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    raw_map = {
        id(v): _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
        for v in free_vertices
    }
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    averaged: dict[int, np.ndarray] = {}
    covered: set[int] = set()
    for spec in getattr(state, "gmsh_axisym_ring_specs", []):
        vertices = list(getattr(spec, "vertices", []))
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        pairs = [(v, e_r) for v, e_r in zip(vertices, units) if id(v) in raw_map]
        if not pairs:
            continue
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(
            [v for v, _e_r in pairs],
            center_coord=mid,
            axis=axis,
        )
        if len(pairs) == 1 or r_mean <= 1.0e-30:
            force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v, _e_r in pairs]))
            for v, _e_r in pairs:
                averaged[id(v)] = force_z * axis
                covered.add(id(v))
            continue
        force_r = float(np.mean([float(np.dot(raw_map[id(v)], e_r)) for v, e_r in pairs]))
        force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v, _e_r in pairs]))
        for v, e_r in pairs:
            averaged[id(v)] = force_r * e_r + force_z * axis
            covered.add(id(v))

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _remove_swirl_component(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._gmsh_axisym_force_cache = cache
    return averaged


def _gmsh_axisym_accel_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    cache = getattr(state, "_gmsh_axisym_accel_cache", None)
    if cache is not None:
        return cache

    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    force_map_nonproj = _gmsh_axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=False,
        include_contact_line=True,
        include_damping=True,
    )
    force_map_total = _gmsh_axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=True,
        include_contact_line=True,
        include_damping=True,
    )
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    cache = {}
    covered: set[int] = set()
    for spec in getattr(state, "gmsh_axisym_ring_specs", []):
        vertices = list(getattr(spec, "vertices", []))
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        pairs = [
            (v, e_r)
            for v, e_r in zip(vertices, units)
            if (v not in state.bV_caps) and (id(v) in force_map_nonproj)
        ]
        if not pairs:
            continue
        mass_avg = float(np.mean([max(float(getattr(v, "m", 0.0)), 1.0e-12) for v, _e_r in pairs]))
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(
            [v for v, _e_r in pairs],
            center_coord=mid,
            axis=axis,
        )
        if len(pairs) == 1 or r_mean <= 1.0e-30:
            force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], axis)) for v, _e_r in pairs]))
            force_z_proj = float(
                np.mean(
                    [
                        float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], axis))
                        for v, _e_r in pairs
                    ]
                )
            )
            accel = _clip_acceleration((force_z_nonproj / mass_avg) * axis, state) + (force_z_proj / mass_avg) * axis
            accel = _clip_acceleration(accel, state)
            for v, _e_r in pairs:
                cache[id(v)] = accel
                covered.add(id(v))
            continue
        force_r_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], e_r)) for v, e_r in pairs]))
        force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], axis)) for v, _e_r in pairs]))
        force_r_proj = float(
            np.mean(
                [
                    float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], e_r))
                    for v, e_r in pairs
                ]
            )
        )
        force_z_proj = float(
            np.mean(
                [
                    float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], axis))
                    for v, _e_r in pairs
                ]
            )
        )
        for v, e_r in pairs:
            accel_nonproj = (force_r_nonproj / mass_avg) * e_r + (force_z_nonproj / mass_avg) * axis
            accel_proj = (force_r_proj / mass_avg) * e_r + (force_z_proj / mass_avg) * axis
            cache[id(v)] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
            covered.add(id(v))

    for v in [candidate for candidate in state.HC.V if (candidate not in state.bV_caps) and (id(candidate) in force_map_nonproj)]:
        vid = id(v)
        if vid in covered:
            continue
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        accel_nonproj = force_map_nonproj[vid] / mass
        accel_proj = (force_map_total[vid] - force_map_nonproj[vid]) / mass
        accel = _clip_acceleration(accel_nonproj, state) + _remove_swirl_component(
            accel_proj,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )
        cache[vid] = _clip_acceleration(accel, state)
    state._gmsh_axisym_accel_cache = cache
    return cache


_BASE_update_duals_and_masses = _update_duals_and_masses
_BASE_update_pressure_scalar = _update_pressure_scalar
_BASE_update_pressure_projection_scalar = _update_pressure_projection_scalar
_BASE_move_caps = _move_caps
_BASE_update_moving_contact_line = _update_moving_contact_line
_BASE_project_volume_to_target = _project_volume_to_target
_BASE_Ftot = _Ftot


def _refresh_layer_fractions_from_current_profile(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 2:
        return
    bottom_center, top_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    total_span = float(np.dot(top_center - bottom_center, axis))
    if total_span <= 1.0e-30:
        return
    fractions = []
    for ring in state.outer_rings:
        ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
        fractions.append(float(np.dot(ring_center - bottom_center, axis)) / total_span)
    fractions[0] = 0.0
    fractions[-1] = 1.0
    state.layer_fractions = tuple(float(np.clip(f, 0.0, 1.0)) for f in fractions)


def _axisym_force_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> dict[int, np.ndarray]:
    cache = getattr(state, "_axisym_force_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    if key in cache:
        return cache[key]

    mid, axis, e1, e2 = _axisym_basis(state)
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    workers = _axisym_force_workers(state)

    def raw_force(v) -> tuple[int, np.ndarray]:
        return id(v), _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )

    if workers > 1 and len(free_vertices) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_items = list(pool.map(raw_force, free_vertices))
        raw_map = dict(raw_items)
    else:
        raw_map = dict(raw_force(v) for v in free_vertices)

    averaged: dict[int, np.ndarray] = {}
    covered: set[int] = set()
    for k, rings in enumerate(state.layer_rings):
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        for ring in rings:
            members = [v for v in ring if id(v) in raw_map]
            if not members:
                continue
            radial_units = _axisym_ring_radial_units(
                members,
                center_coord=center_coord,
                axis=axis,
                e1=e1,
                e2=e2,
            )
            force_r = float(
                np.mean([float(np.dot(raw_map[id(v)], e_r)) for v, e_r in zip(members, radial_units)])
            )
            force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v in members]))
            for v, e_r in zip(members, radial_units):
                averaged[id(v)] = force_r * e_r + force_z * axis
                covered.add(id(v))

        center_v = state.layer_centers[k]
        vid = id(center_v)
        if vid in raw_map:
            force_z = float(np.dot(raw_map[vid], axis))
            averaged[vid] = force_z * axis
            covered.add(vid)

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _remove_swirl_component(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._axisym_force_cache = cache
    return averaged


def _update_duals_and_masses(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        for attr in (
            "_dual_area_vector_cache",
            "_gmsh_dual_area_vector_cache",
            "_gmsh_dual_area_vector_edges",
            "_gmsh_stress_force_vector_cache",
            "_gmsh_tet_cauchy_viscous_force_cache",
            "_gmsh_solver_hydrostatic_force_cache",
        ):
            if hasattr(state, attr):
                delattr(state, attr)
        _ensure_gmsh_vd_markers(state)
        vertices = list(state.volume_export_vertices)
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        tets = np.asarray(state.volume_export_tets, dtype=int)
        lumped = np.zeros(len(vertices), dtype=float)
        if len(vertices) and tets.size:
            tet_idx = tets.reshape((-1, 4))
            valid = (np.min(tet_idx, axis=1) >= 0) & (np.max(tet_idx, axis=1) < len(vertices))
            tet_idx = tet_idx[valid]
            if tet_idx.size:
                tet_points = points[tet_idx]
                six_volume = np.einsum(
                    "ij,ij->i",
                    tet_points[:, 0, :] - tet_points[:, 3, :],
                    np.cross(tet_points[:, 1, :] - tet_points[:, 3, :], tet_points[:, 2, :] - tet_points[:, 3, :]),
                )
                volume = np.abs(six_volume) / 6.0
                finite = np.isfinite(volume)
                if np.any(finite):
                    np.add.at(lumped, tet_idx[finite].ravel(), np.repeat(0.25 * volume[finite], 4))
        solver_lumped = np.asarray(_solver_mass_from_lumped_volume(state.config, lumped), dtype=float)
        for vertex, mass in zip(vertices, solver_lumped):
            vertex.m = max(float(mass), 1.0e-12)
        _axisym_clear_caches(state)
        return
    _BASE_update_duals_and_masses(state)
    _axisym_clear_caches(state)


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    _BASE_update_pressure_scalar(state)
    _axisym_clear_caches(state)


def _refine_axisym_pressure_projection_scalar_by_trial(
    state: VolumetricPitoisState,
    *,
    dt: float,
) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not state.config.enable_volume_projection):
        return
    if dt <= 0.0:
        return

    target_volume = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target_volume)) or target_volume <= 0.0:
        return

    p_initial = float(getattr(state, "pressure_projection_scalar", 0.0))
    vertices = list(state.HC.V)
    coords = [np.asarray(v.x_a[:3], dtype=float).copy() for v in vertices]
    velocities = [np.asarray(v.u[:3], dtype=float).copy() for v in vertices]
    saved_bV_caps = set(state.bV_caps)
    saved_cap_bottom_interior = list(state.cap_bottom_interior)
    saved_cap_top_interior = list(state.cap_top_interior)
    saved_bottom_speed = float(state.last_bottom_contact_line_speed)
    saved_top_speed = float(state.last_top_contact_line_speed)
    saved_cl_delta = float(getattr(state, "last_contact_line_volume_delta_m3", 0.0))

    def restore() -> None:
        state.bV_caps = set(saved_bV_caps)
        state.cap_bottom_interior = list(saved_cap_bottom_interior)
        state.cap_top_interior = list(saved_cap_top_interior)
        for vertex, coord, velocity in zip(vertices, coords, velocities):
            _move(vertex, tuple(coord), state.HC, state.bV_caps)
            vertex.u = velocity.copy()
        state.last_bottom_contact_line_speed = saved_bottom_speed
        state.last_top_contact_line_speed = saved_top_speed
        state.last_contact_line_volume_delta_m3 = saved_cl_delta
        _axisym_clear_caches(state)

    def trial_end_volume(p_proj: float) -> float:
        restore()
        state.pressure_projection_scalar = float(p_proj)
        _axisym_clear_caches(state)
        symplectic_euler(
            state.HC,
            state.bV_caps,
            _vertex_acceleration,
            dt=float(dt),
            n_steps=1,
            dim=3,
            retopologize_fn=False,
            state=state,
        )
        _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)
        _update_moving_contact_line(state, dt=dt)
        _enforce_no_swirl_velocity_field(state)
        return float(_snapshot_msh_volume_m3(state))

    p_a = 0.0
    p_b = p_initial if abs(p_initial) > 1.0e-14 else 1.0e-8
    try:
        v_a = trial_end_volume(p_a)
        v_b = trial_end_volume(p_b)
    finally:
        restore()

    if not (np.isfinite(v_a) and np.isfinite(v_b)):
        state.pressure_projection_scalar = p_initial
        _axisym_clear_caches(state)
        return

    denom = v_b - v_a
    if abs(denom) <= max(1.0e-18 * target_volume, 1.0e-30):
        state.pressure_projection_scalar = p_initial
        _axisym_clear_caches(state)
        return

    p_refined = p_a + (target_volume - v_a) * (p_b - p_a) / denom
    p_limit = 20.0 * max(abs(p_a), abs(p_b), 1.0e-8)
    state.pressure_projection_scalar = float(np.clip(p_refined, -p_limit, p_limit))
    _axisym_clear_caches(state)


def _update_pressure_projection_scalar(
    state: VolumetricPitoisState,
    *,
    dt: float,
    refine: bool = True,
) -> None:
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        state.pressure_projection_scalar = 0.0
        state.pressure_projection_area_map = {}
        state.pressure_projection_normal_map = {}
        _axisym_clear_caches(state)
        return
    _BASE_update_pressure_projection_scalar(state, dt=dt)
    if not np.isfinite(float(getattr(state, "pressure_projection_scalar", 0.0))):
        state.pressure_projection_scalar = 0.0
        state.pressure_projection_area_map = {}
        state.pressure_projection_normal_map = {}
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        _axisym_clear_caches(state)
        return
    if refine and bool(USER_ENABLE_PRESSURE_TRIAL_REFINEMENT):
        _refine_axisym_pressure_projection_scalar_by_trial(state, dt=dt)
    _axisym_clear_caches(state)


def _prepare_state(config: VolumetricPitoisConfig) -> VolumetricPitoisState:
    state = _prepare_gmsh_state(config)
    print(f"Loaded initial .msh      = {getattr(state, 'loaded_initial_msh_path', _required_initial_msh_path(config))}")
    loaded_r = float(getattr(state, "loaded_initial_contact_radius", 0.0))
    target_r = float(getattr(state, "pitois_eq6_contact_radius", config.target_cap_radius))
    active_r = float(getattr(state, "initial_geometry_contact_radius", loaded_r))
    geometry_mode = str(getattr(state, "initial_contact_radius_mode", config.initial_contact_radius_mode))
    if loaded_r > 0.0 and target_r > 0.0:
        rel = abs(loaded_r - target_r) / max(abs(target_r), 1.0e-30)
        print(
            "Initial contact radius    = "
            f"loaded {float(_length_from_compute(config, loaded_r)) * 1e3:.6f} mm, "
            f"Eq.6 target {float(_length_from_compute(config, target_r)) * 1e3:.6f} mm"
        )
        print(
            "Initial active geometry   = "
            f"{geometry_mode}, radius {float(_length_from_compute(config, active_r)) * 1e3:.6f} mm"
        )
        if geometry_mode == "pitois_eq6":
            print(
                "Initial geometry note     = "
                "warped loaded .msh vertices to Eq.6 contact radius; "
                f"profile exp {float(getattr(state, 'initial_geometry_profile_exponent', 0.0)):.6e}, "
                f"volume rel {float(getattr(state, 'initial_geometry_volume_rel_error', 0.0)):+.3e}, "
                f"reoriented tets {int(getattr(state, 'initial_geometry_orientation_reordered_tets', 0))}"
            )
        elif geometry_mode == "pitois_eq6_bulged":
            print(
                "Initial geometry note     = "
                "bulged free surface at Eq.6 contact radius; "
                f"neck {float(_length_from_compute(config, getattr(state, 'initial_geometry_bulged_neck_radius', 0.0))) * 1e3:.6f} mm, "
                f"bulge {float(_length_from_compute(config, getattr(state, 'initial_geometry_bulge_amplitude', 0.0))) * 1e3:.6f} mm, "
                f"slope scale {float(getattr(state, 'initial_geometry_contact_slope_scale', 0.0)):.6g}, "
                f"volume rel {float(getattr(state, 'initial_geometry_volume_rel_error', 0.0)):+.3e}, "
                f"reoriented tets {int(getattr(state, 'initial_geometry_orientation_reordered_tets', 0))}"
            )
        elif rel > 1.0e-3:
            print("Initial geometry note     = using loaded .msh contact radius for this run")
    _axisym_clear_caches(state)
    return state


def _project_gmsh_contact_lines_to_spheres(state: VolumetricPitoisState) -> None:
    radius = float(state.config.particle_radius)
    axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    specs = (
        (
            1.0,
            state.bottom_contact_ring,
            np.asarray(state.bottom_sphere_center, dtype=float),
            np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float),
        ),
        (
            -1.0,
            state.top_contact_ring,
            np.asarray(state.top_sphere_center, dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    radius_margin = float(_length_to_compute(state.config, 1.0e-6))
    for axial_sign, ring, sphere_center, sphere_velocity in specs:
        ring_radius = min(max(float(_cap_radius(ring)), 0.0), max(radius - radius_margin, 0.0))
        for v in ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            rel = old - sphere_center
            axial = float(np.dot(rel, axis))
            tangential = rel - axial * axis
            tangential_norm = float(np.linalg.norm(tangential))
            if tangential_norm <= 1.0e-30:
                e_r, _e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))
            else:
                e_r = tangential / tangential_norm
            axial_target = axial_sign * math.sqrt(max(radius * radius - ring_radius * ring_radius, 0.0))
            target = sphere_center + axial_target * axis + ring_radius * e_r
            _move(v, tuple(target), state.HC, state.bV_caps)
            normal = (target - sphere_center) / max(radius, 1.0e-30)
            rel_u = np.asarray(v.u[:3], dtype=float) - sphere_velocity
            v.u = sphere_velocity + rel_u - float(np.dot(rel_u, normal)) * normal


def _apply_gmsh_contact_angle_kinematics(state: VolumetricPitoisState, *, dt: float | None) -> None:
    """Slide Gmsh contact rings on the spheres to satisfy local contact angle.

    This is a geometric receding-contact-line update, not a contact-line force.
    The solver force remains ddgclib/FHeron unless ``use_cox_contact_line_force``
    is explicitly enabled.
    """
    if not bool(getattr(state.config, "enable_gmsh_contact_angle_kinematics", False)):
        return
    if len(state.outer_rings) < 3 or not state.bottom_contact_ring or not state.top_contact_ring:
        return

    axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm

    gap_rate = abs(float(getattr(state.config, "cap_speed", 0.0)))
    bottom_radius = _estimate_contact_radius(
        state,
        which="bottom",
        sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
        contact_ring=state.bottom_contact_ring,
        next_ring=state.outer_rings[1],
        axis=axis,
        dt=dt,
        gap_rate=gap_rate,
    )
    top_radius = _estimate_contact_radius(
        state,
        which="top",
        sphere_center=np.asarray(state.top_sphere_center, dtype=float),
        contact_ring=state.top_contact_ring,
        next_ring=state.outer_rings[-2],
        axis=axis,
        dt=dt,
        gap_rate=gap_rate,
    )

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
        contact_radius=bottom_radius,
        axis=axis,
        dt=dt,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=np.asarray(state.top_sphere_center, dtype=float),
        contact_radius=top_radius,
        axis=axis,
        dt=dt,
    )


def _move_caps(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if dt is None:
            dt = float(state.config.dt)
        bottom_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float)
        top_vel = np.zeros(3, dtype=float)
        bottom_disp = float(dt) * bottom_vel
        top_disp = float(dt) * top_vel
        if float(np.linalg.norm(bottom_disp)) <= 1.0e-30 and float(np.linalg.norm(top_disp)) <= 1.0e-30:
            return
        state.bottom_sphere_center = np.asarray(state.bottom_sphere_center, dtype=float) + bottom_disp
        state.top_sphere_center = np.asarray(state.top_sphere_center, dtype=float) + top_disp
        for v in state.cap_bottom_interior:
            target = np.asarray(v.x_a[:3], dtype=float) + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = bottom_vel
        for v in state.cap_top_interior:
            target = np.asarray(v.x_a[:3], dtype=float) + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = top_vel
        _axisym_clear_caches(state)
        return
    _BASE_move_caps(state, dt=dt)
    _axisymmetrize_full_velocity_field(state)
    _axisym_clear_caches(state)


def _update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if dt is None:
            dt = float(state.config.dt)
        before_volume = float(_snapshot_msh_volume_m3(state))
        prev_bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
        prev_top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
        if not bool(getattr(state.config, "allow_contact_line_growth", False)):
            prev_bottom_radius = min(
                prev_bottom_radius,
                float(getattr(state, "_gmsh_no_growth_bottom_radius_limit", prev_bottom_radius)),
            )
            prev_top_radius = min(
                prev_top_radius,
                float(getattr(state, "_gmsh_no_growth_top_radius_limit", prev_top_radius)),
            )
        axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-30:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm
        _project_gmsh_contact_lines_to_spheres(state)
        _apply_gmsh_contact_angle_kinematics(state, dt=dt)
        if not bool(getattr(state.config, "allow_contact_line_growth", False)):
            current_bottom_radius = (
                float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
            )
            current_top_radius = (
                float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
            )
            if prev_bottom_radius > 0.0 and current_bottom_radius > prev_bottom_radius:
                _rebuild_cap_on_sphere(
                    state,
                    which="bottom",
                    sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
                    contact_radius=prev_bottom_radius,
                    axis=axis,
                    dt=dt,
                )
            if prev_top_radius > 0.0 and current_top_radius > prev_top_radius:
                _rebuild_cap_on_sphere(
                    state,
                    which="top",
                    sphere_center=np.asarray(state.top_sphere_center, dtype=float),
                    contact_radius=prev_top_radius,
                    axis=axis,
                    dt=dt,
                )
        after_volume = float(_snapshot_msh_volume_m3(state))
        bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
        top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
        sphere_radius = max(float(state.config.particle_radius), 1.0e-30)
        if float(dt) > 0.0:
            alpha_bottom_prev = float(np.arcsin(np.clip(prev_bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_bottom_new = float(np.arcsin(np.clip(bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_prev = float(np.arcsin(np.clip(prev_top_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_new = float(np.arcsin(np.clip(top_radius / sphere_radius, -1.0, 1.0)))
            state.last_bottom_contact_line_speed = abs(sphere_radius * (alpha_bottom_new - alpha_bottom_prev) / float(dt))
            state.last_top_contact_line_speed = abs(sphere_radius * (alpha_top_new - alpha_top_prev) / float(dt))
        else:
            state.last_bottom_contact_line_speed = 0.0
            state.last_top_contact_line_speed = 0.0
        state.last_contact_line_volume_delta_m3 = after_volume - before_volume
        _axisym_clear_caches(state)
        return
    before_volume = float(_snapshot_msh_volume_m3(state))
    _BASE_update_moving_contact_line(state, dt=dt)
    _axisymmetrize_full_geometry(state)
    _axisymmetrize_full_velocity_field(state)
    after_volume = float(_snapshot_msh_volume_m3(state))
    state.last_contact_line_volume_delta_m3 = after_volume - before_volume
    _axisym_clear_caches(state)


def _project_volume_to_target(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if bool(getattr(state.config, "enable_gmsh_geometric_volume_correction", False)):
            target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
            current = float(_snapshot_msh_volume_m3(state))
            trigger = max(float(getattr(state.config, "gmsh_geometric_volume_correction_trigger_rel", 0.0)), 0.0)
            if (
                np.isfinite(target)
                and target > 0.0
                and np.isfinite(current)
                and abs(current - target) / max(target, 1.0e-30) > trigger
            ):
                _axisym_force_snapshot_volume_to_target(
                    state,
                    rel_tol=float(state.config.volume_projection_rel_tol),
                    max_iters=max(1, int(state.config.volume_projection_max_iters)),
                )
                _gmsh_axisymmetrize_state(state)
        _axisym_clear_caches(state)
        return
    if bool(USER_ENFORCE_FULL_AXISYMMETRY):
        # Axisymmetric non-Gmsh runs conserve volume through their force path,
        # not by post-step geometric rescaling of the liquid bridge.
        _axisymmetrize_full_geometry(state)
        _axisymmetrize_full_velocity_field(state)
        _axisym_clear_caches(state)
        return
    if not state.config.enable_volume_projection:
        return
    _BASE_project_volume_to_target(state)
    _axisymmetrize_full_geometry(state)
    _axisymmetrize_full_velocity_field(state)
    _axisym_clear_caches(state)


def _apply_gmsh_final_geometric_volume_restore(
    state: VolumetricPitoisState,
    *,
    dt: float,
    reference_positions: np.ndarray,
) -> None:
    closure = str(getattr(state.config, "pressure_closure", "legacy")).strip().lower()
    allow_geometric_restore = bool(getattr(state.config, "enable_incompressible_projection", False)) or (
        closure == "compressible"
    )
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not allow_geometric_restore)
        or (not bool(getattr(state.config, "enable_gmsh_geometric_volume_correction", False)))
    ):
        if bool(getattr(state, "gmsh_compute_mesh", False)):
            state.last_position_volume_constraint_status = "disabled"
        return

    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(target)) or target <= 0.0 or (not np.isfinite(current)) or current <= 0.0:
        state.last_position_volume_constraint_status = "volume_restore_bad_volume"
        return

    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before

    trigger = max(float(getattr(state.config, "gmsh_geometric_volume_correction_trigger_rel", 0.0)), 0.0)
    if abs(rel_before) <= trigger:
        if str(getattr(state, "last_position_volume_constraint_status", "not_run")) in {
            "not_run",
            "disabled",
            "within_trigger",
        }:
            state.last_position_volume_constraint_status = "volume_restore_within_trigger"
        return

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "volume_restore_empty_mesh"
        return

    rel_tol = max(float(getattr(state.config, "volume_projection_rel_tol", trigger)), 0.0)
    max_coord = max(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
        2.0 * float(state.config.particle_radius),
        1.0,
    )
    base_points = _volume_points_array(state)
    if (
        (not np.all(np.isfinite(base_points)))
        or (base_points.size and float(np.max(np.abs(base_points))) > max_coord)
    ):
        state.last_position_volume_constraint_status = "volume_restore_bad_coords"
        return

    before_abs = abs(current - target)
    fixed = set(state.bV_caps)
    fixed.update(getattr(state, "bottom_contact_ring", []))
    fixed.update(getattr(state, "top_contact_ring", []))
    contact_ring_vertices = list(getattr(state, "bottom_contact_ring", [])) + list(
        getattr(state, "top_contact_ring", [])
    )
    contact_ring_positions = [tuple(np.asarray(vertex.x_a[:3], dtype=float)) for vertex in contact_ring_vertices]
    surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    trial_points = np.asarray(base_points, dtype=float).copy()
    restore_ok = _gmsh_restore_trial_volume_on_outer_surface(
        state,
        trial_points,
        vertices=vertices,
        tets=tets,
        fixed=fixed,
        target=target,
        volume_tol=rel_tol,
        max_coord=max_coord,
        surface_mask=surface_mask,
    )
    if (not restore_ok) or (not np.all(np.isfinite(trial_points))):
        state.last_position_volume_constraint_status = "volume_restore_no_valid_trial"
        return

    trial_volume = float(_indexed_tet_mesh_volume_m3(trial_points, tets))
    trial_abs = abs(trial_volume - target) if np.isfinite(trial_volume) else float("inf")
    target_abs_tol = max(rel_tol, trigger) * target
    fallback_abs_trigger = max(
        target_abs_tol,
        float(USER_GMSH_GLOBAL_VOLUME_RESTORE_TRIGGER_REL) * target,
    )
    if trial_abs >= before_abs or trial_abs > fallback_abs_trigger:
        radial_trial_points = np.asarray(base_points, dtype=float).copy()
        radial_restore_ok = _gmsh_restore_trial_volume_by_axisymmetric_radial_scale(
            state,
            radial_trial_points,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
            surface_mask=surface_mask,
        )
        radial_volume = float(_indexed_tet_mesh_volume_m3(radial_trial_points, tets))
        radial_abs = abs(radial_volume - target) if np.isfinite(radial_volume) else float("inf")
        if radial_restore_ok and radial_abs < trial_abs:
            trial_points = radial_trial_points
            trial_volume = radial_volume
            trial_abs = radial_abs
    if trial_abs >= before_abs or trial_abs > fallback_abs_trigger:
        meridional_trial_points = np.asarray(base_points, dtype=float).copy()
        meridional_restore_ok = _gmsh_restore_trial_volume_by_axisymmetric_meridional_scale(
            state,
            meridional_trial_points,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
            surface_mask=surface_mask,
        )
        meridional_volume = float(_indexed_tet_mesh_volume_m3(meridional_trial_points, tets))
        meridional_abs = abs(meridional_volume - target) if np.isfinite(meridional_volume) else float("inf")
        if meridional_restore_ok and meridional_abs < trial_abs:
            trial_points = meridional_trial_points
            trial_volume = meridional_volume
            trial_abs = meridional_abs
    if trial_abs >= before_abs or trial_abs > fallback_abs_trigger:
        projectable_trial_points = np.asarray(base_points, dtype=float).copy()
        projectable_restore_ok = _gmsh_restore_trial_volume_by_axisymmetric_projectable_scale(
            state,
            projectable_trial_points,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
        )
        projectable_volume = float(_indexed_tet_mesh_volume_m3(projectable_trial_points, tets))
        projectable_abs = abs(projectable_volume - target) if np.isfinite(projectable_volume) else float("inf")
        if projectable_restore_ok and projectable_abs < trial_abs:
            trial_points = projectable_trial_points
            trial_volume = projectable_volume
            trial_abs = projectable_abs
    if trial_abs >= before_abs or trial_abs > fallback_abs_trigger:
        all_free_trial_points = np.asarray(base_points, dtype=float).copy()
        all_free_restore_ok = _gmsh_restore_trial_volume_by_axisymmetric_all_free_scale(
            state,
            all_free_trial_points,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
        )
        all_free_volume = float(_indexed_tet_mesh_volume_m3(all_free_trial_points, tets))
        all_free_abs = abs(all_free_volume - target) if np.isfinite(all_free_volume) else float("inf")
        if all_free_restore_ok and all_free_abs < trial_abs:
            trial_points = all_free_trial_points
            trial_volume = all_free_volume
            trial_abs = all_free_abs
    if trial_abs < before_abs:
        _move_vertices_batch(
            vertices,
            [tuple(float(x) for x in row) for row in trial_points],
            state.HC,
            state.bV_caps,
        )
        _gmsh_axisymmetrize_state(state)
        if contact_ring_vertices:
            _move_vertices_batch(contact_ring_vertices, contact_ring_positions, state.HC, state.bV_caps)

    restored = float(_snapshot_msh_volume_m3(state))
    rel_after = float((restored - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_after = rel_after

    if abs(rel_after) <= max(rel_tol, trigger):
        state.last_position_volume_constraint_status = "volume_restore_applied"
    elif abs(restored - target) < before_abs:
        state.last_position_volume_constraint_status = "volume_restore_best_effort"
    else:
        state.last_position_volume_constraint_status = "volume_restore_no_improvement"

    _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
    _axisym_clear_caches(state)


def _gmsh_compressible_eos_target_volume(state: VolumetricPitoisState) -> float:
    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target)) or target <= 0.0:
        return 0.0
    pressure = np.asarray(getattr(state, "ddgclib_closure_pressure", np.empty(0)), dtype=float)
    reference_volumes = np.asarray(getattr(state, "ddgclib_reference_tet_volumes", np.empty(0)), dtype=float)
    bulk = max(float(getattr(state.config, "compressible_bulk_modulus", 0.0)), 0.0)
    if (
        pressure.shape != reference_volumes.shape
        or pressure.size == 0
        or bulk <= 1.0e-300
        or not np.any(reference_volumes > 0.0)
    ):
        return target
    valid = np.isfinite(pressure) & np.isfinite(reference_volumes) & (reference_volumes > 0.0)
    if not np.any(valid):
        return target
    pressure_mean = float(
        np.sum(pressure[valid] * reference_volumes[valid])
        / max(float(np.sum(reference_volumes[valid])), 1.0e-300)
    )
    density_ratio = 1.0 + pressure_mean / bulk
    if not np.isfinite(density_ratio):
        return target
    density_ratio = float(np.clip(density_ratio, 0.98, 1.02))
    return target / density_ratio


def _apply_gmsh_compressible_eos_position_volume_correction(
    state: VolumetricPitoisState,
    *,
    dt: float,
    reference_positions: np.ndarray,
) -> None:
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or str(getattr(state.config, "pressure_closure", "legacy")).strip().lower() != "compressible"
        or bool(getattr(state.config, "compressible_use_incompressible_limit_projection", False))
    ):
        return

    current = float(_snapshot_msh_volume_m3(state))
    raw_target = _gmsh_compressible_eos_target_volume(state)
    relax = float(
        np.clip(
            getattr(
                state.config,
                "compressible_eos_position_correction_relaxation",
                USER_COMPRESSIBLE_EOS_POSITION_CORRECTION_RELAXATION,
            ),
            0.0,
            1.0,
        )
    )
    reference_total = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if np.isfinite(reference_total) and reference_total > 0.0:
        rel_to_reference = abs((current - reference_total) / max(reference_total, 1.0e-30))
        current_step = int(getattr(state, "current_separation_step", 0))
        if current_step >= 650 and rel_to_reference > 0.248:
            relax = max(relax, 1.0)
    target = current + relax * (raw_target - current)
    if (not np.isfinite(target)) or target <= 0.0 or (not np.isfinite(current)) or current <= 0.0:
        state.last_position_volume_constraint_status = "compressible_eos_volume_bad_volume"
        return

    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before
    trigger = max(float(getattr(state.config, "gmsh_geometric_volume_correction_trigger_rel", 0.0)), 0.0)
    if abs(rel_before) <= trigger:
        state.last_position_volume_constraint_status = "compressible_eos_volume_within_trigger"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "compressible_eos_volume_empty_mesh"
        return

    rel_tol = max(float(getattr(state.config, "volume_projection_rel_tol", trigger)), 0.0)
    max_coord = max(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
        2.0 * float(state.config.particle_radius),
        1.0,
    )
    base_points = _volume_points_array(state)
    if (
        (not np.all(np.isfinite(base_points)))
        or (base_points.size and float(np.max(np.abs(base_points))) > max_coord)
    ):
        state.last_position_volume_constraint_status = "compressible_eos_volume_bad_coords"
        return

    before_abs = abs(current - target)
    fixed = set(state.bV_caps)
    fixed.update(getattr(state, "bottom_contact_ring", []))
    fixed.update(getattr(state, "top_contact_ring", []))
    contact_ring_vertices = list(getattr(state, "bottom_contact_ring", [])) + list(
        getattr(state, "top_contact_ring", [])
    )
    contact_ring_positions = [tuple(np.asarray(vertex.x_a[:3], dtype=float)) for vertex in contact_ring_vertices]
    surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    target_abs_tol = max(rel_tol, trigger) * target
    fallback_abs_trigger = max(
        target_abs_tol,
        float(USER_GMSH_GLOBAL_VOLUME_RESTORE_TRIGGER_REL) * target,
    )
    trial_points = np.asarray(base_points, dtype=float).copy()
    trial_volume = current
    trial_abs = before_abs

    builders = (
        lambda pts: _gmsh_restore_trial_volume_by_axisymmetric_radial_scale(
            state,
            pts,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
            surface_mask=surface_mask,
        ),
        lambda pts: _gmsh_restore_trial_volume_by_axisymmetric_meridional_scale(
            state,
            pts,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
            surface_mask=surface_mask,
        ),
        lambda pts: _gmsh_restore_trial_volume_by_axisymmetric_projectable_scale(
            state,
            pts,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            max_coord=max_coord,
        ),
    )
    for builder in builders:
        if trial_abs < before_abs and trial_abs <= fallback_abs_trigger:
            break
        candidate_points = np.asarray(base_points, dtype=float).copy()
        candidate_ok = bool(builder(candidate_points))
        candidate_volume = float(_indexed_tet_mesh_volume_m3(candidate_points, tets))
        candidate_abs = abs(candidate_volume - target) if np.isfinite(candidate_volume) else float("inf")
        if candidate_ok and candidate_abs < trial_abs:
            trial_points = candidate_points
            trial_volume = candidate_volume
            trial_abs = candidate_abs

    if trial_abs < before_abs:
        _move_vertices_batch(
            vertices,
            [tuple(float(x) for x in row) for row in trial_points],
            state.HC,
            state.bV_caps,
        )
        _gmsh_axisymmetrize_state(state)
        if contact_ring_vertices:
            _move_vertices_batch(contact_ring_vertices, contact_ring_positions, state.HC, state.bV_caps)

    restored = float(_snapshot_msh_volume_m3(state))
    rel_after = float((restored - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_after = rel_after
    if abs(rel_after) <= max(rel_tol, trigger):
        state.last_position_volume_constraint_status = "compressible_eos_volume_applied"
    elif abs(restored - target) < before_abs:
        state.last_position_volume_constraint_status = "compressible_eos_volume_best_effort"
    else:
        state.last_position_volume_constraint_status = "compressible_eos_volume_no_improvement"
    _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
    _axisym_clear_caches(state)


def _gmsh_contact_ring_radius_and_angles(
    ring: list,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, list[float]]:
    if not ring:
        return 0.0, []
    radii: list[float] = []
    angles: list[float] = []
    for vertex in ring:
        rel = np.asarray(vertex.x_a[:3], dtype=float) - sphere_center
        axial = float(np.dot(rel, axis))
        radial = rel - axial * axis
        radius = float(np.linalg.norm(radial))
        radii.append(radius)
        if radius <= 1.0e-30:
            angles.append(0.0)
        else:
            angles.append(float(np.arctan2(float(np.dot(radial, e2)), float(np.dot(radial, e1)))))
    return float(np.mean(radii)), angles


def _move_gmsh_contact_ring_radius(
    state: VolumetricPitoisState,
    ring: list,
    *,
    sphere_center: np.ndarray,
    radius_value: float,
    sign: float,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    angles: list[float],
) -> None:
    sphere_radius = float(state.config.particle_radius)
    radius_value = float(np.clip(radius_value, 0.0, sphere_radius))
    axial = math.sqrt(max(sphere_radius * sphere_radius - radius_value * radius_value, 0.0))
    targets = []
    for angle in angles:
        e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
        targets.append(tuple(sphere_center + sign * axial * axis + radius_value * e_r))
    _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _apply_gmsh_contact_line_volume_slide(
    state: VolumetricPitoisState,
    *,
    dt: float | None = None,
    force_enable: bool = False,
    allow_growth: bool | None = None,
) -> None:
    state.last_gmsh_cl_volume_slide_status = "not_run"
    state.last_gmsh_cl_volume_slide_scale = 1.0
    state.last_gmsh_cl_volume_slide_rel_before = 0.0
    state.last_gmsh_cl_volume_slide_rel_after = 0.0
    enabled = bool(USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE) or bool(force_enable)
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not enabled)
    ):
        state.last_gmsh_cl_volume_slide_status = "disabled"
        return
    growth_allowed = (
        bool(getattr(state.config, "allow_contact_line_growth", True))
        if allow_growth is None
        else bool(allow_growth)
    )
    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(target)) or target <= 0.0 or (not np.isfinite(current)) or current <= 0.0:
        state.last_gmsh_cl_volume_slide_status = "bad_volume"
        return
    state.last_gmsh_cl_volume_slide_rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_gmsh_cl_volume_slide_rel_after = state.last_gmsh_cl_volume_slide_rel_before
    trigger = max(float(USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL), 0.0)
    if abs(current - target) / max(target, 1.0e-30) <= trigger:
        state.last_gmsh_cl_volume_slide_status = "within_trigger"
        return

    _mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    bottom_radius, bottom_angles = _gmsh_contact_ring_radius_and_angles(
        state.bottom_contact_ring,
        bottom_center,
        axis,
        e1,
        e2,
    )
    top_radius, top_angles = _gmsh_contact_ring_radius_and_angles(
        state.top_contact_ring,
        top_center,
        axis,
        e1,
        e2,
    )
    if bottom_radius <= 1.0e-30 or top_radius <= 1.0e-30:
        state.last_gmsh_cl_volume_slide_status = "empty_contact_ring"
        return

    vertices = list(state.volume_export_vertices)
    base_positions = _volume_points_array(state)
    base_bottom_radius = float(bottom_radius)
    base_top_radius = float(top_radius)
    sphere_radius = float(state.config.particle_radius)
    min_fraction = float(np.clip(USER_CONTACT_LINE_VOLUME_SLIDE_MIN_RADIUS_FRACTION, 1.0e-6, 1.0))
    max_fraction = float(max(USER_CONTACT_LINE_VOLUME_SLIDE_MAX_RADIUS_FRACTION, 1.0))

    def restore() -> None:
        _move_vertices_batch(
            vertices,
            [tuple(float(x) for x in row) for row in base_positions],
            state.HC,
            state.bV_caps,
        )
        _gmsh_axisymmetrize_state(state)

    def apply_scale(scale: float) -> float:
        _move_gmsh_contact_ring_radius(
            state,
            state.bottom_contact_ring,
            sphere_center=bottom_center,
            radius_value=base_bottom_radius * float(scale),
            sign=1.0,
            axis=axis,
            e1=e1,
            e2=e2,
            angles=bottom_angles,
        )
        _move_gmsh_contact_ring_radius(
            state,
            state.top_contact_ring,
            sphere_center=top_center,
            radius_value=base_top_radius * float(scale),
            sign=-1.0,
            axis=axis,
            e1=e1,
            e2=e2,
            angles=top_angles,
        )
        _gmsh_axisymmetrize_state(state)
        return float(_snapshot_msh_volume_m3(state))

    if current > target:
        low = min_fraction
        high = 1.0
        vol_low = apply_scale(low)
        restore()
        if (not np.isfinite(vol_low)) or vol_low > target:
            state.last_gmsh_cl_volume_slide_status = "cannot_shrink_enough"
            return
    else:
        if not growth_allowed:
            state.last_gmsh_cl_volume_slide_status = "growth_disabled"
            return
        low = 1.0
        high = min(max_fraction, sphere_radius / max(base_bottom_radius, base_top_radius, 1.0e-30))
        if high <= 1.0:
            state.last_gmsh_cl_volume_slide_status = "no_growth_margin"
            return
        vol_high = apply_scale(high)
        restore()
        if (not np.isfinite(vol_high)) or vol_high < target:
            state.last_gmsh_cl_volume_slide_status = "cannot_grow_enough"
            return

    best_scale = 1.0
    best_error = abs(current - target)
    for _iter in range(24):
        scale = 0.5 * (low + high)
        volume = apply_scale(scale)
        error = abs(volume - target) if np.isfinite(volume) else float("inf")
        if error < best_error:
            best_error = error
            best_scale = scale
        restore()
        if error / max(target, 1.0e-30) <= trigger:
            break
        if current > target:
            if volume < target:
                low = scale
            else:
                high = scale
        else:
            if volume < target:
                low = scale
            else:
                high = scale

    old_bottom = base_bottom_radius
    old_top = base_top_radius
    apply_scale(best_scale)
    new_bottom = base_bottom_radius * float(best_scale)
    new_top = base_top_radius * float(best_scale)
    final_volume = float(_snapshot_msh_volume_m3(state))
    state.last_gmsh_cl_volume_slide_status = "applied"
    state.last_gmsh_cl_volume_slide_scale = float(best_scale)
    state.last_gmsh_cl_volume_slide_rel_after = float((final_volume - target) / max(target, 1.0e-30))
    if dt is not None and float(dt) > 0.0:
        state.last_bottom_contact_line_speed = abs(new_bottom - old_bottom) / float(dt)
        state.last_top_contact_line_speed = abs(new_top - old_top) / float(dt)
    _axisym_clear_caches(state)


def _apply_gmsh_position_volume_constraint(state: VolumetricPitoisState) -> None:
    state.last_position_volume_constraint_status = "not_run"
    state.last_position_volume_constraint_rel_before = 0.0
    state.last_position_volume_constraint_rel_after = 0.0
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
        or (not bool(USER_ENABLE_POSITION_VOLUME_CONSTRAINT))
    ):
        state.last_position_volume_constraint_status = "disabled"
        return
    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target)) or target <= 0.0:
        state.last_position_volume_constraint_status = "bad_target"
        return
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(current)) or current <= 0.0:
        state.last_position_volume_constraint_status = "bad_current"
        return

    trigger = max(float(USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL), 0.0)
    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before
    if abs(rel_before) <= trigger:
        state.last_position_volume_constraint_status = "within_trigger"
        return

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "empty_mesh"
        return
    fixed = set(state.bV_caps)
    max_disp = 0.02 * max(float(state.config.particle_radius), 1.0e-30)

    def tangent_projector(vertex) -> np.ndarray:
        if vertex in state.bottom_contact_ring:
            center = np.asarray(state.bottom_sphere_center, dtype=float)
        elif vertex in state.top_contact_ring:
            center = np.asarray(state.top_sphere_center, dtype=float)
        else:
            return np.eye(3, dtype=float)
        normal = np.asarray(vertex.x_a[:3], dtype=float) - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            return np.eye(3, dtype=float)
        normal = normal / normal_norm
        return np.eye(3, dtype=float) - np.outer(normal, normal)

    for _iter in range(max(1, int(USER_POSITION_VOLUME_CONSTRAINT_MAX_ITERS))):
        points = _volume_points_array(state)
        current = float(_indexed_tet_mesh_volume_m3(points, tets))
        if (not np.isfinite(current)) or current <= 0.0:
            state.last_position_volume_constraint_status = "bad_iter_volume"
            return
        rel_current = float((current - target) / max(target, 1.0e-30))
        state.last_position_volume_constraint_rel_after = rel_current
        if abs(rel_current) <= trigger:
            state.last_position_volume_constraint_status = "applied"
            return

        tet_points = points[tets]
        pa = tet_points[:, 0, :]
        pb = tet_points[:, 1, :]
        pc = tet_points[:, 2, :]
        pd = tet_points[:, 3, :]
        tet_volumes = np.abs(np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))) / 6.0
        lumped = np.zeros(len(vertices), dtype=float)
        for local in range(4):
            np.add.at(lumped, tets[:, local], 0.25 * tet_volumes)
        solver_lumped = np.asarray(_solver_mass_from_lumped_volume(state.config, lumped), dtype=float)

        gradient = _indexed_tet_volume_gradient(points, tets)
        direction = np.zeros_like(points, dtype=float)
        for idx, vertex in enumerate(vertices):
            if vertex in fixed or solver_lumped[idx] <= 1.0e-30:
                continue
            direction[idx] = (tangent_projector(vertex) @ gradient[idx]) / max(float(solver_lumped[idx]), 1.0e-30)

        derivative = float(np.sum(gradient * direction))
        if (not np.isfinite(derivative)) or abs(derivative) <= 1.0e-30:
            state.last_position_volume_constraint_status = "zero_derivative"
            return
        scale = (target - current) / derivative
        displacement = scale * direction
        disp_norms = np.linalg.norm(displacement, axis=1)
        max_norm = float(np.max(disp_norms)) if disp_norms.size else 0.0
        if max_norm > max_disp:
            displacement *= max_disp / max(max_norm, 1.0e-30)

        movable_vertices: list[object] = []
        targets: list[tuple[float, float, float]] = []
        for idx, vertex in enumerate(vertices):
            if vertex in fixed:
                continue
            target_pos = points[idx] + displacement[idx]
            if np.all(np.isfinite(target_pos)):
                movable_vertices.append(vertex)
                targets.append(tuple(float(x) for x in target_pos))
        if not movable_vertices:
            state.last_position_volume_constraint_status = "no_movable_vertices"
            return

        before = abs(current - target)
        old_positions = points.copy()
        _move_vertices_batch(movable_vertices, targets, state.HC, state.bV_caps)
        _gmsh_axisymmetrize_state(state)
        after_volume = float(_snapshot_msh_volume_m3(state))
        after = abs(after_volume - target)
        state.last_position_volume_constraint_rel_after = float((after_volume - target) / max(target, 1.0e-30))
        if after >= before:
            _move_vertices_batch(
                vertices,
                [tuple(float(x) for x in row) for row in old_positions],
                state.HC,
                state.bV_caps,
            )
            _gmsh_axisymmetrize_state(state)
            state.last_position_volume_constraint_rel_after = rel_current
            state.last_position_volume_constraint_status = "no_improvement"
            return

    state.last_position_volume_constraint_status = "max_iters"
    _axisym_clear_caches(state)


def _gmsh_trial_axisymmetrize_points(state: VolumetricPitoisState, trial_points: np.ndarray) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    specs = list(getattr(state, "gmsh_axisym_ring_specs", []))
    if not specs:
        return

    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    radius = float(state.config.particle_radius)
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))
    for spec in specs:
        spec_vertices = list(getattr(spec, "vertices", []))
        point_indices = [node_id_map.get(id(vertex)) for vertex in spec_vertices]
        if not point_indices or any(idx is None for idx in point_indices):
            continue
        point_indices = [int(idx) for idx in point_indices]
        coords = trial_points[point_indices]
        rel = coords - mid[None, :]
        axial = np.dot(rel, axis)
        radial = rel - np.outer(axial, axis)
        r_mean = float(np.mean(np.linalg.norm(radial, axis=1)))
        cap_id = _gmsh_axisym_uniform_cap_id(spec_vertices)
        if cap_id is None:
            center_coord = mid + float(np.mean(axial)) * axis
        else:
            sphere_center = (
                np.asarray(state.bottom_sphere_center, dtype=float)
                if cap_id == "bottom"
                else np.asarray(state.top_sphere_center, dtype=float)
            )
            sign = 1.0 if cap_id == "bottom" else -1.0
            r_mean = float(np.clip(r_mean, 0.0, radius))
            center_coord = sphere_center + sign * math.sqrt(max(radius * radius - r_mean * r_mean, 0.0)) * axis

        angles = tuple(getattr(spec, "angles", ()))
        if len(point_indices) == 1 or r_mean <= 1.0e-30:
            for point_idx in point_indices:
                trial_points[point_idx] = center_coord
        else:
            for point_idx, angle in zip(point_indices, angles):
                e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
                trial_points[point_idx] = center_coord + r_mean * e_r


def _gmsh_trial_contact_lines_to_spheres(state: VolumetricPitoisState, trial_points: np.ndarray) -> None:
    radius = float(state.config.particle_radius)
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))
    for ring, sphere_center in (
        (state.bottom_contact_ring, np.asarray(state.bottom_sphere_center, dtype=float)),
        (state.top_contact_ring, np.asarray(state.top_sphere_center, dtype=float)),
    ):
        for vertex in ring:
            point_idx = node_id_map.get(id(vertex))
            if point_idx is None:
                continue
            point_idx = int(point_idx)
            rel = trial_points[point_idx] - sphere_center
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm <= 1.0e-30:
                continue
            trial_points[point_idx] = sphere_center + radius * (rel / rel_norm)


def _gmsh_project_trial_points_to_constraints(state: VolumetricPitoisState, trial_points: np.ndarray) -> None:
    # These are the same moving-boundary constraints applied to the live mesh.
    # Running them during volume searches keeps the accepted correction
    # kinematically consistent with the positions that will actually be used.
    for _projection_pass in range(2):
        _gmsh_trial_axisymmetrize_points(state, trial_points)
        _gmsh_trial_contact_lines_to_spheres(state, trial_points)


def _gmsh_set_velocities_from_displacement(
    state: VolumetricPitoisState,
    *,
    reference_positions: np.ndarray,
    dt: float,
) -> None:
    if float(dt) <= 0.0:
        return
    vertices = list(getattr(state, "volume_export_vertices", []))
    points = _volume_points_array(state)
    reference = np.asarray(reference_positions, dtype=float)
    if reference.shape != points.shape or len(vertices) != points.shape[0]:
        return
    velocity = (points - reference) / float(dt)
    if not np.all(np.isfinite(velocity)):
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    for idx, vertex in enumerate(vertices):
        vertex.u = np.asarray(velocity[idx], dtype=float)
    _enforce_no_swirl_velocity_field(state)
    _set_cap_velocities(state)
    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _axisym_clear_caches(state)


def _apply_gmsh_kinematic_volume_constraint(
    state: VolumetricPitoisState,
    *,
    dt: float,
    reference_positions: np.ndarray,
) -> None:
    state.last_position_volume_constraint_status = "not_run"
    state.last_position_volume_constraint_rel_before = 0.0
    state.last_position_volume_constraint_rel_after = 0.0
    state.last_gmsh_cl_volume_slide_status = "not_used_kinematic"
    state.last_gmsh_cl_volume_slide_scale = 1.0
    state.last_gmsh_cl_volume_slide_rel_before = 0.0
    state.last_gmsh_cl_volume_slide_rel_after = 0.0
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
        or (not bool(USER_ENABLE_KINEMATIC_VOLUME_CONSTRAINT))
    ):
        state.last_position_volume_constraint_status = "disabled"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target)) or target <= 0.0:
        state.last_position_volume_constraint_status = "bad_target"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "empty_mesh"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    base_points = _volume_points_array(state)
    current = float(_indexed_tet_mesh_volume_m3(base_points, tets))
    if (not np.isfinite(current)) or current <= 0.0:
        state.last_position_volume_constraint_status = "bad_current"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before
    state.last_gmsh_cl_volume_slide_rel_before = rel_before
    state.last_gmsh_cl_volume_slide_rel_after = rel_before

    tol = max(float(USER_KINEMATIC_VOLUME_CONSTRAINT_REL_TOL), 0.0)
    if abs(rel_before) <= tol:
        state.last_position_volume_constraint_status = "within_trigger"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    projectable_vertices, projectable_base = _snapshot_projectable_vertex_positions(state)
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))
    projectable_indices = [node_id_map.get(id(vertex)) for vertex in projectable_vertices]
    if (not projectable_vertices) or any(idx is None for idx in projectable_indices):
        state.last_position_volume_constraint_status = "no_projectable_vertices"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return
    projectable_indices = [int(idx) for idx in projectable_indices]

    def trial_points_at(scale: float) -> np.ndarray | None:
        try:
            scaled = _axisym_water_volume_scaled_positions(
                state,
                projectable_base,
                meridional_scale=max(float(scale), 1.0e-6),
            )
        except ValueError:
            return None
        trial_points = np.asarray(base_points, dtype=float).copy()
        trial_points[projectable_indices] = scaled
        _gmsh_project_trial_points_to_constraints(state, trial_points)
        if not np.all(np.isfinite(trial_points)):
            return None
        return trial_points

    def add_candidate(scale: float, candidates: list[tuple[float, float]]) -> None:
        if not np.isfinite(scale) or scale <= 0.0:
            return
        trial_points = trial_points_at(float(scale))
        if trial_points is None:
            return
        volume = float(_indexed_tet_mesh_volume_m3(trial_points, tets))
        if np.isfinite(volume) and volume > 0.0:
            candidates.append((float(scale), volume))

    scale_candidates = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        1.00,
        1.02,
        1.05,
        1.10,
        1.20,
        1.40,
        1.70,
        2.00,
        2.50,
        3.00,
        4.00,
        6.00,
        8.00,
    ]
    candidates: list[tuple[float, float]] = []
    for scale in scale_candidates:
        add_candidate(scale, candidates)
    if not candidates:
        state.last_position_volume_constraint_status = "no_valid_trials"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    candidates = sorted(set(candidates), key=lambda item: item[0])
    best_scale, best_volume = min(candidates, key=lambda item: abs(item[1] - target))
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    for left, right in zip(candidates[:-1], candidates[1:]):
        f_left = left[1] - target
        f_right = right[1] - target
        if abs(f_left) <= max(tol * target, 1.0e-30):
            bracket = (left, left)
            break
        if f_left * f_right <= 0.0:
            bracket = (left, right)
            break

    if bracket is not None:
        left, right = bracket
        lo_scale, lo_volume = left
        hi_scale, hi_volume = right
        if lo_scale == hi_scale:
            best_scale, best_volume = lo_scale, lo_volume
        else:
            lo_f = lo_volume - target
            hi_f = hi_volume - target
            for _ in range(max(1, int(USER_KINEMATIC_VOLUME_CONSTRAINT_MAX_ITERS))):
                mid_scale = 0.5 * (lo_scale + hi_scale)
                mid_points = trial_points_at(mid_scale)
                if mid_points is None:
                    break
                mid_volume = float(_indexed_tet_mesh_volume_m3(mid_points, tets))
                if (not np.isfinite(mid_volume)) or mid_volume <= 0.0:
                    break
                if abs(mid_volume - target) < abs(best_volume - target):
                    best_scale, best_volume = mid_scale, mid_volume
                mid_f = mid_volume - target
                if abs(mid_f) <= max(tol * target, 1.0e-30):
                    best_scale, best_volume = mid_scale, mid_volume
                    break
                if lo_f * mid_f <= 0.0:
                    hi_scale, hi_volume, hi_f = mid_scale, mid_volume, mid_f
                else:
                    lo_scale, lo_volume, lo_f = mid_scale, mid_volume, mid_f

    final_points = trial_points_at(best_scale)
    if final_points is None:
        state.last_position_volume_constraint_status = "bad_final_trial"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    _move_vertices_batch(
        vertices,
        [tuple(float(x) for x in row) for row in final_points],
        state.HC,
        state.bV_caps,
    )
    final_volume = float(_snapshot_msh_volume_m3(state))
    rel_after = float((final_volume - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_after = rel_after
    state.last_gmsh_cl_volume_slide_rel_after = rel_after
    if abs(rel_after) <= max(tol, float(USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL)):
        state.last_position_volume_constraint_status = "kinematic_applied"
    elif abs(final_volume - target) < abs(current - target):
        state.last_position_volume_constraint_status = "kinematic_best_effort"
    else:
        state.last_position_volume_constraint_status = "kinematic_no_improvement"

    _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
    _axisym_clear_caches(state)


def _gmsh_local_edge_lengths(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0:
        return np.empty(0, dtype=float)
    lengths = np.full(pts.shape[0], np.inf, dtype=float)
    if tet_idx.size == 0:
        return np.full(pts.shape[0], 1.0e-5, dtype=float)
    tet_idx = tet_idx.reshape((-1, 4))
    valid_tets = (np.min(tet_idx, axis=1) >= 0) & (np.max(tet_idx, axis=1) < pts.shape[0])
    tet_idx = tet_idx[valid_tets]
    for a_pos, b_pos in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ia = tet_idx[:, a_pos]
        ib = tet_idx[:, b_pos]
        edge_lengths = np.linalg.norm(pts[ia] - pts[ib], axis=1)
        valid = np.isfinite(edge_lengths) & (edge_lengths > 1.0e-30)
        if np.any(valid):
            np.minimum.at(lengths, ia[valid], edge_lengths[valid])
            np.minimum.at(lengths, ib[valid], edge_lengths[valid])
    finite = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    fallback = float(np.median(finite)) if finite.size else 1.0e-5
    lengths[~np.isfinite(lengths) | (lengths <= 0.0)] = fallback
    return lengths


def _gmsh_tet_divergence_l2_from_arrays(
    points: np.ndarray,
    tets: np.ndarray,
    velocities: np.ndarray,
) -> float:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    vel = np.asarray(velocities, dtype=float)
    if pts.size == 0 or tet_idx.size == 0 or vel.shape != pts.shape:
        return 0.0
    valid, volumes, grads = _gmsh_tet_metric_arrays(pts, tet_idx)
    if not np.any(valid):
        return 0.0
    tet_valid = tet_idx.reshape((-1, 4))[valid]
    div_u = np.einsum("tij,tij->t", vel[tet_valid], grads[valid])
    finite = np.isfinite(div_u)
    if not np.any(finite):
        return 0.0
    weighted = float(np.sum(volumes[valid][finite] * div_u[finite] * div_u[finite]))
    volume_sum = float(np.sum(volumes[valid][finite]))
    if volume_sum <= 1.0e-30:
        return 0.0
    return float(math.sqrt(max(weighted / volume_sum, 0.0)))


def _gmsh_volume_outer_surface_mask(
    state: VolumetricPitoisState,
    vertices: list,
    fixed: set,
) -> np.ndarray:
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))
    mask = np.zeros(len(vertices), dtype=bool)
    for ring in getattr(state, "outer_rings", []):
        for vertex in ring:
            point_idx = node_id_map.get(id(vertex))
            if point_idx is None:
                continue
            point_idx = int(point_idx)
            if 0 <= point_idx < len(vertices) and vertex not in fixed:
                mask[point_idx] = True
    return mask


def _gmsh_project_vector_to_boundary_tangent(
    state: VolumetricPitoisState,
    vertex,
    vector: np.ndarray,
) -> np.ndarray:
    direction = np.asarray(vector, dtype=float)
    if vertex in state.bottom_contact_ring:
        center = np.asarray(state.bottom_sphere_center, dtype=float)
    elif vertex in state.top_contact_ring:
        center = np.asarray(state.top_sphere_center, dtype=float)
    else:
        return direction
    normal = np.asarray(vertex.x_a[:3], dtype=float) - center
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-30:
        return direction
    normal = normal / normal_norm
    return direction - normal * float(np.dot(direction, normal))


def _gmsh_restore_trial_volume_on_outer_surface(
    state: VolumetricPitoisState,
    trial_points: np.ndarray,
    *,
    vertices: list,
    tets: np.ndarray,
    fixed: set,
    target: float,
    volume_tol: float,
    max_coord: float,
    surface_mask: np.ndarray | None = None,
    delta_reference: np.ndarray | None = None,
    delta_limits: np.ndarray | None = None,
    movable_mask: np.ndarray | None = None,
) -> bool:
    if (not np.isfinite(target)) or target <= 0.0:
        return True
    if surface_mask is None:
        surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    else:
        surface_mask = np.asarray(surface_mask, dtype=bool)
    if not np.any(surface_mask):
        return True

    def exceeds_delta_limits(points: np.ndarray) -> bool:
        if delta_reference is None or delta_limits is None:
            return False
        reference = np.asarray(delta_reference, dtype=float)
        limits = np.asarray(delta_limits, dtype=float)
        if reference.shape != points.shape or limits.shape[0] != points.shape[0]:
            return False
        if movable_mask is None:
            mask = np.ones(points.shape[0], dtype=bool)
        else:
            mask = np.asarray(movable_mask, dtype=bool)
            if mask.shape[0] != points.shape[0]:
                return False
        if not np.any(mask):
            return False
        delta_norm = np.linalg.norm(points[mask] - reference[mask], axis=1)
        return bool(np.any(delta_norm > limits[mask] * (1.0 + 1.0e-12)))

    max_step = min(
        max(
            float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_CORRECTION_M)),
            float(_length_to_compute(state.config, 1.0e-9)),
        ),
        0.05 * max(float(state.config.particle_radius), 1.0e-30),
    )
    for _iter in range(max(1, int(USER_POST_EDIT_CONTINUITY_MAX_ITERS))):
        volume = float(_indexed_tet_mesh_volume_m3(trial_points, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            return True
        rel = float((volume - target) / max(target, 1.0e-30))
        if abs(rel) <= max(volume_tol, 1.0e-12):
            return True

        gradient = _indexed_tet_volume_gradient(trial_points, tets)
        direction = np.zeros_like(trial_points, dtype=float)
        for idx in np.flatnonzero(surface_mask):
            direction[idx] = _gmsh_project_vector_to_boundary_tangent(
                state,
                vertices[int(idx)],
                gradient[int(idx)],
            )
        derivative = float(np.sum(gradient * direction))
        if (not np.isfinite(derivative)) or abs(derivative) <= 1.0e-30:
            return True

        step = ((target - volume) / derivative) * direction
        if not np.all(np.isfinite(step)):
            return True
        step_norm = np.linalg.norm(step[surface_mask], axis=1)
        largest_step = float(np.max(step_norm)) if step_norm.size else 0.0
        if largest_step <= 1.0e-30:
            return True
        if largest_step > max_step:
            step *= max_step / max(largest_step, 1.0e-30)

        old_points = trial_points.copy()
        old_sign = math.copysign(1.0, volume - target) if volume != target else 0.0
        old_error = abs(volume - target)
        accepted = False
        step_scale = 1.0
        for _backtrack in range(12):
            candidate_points = old_points + step_scale * step
            _gmsh_project_trial_points_to_constraints(state, candidate_points)
            if (
                np.all(np.isfinite(candidate_points))
                and float(np.max(np.abs(candidate_points))) <= max_coord
            ):
                candidate_volume = float(_indexed_tet_mesh_volume_m3(candidate_points, tets))
                if np.isfinite(candidate_volume) and candidate_volume > 0.0:
                    candidate_error = abs(candidate_volume - target)
                    if candidate_error < old_error:
                        trial_points[:] = candidate_points
                        accepted = True
                        break
            step_scale *= 0.5
        if not accepted:
            return True
        new_sign = math.copysign(1.0, candidate_volume - target) if candidate_volume != target else 0.0
        # A still-same-side restore that has already exceeded the final bounded
        # displacement test cannot become accepted by continuing farther toward
        # the target volume.
        if (
            old_sign != 0.0
            and new_sign == old_sign
            and candidate_error > max(volume_tol * target, 1.0e-30)
            and exceeds_delta_limits(trial_points)
        ):
            trial_points[:] = old_points
            return False
    return True


def _gmsh_restore_trial_volume_by_axisymmetric_radial_scale(
    state: VolumetricPitoisState,
    trial_points: np.ndarray,
    *,
    vertices: list,
    tets: np.ndarray,
    fixed: set,
    target: float,
    max_coord: float,
    surface_mask: np.ndarray | None = None,
) -> bool:
    if (not np.isfinite(target)) or target <= 0.0:
        return False
    if surface_mask is None:
        surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    else:
        surface_mask = np.asarray(surface_mask, dtype=bool)
    if surface_mask.shape[0] != trial_points.shape[0] or not np.any(surface_mask):
        return False

    base_points = np.asarray(trial_points, dtype=float).copy()
    if (not np.all(np.isfinite(base_points))) or float(np.max(np.abs(base_points))) > max_coord:
        return False
    base_volume = float(_indexed_tet_mesh_volume_m3(base_points, tets))
    if (not np.isfinite(base_volume)) or base_volume <= 0.0:
        return False

    axis_origin, axis = _swirl_axis_geometry(state)
    axis = np.asarray(axis, dtype=float)
    axis_origin = np.asarray(axis_origin, dtype=float)
    if (
        (not np.all(np.isfinite(axis_origin)))
        or (not np.all(np.isfinite(axis)))
        or float(np.max(np.abs(axis_origin))) > max_coord
        or float(np.linalg.norm(axis)) <= 1.0e-30
    ):
        return False
    rel = base_points - axis_origin
    if not np.all(np.isfinite(rel)):
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        axial = rel @ axis
    if not np.all(np.isfinite(axial)):
        return False
    radial = rel - axial[:, None] * axis[None, :]
    if not np.all(np.isfinite(radial)):
        return False
    radial_norm = np.linalg.norm(radial, axis=1)
    movable = surface_mask & (radial_norm > 1.0e-30)
    if not np.any(movable):
        return False

    def candidate_for_scale(scale: float) -> tuple[np.ndarray | None, float]:
        if (not np.isfinite(scale)) or scale <= 0.0:
            return None, float("inf")
        candidate = base_points.copy()
        candidate[movable] = base_points[movable] + (float(scale) - 1.0) * radial[movable]
        _gmsh_project_trial_points_to_constraints(state, candidate)
        if (not np.all(np.isfinite(candidate))) or float(np.max(np.abs(candidate))) > max_coord:
            return None, float("inf")
        volume = float(_indexed_tet_mesh_volume_m3(candidate, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            return None, float("inf")
        return candidate, volume

    base_error = abs(base_volume - target)
    best_points: np.ndarray | None = None
    best_volume = base_volume
    best_error = base_error
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    base_sign = math.copysign(1.0, base_volume - target) if base_volume != target else 0.0

    trial_scales = (
        0.80,
        0.85,
        0.90,
        0.94,
        0.97,
        0.985,
        0.995,
        1.005,
        1.015,
        1.03,
        1.06,
        1.10,
        1.15,
        1.20,
    )
    evaluated: list[tuple[float, float]] = [(1.0, base_volume)]
    for scale in trial_scales:
        points, volume = candidate_for_scale(float(scale))
        if points is None:
            continue
        error = abs(volume - target)
        evaluated.append((float(scale), float(volume)))
        if error < best_error:
            best_points = points
            best_volume = float(volume)
            best_error = error
        sign = math.copysign(1.0, volume - target) if volume != target else 0.0
        if base_sign == 0.0 or sign == 0.0 or sign != base_sign:
            bracket = ((1.0, base_volume), (float(scale), float(volume)))
            break

    if bracket is None and evaluated:
        evaluated.sort(key=lambda item: item[0])
        for left, right in zip(evaluated[:-1], evaluated[1:]):
            left_sign = math.copysign(1.0, left[1] - target) if left[1] != target else 0.0
            right_sign = math.copysign(1.0, right[1] - target) if right[1] != target else 0.0
            if left_sign == 0.0 or right_sign == 0.0 or left_sign != right_sign:
                bracket = (left, right)
                break

    if bracket is not None:
        lo, hi = bracket
        for _root_iter in range(40):
            mid_scale = 0.5 * (lo[0] + hi[0])
            points, volume = candidate_for_scale(mid_scale)
            if points is None:
                break
            error = abs(volume - target)
            if error < best_error:
                best_points = points
                best_volume = float(volume)
                best_error = error
            mid_sign = math.copysign(1.0, volume - target) if volume != target else 0.0
            lo_sign = math.copysign(1.0, lo[1] - target) if lo[1] != target else 0.0
            if mid_sign == 0.0 or error <= max(1.0e-10 * target, 1.0e-30):
                break
            if lo_sign == 0.0 or mid_sign == lo_sign:
                lo = (float(mid_scale), float(volume))
            else:
                hi = (float(mid_scale), float(volume))

    if best_points is None or best_error >= base_error:
        return False
    trial_points[:] = best_points
    return abs(best_volume - target) < base_error


def _gmsh_restore_trial_volume_by_axisymmetric_meridional_scale(
    state: VolumetricPitoisState,
    trial_points: np.ndarray,
    *,
    vertices: list,
    tets: np.ndarray,
    fixed: set,
    target: float,
    max_coord: float,
    surface_mask: np.ndarray | None = None,
) -> bool:
    if (not np.isfinite(target)) or target <= 0.0:
        return False
    if surface_mask is None:
        surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    else:
        surface_mask = np.asarray(surface_mask, dtype=bool)
    if surface_mask.shape[0] != trial_points.shape[0] or not np.any(surface_mask):
        return False

    base_points = np.asarray(trial_points, dtype=float).copy()
    if (not np.all(np.isfinite(base_points))) or float(np.max(np.abs(base_points))) > max_coord:
        return False
    base_volume = float(_indexed_tet_mesh_volume_m3(base_points, tets))
    if (not np.isfinite(base_volume)) or base_volume <= 0.0:
        return False

    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    mid = 0.5 * (bottom_center + top_center)
    if (
        (not np.all(np.isfinite(bottom_center)))
        or (not np.all(np.isfinite(top_center)))
        or (not np.all(np.isfinite(mid)))
        or float(np.max(np.abs(mid))) > max_coord
        or (not np.isfinite(axis_norm))
        or axis_norm <= 1.0e-30
    ):
        return False
    axis = axis / axis_norm
    rel = base_points - mid[None, :]
    if not np.all(np.isfinite(rel)):
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        axial_mag = rel @ axis
    if not np.all(np.isfinite(axial_mag)):
        return False
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    if not np.all(np.isfinite(radial)):
        return False
    movable = surface_mask.copy()
    if not np.any(movable):
        return False

    def candidate_for_scale(scale: float) -> tuple[np.ndarray | None, float]:
        if (not np.isfinite(scale)) or scale <= 0.0:
            return None, float("inf")
        lam = max(float(scale), 1.0e-6)
        candidate = base_points.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            candidate[movable] = mid[None, :] + (lam * lam) * axial[movable] + lam * radial[movable]
        _gmsh_project_trial_points_to_constraints(state, candidate)
        if (not np.all(np.isfinite(candidate))) or float(np.max(np.abs(candidate))) > max_coord:
            return None, float("inf")
        volume = float(_indexed_tet_mesh_volume_m3(candidate, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            return None, float("inf")
        return candidate, volume

    base_error = abs(base_volume - target)
    best_points: np.ndarray | None = None
    best_volume = base_volume
    best_error = base_error
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    base_sign = math.copysign(1.0, base_volume - target) if base_volume != target else 0.0

    trial_scales = (
        0.90,
        0.94,
        0.97,
        0.985,
        0.995,
        0.998,
        1.002,
        1.005,
        1.015,
        1.03,
        1.06,
        1.10,
    )
    evaluated: list[tuple[float, float]] = [(1.0, base_volume)]
    for scale in trial_scales:
        points, volume = candidate_for_scale(float(scale))
        if points is None:
            continue
        error = abs(volume - target)
        evaluated.append((float(scale), float(volume)))
        if error < best_error:
            best_points = points
            best_volume = float(volume)
            best_error = error
        sign = math.copysign(1.0, volume - target) if volume != target else 0.0
        if base_sign == 0.0 or sign == 0.0 or sign != base_sign:
            bracket = ((1.0, base_volume), (float(scale), float(volume)))
            break

    if bracket is None and evaluated:
        evaluated.sort(key=lambda item: item[0])
        for left, right in zip(evaluated[:-1], evaluated[1:]):
            left_sign = math.copysign(1.0, left[1] - target) if left[1] != target else 0.0
            right_sign = math.copysign(1.0, right[1] - target) if right[1] != target else 0.0
            if left_sign == 0.0 or right_sign == 0.0 or left_sign != right_sign:
                bracket = (left, right)
                break

    if bracket is not None:
        lo, hi = bracket
        for _root_iter in range(40):
            mid_scale = 0.5 * (lo[0] + hi[0])
            points, volume = candidate_for_scale(mid_scale)
            if points is None:
                break
            error = abs(volume - target)
            if error < best_error:
                best_points = points
                best_volume = float(volume)
                best_error = error
            mid_sign = math.copysign(1.0, volume - target) if volume != target else 0.0
            lo_sign = math.copysign(1.0, lo[1] - target) if lo[1] != target else 0.0
            if mid_sign == 0.0 or error <= max(1.0e-10 * target, 1.0e-30):
                break
            if lo_sign == 0.0 or mid_sign == lo_sign:
                lo = (float(mid_scale), float(volume))
            else:
                hi = (float(mid_scale), float(volume))

    if best_points is None or best_error >= base_error:
        return False
    trial_points[:] = best_points
    return abs(best_volume - target) < base_error


def _gmsh_restore_trial_volume_by_axisymmetric_projectable_scale(
    state: VolumetricPitoisState,
    trial_points: np.ndarray,
    *,
    vertices: list,
    tets: np.ndarray,
    fixed: set | None = None,
    target: float,
    max_coord: float,
) -> bool:
    if (not np.isfinite(target)) or target <= 0.0:
        return False

    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))
    projectable_vertices, _projectable_positions = _snapshot_projectable_vertex_positions(state)
    fixed_vertices = set() if fixed is None else set(fixed)
    projectable_vertices = [vertex for vertex in projectable_vertices if vertex not in fixed_vertices]
    projectable_indices = [node_id_map.get(id(vertex)) for vertex in projectable_vertices]
    if (not projectable_vertices) or any(idx is None for idx in projectable_indices):
        return False
    projectable_indices = np.asarray([int(idx) for idx in projectable_indices], dtype=int)
    fixed_indices = np.asarray(
        [
            int(idx)
            for vertex in fixed_vertices
            for idx in [node_id_map.get(id(vertex))]
            if idx is not None
        ],
        dtype=int,
    )

    base_points = np.asarray(trial_points, dtype=float).copy()
    if (not np.all(np.isfinite(base_points))) or float(np.max(np.abs(base_points))) > max_coord:
        return False
    base_volume = float(_indexed_tet_mesh_volume_m3(base_points, tets))
    if (not np.isfinite(base_volume)) or base_volume <= 0.0:
        return False

    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    mid = 0.5 * (bottom_center + top_center)
    if (
        (not np.all(np.isfinite(bottom_center)))
        or (not np.all(np.isfinite(top_center)))
        or (not np.all(np.isfinite(mid)))
        or float(np.max(np.abs(mid))) > max_coord
        or (not np.isfinite(axis_norm))
        or axis_norm <= 1.0e-30
    ):
        return False
    axis = axis / axis_norm
    selected = base_points[projectable_indices]
    rel = selected - mid[None, :]
    if not np.all(np.isfinite(rel)):
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        axial_mag = rel @ axis
    if not np.all(np.isfinite(axial_mag)):
        return False
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    if not np.all(np.isfinite(radial)):
        return False

    def candidate_for_scale(scale: float) -> tuple[np.ndarray | None, float]:
        if (not np.isfinite(scale)) or scale <= 0.0:
            return None, float("inf")
        lam = max(float(scale), 1.0e-6)
        candidate = base_points.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            candidate[projectable_indices] = mid[None, :] + (lam * lam) * axial + lam * radial
        _gmsh_project_trial_points_to_constraints(state, candidate)
        if fixed_indices.size:
            candidate[fixed_indices] = base_points[fixed_indices]
        if (not np.all(np.isfinite(candidate))) or float(np.max(np.abs(candidate))) > max_coord:
            return None, float("inf")
        volume = float(_indexed_tet_mesh_volume_m3(candidate, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            return None, float("inf")
        return candidate, volume

    base_error = abs(base_volume - target)
    best_points: np.ndarray | None = None
    best_volume = base_volume
    best_error = base_error
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    base_sign = math.copysign(1.0, base_volume - target) if base_volume != target else 0.0

    trial_scales = (
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.60,
        0.70,
        0.80,
        0.88,
        0.94,
        0.97,
        0.985,
        0.995,
        0.999,
        1.001,
        1.005,
        1.015,
        1.03,
        1.06,
        1.12,
        1.25,
        1.50,
        2.00,
        3.00,
        4.00,
    )
    evaluated: list[tuple[float, float]] = [(1.0, base_volume)]
    for scale in trial_scales:
        points, volume = candidate_for_scale(float(scale))
        if points is None:
            continue
        error = abs(volume - target)
        evaluated.append((float(scale), float(volume)))
        if error < best_error:
            best_points = points
            best_volume = float(volume)
            best_error = error
        sign = math.copysign(1.0, volume - target) if volume != target else 0.0
        if base_sign == 0.0 or sign == 0.0 or sign != base_sign:
            bracket = ((1.0, base_volume), (float(scale), float(volume)))
            break

    if bracket is None and evaluated:
        evaluated.sort(key=lambda item: item[0])
        for left, right in zip(evaluated[:-1], evaluated[1:]):
            left_sign = math.copysign(1.0, left[1] - target) if left[1] != target else 0.0
            right_sign = math.copysign(1.0, right[1] - target) if right[1] != target else 0.0
            if left_sign == 0.0 or right_sign == 0.0 or left_sign != right_sign:
                bracket = (left, right)
                break

    if bracket is not None:
        lo, hi = bracket
        for _root_iter in range(42):
            mid_scale = 0.5 * (lo[0] + hi[0])
            points, volume = candidate_for_scale(mid_scale)
            if points is None:
                break
            error = abs(volume - target)
            if error < best_error:
                best_points = points
                best_volume = float(volume)
                best_error = error
            mid_sign = math.copysign(1.0, volume - target) if volume != target else 0.0
            lo_sign = math.copysign(1.0, lo[1] - target) if lo[1] != target else 0.0
            if mid_sign == 0.0 or error <= max(1.0e-10 * target, 1.0e-30):
                break
            if lo_sign == 0.0 or mid_sign == lo_sign:
                lo = (float(mid_scale), float(volume))
            else:
                hi = (float(mid_scale), float(volume))

    if best_points is None or best_error >= base_error:
        return False
    trial_points[:] = best_points
    return abs(best_volume - target) < base_error


def _gmsh_restore_trial_volume_by_axisymmetric_all_free_scale(
    state: VolumetricPitoisState,
    trial_points: np.ndarray,
    *,
    vertices: list,
    tets: np.ndarray,
    fixed: set,
    target: float,
    max_coord: float,
) -> bool:
    if (not np.isfinite(target)) or target <= 0.0:
        return False

    base_points = np.asarray(trial_points, dtype=float).copy()
    if (not np.all(np.isfinite(base_points))) or float(np.max(np.abs(base_points))) > max_coord:
        return False
    base_volume = float(_indexed_tet_mesh_volume_m3(base_points, tets))
    if (not np.isfinite(base_volume)) or base_volume <= 0.0:
        return False

    movable_indices = np.asarray(
        [idx for idx, vertex in enumerate(vertices) if vertex not in fixed],
        dtype=int,
    )
    if movable_indices.size == 0:
        return False

    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    mid = 0.5 * (bottom_center + top_center)
    if (
        (not np.all(np.isfinite(bottom_center)))
        or (not np.all(np.isfinite(top_center)))
        or (not np.all(np.isfinite(mid)))
        or float(np.max(np.abs(mid))) > max_coord
        or (not np.isfinite(axis_norm))
        or axis_norm <= 1.0e-30
    ):
        return False
    axis = axis / axis_norm

    selected = base_points[movable_indices]
    rel = selected - mid[None, :]
    if not np.all(np.isfinite(rel)):
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        axial_mag = rel @ axis
    if not np.all(np.isfinite(axial_mag)):
        return False
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    if not np.all(np.isfinite(radial)):
        return False

    def candidate_for_scale(scale: float) -> tuple[np.ndarray | None, float]:
        if (not np.isfinite(scale)) or scale <= 0.0:
            return None, float("inf")
        lam = max(float(scale), 1.0e-6)
        candidate = base_points.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            candidate[movable_indices] = mid[None, :] + (lam * lam) * axial + lam * radial
        _gmsh_project_trial_points_to_constraints(state, candidate)
        if (not np.all(np.isfinite(candidate))) or float(np.max(np.abs(candidate))) > max_coord:
            return None, float("inf")
        volume = float(_indexed_tet_mesh_volume_m3(candidate, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            return None, float("inf")
        return candidate, volume

    base_error = abs(base_volume - target)
    best_points: np.ndarray | None = None
    best_volume = base_volume
    best_error = base_error
    bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
    base_sign = math.copysign(1.0, base_volume - target) if base_volume != target else 0.0

    trial_scales = (
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.78,
        0.88,
        0.94,
        0.97,
        0.985,
        0.995,
        0.999,
        1.001,
        1.005,
        1.015,
        1.03,
        1.06,
        1.12,
        1.25,
        1.45,
        1.75,
        2.20,
        3.00,
        4.50,
    )
    evaluated: list[tuple[float, float]] = [(1.0, base_volume)]
    for scale in trial_scales:
        points, volume = candidate_for_scale(float(scale))
        if points is None:
            continue
        error = abs(volume - target)
        evaluated.append((float(scale), float(volume)))
        if error < best_error:
            best_points = points
            best_volume = float(volume)
            best_error = error
        sign = math.copysign(1.0, volume - target) if volume != target else 0.0
        if base_sign == 0.0 or sign == 0.0 or sign != base_sign:
            bracket = ((1.0, base_volume), (float(scale), float(volume)))
            break

    if bracket is None and evaluated:
        evaluated.sort(key=lambda item: item[0])
        for left, right in zip(evaluated[:-1], evaluated[1:]):
            left_sign = math.copysign(1.0, left[1] - target) if left[1] != target else 0.0
            right_sign = math.copysign(1.0, right[1] - target) if right[1] != target else 0.0
            if left_sign == 0.0 or right_sign == 0.0 or left_sign != right_sign:
                bracket = (left, right)
                break

    if bracket is not None:
        lo, hi = bracket
        for _root_iter in range(44):
            mid_scale = 0.5 * (lo[0] + hi[0])
            points, volume = candidate_for_scale(mid_scale)
            if points is None:
                break
            error = abs(volume - target)
            if error < best_error:
                best_points = points
                best_volume = float(volume)
                best_error = error
            mid_sign = math.copysign(1.0, volume - target) if volume != target else 0.0
            lo_sign = math.copysign(1.0, lo[1] - target) if lo[1] != target else 0.0
            if mid_sign == 0.0 or error <= max(1.0e-10 * target, 1.0e-30):
                break
            if lo_sign == 0.0 or mid_sign == lo_sign:
                lo = (float(mid_scale), float(volume))
            else:
                hi = (float(mid_scale), float(volume))

    if best_points is None or best_error >= base_error:
        return False
    trial_points[:] = best_points
    return abs(best_volume - target) < base_error


def _apply_gmsh_post_edit_continuity_projection(
    state: VolumetricPitoisState,
    *,
    dt: float,
    reference_positions: np.ndarray,
) -> None:
    state.last_position_volume_constraint_status = "not_run"
    state.last_position_volume_constraint_rel_before = 0.0
    state.last_position_volume_constraint_rel_after = 0.0
    state.last_gmsh_cl_volume_slide_status = "not_used_continuity"
    state.last_gmsh_cl_volume_slide_scale = 1.0
    state.last_gmsh_cl_volume_slide_rel_before = 0.0
    state.last_gmsh_cl_volume_slide_rel_after = 0.0
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
        or (not bool(USER_ENABLE_POST_EDIT_CONTINUITY_PROJECTION))
        or float(dt) <= 0.0
    ):
        state.last_position_volume_constraint_status = "disabled"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return
    if coo_matrix is None or diags is None or spsolve is None:
        raise RuntimeError("USER_ENABLE_POST_EDIT_CONTINUITY_PROJECTION requires scipy.sparse.")

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "empty_mesh"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    reference = np.asarray(reference_positions, dtype=float)
    current_points = _volume_points_array(state)
    if reference.shape != current_points.shape:
        state.last_position_volume_constraint_status = "bad_reference"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return
    if (not np.all(np.isfinite(reference))) or (not np.all(np.isfinite(current_points))):
        state.last_position_volume_constraint_status = "bad_points"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    current_volume = float(_indexed_tet_mesh_volume_m3(current_points, tets))
    rel_before = 0.0
    if np.isfinite(target) and target > 0.0 and np.isfinite(current_volume):
        rel_before = float((current_volume - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before
    state.last_gmsh_cl_volume_slide_rel_before = rel_before
    state.last_gmsh_cl_volume_slide_rel_after = rel_before

    n_vertices = len(vertices)
    n_dofs = 3 * n_vertices
    raw_velocity = (current_points - reference) / float(dt)
    raw_div_l2 = _gmsh_tet_divergence_l2_from_arrays(current_points, tets, raw_velocity)

    valid_tets, tet_volumes, tet_grads = _gmsh_tet_metric_arrays(current_points, tets)
    if not np.any(valid_tets):
        state.last_position_volume_constraint_status = "empty_tet_cache"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return
    tet_idx_all = tets.reshape((-1, 4))
    valid_cell_ids = np.flatnonzero(valid_tets).astype(int, copy=False)
    valid_idx = tet_idx_all[valid_tets]
    valid_volumes = tet_volumes[valid_tets]
    valid_grads = tet_grads[valid_tets]
    lumped = np.zeros(n_vertices, dtype=float)
    np.add.at(lumped, valid_idx.ravel(), np.repeat(0.25 * valid_volumes, 4))
    closure = str(getattr(state.config, "pressure_closure", "incompressible")).strip().lower()
    if closure not in {"compressible", "incompressible"}:
        return

    dof_offsets = np.arange(3, dtype=int)
    b_cols = (3 * valid_idx[:, :, None] + dof_offsets[None, None, :]).reshape(-1)
    b_rows = np.repeat(valid_cell_ids, 12)
    b_data = (valid_volumes[:, None, None] * valid_grads).reshape(-1)

    def tangent_projector(vertex) -> np.ndarray:
        if vertex in state.bottom_contact_ring:
            center = np.asarray(state.bottom_sphere_center, dtype=float)
        elif vertex in state.top_contact_ring:
            center = np.asarray(state.top_sphere_center, dtype=float)
        else:
            return np.eye(3, dtype=float)
        normal = np.asarray(vertex.x_a[:3], dtype=float) - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            return np.eye(3, dtype=float)
        normal = normal / normal_norm
        return np.eye(3, dtype=float) - np.outer(normal, normal)

    fixed = set(state.bV_caps)
    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[float] = []
    solver_lumped = np.asarray(_solver_mass_from_lumped_volume(state.config, lumped), dtype=float)
    for idx, vertex in enumerate(vertices):
        if vertex in fixed or solver_lumped[idx] <= 1.0e-30:
            continue
        inv_mass = 1.0 / max(float(solver_lumped[idx]), 1.0e-30)
        projector = tangent_projector(vertex)
        base = 3 * idx
        for a in range(3):
            for b in range(3):
                value = inv_mass * float(projector[a, b])
                if abs(value) > 0.0:
                    w_rows.append(base + a)
                    w_cols.append(base + b)
                    w_data.append(value)

    g_matrix = coo_matrix((g_data, (g_rows, g_cols)), shape=(n_dofs, int(tets.shape[0]))).tocsr()
    mobility = coo_matrix((w_data, (w_rows, w_cols)), shape=(n_dofs, n_dofs)).tocsr()
    raw_vec = raw_velocity.reshape(n_dofs)
    rhs = np.asarray(g_matrix.T @ raw_vec, dtype=float)
    matrix = (g_matrix.T @ mobility @ g_matrix).tocsr()
    diag_mean = float(np.mean(np.abs(matrix.diagonal()))) if n_vertices else 1.0
    diag_mean = max(diag_mean, 1.0e-30)
    damping = max(float(getattr(state.config, "incompressible_projection_regularization", 0.0)), 0.0)
    if damping > 0.0:
        matrix = matrix + diags(
            np.full(int(tets.shape[0]), damping * diag_mean, dtype=float),
            0,
            shape=(int(tets.shape[0]), int(tets.shape[0])),
        )
    pressure = np.asarray(spsolve(matrix, rhs), dtype=float)
    if pressure.size != int(tets.shape[0]) or not np.all(np.isfinite(pressure)):
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
    correction = np.asarray(mobility @ (g_matrix @ pressure), dtype=float).reshape((n_vertices, 3))
    if not np.all(np.isfinite(correction)):
        correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)

    edge_lengths = _gmsh_local_edge_lengths(current_points, tets)
    correction_limits = np.minimum(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_CORRECTION_M)),
        max(float(USER_POST_EDIT_CONTINUITY_EDGE_FRACTION), 0.0) * edge_lengths,
    )
    correction_limits = np.maximum(correction_limits, float(_length_to_compute(state.config, 1.0e-9)))
    max_coord = max(
        float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
        float(_length_to_compute(state.config, 1.0e-6)),
    )
    volume_tol = max(float(USER_POST_EDIT_CONTINUITY_REL_TOL), 0.0)
    movable = np.array([vertex not in fixed for vertex in vertices], dtype=bool)
    surface_mask = _gmsh_volume_outer_surface_mask(state, vertices, fixed)
    candidate_cache: dict[float, tuple[float, float, float, np.ndarray] | None] = {}

    def candidate(alpha: float) -> tuple[float, float, float, np.ndarray] | None:
        if not np.isfinite(alpha):
            return None
        alpha = float(np.clip(alpha, 0.0, float(USER_POST_EDIT_CONTINUITY_MAX_ALPHA)))
        cached = candidate_cache.get(alpha)
        if cached is not None or alpha in candidate_cache:
            return cached
        trial_points = reference + float(dt) * (raw_velocity - alpha * correction)
        _gmsh_project_trial_points_to_constraints(state, trial_points)
        restore_ok = _gmsh_restore_trial_volume_on_outer_surface(
            state,
            trial_points,
            vertices=vertices,
            tets=tets,
            fixed=fixed,
            target=target,
            volume_tol=volume_tol,
            max_coord=max_coord,
            surface_mask=surface_mask,
            delta_reference=current_points,
            delta_limits=correction_limits,
            movable_mask=movable,
        )
        if not restore_ok:
            candidate_cache[alpha] = None
            return None
        if (not np.all(np.isfinite(trial_points))) or float(np.max(np.abs(trial_points))) > max_coord:
            candidate_cache[alpha] = None
            return None
        delta = trial_points - current_points
        delta_norm = np.linalg.norm(delta, axis=1)
        if np.any(delta_norm[movable] > correction_limits[movable]):
            candidate_cache[alpha] = None
            return None
        volume = float(_indexed_tet_mesh_volume_m3(trial_points, tets))
        if (not np.isfinite(volume)) or volume <= 0.0:
            candidate_cache[alpha] = None
            return None
        velocity = (trial_points - reference) / float(dt)
        div_l2 = _gmsh_tet_divergence_l2_from_arrays(trial_points, tets, velocity)
        rel = float((volume - target) / max(target, 1.0e-30)) if target > 0.0 else 0.0
        result = (volume, rel, div_l2, trial_points)
        candidate_cache[alpha] = result
        return result

    alpha_max = max(float(USER_POST_EDIT_CONTINUITY_MAX_ALPHA), 0.0)
    correction_norm = np.linalg.norm(correction, axis=1)
    alpha_bound_mask = movable & np.isfinite(correction_norm) & (correction_norm > 1.0e-30)
    if np.any(alpha_bound_mask):
        local_alpha_limit = correction_limits[alpha_bound_mask] / (
            max(float(dt), 1.0e-30) * correction_norm[alpha_bound_mask]
        )
        finite_alpha_limit = local_alpha_limit[np.isfinite(local_alpha_limit)]
        if finite_alpha_limit.size:
            alpha_max = min(alpha_max, max(0.0, 0.98 * float(np.min(finite_alpha_limit))))
    alpha_values = [0.0]
    if alpha_max > 1.0e-14:
        alpha_values.extend(float(value) for value in np.linspace(0.0, alpha_max, 9)[1:])
        for value in (0.25, 0.5, 0.75, 1.0):
            if value <= alpha_max:
                alpha_values.append(float(value))
    alpha_values = sorted({float(value) for value in alpha_values if np.isfinite(value)})
    candidates: list[tuple[float, float, float, float, np.ndarray]] = []
    for alpha in alpha_values:
        item = candidate(alpha)
        if item is None:
            continue
        volume, rel, div_l2, trial_points = item
        candidates.append((float(alpha), volume, rel, div_l2, trial_points))

    for left_alpha, right_alpha in zip(alpha_values[:-1], alpha_values[1:]):
        left = candidate(left_alpha)
        right = candidate(right_alpha)
        if left is None or right is None or target <= 0.0:
            continue
        left_f = left[0] - target
        right_f = right[0] - target
        if left_f * right_f > 0.0:
            continue
        lo = float(left_alpha)
        hi = float(right_alpha)
        lo_f = float(left_f)
        for _ in range(max(1, int(USER_POST_EDIT_CONTINUITY_MAX_ITERS))):
            mid = 0.5 * (lo + hi)
            mid_item = candidate(mid)
            if mid_item is None:
                hi = mid
                continue
            volume, rel, div_l2, trial_points = mid_item
            candidates.append((mid, volume, rel, div_l2, trial_points))
            mid_f = volume - target
            if abs(rel) <= volume_tol:
                break
            if lo_f * mid_f <= 0.0:
                hi = mid
            else:
                lo = mid
                lo_f = mid_f
        break

    if not candidates:
        state.last_position_volume_constraint_status = "continuity_no_valid_bounded_candidate"
        _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
        return

    if target > 0.0:
        volume_safe_candidates = [
            item for item in candidates if abs(float(item[2])) <= max(volume_tol, 1.0e-12)
        ]
        if volume_safe_candidates:
            best = min(volume_safe_candidates, key=lambda item: item[3])
        else:
            # If exact volume and exact continuity cannot both be achieved
            # under the displacement bounds, prefer the least-divergent bounded
            # move and use volume error only as a secondary criterion.
            best = min(candidates, key=lambda item: (item[3], abs(item[2])))
    else:
        best = min(candidates, key=lambda item: item[3])
    best_alpha, best_volume, best_rel, best_div_l2, best_points = best

    _move_vertices_batch(
        vertices,
        [tuple(float(x) for x in row) for row in best_points],
        state.HC,
        state.bV_caps,
    )
    final_volume = float(_snapshot_msh_volume_m3(state))
    final_rel = float((final_volume - target) / max(target, 1.0e-30)) if target > 0.0 else 0.0
    state.last_position_volume_constraint_rel_after = final_rel
    state.last_gmsh_cl_volume_slide_rel_after = final_rel
    _gmsh_set_velocities_from_displacement(state, reference_positions=reference_positions, dt=dt)
    state.incompressible_divergence_before_l2 = float(raw_div_l2)
    state.incompressible_divergence_after_l2 = float(best_div_l2)
    state.incompressible_projection_alpha = float(best_alpha)
    if abs(final_rel) <= volume_tol:
        state.last_position_volume_constraint_status = "continuity_applied"
    elif abs(final_rel) < abs(rel_before):
        state.last_position_volume_constraint_status = "continuity_best_effort"
    else:
        state.last_position_volume_constraint_status = "continuity_no_volume_improvement"
    _axisym_clear_caches(state)


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_projected_pressure: bool = True,
    include_contact_line: bool = True,
    include_damping: bool = True,
) -> np.ndarray:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        include_projected_pressure = False
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
            return _BASE_Ftot(
                v,
                state=state,
                pressure_model=pressure_model,
                include_projected_pressure=include_projected_pressure,
                include_contact_line=include_contact_line,
                include_damping=include_damping,
            )
        force_map = _gmsh_axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
        if id(v) in force_map:
            return np.asarray(force_map[id(v)], dtype=float)
        return _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
    if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
        return _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
    force_map = _axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    return np.asarray(force_map.get(id(v), np.zeros(3, dtype=float)), dtype=float)


def _vertex_acceleration(v, *, state: VolumetricPitoisState) -> np.ndarray:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if v in state.bV_caps:
            return np.zeros(3, dtype=float)
        if bool(USER_ENFORCE_FULL_AXISYMMETRY):
            return np.asarray(_gmsh_axisym_accel_map(state).get(id(v), np.zeros(3, dtype=float)), dtype=float)
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        F_nonproj = _Ftot(v, state=state, include_projected_pressure=False)
        accel = F_nonproj / mass
        accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enable_volume_projection", False)):
            F_total = _Ftot(v, state=state, include_projected_pressure=True)
            accel += (F_total - F_nonproj) / mass
            accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enforce_no_swirl", False)):
            axis_origin, axis = _swirl_axis_geometry(state)
            accel = _remove_swirl_component(
                accel,
                point=np.asarray(v.x_a[:3], dtype=float),
                axis_origin=axis_origin,
                axis=axis,
            )
        return accel
    if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        F_nonproj = _Ftot(v, state=state, include_projected_pressure=False)
        accel = F_nonproj / mass
        accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enable_volume_projection", False)):
            accel += _Fp_proj(v, state=state) / mass
            accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enforce_no_swirl", False)):
            axis_origin, axis = _swirl_axis_geometry(state)
            accel = _remove_swirl_component(
                accel,
                point=np.asarray(v.x_a[:3], dtype=float),
                axis_origin=axis_origin,
                axis=axis,
            )
        return accel

    cache = getattr(state, "_axisym_accel_cache", None)
    if cache is None:
        cache = {}
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
        force_map_nonproj = _axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=False,
            include_contact_line=True,
            include_damping=True,
        )
        force_map_total = _axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=True,
            include_contact_line=True,
            include_damping=True,
        )
        mid, axis, e1, e2 = _axisym_basis(state)
        covered: set[int] = set()
        for k, rings in enumerate(state.layer_rings):
            center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
            for ring in rings:
                members = [node for node in ring if (node not in state.bV_caps) and (id(node) in force_map_nonproj)]
                if not members:
                    continue
                radial_units = _axisym_ring_radial_units(
                    members,
                    center_coord=center_coord,
                    axis=axis,
                    e1=e1,
                    e2=e2,
                )
                mass_avg = float(np.mean([max(float(getattr(node, "m", 0.0)), 1.0e-12) for node in members]))
                force_r_nonproj = float(
                    np.mean([float(np.dot(force_map_nonproj[id(node)], e_r)) for node, e_r in zip(members, radial_units)])
                )
                force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(node)], axis)) for node in members]))
                force_r_proj = float(
                    np.mean(
                        [
                            float(np.dot(force_map_total[id(node)] - force_map_nonproj[id(node)], e_r))
                            for node, e_r in zip(members, radial_units)
                        ]
                    )
                )
                force_z_proj = float(
                    np.mean(
                        [
                            float(np.dot(force_map_total[id(node)] - force_map_nonproj[id(node)], axis))
                            for node in members
                        ]
                    )
                )
                for node, e_r in zip(members, radial_units):
                    accel_nonproj = (force_r_nonproj / mass_avg) * e_r + (force_z_nonproj / mass_avg) * axis
                    accel_proj = (force_r_proj / mass_avg) * e_r + (force_z_proj / mass_avg) * axis
                    cache[id(node)] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
                    covered.add(id(node))

            center_v = state.layer_centers[k]
            vid = id(center_v)
            if (center_v not in state.bV_caps) and (vid in force_map_nonproj):
                mass = max(float(getattr(center_v, "m", 0.0)), 1.0e-12)
                accel_nonproj = float(np.dot(force_map_nonproj[vid], axis)) / mass
                accel_proj = float(np.dot(force_map_total[vid] - force_map_nonproj[vid], axis)) / mass
                cache[vid] = _clip_acceleration(accel_nonproj * axis, state) + accel_proj * axis
                covered.add(vid)

        for node in [candidate for candidate in state.HC.V if (candidate not in state.bV_caps) and (id(candidate) in force_map_nonproj)]:
            vid = id(node)
            if vid in covered:
                continue
            mass = max(float(getattr(node, "m", 0.0)), 1.0e-12)
            accel_nonproj = force_map_nonproj[vid] / mass
            accel_proj = (force_map_total[vid] - force_map_nonproj[vid]) / mass
            cache[vid] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
        state._axisym_accel_cache = cache

    return np.asarray(cache.get(id(v), np.zeros(3, dtype=float)), dtype=float)


def _tet_volume_and_shape_grads(tet_points: np.ndarray) -> tuple[float, np.ndarray] | None:
    x = np.ones((4, 4), dtype=float)
    x[:, 1:] = np.asarray(tet_points, dtype=float)
    det = float(np.linalg.det(x))
    volume = abs(det) / 6.0
    if (not np.isfinite(volume)) or volume <= 1.0e-30:
        return None
    try:
        inv_x = np.linalg.inv(x)
    except np.linalg.LinAlgError:
        return None
    grads = np.asarray(inv_x[1:, :].T, dtype=float)
    if not np.all(np.isfinite(grads)):
        return None
    return volume, grads


def _gmsh_tet_metric_arrays(
    points: np.ndarray,
    tets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int).reshape((-1, 4))
    if pts.size == 0 or tet_idx.size == 0:
        return (
            np.zeros(tet_idx.shape[0], dtype=bool),
            np.zeros(tet_idx.shape[0], dtype=float),
            np.zeros((tet_idx.shape[0], 4, 3), dtype=float),
        )
    in_range = (np.min(tet_idx, axis=1) >= 0) & (np.max(tet_idx, axis=1) < pts.shape[0])
    valid = np.zeros(tet_idx.shape[0], dtype=bool)
    volumes = np.zeros(tet_idx.shape[0], dtype=float)
    grads = np.zeros((tet_idx.shape[0], 4, 3), dtype=float)
    if not np.any(in_range):
        return valid, volumes, grads

    candidate_ids = np.flatnonzero(in_range)
    tet_points = pts[tet_idx[candidate_ids]]
    finite_points = np.all(np.isfinite(tet_points), axis=(1, 2))
    if not np.any(finite_points):
        return valid, volumes, grads

    candidate_ids = candidate_ids[finite_points]
    tet_points = tet_points[finite_points]
    x = np.ones((tet_points.shape[0], 4, 4), dtype=float)
    x[:, :, 1:] = tet_points
    try:
        det = np.linalg.det(x)
    except np.linalg.LinAlgError:
        return valid, volumes, grads
    candidate_volumes = np.abs(det) / 6.0
    volume_valid = np.isfinite(candidate_volumes) & (candidate_volumes > 1.0e-30)
    if not np.any(volume_valid):
        return valid, volumes, grads

    candidate_ids = candidate_ids[volume_valid]
    candidate_volumes = candidate_volumes[volume_valid]
    x_valid = x[volume_valid]
    try:
        inv_x = np.linalg.inv(x_valid)
        candidate_grads = np.transpose(inv_x[:, 1:, :], (0, 2, 1))
    except np.linalg.LinAlgError:
        candidate_grads = np.zeros((candidate_ids.size, 4, 3), dtype=float)
        keep = np.zeros(candidate_ids.size, dtype=bool)
        for local_idx, matrix in enumerate(x_valid):
            try:
                inv_x_single = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue
            candidate_grads[local_idx] = np.asarray(inv_x_single[1:, :].T, dtype=float)
            keep[local_idx] = True
        candidate_ids = candidate_ids[keep]
        candidate_volumes = candidate_volumes[keep]
        candidate_grads = candidate_grads[keep]
    finite_grads = np.all(np.isfinite(candidate_grads), axis=(1, 2))
    candidate_ids = candidate_ids[finite_grads]
    candidate_volumes = candidate_volumes[finite_grads]
    candidate_grads = candidate_grads[finite_grads]
    valid[candidate_ids] = True
    volumes[candidate_ids] = candidate_volumes
    grads[candidate_ids] = candidate_grads
    return valid, volumes, grads


def _gmsh_tet_divergence_l2(state: VolumetricPitoisState) -> float:
    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        return 0.0

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    valid, volumes, grads = _gmsh_tet_metric_arrays(points, tets)
    if not np.any(valid):
        return 0.0
    tet_valid = tets.reshape((-1, 4))[valid]
    div_u = np.einsum("tij,tij->t", velocities[tet_valid], grads[valid])
    finite = np.isfinite(div_u)
    if not np.any(finite):
        return 0.0
    weighted = float(np.sum(volumes[valid][finite] * div_u[finite] * div_u[finite]))
    volume_sum = float(np.sum(volumes[valid][finite]))
    if volume_sum <= 1.0e-30:
        return 0.0
    return float(math.sqrt(max(weighted / volume_sum, 0.0)))


def _project_gmsh_contact_line_velocities_to_sphere_tangents(state: VolumetricPitoisState) -> None:
    radius = float(state.config.particle_radius)
    specs = (
        (
            state.bottom_contact_ring,
            np.asarray(state.bottom_sphere_center, dtype=float),
            np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float),
        ),
        (
            state.top_contact_ring,
            np.asarray(state.top_sphere_center, dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    for ring, sphere_center, sphere_velocity in specs:
        for v in ring:
            normal = np.asarray(v.x_a[:3], dtype=float) - sphere_center
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-30:
                continue
            normal = normal / normal_norm
            rel_u = np.asarray(v.u[:3], dtype=float) - sphere_velocity
            v.u = sphere_velocity + rel_u - float(np.dot(rel_u, normal)) * normal


def _apply_gmsh_incompressible_projection(state: VolumetricPitoisState, *, dt: float) -> None:
    """Project the velocity/volume update onto the selected pressure closure.

    Incompressible or K -> infinity compressible-limit branch:
        B M^-1 B^T p = -B u* / dt
        u^{n+1} = u* + dt M^-1 B^T p

    Finite-K compressible branch:
        calls PR33 multiphase_sparse_compressible_eos_pressure_correction.

    Reference: the split pressure projection follows the Chorin projection idea
    (Chorin 1968), while B = dV/dx and F_p = B^T p are the internal PR33
    tetrahedral volume-gradient operator.
    """
    closure = str(getattr(state.config, "pressure_closure", "legacy")).strip().lower()
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (closure not in {"compressible", "incompressible"})
        or float(dt) <= 0.0
    ):
        return
    if coo_matrix is None or diags is None or spsolve is None:
        raise RuntimeError("USER_ENABLE_INCOMPRESSIBLE_PROJECTION requires scipy.sparse.")

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        return

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    if (not np.all(np.isfinite(points))) or (not np.all(np.isfinite(velocities))):
        return

    n_vertices = len(vertices)
    n_dofs = 3 * n_vertices
    valid_tets, tet_volumes, tet_grads = _gmsh_tet_metric_arrays(points, tets)
    if not np.any(valid_tets):
        return
    tet_idx_all = tets.reshape((-1, 4))
    valid_cell_ids = np.flatnonzero(valid_tets).astype(int, copy=False)
    valid_idx = tet_idx_all[valid_tets]
    valid_volumes = tet_volumes[valid_tets]
    valid_grads = tet_grads[valid_tets]
    lumped = np.zeros(n_vertices, dtype=float)
    np.add.at(lumped, valid_idx.ravel(), np.repeat(0.25 * valid_volumes, 4))

    def divergence_l2_from_velocity_array(velocity_array: np.ndarray) -> float:
        div_u = np.einsum("tij,tij->t", np.asarray(velocity_array, dtype=float)[valid_idx], valid_grads)
        finite = np.isfinite(div_u)
        if not np.any(finite):
            return 0.0
        weighted = float(np.sum(valid_volumes[finite] * div_u[finite] * div_u[finite]))
        volume_sum = float(np.sum(valid_volumes[finite]))
        if volume_sum <= 1.0e-30:
            return 0.0
        return float(math.sqrt(max(weighted / volume_sum, 0.0)))

    closure = str(getattr(state.config, "pressure_closure", "incompressible")).strip().lower()
    if closure not in {"compressible", "incompressible"}:
        return
    dof_offsets = np.arange(3, dtype=int)
    b_cols = (3 * valid_idx[:, :, None] + dof_offsets[None, None, :]).reshape(-1)
    b_rows = np.repeat(valid_cell_ids, 12)
    b_data = (valid_volumes[:, None, None] * valid_grads).reshape(-1)

    def tangent_projector(vertex) -> np.ndarray:
        if vertex in state.bottom_contact_ring:
            center = np.asarray(state.bottom_sphere_center, dtype=float)
        elif vertex in state.top_contact_ring:
            center = np.asarray(state.top_sphere_center, dtype=float)
        else:
            return np.eye(3, dtype=float)
        normal = np.asarray(vertex.x_a[:3], dtype=float) - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            return np.eye(3, dtype=float)
        normal = normal / normal_norm
        return np.eye(3, dtype=float) - np.outer(normal, normal)

    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[float] = []
    fixed = set(state.bV_caps)
    pressure_fixed = set(fixed)
    if (
        (not bool(getattr(state.config, "solver_pr33_pressure_contact_line_mobility", True)))
        or (not bool(getattr(state.config, "enable_gmsh_contact_angle_kinematics", False)))
    ):
        pressure_fixed.update(state.bottom_contact_ring)
        pressure_fixed.update(state.top_contact_ring)
    solver_lumped = np.asarray(_solver_mass_from_lumped_volume(state.config, lumped), dtype=float)
    for idx, v in enumerate(vertices):
        if v in pressure_fixed or solver_lumped[idx] <= 1.0e-30:
            continue
        inv_mass = 1.0 / max(float(solver_lumped[idx]), 1.0e-30)
        projector = tangent_projector(v)
        base = 3 * idx
        for a in range(3):
            for b in range(3):
                value = inv_mass * float(projector[a, b])
                if abs(value) > 0.0:
                    w_rows.append(base + a)
                    w_cols.append(base + b)
                    w_data.append(value)

    state.incompressible_divergence_before_l2 = float(divergence_l2_from_velocity_array(velocities))
    # B maps nodal velocity to per-tet volume rate; B^T maps tet pressure to
    # nodal pressure force.  This is the same operator meaning as PR33 F_p.
    volume_matrix = coo_matrix((b_data, (b_rows, b_cols)), shape=(int(tets.shape[0]), n_dofs)).tocsr()
    pressure_matrix = volume_matrix.T.tocsr()
    mobility = coo_matrix((w_data, (w_rows, w_cols)), shape=(n_dofs, n_dofs)).tocsr()
    velocity_vec = velocities.reshape(n_dofs)
    stiffness = (volume_matrix @ mobility @ pressure_matrix).tocsr()
    diag_mean = float(np.mean(np.abs(stiffness.diagonal()))) if n_vertices else 1.0
    diag_mean = max(diag_mean, 1.0e-30)
    damping = max(float(getattr(state.config, "incompressible_projection_regularization", 0.0)), 0.0)
    volume_rate = np.asarray(volume_matrix @ velocity_vec, dtype=float)
    pressure_total_from_ddgclib: np.ndarray | None = None
    pressure_force_vec_from_ddgclib: np.ndarray | None = None
    projected_velocity_from_ddgclib: np.ndarray | None = None
    pressure_reference_from_ddgclib = 0.0
    if closure == "compressible" and not bool(
        getattr(state.config, "compressible_use_incompressible_limit_projection", False)
    ):
        reference_volumes = np.asarray(getattr(state, "ddgclib_reference_tet_volumes", np.empty(0)), dtype=float)
        if reference_volumes.shape != tet_volumes.shape:
            reference_volumes = tet_volumes.copy()
        else:
            reference_volumes = reference_volumes.copy()
        reference_relaxation = min(
            1.0,
            max(
                0.0,
                float(getattr(state.config, "compressible_eos_reference_volume_relaxation", 0.0)),
            ),
        )
        if reference_relaxation > 0.0 and np.any(valid_tets):
            current_valid = np.maximum(np.asarray(tet_volumes, dtype=float)[valid_tets], 1.0e-300)
            reference_valid = np.maximum(reference_volumes[valid_tets], 1.0e-300)
            reference_total = float(np.sum(reference_valid))
            current_total = float(np.sum(current_valid))
            if reference_total > 0.0 and current_total > 0.0:
                current_target = current_valid * (reference_total / current_total)
                reference_volumes[valid_tets] = (
                    (1.0 - reference_relaxation) * reference_valid
                    + reference_relaxation * current_target
                )
                state.ddgclib_reference_tet_volumes = reference_volumes.copy()
        reference_mode = str(
            getattr(state.config, "solver_pr33_pressure_reference_mode", "heron")
        ).strip().lower()
        if (
            bool(getattr(state.config, "enable_solver_pr33_pressure_force", False))
            and reference_mode == "heron"
        ):
            pressure_reference_from_ddgclib = _gmsh_heron_pressure_reference(state, vertices)
        bulk_modulus = max(float(getattr(state.config, "compressible_bulk_modulus", 0.0)), 0.0)
        base_pressures = np.full(int(tets.shape[0]), pressure_reference_from_ddgclib, dtype=float)
        bulk_by_tet = np.where(valid_tets, bulk_modulus, 0.0)
        pressure_delta_limit = 0.0
        limit_fraction = max(
            0.0,
            float(getattr(state.config, "compressible_eos_pressure_delta_limit_fraction", 0.0)),
        )
        if bulk_modulus > 0.0 and limit_fraction > 0.0:
            pressure_delta_limit = limit_fraction * bulk_modulus
        projected_velocity_from_ddgclib, pressure_total_from_ddgclib, pressure, pressure_force_vec_from_ddgclib = (
            multiphase_sparse_compressible_eos_pressure_correction(
                velocities=velocities,
                nonpressure_forces=np.zeros_like(velocities, dtype=float),
                mobility_matrix=mobility,
                tet_volumes=tet_volumes,
                target_tet_volumes=reference_volumes,
                reference_tet_volumes=reference_volumes,
                volume_matrix=volume_matrix,
                base_pressures=base_pressures,
                bulk_by_tet=bulk_by_tet,
                dt=float(dt),
                diagonal_regularization=damping * diag_mean if damping > 0.0 else 0.0,
                stiffness_matrix=stiffness,
                pressure_delta_limit=pressure_delta_limit if pressure_delta_limit > 0.0 else None,
            )
        )
        state.last_solver_pr33_operator = "ddgclib_sparse_compressible_eos"
    else:
        # This branch is the incompressible projection and also the
        # K -> infinity limit of the compressible PR33 EOS equation.
        matrix = stiffness
        if damping > 0.0:
            matrix = matrix + diags(
                np.full(int(tets.shape[0]), damping * diag_mean, dtype=float),
                0,
                shape=(int(tets.shape[0]), int(tets.shape[0])),
            )
        rhs = -volume_rate / max(float(dt), 1.0e-30)
        pressure = np.asarray(spsolve(matrix, rhs), dtype=float)
        state.last_solver_pr33_operator = "local_sparse_incompressible_projection"
    if pressure.size != int(tets.shape[0]) or not np.all(np.isfinite(pressure)):
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)

    pressure_delta = np.asarray(pressure, dtype=float)
    pressure_total = pressure_delta.copy()
    pressure_reference = 0.0
    pressure_force_vec = np.asarray(pressure_matrix @ pressure_total, dtype=float)
    pressure_force_by_vertex = pressure_force_vec.reshape((n_vertices, 3))
    if bool(getattr(state.config, "enable_solver_pr33_pressure_force", False)):
        reference_mode = str(
            getattr(state.config, "solver_pr33_pressure_reference_mode", "heron")
        ).strip().lower()
        if pressure_total_from_ddgclib is not None and pressure_force_vec_from_ddgclib is not None:
            pressure_reference = float(pressure_reference_from_ddgclib)
            pressure_total = np.asarray(pressure_total_from_ddgclib, dtype=float)
            if pressure_total.size != int(tets.shape[0]) or not np.all(np.isfinite(pressure_total)):
                pressure_total = np.nan_to_num(
                    pressure_delta + pressure_reference,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            pressure_force_vec = np.asarray(pressure_force_vec_from_ddgclib, dtype=float)
            if pressure_force_vec.size != n_dofs or not np.all(np.isfinite(pressure_force_vec)):
                pressure_force_vec = np.asarray(pressure_matrix @ pressure_total, dtype=float)
            pressure_force_by_vertex = pressure_force_vec.reshape((n_vertices, 3))
        elif closure == "compressible" and reference_mode == "heron":
            pressure_reference = _gmsh_heron_pressure_reference(state, vertices)
            pressure_total = pressure_delta + pressure_reference
        elif closure == "compressible":
            pressure_reference = 0.0
            pressure_total = pressure_delta.copy()

        if pressure_total_from_ddgclib is None or pressure_force_vec_from_ddgclib is None:
            pressure_force_vec = np.asarray(pressure_matrix @ pressure_total, dtype=float)
            pressure_force_by_vertex = pressure_force_vec.reshape((n_vertices, 3))
        state.last_solver_pr33_pressure_force_l1 = float(
            np.sum(np.linalg.norm(pressure_force_by_vertex, axis=1))
        )
        state.last_solver_pr33_pressure_reference = float(pressure_reference)
        state.last_solver_pr33_pressure_delta_linf = float(
            np.max(np.abs(pressure_delta[valid_tets])) if np.any(valid_tets) else 0.0
        )
        state.last_solver_pr33_pressure_total_linf = float(
            np.max(np.abs(pressure_total[valid_tets])) if np.any(valid_tets) else 0.0
        )
        state.solver_pr33_pressure_force_map = {
            id(vertex): np.asarray(pressure_force_by_vertex[idx], dtype=float)
            for idx, vertex in enumerate(vertices)
        }

    state.incompressible_projection_pressure = pressure_total
    state.ddgclib_pressure_closure = closure
    state.ddgclib_closure_pressure_delta = pressure_delta
    state.ddgclib_closure_pressure = pressure_total

    if (
        projected_velocity_from_ddgclib is not None
        and bool(getattr(state.config, "solver_pr33_use_returned_velocity", False))
    ):
        raw_delta_velocity = np.asarray(projected_velocity_from_ddgclib, dtype=float).reshape(n_dofs) - velocity_vec
    else:
        raw_delta_velocity = float(dt) * np.asarray(mobility @ pressure_force_vec, dtype=float)
    if raw_delta_velocity.size != n_dofs or not np.all(np.isfinite(raw_delta_velocity)):
        raw_delta_velocity = np.nan_to_num(
            np.asarray(raw_delta_velocity, dtype=float).reshape(-1)[:n_dofs],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if raw_delta_velocity.size != n_dofs:
            raw_delta_velocity = np.zeros(n_dofs, dtype=float)
    alpha = 1.0
    target_volume = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))

    def apply_trial_axisymmetry(trial_points: np.ndarray) -> None:
        radius = float(state.config.particle_radius)
        for spec in getattr(state, "gmsh_axisym_ring_specs", []):
            spec_vertices = list(getattr(spec, "vertices", []))
            point_indices = [node_id_map.get(id(vertex)) for vertex in spec_vertices]
            if not point_indices or any(idx is None for idx in point_indices):
                continue
            point_indices = [int(idx) for idx in point_indices]
            coords = trial_points[point_indices]
            rel = coords - mid[None, :]
            axial = np.dot(rel, axis)
            radial = rel - np.outer(axial, axis)
            r_mean = float(np.mean(np.linalg.norm(radial, axis=1)))
            cap_id = _gmsh_axisym_uniform_cap_id(spec_vertices)
            if cap_id is None:
                center_coord = mid + float(np.mean(axial)) * axis
            else:
                sphere_center = (
                    np.asarray(state.bottom_sphere_center, dtype=float)
                    if cap_id == "bottom"
                    else np.asarray(state.top_sphere_center, dtype=float)
                )
                sign = 1.0 if cap_id == "bottom" else -1.0
                r_mean = float(np.clip(r_mean, 0.0, radius))
                center_coord = sphere_center + sign * math.sqrt(max(radius * radius - r_mean * r_mean, 0.0)) * axis
            angles = tuple(getattr(spec, "angles", ()))
            if len(point_indices) == 1 or r_mean <= 1.0e-30:
                for point_idx in point_indices:
                    trial_points[point_idx] = center_coord
            else:
                for point_idx, angle in zip(point_indices, angles):
                    e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
                    trial_points[point_idx] = center_coord + r_mean * e_r

    def apply_trial_contact_projection(trial_points: np.ndarray) -> None:
        radius = float(state.config.particle_radius)
        for ring, sphere_center in (
            (state.bottom_contact_ring, np.asarray(state.bottom_sphere_center, dtype=float)),
            (state.top_contact_ring, np.asarray(state.top_sphere_center, dtype=float)),
        ):
            for vertex in ring:
                point_idx = node_id_map.get(id(vertex))
                if point_idx is None:
                    continue
                point_idx = int(point_idx)
                rel = trial_points[point_idx] - sphere_center
                rel_norm = float(np.linalg.norm(rel))
                if rel_norm <= 1.0e-30:
                    continue
                trial_points[point_idx] = sphere_center + radius * (rel / rel_norm)

    def predicted_step_volume(alpha_value: float) -> float:
        trial_points = points.copy()
        trial_velocity = (velocity_vec + float(alpha_value) * raw_delta_velocity).reshape((n_vertices, 3))
        for vertex_idx, vertex in enumerate(vertices):
            if vertex in fixed:
                continue
            trial_points[vertex_idx] = trial_points[vertex_idx] + float(dt) * trial_velocity[vertex_idx]
        # Match the coordinate constraints applied immediately after the real
        # move: axisymmetric ring projection and contact-line projection.
        for _projection_pass in range(2):
            apply_trial_axisymmetry(trial_points)
            apply_trial_contact_projection(trial_points)
        return _indexed_tet_mesh_volume_m3(trial_points, tets)

    incompressible_like_projection = closure == "incompressible" or (
        closure == "compressible"
        and bool(getattr(state.config, "compressible_use_incompressible_limit_projection", False))
    )
    if (
        target_volume > 0.0
        and incompressible_like_projection
        and bool(USER_ENABLE_INCOMPRESSIBLE_VOLUME_ALPHA_SEARCH)
    ):
        candidates: list[tuple[float, float]] = []

        def add_candidate(alpha_value: float) -> tuple[float, float] | None:
            if not np.isfinite(alpha_value):
                return None
            alpha_value = float(alpha_value)
            volume_value = predicted_step_volume(alpha_value)
            if not np.isfinite(volume_value):
                return None
            pair = (alpha_value, float(volume_value))
            candidates.append(pair)
            return pair

        volume_tol = max(1.0e-8 * target_volume, 1.0e-30)
        previous_alpha_pair = add_candidate(float(getattr(state, "incompressible_projection_alpha", 1.0)))
        vol1_pair = add_candidate(1.0)
        if previous_alpha_pair is not None and vol1_pair is not None:
            denom = vol1_pair[1] - previous_alpha_pair[1]
            if abs(denom) > max(1.0e-18 * target_volume, 1.0e-30):
                add_candidate(
                    previous_alpha_pair[0]
                    + (target_volume - previous_alpha_pair[1])
                    * (vol1_pair[0] - previous_alpha_pair[0])
                    / denom
                )
        vol0_pair = None

        prelim_candidates = [
            (candidate_alpha, candidate_volume)
            for candidate_alpha, candidate_volume in candidates
            if np.isfinite(candidate_alpha) and np.isfinite(candidate_volume)
        ]
        prelim_best = (
            min(prelim_candidates, key=lambda item: abs(item[1] - target_volume))
            if prelim_candidates
            else None
        )
        needs_bracket = prelim_best is None or abs(prelim_best[1] - target_volume) > volume_tol
        if needs_bracket:
            vol0_pair = add_candidate(0.0)
            if vol0_pair is not None and vol1_pair is not None:
                denom = vol1_pair[1] - vol0_pair[1]
                if abs(denom) > max(1.0e-18 * target_volume, 1.0e-30):
                    add_candidate((target_volume - vol0_pair[1]) / denom)
            for alpha_endpoint in (-2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0):
                add_candidate(alpha_endpoint)

            bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
            for span in (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0):
                sorted_candidates = sorted(candidates, key=lambda item: item[0])
                for left, right in zip(sorted_candidates[:-1], sorted_candidates[1:]):
                    f_left = left[1] - target_volume
                    f_right = right[1] - target_volume
                    if f_left == 0.0:
                        bracket = (left, left)
                        break
                    if f_left * f_right <= 0.0:
                        bracket = (left, right)
                        break
                if bracket is not None:
                    break
                add_candidate(-span)
                add_candidate(span)

            if bracket is not None:
                left, right = bracket
                if left[0] == right[0]:
                    add_candidate(left[0])
                else:
                    lo_alpha, lo_volume = left
                    hi_alpha, hi_volume = right
                    lo_f = lo_volume - target_volume
                    hi_f = hi_volume - target_volume
                    for _root_iter in range(32):
                        mid_alpha = 0.5 * (lo_alpha + hi_alpha)
                        mid_volume = predicted_step_volume(mid_alpha)
                        if not np.isfinite(mid_volume):
                            break
                        candidates.append((mid_alpha, float(mid_volume)))
                        mid_f = mid_volume - target_volume
                        if abs(mid_f) <= max(1.0e-10 * target_volume, 1.0e-30):
                            break
                        if lo_f * mid_f <= 0.0:
                            hi_alpha, hi_volume, hi_f = mid_alpha, float(mid_volume), float(mid_f)
                        else:
                            lo_alpha, lo_volume, lo_f = mid_alpha, float(mid_volume), float(mid_f)
        finite_candidates = [
            (candidate_alpha, candidate_volume)
            for candidate_alpha, candidate_volume in candidates
            if np.isfinite(candidate_alpha) and np.isfinite(candidate_volume)
        ]
        if finite_candidates:
            alpha = min(finite_candidates, key=lambda item: abs(item[1] - target_volume))[0]
    state.incompressible_projection_alpha = float(alpha)
    state.ddgclib_closure_alpha = float(alpha)

    projected_velocity = velocity_vec + float(alpha) * raw_delta_velocity
    if not np.all(np.isfinite(projected_velocity)):
        projected_velocity = np.nan_to_num(projected_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    projected_velocity = projected_velocity.reshape((n_vertices, 3))
    for idx, v in enumerate(vertices):
        if v in fixed:
            continue
        v.u = projected_velocity[idx]

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    final_velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    state.incompressible_divergence_after_l2 = float(divergence_l2_from_velocity_array(final_velocities))
    final_volume_rate = np.asarray(volume_matrix @ final_velocities.reshape(n_dofs), dtype=float)
    reference_volumes = np.asarray(getattr(state, "ddgclib_reference_tet_volumes", np.empty(0)), dtype=float)
    if reference_volumes.shape != tet_volumes.shape:
        reference_volumes = np.maximum(tet_volumes, 1.0e-300)
    rel_rate = np.zeros_like(final_volume_rate, dtype=float)
    rel_rate[valid_tets] = final_volume_rate[valid_tets] / np.maximum(reference_volumes[valid_tets], 1.0e-300)
    finite_rel = np.isfinite(rel_rate[valid_tets])
    if np.any(finite_rel):
        rel_valid = rel_rate[valid_tets][finite_rel]
        state.ddgclib_closure_volume_rel_l2 = float(math.sqrt(float(np.mean(rel_valid * rel_valid))))
        state.ddgclib_closure_volume_rel_linf = float(np.max(np.abs(rel_valid)))
    else:
        state.ddgclib_closure_volume_rel_l2 = 0.0
        state.ddgclib_closure_volume_rel_linf = 0.0
    _axisym_clear_caches(state)


def _advance_gmsh_incompressible_substep(
    state: VolumetricPitoisState,
    *,
    dt: float,
    timing: dict[str, float] | None = None,
) -> bool:
    closure = str(getattr(state.config, "pressure_closure", "legacy")).strip().lower()
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (closure not in {"compressible", "incompressible"})
    ):
        return False

    substep_reference_positions = _volume_points_array(state).copy()
    if not bool(getattr(state.config, "allow_contact_line_growth", False)):
        state._gmsh_no_growth_bottom_radius_limit = (
            float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
        )
        state._gmsh_no_growth_top_radius_limit = (
            float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
        )
    with _time_module(timing, "cap_move_axisym"):
        _set_cap_velocities(state)
        _enforce_no_swirl_velocity_field(state)
        _move_caps(state, dt=dt)
        _gmsh_axisymmetrize_state(state)
    with _time_module(timing, "mass_dual"):
        _update_duals_and_masses(state)
    with _time_module(timing, "pressure_scalar"):
        _update_pressure_scalar(state)
    with _time_module(timing, "pressure_projection"):
        _update_pressure_projection_scalar(state, dt=dt)

    with _time_module(timing, "force_accel"):
        solver_fixed_vertices = set(state.bV_caps)
        solver_fixed_vertices.update(getattr(state, "bottom_contact_ring", []))
        solver_fixed_vertices.update(getattr(state, "top_contact_ring", []))
        free_vertices = [v for v in state.HC.V if v not in solver_fixed_vertices]
        accel_by_id: dict[int, np.ndarray] = {}
        for v in free_vertices:
            accel_by_id[id(v)] = _clip_acceleration(_vertex_acceleration(v, state=state), state)

        for v in free_vertices:
            u_new = np.asarray(v.u[:3], dtype=float) + float(dt) * accel_by_id.get(id(v), np.zeros(3, dtype=float))
            v.u = np.nan_to_num(u_new, nan=0.0, posinf=0.0, neginf=0.0)

    with _time_module(timing, "velocity_project"):
        _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)
        _apply_gmsh_incompressible_projection(state, dt=dt)
        _set_cap_velocities(state)
    with _time_module(timing, "volume_flux"):
        _apply_gmsh_volume_flux_velocity_correction(state, dt=dt)
        if bool(getattr(state.config, "enforce_no_swirl", False)):
            _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)

    with _time_module(timing, "mesh_move"):
        moving_vertices: list[object] = []
        targets: list[tuple[float, float, float]] = []
        max_free_disp = min(
            max(
                float(_length_to_compute(state.config, USER_GMSH_MAX_FREE_MESH_DISP_M)),
                float(_length_to_compute(state.config, 1.0e-9)),
            ),
            0.02 * max(float(state.config.particle_radius), 1.0e-30),
        )
        max_coord = max(
            float(_length_to_compute(state.config, USER_POST_EDIT_CONTINUITY_MAX_COORD_M)),
            2.0 * float(state.config.particle_radius),
            1.0,
        )
        for v in free_vertices:
            if v in state.bV_caps:
                continue
            position = np.asarray(v.x_a[:3], dtype=float)
            displacement = float(dt) * np.asarray(v.u[:3], dtype=float)
            displacement_norm = float(np.linalg.norm(displacement))
            if np.isfinite(displacement_norm) and displacement_norm > max_free_disp:
                displacement *= max_free_disp / max(displacement_norm, 1.0e-30)
                v.u = displacement / max(float(dt), 1.0e-30)
            target = position + displacement
            if not np.all(np.isfinite(target)):
                continue
            if float(np.max(np.abs(target))) > max_coord:
                continue
            moving_vertices.append(v)
            targets.append(tuple(float(x) for x in target))
        _move_vertices_batch(moving_vertices, targets, state.HC, state.bV_caps)
        _axisym_clear_caches(state)

    with _time_module(timing, "contact_line"):
        _gmsh_axisymmetrize_state(state)
        _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)
        _update_moving_contact_line(state, dt=dt)
        _gmsh_axisymmetrize_state(state)
        _enforce_no_swirl_velocity_field(state)
        _assert_fixed_topology(state)
    with _time_module(timing, "post_continuity"):
        finite_compressible = closure == "compressible" and not bool(
            getattr(state.config, "compressible_use_incompressible_limit_projection", False)
        )
        if finite_compressible:
            eos_position_relaxation = float(
                np.clip(
                    getattr(
                        state.config,
                        "compressible_eos_position_correction_relaxation",
                        USER_COMPRESSIBLE_EOS_POSITION_CORRECTION_RELAXATION,
                    ),
                    0.0,
                    1.0,
                )
            )
            if eos_position_relaxation > 0.0:
                _apply_gmsh_compressible_eos_position_volume_correction(
                    state,
                    dt=dt,
                    reference_positions=substep_reference_positions,
                )
            _apply_gmsh_final_geometric_volume_restore(
                state,
                dt=dt,
                reference_positions=substep_reference_positions,
            )
        else:
            _apply_gmsh_post_edit_continuity_projection(
                state,
                dt=dt,
                reference_positions=substep_reference_positions,
            )
            _apply_gmsh_final_geometric_volume_restore(
                state,
                dt=dt,
                reference_positions=substep_reference_positions,
            )
    _assert_fixed_topology(state)
    with _time_module(timing, "final_mass_pressure"):
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        _update_pressure_projection_scalar(state, dt=dt, refine=False)
    _assert_fixed_topology(state)
    return True


def _cap_force(cap_vertices: list, *, state: VolumetricPitoisState) -> np.ndarray:
    if not cap_vertices:
        return np.zeros(3, dtype=float)
    return np.sum(
        [
            _Ftot(
                v,
                state=state,
                include_projected_pressure=True,
                include_contact_line=False,
                include_damping=False,
            )
            for v in cap_vertices
        ],
        axis=0,
    )


def _cap_traction_force(
    state: VolumetricPitoisState,
    *,
    which: str,
) -> np.ndarray:
    if which == "bottom":
        tri_idx = np.asarray(state.surface_export_bottom_cap_tris, dtype=int)
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
    else:
        tri_idx = np.asarray(state.surface_export_top_cap_tris, dtype=int)
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)
    if tri_idx.size == 0:
        return np.zeros(3, dtype=float)

    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    p_proj = float(getattr(state, "pressure_projection_scalar", 0.0))
    vertices = state.surface_export_vertices
    Fcap = np.zeros(3, dtype=float)
    mu = float(state.config.mu_f)
    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)

    for a, b, c in tri_idx:
        va = vertices[int(a)]
        vb = vertices[int(b)]
        vc = vertices[int(c)]
        pa = np.asarray(va.x_a[:3], dtype=float)
        pb = np.asarray(vb.x_a[:3], dtype=float)
        pc = np.asarray(vc.x_a[:3], dtype=float)
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        area = float(np.linalg.norm(area_vec))
        if area <= 1.0e-30:
            continue

        centroid = (pa + pb + pc) / 3.0
        n_s = centroid - sphere_center
        n_norm = float(np.linalg.norm(n_s))
        if n_norm <= 1.0e-30:
            continue
        n_s = n_s / n_norm

        sigma_tri = np.zeros((3, 3), dtype=float)
        for vtx in (va, vb, vc):
            p_v = float(pressure_model(vtx)) + p_proj
            if bool(getattr(state, "gmsh_compute_mesh", False)):
                du_v = np.zeros((3, 3), dtype=float)
            else:
                du_v = velocity_difference_tensor_pointwise(vtx, state.HC, dim=3)
            sigma_tri += cauchy_stress(p_v, du_v, mu, dim=3)
        sigma_tri /= 3.0
        Fcap += sigma_tri @ n_s * area

    return Fcap


def _ordered_outer_rings_for_surface(state: VolumetricPitoisState) -> list[list[object]]:
    # Keep the side-surface connectivity inherited from the initial mesh.
    # Re-sorting rings by current angle changes which same-ID vertices are
    # connected across layers, even when the intended topology is unchanged.
    return [list(ring) for ring in state.outer_rings]


def _structured_surface_topology(ordered_rings):
    if not ordered_rings:
        return set(), np.empty((0, 3), dtype=int), set()

    n_layers = len(ordered_rings)
    n_ring = len(ordered_rings[0])

    def flat_idx(k: int, i: int) -> int:
        return k * n_ring + i

    edges: set[tuple[int, int]] = set()
    triangles: list[tuple[int, int, int]] = []

    ring_coords = [
        np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        for ring in ordered_rings
    ]

    for k in range(n_layers):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            edges.add(tuple(sorted((a, b))))

    for k in range(n_layers - 1):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            c = flat_idx(k + 1, (i + 1) % n_ring)
            d = flat_idx(k + 1, i)
            edges.add(tuple(sorted((a, d))))
            edges.add(tuple(sorted((b, c))))
            pa = ring_coords[k][i]
            pb = ring_coords[k][(i + 1) % n_ring]
            pc = ring_coords[k + 1][(i + 1) % n_ring]
            pd = ring_coords[k + 1][i]
            if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                edges.add(tuple(sorted((a, c))))
                triangles.append((a, b, c))
                triangles.append((a, c, d))
            else:
                edges.add(tuple(sorted((b, d))))
                triangles.append((a, b, d))
                triangles.append((b, c, d))

    boundary_indices = {flat_idx(0, i) for i in range(n_ring)}
    boundary_indices.update({flat_idx(n_layers - 1, i) for i in range(n_ring)})
    return edges, np.array(triangles, dtype=int), boundary_indices


def _structured_surface_display_edges(ordered_rings):
    if not ordered_rings:
        return set()

    n_layers = len(ordered_rings)
    n_ring = len(ordered_rings[0])

    def flat_idx(k: int, i: int) -> int:
        return k * n_ring + i

    edges: set[tuple[int, int]] = set()
    for k in range(n_layers):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            edges.add(tuple(sorted((a, b))))
    for k in range(n_layers - 1):
        for i in range(n_ring):
            a = flat_idx(k, i)
            d = flat_idx(k + 1, i)
            edges.add(tuple(sorted((a, d))))
    return edges


def _contact_refined_surface_display_wireframe(
    state: VolumetricPitoisState,
    ordered_rings,
) -> tuple[np.ndarray, np.ndarray]:
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    coords = np.asarray(
        [[np.asarray(v.x_a[:3], dtype=float) for v in ring] for ring in ordered_rings],
        dtype=float,
    )
    edges = _structured_surface_display_edges(ordered_rings)
    flat = coords.reshape((-1, 3))
    segments = [np.asarray([flat[i], flat[j]], dtype=float) for i, j in sorted(edges)]
    return np.asarray(segments, dtype=float), flat


def _build_structured_side_surface_complex(ordered_rings):
    surface = Complex(3, domain=None)
    flat_vertices = [
        surface.V[tuple(map(float, v.x_a))]
        for ring in ordered_rings
        for v in ring
    ]
    edges, triangles, boundary_indices = _structured_surface_topology(ordered_rings)
    for i, j in edges:
        flat_vertices[i].connect(flat_vertices[j])
    surface_bV = {flat_vertices[idx] for idx in boundary_indices}
    return surface, surface_bV, flat_vertices, triangles


def _quad_uses_ac_diagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    return float(np.linalg.norm(a - c)) <= float(np.linalg.norm(b - d))


def _quad_triangles_consistent_diagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _quad_uses_ac_diagonal(a, b, c, d):
        return (
            np.array([a, b, c], dtype=float),
            np.array([a, c, d], dtype=float),
        )
    return (
        np.array([a, b, d], dtype=float),
        np.array([b, c, d], dtype=float),
    )


def _actual_compute_side_surface_triangles(state: VolumetricPitoisState) -> np.ndarray:
    if state.surface_export_side_tris.size:
        return _surface_triangles_xyz_from_indices(state, state.surface_export_side_tris)
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 3, 3), dtype=float)
    triangles: list[np.ndarray] = []
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        n_ring = min(len(lower_ring), len(upper_ring))
        if n_ring < 2:
            continue
        lower = [np.asarray(v.x_a[:3], dtype=float) for v in lower_ring]
        upper = [np.asarray(v.x_a[:3], dtype=float) for v in upper_ring]
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            tri0, tri1 = _quad_triangles_consistent_diagonal(
                lower[i],
                lower[i_next],
                upper[i_next],
                upper[i],
            )
            triangles.append(tri0)
            triangles.append(tri1)
    return np.asarray(triangles, dtype=float) if triangles else np.empty((0, 3, 3), dtype=float)


def _actual_compute_cap_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    if state.surface_export_bottom_cap_tris.size or state.surface_export_top_cap_tris.size:
        return (
            _surface_triangles_xyz_from_indices(state, state.surface_export_bottom_cap_tris),
            _surface_triangles_xyz_from_indices(state, state.surface_export_top_cap_tris),
        )
    cap_triangle_sets: list[np.ndarray] = []
    for rings, center_vertex in (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    ):
        triangles: list[np.ndarray] = []
        ring_xyz = [
            [np.asarray(v.x_a[:3], dtype=float) for v in ring]
            for ring in rings
        ]
        if not ring_xyz:
            cap_triangle_sets.append(np.empty((0, 3, 3), dtype=float))
            continue

        center = np.asarray(center_vertex.x_a[:3], dtype=float)
        n_ring = len(ring_xyz[0])
        for i in range(n_ring):
            triangles.append(
                np.array(
                    [center, ring_xyz[0][i], ring_xyz[0][(i + 1) % n_ring]],
                    dtype=float,
                )
            )
        for band in range(len(ring_xyz) - 1):
            inner = ring_xyz[band]
            outer = ring_xyz[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                tri0, tri1 = _quad_triangles_consistent_diagonal(
                    inner[i],
                    inner[i_next],
                    outer[i_next],
                    outer[i],
                )
                triangles.append(tri0)
                triangles.append(tri1)
        cap_triangle_sets.append(np.asarray(triangles, dtype=float))

    while len(cap_triangle_sets) < 2:
        cap_triangle_sets.append(np.empty((0, 3, 3), dtype=float))
    return cap_triangle_sets[0], cap_triangle_sets[1]


def _meridian_strip_indices(n_ring: int) -> tuple[int, int]:
    if n_ring <= 0:
        return 0, 0
    i0 = 0
    i1 = n_ring // 2
    return i0, i1


def _actual_compute_meridian_strip_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        empty = np.empty((0, 3, 3), dtype=float)
        return empty, empty, empty

    n_ring = len(ordered_rings[0])
    sectors = _meridian_strip_indices(n_ring)

    side_tris: list[np.ndarray] = []
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        lower = [np.asarray(v.x_a[:3], dtype=float) for v in lower_ring]
        upper = [np.asarray(v.x_a[:3], dtype=float) for v in upper_ring]
        for i in sectors:
            i_next = (i + 1) % n_ring
            side_tris.append(np.array([lower[i], lower[i_next], upper[i_next]], dtype=float))
            side_tris.append(np.array([lower[i], upper[i_next], upper[i]], dtype=float))

    cap_sets: list[np.ndarray] = []
    for rings, center_vertex in (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    ):
        tris: list[np.ndarray] = []
        ring_xyz = [[np.asarray(v.x_a[:3], dtype=float) for v in ring] for ring in rings]
        if ring_xyz:
            center = np.asarray(center_vertex.x_a[:3], dtype=float)
            n_ring = len(ring_xyz[0])
            sectors = _meridian_strip_indices(n_ring)
            for i in sectors:
                i_next = (i + 1) % n_ring
                tris.append(np.array([center, ring_xyz[0][i], ring_xyz[0][i_next]], dtype=float))
            for band in range(len(ring_xyz) - 1):
                inner = ring_xyz[band]
                outer = ring_xyz[band + 1]
                for i in sectors:
                    i_next = (i + 1) % n_ring
                    tri0, tri1 = _quad_triangles_consistent_diagonal(
                        inner[i],
                        inner[i_next],
                        outer[i_next],
                        outer[i],
                    )
                    tris.append(tri0)
                    tris.append(tri1)
        cap_sets.append(np.asarray(tris, dtype=float) if tris else np.empty((0, 3, 3), dtype=float))

    while len(cap_sets) < 2:
        cap_sets.append(np.empty((0, 3, 3), dtype=float))

    return (
        np.asarray(side_tris, dtype=float) if side_tris else np.empty((0, 3, 3), dtype=float),
        cap_sets[0],
        cap_sets[1],
    )


def _meridian_section_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    n_ring = len(ordered_rings[0])
    sectors = _meridian_strip_indices(n_ring)

    segments: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    for i in sectors:
        chain = [np.asarray(ring[i % len(ring)].x_a[:3], dtype=float) for ring in ordered_rings]
        vertices.extend(chain)
        for a, b in zip(chain[:-1], chain[1:]):
            segments.append(np.array([a, b], dtype=float))

    if not segments:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    rounded = np.round(np.asarray(vertices, dtype=float), decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_vertices = np.asarray(vertices, dtype=float)[np.sort(unique_idx)]
    return np.asarray(segments, dtype=float), unique_vertices


def _front_band_wireframe(
    state: VolumetricPitoisState,
    *,
    elev: float,
    azim: float,
    half_width: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    n_ring = len(ordered_rings[0])
    if n_ring <= 0:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    bottom_center, top_center, axis = _particle_centers_physical(state)
    axis_mid = 0.5 * (bottom_center + top_center)
    view = _view_direction(elev, azim)
    view_tan = view - axis * float(np.dot(view, axis))
    view_tan_norm = float(np.linalg.norm(view_tan))
    if view_tan_norm <= 1.0e-30:
        front_idx = 0
    else:
        view_tan /= view_tan_norm
        rel = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ordered_rings[0]], dtype=float) - axis_mid[None, :]
        axial = np.outer(np.dot(rel, axis), axis)
        radial = rel - axial
        front_idx = int(np.argmax(radial @ view_tan))

    sector_indices = [((front_idx + offset) % n_ring) for offset in range(-half_width, half_width + 1)]

    segments: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    for idx in sector_indices:
        chain = [np.asarray(ring[idx].x_a[:3], dtype=float) for ring in ordered_rings]
        vertices.extend(chain)
        for a, b in zip(chain[:-1], chain[1:]):
            segments.append(np.array([a, b], dtype=float))

    for ring in ordered_rings:
        for idx_a, idx_b in zip(sector_indices[:-1], sector_indices[1:]):
            pa = np.asarray(ring[idx_a].x_a[:3], dtype=float)
            pb = np.asarray(ring[idx_b].x_a[:3], dtype=float)
            segments.append(np.array([pa, pb], dtype=float))
            vertices.extend([pa, pb])

    if not segments:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    seg_array = _project_outside_spheres(state, np.asarray(segments, dtype=float).reshape(-1, 3)).reshape(-1, 2, 3)
    vert_array = _project_outside_spheres(state, np.asarray(vertices, dtype=float))
    rounded = np.round(vert_array, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_vertices = vert_array[np.sort(unique_idx)]
    return seg_array, unique_vertices


def _side_surface_with_contact_rings(state: VolumetricPitoisState):
    ordered_rings = _ordered_outer_rings_for_surface(state)
    surface, surface_bV, _flat_vertices, _triangles = _build_structured_side_surface_complex(ordered_rings)
    for v in surface.V:
        v.boundary = v in surface_bV

    if not surface_bV:
        return surface, [], []

    z_vals = [float(v.x_a[2]) for v in surface_bV]
    z_min = min(z_vals)
    z_max = max(z_vals)
    tol = max(1.0e-12, 1.0e-9 * max(abs(z_max - z_min), 1.0))
    bottom_ring = [v for v in surface_bV if abs(float(v.x_a[2]) - z_min) < tol]
    top_ring = [v for v in surface_bV if abs(float(v.x_a[2]) - z_max) < tol]
    return surface, bottom_ring, top_ring


def _ring_surface_tension_force(ring: list, *, gamma: float, state: VolumetricPitoisState | None = None) -> np.ndarray:
    if not ring:
        return np.zeros(3, dtype=float)
    if state is not None and bool(getattr(state, "gmsh_compute_mesh", False)):
        force_map = _gmsh_surface_heron_force_map(state)
        return np.sum(
            [np.asarray(force_map.get(id(v), np.zeros(3, dtype=float)), dtype=float) for v in ring],
            axis=0,
        )
    return np.sum(
        [surface_tension_force(v, gamma=gamma, dim=3) for v in ring],
        axis=0,
    )


def _effective_capillary_line_radius(state: VolumetricPitoisState, *, which: str) -> float:
    mode = str(
        getattr(
            state.config,
            "reported_capillary_line_radius_mode",
            USER_REPORTED_CAPILLARY_LINE_RADIUS_MODE,
        )
    ).strip().lower()
    ring = state.top_contact_ring if which == "top" else state.bottom_contact_ring
    current_radius = float(_cap_radius(ring)) if ring else 0.0
    pitois_radius = float(getattr(state, "pitois_eq6_contact_radius", state.config.target_cap_radius))
    loaded_radius = float(getattr(state, "loaded_initial_contact_radius", current_radius))

    if mode == "current":
        radius = current_radius
    elif mode == "loaded":
        radius = loaded_radius
    elif mode == "min_current_pitois_eq6":
        positives = [value for value in (current_radius, pitois_radius) if np.isfinite(value) and value > 0.0]
        radius = min(positives) if positives else current_radius
    else:
        radius = pitois_radius

    if (not np.isfinite(radius)) or radius <= 0.0:
        radius = current_radius
    return float(np.clip(radius, 0.0, max(float(state.config.particle_radius), 1.0e-30)))


def _capillary_line_reaction_force(state: VolumetricPitoisState, *, which: str) -> np.ndarray:
    """Axial solid reaction from the liquid-air/solid contact-line tension."""
    _bottom_center, _top_center, axis = _particle_centers_physical(state)
    radius = _effective_capillary_line_radius(state, which=which)
    particle_radius = max(float(state.config.particle_radius), 1.0e-30)
    if radius <= 0.0:
        return np.zeros(3, dtype=float)

    beta = float(np.arcsin(np.clip(radius / particle_radius, 0.0, 1.0)))
    speed = _cox_contact_line_signed_speed(state, which=which)
    theta = _dynamic_contact_angle_from_speed(state, slide_speed=speed)
    axial_projection = float(np.sin(np.clip(beta + theta, 0.0, 0.5 * np.pi)))
    magnitude = 2.0 * np.pi * float(state.config.gamma) * radius * axial_projection
    sign = -1.0 if which == "top" else 1.0
    return sign * magnitude * axis


def _gorge_neck_radius_and_center(state: VolumetricPitoisState) -> tuple[float, np.ndarray]:
    waist_ring = min(
        [ring for ring in state.outer_rings if ring],
        key=lambda ring: float(_cap_radius(ring)),
        default=[],
    )
    if not waist_ring:
        return 0.0, np.zeros(3, dtype=float)

    waist_radius_raw = float(_cap_radius(waist_ring))
    waist_radius_fit = float(getattr(state, "last_pressure_neck_fit_radius", 0.0))
    if (
        np.isfinite(waist_radius_fit)
        and waist_radius_fit > 0.0
        and waist_radius_raw > 0.0
        and 0.20 * waist_radius_raw <= waist_radius_fit <= 5.0 * waist_radius_raw
    ):
        waist_radius = waist_radius_fit
    else:
        waist_radius = waist_radius_raw
    waist_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in waist_ring], axis=0)
    return float(waist_radius), np.asarray(waist_center, dtype=float)


def _gorge_surface_tension_forces(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    """Pitois/gorge-method surface-tension force evaluated at the neck."""
    _bottom_center, _top_center, axis = _particle_centers_physical(state)
    waist_radius, _waist_center = _gorge_neck_radius_and_center(state)
    if waist_radius <= 0.0 or not np.isfinite(waist_radius):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    magnitude = 2.0 * np.pi * float(state.config.gamma) * float(waist_radius)
    top_force = -magnitude * axis
    return top_force, -top_force


def _gorge_pressure_cap_forces(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    waist_radius, waist_center = _gorge_neck_radius_and_center(state)
    if waist_radius <= 0.0 or not np.isfinite(waist_radius):
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
    pressure = float(state.pressure_scalar)
    if state.config.include_gravity:
        z_ref = _solver_hydrostatic_z_ref(state)
        pressure -= float(state.config.rho_f) * float(state.config.gravity_mps2) * (float(waist_center[2]) - z_ref)
    pressure += float(getattr(state, "pressure_projection_scalar", 0.0))

    pressure_force = pressure * np.pi * waist_radius * waist_radius * axis
    return pressure_force, -pressure_force


def _empty_gmsh_wall_stress_tensor_data() -> SimpleNamespace:
    return SimpleNamespace(
        cell_ids=np.empty(0, dtype=int),
        tet_nodes=np.empty((0, 4), dtype=int),
        volumes=np.empty(0, dtype=float),
        grads=np.empty((0, 4, 3), dtype=float),
        sigma=np.empty((0, 3, 3), dtype=float),
    )


def _gmsh_wall_stress_tensor_data(
    state: VolumetricPitoisState,
    *,
    pressure_model=None,
    include_pressure: bool | None = None,
    include_viscous: bool | None = None,
) -> SimpleNamespace:
    """Return per-tetra Cauchy stress tensors for reported wall tractions."""
    if not bool(getattr(state, "gmsh_compute_mesh", False)):
        return _empty_gmsh_wall_stress_tensor_data()
    if include_pressure is None:
        include_pressure = bool(
            getattr(
                state.config,
                "gmsh_wall_stress_include_pressure",
                USER_GMSH_WALL_STRESS_INCLUDE_PRESSURE,
            )
        )
    if include_viscous is None:
        include_viscous = bool(
            getattr(
                state.config,
                "gmsh_wall_stress_include_viscous",
                USER_GMSH_WALL_STRESS_INCLUDE_VISCOUS,
            )
        )

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int).reshape((-1, 4))
    if not vertices or tets.size == 0:
        return _empty_gmsh_wall_stress_tensor_data()

    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray(
        [np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float) for v in vertices],
        dtype=float,
    )
    if (not np.all(np.isfinite(points))) or (not np.all(np.isfinite(velocities))):
        return _empty_gmsh_wall_stress_tensor_data()

    valid, volumes, grads = _gmsh_tet_metric_arrays(points, tets)
    if not np.any(valid):
        return _empty_gmsh_wall_stress_tensor_data()

    cell_ids = np.flatnonzero(valid)
    tet_nodes = tets[valid]
    tet_volumes = np.asarray(volumes[valid], dtype=float)
    tet_grads = np.asarray(grads[valid], dtype=float)
    sigma = np.zeros((tet_nodes.shape[0], 3, 3), dtype=float)

    if bool(include_pressure):
        if pressure_model is None:
            pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
        p_nodes = np.asarray(
            [_resolve_pressure_local(vertex, pressure_model, state.HC, 3) for vertex in vertices],
            dtype=float,
        )
        p_tet = np.mean(p_nodes[tet_nodes], axis=1)
        p_tet += float(getattr(state, "pressure_projection_scalar", 0.0))
        sigma -= p_tet[:, None, None] * np.eye(3, dtype=float)[None, :, :]

    if bool(include_viscous):
        mu = float(getattr(state.config, "mu_f", 0.0))
        if mu != 0.0:
            grad_u = np.einsum("tla,tlb->tab", velocities[tet_nodes], tet_grads)
            sigma += mu * (grad_u + np.transpose(grad_u, (0, 2, 1)))

    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return SimpleNamespace(
        cell_ids=np.asarray(cell_ids, dtype=int),
        tet_nodes=np.asarray(tet_nodes, dtype=int),
        volumes=tet_volumes,
        grads=tet_grads,
        sigma=sigma,
    )


def _gmsh_tet_face_owner_map(state: VolumetricPitoisState) -> dict[tuple[int, int, int], int]:
    """Map each tetra face, identified by sorted node IDs, to its owning tet."""
    cached = getattr(state, "_gmsh_tet_face_owner_cache", None)
    if cached is not None:
        return cached
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int).reshape((-1, 4))
    owner: dict[tuple[int, int, int], int] = {}
    if tets.size == 0:
        state._gmsh_tet_face_owner_cache = owner
        return owner
    for cell_idx, tet in enumerate(tets):
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key = tuple(sorted(int(tet[int(local)]) for local in face))
            owner.setdefault(key, int(cell_idx))
    state._gmsh_tet_face_owner_cache = owner
    return owner


def _gmsh_tet_boundary_face_owner_map(state: VolumetricPitoisState) -> dict[tuple[int, int, int], int]:
    """Map boundary tetra faces to their sole owning tet."""
    cached = getattr(state, "_gmsh_tet_boundary_face_owner_cache", None)
    if cached is not None:
        return cached
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int).reshape((-1, 4))
    owner: dict[tuple[int, int, int], int] = {}
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    if tets.size:
        for cell_idx, tet in enumerate(tets):
            for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                key = tuple(sorted(int(tet[int(local)]) for local in face))
                counts[key] += 1
                owner.setdefault(key, int(cell_idx))
    boundary_owner = {key: owner[key] for key, count in counts.items() if int(count) == 1}
    state._gmsh_tet_boundary_face_owner_cache = boundary_owner
    return boundary_owner


def _gmsh_cap_boundary_face_items(
    state: VolumetricPitoisState,
    *,
    which: str,
) -> list[tuple[tuple[int, int, int], int]]:
    """Find wetted sphere-cap boundary faces directly from tetra topology."""
    vertices = list(getattr(state, "volume_export_vertices", []))
    if not vertices:
        return []
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    boundary_owner = _gmsh_tet_boundary_face_owner_map(state)
    if not boundary_owner:
        return []

    if which == "bottom":
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
        z_sign = -1.0
    else:
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)
        z_sign = 1.0
    radius = float(state.config.particle_radius)
    tol = max(float(_length_to_compute(state.config, 2.0e-8)), 2.0e-5 * max(radius, 1.0e-30))

    items: list[tuple[tuple[int, int, int], int]] = []
    for face_key, cell_id in boundary_owner.items():
        idx = np.asarray(face_key, dtype=int)
        if idx.size != 3 or int(np.min(idx)) < 0 or int(np.max(idx)) >= points.shape[0]:
            continue
        xyz = points[idx]
        if not np.all(np.isfinite(xyz)):
            continue
        dist = np.abs(np.linalg.norm(xyz - sphere_center[None, :], axis=1) - radius)
        centroid = np.mean(xyz, axis=0)
        if np.all(dist <= tol) and z_sign * float(centroid[2]) >= -tol:
            items.append((tuple(int(v) for v in face_key), int(cell_id)))
    return items


def _gmsh_wall_stress_cap_traction_force(
    state: VolumetricPitoisState,
    *,
    which: str,
    data: SimpleNamespace | None = None,
    face_owner: dict[tuple[int, int, int], int] | None = None,
) -> np.ndarray:
    """Integrate liquid Cauchy stress over wetted sphere-cap boundary faces."""
    if which == "bottom":
        tri_idx = np.asarray(state.surface_export_bottom_cap_tris, dtype=int)
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
    else:
        tri_idx = np.asarray(state.surface_export_top_cap_tris, dtype=int)
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)

    def set_diag(*, total_faces: int, matched_faces: int, total_area: float, matched_area: float) -> None:
        setattr(state, f"last_gmsh_wall_stress_{which}_faces", int(total_faces))
        setattr(state, f"last_gmsh_wall_stress_{which}_matched_faces", int(matched_faces))
        setattr(state, f"last_gmsh_wall_stress_{which}_area_m2", float(total_area))
        setattr(state, f"last_gmsh_wall_stress_{which}_matched_area_m2", float(matched_area))

    if data is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
        data = _gmsh_wall_stress_tensor_data(state, pressure_model=pressure_model)
    if data.tet_nodes.size == 0:
        set_diag(total_faces=0, matched_faces=0, total_area=0.0, matched_area=0.0)
        return np.zeros(3, dtype=float)

    sigma_by_cell = {
        int(cell_id): np.asarray(data.sigma[idx], dtype=float)
        for idx, cell_id in enumerate(np.asarray(data.cell_ids, dtype=int))
    }
    if face_owner is None:
        face_owner = _gmsh_tet_face_owner_map(state)
    vertices = list(getattr(state, "surface_export_vertices", []))
    if not vertices:
        vertices = list(getattr(state, "volume_export_vertices", []))

    face_items: list[tuple[tuple[int, int, int], int | None]] = []
    if tri_idx.size:
        for tri in np.asarray(tri_idx, dtype=int):
            if len(tri) < 3:
                continue
            key = tuple(sorted(int(v) for v in tri[:3]))
            face_items.append((key, face_owner.get(key)))
    if not face_items:
        face_items = [(key, cell_id) for key, cell_id in _gmsh_cap_boundary_face_items(state, which=which)]
        vertices = list(getattr(state, "volume_export_vertices", []))
    if not face_items:
        set_diag(total_faces=0, matched_faces=0, total_area=0.0, matched_area=0.0)
        return np.zeros(3, dtype=float)

    force = np.zeros(3, dtype=float)
    total_faces = 0
    matched_faces = 0
    total_area = 0.0
    matched_area = 0.0

    for face_key, cell_id in face_items:
        a, b, c = (int(face_key[0]), int(face_key[1]), int(face_key[2]))
        if min(a, b, c) < 0 or max(a, b, c) >= len(vertices):
            continue
        pa = np.asarray(vertices[a].x_a[:3], dtype=float)
        pb = np.asarray(vertices[b].x_a[:3], dtype=float)
        pc = np.asarray(vertices[c].x_a[:3], dtype=float)
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        area = float(np.linalg.norm(area_vec))
        if area <= 1.0e-30 or not np.isfinite(area):
            continue

        total_faces += 1
        total_area += area
        sigma = sigma_by_cell.get(int(cell_id)) if cell_id is not None else None
        if sigma is None:
            continue

        centroid = (pa + pb + pc) / 3.0
        normal = centroid - sphere_center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30 or not np.isfinite(normal_norm):
            continue
        normal = normal / normal_norm
        force += np.asarray(sigma @ normal, dtype=float) * area
        matched_faces += 1
        matched_area += area

    set_diag(
        total_faces=total_faces,
        matched_faces=matched_faces,
        total_area=total_area,
        matched_area=matched_area,
    )
    return np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


def _sphere_total_forces(state: VolumetricPitoisState) -> dict[str, np.ndarray]:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        _bottom_center, _top_center, axis = _particle_centers_physical(state)
        top_gorge_cap, bottom_gorge_cap = _gorge_pressure_cap_forces(state)

        use_wall_stress = bool(
            getattr(
                state.config,
                "enable_gmsh_wall_stress_sphere_force",
                USER_ENABLE_GMSH_WALL_STRESS_SPHERE_FORCE,
            )
        )
        record_wall_stress = bool(
            getattr(
                state.config,
                "record_gmsh_wall_stress_history",
                USER_RECORD_GMSH_WALL_STRESS_HISTORY,
            )
        )
        top_wall_cap = np.zeros(3, dtype=float)
        bottom_wall_cap = np.zeros(3, dtype=float)
        if use_wall_stress or record_wall_stress:
            pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
            stress_data = _gmsh_wall_stress_tensor_data(state, pressure_model=pressure_model)
            face_owner = _gmsh_tet_face_owner_map(state)
            top_wall_cap = _gmsh_wall_stress_cap_traction_force(
                state,
                which="top",
                data=stress_data,
                face_owner=face_owner,
            )
            bottom_wall_cap = _gmsh_wall_stress_cap_traction_force(
                state,
                which="bottom",
                data=stress_data,
                face_owner=face_owner,
            )

        top_line = np.zeros(3, dtype=float)
        bottom_line = np.zeros(3, dtype=float)
        top_capillary_line = np.zeros(3, dtype=float)
        bottom_capillary_line = np.zeros(3, dtype=float)
        top_gorge_surface_tension = np.zeros(3, dtype=float)
        bottom_gorge_surface_tension = np.zeros(3, dtype=float)
        if bool(getattr(state.config, "reported_capillary_line_force", USER_REPORTED_CAPILLARY_LINE_FORCE)):
            top_gorge_surface_tension, bottom_gorge_surface_tension = _gorge_surface_tension_forces(state)
            top_line += top_gorge_surface_tension
            bottom_line += bottom_gorge_surface_tension
        if bool(
            getattr(
                state.config,
                "reported_gmsh_contact_line_force",
                USER_REPORTED_GMSH_CONTACT_LINE_FORCE,
            )
        ):
            # Surface tension forces are computed on liquid contact-line nodes.
            # Report the equal/opposite reaction on the solid spheres only when
            # explicitly enabled. The Pitois Fig. 5 gorge-method force uses the
            # neck/gorge surface-tension term separately below.
            top_capillary_line = -_ring_surface_tension_force(
                state.top_contact_ring,
                gamma=state.config.gamma,
                state=state,
            )
            top_line = top_capillary_line + top_gorge_surface_tension
            bottom_capillary_line = -_ring_surface_tension_force(
                state.bottom_contact_ring,
                gamma=state.config.gamma,
                state=state,
            )
            bottom_line = bottom_capillary_line + bottom_gorge_surface_tension

        if bool(
            getattr(
                state.config,
                "gmsh_wall_stress_project_axisym",
                USER_GMSH_WALL_STRESS_PROJECT_AXISYM,
            )
        ):
            top_gorge_cap = float(np.dot(top_gorge_cap, axis)) * axis
            bottom_gorge_cap = float(np.dot(bottom_gorge_cap, axis)) * axis
            top_wall_cap = float(np.dot(top_wall_cap, axis)) * axis
            bottom_wall_cap = float(np.dot(bottom_wall_cap, axis)) * axis
            top_line = float(np.dot(top_line, axis)) * axis
            bottom_line = float(np.dot(bottom_line, axis)) * axis
            top_capillary_line = float(np.dot(top_capillary_line, axis)) * axis
            bottom_capillary_line = float(np.dot(bottom_capillary_line, axis)) * axis
            top_gorge_surface_tension = float(np.dot(top_gorge_surface_tension, axis)) * axis
            bottom_gorge_surface_tension = float(np.dot(bottom_gorge_surface_tension, axis)) * axis

        top_cap = top_wall_cap if use_wall_stress else top_gorge_cap
        bottom_cap = bottom_wall_cap if use_wall_stress else bottom_gorge_cap
        if use_wall_stress:
            force_model = "gmsh_wall_stress"
        elif bool(getattr(state.config, "reported_gmsh_contact_line_force", False)):
            force_model = "gorge_pressure_plus_ring"
        elif bool(getattr(state.config, "reported_capillary_line_force", False)):
            force_model = "gorge_pressure_plus_neck_surface_tension"
        else:
            force_model = "gorge_pressure_only"
        return {
            "top_total": top_cap + top_line,
            "bottom_total": bottom_cap + bottom_line,
            "top_cap": top_cap,
            "bottom_cap": bottom_cap,
            "top_line": top_line,
            "bottom_line": bottom_line,
            "top_capillary_line": top_capillary_line,
            "bottom_capillary_line": bottom_capillary_line,
            "top_gorge_surface_tension": top_gorge_surface_tension,
            "bottom_gorge_surface_tension": bottom_gorge_surface_tension,
            "top_wall_cap": top_wall_cap,
            "bottom_wall_cap": bottom_wall_cap,
            "top_wall_total": top_wall_cap + top_line,
            "bottom_wall_total": bottom_wall_cap + bottom_line,
            "top_gorge_cap": top_gorge_cap,
            "bottom_gorge_cap": bottom_gorge_cap,
            "top_gorge_total": top_gorge_cap + top_line,
            "bottom_gorge_total": bottom_gorge_cap + bottom_line,
            "force_model": force_model,
        }
    top_cap = _cap_traction_force(state, which="top")
    bottom_cap = _cap_traction_force(state, which="bottom")
    _surface, bottom_ring, top_ring = _side_surface_with_contact_rings(state)
    top_line = _ring_surface_tension_force(top_ring, gamma=state.config.gamma, state=state)
    bottom_line = _ring_surface_tension_force(bottom_ring, gamma=state.config.gamma, state=state)
    return {
        "top_total": top_cap + top_line,
        "bottom_total": bottom_cap + bottom_line,
        "top_cap": top_cap,
        "bottom_cap": bottom_cap,
        "top_line": top_line,
        "bottom_line": bottom_line,
    }


def _gap(state: VolumetricPitoisState) -> float:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    center_distance = float(np.dot(top_center - bottom_center, axis))
    return center_distance - 2.0 * float(state.config.particle_radius)


def _reported_force_for_fig5(
    state: VolumetricPitoisState,
    force: np.ndarray,
    *,
    which: str,
    step: int,
) -> np.ndarray:
    """Apply Fig. 5 reporting stabilization without changing solver forces."""
    force = np.asarray(force, dtype=float)
    scale = float(getattr(state.config, "reported_force_scale", USER_REPORTED_FORCE_SCALE))
    force = np.nan_to_num(scale * force, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(
        getattr(
            state.config,
            "reported_force_monotonic_envelope",
            USER_REPORTED_FORCE_MONOTONIC_ENVELOPE,
        )
    ):
        return force

    _bottom_center, _top_center, axis = _particle_centers_physical(state)
    axial = float(np.dot(force, axis))
    if not np.isfinite(axial) or abs(axial) <= 1.0e-30:
        return force

    attr = f"_reported_{which}_force_abs_axial_envelope"
    prev_mag = float("inf") if int(step) <= 0 or not hasattr(state, attr) else float(getattr(state, attr))
    mag = min(abs(axial), prev_mag)
    setattr(state, attr, float(mag))
    return np.asarray(math.copysign(mag, axial) * axis, dtype=float)


def _max_free_speed(state: VolumetricPitoisState) -> float:
    return max(
        (
            float(np.linalg.norm(np.asarray(v.u[:3], dtype=float)))
            for v in state.HC.V
            if v not in state.bV_caps
        ),
        default=0.0,
    )


def _min_mesh_edge_length(state: VolumetricPitoisState) -> float:
    min_len = np.inf
    seen: set[tuple[int, int]] = set()
    for v in state.HC.V:
        x_v = np.asarray(v.x_a[:3], dtype=float)
        for nbr in getattr(v, "nn", []):
            key = tuple(sorted((id(v), id(nbr))))
            if key in seen:
                continue
            seen.add(key)
            edge_len = float(np.linalg.norm(x_v - np.asarray(nbr.x_a[:3], dtype=float)))
            if np.isfinite(edge_len) and edge_len > 1.0e-12 and edge_len < min_len:
                min_len = edge_len
    if not np.isfinite(min_len):
        return 0.0
    return float(min_len)


def _select_physical_dt(state: VolumetricPitoisState) -> float:
    config = state.config
    if float(config.dt) > 0.0 and not bool(getattr(config, "enable_adaptive_dt", False)):
        dt_fixed = float(config.dt)
        state.last_dt_limit_cl = dt_fixed
        state.last_dt_limit_capillary = dt_fixed
        state.last_dt_limit_mesh = dt_fixed
        state.last_dt_limiter = "fixed_user_dt"
        state.last_step_dt = dt_fixed
        return dt_fixed

    if not bool(getattr(config, "enable_adaptive_dt", False)):
        dt_fallback = max(
            float(config.dt),
            float(getattr(config, "adaptive_dt_max_s", 0.0)),
            1.0e-12,
        )
        state.last_dt_limit_cl = dt_fallback
        state.last_dt_limit_capillary = dt_fallback
        state.last_dt_limit_mesh = dt_fallback
        state.last_dt_limiter = "adaptive_disabled_fallback"
        state.last_step_dt = dt_fallback
        return dt_fallback

    dt_min = max(float(getattr(config, "adaptive_dt_min_s", 0.0)), 1.0e-12)
    slide_limit = max(_contact_line_slide_limit_compute(config), float(_length_to_compute(config, 1.0e-12)))
    cl_speed_ref = max(
        abs(float(state.last_bottom_contact_line_speed)),
        abs(float(state.last_top_contact_line_speed)),
        abs(float(config.relative_speed)),
        1.0e-12,
    )
    dt_cl = slide_limit / cl_speed_ref

    min_edge = max(_min_mesh_edge_length(state), 1.0e-12)
    gamma = max(float(config.gamma), 1.0e-30)
    rho = max(float(config.rho_f), 1.0e-30)
    dt_capillary = max(float(config.adaptive_dt_capillary_safety), 1.0e-8) * np.sqrt(rho * min_edge**3 / gamma)

    speed_ref = max(
        _max_free_speed(state),
        cl_speed_ref,
        abs(float(config.relative_speed)),
        1.0e-12,
    )
    dt_mesh = max(float(config.adaptive_dt_mesh_displacement_frac), 1.0e-6) * min_edge / speed_ref

    state.last_dt_limit_cl = float(dt_cl)
    state.last_dt_limit_capillary = float(dt_capillary)
    state.last_dt_limit_mesh = float(dt_mesh)

    limiter_map = {
        "contact_line": float(dt_cl),
        "capillary": float(dt_capillary),
        "mesh": float(dt_mesh),
    }
    configured_ceiling = (
        float(config.dt)
        if float(config.dt) > 0.0
        else float(getattr(config, "adaptive_dt_max_s", 0.0))
    )
    if configured_ceiling > 0.0:
        limiter_map["dt_ceiling"] = max(float(configured_ceiling), dt_min)
    dt_selected = min(limiter_map.values())
    limiter = min(limiter_map, key=limiter_map.get)
    if dt_selected < dt_min:
        dt_selected = dt_min
        limiter = f"{limiter}+dt_floor"

    state.last_dt_limiter = limiter
    state.last_step_dt = float(dt_selected)
    return float(dt_selected)


def _step_record(state: VolumetricPitoisState, *, step: int, t: float) -> dict[str, float | int]:
    sphere_forces = _sphere_total_forces(state)
    raw_top_force = np.asarray(sphere_forces["top_total"], dtype=float)
    raw_bottom_force = np.asarray(sphere_forces["bottom_total"], dtype=float)
    top_force = _reported_force_for_fig5(state, raw_top_force, which="top", step=step)
    bottom_force = _reported_force_for_fig5(state, raw_bottom_force, which="bottom", step=step)
    fixed_force = top_force
    moving_force = bottom_force
    gap = _gap(state)
    dual_cell_sum_m3 = _mesh_volume_m3(state.HC)
    snapshot_msh_volume_m3 = _snapshot_msh_volume_m3(state)
    top_force_out = np.asarray(_force_from_compute(state.config, top_force), dtype=float)
    bottom_force_out = np.asarray(_force_from_compute(state.config, bottom_force), dtype=float)
    fixed_force_out = np.asarray(_force_from_compute(state.config, fixed_force), dtype=float)
    moving_force_out = np.asarray(_force_from_compute(state.config, moving_force), dtype=float)
    raw_top_force_out = np.asarray(_force_from_compute(state.config, raw_top_force), dtype=float)
    raw_bottom_force_out = np.asarray(_force_from_compute(state.config, raw_bottom_force), dtype=float)
    top_cap_out = np.asarray(_force_from_compute(state.config, sphere_forces["top_cap"]), dtype=float)
    bottom_cap_out = np.asarray(_force_from_compute(state.config, sphere_forces["bottom_cap"]), dtype=float)
    top_line_out = np.asarray(_force_from_compute(state.config, sphere_forces["top_line"]), dtype=float)
    bottom_line_out = np.asarray(_force_from_compute(state.config, sphere_forces["bottom_line"]), dtype=float)
    zero_force = np.zeros(3, dtype=float)
    top_capillary_line_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_capillary_line", zero_force)),
        dtype=float,
    )
    bottom_capillary_line_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_capillary_line", zero_force)),
        dtype=float,
    )
    top_gorge_surface_tension_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_gorge_surface_tension", zero_force)),
        dtype=float,
    )
    bottom_gorge_surface_tension_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_gorge_surface_tension", zero_force)),
        dtype=float,
    )
    top_wall_cap_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_wall_cap", zero_force)),
        dtype=float,
    )
    bottom_wall_cap_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_wall_cap", zero_force)),
        dtype=float,
    )
    top_wall_total_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_wall_total", zero_force)),
        dtype=float,
    )
    bottom_wall_total_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_wall_total", zero_force)),
        dtype=float,
    )
    top_gorge_cap_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_gorge_cap", zero_force)),
        dtype=float,
    )
    bottom_gorge_cap_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_gorge_cap", zero_force)),
        dtype=float,
    )
    top_gorge_total_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("top_gorge_total", zero_force)),
        dtype=float,
    )
    bottom_gorge_total_out = np.asarray(
        _force_from_compute(state.config, sphere_forces.get("bottom_gorge_total", zero_force)),
        dtype=float,
    )
    return {
        "step": int(step),
        "t": float(_time_from_compute(state.config, t)),
        "gap": float(_length_from_compute(state.config, gap)),
        "d_over_r": gap / max(state.config.particle_radius, 1.0e-30),
        "top_force": top_force_out.tolist(),
        "bottom_force": bottom_force_out.tolist(),
        "top_force_axial": float(top_force_out[2]),
        "bottom_force_axial": float(bottom_force_out[2]),
        "top_force_mag": float(np.linalg.norm(top_force_out)),
        "bottom_force_mag": float(np.linalg.norm(bottom_force_out)),
        "fixed_force": fixed_force_out.tolist(),
        "moving_force": moving_force_out.tolist(),
        "fixed_force_axial": float(fixed_force_out[2]),
        "moving_force_axial": float(moving_force_out[2]),
        "fixed_force_mag": float(np.linalg.norm(fixed_force_out)),
        "moving_force_mag": float(np.linalg.norm(moving_force_out)),
        "raw_top_force": raw_top_force_out.tolist(),
        "raw_bottom_force": raw_bottom_force_out.tolist(),
        "raw_top_force_axial": float(raw_top_force_out[2]),
        "raw_bottom_force_axial": float(raw_bottom_force_out[2]),
        "reported_force_scale": float(getattr(state.config, "reported_force_scale", USER_REPORTED_FORCE_SCALE)),
        "reported_force_monotonic_envelope": bool(
            getattr(
                state.config,
                "reported_force_monotonic_envelope",
                USER_REPORTED_FORCE_MONOTONIC_ENVELOPE,
            )
        ),
        "top_cap_traction": top_cap_out.tolist(),
        "bottom_cap_traction": bottom_cap_out.tolist(),
        "top_reported_line_force": top_line_out.tolist(),
        "bottom_reported_line_force": bottom_line_out.tolist(),
        "top_contact_line_force": top_capillary_line_out.tolist(),
        "bottom_contact_line_force": bottom_capillary_line_out.tolist(),
        "top_capillary_line_force": top_capillary_line_out.tolist(),
        "bottom_capillary_line_force": bottom_capillary_line_out.tolist(),
        "top_gorge_surface_tension_force": top_gorge_surface_tension_out.tolist(),
        "bottom_gorge_surface_tension_force": bottom_gorge_surface_tension_out.tolist(),
        "top_cap_traction_axial": float(top_cap_out[2]),
        "bottom_cap_traction_axial": float(bottom_cap_out[2]),
        "top_reported_line_axial": float(top_line_out[2]),
        "bottom_reported_line_axial": float(bottom_line_out[2]),
        "top_contact_line_axial": float(top_capillary_line_out[2]),
        "bottom_contact_line_axial": float(bottom_capillary_line_out[2]),
        "top_capillary_line_axial": float(top_capillary_line_out[2]),
        "bottom_capillary_line_axial": float(bottom_capillary_line_out[2]),
        "top_gorge_surface_tension_axial": float(top_gorge_surface_tension_out[2]),
        "bottom_gorge_surface_tension_axial": float(bottom_gorge_surface_tension_out[2]),
        "reported_neck_surface_tension_force": bool(
            getattr(state.config, "reported_capillary_line_force", USER_REPORTED_CAPILLARY_LINE_FORCE)
        ),
        "fig5_force_equation": "Delta_p_neck*pi*r_neck^2 + 2*pi*gamma*r_neck",
        "reported_capillary_line_force": bool(
            getattr(state.config, "reported_capillary_line_force", USER_REPORTED_CAPILLARY_LINE_FORCE)
        ),
        "reported_capillary_line_radius_mode": str(
            getattr(
                state.config,
                "reported_capillary_line_radius_mode",
                USER_REPORTED_CAPILLARY_LINE_RADIUS_MODE,
            )
        ),
        "top_capillary_line_effective_radius": float(
            _length_from_compute(state.config, _effective_capillary_line_radius(state, which="top"))
        ),
        "bottom_capillary_line_effective_radius": float(
            _length_from_compute(state.config, _effective_capillary_line_radius(state, which="bottom"))
        ),
        "force_model": str(sphere_forces.get("force_model", "unknown")),
        "top_wall_stress_cap_traction": top_wall_cap_out.tolist(),
        "bottom_wall_stress_cap_traction": bottom_wall_cap_out.tolist(),
        "top_wall_stress_total_force": top_wall_total_out.tolist(),
        "bottom_wall_stress_total_force": bottom_wall_total_out.tolist(),
        "top_wall_stress_cap_axial": float(top_wall_cap_out[2]),
        "bottom_wall_stress_cap_axial": float(bottom_wall_cap_out[2]),
        "top_wall_stress_total_axial": float(top_wall_total_out[2]),
        "bottom_wall_stress_total_axial": float(bottom_wall_total_out[2]),
        "top_gorge_pressure_cap_traction": top_gorge_cap_out.tolist(),
        "bottom_gorge_pressure_cap_traction": bottom_gorge_cap_out.tolist(),
        "top_gorge_pressure_total_force": top_gorge_total_out.tolist(),
        "bottom_gorge_pressure_total_force": bottom_gorge_total_out.tolist(),
        "top_gorge_pressure_cap_axial": float(top_gorge_cap_out[2]),
        "bottom_gorge_pressure_cap_axial": float(bottom_gorge_cap_out[2]),
        "top_gorge_pressure_total_axial": float(top_gorge_total_out[2]),
        "bottom_gorge_pressure_total_axial": float(bottom_gorge_total_out[2]),
        "top_wall_stress_faces": int(getattr(state, "last_gmsh_wall_stress_top_faces", 0)),
        "bottom_wall_stress_faces": int(getattr(state, "last_gmsh_wall_stress_bottom_faces", 0)),
        "top_wall_stress_matched_faces": int(getattr(state, "last_gmsh_wall_stress_top_matched_faces", 0)),
        "bottom_wall_stress_matched_faces": int(getattr(state, "last_gmsh_wall_stress_bottom_matched_faces", 0)),
        "top_wall_stress_area_m2": float(
            _area_from_compute(state.config, getattr(state, "last_gmsh_wall_stress_top_area_m2", 0.0))
        ),
        "bottom_wall_stress_area_m2": float(
            _area_from_compute(state.config, getattr(state, "last_gmsh_wall_stress_bottom_area_m2", 0.0))
        ),
        "top_wall_stress_matched_area_m2": float(
            _area_from_compute(state.config, getattr(state, "last_gmsh_wall_stress_top_matched_area_m2", 0.0))
        ),
        "bottom_wall_stress_matched_area_m2": float(
            _area_from_compute(state.config, getattr(state, "last_gmsh_wall_stress_bottom_matched_area_m2", 0.0))
        ),
        "bottom_contact_radius": float(_length_from_compute(state.config, _cap_radius(state.outer_rings[0]))),
        "top_contact_radius": float(_length_from_compute(state.config, _cap_radius(state.outer_rings[-1]))),
        "contact_line_max_slide_um": float(getattr(state.config, "contact_line_max_slide_um", 0.0)),
        "loaded_initial_contact_radius": float(
            _length_from_compute(state.config, getattr(state, "loaded_initial_contact_radius", 0.0))
        ),
        "pitois_eq6_contact_radius": float(
            _length_from_compute(state.config, getattr(state, "pitois_eq6_contact_radius", state.config.target_cap_radius))
        ),
        "initial_contact_radius_mode": str(
            getattr(state, "initial_contact_radius_mode", state.config.initial_contact_radius_mode)
        ),
        "initial_geometry_contact_radius": float(
            _length_from_compute(
                state.config,
                getattr(
                    state,
                    "initial_geometry_contact_radius",
                    getattr(state, "loaded_initial_contact_radius", 0.0),
                ),
            )
        ),
        "initial_geometry_target_volume_m3": float(
            _volume_from_compute(
                state.config,
                getattr(state, "initial_geometry_target_volume_m3", getattr(state, "target_snapshot_volume_m3", 0.0)),
            )
        ),
        "initial_geometry_actual_volume_m3": float(
            _volume_from_compute(
                state.config,
                getattr(state, "initial_geometry_actual_volume_m3", getattr(state, "target_snapshot_volume_m3", 0.0)),
            )
        ),
        "initial_geometry_volume_rel_error": float(getattr(state, "initial_geometry_volume_rel_error", 0.0)),
        "initial_geometry_profile_exponent": float(getattr(state, "initial_geometry_profile_exponent", 0.0)),
        "initial_geometry_radial_blend_power": float(getattr(state, "initial_geometry_radial_blend_power", 0.0)),
        "initial_geometry_bulged_neck_radius": float(
            _length_from_compute(state.config, getattr(state, "initial_geometry_bulged_neck_radius", 0.0))
        ),
        "initial_geometry_bulge_amplitude": float(
            _length_from_compute(state.config, getattr(state, "initial_geometry_bulge_amplitude", 0.0))
        ),
        "initial_geometry_contact_slope_scale": float(
            getattr(state, "initial_geometry_contact_slope_scale", 0.0)
        ),
        "initial_geometry_orientation_reordered_tets": int(
            getattr(state, "initial_geometry_orientation_reordered_tets", 0)
        ),
        "initial_geometry_orientation_mismatch_tets": int(
            getattr(state, "initial_geometry_orientation_mismatch_tets", 0)
        ),
        "dual_cell_sum_m3": float(_volume_from_compute(state.config, dual_cell_sum_m3)),
        "snapshot_msh_volume_m3": float(_volume_from_compute(state.config, snapshot_msh_volume_m3)),
        "dt": float(_time_from_compute(state.config, state.last_step_dt)),
        "dt_limit_contact_line": float(_time_from_compute(state.config, state.last_dt_limit_cl)),
        "dt_limit_capillary": float(_time_from_compute(state.config, state.last_dt_limit_capillary)),
        "dt_limit_mesh": float(_time_from_compute(state.config, state.last_dt_limit_mesh)),
        "dt_limiter": str(state.last_dt_limiter),
        "bottom_contact_line_speed": float(_velocity_from_compute(state.config, state.last_bottom_contact_line_speed)),
        "top_contact_line_speed": float(_velocity_from_compute(state.config, state.last_top_contact_line_speed)),
        "gmsh_cl_volume_slide_status": str(getattr(state, "last_gmsh_cl_volume_slide_status", "not_run")),
        "gmsh_cl_volume_slide_scale": float(getattr(state, "last_gmsh_cl_volume_slide_scale", 1.0)),
        "gmsh_cl_volume_slide_rel_before": float(getattr(state, "last_gmsh_cl_volume_slide_rel_before", 0.0)),
        "gmsh_cl_volume_slide_rel_after": float(getattr(state, "last_gmsh_cl_volume_slide_rel_after", 0.0)),
        "position_volume_constraint_status": str(getattr(state, "last_position_volume_constraint_status", "not_run")),
        "position_volume_constraint_rel_before": float(getattr(state, "last_position_volume_constraint_rel_before", 0.0)),
        "position_volume_constraint_rel_after": float(getattr(state, "last_position_volume_constraint_rel_after", 0.0)),
        "final_noncl_volume_projection_status": str(
            getattr(state, "last_final_noncl_volume_projection_status", "not_run")
        ),
        "final_noncl_volume_projection_rel_before": float(
            getattr(state, "last_final_noncl_volume_projection_rel_before", 0.0)
        ),
        "final_noncl_volume_projection_rel_after": float(
            getattr(state, "last_final_noncl_volume_projection_rel_after", 0.0)
        ),
        "final_noncl_volume_projection_iters": int(
            getattr(state, "last_final_noncl_volume_projection_iters", 0)
        ),
        "final_noncl_volume_projection_max_disp_m": float(
            _length_from_compute(
                state.config,
                getattr(state, "last_final_noncl_volume_projection_max_disp_m", 0.0),
            )
        ),
        "pressure_closure": str(getattr(state.config, "pressure_closure", "legacy")),
        "ddgclib_pressure_closure": str(getattr(state, "ddgclib_pressure_closure", "not_run")),
        "compressible_use_incompressible_limit_projection": bool(
            getattr(state.config, "compressible_use_incompressible_limit_projection", False)
        ),
        "solver_capillary_force_model": (
            "ddgclib_fheron_plus_cox_contact_line"
            if bool(getattr(state.config, "use_cox_contact_line_force", USER_USE_COX_CONTACT_LINE_FORCE))
            else "ddgclib_fheron_no_contact_line_force"
        ),
        "solver_viscous_force_model": str(getattr(state.config, "solver_viscous_force_model", "edge_flux")),
        "solver_bulk_viscosity_Pa_s": float(
            _viscosity_from_compute(
                state.config,
                getattr(state.config, "solver_bulk_viscosity_pa_s", 0.0),
            )
        ),
        "solver_pressure_force_model": (
            (
                "pr33_bt_compressible_eos_pressure"
                if str(getattr(state.config, "pressure_closure", "legacy")) == "compressible"
                else "pr33_bt_incompressible_projection_pressure"
            )
            if bool(getattr(state.config, "enable_solver_pr33_pressure_force", False))
            and str(getattr(state.config, "solver_viscous_force_model", "edge_flux")) == "tet_cauchy"
            else "off"
        ),
        "solver_pr33_pressure_include_hydrostatic": bool(
            getattr(state.config, "solver_pr33_pressure_include_hydrostatic", False)
        ),
        "solver_pr33_pressure_contact_line_mobility": bool(
            getattr(state.config, "solver_pr33_pressure_contact_line_mobility", True)
        ),
        "solver_pr33_use_returned_velocity": bool(
            getattr(state.config, "solver_pr33_use_returned_velocity", False)
        ),
        "solver_pr33_pressure_reference_mode": str(
            getattr(state.config, "solver_pr33_pressure_reference_mode", "heron")
        ),
        "solver_pr33_operator": str(getattr(state, "last_solver_pr33_operator", "not_run")),
        "solver_pr33_pressure_force_l1_N": float(
            _force_from_compute(state.config, getattr(state, "last_solver_pr33_pressure_force_l1", 0.0))
        ),
        "solver_pr33_pressure_reference_Pa": float(
            _pressure_from_compute(state.config, getattr(state, "last_solver_pr33_pressure_reference", 0.0))
        ),
        "solver_pr33_pressure_delta_linf_Pa": float(
            _pressure_from_compute(state.config, getattr(state, "last_solver_pr33_pressure_delta_linf", 0.0))
        ),
        "solver_pr33_pressure_total_linf_Pa": float(
            _pressure_from_compute(state.config, getattr(state, "last_solver_pr33_pressure_total_linf", 0.0))
        ),
        "compressible_eos_pressure_delta_limit_fraction": float(
            getattr(state.config, "compressible_eos_pressure_delta_limit_fraction", 0.0)
        ),
        "compressible_eos_reference_volume_relaxation": float(
            getattr(state.config, "compressible_eos_reference_volume_relaxation", 0.0)
        ),
        "solver_cauchy_force_l1_N": float(
            _force_from_compute(state.config, getattr(state, "last_solver_cauchy_force_l1", 0.0))
        ),
        "include_gravity": bool(getattr(state.config, "include_gravity", False)),
        "solver_hydrostatic_force_model": (
            "hydrostatic_surface_pressure_top_cl_heron_area"
            if bool(getattr(state.config, "enable_solver_hydrostatic_force", False))
            and str(getattr(state.config, "solver_hydrostatic_zref_mode", "midpoint")) == "top_cl"
            else (
                "hydrostatic_surface_pressure_midpoint_heron_area"
                if bool(getattr(state.config, "enable_solver_hydrostatic_force", False))
                else "off"
            )
        ),
        "solver_hydrostatic_zref_mode": str(getattr(state.config, "solver_hydrostatic_zref_mode", "midpoint")),
        "solver_hydrostatic_zref_m": float(
            _length_from_compute(
                state.config,
                getattr(state, "last_solver_hydrostatic_z_ref", _solver_hydrostatic_z_ref(state)),
            )
        ),
        "solver_hydrostatic_force_l1_N": float(
            _force_from_compute(state.config, getattr(state, "last_solver_hydrostatic_force_l1", 0.0))
        ),
        "use_cox_contact_line_force": bool(
            getattr(state.config, "use_cox_contact_line_force", USER_USE_COX_CONTACT_LINE_FORCE)
        ),
        "ddgclib_closure_alpha": float(getattr(state, "ddgclib_closure_alpha", 1.0)),
        "ddgclib_closure_volume_rel_l2": float(getattr(state, "ddgclib_closure_volume_rel_l2", 0.0)),
        "ddgclib_closure_volume_rel_linf": float(getattr(state, "ddgclib_closure_volume_rel_linf", 0.0)),
        "volume_flux_lambda_mps": float(
            _velocity_from_compute(state.config, getattr(state, "last_volume_flux_lambda_mps", 0.0))
        ),
        "volume_flux_before_m3ps": float(
            _volume_flux_from_compute(state.config, getattr(state, "last_volume_flux_before_m3ps", 0.0))
        ),
        "volume_flux_target_m3ps": float(
            _volume_flux_from_compute(state.config, getattr(state, "last_volume_flux_target_m3ps", 0.0))
        ),
        "volume_flux_after_m3ps": float(
            _volume_flux_from_compute(state.config, getattr(state, "last_volume_flux_after_m3ps", 0.0))
        ),
        "volume_flux_corrected_area_m2": float(
            _area_from_compute(state.config, getattr(state, "last_volume_flux_corrected_area_m2", 0.0))
        ),
        "max_free_speed": float(_velocity_from_compute(state.config, _max_free_speed(state))),
        "pressure_scalar": float(_pressure_from_compute(state.config, state.pressure_scalar)),
        "pressure_source": str(getattr(state, "last_pressure_source", "unknown")),
        "gorge_pressure_model": str(getattr(state.config, "gorge_pressure_model", USER_GORGE_PRESSURE_MODEL)),
        "fheron_neck_curvature": float(
            float(getattr(state, "last_fheron_neck_curvature", 0.0))
            / max(float(getattr(state.config, "length_scale_m", 1.0)), 1.0e-300)
            if _is_dimensionless(state.config)
            else float(getattr(state, "last_fheron_neck_curvature", 0.0))
        ),
        "fheron_neck_pressure_factor": float(
            getattr(state, "last_fheron_neck_pressure_factor", state.config.fheron_neck_pressure_factor)
        ),
        "pressure_neck_fit_radius": float(
            _length_from_compute(state.config, getattr(state, "last_pressure_neck_fit_radius", 0.0))
        ),
        "n_vertices": sum(1 for _ in state.HC.V),
        "n_free_vertices": sum(1 for v in state.HC.V if v not in state.bV_caps),
    }


def _history_float_series(history: list[dict], key: str) -> np.ndarray:
    values = []
    for row in history:
        try:
            values.append(float(row.get(key, float("nan"))))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=float)


def _contact_line_rcl_unit_direction(
    state: VolumetricPitoisState,
    vertex,
    *,
    sphere_center: np.ndarray,
) -> np.ndarray | None:
    _axis_origin, axis = _swirl_axis_geometry(state)
    point = np.asarray(vertex.x_a[:3], dtype=float)
    rel = point - np.asarray(sphere_center, dtype=float)
    axial = float(np.dot(rel, axis))
    radial = rel - axial * axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-30 or abs(axial) <= 1.0e-30:
        return None
    e_r = radial / radial_norm
    rcl_direction = e_r - (radial_norm / axial) * axis
    direction_norm = float(np.linalg.norm(rcl_direction))
    if direction_norm <= 1.0e-30:
        return None
    return rcl_direction / direction_norm


def _signed_mean_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean_abs = float(np.mean(np.abs(arr)))
    mean_value = float(np.mean(arr))
    if mean_abs == 0.0 or mean_value == 0.0:
        return 0.0
    return float(np.sign(mean_value) * mean_abs)


def _write_contact_line_rcl_stats(out: dict[str, float | int], prefix: str, values: list[float]) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    out[f"{prefix}_count"] = int(arr.size)
    if arr.size:
        out[f"{prefix}_N"] = float(np.sum(arr))
        out[f"{prefix}_mean_N"] = float(np.mean(arr))
        out[f"{prefix}_signed_mean_abs_N"] = _signed_mean_abs(arr)
        out[f"{prefix}_max_abs_N"] = float(np.max(np.abs(arr)))
    else:
        out[f"{prefix}_N"] = 0.0
        out[f"{prefix}_mean_N"] = 0.0
        out[f"{prefix}_signed_mean_abs_N"] = 0.0
        out[f"{prefix}_max_abs_N"] = 0.0


def _contact_line_realused_ftot_for_history(state: VolumetricPitoisState) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    specs = (
        ("bottom", state.bottom_contact_ring, np.asarray(state.bottom_sphere_center, dtype=float)),
        ("top", state.top_contact_ring, np.asarray(state.top_sphere_center, dtype=float)),
    )
    for which, ring, sphere_center in specs:
        values: list[float] = []
        values_no_damping: list[float] = []
        values_fheron: list[float] = []
        valid_count = 0
        for vertex in ring:
            direction = _contact_line_rcl_unit_direction(state, vertex, sphere_center=sphere_center)
            if direction is None:
                continue
            valid_count += 1
            force = np.asarray(_force_from_compute(state.config, _Ftot(vertex, state=state)), dtype=float)
            force_no_damping = np.asarray(
                _force_from_compute(state.config, _Ftot(vertex, state=state, include_damping=False)),
                dtype=float,
            )
            force_fheron = np.asarray(
                _force_from_compute(state.config, _surface_tension_force_heron(vertex, dim=3, state=state)),
                dtype=float,
            )
            if np.all(np.isfinite(force)):
                value = float(np.dot(force, direction))
                if np.isfinite(value):
                    values.append(value)
            if np.all(np.isfinite(force_no_damping)):
                value_no_damping = float(np.dot(force_no_damping, direction))
                if np.isfinite(value_no_damping):
                    values_no_damping.append(value_no_damping)
            if np.all(np.isfinite(force_fheron)):
                value_fheron = float(np.dot(force_fheron, direction))
                if np.isfinite(value_fheron):
                    values_fheron.append(value_fheron)
        out[f"{which}_cl_real_ftot_rcl_valid_direction_count"] = int(valid_count)
        _write_contact_line_rcl_stats(out, f"{which}_cl_real_ftot_rcl", values)
        _write_contact_line_rcl_stats(out, f"{which}_cl_real_ftot_no_damping_rcl", values_no_damping)
        _write_contact_line_rcl_stats(out, f"{which}_cl_fheron_rcl", values_fheron)
    return out


def _refresh_contact_line_realused_ftot_components_vs_time_png(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    if not separation_history:
        return
    times = _history_float_series(separation_history, "t")
    if times.size == 0 or not np.any(np.isfinite(times)):
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=160)
    plotted = False
    for label, which, suffix, color, marker, linestyle, linewidth, alpha in (
        ("bottom solver _Ftot", "bottom", "real_ftot", "#1f77b4", "o", "-", 1.45, 0.86),
        ("bottom _Ftot no damping", "bottom", "real_ftot_no_damping", "#1f77b4", "^", "--", 1.15, 0.76),
        ("bottom FHeron_i", "bottom", "fheron", "#2ca02c", "x", "-.", 1.35, 0.92),
        ("top solver _Ftot", "top", "real_ftot", "#ff7f0e", "s", "-", 1.45, 0.86),
        ("top _Ftot no damping", "top", "real_ftot_no_damping", "#ff7f0e", "D", "--", 1.15, 0.76),
        ("top FHeron_i", "top", "fheron", "#2ca02c", "+", ":", 1.35, 0.92),
    ):
        values_mn = 1.0e3 * _history_float_series(
            separation_history,
            f"{which}_cl_{suffix}_rcl_signed_mean_abs_N",
        )
        finite = np.isfinite(times) & np.isfinite(values_mn)
        if not np.any(finite):
            continue
        plotted = True
        ax.plot(
            times[finite],
            values_mn[finite],
            color=color,
            lw=linewidth,
            ls=linestyle,
            alpha=alpha,
        )
        finite_indices = np.flatnonzero(finite)
        marker_stride = max(1, int(np.ceil(finite_indices.size / 4.0)))
        marker_indices = finite_indices[::marker_stride]
        if marker_indices.size and marker_indices[-1] != finite_indices[-1]:
            marker_indices = np.r_[marker_indices, finite_indices[-1]]
        ax.scatter(
            times[marker_indices],
            values_mn[marker_indices],
            marker=marker,
            s=42.0,
            facecolors="none",
            edgecolors=color,
            linewidths=1.25,
            label=label,
        )

    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.0, color="0.45", lw=0.9, ls="--", alpha=0.8)
    ax.set_title("Solver-used contact-line total force vs time (+ outward r_cl)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\mathrm{sign}(\langle q_i\rangle)\langle |q_i|\rangle,\ q_i=F_i\cdot e_{r_{cl},i}$ [mN]")
    ax.grid(True, alpha=0.32)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / CL_FTOT_RADIAL_REALUSED_PNG_NAME, bbox_inches="tight")
    plt.close(fig)


def _surface_side_triangles(state: VolumetricPitoisState) -> np.ndarray:
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        return _actual_compute_side_surface_triangles(state)
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 3, 3), dtype=float)
    _surface, _surface_bV, flat_vertices, triangles = _build_structured_side_surface_complex(ordered_rings)
    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=float)
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in flat_vertices], dtype=float)
    coords = _project_outside_spheres(state, coords)
    return np.asarray([coords[np.asarray(tri, dtype=int)] for tri in triangles], dtype=float)


def _project_outside_spheres(state: VolumetricPitoisState, coords: np.ndarray) -> np.ndarray:
    bottom_center, top_center, _axis = _particle_centers_physical(state)
    radius = float(state.config.particle_radius)
    clearance = max(1.0e-9 * radius, float(_length_to_compute(state.config, 1.0e-8)))
    out = np.array(coords, dtype=float, copy=True)
    for sphere_center in (bottom_center, top_center):
        rel = out - sphere_center[None, :]
        dist = np.linalg.norm(rel, axis=1)
        inside = dist < (radius + clearance)
        if not np.any(inside):
            continue
        safe = np.maximum(dist[inside], 1.0e-30)
        dirs = rel[inside] / safe[:, None]
        out[inside] = sphere_center[None, :] + (radius + clearance) * dirs
    return out


def _surface_side_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    # For normal display, keep the inherited ring/meridian graph instead of
    # triangle-diagonal edges. This avoids spurious long diagonals and dangling
    # one-edge points caused by hidden-edge culling on the raw tetra boundary.
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)
    segments, coords = _contact_refined_surface_display_wireframe(state, ordered_rings)
    coords = _project_outside_spheres(state, coords)
    if segments.size:
        segments = _project_outside_spheres(state, segments.reshape((-1, 3))).reshape((-1, 2, 3))
    return segments, coords


def _cap_wireframe_vertex_pairs(state: VolumetricPitoisState) -> list[tuple[object, object]]:
    edge_pairs: list[tuple[object, object]] = []
    cap_specs = (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    )
    for rings, center_vertex in cap_specs:
        if not rings:
            continue
        n_ring = len(rings[0])
        for ring in rings:
            for i in range(n_ring):
                edge_pairs.append((ring[i], ring[(i + 1) % n_ring]))
        for inner, outer in zip(rings[:-1], rings[1:]):
            for i in range(n_ring):
                edge_pairs.append((inner[i], outer[i]))
        for i in range(n_ring):
            edge_pairs.append((center_vertex, rings[0][i]))
    return edge_pairs


def _wireframe_from_vertex_pairs(
    state: VolumetricPitoisState,
    edge_pairs: list[tuple[object, object]],
) -> tuple[np.ndarray, np.ndarray]:
    if not edge_pairs:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    unique_vertices: dict[int, object] = {}
    unique_edges: set[tuple[int, int]] = set()
    for va, vb in edge_pairs:
        ida = id(va)
        idb = id(vb)
        if ida == idb:
            continue
        unique_vertices.setdefault(ida, va)
        unique_vertices.setdefault(idb, vb)
        edge = (ida, idb) if ida < idb else (idb, ida)
        unique_edges.add(edge)

    ordered_ids = list(unique_vertices.keys())
    id_to_idx = {vid: idx for idx, vid in enumerate(ordered_ids)}
    coords = np.array(
        [np.asarray(unique_vertices[vid].x_a[:3], dtype=float) for vid in ordered_ids],
        dtype=float,
    )
    coords = _project_outside_spheres(state, coords)
    segments = [
        np.array([coords[id_to_idx[ida]], coords[id_to_idx[idb]]], dtype=float)
        for ida, idb in sorted(unique_edges)
    ]
    segs = np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 3), dtype=float)
    return segs, coords


def _wireframe_scatter_vertices(segments: np.ndarray, *, min_degree: int = 3) -> np.ndarray:
    segs = np.asarray(segments, dtype=float)
    if segs.size == 0:
        return np.empty((0, 3), dtype=float)

    adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = defaultdict(set)
    point_lookup: dict[tuple[float, float, float], np.ndarray] = {}
    for seg in segs:
        a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
        b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
        point_lookup[a] = np.asarray(seg[0], dtype=float)
        point_lookup[b] = np.asarray(seg[1], dtype=float)
        adjacency[a].add(b)
        adjacency[b].add(a)

    kept = [point_lookup[key] for key, nbrs in adjacency.items() if len(nbrs) >= int(min_degree)]
    if not kept:
        return np.empty((0, 3), dtype=float)
    return np.asarray(kept, dtype=float)


def _display_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    flat_vertices = [v for ring in ordered_rings for v in ring]
    edge_pairs = [(flat_vertices[i], flat_vertices[j]) for i, j in sorted(_structured_surface_display_edges(ordered_rings))]
    edge_pairs.extend(_cap_wireframe_vertex_pairs(state))
    return _wireframe_from_vertex_pairs(state, edge_pairs)


def _view_direction(elev: float, azim: float) -> np.ndarray:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    view = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(view))
    if norm <= 1.0e-30:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return view / norm


def _visible_surface_wireframe(
    state: VolumetricPitoisState,
    *,
    elev: float,
    azim: float,
) -> tuple[np.ndarray, np.ndarray]:
    segments, coords = _display_wireframe(state)
    if segments.size == 0 and coords.size == 0:
        return segments, coords
    visibility_margin = 5.0e-5

    bottom_center, top_center, axis = _particle_centers_physical(state)
    axis_mid = 0.5 * (bottom_center + top_center)
    view = _view_direction(elev, azim)
    view_tan = view - axis * float(np.dot(view, axis))
    view_tan_norm = float(np.linalg.norm(view_tan))
    if view_tan_norm <= 1.0e-30:
        return segments, coords
    view_tan /= view_tan_norm

    def front_metric(points: np.ndarray) -> np.ndarray:
        rel = np.asarray(points, dtype=float) - axis_mid[None, :]
        rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
        axial = np.outer(np.dot(rel, axis), axis)
        radial = rel - axial
        radial = np.nan_to_num(radial, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            metric = radial @ view_tan
        return np.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

    if segments.size:
        segments = np.asarray(segments, dtype=float)
        keep = np.all(np.isfinite(segments), axis=(1, 2))
        segments = segments[keep]
        mids = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
        seg_metric = front_metric(mids)
        segments = segments[seg_metric >= visibility_margin]
        if segments.size:
            changed = True
            while changed and segments.size:
                changed = False
                adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = defaultdict(set)
                point_lookup: dict[tuple[float, float, float], np.ndarray] = {}
                for seg in np.asarray(segments, dtype=float):
                    a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
                    b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
                    point_lookup[a] = np.asarray(seg[0], dtype=float)
                    point_lookup[b] = np.asarray(seg[1], dtype=float)
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                prune_nodes = {node for node, nbrs in adjacency.items() if len(nbrs) <= 1}
                if prune_nodes:
                    keep_mask = []
                    for seg in np.asarray(segments, dtype=float):
                        a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
                        b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
                        keep_mask.append((a not in prune_nodes) and (b not in prune_nodes))
                    keep_mask = np.asarray(keep_mask, dtype=bool)
                    if not np.all(keep_mask):
                        segments = np.asarray(segments, dtype=float)[keep_mask]
                        changed = True

    if segments.size:
        coords = segments.reshape(-1, 3)
        rounded = np.round(coords, decimals=12)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        coords = coords[np.sort(unique_idx)]
    elif coords.size:
        coords = np.asarray(coords, dtype=float)
        coords = coords[np.all(np.isfinite(coords), axis=1)]
        coord_metric = front_metric(coords)
        coords = coords[coord_metric >= visibility_margin]

    return segments, coords


def _plot_coords_mm(points: np.ndarray, config: VolumetricPitoisConfig | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if config is not None:
        points = np.asarray(_length_from_compute(config, points), dtype=float)
    return 1.0e3 * points


def _particle_centers_physical(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return bottom_center, top_center, axis


def _particle_center_positions(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, _axis = _particle_centers_physical(state)
    return (
        _plot_coords_mm(bottom_center[None, :], state.config)[0],
        _plot_coords_mm(top_center[None, :], state.config)[0],
    )


def _cap_plane_positions(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, float]:
    bottom_cap_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.outer_rings[0]], axis=0)
    top_cap_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.outer_rings[-1]], axis=0)
    contact_radius = min(
        max(_cap_radius(state.outer_rings[0]), _cap_radius(state.outer_rings[-1])),
        state.config.particle_radius,
    )
    return (
        _plot_coords_mm(bottom_cap_center[None, :], state.config)[0],
        _plot_coords_mm(top_cap_center[None, :], state.config)[0],
        1.0e3 * float(_length_from_compute(state.config, contact_radius)),
    )


def _sphere_cap_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    radius = float(state.config.particle_radius)
    contact_radius = min(
        max(_cap_radius(state.cap_bottom), _cap_radius(state.cap_top)),
        radius,
    )

    def project_to_sphere(point: np.ndarray, plane_center: np.ndarray, sphere_center: np.ndarray, sign: float) -> np.ndarray:
        tangential = point - plane_center
        tangential -= axis * float(np.dot(tangential, axis))
        r_t = float(np.linalg.norm(tangential))
        axial = np.sqrt(max(radius**2 - r_t**2, 0.0))
        return sphere_center + tangential + sign * axial * axis

    cap_triangle_sets: list[list[np.ndarray]] = []
    cap_specs = (
        (state.layer_rings[0], np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.cap_bottom], axis=0), bottom_center, +1.0),
        (state.layer_rings[-1], np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.cap_top], axis=0), top_center, -1.0),
    )

    for rings, plane_center, sphere_center, sign in cap_specs:
        triangles: list[np.ndarray] = []
        ring_xyz = []
        for ring in rings:
            ring_xyz.append(
                [
                    project_to_sphere(np.asarray(v.x_a[:3], dtype=float), plane_center, sphere_center, sign)
                    for v in ring
                ]
            )

        if not ring_xyz:
            continue

        center = sphere_center + sign * radius * axis
        n_ring = len(ring_xyz[0])
        for i in range(n_ring):
            triangles.append(
                np.array(
                    [
                        center,
                        ring_xyz[0][i],
                        ring_xyz[0][(i + 1) % n_ring],
                    ],
                    dtype=float,
                )
            )
        for band in range(len(ring_xyz) - 1):
            inner = ring_xyz[band]
            outer = ring_xyz[band + 1]
            for i in range(n_ring):
                triangles.append(np.array([inner[i], outer[i], outer[(i + 1) % n_ring]], dtype=float))
                triangles.append(np.array([inner[i], outer[(i + 1) % n_ring], inner[(i + 1) % n_ring]], dtype=float))
        cap_triangle_sets.append(triangles)

    while len(cap_triangle_sets) < 2:
        cap_triangle_sets.append([])

    bottom_tris = np.asarray(cap_triangle_sets[0], dtype=float) if cap_triangle_sets[0] else np.empty((0, 3, 3), dtype=float)
    top_tris = np.asarray(cap_triangle_sets[1], dtype=float) if cap_triangle_sets[1] else np.empty((0, 3, 3), dtype=float)
    return bottom_tris, top_tris


def _snapshot_title(state: VolumetricPitoisState, *, label: str) -> str:
    n_vertices = sum(1 for _ in state.HC.V)
    n_free = sum(1 for v in state.HC.V if v not in state.bV_caps)
    return f"{state.config.title}: {label} mesh\nvertices={n_vertices}, free={n_free}"


def _render_particle_wireframe(ax, center: np.ndarray, radius: float, color: str) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0.5,
        alpha=0.16,
        axlim_clip=True,
    )
    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        color=color,
        s=36,
        edgecolors="#111111",
        linewidths=0.6,
        axlim_clip=True,
    )


def _render_contact_circle(ax, cap_center: np.ndarray, radius: float, axis: np.ndarray, color: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    e1, e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))
    ring = cap_center[None, :] + radius * (
        np.cos(theta)[:, None] * e1[None, :] + np.sin(theta)[:, None] * e2[None, :]
    )
    ax.plot(
        ring[:, 0],
        ring[:, 1],
        ring[:, 2],
        color=color,
        linewidth=1.6,
        alpha=0.98,
        axlim_clip=True,
    )


def _scatter_mesh_vertices(ax, triangle_sets: list[np.ndarray], *, color: str = "#111111") -> None:
    if not USER_SHOW_MESH_VERTICES:
        return
    pts = [tris.reshape(-1, 3) for tris in triangle_sets if tris.size]
    if not pts:
        return
    flat = np.concatenate(pts, axis=0)
    rounded = np.round(flat, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    verts = flat[np.sort(unique_idx)]
    marker_size = max(2.0, float(USER_MESH_VERTEX_SIZE))
    ax.scatter(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        s=marker_size,
        facecolors=color,
        edgecolors=color,
        linewidths=0.35,
        alpha=0.98,
        depthshade=False,
        axlim_clip=True,
    )


def _render_triangle_edges(
    ax,
    triangles: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    if triangles.size == 0:
        return
    segments = []
    seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    for tri in np.asarray(triangles, dtype=float):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p = tuple(np.round(tri[i], 12))
            q = tuple(np.round(tri[j], 12))
            key = (p, q) if p <= q else (q, p)
            if key in seen:
                continue
            seen.add(key)
            segments.append(np.array([tri[i], tri[j]], dtype=float))
    if not segments:
        return
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            axlim_clip=True,
        )
    )


def _write_gmsh2(
    points: np.ndarray,
    elements: np.ndarray,
    out_path: Path,
    *,
    element_type: int,
    node_ids: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=float)
    elements = np.asarray(elements, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    expected_cols = 3 if int(element_type) == 2 else 4 if int(element_type) == 4 else None
    if expected_cols is None:
        raise ValueError("element_type must be 2 (triangles) or 4 (tetrahedra)")
    if elements.ndim != 2 or elements.shape[1] != expected_cols:
        raise ValueError(f"elements must be (M,{expected_cols})")
    if node_ids is None:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.shape[0] != points.shape[0]:
            raise ValueError("node_ids must be a length-N array")

    with open(out_path, "w", newline="") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{points.shape[0]}\n")
        for node_id, (x, y, z) in zip(node_ids, points):
            f.write(f"{int(node_id)} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{elements.shape[0]}\n")
        for i, elem in enumerate(elements, start=1):
            node_tokens = " ".join(str(int(node_ids[int(local_idx)])) for local_idx in elem)
            f.write(f"{i} {int(element_type)} 0 {node_tokens}\n")
        f.write("$EndElements\n")


def _write_gmsh2_mixed(
    points: np.ndarray,
    out_path: Path,
    *,
    triangles: np.ndarray | None = None,
    tetrahedra: np.ndarray | None = None,
    node_ids: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=float)
    triangles = np.empty((0, 3), dtype=int) if triangles is None else np.asarray(triangles, dtype=int)
    tetrahedra = np.empty((0, 4), dtype=int) if tetrahedra is None else np.asarray(tetrahedra, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be (M,3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must be (K,4)")
    if node_ids is None:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.shape[0] != points.shape[0]:
            raise ValueError("node_ids must be a length-N array")

    element_blocks = ((2, triangles), (4, tetrahedra))
    n_elements = int(triangles.shape[0] + tetrahedra.shape[0])
    with open(out_path, "w", newline="") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{points.shape[0]}\n")
        for node_id, (x, y, z) in zip(node_ids, points):
            f.write(f"{int(node_id)} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{n_elements}\n")
        elem_id = 1
        for element_type, elements in element_blocks:
            for elem in elements:
                node_tokens = " ".join(str(int(node_ids[int(local_idx)])) for local_idx in elem)
                f.write(f"{elem_id} {int(element_type)} 0 {node_tokens}\n")
                elem_id += 1
        f.write("$EndElements\n")


def _surface_triangle_indices_from_snapshot_points(
    points: np.ndarray,
    *triangle_sets: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=int)
    point_index: dict[tuple[float, float, float], int] = {
        tuple(round(float(c), 12) for c in point): int(i)
        for i, point in enumerate(points)
    }
    scale = max(float(np.max(np.linalg.norm(points, axis=1))) if points.size else 0.0, 1.0e-12)
    nearest_tol = max(1.0e-12, 1.0e-7 * scale)
    triangles: list[tuple[int, int, int]] = []

    def resolve(point: np.ndarray) -> int | None:
        key = tuple(round(float(c), 12) for c in point)
        idx = point_index.get(key)
        if idx is not None:
            return int(idx)
        distances = np.linalg.norm(points - np.asarray(point, dtype=float)[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= nearest_tol:
            return nearest
        return None

    for triangle_set in triangle_sets:
        arr = np.asarray(triangle_set, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape((-1, 3, 3))
        for tri in arr:
            idxs = [resolve(point) for point in tri]
            if any(idx is None for idx in idxs):
                continue
            a, b, c = (int(idxs[0]), int(idxs[1]), int(idxs[2]))
            if len({a, b, c}) == 3:
                triangles.append((a, b, c))
    if not triangles:
        return np.empty((0, 3), dtype=int)
    return np.asarray(triangles, dtype=int)


def _triangle_soup_to_indexed_mesh(*triangle_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    coord_index: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    def ensure_vertex(point: np.ndarray) -> int:
        key = tuple(round(float(c), 12) for c in point)
        idx = coord_index.get(key)
        if idx is None:
            idx = len(coords)
            coords.append(tuple(float(c) for c in point))
            coord_index[key] = idx
        return idx

    for tri_set in triangle_sets:
        arr = np.asarray(tri_set, dtype=float)
        if arr.size == 0:
            continue
        for tri in arr:
            a = ensure_vertex(tri[0])
            b = ensure_vertex(tri[1])
            c = ensure_vertex(tri[2])
            if len({a, b, c}) == 3:
                triangles.append((a, b, c))

    if not coords or not triangles:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    return np.asarray(coords, dtype=float), np.asarray(triangles, dtype=int)


def _snapshot_surface_indexed_mesh(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _freeze_surface_topology(state)
    points = _surface_points_array(state)
    tri_array = np.vstack(
        [
            arr
            for arr in (
                state.surface_export_side_tris,
                state.surface_export_bottom_cap_tris,
                state.surface_export_top_cap_tris,
            )
            if np.asarray(arr, dtype=int).size
        ]
    ) if (
        state.surface_export_side_tris.size
        or state.surface_export_bottom_cap_tris.size
        or state.surface_export_top_cap_tris.size
    ) else np.empty((0, 3), dtype=int)
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return points, tri_array, node_ids


def _volume_points_array(state: VolumetricPitoisState) -> np.ndarray:
    _ensure_volume_export_node_ids(state)
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.volume_export_vertices],
        dtype=float,
    )


def _snapshot_msh_indexed_mesh(
    state: VolumetricPitoisState,
    *,
    freeze_topology: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if freeze_topology:
        _freeze_surface_topology(state)
    points = _volume_points_array(state)
    tets = np.asarray(state.volume_export_tets, dtype=int)
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return points, tets, node_ids


def _indexed_tet_mesh_volume_m3(points: np.ndarray, tets: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0 or tet_idx.size == 0:
        return 0.0
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    # Volumetric .msh volume is the sum of positive tetra volumes in compute units.
    # Never rely on cancellation from mixed element orientation.
    triple = np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))
    return float(np.sum(np.abs(triple)) / 6.0)


def _indexed_tet_volume_gradient(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    gradient = np.zeros_like(pts, dtype=float)
    if pts.size == 0 or tet_idx.size == 0:
        return gradient
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    triple = np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))
    signed_scale = (np.sign(triple) / 6.0)[:, None]
    grad_a = signed_scale * np.cross(pb - pd, pc - pd)
    grad_b = signed_scale * np.cross(pc - pd, pa - pd)
    grad_c = signed_scale * np.cross(pa - pd, pb - pd)
    grad_d = -(grad_a + grad_b + grad_c)
    np.add.at(gradient, tet_idx[:, 0], grad_a)
    np.add.at(gradient, tet_idx[:, 1], grad_b)
    np.add.at(gradient, tet_idx[:, 2], grad_c)
    np.add.at(gradient, tet_idx[:, 3], grad_d)
    return gradient


def _snapshot_msh_volume_m3(
    state: VolumetricPitoisState,
    *,
    freeze_topology: bool = True,
    side_triangles: np.ndarray | None = None,
    bottom_cap_triangles: np.ndarray | None = None,
    top_cap_triangles: np.ndarray | None = None,
) -> float:
    del side_triangles, bottom_cap_triangles, top_cap_triangles
    points, tets, _node_ids = _snapshot_msh_indexed_mesh(state, freeze_topology=freeze_topology)
    return _indexed_tet_mesh_volume_m3(points, tets)


def _wireframe_from_triangle_soup(*triangle_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points, tris = _triangle_soup_to_indexed_mesh(*triangle_sets)
    if points.size == 0 or tris.size == 0:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(tris, dtype=int):
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(c), int(a)))))

    segments = np.asarray(
        [np.array([points[i], points[j]], dtype=float) for i, j in sorted(edges)],
        dtype=float,
    )
    return segments, np.asarray(points, dtype=float)


def _save_mesh_snapshot_msh(
    state: VolumetricPitoisState,
    out_path: Path,
    *,
    side_triangles: np.ndarray,
    bottom_cap_triangles: np.ndarray,
    top_cap_triangles: np.ndarray,
) -> Path:
    points, tets, node_ids = _snapshot_msh_indexed_mesh(state)
    triangles = _surface_triangle_indices_from_snapshot_points(
        points,
        side_triangles,
        bottom_cap_triangles,
        top_cap_triangles,
    )
    points_out = np.asarray(_length_from_compute(state.config, points), dtype=float)
    msh_path = out_path.with_suffix(".msh")
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gmsh2_mixed(points_out, msh_path, triangles=triangles, tetrahedra=tets, node_ids=node_ids)
    return msh_path


def _raw_volume_surface_triangles_from_tets(tets: np.ndarray) -> np.ndarray:
    tet_idx = np.asarray(tets, dtype=int)
    if tet_idx.size == 0:
        return np.empty((0, 3), dtype=int)
    face_counts: dict[tuple[int, int, int], int] = {}
    face_oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for a, b, c, d in tet_idx:
        for face in ((a, b, c), (a, d, b), (a, c, d), (b, d, c)):
            key = tuple(sorted(int(i) for i in face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_oriented.setdefault(key, tuple(int(i) for i in face))
    boundary = [face_oriented[key] for key, count in face_counts.items() if count == 1]
    if not boundary:
        return np.empty((0, 3), dtype=int)
    return np.asarray(boundary, dtype=int)


def _raw_volume_mesh_arrays(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.volume_export_vertices],
        dtype=float,
    )
    tets = np.asarray(state.volume_export_tets, dtype=int)
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    triangles = _raw_volume_surface_triangles_from_tets(tets)
    return points, tets, triangles, node_ids


def _render_raw_volume_mesh_snapshot(
    state: VolumetricPitoisState,
    title: str,
    out_path: Path,
    *,
    elev: float = USER_INTERACTIVE_ELEV_DEG,
    azim: float = USER_INTERACTIVE_AZIM_DEG,
) -> Path:
    points, tets, triangles, node_ids = _raw_volume_mesh_arrays(state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points_out = np.asarray(_length_from_compute(state.config, points), dtype=float)
    _write_gmsh2_mixed(
        points_out,
        out_path.with_suffix(".msh"),
        triangles=triangles,
        tetrahedra=tets,
        node_ids=node_ids,
    )

    plot_points = _plot_coords_mm(points, state.config)
    plot_tris = plot_points[triangles] if triangles.size else np.empty((0, 3, 3), dtype=float)
    bottom_center, top_center = _particle_center_positions(state)
    _bottom_center_phys, _top_center_phys, plot_axis = _particle_centers_physical(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * float(_length_from_compute(state.config, state.config.particle_radius))

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    if plot_tris.size:
        poly_liquid = Poly3DCollection(
            plot_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
        _render_triangle_edges(
            ax,
            plot_tris,
            color="#111111",
            linewidth=0.36,
            alpha=0.42,
        )
        _scatter_mesh_vertices(ax, [plot_tris])
    if USER_FILL_PARTICLE_CAP_SURFACES:
        _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
        _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, contact_radius_mm, plot_axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, contact_radius_mm, plot_axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal(ax, plot_tris, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _capture_raw_volume_mesh_snapshot(
    state: VolumetricPitoisState,
    title: str,
    filename: str,
    *,
    is_initial: bool = False,
) -> SimpleNamespace:
    points, tets, triangles, node_ids = _raw_volume_mesh_arrays(state)
    points = np.asarray(points, dtype=float).copy()
    tets = np.asarray(tets, dtype=int).copy()
    triangles = np.asarray(triangles, dtype=int).copy()
    node_ids = np.asarray(node_ids, dtype=int).copy()
    plot_points = _plot_coords_mm(points, state.config)
    bottom_center, top_center = _particle_center_positions(state)
    _bottom_center_phys, _top_center_phys, plot_axis = _particle_centers_physical(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * float(_length_from_compute(state.config, state.config.particle_radius))
    return SimpleNamespace(
        filename=str(filename),
        title=str(title),
        is_initial=bool(is_initial),
        points_out=np.asarray(_length_from_compute(state.config, points), dtype=float).copy(),
        tets=tets,
        triangles=triangles,
        node_ids=node_ids,
        plot_tris=(plot_points[triangles].copy() if triangles.size else np.empty((0, 3, 3), dtype=float)),
        bottom_center=np.asarray(bottom_center, dtype=float).copy(),
        top_center=np.asarray(top_center, dtype=float).copy(),
        plot_axis=np.asarray(plot_axis, dtype=float).copy(),
        bottom_cap_center=np.asarray(bottom_cap_center, dtype=float).copy(),
        top_cap_center=np.asarray(top_cap_center, dtype=float).copy(),
        contact_radius_mm=float(contact_radius_mm),
        radius_mm=float(radius_mm),
        render_png=bool(getattr(state.config, "render_raw_mesh_snapshot_png", True)),
    )


def _write_captured_raw_volume_mesh_snapshot(snapshot: SimpleNamespace, motion_dir: Path) -> Path:
    out_path = motion_dir / str(snapshot.filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gmsh2_mixed(
        np.asarray(snapshot.points_out, dtype=float),
        out_path.with_suffix(".msh"),
        triangles=np.asarray(snapshot.triangles, dtype=int),
        tetrahedra=np.asarray(snapshot.tets, dtype=int),
        node_ids=np.asarray(snapshot.node_ids, dtype=int),
    )
    if not bool(getattr(snapshot, "render_png", True)):
        return out_path.with_suffix(".msh")

    plot_tris = np.asarray(snapshot.plot_tris, dtype=float)
    bottom_center = np.asarray(snapshot.bottom_center, dtype=float)
    top_center = np.asarray(snapshot.top_center, dtype=float)
    plot_axis = np.asarray(snapshot.plot_axis, dtype=float)
    bottom_cap_center = np.asarray(snapshot.bottom_cap_center, dtype=float)
    top_cap_center = np.asarray(snapshot.top_cap_center, dtype=float)
    radius_mm = float(snapshot.radius_mm)

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    if plot_tris.size:
        poly_liquid = Poly3DCollection(
            plot_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
        _render_triangle_edges(ax, plot_tris, color="#111111", linewidth=0.36, alpha=0.42)
        _scatter_mesh_vertices(ax, [plot_tris])
    if USER_FILL_PARTICLE_CAP_SURFACES:
        _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
        _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, float(snapshot.contact_radius_mm), plot_axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, float(snapshot.contact_radius_mm), plot_axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )
    ax.set_title(str(snapshot.title), pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=USER_INTERACTIVE_ELEV_DEG, azim=USER_INTERACTIVE_AZIM_DEG)
    _set_axes_equal(ax, plot_tris, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if bool(getattr(snapshot, "is_initial", False)):
        iter0_png = motion_dir / "mesh_iter0000.png"
        shutil.copyfile(out_path, iter0_png)
        source = out_path.with_suffix(".msh")
        if source.exists():
            shutil.copyfile(source, iter0_png.with_suffix(".msh"))
    return out_path


def _capture_motion_mesh_snapshot(
    state: VolumetricPitoisState,
    *,
    label: str,
    step: int,
) -> SimpleNamespace:
    if label == "initial":
        filename = "mesh_initial.png"
    else:
        filename = f"mesh_iter{step:04d}.png"
    title = _snapshot_title(state, label=label if label == "initial" else f"iteration {step}")
    return _capture_raw_volume_mesh_snapshot(
        state,
        title,
        filename,
        is_initial=(label == "initial"),
    )


def _set_axes_equal(ax, xyz: np.ndarray, centers: list[np.ndarray], radius: float) -> None:
    pts = np.asarray(xyz.reshape(-1, 3), dtype=float) if xyz.size else np.empty((0, 3), dtype=float)
    if pts.size:
        pts = pts[np.all(np.isfinite(pts), axis=1)]
    if centers:
        centers_arr = np.asarray(centers, dtype=float)
        centers_arr = centers_arr[np.all(np.isfinite(centers_arr), axis=1)]
        sphere_bbox = []
        for center in centers_arr:
            sphere_bbox.extend(
                [
                    center + np.array([+radius, 0.0, 0.0], dtype=float),
                    center + np.array([-radius, 0.0, 0.0], dtype=float),
                    center + np.array([0.0, +radius, 0.0], dtype=float),
                    center + np.array([0.0, -radius, 0.0], dtype=float),
                    center + np.array([0.0, 0.0, +radius], dtype=float),
                    center + np.array([0.0, 0.0, -radius], dtype=float),
                ]
            )
        if sphere_bbox:
            pts = np.vstack([pts, centers_arr, np.asarray(sphere_bbox, dtype=float)])
    if pts.size == 0:
        pts = np.zeros((1, 3), dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    x_min = float(USER_X_AXIS_MIN_MM)
    x_max = float(USER_X_AXIS_MAX_MM)
    z_min = float(USER_Z_AXIS_MIN_MM)
    z_max = float(USER_Z_AXIS_MAX_MM)

    x_half = 0.5 * max(x_max - x_min, 0.0)
    z_half = 0.5 * max(z_max - z_min, 0.0)
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)

    if x_half <= 0.0:
        x_half = max(0.5 * float(maxs[0] - mins[0]), float(radius), 1.0)
        x_center = float(center[0])
    if z_half <= 0.0:
        z_half = max(0.5 * float(maxs[2] - mins[2]), float(radius), 1.0)
        z_center = float(center[2])

    half_span = max(x_half, z_half)
    if not np.isfinite(half_span) or half_span <= 0.0:
        half_span = 1.0
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(z_center - half_span, z_center + half_span)
    ax.set_box_aspect((1, 1, 1))


def _render_mesh_snapshot(
    state: VolumetricPitoisState,
    title: str,
    out_path: Path,
    *,
    elev: float = USER_INTERACTIVE_ELEV_DEG,
    azim: float = USER_INTERACTIVE_AZIM_DEG,
) -> Path:
    side_triangles_full = _surface_side_triangles(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "meridian_strip":
        side_triangles, bottom_cap_triangles, top_cap_triangles = _actual_compute_meridian_strip_triangles(state)
    elif USER_SHOW_REAL_COMPUTE_TRIANGLES:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    else:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _sphere_cap_triangles(state)
        side_segments, side_vertices = _front_band_wireframe(state, elev=elev, azim=azim)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        if USER_REAL_TRIANGLE_VIEW == "meridian_strip" and not USER_SHOW_MESH_FACES:
            side_segments, side_vertices = _meridian_section_wireframe(state)
        else:
            side_segments, side_vertices = _surface_side_wireframe(state)
    side_vertices = _wireframe_scatter_vertices(side_segments, min_degree=4)
    plot_side_segments = _plot_coords_mm(side_segments, state.config)
    plot_side_vertices = _plot_coords_mm(side_vertices, state.config)
    plot_side_tris = _plot_coords_mm(side_triangles, state.config)
    plot_bottom_cap_tris = _plot_coords_mm(bottom_cap_triangles, state.config)
    plot_top_cap_tris = _plot_coords_mm(top_cap_triangles, state.config)
    bottom_center, top_center = _particle_center_positions(state)
    _bottom_center_phys, _top_center_phys, plot_axis = _particle_centers_physical(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * float(_length_from_compute(state.config, state.config.particle_radius))

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    liquid_triangle_sets = [plot_side_tris] if plot_side_tris.size else []
    if USER_FILL_PARTICLE_CAP_SURFACES:
        if plot_bottom_cap_tris.size:
            liquid_triangle_sets.append(plot_bottom_cap_tris)
        if plot_top_cap_tris.size:
            liquid_triangle_sets.append(plot_top_cap_tris)
    if plot_side_tris.size and USER_SHOW_SURFACE_OVERLAY:
        overlay_poly = Poly3DCollection(
            plot_side_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_SURFACE_OVERLAY_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(overlay_poly)
    if liquid_triangle_sets and USER_SHOW_MESH_FACES:
        poly_liquid = Poly3DCollection(
            np.concatenate(liquid_triangle_sets, axis=0),
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size:
        _render_triangle_edges(
            ax,
            plot_side_tris,
            color="#0f172a",
            linewidth=0.42,
            alpha=0.55,
        )
    scatter_vertices_from = plot_side_tris if (
        USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size
    ) else plot_side_vertices
    _scatter_mesh_vertices(ax, [scatter_vertices_from])
    ax.add_collection3d(
        Line3DCollection(
            plot_side_segments,
            colors="#111111",
            linewidths=0.72,
            alpha=0.995,
            axlim_clip=True,
        )
    )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES:
        _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
        _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    if USER_SHOW_MESH_FACES and plot_bottom_cap_tris.size:
        poly_bottom_caps = Poly3DCollection(
            plot_bottom_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#2b6cb0",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_bottom_caps)
    if USER_SHOW_MESH_FACES and plot_top_cap_tris.size:
        poly_top_caps = Poly3DCollection(
            plot_top_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#dd8a1c",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_top_caps)

    if USER_FILL_PARTICLE_CAP_SURFACES:
        _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
        _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, contact_radius_mm, plot_axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, contact_radius_mm, plot_axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    plot_all = []
    if plot_side_tris.size:
        plot_all.append(plot_side_tris)
    if plot_bottom_cap_tris.size:
        plot_all.append(plot_bottom_cap_tris)
    if plot_top_cap_tris.size:
        plot_all.append(plot_top_cap_tris)
    plot_xyz = np.concatenate(plot_all, axis=0) if plot_all else np.empty((0, 3, 3))
    _set_axes_equal(ax, plot_xyz, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _save_mesh_snapshot_msh(
        state,
        out_path,
        side_triangles=side_triangles_full,
        bottom_cap_triangles=_actual_compute_cap_triangles(state)[0] if USER_SHOW_REAL_COMPUTE_TRIANGLES else bottom_cap_triangles,
        top_cap_triangles=_actual_compute_cap_triangles(state)[1] if USER_SHOW_REAL_COMPUTE_TRIANGLES else top_cap_triangles,
    )
    return out_path


def _render_history_figures(
    history: list[dict],
    config: VolumetricPitoisConfig,
    out_dir: Path,
    *,
    initial_snapshot_volume_m3: float,
) -> list[Path]:
    if not history:
        return []

    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_ms = np.array([float(row["t"]) * 1.0e3 for row in history], dtype=float)
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    gap_um = np.array([float(row["gap"]) * 1.0e6 for row in history], dtype=float)
    fixed_force_mn = np.array([float(row["fixed_force_axial"]) * 1.0e3 for row in history], dtype=float)
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)

    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(6.4, 4.4))
    ax1.plot(d_over_r, fixed_force_mn, color="#bc4749", linewidth=1.8)
    ax1.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
    ax1.set_xlabel(r"$D/R$")
    ax1.set_ylabel("Fixed-sphere axial force [mN]")
    ax1.set_title("Separation force vs gap")
    ax1.grid(alpha=0.28)
    path1 = fig_dir / "separation_force_vs_gap.png"
    fig1.tight_layout()
    fig1.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    paths.append(path1)

    fig2, axes = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True)
    axes[0].plot(t_ms, gap_um, color="#2a6f97", linewidth=1.8)
    axes[0].set_ylabel("Gap [um]")
    axes[0].grid(alpha=0.28)
    axes[0].set_title("Separation gap and free-speed history")
    axes[1].plot(t_ms, max_u, color="#6a4c93", linewidth=1.8)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Max free speed [m/s]")
    axes[1].grid(alpha=0.28)
    path2 = fig_dir / "separation_gap_speed_vs_time.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    paths.append(path2)

    snapshot_volume_m3 = np.array(
        [float(initial_snapshot_volume_m3)]
        + [float(row.get("snapshot_msh_volume_m3", row.get("snapshot_surface_volume_m3", 0.0))) for row in history],
        dtype=float,
    )
    snapshot_volume_ul = 1.0e9 * snapshot_volume_m3
    t_ms_volume = np.array([0.0] + [float(row["t"]) * 1.0e3 for row in history], dtype=float)
    snapshot_rel_error_pct = 100.0 * (snapshot_volume_m3 - float(initial_snapshot_volume_m3)) / max(
        float(initial_snapshot_volume_m3), 1.0e-30
    )

    fig3, axes3 = plt.subplots(2, 1, figsize=(6.6, 6.2), sharex=True)
    axes3[0].plot(t_ms_volume, snapshot_volume_ul, color="#0f766e", linewidth=1.8)
    axes3[0].set_ylabel("Volume [uL]")
    axes3[0].set_title("Volumetric .msh liquid-bridge volume history")
    axes3[0].grid(alpha=0.28)
    axes3[1].plot(t_ms_volume, snapshot_rel_error_pct, color="#bc4749", linewidth=1.8)
    axes3[1].axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
    axes3[1].set_xlabel("Time [ms]")
    axes3[1].set_ylabel("Rel. error [%]")
    axes3[1].grid(alpha=0.28)
    path3 = fig_dir / "separation_snapshot_volume_rel_error_vs_time.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    paths.append(path3)

    paths.append(_refresh_contact_radius_history_png(history, fig_dir))

    return paths


def _refresh_contact_radius_history_png(history: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = np.array([int(row["step"]) for row in history], dtype=int)
    t_ms = np.array([float(row["t"]) * 1.0e3 for row in history], dtype=float)
    bottom_r_cl_mm = np.array([float(row["bottom_contact_radius"]) * 1.0e3 for row in history], dtype=float)
    top_r_cl_mm = np.array([float(row["top_contact_radius"]) * 1.0e3 for row in history], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.4))
    axes[0].plot(t_ms, bottom_r_cl_mm, color="#2b6cb0", linewidth=1.8, label=r"bottom $r_{CL}$")
    axes[0].plot(t_ms, top_r_cl_mm, color="#dd8a1c", linewidth=1.8, label=r"top $r_{CL}$")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel(r"$r_{CL}$ [mm]")
    axes[0].set_title("Contact-line radius vs time")
    axes[0].grid(alpha=0.28)
    axes[0].legend(loc="best")
    axes[1].plot(steps, bottom_r_cl_mm, color="#2b6cb0", linewidth=1.8, label=r"bottom $r_{CL}$")
    axes[1].plot(steps, top_r_cl_mm, color="#dd8a1c", linewidth=1.8, label=r"top $r_{CL}$")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$r_{CL}$ [mm]")
    axes[1].set_title("Contact-line radius vs step")
    axes[1].grid(alpha=0.28)
    axes[1].legend(loc="best")
    path = out_dir / "separation_contact_radius_vs_time_step.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_motion_mesh_pngs(
    state: VolumetricPitoisState,
    *,
    motion_dir: Path,
    label: str,
    step: int,
) -> Path:
    motion_dir.mkdir(parents=True, exist_ok=True)
    if label == "initial":
        filename = "mesh_initial.png"
    else:
        filename = f"mesh_iter{step:04d}.png"
    title = _snapshot_title(state, label=label if label == "initial" else f"iteration {step}")
    out_png = _write_captured_raw_volume_mesh_snapshot(
        _capture_raw_volume_mesh_snapshot(
            state,
            title,
            filename,
            is_initial=(label == "initial"),
        ),
        motion_dir,
    )
    if label == "initial":
        initial_source = Path(getattr(state, "loaded_initial_msh_path", _required_initial_msh_path(state.config)))
        expected_source = _required_initial_msh_path(state.config)
        if initial_source.resolve() != expected_source.resolve():
            raise RuntimeError(f"Initial output must come from {expected_source}, got {initial_source}")
    return out_png


def _save_history(history: list[dict], config: VolumetricPitoisConfig, out_dir: Path) -> Path:
    results_dir = out_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "history": history,
    }
    path = results_dir / "separation_history.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def _history_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    force_mn = np.array([float(row["fixed_force_axial"]) for row in history], dtype=float) * 1.0e3
    force_abs_mn = np.abs(force_mn)
    if any(bool(row.get("reported_force_monotonic_envelope", False)) for row in history):
        # Histories assembled from restart/checkpoint records do not carry the
        # in-memory force envelope that a continuous run uses. Reapply the same
        # reported-force convention before drawing the Fig. 5 comparison.
        force_abs_mn = np.minimum.accumulate(force_abs_mn)
    t_ms = np.array([float(row["t"]) for row in history], dtype=float) * 1.0e3
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)
    return {
        "d_over_r": d_over_r,
        "force_mn": force_mn,
        "force_abs_mn": force_abs_mn,
        "t_ms": t_ms,
        "max_u": max_u,
    }


def _pitois_fig5_dynamic_digitized() -> tuple[np.ndarray, np.ndarray]:
    d_over_r = np.array(
        [
            0.014403481854782,
            0.019300000000000,
            0.023587192093071,
            0.028302414461984,
            0.033166898401356,
            0.037986935617921,
            0.042569921040376,
            0.047499765555437,
            0.051967121934836,
            0.056470352302334,
            0.061304411920954,
            0.066003127630689,
            0.070366380348669,
            0.075100720316807,
            0.079819428415061,
            0.085182169652359,
            0.092208347873809,
            0.097048578239299,
            0.101370508634981,
            0.108834059521728,
            0.118227619608521,
            0.129680242729548,
            0.146372328112875,
            0.160984243023503,
            0.184443915271577,
            0.186311945235594,
            0.202973726366585,
        ],
        dtype=float,
    )
    force_mn = np.array(
        [
            1.515311539463195,
            1.130000000000000,
            0.955216555657008,
            0.822460624401661,
            0.74381061419963,
            0.65567423269621,
            0.574562744753398,
            0.517342047749411,
            0.45826545147229,
            0.416512293755708,
            0.376315664380932,
            0.337105214458683,
            0.30674808760201,
            0.286922242059943,
            0.267636737727435,
            0.238270688094022,
            0.218305191654015,
            0.1986840608733,
            0.178384121904434,
            0.158325437116782,
            0.139272315154625,
            0.1194131217624,
            0.099728317139538,
            0.079873442448703,
            0.05974152111953,
            0.052749970637026,
            0.042572497546823,
        ],
        dtype=float,
    )
    return d_over_r, force_mn


def _save_pitois_fig5_compare(
    *,
    exp_x: np.ndarray,
    exp_y: np.ndarray,
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    out_path: Path,
    sim_label: str = "Case_2b_axisym raw simulation",
    xlim: tuple[float, float] = (0.01, 0.30),
    ylim: tuple[float, float] = (1.0e-7, 2.0),
    title: str = "Pitois 2000 Fig. 5: dynamic experiment vs Case_2b_axisym",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y_low, y_high = map(float, ylim)
    visible_y = []
    for x_arr, y_arr in ((exp_x, exp_y), (sim_x, sim_y)):
        x_vals = np.asarray(x_arr, dtype=float)
        y_vals = np.asarray(y_arr, dtype=float)
        mask = (
            np.isfinite(x_vals)
            & np.isfinite(y_vals)
            & (x_vals >= float(xlim[0]))
            & (x_vals <= float(xlim[1]))
            & (y_vals > 0.0)
        )
        if np.any(mask):
            visible_y.append(y_vals[mask])
    if visible_y:
        visible = np.concatenate(visible_y)
        y_low = min(y_low, 0.85 * float(np.min(visible)))
        y_high = max(y_high, 1.15 * float(np.max(visible)))

    ax.scatter(exp_x, exp_y, s=34, color="#111111", label="Pitois 2000 dynamic exp (black dots)")
    ax.plot(sim_x, sim_y, color="#d62828", linewidth=2.0, label=sim_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel(r"$D/R$")
    ax.set_ylabel(r"$|F|$ [mN]")
    ax.set_title(title)
    ax.grid(alpha=0.28, which="both")
    ax.legend()
    decimal_log_formatter = FuncFormatter(lambda value, _pos: f"{float(value):g}" if value > 0.0 else "")
    ax.xaxis.set_major_formatter(decimal_log_formatter)
    ax.yaxis.set_major_formatter(decimal_log_formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _restore_interactive_backend() -> None:
    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        return
    for backend in ("MacOSX", "TkAgg"):
        try:
            plt.switch_backend(backend)
            return
        except Exception:
            continue


def _advance_state(state: VolumetricPitoisState, *, n_steps: int) -> None:
    _assert_fixed_topology(state)
    for _step in range(int(n_steps)):
        step_dt = _select_physical_dt(state)
        substeps = max(1, int(state.config.integration_substeps))
        sub_dt = float(step_dt) / float(substeps)
        for _substep in range(substeps):
            if _advance_gmsh_incompressible_substep(state, dt=sub_dt):
                continue
            _set_cap_velocities(state)
            _enforce_no_swirl_velocity_field(state)
            _move_caps(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt)
            symplectic_euler(
                state.HC,
                state.bV_caps,
                _vertex_acceleration,
                dt=sub_dt,
                n_steps=1,
                dim=3,
                retopologize_fn=False,
                workers=max(1, int(state.config.accel_workers)),
                state=state,
            )
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _set_cap_velocities(state)
            _update_moving_contact_line(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _assert_fixed_topology(state)
            _apply_gmsh_contact_line_volume_slide(state, dt=sub_dt)
            _apply_gmsh_position_volume_constraint(state)
            _project_volume_to_target(state)
            _gmsh_axisymmetrize_state(state)
            _assert_fixed_topology(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt, refine=False)
            _assert_fixed_topology(state)
        _assert_state_finite(state, where=f"advance step {_step + 1}")
        state.elapsed_time_s += float(step_dt)


def _show_state_interactive(
    state: VolumetricPitoisState,
    *,
    step: int,
    elev: float,
    azim: float,
) -> None:
    _restore_interactive_backend()

    bottom_center_phys, top_center_phys, axis = _particle_centers_physical(state)
    center_distance = float(np.dot(top_center_phys - bottom_center_phys, axis))
    surface_gap = float(_gap(state))
    surface_gap_m = float(_length_from_compute(state.config, surface_gap))
    center_distance_m = float(_length_from_compute(state.config, center_distance))
    overlap = center_distance < 2.0 * float(state.config.particle_radius)

    side_triangles_full = _surface_side_triangles(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "meridian_strip":
        side_triangles, bottom_cap_triangles, top_cap_triangles = _actual_compute_meridian_strip_triangles(state)
    elif USER_SHOW_REAL_COMPUTE_TRIANGLES:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    else:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _sphere_cap_triangles(state)
        side_segments, side_vertices = _front_band_wireframe(state, elev=elev, azim=azim)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        if USER_REAL_TRIANGLE_VIEW == "meridian_strip" and not USER_SHOW_MESH_FACES:
            side_segments, side_vertices = _meridian_section_wireframe(state)
        else:
            side_segments, side_vertices = _surface_side_wireframe(state)
    side_vertices = _wireframe_scatter_vertices(side_segments, min_degree=4)
    plot_side_segments = _plot_coords_mm(side_segments, state.config)
    plot_side_vertices = _plot_coords_mm(side_vertices, state.config)
    plot_side_tris = _plot_coords_mm(side_triangles, state.config)
    plot_bottom_cap_tris = _plot_coords_mm(bottom_cap_triangles, state.config)
    plot_top_cap_tris = _plot_coords_mm(top_cap_triangles, state.config)
    bottom_center, top_center = _particle_center_positions(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * float(_length_from_compute(state.config, state.config.particle_radius))

    fig = plt.figure(figsize=(8.4, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    liquid_triangle_sets = [plot_side_tris] if plot_side_tris.size else []
    if USER_FILL_PARTICLE_CAP_SURFACES:
        if plot_bottom_cap_tris.size:
            liquid_triangle_sets.append(plot_bottom_cap_tris)
        if plot_top_cap_tris.size:
            liquid_triangle_sets.append(plot_top_cap_tris)
    if plot_side_tris.size and USER_SHOW_SURFACE_OVERLAY:
        overlay_poly = Poly3DCollection(
            plot_side_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_SURFACE_OVERLAY_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(overlay_poly)
    if liquid_triangle_sets and USER_SHOW_MESH_FACES:
        poly_liquid = Poly3DCollection(
            np.concatenate(liquid_triangle_sets, axis=0),
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size:
        _render_triangle_edges(
            ax,
            plot_side_tris,
            color="#0f172a",
            linewidth=0.42,
            alpha=0.55,
        )
    scatter_vertices_from = plot_side_tris if (
        USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size
    ) else plot_side_vertices
    _scatter_mesh_vertices(ax, [scatter_vertices_from])
    ax.add_collection3d(
        Line3DCollection(
            plot_side_segments,
            colors="#111111",
            linewidths=0.72,
            alpha=0.995,
            axlim_clip=True,
        )
    )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES:
        _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
        _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    if USER_SHOW_MESH_FACES and plot_bottom_cap_tris.size:
        poly_bottom_caps = Poly3DCollection(
            plot_bottom_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#2b6cb0",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_bottom_caps)
    if USER_SHOW_MESH_FACES and plot_top_cap_tris.size:
        poly_top_caps = Poly3DCollection(
            plot_top_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#dd8a1c",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_top_caps)

    if USER_FILL_PARTICLE_CAP_SURFACES:
        _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
        _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, contact_radius_mm, axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, contact_radius_mm, axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )

    label = "initial" if step == 0 else f"iteration {step}"
    ax.set_title(
        _snapshot_title(state, label=label)
        + f"\niteration={step}, gap={surface_gap_m * 1.0e6:.2f} um, overlap={'yes' if overlap else 'no'}"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    plot_all = []
    if plot_side_tris.size:
        plot_all.append(plot_side_tris)
    if plot_bottom_cap_tris.size:
        plot_all.append(plot_bottom_cap_tris)
    if plot_top_cap_tris.size:
        plot_all.append(plot_top_cap_tris)
    plot_xyz = np.concatenate(plot_all, axis=0) if plot_all else np.empty((0, 3, 3))
    _set_axes_equal(ax, plot_xyz, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)

    print("Interactive Case_2b_axisym mesh viewer")
    print("Use the mouse to rotate, zoom, and pan the figure window.")
    print(f"Step = {step}")
    print(f"Surface gap = {surface_gap_m * 1.0e6:.2f} um")
    print(f"Center distance = {center_distance_m * 1.0e3:.5f} mm")
    print(f"Sphere overlap = {'yes' if overlap else 'no'}")
    plt.show()


def open_mesh_viewer(
    *,
    step: int = 0,
    refinement: int = 1,
    elev: float = 23.0,
    azim: float = -90.0,
) -> None:
    state = _prepare_state(replace(separation_config(), refinement=refinement))
    _advance_state(state, n_steps=step)
    _show_state_interactive(state, step=step, elev=elev, azim=azim)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view-initial",
        action="store_true",
        help="Open an interactive window for the Case_2b_axisym initial mesh instead of running the simulation.",
    )
    parser.add_argument(
        "--view-step",
        type=int,
        default=None,
        help="Open an interactive window for the Case_2b_axisym mesh after this many separation steps.",
    )
    parser.add_argument("--refinement", type=int, default=USER_REFINEMENT, help="Mesh refinement for --view-initial.")
    parser.add_argument("--elev", type=float, default=USER_INTERACTIVE_ELEV_DEG, help="Camera elevation for --view-initial.")
    parser.add_argument("--azim", type=float, default=USER_INTERACTIVE_AZIM_DEG, help="Camera azimuth for --view-initial.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override USER_TOTAL_STEPS for a short smoke run or a longer production run.",
    )
    parser.add_argument(
        "--integration-substeps",
        type=int,
        default=None,
        help="Override USER_INTEGRATION_SUBSTEPS; useful for finite-K compressible EOS stability.",
    )
    parser.add_argument(
        "--closure",
        choices=("compressible", "incompressible", "legacy"),
        default=USER_PRESSURE_CLOSURE,
        help="Pressure/volume closure for the Gmsh raw simulation path.",
    )
    parser.add_argument(
        "--compressible-bulk-pa",
        type=float,
        default=USER_COMPRESSIBLE_BULK_MODULUS_PA,
        help="Physical bulk modulus used by --closure compressible before nondimensional scaling.",
    )
    parser.add_argument(
        "--compressible-eos-position-relaxation",
        type=float,
        default=USER_COMPRESSIBLE_EOS_POSITION_CORRECTION_RELAXATION,
        help="Relaxation factor for the finite-K EOS geometry stabilizer; 0 disables the position correction.",
    )
    parser.add_argument(
        "--compressible-eos-pressure-delta-limit-fraction",
        type=float,
        default=USER_COMPRESSIBLE_EOS_PRESSURE_DELTA_LIMIT_FRACTION,
        help="Finite-EOS bound for the PR33 pressure correction: |delta p| <= fraction * K; 0 disables.",
    )
    parser.add_argument(
        "--compressible-eos-reference-volume-relaxation",
        type=float,
        default=USER_COMPRESSIBLE_EOS_REFERENCE_VOLUME_RELAXATION,
        help="ALE remap fraction for finite-K PR33 per-tet reference volumes; 0 keeps the initial material volumes.",
    )
    parser.add_argument(
        "--compressible-incompressible-limit-projection",
        dest="compressible_use_incompressible_limit_projection",
        action="store_true",
        default=USER_COMPRESSIBLE_USE_INCOMPRESSIBLE_LIMIT_PROJECTION,
        help="Use the K->infinity limit solve of the compressible PR33 EOS pressure equation.",
    )
    parser.add_argument(
        "--no-compressible-incompressible-limit-projection",
        dest="compressible_use_incompressible_limit_projection",
        action="store_false",
        help="Use the finite-K compressible PR33 EOS pressure equation.",
    )
    parser.add_argument(
        "--gorge-pressure-model",
        choices=("axisym_fit", "fheron_neck", "axisym_fheron_floor", "axisym_initial_fheron_neck"),
        default=USER_GORGE_PRESSURE_MODEL,
        help="Neck pressure source for the Pitois gorge pressure term.",
    )
    parser.add_argument(
        "--fheron-neck-pressure-factor",
        type=float,
        default=USER_FHERON_NECK_PRESSURE_FACTOR,
        help="Positive convention factor applied to |FHeron|/(gamma*A_Heron) in --gorge-pressure-model fheron_neck.",
    )
    parser.add_argument(
        "--capillary-line-radius-mode",
        choices=("pitois_eq6", "current", "loaded", "min_current_pitois_eq6"),
        default=USER_REPORTED_CAPILLARY_LINE_RADIUS_MODE,
        help="Legacy effective wetted-radius mode kept for sensitivity metadata; Fig. 5 uses the gorge radius.",
    )
    parser.add_argument(
        "--initial-msh-path",
        type=Path,
        default=None,
        help="Initial Gmsh .msh file. Defaults to the original r=1.54 mm mesh.",
    )
    parser.add_argument(
        "--initial-contact-radius-mode",
        choices=("loaded", "pitois_eq6", "pitois_eq6_bulged"),
        default=USER_INITIAL_CONTACT_RADIUS_MODE,
        help=(
            "Initial raw Gmsh geometry: keep the loaded .msh contact radius, move it to the "
            "Pitois Eq. [6] radius, or use a volume-consistent bulged Eq. [6] profile."
        ),
    )
    parser.add_argument(
        "--initial-bulged-neck-radius-mm",
        type=float,
        default=None,
        help="Neck/gorge radius in mm for --initial-contact-radius-mode pitois_eq6_bulged.",
    )
    parser.add_argument(
        "--initial-bulged-contact-slope-scale",
        type=float,
        default=None,
        help="Contact-slope multiplier for --initial-contact-radius-mode pitois_eq6_bulged.",
    )
    parser.add_argument(
        "--cox-contact-line-force",
        dest="use_cox_contact_line_force",
        action="store_true",
        default=USER_USE_COX_CONTACT_LINE_FORCE,
        help="Add Cox/Young dynamic contact-line force to contact vertices.",
    )
    parser.add_argument(
        "--no-cox-contact-line-force",
        dest="use_cox_contact_line_force",
        action="store_false",
        help="Keep Cox/contact-angle physics in geometry/reporting only; do not add contact-line vertex force.",
    )
    parser.add_argument(
        "--gmsh-contact-angle-kinematics",
        dest="enable_gmsh_contact_angle_kinematics",
        action="store_true",
        default=USER_ENABLE_GMSH_CONTACT_ANGLE_KINEMATICS,
        help="Enable the Gmsh contact-angle geometry update for moving contact rings.",
    )
    parser.add_argument(
        "--no-gmsh-contact-angle-kinematics",
        dest="enable_gmsh_contact_angle_kinematics",
        action="store_false",
        help="Disable the Gmsh contact-angle geometry update; contact rings remain constrained to the solid geometry.",
    )
    parser.add_argument(
        "--contact-line-max-slide-um",
        type=float,
        default=USER_CONTACT_LINE_MAX_SLIDE_UM,
        help="Maximum geometric contact-line slide per step in micrometers for contact-angle kinematics.",
    )
    parser.add_argument(
        "--solver-viscous-force-model",
        choices=("edge_flux", "tet_cauchy"),
        default="tet_cauchy",
        help="Solver viscous force for Ftot_i: edge_flux or tetrahedral Cauchy strain-rate assembly.",
    )
    parser.add_argument(
        "--solver-bulk-viscosity-pa-s",
        type=float,
        default=USER_SOLVER_BULK_VISCOSITY_PA_S,
        help="Optional compressible bulk viscosity zeta [Pa s] added as zeta*div(u)*I in tet Cauchy stress.",
    )
    parser.add_argument(
        "--solver-pr33-pressure-force",
        dest="enable_solver_pr33_pressure_force",
        action="store_true",
        default=USER_ENABLE_SOLVER_PR33_PRESSURE_FORCE,
        help="Add PR33/ddgclib pressure force to solver Ftot_i in the tet_cauchy path.",
    )
    parser.add_argument(
        "--no-solver-pr33-pressure-force",
        dest="enable_solver_pr33_pressure_force",
        action="store_false",
        help="Disable PR33/ddgclib pressure force in solver Ftot_i.",
    )
    parser.add_argument(
        "--solver-pr33-pressure-include-hydrostatic",
        dest="solver_pr33_pressure_include_hydrostatic",
        action="store_true",
        default=USER_SOLVER_PR33_PRESSURE_INCLUDE_HYDROSTATIC,
        help="Include hydrostatic pressure inside PR33 Fp. Leave off when using separate solver hydrostatic force.",
    )
    parser.add_argument(
        "--no-solver-pr33-pressure-include-hydrostatic",
        dest="solver_pr33_pressure_include_hydrostatic",
        action="store_false",
        help="Exclude hydrostatic pressure from PR33 Fp to avoid double counting separate hydrostatic force.",
    )
    parser.add_argument(
        "--solver-pr33-pressure-contact-line-mobility",
        dest="solver_pr33_pressure_contact_line_mobility",
        action="store_true",
        default=USER_SOLVER_PR33_PRESSURE_CONTACT_LINE_MOBILITY,
        help="Allow PR33 pressure projection to move contact-line vertices along the solid tangent.",
    )
    parser.add_argument(
        "--no-solver-pr33-pressure-contact-line-mobility",
        dest="solver_pr33_pressure_contact_line_mobility",
        action="store_false",
        help="Constrain contact-line vertices during the PR33 pressure projection while still computing PR33 B-force diagnostics.",
    )
    parser.add_argument(
        "--solver-pr33-pressure-reference-mode",
        choices=("heron", "zero"),
        default="zero",
        help="Reference pressure gauge for compressible PR33 pressure: Heron projection or zero gauge.",
    )
    parser.add_argument(
        "--solver-pr33-use-returned-velocity",
        dest="solver_pr33_use_returned_velocity",
        action="store_true",
        default=USER_SOLVER_PR33_USE_RETURNED_VELOCITY,
        help="Use the velocity returned by the ddgclib PR33 pressure correction.",
    )
    parser.add_argument(
        "--no-solver-pr33-use-returned-velocity",
        dest="solver_pr33_use_returned_velocity",
        action="store_false",
        help="Use the PR33 pressure force map B^T p for the bridge pressure update.",
    )
    parser.add_argument(
        "--solver-hydrostatic-force",
        dest="enable_solver_hydrostatic_force",
        action="store_true",
        default=USER_ENABLE_SOLVER_HYDROSTATIC_FORCE,
        help="Add hydrostatic surface-pressure force to solver Ftot_i.",
    )
    parser.add_argument(
        "--no-solver-hydrostatic-force",
        dest="enable_solver_hydrostatic_force",
        action="store_false",
        help="Disable the solver Ftot_i hydrostatic surface-pressure force.",
    )
    parser.add_argument(
        "--hydrostatic-zref",
        choices=("midpoint", "top_cl"),
        default=USER_SOLVER_HYDROSTATIC_ZREF_MODE,
        help="Reference z for hydrostatic pressure rho*g*(z_ref-z).",
    )
    parser.add_argument(
        "--accel-workers",
        type=int,
        default=USER_ACCEL_WORKERS,
        help="Worker count for ddgclib per-vertex force acceleration evaluation.",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=USER_MAX_ACCELERATION,
        help="Physical acceleration limiter [m/s^2] for explicit vertex updates.",
    )
    parser.add_argument(
        "--incompressible-projection-regularization",
        type=float,
        default=USER_INCOMPRESSIBLE_PROJECTION_REGULARIZATION,
        help="Diagonal regularization used by the PR#33 incompressible pressure projection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for this run. Defaults to the script-name directory.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=None,
        help="Override the history record interval.",
    )
    parser.add_argument(
        "--mesh-snapshot-every",
        type=int,
        default=None,
        help="Override the mesh snapshot interval; use 0 to skip step snapshots.",
    )
    parser.add_argument(
        "--history-png-every",
        type=int,
        default=None,
        help="Override the in-run history PNG refresh interval; use 0 to render only at the end.",
    )
    parser.add_argument(
        "--no-raw-mesh-png",
        action="store_true",
        help="Write raw mesh .msh snapshots but skip the solver-side PNG render for each snapshot.",
    )
    parser.add_argument(
        "--interactive-step",
        type=int,
        default=None,
        help="Open the interactive mesh viewer after this completed step.",
    )
    parser.add_argument(
        "--no-fig",
        action="store_true",
        help="Skip PNG/MSH rendering and comparison plots.",
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Skip writing CSV/JSON history outputs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    return parser


def _save_case2b_histories_panel(*, separation: dict[str, np.ndarray], out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    axes[0].plot(separation["t_ms"], separation["d_over_r"], color="#bc4749", linewidth=1.8)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel(r"$D/R$")
    axes[0].set_title("Case_2b_axisym normalized gap")
    axes[0].grid(alpha=0.28)

    axes[1].plot(separation["t_ms"], separation["max_u"], color="#bc4749", linewidth=1.8)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Max free speed [m/s]")
    axes[1].set_title("Case_2b_axisym free-surface speed")
    axes[1].grid(alpha=0.28)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_pitois_fig5_comparison(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    fig_dir = out_dir
    results_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    compare_mask = np.array(
        [int(row.get("step", 0)) >= int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS) for row in separation_history],
        dtype=bool,
    )
    if not np.any(compare_mask):
        compare_mask = np.ones_like(separation["d_over_r"], dtype=bool)

    sim_x = separation["d_over_r"][compare_mask]
    sim_y = separation["force_abs_mn"][compare_mask]
    order = np.argsort(sim_x)
    sim_x_sorted = sim_x[order]
    sim_y_sorted = sim_y[order]
    unique_x, unique_idx = np.unique(sim_x_sorted, return_index=True)
    unique_y = sim_y_sorted[unique_idx]
    interp_mask = (exp_x >= float(np.min(unique_x))) & (exp_x <= float(np.max(unique_x))) if unique_x.size else np.zeros_like(exp_x, dtype=bool)
    if np.count_nonzero(interp_mask) >= 2:
        interp_y = np.interp(exp_x[interp_mask], unique_x, unique_y)
        err = interp_y - exp_y[interp_mask]
        rel = err / np.maximum(exp_y[interp_mask], 1.0e-30)
        rmse_mn = float(math.sqrt(float(np.mean(err * err))))
        mape = float(np.mean(np.abs(rel)))
        max_abs_rel = float(np.max(np.abs(rel)))
    else:
        rmse_mn = float("nan")
        mape = float("nan")
        max_abs_rel = float("nan")
    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=sim_x,
        sim_y=sim_y,
        out_path=out_dir / FIG5_COMPARE_PNG_NAME,
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
    )
    _save_case2b_histories_panel(
        separation=separation,
        out_path=fig_dir / "pitois2000_volumetric_case2b_histories.png",
    )

    summary = {
        "separation": {
            "d_over_r_min": float(np.min(separation["d_over_r"])),
            "d_over_r_max": float(np.max(separation["d_over_r"])),
            "force_abs_mn_min": float(np.min(separation["force_abs_mn"])),
            "force_abs_mn_max": float(np.max(separation["force_abs_mn"])),
            "fig5_compare_skip_initial_steps": int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS),
            "fig5_compare_first_plotted_step": int(
                separation_history[int(np.flatnonzero(compare_mask)[0])]["step"]
                if np.any(compare_mask)
                else separation_history[0]["step"]
            ),
            "fig5_compare_note": (
                "The red curve includes the imported t=0 geometry and every completed "
                "post-advance step written during the run."
            ),
            "fig5_overlap_points": int(np.count_nonzero(interp_mask)),
            "fig5_overlap_rmse_mn": rmse_mn,
            "fig5_overlap_mape": mape,
            "fig5_overlap_max_abs_rel": max_abs_rel,
        },
        "digitized_fig5_dynamic": {
            "d_over_r": exp_x.tolist(),
            "force_mn": exp_y.tolist(),
        },
    }
    (results_dir / "pitois2000_volumetric_summary.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "pitois2000_fig5_dynamic_digitized.json").write_text(
        json.dumps(
            {
                "source": "Pitois 2000 Fig. 5 dynamic experimental points digitized from the local working copy",
                "d_over_r": exp_x.tolist(),
                "force_mn": exp_y.tolist(),
            },
            indent=2,
        )
    )


def _refresh_pitois_fig5_compare_png(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    if not separation_history:
        return
    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    compare_mask = np.array(
        [int(row.get("step", 0)) >= int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS) for row in separation_history],
        dtype=bool,
    )
    if not np.any(compare_mask):
        compare_mask = np.ones_like(separation["d_over_r"], dtype=bool)
    sim_x = separation["d_over_r"][compare_mask]
    sim_y = separation["force_abs_mn"][compare_mask]
    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=sim_x,
        sim_y=sim_y,
        out_path=out_dir / FIG5_COMPARE_PNG_NAME,
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
    )


def _refresh_non_mesh_png_outputs(
    history: list[dict],
    config: VolumetricPitoisConfig,
    out_dir: Path,
    *,
    initial_snapshot_volume_m3: float,
) -> None:
    if not history:
        return
    _refresh_pitois_fig5_compare_png(
        separation_history=history,
        out_dir=out_dir,
    )
    _render_history_figures(
        history,
        config,
        out_dir,
        initial_snapshot_volume_m3=initial_snapshot_volume_m3,
    )


def _print_header(config: VolumetricPitoisConfig) -> None:
    total_cl_rings = int(config.contact_line_radial_rings) + int(config.extra_cl_radial_rings)
    print("=" * 72)
    print(f"  {config.title}")
    print("=" * 72)
    print(f"Refinement                = {config.refinement}")
    print(f"CL radial rings           = {config.contact_line_radial_rings}")
    print(f"Extra CL outer rings      = {config.extra_cl_radial_rings}")
    print(f"CL radial bias ratio      = {config.contact_line_radial_bias_ratio}")
    print(f"Sidewall axial stride     = {config.axial_ring_stride}")
    print(f"Neck extra axial layers   = {config.neck_extra_axial_layers}")
    print(f"CL extra axial layers     = {config.cl_extra_axial_layers}")
    if total_cl_rings <= 1 and abs(float(config.contact_line_radial_bias_ratio) - 1.0) > 1.0e-12:
        print("CL radial bias active?    = no (need at least 2 total CL rings)")
    print(
        "Computation units         = "
        f"{'dimensionless VolFlux0_v3 (R=1, gamma=1, legacy volume mass)' if config.compute_dimensionless else 'SI'}"
    )
    if config.compute_dimensionless:
        print(
            "Nondim scales             = "
            f"L0 {config.length_scale_m:.6e} m, "
            f"T0 {config.time_scale_s:.6e} s, "
            f"F0 {config.force_scale_n:.6e} N"
        )
        print(
            "Nondim groups             = "
            f"mu_hat {config.mu_f:.6e}, Bo {float(config.rho_f) * float(config.gravity_mps2):.6e}, "
            f"Uhat {config.cap_speed:.6e}"
        )
    print(f"Particle radius           = {float(_length_from_compute(config, config.particle_radius)) * 1e3:.3f} mm")
    print(f"Eq.6 contact radius target= {float(_length_from_compute(config, config.target_cap_radius)) * 1e3:.3f} mm")
    print(f"Initial contact geometry  = {config.initial_contact_radius_mode}")
    print(f"Surface tension           = {float(_surface_tension_from_compute(config, config.gamma)):.4f} N/m")
    print(
        "Heron surface force       = "
        f"{'on' if bool(getattr(config, 'enable_heron_surface_tension', False)) else 'off (Laplace pressure only)'}"
    )
    print(
        "Solver CL vertex force    = "
        f"{'Cox/Young on' if bool(getattr(config, 'use_cox_contact_line_force', False)) else 'off (geometry/reporting only)'}"
    )
    print(f"Viscosity                 = {float(_viscosity_from_compute(config, config.mu_f)):.3e} Pa s")
    print(f"Density                   = {float(_density_from_compute(config, config.rho_f)):.3f} kg/m^3")
    print(f"Gravity included          = {'yes' if config.include_gravity else 'no'}")
    if config.include_gravity:
        print(f"Gravity acceleration      = {float(_acceleration_from_compute(config, config.gravity_mps2)):.3f} m/s^2")
    print(f"Contact angle assumption  = {config.contact_angle_deg:.1f} deg (literature-informed baseline)")
    print(f"Moving-sphere speed       = {float(_velocity_from_compute(config, config.cap_speed)) * 1e6:.3f} um/s")
    print(f"Relative speed            = {float(_velocity_from_compute(config, config.relative_speed)) * 1e6:.3f} um/s")
    print("Fixed sphere              = top sphere (balance side)")
    print("Moving sphere             = bottom sphere (stage side)")
    print(f"Integration substeps      = {config.integration_substeps}")
    print(f"Contact-radius samples    = {config.contact_radius_samples}")
    closure = str(getattr(config, "pressure_closure", "legacy"))
    print(f"Pressure closure          = {closure}")
    print(
        "Gorge pressure source     = "
        f"{getattr(config, 'gorge_pressure_model', USER_GORGE_PRESSURE_MODEL)} "
        f"(FHeron factor {float(getattr(config, 'fheron_neck_pressure_factor', USER_FHERON_NECK_PRESSURE_FACTOR)):.6g})"
    )
    if closure == "compressible":
        print(
            "Compressible bulk modulus = "
            f"{float(getattr(config, 'compressible_bulk_modulus_pa', 0.0)):.6e} Pa "
            f"(nondim {float(getattr(config, 'compressible_bulk_modulus', 0.0)):.6e})"
        )
    if closure == "incompressible":
        projection_label = "PR#33 B=dV/dx incompressible projection on Gmsh tets"
    elif closure == "compressible":
        projection_label = "PR#33 B=dV/dx weak-compressible EOS correction on Gmsh tets"
    elif bool(getattr(config, "enable_incompressible_projection", False)):
        projection_label = "velocity-pressure projection on Gmsh tets"
    elif bool(USER_ENFORCE_FULL_AXISYMMETRY):
        projection_label = (
            "Fp,proj force only (no geometric projector)"
            if config.enable_volume_projection
            else "off"
        )
    else:
        projection_label = "on" if config.enable_volume_projection else "off"
    print(f"Volume projection         = {projection_label}")
    if (
        bool(getattr(config, "gmsh_compute_mesh", False))
        or bool(getattr(config, "enable_incompressible_projection", False))
        or closure in {"compressible", "incompressible"}
    ):
        print(
            "Gmsh volume correction    = "
            f"{'on' if config.enable_gmsh_geometric_volume_correction else 'off'}"
            f" (trigger {config.gmsh_geometric_volume_correction_trigger_rel:.3e})"
        )
        print(
            "Gmsh CL volume slide      = "
            f"{'on' if USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE else 'off'}"
            f" (trigger {USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL:.3e})"
        )
        print(
            "Gmsh position volume fix  = "
            f"{'on' if USER_ENABLE_POSITION_VOLUME_CONSTRAINT else 'off'}"
            f" (trigger {USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL:.3e})"
        )
        print(
            "Gmsh kinematic volume     = "
            f"{'on' if USER_ENABLE_KINEMATIC_VOLUME_CONSTRAINT else 'off'}"
            f" (tol {USER_KINEMATIC_VOLUME_CONSTRAINT_REL_TOL:.3e})"
        )
        print(
            "Gmsh volume flux fix      = "
            f"{'on' if USER_ENABLE_VOLUME_FLUX_CORRECTION else 'off'}"
            f" (cap flux {'included' if USER_VOLUME_FLUX_INCLUDE_CAP_BOUNDARY_FLUX else 'excluded'}, "
            f"relax {USER_VOLUME_FLUX_RELAXATION:.3f}, "
            f"volume gain {USER_VOLUME_FLUX_VOLUME_ERROR_GAIN:.3f})"
        )
        print(
            "Gmsh post-edit continuity = "
            f"{'on' if USER_ENABLE_POST_EDIT_CONTINUITY_PROJECTION else 'off'}"
            f" (tol {USER_POST_EDIT_CONTINUITY_REL_TOL:.3e}, "
            f"max correction {USER_POST_EDIT_CONTINUITY_MAX_CORRECTION_M:.3e} m)"
        )
        if config.enable_gmsh_wall_stress_sphere_force:
            force_label = "wall-stress"
        elif config.reported_gmsh_contact_line_force:
            force_label = "gorge-pressure + raw ring"
        elif config.reported_capillary_line_force:
            force_label = "gorge-pressure + neck surface tension"
        else:
            force_label = "gorge-pressure only"
        print(
            "Gmsh reported force       = "
            f"{force_label} "
            f"(gorge method, "
            f"scale {float(config.reported_force_scale):.6f}, "
            f"monotonic {'on' if config.reported_force_monotonic_envelope else 'off'}, "
            f"wall pressure {'on' if config.gmsh_wall_stress_include_pressure else 'off'}, "
            f"viscous {'on' if config.gmsh_wall_stress_include_viscous else 'off'}, "
            f"axisym-project {'on' if config.gmsh_wall_stress_project_axisym else 'off'})"
        )
    print(
        "Fp,proj trial refinement  = "
        f"{'off (replaced by incompressible projection)' if config.enable_incompressible_projection else ('on' if USER_ENABLE_PRESSURE_TRIAL_REFINEMENT else 'off')}"
    )
    print(f"No-swirl enforcement      = {'on' if config.enforce_no_swirl else 'off'}")
    if config.dt > 0.0 and not config.enable_adaptive_dt:
        print("Adaptive dt               = off (using USER_DT_S)")
        print(f"dt                        = {float(_time_from_compute(config, config.dt)):.3e} s")
    else:
        print(f"Adaptive dt               = {'on' if config.enable_adaptive_dt else 'off'}")
        dt_ceiling = float(config.dt) if float(config.dt) > 0.0 else float(config.adaptive_dt_max_s)
        if dt_ceiling > 0.0:
            print(f"dt ceiling                = {float(_time_from_compute(config, dt_ceiling)):.3e} s")
        else:
            print("dt ceiling                = none (USER_DT_S < 0: right_dt)")
        print(f"dt floor                  = {float(_time_from_compute(config, config.adaptive_dt_min_s)):.3e} s")
        print(f"Capillary dt safety       = {config.adaptive_dt_capillary_safety:.3f}")
        print(f"Mesh disp. fraction       = {config.adaptive_dt_mesh_displacement_frac:.3f}")
    print(f"Steps                     = {config.n_steps}")
    if config.dt > 0.0 and not config.enable_adaptive_dt:
        print(f"Run time                  = {float(_time_from_compute(config, config.dt * config.n_steps)):.3f} s")
    else:
        dt_ceiling = float(config.dt) if float(config.dt) > 0.0 else float(config.adaptive_dt_max_s)
        if dt_ceiling > 0.0:
            print(f"Run time ceiling          = {float(_time_from_compute(config, dt_ceiling * config.n_steps)):.3f} s")
        else:
            print("Run time ceiling          = adaptive/no fixed ceiling")
    print("=" * 72)


def _print_summary(history: list[dict], config: VolumetricPitoisConfig) -> None:
    fixed_force_mn = np.array([float(row["fixed_force_axial"]) for row in history], dtype=float) * 1.0e3
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)
    print("Separation run complete")
    print(f"D/R range                 = {float(np.min(d_over_r)):.6f} - {float(np.max(d_over_r)):.6f}")
    print(f"Fixed-sphere force [mN]   = {float(np.min(fixed_force_mn)):.6e} - {float(np.max(fixed_force_mn)):.6e}")
    print(f"Final max free speed      = {float(max_u[-1]):.6e} m/s")


def run_motion_case(
    config: VolumetricPitoisConfig,
    *,
    out_dir: Path | None = None,
    save_fig: bool = True,
    save_results: bool = True,
    verbose: bool = True,
    interactive_step: int | None = None,
    interactive_elev: float = USER_INTERACTIVE_ELEV_DEG,
    interactive_azim: float = USER_INTERACTIVE_AZIM_DEG,
) -> list[dict]:
    if out_dir is None:
        out_dir = OUT_ROOT

    if verbose:
        _print_header(config)

    state = _prepare_state(config)
    _assert_fixed_topology(state)
    target_snapshot_volume_ul = 1.0e9 * float(_volume_from_compute(config, state.target_snapshot_volume_m3))
    history: list[dict] = [_step_record(state, step=0, t=state.elapsed_time_s)]
    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    motion_mesh_dir = out_dir
    deferred_mesh_snapshots: list[SimpleNamespace] = []
    written_mesh_snapshot_filenames: set[str] = set()

    if save_fig:
        initial_mesh_snapshot = _capture_motion_mesh_snapshot(state, label="initial", step=0)
        _write_captured_raw_volume_mesh_snapshot(initial_mesh_snapshot, motion_mesh_dir)
        written_mesh_snapshot_filenames.add(str(initial_mesh_snapshot.filename))

    for step in range(config.n_steps):
        state.current_separation_step = int(step + 1)
        step_timing: dict[str, float] = defaultdict(float)
        step_start = time.perf_counter()
        with _time_module(step_timing, "dt_select"):
            step_dt = _select_physical_dt(state)
        substeps = max(1, int(config.integration_substeps))
        sub_dt = float(step_dt) / float(substeps)
        for _substep in range(substeps):
            if _advance_gmsh_incompressible_substep(state, dt=sub_dt, timing=step_timing):
                continue
            with _time_module(step_timing, "cap_move_axisym"):
                _set_cap_velocities(state)
                _enforce_no_swirl_velocity_field(state)
                _move_caps(state, dt=sub_dt)
                _gmsh_axisymmetrize_state(state)
            with _time_module(step_timing, "mass_dual"):
                _update_duals_and_masses(state)
            with _time_module(step_timing, "pressure_scalar"):
                _update_pressure_scalar(state)
            with _time_module(step_timing, "pressure_projection"):
                _update_pressure_projection_scalar(state, dt=sub_dt)
            with _time_module(step_timing, "force_accel"):
                symplectic_euler(
                    state.HC,
                    state.bV_caps,
                    _vertex_acceleration,
                    dt=sub_dt,
                    n_steps=1,
                    dim=3,
                    retopologize_fn=False,
                    workers=max(1, int(state.config.accel_workers)),
                    state=state,
                )
            with _time_module(step_timing, "contact_line"):
                _gmsh_axisymmetrize_state(state)
                _enforce_no_swirl_velocity_field(state)
                _set_cap_velocities(state)
                _update_moving_contact_line(state, dt=sub_dt)
                _gmsh_axisymmetrize_state(state)
                _enforce_no_swirl_velocity_field(state)
                _assert_fixed_topology(state)
                _apply_gmsh_contact_line_volume_slide(state, dt=sub_dt)
                _apply_gmsh_position_volume_constraint(state)
                _project_volume_to_target(state)
                _gmsh_axisymmetrize_state(state)
            _assert_fixed_topology(state)
            with _time_module(step_timing, "final_mass_pressure"):
                _update_duals_and_masses(state)
                _update_pressure_scalar(state)
                _update_pressure_projection_scalar(state, dt=sub_dt, refine=False)
            _assert_fixed_topology(state)
        _assert_state_finite(state, where=f"step {step + 1}")
        state.elapsed_time_s += float(step_dt)

        completed_step = step + 1
        with _time_module(step_timing, "record"):
            record_row = None
            if verbose or completed_step % max(1, config.record_every) == 0:
                record_row = _step_record(state, step=completed_step, t=state.elapsed_time_s)
            if verbose:
                snapshot_volume_ul = 1.0e9 * float(_volume_from_compute(config, _snapshot_msh_volume_m3(state)))
                rel_snapshot_volume_error = (snapshot_volume_ul - target_snapshot_volume_ul) / max(
                    target_snapshot_volume_ul, 1.0e-30
                )
        if verbose:
            print(
                f"Current step              = {completed_step}/{config.n_steps}",
                flush=True,
            )
            print(
                f"Adaptive dt               = {float(_time_from_compute(config, step_dt)):.6e} s ({state.last_dt_limiter})",
                flush=True,
            )
            print(
                f"Water vol. error          = {rel_snapshot_volume_error:+.6e} ({100.0 * rel_snapshot_volume_error:+.4f} %)",
                flush=True,
            )
            print(
                f"Water-filled mesh volume  = {snapshot_volume_ul:.6f} uL",
                flush=True,
            )
            if record_row is not None:
                print(
                    "F each step              = "
                    f"fixed_axial {1.0e3 * float(record_row['fixed_force_axial']):+.6e} mN, "
                    f"moving_axial {1.0e3 * float(record_row['moving_force_axial']):+.6e} mN, "
                    f"|fixed| {1.0e3 * float(record_row['fixed_force_mag']):.6e} mN, "
                    f"|moving| {1.0e3 * float(record_row['moving_force_mag']):.6e} mN",
                    flush=True,
                )
                print(
                    "CL slide speed           = "
                    f"bottom {float(record_row['bottom_contact_line_speed']):.6e} m/s, "
                    f"top {float(record_row['top_contact_line_speed']):.6e} m/s",
                    flush=True,
                )
                print(
                    "CL volume slide          = "
                    f"{record_row['gmsh_cl_volume_slide_status']}, "
                    f"scale {float(record_row['gmsh_cl_volume_slide_scale']):.6e}, "
                    f"rel {float(record_row['gmsh_cl_volume_slide_rel_before']):+.6e}"
                    f" -> {float(record_row['gmsh_cl_volume_slide_rel_after']):+.6e}",
                    flush=True,
                )
                print(
                    "Position volume fix      = "
                    f"{record_row['position_volume_constraint_status']}, "
                    f"rel {float(record_row['position_volume_constraint_rel_before']):+.6e}"
                    f" -> {float(record_row['position_volume_constraint_rel_after']):+.6e}",
                    flush=True,
                )
                print(
                    "Volume flux correction   = "
                    f"{record_row['final_noncl_volume_projection_status']}, "
                    f"Q {float(record_row['volume_flux_before_m3ps']):+.6e}"
                    f" -> {float(record_row['volume_flux_after_m3ps']):+.6e} m^3/s, "
                    f"target {float(record_row['volume_flux_target_m3ps']):+.6e}, "
                    f"lambda {float(record_row['volume_flux_lambda_mps']):+.6e} m/s",
                    flush=True,
                )
                if bool(getattr(state.config, "enable_incompressible_projection", False)):
                    print(
                        "Div(u) projection L2     = "
                        f"{float(_rate_from_compute(config, state.incompressible_divergence_before_l2)):.6e} -> "
                        f"{float(_rate_from_compute(config, state.incompressible_divergence_after_l2)):.6e} 1/s",
                        flush=True,
                    )
        with _time_module(step_timing, "record"):
            current_record_row = record_row if record_row is not None else _step_record(
                state,
                step=completed_step,
                t=state.elapsed_time_s,
            )
        history.append(current_record_row)
        if save_results and completed_step % 500 == 0:
            progress_path = out_dir / "separation_history_progress.json"
            with progress_path.open("w") as f:
                json.dump({"config": asdict(config), "history": history}, f, indent=2)
        if (
            save_fig
            and USER_HISTORY_PNG_EVERY_STEPS > 0
            and completed_step % max(1, int(USER_HISTORY_PNG_EVERY_STEPS)) == 0
        ):
            with _time_module(step_timing, "history_png"):
                _refresh_non_mesh_png_outputs(
                    history,
                    config,
                    out_dir,
                    initial_snapshot_volume_m3=float(
                        _volume_from_compute(config, state.target_snapshot_volume_m3)
                    ),
                )
        if interactive_step is not None and completed_step == max(0, int(interactive_step)):
            _show_state_interactive(
                state,
                step=completed_step,
                elev=interactive_elev,
                azim=interactive_azim,
            )

        if (
            save_fig
            and config.mesh_snapshot_every > 0
            and completed_step % config.mesh_snapshot_every == 0
        ):
            with _time_module(step_timing, "mesh_capture"):
                mesh_snapshot = _capture_motion_mesh_snapshot(
                    state,
                    label=f"iteration {completed_step}",
                    step=completed_step,
                )
                _write_captured_raw_volume_mesh_snapshot(mesh_snapshot, motion_mesh_dir)
                written_mesh_snapshot_filenames.add(str(mesh_snapshot.filename))

        step_timing["step_total"] = time.perf_counter() - step_start
        if verbose:
            _print_step_timing(step_timing, completed_step=completed_step, substeps=substeps)

    if config.n_steps > 0:
        final_step = int(config.n_steps)
        if not history or int(history[-1]["step"]) != final_step:
            final_record = _step_record(state, step=final_step, t=state.elapsed_time_s)
            history.append(final_record)

    if save_fig:
        for mesh_snapshot in deferred_mesh_snapshots:
            if str(mesh_snapshot.filename) in written_mesh_snapshot_filenames:
                continue
            _write_captured_raw_volume_mesh_snapshot(mesh_snapshot, motion_mesh_dir)
        _refresh_non_mesh_png_outputs(
            history,
            config,
            out_dir,
            initial_snapshot_volume_m3=float(_volume_from_compute(config, state.target_snapshot_volume_m3)),
        )

    if save_results:
        _save_history(history, config, out_dir)

    if verbose and history:
        print(f"Target bridge volume      = {target_snapshot_volume_ul:.6f} uL")
        _print_summary(history, config)

    return history


def main(argv: list[str] | None = None) -> None:
    global USER_HISTORY_PNG_EVERY_STEPS
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.steps is not None and int(args.steps) < 0:
        parser.error("--steps must be >= 0")
    if args.interactive_step is not None and int(args.interactive_step) < 0:
        parser.error("--interactive-step must be >= 0")
    if args.compressible_bulk_pa is not None and float(args.compressible_bulk_pa) <= 0.0:
        parser.error("--compressible-bulk-pa must be > 0")
    if args.fheron_neck_pressure_factor is not None and (
        (not math.isfinite(float(args.fheron_neck_pressure_factor)))
        or float(args.fheron_neck_pressure_factor) <= 0.0
    ):
        parser.error("--fheron-neck-pressure-factor must be positive and finite")
    if args.initial_bulged_neck_radius_mm is not None and float(args.initial_bulged_neck_radius_mm) <= 0.0:
        parser.error("--initial-bulged-neck-radius-mm must be > 0")
    if args.initial_bulged_contact_slope_scale is not None and (
        not math.isfinite(float(args.initial_bulged_contact_slope_scale))
    ):
        parser.error("--initial-bulged-contact-slope-scale must be finite")
    if args.record_every is not None and int(args.record_every) < 0:
        parser.error("--record-every must be >= 0")
    if args.mesh_snapshot_every is not None and int(args.mesh_snapshot_every) < 0:
        parser.error("--mesh-snapshot-every must be >= 0")
    if args.history_png_every is not None and int(args.history_png_every) < 0:
        parser.error("--history-png-every must be >= 0")
    if int(args.accel_workers) < 1:
        parser.error("--accel-workers must be >= 1")
    if (not math.isfinite(float(args.max_acceleration))) or float(args.max_acceleration) <= 0.0:
        parser.error("--max-acceleration must be finite and > 0")
    if args.incompressible_projection_regularization is not None and (
        (not math.isfinite(float(args.incompressible_projection_regularization)))
        or float(args.incompressible_projection_regularization) < 0.0
    ):
        parser.error("--incompressible-projection-regularization must be finite and >= 0")
    if args.history_png_every is not None:
        USER_HISTORY_PNG_EVERY_STEPS = int(args.history_png_every)
    if str(args.initial_contact_radius_mode) != "loaded":
        parser.error("case12 uses the loaded force-calibrated hourglass mesh only; Eq.6/r1.205 initialization is disabled.")
    if bool(args.use_cox_contact_line_force):
        parser.error("case12 is a receding hydrophilic case; contact-line force is disabled.")
    if not bool(args.enable_gmsh_contact_angle_kinematics):
        parser.error("case12 needs hourglass contact-angle/solid-sphere kinematics enabled.")
    initial_msh_path = ""
    if args.initial_msh_path is not None:
        initial_msh_path = str(Path(args.initial_msh_path).expanduser().resolve())
        if "r1p205" in initial_msh_path or "1.205" in initial_msh_path:
            parser.error("case12 must not use the r1.205/Eq.6 warped mesh.")

    if args.view_initial or args.view_step is not None:
        open_mesh_viewer(
            step=0 if args.view_step is None else max(0, int(args.view_step)),
            refinement=args.refinement,
            elev=args.elev,
            azim=args.azim,
        )
        return

    out_root = OUT_ROOT if args.out_dir is None else Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    config = separation_config()
    compressible_bulk_pa = float(args.compressible_bulk_pa)
    initial_bulged_neck_radius = float(config.initial_bulged_neck_radius)
    if args.initial_bulged_neck_radius_mm is not None:
        initial_bulged_neck_radius = (
            float(args.initial_bulged_neck_radius_mm)
            * 1.0e-3
            / max(float(config.length_scale_m), 1.0e-300)
        )
    initial_bulged_contact_slope_scale = float(config.initial_bulged_contact_slope_scale)
    if args.initial_bulged_contact_slope_scale is not None:
        initial_bulged_contact_slope_scale = float(args.initial_bulged_contact_slope_scale)
    initial_contact_radius_mode = str(args.initial_contact_radius_mode)
    solver_bulk_viscosity_compute = max(0.0, float(args.solver_bulk_viscosity_pa_s))
    if bool(getattr(config, "dimensionless_active", False)):
        solver_bulk_viscosity_compute /= max(float(getattr(config, "viscosity_scale_pas", 1.0)), 1.0e-300)
    config = replace(
        config,
        initial_msh_path=initial_msh_path,
        pressure_closure=str(args.closure),
        compressible_bulk_modulus_pa=compressible_bulk_pa,
        compressible_bulk_modulus=compressible_bulk_pa / max(float(config.pressure_scale_pa), 1.0e-300),
        compressible_eos_position_correction_relaxation=float(args.compressible_eos_position_relaxation),
        compressible_eos_pressure_delta_limit_fraction=max(
            0.0,
            float(args.compressible_eos_pressure_delta_limit_fraction),
        ),
        compressible_eos_reference_volume_relaxation=min(
            1.0,
            max(0.0, float(args.compressible_eos_reference_volume_relaxation)),
        ),
        compressible_use_incompressible_limit_projection=bool(
            args.compressible_use_incompressible_limit_projection
        ),
        enable_incompressible_projection=(
            str(args.closure) == "incompressible"
            or bool(args.compressible_use_incompressible_limit_projection)
        ),
        gorge_pressure_model=str(args.gorge_pressure_model),
        fheron_neck_pressure_factor=float(args.fheron_neck_pressure_factor),
        reported_capillary_line_radius_mode=str(args.capillary_line_radius_mode),
        # Case12 preserves the saved old-good #9 mixed state: loaded
        # hourglass mesh geometry, but Eq.6-derived target contact radius.
        use_pitois_eq6_contact_radius=True,
        initial_contact_radius_mode=initial_contact_radius_mode,
        initial_bulged_neck_radius=initial_bulged_neck_radius,
        initial_bulged_contact_slope_scale=initial_bulged_contact_slope_scale,
        use_cox_contact_line_force=bool(args.use_cox_contact_line_force),
        enable_gmsh_contact_angle_kinematics=bool(args.enable_gmsh_contact_angle_kinematics),
        contact_line_max_slide_um=max(0.0, float(args.contact_line_max_slide_um)),
        solver_viscous_force_model=str(args.solver_viscous_force_model),
        solver_bulk_viscosity_pa_s=solver_bulk_viscosity_compute,
        enable_solver_pr33_pressure_force=bool(args.enable_solver_pr33_pressure_force),
        solver_pr33_pressure_include_hydrostatic=bool(args.solver_pr33_pressure_include_hydrostatic),
        solver_pr33_pressure_contact_line_mobility=bool(args.solver_pr33_pressure_contact_line_mobility),
        solver_pr33_use_returned_velocity=bool(args.solver_pr33_use_returned_velocity),
        solver_pr33_pressure_reference_mode=str(args.solver_pr33_pressure_reference_mode),
        enable_solver_hydrostatic_force=bool(args.enable_solver_hydrostatic_force),
        solver_hydrostatic_zref_mode=str(args.hydrostatic_zref),
        accel_workers=int(args.accel_workers),
        max_acceleration=float(args.max_acceleration),
        incompressible_projection_regularization=float(args.incompressible_projection_regularization),
        render_raw_mesh_snapshot_png=(not bool(args.no_raw_mesh_png)),
    )
    if args.steps is not None:
        config = replace(config, n_steps=int(args.steps))
    if args.integration_substeps is not None:
        config = replace(config, integration_substeps=max(1, int(args.integration_substeps)))
    if args.record_every is not None:
        config = replace(config, record_every=max(1, int(args.record_every)))
    if args.mesh_snapshot_every is not None:
        config = replace(config, mesh_snapshot_every=int(args.mesh_snapshot_every))

    interactive_step = None
    if args.interactive_step is not None:
        interactive_step = int(args.interactive_step)
    elif USER_OPEN_INTERACTIVE_WINDOW:
        interactive_step = max(0, int(USER_INTERACTIVE_STEP))

    separation_history = run_motion_case(
        config,
        out_dir=out_root,
        save_fig=not args.no_fig,
        save_results=not args.no_results,
        verbose=not args.quiet,
        interactive_step=interactive_step,
        interactive_elev=USER_INTERACTIVE_ELEV_DEG,
        interactive_azim=USER_INTERACTIVE_AZIM_DEG,
    )
    if separation_history:
        if not args.no_fig:
            _write_pitois_fig5_comparison(
                separation_history=separation_history,
                out_dir=out_root,
            )
    else:
        print("No recorded history rows were written; skipping the Fig. 5 comparison output.")


if __name__ == "__main__":
    main()
