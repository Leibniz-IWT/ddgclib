#!/usr/bin/env python3
# ===================== _AR1.1_fix_mi_fine_curved.py =====================

# -------- easy knobs --------
N_STEPS    = 2000  # how many steps to run
SAVE_EVERY = 10    # save outputs every N steps (1 = every step)

# physical constants (AlCu10 droplet at 973 K, from PDF)
RHO0   = 2587.35          # kg/m^3 (AlCu10 at 973 K)
P0_ATM = 101325.0         # Pa
SIGMA  = 0.858            # N/m  (surface tension AlCu10/Ar)
MU     = 1.34e-3          # Pa·s (dynamic viscosity of AlCu10)
RHO_MIN, RHO_MAX = 6000.0, 8000.0   # plotting range (not used to clamp)
DT_CAP   = 1e-5  # time step cap

# ---- Rayleigh ellipsoid parameters (global, used for both mesh + reference) ----
AR_INIT = 1.05     # a_z / b_x
R_EQ    = 1.0e-3   # equivalent sphere radius (2 mm droplet, r = 1 mm)

# -------- curved-surface toggle (global) --------
USE_CURVED_VOLUME = False  # set to False for pure PL run

# -------- hard-guard toggle (post-step edge projection) --------
USE_HARD_GUARD = False   # set to False to disable the short-edge tangential projection

# -------- mass update toggle (dynamic vs fixed m_i) --------
UPDATE_MASS_EACH_STEP = True  # True => MASS_I_VEC recomputed from Vi_true each step

# -------- NEW: toggles to freeze Vi and Ai at t=0 --------
FIX_VI_AT_T0 = False   # True => use Vi_true0 for all steps (V_i fixed at t=0)
FIX_AI_AT_T0 = False   # True => use Astar_safe0 for all steps (A_i fixed at t=0)

# -------- surface-tension force method --------
FS_METHOD = "cotan"   # "heron" or "cotan"

# -------- mesh source toggle --------
# "gmsh" -> Gmsh ellipsoid mesh (current behavior)
# "hc"   -> ddgclib cube-surface mesh, then snapped to ellipsoid
MESH_SOURCE = "gmsh"   # or "hc"

# -------- stabilizers --------
# REMOVE_MEAN_DP now used to pin droplet: remove COM velocity + recenter positions
REMOVE_MEAN_DP = True       # True => pin droplet (no bulk COM drift)
C_DAMP      = 1.0e-4        # linear drag coeff in Fdamp = -C_DAMP * Astar * U
CFL_EDGE_FRAC = 100         # fraction of min edge per step
N_SUBSTEPS_PRESSURE = 1     # pressure substeps (keep 1 for more motion)

# -------- curved-volume behavior --------
STRICT_CV      = True          # fail hard if _curved_volume.py can’t run/parse
TIMEOUT_CV_S   = 60            # wall-clock timeout per CV attempt (seconds)
PLANE_REL_TOL  = 5e-3          # <- locked to 0.005 as requested

# -------- NEW: mass matrix type toggle --------
USE_CONSISTENT_MASS = False  # True => surface consistent mass matrix, False => lumped m_i

# -------- NEW: pressure gradient toggle --------
USE_PRESSURE_GRADIENT = True   # False => old uniform pressure only
P_GRAD_SCALE          = 1.0    # scale of curvature-based pressure variations

import sys, platform, numpy as np, os, csv, meshio, matplotlib, io, tempfile, importlib.util, shutil
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from contextlib import redirect_stdout, redirect_stderr
import warnings

# NEW: sparse matrix + CG for consistent mass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg

# ----- Heron curvature import -----
from _curvatures_heron import HNdC_ijk

# try gmsh (for auto mesh generation)
try:
    import gmsh
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False

# --------------------- noise / logging controls ---------------------
def pflush(*a, **kw):
    print(*a, **kw); sys.stdout.flush()

class silence_all:
    def __enter__(self):
        self._out = io.StringIO(); self._err = io.StringIO()
        self._ctx1 = redirect_stdout(self._out); self._ctx2 = redirect_stderr(self._err)
        self._ctx1.__enter__(); self._ctx2.__enter__()
        return self
    def __exit__(self, *exc):
        self._ctx2.__exit__(*exc); self._ctx1.__exit__(*exc)

warnings.filterwarnings("ignore")

# --------------------- basic setup ---------------------
INTERACTIVE = True
if not INTERACTIVE:
    matplotlib.use("Agg")

STYLE = Path("/Volumes/Songyi Deng/songyidengx/Downloads/neatplot-main/standard.mplstyle")
if STYLE.exists():
    plt.style.use(STYLE)

print(sys.executable); print(platform.python_version())

# --------------------- helpers (geometry, areas, normals) ---------------------
# NOTE: _vertex_array now just returns the current global xyz, we no longer use ddgclib.Complex
def _vertex_array(HC_unused=None):
    """
    Kept for API compatibility. Returns the current global vertex array.
    """
    global xyz
    return xyz, None

def _surface_area_PL(xyz, faces):
    A = 0.0
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        A += 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    return float(A)

def _per_triangle_area(xyz, faces):
    areas = []
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        areas.append(0.5 * np.linalg.norm(np.cross(b - a, c - a)))
    return areas

def _orient_faces_outward(xyz, faces, r0=None):
    if r0 is None:
        r0 = np.mean(xyz, axis=0)
    out = []
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        n = np.cross(b - a, c - a)
        ctr = (a + b + c) / 3.0
        out.append((i, k, j) if np.dot(n, ctr - r0) < 0 else (i, j, k))
    return np.asarray(out, int)

def _enclosed_volume_PL(xyz, faces, r0=None, signed=False):
    if r0 is None:
        r0 = np.mean(xyz, axis=0)
    V = 0.0
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        v = np.dot(a - r0, np.cross(b - r0, c - r0)) / 6.0
        V += v
    return float(V if signed else abs(V))

def _vertex_normals_dual_areas_outward(xyz, faces):
    N = np.zeros_like(xyz)
    Astar = np.zeros(len(xyz))
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        n = np.cross(b - a, c - a)
        Atri = 0.5 * np.linalg.norm(n)
        if Atri == 0:
            continue
        for s in (i, j, k):
            N[s] += n
            Astar[s] += Atri / 3.0
    nn = np.linalg.norm(N, axis=1); nn[nn==0.0] = 1.0
    N = (N.T / nn).T
    c = np.mean(xyz, axis=0)
    dots = np.sum(N * (xyz - c), axis=1)
    N[dots < 0] *= -1.0
    return N, Astar

def _build_cube_surface_faces(xyz, tol=1e-12):
    # kept in case you ever want it; not used in Gmsh version
    idx = np.arange(len(xyz)); faces = []; ctr = np.mean(xyz, axis=0)
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    xtol = max(tol, 1e-6 * max(1.0, abs(xmax - xmin)))
    ytol = max(tol, 1e-6 * max(1.0, abs(ymax - ymin)))
    ztol = max(tol, 1e-6 * max(1.0, abs(zmax - zmin)))
    def add(ids, uv2):
        if len(ids) < 3: return
        tri = Delaunay(uv2)
        g = ids[tri.simplices]
        g = _orient_faces_outward(xyz, g, r0=ctr)
        faces.extend(map(tuple, g))
    m = np.isclose(x, xmin, atol=xtol); add(idx[m], np.c_[y[m], z[m]])
    p = np.isclose(x, xmax, atol=xtol); add(idx[p], np.c_[y[p], z[p]])
    m = np.isclose(y, ymin, atol=ytol); add(idx[m], np.c_[x[m], z[m]])
    p = np.isclose(y, ymax, atol=ytol); add(idx[p], np.c_[x[p], z[p]])
    m = np.isclose(z, zmin, atol=ztol); add(idx[m], np.c_[x[m], y[m]])
    p = np.isclose(z, zmax, atol=ztol); add(idx[p], np.c_[x[p], y[p]])
    return np.asarray(faces, int)

def _min_edge_length(xyz, faces):
    m = np.inf
    for (i,j,k) in faces:
        for a,b in ((i,j),(j,k),(k,i)):
            m = min(m, np.linalg.norm(xyz[a]-xyz[b]))
    return m if np.isfinite(m) else 0.0

# --------------------- cotangent stuff (LB weights & curvature) ---------------------
def _per_triangle_cots(a, b, c):
    ba, ca = b - a, c - a
    cb, ab = c - b, a - b
    ac, bc = a - c, b - c
    n2a = np.linalg.norm(np.cross(ba, ca))
    n2b = np.linalg.norm(np.cross(cb, ab))
    n2c = np.linalg.norm(np.cross(ac, bc))
    cotA = np.dot(ba, ca) / max(n2a, 1e-18)
    cotB = np.dot(cb, ab) / max(n2b, 1e-18)
    cotC = np.dot(ac, bc) / max(n2c, 1e-18)
    return cotA, cotB, cotC

def cotangent_weights(xyz, faces):
    W = defaultdict(float)
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        cotA, cotB, cotC = _per_triangle_cots(a, b, c)
        W[(j, k)] += 0.5 * cotA; W[(k, j)] += 0.5 * cotA
        W[(k, i)] += 0.5 * cotB; W[(i, k)] += 0.5 * cotB
        W[(i, j)] += 0.5 * cotC; W[(j, i)] += 0.5 * cotC
    return W

def mean_curvature_scalar(xyz, faces):
    N, Astar = _vertex_normals_dual_areas_outward(xyz, faces)
    W = cotangent_weights(xyz, faces)
    Lx = np.zeros_like(xyz)
    for (i, j), w in W.items():
        Lx[i] += w * (xyz[j] - xyz[i])
    HN = (Lx.T / (2 * Astar)).T
    Hn = np.sum(HN * N, axis=1)
    return Hn, N, Astar, W, Lx   # also return Lx

# --------------------- Heron-based curvature vectors (HNdA_i) ---------------------
def heron_mean_curvature_vectors(points, faces, neighbors, vertex_faces):
    """
    Heron-based discrete mean curvature:
      - Uses HNdC_ijk from _curvatures_heron.py
      - Returns per-vertex HNdA_i (integrated mean-curvature vector) and
        A_dual (barycentric dual area = sum area/3).
    """
    n = len(points)
    H_vecs = np.zeros((n, 3))
    A_dual = np.zeros(n)

    # Barycentric dual areas from flat triangle areas
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = points[i], points[j], points[k]
        area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        A_dual[i] += area / 3.0
        A_dual[j] += area / 3.0
        A_dual[k] += area / 3.0

    # Integrated curvature vectors HNdA_i
    for i in range(n):
        HNdA_i = np.zeros(3)
        for j in neighbors[i]:
            # faces that contain both i and j
            shared_faces = [idx for idx in vertex_faces[i] if j in faces[idx]]
            for face_idx in shared_faces:
                tri = faces[face_idx]
                if i in tri and j in tri:
                    k = [v for v in tri if v != i and v != j][0]
                    vi, vj, vk = points[i], points[j], points[k]
                    e_ij = vj - vi
                    e_ik = vk - vi
                    e_jk = vk - vj
                    l_ij = np.linalg.norm(e_ij)
                    l_ik = np.linalg.norm(e_ik)
                    l_jk = np.linalg.norm(e_jk)
                    hnda_ijk, c_ijk = HNdC_ijk(e_ij, l_ij, l_jk, l_ik)
                    HNdA_i += hnda_ijk
        H_vecs[i] = HNdA_i

    return H_vecs, A_dual

# --------------------- control volumes via O + surface triangles ---------------------
def _barycentric_dual_volumes_from_surface_with_O(xyz, faces):
    nV = xyz.shape[0]
    Vi = np.zeros(nV, dtype=float)
    O = xyz.mean(axis=0)
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        V_tet = abs(np.dot(a - O, np.cross(b - O, c - O))) / 6.0
        share = 1/3*1.5 * V_tet
        Vi[i] += share; Vi[j] += share; Vi[k] += share
    return Vi

# --------------------- NEW: surface consistent mass matrix --------------------------
MASS_MATRIX = None  # global sparse mass matrix

def _build_surface_consistent_mass_matrix(xyz_init, faces_init, rho0, ref_volume):
    """
    Build a surface consistent mass matrix for a thin shell with thickness h such that
    total mass = rho0 * ref_volume.

    Element (triangle) consistent mass:
        M_e = (rho0 * h * A / 12) * [[2,1,1],[1,2,1],[1,1,2]]
    """
    nV = xyz_init.shape[0]
    M = lil_matrix((nV, nV), dtype=float)

    # shell thickness h such that sum_e rho0 * h * A_e = rho0 * ref_volume
    A_shell = _surface_area_PL(xyz_init, faces_init)
    h_shell = ref_volume / max(A_shell, 1e-18)

    for (i, j, k) in faces_init:
        vi, vj, vk = xyz_init[i], xyz_init[j], xyz_init[k]
        Atri = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        m_e = rho0 * h_shell * Atri
        coeff = m_e / 12.0
        nodes = [i, j, k]
        for r in range(3):
            for s in range(3):
                val = 2.0 if r == s else 1.0
                M[nodes[r], nodes[s]] += coeff * val

    return M.tocsr()

# --------------------- forces (no EOS pressure; only viscous here) ------------------
def F_viscous(mu, W_cotan, U, Astar):
    Lu = np.zeros_like(U)
    for (i, j), w in W_cotan.items():
        Lu[i] += w * (U[j] - U[i])
    DeltaU = (Lu.T / (2.0 * Astar)).T
    return mu * (Astar[:, None] * DeltaU)

def accel_from_forces(F_total, m_i_fixed):
    """
    If USE_CONSISTENT_MASS:
        solve M a = F using CG for each coordinate.
    Else:
        diagonal a_i = F_i / m_i_fixed.
    """
    global MASS_MATRIX
    if USE_CONSISTENT_MASS and (MASS_MATRIX is not None):
        nV = F_total.shape[0]
        a = np.zeros_like(F_total)

        for d in range(3):
            rhs = F_total[:, d]

            # Try old SciPy API first (tol), fall back to new API (rtol)
            try:
                sol, info = cg(MASS_MATRIX, rhs, tol=1e-10, maxiter=300)
            except TypeError:
                sol, info = cg(MASS_MATRIX, rhs, rtol=1e-10, maxiter=300)

            if info != 0:
                pflush(
                    f"[mass-CG] warning: cg did not converge cleanly (info={info}); "
                    f"falling back to diagonal mass for this coordinate."
                )
                m = np.asarray(m_i_fixed, float)
                m_safe = np.where(m > 1e-30, m, 1e-30)
                return (F_total.T / m_safe).T

            a[:, d] = sol

        return a
    else:
        m = np.asarray(m_i_fixed, float)
        m_safe = np.where(m > 1e-30, m, 1e-30)
        return (F_total.T / m_safe).T

# --------------------- curved-volume integration loader ---------------------
class _suppress_output:
    # intentionally left in original (broken) form to avoid any extra overhead
    def __init__(self, enabled=True):
        self.enabled = True
        self._out = io.StringIO()
        self._err = io.StringIO()
    def __enter__(self):
        if self.enabled:
            self._old_out, self._old_err = self._old_out, self._stderr
            sys.stdout, self._stderr = self._out, self._err
    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._old_out, self._old_err = self._old_out, self._old_err

def _run_curved_volume_in_subprocess(xyz_np, faces_np, workdir: Path, tol: float, cv_path: Path, mesh_path: Path):
    class _Result:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    global _CURVED_VOL_MOD
    try:
        if _CURVED_VOL_MOD is None:
            spec = importlib.util.spec_from_file_location("_curved_volume_file", str(cv_path))
            mod = module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            _CURVED_VOL_MOD = mod
        else:
            mod = _CURVED_VOL_MOD

        curved = getattr(mod, "curved_volume", None)
        if not callable(curved):
            return _Result(returncode=1, stderr="curved_volume() not found in _curved_volume.py")
        
        orig_TD = tempfile.TemporaryDirectory
        tempfile.TemporaryDirectory = _CVTempDir
        try:
            curved(
                (np.asarray(xyz_np, float), np.asarray(faces_np, int)),
                workdir=workdir,
                coeffs_kwargs={"plane_rel_tol": float(tol)},
                enforce_split=True,
                complex_dtype="vf",
                msh_path=str(mesh_path.resolve())
            )
        finally:
            tempfile.TemporaryDirectory = orig_TD

        return _Result(returncode=0, stdout="", stderr="")
    except Exception as e:
        return _Result(returncode=1, stdout="", stderr=str(e))

_CURVED_VOL_MOD = None
class _CVTempDir(tempfile.TemporaryDirectory):
    def __init__(self, *a, **kw):
        kw.setdefault("dir", "/tmp")
        super().__init__(*a, **kw)
    def cleanup(self):
        pass

def _load_good_tids_from_coeffs(workdir: Path, mesh_stem: str) -> set[int] | None:
    coeffs_path = workdir / f"{mesh_stem}_COEFFS.csv"
    if not coeffs_path.exists():
        return None
    good: set[int] = set()
    with open(coeffs_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return None
        try:
            j_tid = header.index("triangle_id")
            j_res = header.index("Max_Residual_ABC")
            j_thr = header.index("Residual_Threshold")
        except ValueError:
            return None
        for row in reader:
            if j_tid >= len(row):
                continue
            try:
                tid = int(float(row[j_tid]))
            except Exception:
                continue
            if j_res < len(row) and j_thr < len(row):
                try:
                    res_val = float(row[j_res])
                    thr_val = float(row[j_thr])
                except Exception:
                    continue
                if res_val < thr_val:
                    good.add(tid)
    return good if good else set()

def _load_curved_volume():
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / "_curved_volume.py")
    candidates.append(Path.cwd() / "_curved_volume.py")
    candidates.append(Path("/mnt/data/_curved_volume.py"))
    last_err = None
    for p in candidates:
        try:
            if p.exists():
                spec = importlib.util.spec_from_file_location("_curved_volume_file", str(p))
                mod = module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                if hasattr(mod, "curved_volume"):
                    return mod.curved_volume
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Could not locate a working _curved_volume.py ({last_err})")

def _find_transformed_volume_csv(workdir: Path, mesh_stem: str) -> Path | None:
    cand = workdir / f"{mesh_stem}_COEFFS_Transformed_Volume.csv"
    if cand.exists():
        return cand
    for p in workdir.glob("*_COEFFS_Transformed_Volume.csv"):
        return p
    return None

def _sum_vcorrection_from_csv(csv_path: Path, good_tids: set[int] | None = None) -> float:
    global _CV_LAST_GOOD_TIDS
    _CV_LAST_GOOD_TIDS = set()
    if not csv_path or not csv_path.exists():
        return float("nan")
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f); header = next(reader, None)
        if not header:
            return float("nan")
        try:
            j_vcorr = header.index("Vcorrection")
        except ValueError:
            return float("nan")
        try:
            j_tid = header.index("triangle_id")
        except ValueError:
            j_tid = None

        s = 0.0
        for row in reader:
            if j_vcorr >= len(row) or row[j_vcorr] == "":
                continue
            tid_val = None
            if j_tid is not None and j_tid < len(row):
                try:
                    tid_val = int(float(row[j_tid]))
                except Exception:
                    tid_val = None
            if good_tids is not None:
                if tid_val is None or tid_val not in good_tids:
                    continue
            try:
                vc = float(row[j_vcorr])
            except Exception:
                vc = 0.0
            s += vc
            if tid_val is not None:
                _CV_LAST_GOOD_TIDS.add(tid_val)
        return float(s)

def _read_Acurved_map(csv_path: Path, good_tids: set[int] | None = None) -> dict[int, float] | None:
    global _CV_LAST_GOOD_TIDS
    if not csv_path or not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f); header = next(reader, None)
            if not header:
                return None
            j_tid = header.index("triangle_id")
            j_Ac  = header.index("A_curved")
            out = {}
            for row in reader:
                tid = int(float(row[j_tid]))
                if good_tids is not None and tid not in good_tids:
                    continue
                Ac  = float(row[j_Ac]) if row[j_Ac] != "" else np.nan
                if np.isfinite(Ac):
                    out[tid] = Ac
                    _CV_LAST_GOOD_TIDS.add(tid)
            return out
    except Exception:
        return None

# --- NEW: point-wise DualVolume + DualArea (from v44, adapted to tmpdir workdir) ---
def _find_dualvolume_csv(workdir: Path) -> Path | None:
    cand = workdir / "_COEFFS_Transformed_DualVolume.csv"
    if cand.exists():
        return cand
    for p in workdir.glob("*_COEFFS_Transformed_DualVolume.csv"):
        return p
    return None

def _load_point_dualvolume_map(workdir: Path) -> dict[int, tuple[float, float]]:
    """
    Expect CSV like:
    PointID,DualVolume,DualArea,Type
    0,-6.58e-15,0.0,rotation
    1, 9.65e-16,0.0,rotation
    -> return {pid: (DualVolume, DualArea)}
    """
    path = _find_dualvolume_csv(workdir)
    out: dict[int, tuple[float, float]] = {}
    if not path:
        return out
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return out
        try:
            j_pid = header.index("PointID")
            j_dv  = header.index("DualVolume")
            j_da  = header.index("DualArea")
        except ValueError:
            return out
        for row in r:
            if j_pid >= len(row) or j_dv >= len(row) or j_da >= len(row):
                continue
            try:
                pid = int(float(row[j_pid]))
                dv  = float(row[j_dv])
                da  = float(row[j_da])
            except Exception:
                continue
            out[pid] = (dv, da)
    return out

def _curved_volume_and_sumAcurved_with_flat_fallback(
    xyz,
    faces,
    plane_rel_tol: float = PLANE_REL_TOL,
    step_idx: int = 0,
    mesh_path: Path | None = None
):
    if mesh_path is None:
        raise RuntimeError("mesh_path is required so _curved_volume.py uses the on-disk .msh")

    A_flat_list = _per_triangle_area(np.asarray(xyz, float), np.asarray(faces, int))
    sum_A_flat  = float(np.sum(A_flat_list))

    cv_path = (Path(__file__).resolve().parent / "_curved_volume.py").resolve()
    mesh_stem = mesh_path.stem

    workdir = OUTPUT_DIR
    tol = float(plane_rel_tol)
    pflush(f"[CV] start tol={tol:g}  tmp={workdir}")
    try:
        proc = _run_curved_volume_in_subprocess(xyz, faces, workdir, tol, cv_path, mesh_path)
        if proc.returncode != 0:
            last_err = RuntimeError(f"_curved_volume rc={proc.returncode}\nSTDERR:\n{proc.stderr}")
            pflush(f"[CV] failed tol={tol:g}: rc={proc.returncode}")
            if STRICT_CV:
                raise RuntimeError(f"[STRICT_CV] _curved_volume.py produced no usable CSV (last_err={last_err}).")
            pflush(f"[warn] CSV missing/invalid (last_err={last_err}). Using A_flat-only fallback.")
            return 0.0, sum_A_flat, sum_A_flat, None
    except Exception as e:
        pflush(f"[CV] exception tol={tol:g}: {e}")
        if STRICT_CV:
            raise RuntimeError(f"[STRICT_CV] _curved_volume.py produced no usable CSV (last_err={e}).")
        pflush(f"[warn] CSV missing/invalid (last_err={e}). Using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None

    good_tids = _load_good_tids_from_coeffs(workdir, mesh_stem)

    volcsv = _find_transformed_volume_csv(workdir, mesh_stem)
    if (not volcsv) or (not volcsv.exists()) or (volcsv.stat().st_size == 0):
        last_err = RuntimeError(f"_curved_volume: missing/empty _COEFFS_Transformed_Volume.csv in {workdir}")
        pflush(f"[CV] invalid/missing Transformed_Volume at tol={tol:g}")
        if STRICT_CV:
            raise RuntimeError(f"[STRICT_CV] _curved_volume.py produced no usable CSV (last_err={last_err}).")
        pflush(f"[warn] CSV missing/invalid (last_err={last_err}). Using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None

    Vcorr  = _sum_vcorrection_from_csv(volcsv, good_tids=good_tids)
    Amap   = _read_Acurved_map(volcsv, good_tids=good_tids)
    if Amap is None:
        sumA = sum_A_flat
    else:
        sumA = 0.0
        for tid, Aflat in enumerate(A_flat_list):
            Ac = Amap.get(tid, np.nan)
            sumA += (max(Ac, Aflat) if np.isfinite(Ac) else Aflat)

    if np.isfinite(Vcorr) and np.isfinite(sumA):
        pflush(f"[CV] done  tol={tol:g}  tmp={workdir}")
        return float(Vcorr), float(sumA), float(sum_A_flat), workdir
    else:
        last_err = RuntimeError("CSV parsed but missing finite Vcorr/sumA")
        pflush(f"[CV] invalid CSV tol={tol:g}")
        if STRICT_CV:
            raise RuntimeError(f"[STRICT_CV] _curved_volume.py produced no usable CSV (last_err={last_err}).")
        pflush(f"[warn] CSV missing/invalid (last_err={last_err}). Using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None

# --------------------- ALWAYS-WRITTEN mesh/topology CSVs ---------------------
def _vertex_adjacency_list(faces, nV):
    nbr = [set() for _ in range(nV)]
    for (i, j, k) in faces:
        nbr[i].update((j, k)); nbr[j].update((i, k)); nbr[k].update((i, j))
    return [sorted(s) for s in nbr]

def write_mesh_topology_csvs(xyz, faces, out_dir: Path, iter_k: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    nV = len(xyz)
    with open(out_dir / f"iter_{iter_k:04d}_mesh_POINT_TYPES.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["vertex_id","type"])
        for i in range(nV): w.writerow([i, 1])
    rings = _vertex_adjacency_list(faces, nV)
    with open(out_dir / f"iter_{iter_k:04d}_mesh_RINGS.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["center_id","ring"])
        for i, nbrs in enumerate(rings): w.writerow([i, " ".join(map(str, nbrs))])
    with open(out_dir / f"iter_{iter_k:04d}_mesh_Tri_TYPE.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["triangle_id","type"])
        for tid, _ in enumerate(faces): w.writerow([tid, 1])

# --------------------- initial conditions & OUTPUT DIR ---------------------
try:
    script_path = Path(__file__).resolve()
    SCRIPT_STEM = script_path.stem          # e.g. "_AR1.1_fix_mi_fine_curved"
    script_dir  = script_path.parent
except NameError:
    # Fallback for interactive runs
    script_dir  = Path.cwd()
    SCRIPT_STEM = "_AR1.1_fix_mi_fine_curved"

OUTPUT_DIR = script_dir / SCRIPT_STEM
OUTPUT_DIR.mkdir(exist_ok=True)

# -------- Gmsh mesh generator (ellipsoid AR_INIT, R_EQ) --------
def _generate_gmsh_ellipsoid_mesh(path: Path, AR: float, R_eq: float):
    if not HAVE_GMSH:
        raise RuntimeError(
            f"Requested auto-generation of {path}, but gmsh python module is not available.\n"
            f"Install with: pip install gmsh"
        )
    pflush(f"[GMSH] Generating ellipsoid mesh at {path} (AR={AR}, R_eq={R_eq})")

    gmsh.initialize()
    gmsh.model.add("ellipsoid_AR")

    # Semi-axes: a_z = AR * b, b_x = b_y = b, with same volume as sphere of radius R_eq
    b = R_eq / (AR ** (1.0 / 3.0))
    a = AR * b

    # 1. Create a unit sphere centered at origin
    sph_tag = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)

    # 2. Non-uniform scaling: x→b, y→b, z→a
    gmsh.model.occ.dilate([(3, sph_tag)], 0.0, 0.0, 0.0, b, b, a)

    # 3. Sync CAD → model
    gmsh.model.occ.synchronize()

    # 4. Set mesh size (tune lc for coarser/finer)
    lc = 0.20 * R_eq
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    # 5. Generate surface mesh (2D) on ellipsoid boundary
    gmsh.model.mesh.generate(2)

    # 6. Write mesh; meshio will read this
    gmsh.write(str(path))
    gmsh.finalize()

    pflush(f"[GMSH] Done generating mesh: {path}")

# --------------------- ddg mesh ---------------------
# and mesh source selection
xyz0 = None
faces0 = None

if MESH_SOURCE.lower() == "gmsh":
    # ---- Gmsh-based ellipsoid mesh ----
    MESH_IN_PATH = script_dir / "surface_init.msh"
    _generate_gmsh_ellipsoid_mesh(MESH_IN_PATH, AR_INIT, R_EQ)

    mesh_in = meshio.read(MESH_IN_PATH)

    # points
    xyz0 = np.asarray(mesh_in.points[:, :3], float)

    # triangle faces
    tri_cells = None
    for cell_block in mesh_in.cells:
        if cell_block.type in ("triangle", "tri"):
            tri_cells = cell_block.data
            break
    if tri_cells is None:
        raise RuntimeError("No triangle cells found in Gmsh mesh for droplet surface.")

    faces0 = np.asarray(tri_cells, dtype=int)

elif MESH_SOURCE.lower() == "hc":
    # ---- ddgclib-based cube-surface mesh (then snapped to ellipsoid) ----
    try:
        from ddgclib import *
        from hyperct import Complex
    except ImportError as e:
        raise RuntimeError(
            "MESH_SOURCE='hc' requires ddgclib to be installed. "
            "Install it or set MESH_SOURCE='gmsh'."
        ) from e

    pflush("[DDG] Building ddg cube-surface mesh (HC)")

    HC = Complex(3)
    HC.triangulate()
    for _ in range(3):
        HC.refine_all()

    # keep only boundary vertices (x,y,z in {0,1})
    interior = []
    for v in HC.V:
        p = np.asarray(v.x_a, float)
        if np.any(np.isclose(p, 0.0)) or np.any(np.isclose(p, 1.0)):
            continue
        interior.append(v)
    for v in interior:
        HC.V.remove(v)

    verts = list(HC.V)
    xyz0 = np.array([np.asarray(v.x_a, float) for v in verts])

    # center and scale to a ~1 mm cube
    c0_tmp = np.mean(xyz0, axis=0)
    xyz0 = (xyz0 - c0_tmp) * 1.0e-3

    # build cube-surface triangulation
    faces0 = _build_cube_surface_faces(xyz0)

    pflush(f"[DDG] ddg mesh ready: nV={xyz0.shape[0]}, nF={faces0.shape[0]}")

else:
    raise ValueError(f"Unknown MESH_SOURCE={MESH_SOURCE!r} (expected 'gmsh' or 'hc')")

# center mesh at origin (again; keeps behavior consistent)
c0 = np.mean(xyz0, axis=0)
xyz0 = xyz0 - c0

# ---- SNAP xyz0 onto ideal ellipsoid (AR_INIT, R_EQ) ----
# Ellipsoid: (x^2 + y^2)/b^2 + z^2/a^2 = 1,
# with a_z = AR_INIT * b, and volume equal to sphere of radius R_EQ.
b_snap = R_EQ / (AR_INIT ** (1.0 / 3.0))
a_snap = AR_INIT * b_snap

x = xyz0[:, 0]
y = xyz0[:, 1]
z = xyz0[:, 2]

denom = (x**2 + y**2) / (b_snap**2) + (z**2) / (a_snap**2)
scale = np.ones_like(denom)
mask = denom > 0.0
scale[mask] = 1.0 / np.sqrt(denom[mask])

xyz0[:, 0] *= scale
xyz0[:, 1] *= scale
xyz0[:, 2] *= scale

# global current coordinates (will be updated during the run)
xyz = xyz0.copy()

# export initial mesh as surface_iter_0000.msh (for curved volume etc.)
mesh0 = meshio.Mesh(points=xyz0, cells=[("triangle", faces0.astype(np.int32))])
mesh0_path = (OUTPUT_DIR / "surface_iter_0000.msh").resolve()
with silence_all():
    plt.close('all')
    meshio.write(mesh0_path, mesh0, file_format="gmsh22", binary=False)

# ---- NEW: build Vi_true0 = Vi_PL0 + DualVolume_vec0 at t=0 for volumes ----
Vi_PL0 = _barycentric_dual_volumes_from_surface_with_O(xyz0, faces0)

if USE_CURVED_VOLUME:
    Vcorr0, sum_A_curved0, sum_A_flat0, workdir0 = _curved_volume_and_sumAcurved_with_flat_fallback(
        xyz0, faces0, plane_rel_tol=PLANE_REL_TOL, step_idx=0, mesh_path=mesh0_path
    )
    DualVolume_vec0 = np.zeros_like(Vi_PL0)
    if workdir0 is not None:
        dv_map0 = _load_point_dualvolume_map(workdir0)
        if dv_map0:
            for vid, (dv, _da) in dv_map0.items():
                if 0 <= vid < len(DualVolume_vec0):
                    DualVolume_vec0[vid] = dv
else:
    # Pure PL: no curved-volume call, no CSV usage
    Vcorr0 = 0.0
    sum_A_flat0 = _surface_area_PL(xyz0, faces0)
    sum_A_curved0 = sum_A_flat0
    workdir0 = None
    DualVolume_vec0 = np.zeros_like(Vi_PL0)

Vi_true0 = Vi_PL0 + DualVolume_vec0

# -------------- reference volume & area (constants) --------------
# Same ellipsoid as used for mesh: a_z = AR_INIT * b, b_x = b_y = b
b_ref = R_EQ / (AR_INIT ** (1.0 / 3.0))
a_ref = AR_INIT * b_ref

REF_VOLUME = (4.0 / 3.0) * np.pi * a_ref * (b_ref ** 2)
# Reference area: sphere of radius R_EQ (Case II) – full surface area
REF_AREA   = 4.0 * np.pi * R_EQ**2

# ---- target area for a sphere with the same REF_VOLUME ----
def _sphere_area_from_volume(V):
    # r = (3V / 4π)^(1/3),  A = 4π r^2 = 4π * (3V / 4π)^(2/3)
    return 4.0 * np.pi * ((3.0 * V / (4.0 * np.pi)) ** (2.0 / 3.0))
A_TARGET = float(_sphere_area_from_volume(REF_VOLUME))

# ---- Rayleigh frequency and reference cross-sectional area (for dimensionless plots) ----
# Rayleigh angular frequency: ω_R^2 = 8 σ / (ρ R^3),  f_R = ω_R / (2π)
F_RAYL = (1.0 / (2.0 * np.pi)) * np.sqrt(8.0 * SIGMA / (RHO0 * R_EQ**3))  # Hz
# Cross-sectional reference area: unperturbed sphere, A_ref = π R^2
A_REF_CROSS = np.pi * R_EQ**2

# ---- FIXED per-vertex masses / mass matrix ----
MASS_I_VEC = None
MASS_TOTAL = None

if USE_CONSISTENT_MASS:
    pflush("[MASS] Building surface consistent mass matrix...")
    MASS_MATRIX = _build_surface_consistent_mass_matrix(xyz0, faces0, RHO0, REF_VOLUME)
    MASS_I_VEC = np.asarray(MASS_MATRIX.sum(axis=1)).ravel()  # per-vertex "lumped" mass (row sums)
    MASS_TOTAL = float(np.sum(MASS_I_VEC))
    pflush(f"[MASS] Consistent mass built. total mass={MASS_TOTAL:.6e} kg")
else:
    MASS_MATRIX = None
    VI_REF       = Vi_true0
    MASS_I_VEC   = RHO0 * VI_REF  # initial masses (may be updated if UPDATE_MASS_EACH_STEP=True)
    MASS_TOTAL   = float(np.sum(MASS_I_VEC))

# ---- storage for Ai(t=0) if we freeze Ai later ----
Astar_safe0 = None

# --------------------- HARD GUARD: post-step min-edge projection (tangential) ----
def _enforce_min_edge_length_tangent(xyz_local, faces, alpha=0.4, *,
                                     pole_only=False, pole_frac=0.85, axis=2):
    """
    Enforce L_ij >= L_thr on unique edges by a symmetric, tangential projection.
    L_thr = alpha * median(edge lengths).

    Optional scoping to polar caps for prolate/oblate control:
      - pole_only : if True, act only on edges whose endpoints both lie inside the caps
      - pole_frac : cap half-height as a fraction of max |coordinate along `axis`|
      - axis      : 0->x, 1->y, 2->z (use 2 for a z-major prolate)
    """
    if xyz_local.size == 0:
        return xyz_local

    # normals (for tangential projection)
    N, _A = _vertex_normals_dual_areas_outward(xyz_local, faces)
    I = np.eye(3)

    # Optional polar-cap mask
    if pole_only:
        coord = xyz_local[:, axis]
        cap_threshold = float(np.max(np.abs(coord))) * float(pole_frac)
        in_cap = np.abs(coord) >= cap_threshold
    else:
        in_cap = None  # no masking

    # Gather unique edges (optionally restricted to polar caps)
    edge_set = set()
    for (i, j, k) in faces:
        for a, b in ((i, j), (j, k), (k, i)):
            if a > b:
                a, b = b, a
            if pole_only and not (in_cap[a] and in_cap[b]):
                continue
            edge_set.add((a, b))

    if not edge_set:
        return xyz_local

    # Edge lengths (based on the selected set)
    lengths = []
    for (i, j) in edge_set:
        L = float(np.linalg.norm(xyz_local[j] - xyz_local[i]))
        if L > 0:
            lengths.append(L)
    if not lengths:
        return xyz_local

    L_med = float(np.median(lengths))
    L_thr = alpha * L_med

    X = xyz_local.copy()
    for (i, j) in edge_set:
        e = X[j] - X[i]
        L = float(np.linalg.norm(e))
        if L <= 1e-16 or L >= L_thr:
            continue
        t = e / L
        d = 0.5 * (L_thr - L) * t  # symmetric split

        # Tangential projection at each endpoint
        ni = N[i]; Pi = I - np.outer(ni, ni)
        nj = N[j]; Pj = I - np.outer(nj, nj)
        X[i] = X[i] - Pi @ d
        X[j] = X[j] + Pj @ d

    return X
# ------------------------------------------------------------------------------

# --------------------- ONE STEP (pressure projection + curved Ai/Vi) ---------
_prev_U = np.zeros_like(xyz0)  # previous-step velocity field

def step_forces_dualvolume(HC_unused, faces, step_idx: int):
    global xyz, _prev_U, MASS_I_VEC, MASS_TOTAL, Astar_safe0

    # current geometry from global xyz
    xyz_curr, _ = _vertex_array(None)
    xyz_loc = xyz_curr.copy()

    Aprev_PL = _surface_area_PL(xyz_loc, faces)
    V_tet    = _enclosed_volume_PL(xyz_loc, faces)

    # write current surface mesh for this iteration
    mesh_iter_path = (OUTPUT_DIR / f"surface_iter_{step_idx:04d}.msh").resolve()
    mesh_iter = meshio.Mesh(points=xyz_loc, cells=[("triangle", np.asarray(faces, np.int32))])
    with silence_all():
        meshio.write(mesh_iter_path, mesh_iter, file_format="gmsh22", binary=False)

    # per-triangle flat areas (same ordering as 'faces')
    A_flat_list_local = _per_triangle_area(xyz_loc, faces)

    # --- curved-volume run + access to A_curved & DualVolume via tmpdir (optional) ---
    if USE_CURVED_VOLUME:
        V_corr, sum_A_curved, sum_A_flat, tmpdir = _curved_volume_and_sumAcurved_with_flat_fallback(
            xyz_loc, faces, plane_rel_tol=PLANE_REL_TOL, step_idx=step_idx, mesh_path=mesh_iter_path
        )
    else:
        V_corr = 0.0
        sum_A_flat = float(np.sum(A_flat_list_local))
        sum_A_curved = sum_A_flat
        tmpdir = None
    
    # --- Ai from v44: per-vertex Aflat_pt and Acurved_pt -------------------
    nV = xyz_loc.shape[0]
    Aflat_pt  = np.zeros(nV, dtype=float)
    Ac_pt_raw = np.zeros(nV, dtype=float)

    mesh_stem = mesh_iter_path.stem
    workdir = OUTPUT_DIR
    Amap_tri = None
    if tmpdir is not None:
        volcsv = _find_transformed_volume_csv(workdir, mesh_stem)
        if volcsv:
            Amap_tri = _read_Acurved_map(volcsv)

    for tid, (i, j, k) in enumerate(faces):
        Atri_flat = float(A_flat_list_local[tid])
        if Amap_tri is not None:
            val = Amap_tri.get(tid, np.nan)
            Atri_curv = float(val) if np.isfinite(val) else Atri_flat
        else:
            Atri_curv = Atri_flat

        share_flat = Atri_flat / 3.0
        share_curv = Atri_curv / 3.0

        for idx in (i, j, k):
            Aflat_pt[idx]  += share_flat
            Ac_pt_raw[idx] += share_curv

    Ac_pt = np.maximum(Ac_pt_raw, Aflat_pt)

    # curvature etc. from cotan-LB (for normals, viscous operator, etc.)
    Hn, N, Astar_flat_cotan, W_cotan, Lx = mean_curvature_scalar(xyz_loc, faces)

    # A_dual (Ai) used everywhere else in this script
    Astar_safe = np.maximum(Aflat_pt, Ac_pt)

    # store Ai at t=0 if we want to potentially freeze it
    if step_idx == 0:
        Astar_safe0 = Astar_safe.copy()
    elif FIX_AI_AT_T0 and (Astar_safe0 is not None):
        # use frozen Ai from t=0
        Astar_safe = Astar_safe0.copy()

    Astar_safe_total = float(np.sum(Astar_safe))

    # --- Vi from v44: barycentric + DualVolume_vec from CSV (if present) ---
    Vi_true_base = _barycentric_dual_volumes_from_surface_with_O(xyz_loc, faces)
    DualVolume_vec = np.zeros_like(Vi_true_base)

    if tmpdir is not None:
        dv_map = _load_point_dualvolume_map(tmpdir)
        if dv_map:
            for vid, (dv, _da) in dv_map.items():
                if 0 <= vid < nV:
                    DualVolume_vec[vid] = dv

    Vi_true_plus  = Vi_true_base + DualVolume_vec
    Vi_true       = Vi_true_plus.copy()
    Vi_true_plus_total = float(np.sum(Vi_true_plus))

    # If requested, freeze Vi to t=0 (only affects geometry/diagnostics; mass matrix stays fixed)
    if FIX_VI_AT_T0:
        Vi_true = Vi_true0.copy()
        Vi_true_plus_total = float(np.sum(Vi_true0))

    # define global V_total as Vtrue+total(k)
    V_center_O = V_tet - Vi_true_plus_total
    V_total    = Vi_true_plus_total + V_center_O

    # --- optional dynamic masses: m_i = rho0 * Vi_true_plus(k) each step ---
    if UPDATE_MASS_EACH_STEP and (not USE_CONSISTENT_MASS):
        if FIX_VI_AT_T0:
            MASS_I_VEC = RHO0 * Vi_true0
        else:
            MASS_I_VEC = RHO0 * Vi_true_plus
        MASS_TOTAL = float(np.sum(MASS_I_VEC))

    # --------- GLOBAL VOLUME-PROJECTION PRESSURE (no EOS) ---------

    # choose surface-tension force method
    method = FS_METHOD.lower()

    # we will store a scalar mean curvature per vertex for CSV output
    H_scalar = np.zeros(nV, dtype=float)

    if method == "heron":
        # Heron-based curvature vectors (HNdA_i) for surface tension Fs
        neighbors = _vertex_adjacency_list(faces, nV)
        vertex_faces = [[] for _ in range(nV)]
        for fid, (i, j, k) in enumerate(faces):
            vertex_faces[i].append(fid)
            vertex_faces[j].append(fid)
            vertex_faces[k].append(fid)
        H_vecs_heron, A_dual_heron = heron_mean_curvature_vectors(
            xyz_loc, faces, neighbors, vertex_faces
        )
        Fs = SIGMA * H_vecs_heron

        # scalar mean curvature from Heron: H ≈ (HNdA · n) / A_dual
        epsA = 1e-30
        for i in range(nV):
            denom = A_dual_heron[i] if abs(A_dual_heron[i]) > epsA else epsA
            H_scalar[i] = float(np.dot(H_vecs_heron[i], N[i]) / denom)

    elif method == "cotan":
        # cotan-based integrated mean curvature vector: HNdA_i ≈ Lx
        Fs = SIGMA * Lx

        # scalar mean curvature from cotan-LB (already computed as Hn)
        H_scalar = Hn.copy()

    else:
        raise ValueError(f"Unknown FS_METHOD={FS_METHOD!r} (expected 'heron' or 'cotan')")

    # area-weighted mean capillary pressure proxy: tau_bar = < -σ Hn >
    A_sum = float(np.sum(Astar_safe) + 1e-30)
    tau_bar = float(np.sum((-SIGMA * Hn) * Astar_safe) / A_sum)

    # surface-proxy viscosity using bulk mu * h_eff
    h_eff  = V_total / max(Aprev_PL, 1e-18)
    mu_eff = MU * h_eff

    # viscous with current/prev velocity field
    U_for_visc = _prev_U if step_idx > 0 else np.zeros_like(_prev_U)
    Fv    = F_viscous(mu_eff, W_cotan, U_for_visc, Astar_safe)

    # linear drag (reduced)
    Fdrag = -C_DAMP * (Astar_safe[:, None] * U_for_visc)

    # Non-pressure forces and provisional acceleration
    F_nonp = Fs   # + Fv + Fdrag    # (you can re-enable viscous/drag here if you want)
    a_nonp = accel_from_forces(F_nonp, MASS_I_VEC)

    # Provisional dt from non-pressure dynamics
    U_prov    = _prev_U + DT_CAP * a_nonp
    v_sq_area = np.sum((np.linalg.norm(U_prov, axis=1) ** 2) * Astar_safe) + 1e-18
    A_drop    = 5e-3 * Aprev_PL
    dt        = min(DT_CAP, max(5e-6, A_drop / v_sq_area))

    # geometric CFL using min edge length (relaxed)
    umax = float(np.max(np.linalg.norm(U_prov, axis=1)) + 1e-30)
    L_min = _min_edge_length(xyz_loc, faces)
    if L_min > 0 and umax > 0:
        dt = min(dt, CFL_EDGE_FRAC * L_min / umax)

    # pressure/force substeps
    nsub   = max(1, int(N_SUBSTEPS_PRESSURE))
    sub_dt = dt / nsub

    # Predicted volume rate from non-pressure motion
    U_pred   = _prev_U + sub_dt * a_nonp
    dVdt_pred = float(np.sum(Astar_safe * np.sum(U_pred * N, axis=1)))

    # Sensitivity of dV/dt to a uniform pressure
    C = float(np.sum((Astar_safe**2) / np.maximum(MASS_I_VEC, 1e-30)))

    # Pressure to enforce dV/dt_new = 0 (global mean)
    p_proj = 0.0 if C == 0.0 else (-dVdt_pred / (sub_dt * C))

    # --- NEW: build a non-uniform pressure pattern on the surface ---
    if USE_PRESSURE_GRADIENT:
        # area-weighted mean curvature
        H_mean = float(np.sum(Hn * Astar_safe) / (A_sum + 1e-30))

        # curvature fluctuation around the mean (area-weighted mean = 0)
        H_fluct = Hn - H_mean

        # curvature-based extra pressure (higher |H| → higher |p|, sign chosen to oppose Fs)
        # units: [Pa] because SIGMA [N/m] * H [1/m]
        p_grad = 2/3 * (-SIGMA * H_fluct) *0

        # total internal pressure per vertex
        p_i = p_proj + p_grad
    else:
        # fallback: purely uniform pressure (old behavior)
        p_i = np.full(len(Astar_safe), p_proj, dtype=float)

    # Pressure force and "book-keeping" fields for logs/CSV
    Fp      = (p_i * Astar_safe)[:, None] * N
    rho_i   = np.full(len(Astar_safe), RHO0)
    P_abs_i = np.full(len(Astar_safe), P0_ATM) + p_i
    dP_i    = p_i

    pflush(f"*⟨P_abs⟩ (Pa): {float(np.mean(P_abs_i)):.5f}  *⟨rho⟩ {float(np.mean(rho_i)):.3f}")

    # ---- area-weighted means for summary (after pressure) ----
    P_inside_mean = float(np.sum(P_abs_i * Astar_safe) / A_sum)
    dP_mean       = float(np.sum(dP_i * Astar_safe) / A_sum)
    surface_tension_mean = tau_bar  # ⟨-σ Hn⟩

    # total force with projected pressure + gradient part
    F_total = Fp + F_nonp

    # acceleration (fixed or dynamic masses)
    a_vec = accel_from_forces(F_total, MASS_I_VEC)

    # pressure/force substeps using the finalized a_vec
    U = _prev_U.copy()
    for _ in range(nsub):
        U = U + sub_dt * a_vec

    # --- NEW: remove center-of-mass (COM) velocity to pin droplet ---
    if REMOVE_MEAN_DP:
        m = np.asarray(MASS_I_VEC, float)
        M_tot = float(np.sum(m) + 1e-30)
        v_com = np.sum(m[:, None] * U, axis=0) / M_tot
        U = U - v_com
    # ---------------------------------------------------------------

    # AB2 displacement (Euler on first step)
    if step_idx == 0:
        disp = dt * U
    else:
        disp = dt * (1.5 * U - 0.5 * _prev_U)

    # move vertices: update global xyz
    xyz2 = xyz_loc + disp
    xyz = xyz2
    _prev_U = U.copy()

    # ---------- HARD GUARD (post-move): project short edges to L_thr tangentially ----------
    if USE_HARD_GUARD:
        xyz_tmp, _ = _vertex_array(None)
        xyz_proj = _enforce_min_edge_length_tangent(xyz_tmp, faces, alpha=0.4)
        if xyz_proj is not None and xyz_proj.shape == xyz_tmp.shape:
            xyz = xyz_proj
    # --------------------------------------------------------------------------------------

    # --- NEW: recenter positions so COM stays at origin (pinning) ---
    if REMOVE_MEAN_DP:
        m = np.asarray(MASS_I_VEC, float)
        M_tot = float(np.sum(m) + 1e-30)
        x_com = np.sum(m[:, None] * xyz, axis=0) / M_tot
        xyz = xyz - x_com
    # ---------------------------------------------------------------

    # report (post-move, after hard guard and recentering)
    xyz2, _ = _vertex_array(None)
    A_PL = _surface_area_PL(xyz2, faces)
    V_PL = _enclosed_volume_PL(xyz2, faces)

    # === average distance between neighboring points (unique edges) ===
    edge_set = set()
    for (i, j, k) in faces:
        if i < j: edge_set.add((i, j))
        else:     edge_set.add((j, i))
        if j < k: edge_set.add((j, k))
        else:     edge_set.add((k, j))
        if k < i: edge_set.add((k, i))
        else:     edge_set.add((i, k))
    total_edge_len = 0.0
    for (a_e, b_e) in edge_set:
        pa = xyz2[a_e]; pb = xyz2[b_e]
        total_edge_len += float(np.linalg.norm(pb - pa))
    ave_dist_nnpt = total_edge_len / max(len(edge_set), 1)

    # === average distance to origin (for this iteration) ===
    dist_O   = np.linalg.norm(xyz2, axis=1)
    ave_dist_O = float(np.mean(dist_O))

    # --- cross-sectional area via area-weighted RMS semi-axes (RMS definition) ---
    z = xyz2[:, 2]
    r = np.sqrt(xyz2[:, 0]**2 + xyz2[:, 1]**2)

    weights = Astar_safe
    w_sum   = float(np.sum(weights) + 1e-30)

    a_z = float(np.sqrt(np.sum(weights * z**2) / w_sum))     # RMS |z|
    b_r = float(np.sqrt(np.sum(weights * r**2) / w_sum))     # RMS radius

    A_cross = float(np.pi * a_z * b_r)

    # --- NEW: ellipsoid-fit cross-sectional area (volume-consistent) ---
    A_cross_ellip = A_cross  # fallback
    s = r**2
    t_sq = z**2
    w = weights

    S11 = float(np.sum(w * s * s))
    S22 = float(np.sum(w * t_sq * t_sq))
    S12 = float(np.sum(w * s * t_sq))
    b1  = float(np.sum(w * s))
    b2  = float(np.sum(w * t_sq))

    det = S11 * S22 - S12 * S12
    if abs(det) > 1e-50:
        alpha = (b1 * S22 - b2 * S12) / det   # ~ 1/b^2
        beta  = (S11 * b2 - S12 * b1) / det   # ~ 1/a^2
        if alpha > 0.0 and beta > 0.0:
            b_eff = 1.0 / np.sqrt(alpha)
            a_eff = 1.0 / np.sqrt(beta)

            V_fit = (4.0 / 3.0) * np.pi * a_eff * (b_eff ** 2)
            if V_fit > 0.0:
                s_vol = (V_total / V_fit) ** (1.0 / 3.0)
                a_eff *= s_vol
                b_eff *= s_vol

            A_cross_ellip = float(np.pi * a_eff * b_eff)

    Fp_n = np.linalg.norm(Fp, axis=1)
    Fs_n = np.linalg.norm(Fs, axis=1)
    Fv_n = np.linalg.norm(Fv, axis=1)
    Fs_n_along_n = np.sum(Fs * N, axis=1)
    total_Fs_outward = float(np.sum(Fs_n_along_n[Fs_n_along_n > 0.0]))
    total_Fs_inward  = float(-np.sum(Fs_n_along_n[Fs_n_along_n < 0.0]))
    pflush(f"   [Fs only] total outward={total_Fs_outward:.6e}   total inward={total_Fs_inward:.6e}")

    pflush(f"   A_init(PL)={REF_AREA:.6e}  V_init(PL)={REF_VOLUME:.6e}")

    dA_PL_total_rel = (A_PL - REF_AREA) / max(REF_AREA, 1e-300)
    dV_total_rel    = (V_total - REF_VOLUME) / max(REF_VOLUME, 1e-300)
    pflush(f"   [TOTAL] ΔA_PL_total/A0={dA_PL_total_rel:+.6e} ({dA_PL_total_rel*100:+.3f}%)   "
           f"ΔV_total/V0={dV_total_rel:+.6e} ({dV_total_rel*100:+.3f}%)")

    rho_mean        = float(np.mean(rho_i))
    P_mean          = float(np.mean(P_abs_i))
    dP_mean_print   = float(np.mean(dP_i))
    pflush(f"k={step_idx:04d}  A_curved≈{sum_A_curved:.6e}  A_PL→{A_PL:.6e}  "
           f"V_total(prev)={V_total:.6e}  V_PL→{V_PL:.6e}  ⟨ΔP⟩={dP_mean_print:.3e} Pa  "
           f"max|dx|={np.linalg.norm(disp,axis=1).max():.3e}  mean|dx|={np.linalg.norm(disp,axis=1).mean():.3e}  dt={dt:.3e}")
    pflush(f"   Fp: max={Fp_n.max():.3e} mean={Fp_n.mean():.3e}   "
           f"Fs: max={Fs_n.max():.3e} mean={Fs_n.mean():.3e}   "
           f"Fv: max={Fv_n.max():.3e} mean={Fv_n.mean():.3e}")

    # Persist artifacts
    if (step_idx % SAVE_EVERY) == 0:
        tri_csv = OUTPUT_DIR / f"iter_{step_idx:04d}_tri_stats.csv"
        with open(tri_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["triangle_id","area_PL","signed_volume_about_centroid",
                        "Ax","Ay","Az","Bx","By","Bz","Cx","Cy","Cz"])
            r0 = xyz2.mean(0)
            for tid, (i, j, k) in enumerate(faces):
                a_p, b_p, c_p = xyz2[i], xyz2[j], xyz2[k]
                area_pl = 0.5 * np.linalg.norm(np.cross(b_p - a_p, c_p - a_p))
                vol  = np.dot(a_p - r0, np.cross(b_p - r0, c_p - r0)) / 6
                w.writerow([tid, area_pl, vol, *a_p, *b_p, *c_p])

        mesh = meshio.Mesh(points=xyz2, cells=[("triangle", faces.astype(np.int32))])
        with silence_all():
            meshio.write(OUTPUT_DIR / f"surface_iter_{step_idx:04d}.msh", mesh, file_format="gmsh22", binary=False)

        # --- save 3D mesh view as PNG ---
        tris_plot = [xyz2[list(t)] for t in faces]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris_plot, alpha=0.9, linewidth=0.3, edgecolor="k"))
        ax.set_box_aspect((1, 1, 1))

        mm = 1.0e-3
        ax.set_xlim(-1.5 * mm,  1.5 * mm)
        ax.set_ylim(-1.5 * mm,  1.5 * mm)
        ax.set_zlim(-1.5 * mm,  1.5 * mm)

        ax.grid(False)
        plt.tight_layout()
        with silence_all():
            plt.savefig(OUTPUT_DIR / f"org_surface_iter_{step_idx:04d}.png", dpi=150)
        plt.close(fig)

        # ---- per-vertex dump (Ai = Astar_safe, Vi = Vi_true, plus H_scalar, p_i) ----
        pv_csv = OUTPUT_DIR / f"iter_{step_idx:04d}_per_vertex.csv"
        with open(pv_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "vertex_id",
                "x","y","z",
                "distanceToO",
                "Fs_x","Fs_y","Fs_z",
                "Fv_x","Fv_y","Fv_z",
                "Fp_x","Fp_y","Fp_z",
                "Ftot_x","Ftot_y","Ftot_z","|Ftot|",
                "P_abs","dP","rho",
                "V_dual","A_dual","H_scalar",
                "U_x","U_y","U_z","|U|",
                "dx","dy","dz","|dx|",
                "dt",
                "m_i",
                "a_x","a_y","a_z","|a|"
            ])
            U_mag    = np.linalg.norm(U, axis=1)
            a_mag    = np.linalg.norm(a_vec, axis=1)
            Ftot_mag = np.linalg.norm(F_total, axis=1)
            for i in range(xyz2.shape[0]):
                dist_to_O = float(np.linalg.norm(xyz2[i]))
                w.writerow([
                    i,
                    xyz2[i,0], xyz2[i,1], xyz2[i,2],
                    dist_to_O,
                    Fs[i,0], Fs[i,1], Fs[i,2],
                    Fv[i,0], Fv[i,1], Fv[i,2],
                    Fp[i,0], Fp[i,1], Fp[i,2],
                    F_total[i,0], F_total[i,1], F_total[i,2], Ftot_mag[i],
                    P_abs_i[i], dP_i[i], rho_i[i],
                    Vi_true[i], Astar_safe[i], H_scalar[i],
                    U[i,0], U[i,1], U[i,2], U_mag[i],
                    disp[i,0], disp[i,1], disp[i,2], np.linalg.norm(disp[i]),
                    dt,
                    MASS_I_VEC[i],
                    a_vec[i,0], a_vec[i,1], a_vec[i,2], a_mag[i],
                ])

    return (
        A_PL,
        sum_A_curved,
        V_PL,
        V_total,
        Astar_safe_total,
        Vi_true_plus_total,
        float(np.mean(rho_i)),
        P_inside_mean,
        dP_mean,
        surface_tension_mean,
        A_cross,
        A_cross_ellip,
        dt,
        ave_dist_nnpt,
        ave_dist_O,
    )

# --------------------- run loop (collect per-step stats) ---------------------
iters, Apl_list, Vtotal_list, rhobar_list = [], [], [], []
Pbar_list, dPbar_list, taubar_list = [], [], []
dt_list = []
ave_dist_nnpt_list = []
ave_dist_O_list = []
Across_list = []          # RMS A(t)
Across_ellip_list = []    # ellip-fit A(t)
t_phys_list = []          # physical time list
Astar_safe_total_list = []
Vi_true_plus_total_list = []
Vpl_list = []

t_phys = 0.0
for k in range(N_STEPS):
    print("---Step---: ", k, "\n")
    (
        A_PL,
        A_curved,
        V_PL,
        V_total,
        Astar_safe_total,
        Vi_true_plus_total,
        rho_mean,
        Pbar,
        dPbar,
        taubar,
        A_cross,
        A_cross_ellip,
        dt,
        ave_dist_nnpt,
        ave_dist_O,
    ) = step_forces_dualvolume(None, faces0, step_idx=k)
    iters.append(k)
    Apl_list.append(A_PL)
    Vpl_list.append(V_PL)
    Vtotal_list.append(V_total)
    rhobar_list.append(rho_mean)
    Pbar_list.append(Pbar)
    dPbar_list.append(dPbar)
    taubar_list.append(taubar)
    dt_list.append(dt)
    ave_dist_nnpt_list.append(ave_dist_nnpt)
    ave_dist_O_list.append(ave_dist_O)
    Across_list.append(A_cross)
    Across_ellip_list.append(A_cross_ellip)
    Astar_safe_total_list.append(Astar_safe_total)
    Vi_true_plus_total_list.append(Vi_true_plus_total)

    t_phys += dt
    t_phys_list.append(t_phys)

# dimensionless time t* = f_R t, and dimensionless cross-sectional area
t_dimless_list = [F_RAYL * t for t in t_phys_list]
Across_dimless_list = [A / A_REF_CROSS for A in Across_list]                  # RMS
Across_ellip_dimless_list = [A / A_REF_CROSS for A in Across_ellip_list]      # ellip-fit

# rescale RMS so that sphere baseline is 1 (factor 3/sqrt(2))
scale_rms_to_one = 3.0 / np.sqrt(2.0)
Across_dimless_rescaled_list = [A_dim * scale_rms_to_one for A_dim in Across_dimless_list]

# --------------------- save per-step summary ---------------------
summary_csv = OUTPUT_DIR / f"{SCRIPT_STEM}_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "iter",
        "A_PL_total",
        "Astar_safe_total",
        "A_target",
        "REF_VOLUME",
        "V_PL_total",
        "V_total",
        "Vi_true_plus_total",
        "rho_mean",
        "mass_total",
        "P_inside_mean",
        "dP_mean",
        "surface_tension_mean",
        "P_outside_mean",
        "dt",
        "rel.error.V%",
        "rel.error.A%",
        "ave_dist_nnpt",
        "ave_dist_O",
        "A_ref_cross",
        "A_cross",
        "A_cross_over_A_ref",
        "A_cross_ellip",
        "A_cross_ellip_over_A_ref",
        "A_cross_over_A_ref_rescaled",
        "t_dimless"
    ])
    for (
        k,
        A_PL,
        V_PL,
        V_total,
        Astar_safe_total,
        Vi_true_plus_total,
        rbar,
        Pbar,
        dPbar,
        taubar,
        dt,
        ave_dist_nnpt,
        ave_dist_O,
        A_cross,
        A_dimless,
        A_cross_ellip,
        A_ellip_dimless,
        A_dimless_rescaled,
        t_dimless,
    ) in zip(
        iters,
        Apl_list,
        Vpl_list,
        Vtotal_list,
        Astar_safe_total_list,
        Vi_true_plus_total_list,
        rhobar_list,
        Pbar_list,
        dPbar_list,
        taubar_list,
        dt_list,
        ave_dist_nnpt_list,
        ave_dist_O_list,
        Across_list,
        Across_dimless_list,
        Across_ellip_list,
        Across_ellip_dimless_list,
        Across_dimless_rescaled_list,
        t_dimless_list,
    ):
        rel_error_V = ((V_total - REF_VOLUME) / max(REF_VOLUME, 1e-300)) * 100.0
        rel_error_A = ((Astar_safe_total - A_TARGET) / max(A_TARGET, 1e-300)) * 100.0
        w.writerow([
            k,
            f"{A_PL:.16e}",
            f"{Astar_safe_total:.16e}",
            f"{A_TARGET:.16e}",
            f"{REF_VOLUME:.16e}",
            f"{V_PL:.16e}",
            f"{V_total:.16e}",
            f"{Vi_true_plus_total:.16e}",
            f"{rbar:.6e}",
            f"{MASS_TOTAL:.16e}",
            f"{Pbar:.6e}",
            f"{dPbar:.6e}",
            f"{taubar:.6e}",
            f"{P0_ATM:.6e}",
            f"{dt:.6e}",
            f"{rel_error_V:.6e}",
            f"{rel_error_A:.6e}",
            f"{ave_dist_nnpt:.16e}",
            f"{ave_dist_O:.16e}",
            f"{A_REF_CROSS:.16e}",
            f"{A_cross:.16e}",
            f"{A_dimless:.16e}",
            f"{A_cross_ellip:.16e}",
            f"{A_ellip_dimless:.16e}",
            f"{A_dimless_rescaled:.16e}",
            f"{t_dimless:.16e}"
        ])

# --------------------- plots → PNG (ONE ROW + A/Aref vs t*) ---------------------
plt.close('all')

import pandas as pd
import matplotlib.ticker as mticker
df = pd.read_csv(summary_csv)

iters_csv = df["iter"].to_numpy()
rel_err_A = df["rel.error.A%"].to_numpy()
rel_err_V = df["rel.error.V%"].to_numpy()

# One row, two panels for rel.error
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Left: rel.error.A%
ax[0].plot(iters_csv, rel_err_A)
ax[0].set_title("rel.error.A%")
ax[0].set_xlabel("iter")
ax[0].set_ylabel("%")
ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[0].grid(True)  # <<< grid on

# Right: rel.error.V%
ax[1].plot(iters_csv, rel_err_V)
ax[1].set_title("rel.error.V%")
ax[1].set_xlabel("iter")
ax[1].set_ylabel("%")
ax[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[1].grid(True)  # <<< grid on

plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "_manifold_cotan_progress_org.png", dpi=150)
plt.close(fig)

# ---- Dimensionless cross-sectional area over dimensionless time (original RMS) ----
t_dimless = df["t_dimless"].to_numpy()
A_dimless = df["A_cross_over_A_ref"].to_numpy()

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
ax2.plot(t_dimless, A_dimless)
ax2.set_xlabel(r"Dimensionless time $t^* = f_R t$")
ax2.set_ylabel(r"$A(t) / A_{\mathrm{ref}}$")
ax2.set_title("Dimensionless cross-sectional area over dimensionless time (RMS, unscaled)")
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax2.grid(True)  # <<< grid on
plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "A_over_Aref_vs_tstar_org.png", dpi=150)
plt.close(fig2)

# ---- RMS (rescaled) -> A_over_Aref_vs_tstar_rescaled.png ----
A_dimless_rescaled = df["A_cross_over_A_ref_rescaled"].to_numpy()

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
ax3.plot(t_dimless, A_dimless_rescaled)
ax3.set_xlabel(r"Dimensionless time $t^* = f_R t$")
ax3.set_ylabel(r"$A_{\mathrm{RMS,rescaled}}(t) / A_{\mathrm{ref}}$")
ax3.set_title("Dimensionless cross-sectional area (RMS, rescaled)")
ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax3.grid(True)  # <<< grid on
plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "A_over_Aref_vs_tstar_rescaled.png", dpi=150)
plt.close(fig3)

# ---- Ellipsoid-fit -> A_over_Aref_vs_tstar_ellip.png ----
A_dimless_ellip = df["A_cross_ellip_over_A_ref"].to_numpy()

fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4))
ax4.plot(t_dimless, A_dimless_ellip)
ax4.set_xlabel(r"Dimensionless time $t^* = f_R t$")
ax4.set_ylabel(r"$A_{\mathrm{ellip}}(t) / A_{\mathrm{ref}}$")
ax4.set_title("Dimensionless cross-sectional area (ellipsoid fit)")
ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax4.grid(True)  # <<< grid on
plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "A_over_Aref_vs_tstar_ellip.png", dpi=150)
plt.close(fig4)

# --------------------- final view ---------------------
if INTERACTIVE:
    xyz_final, _ = _vertex_array(None)
    tris = [xyz_final[list(t)] for t in faces0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, alpha=0.9, linewidth=0.3, edgecolor="k"))
    ax.set_box_aspect((1, 1, 1))

    mm = 1.0e-3
    ax.set_xlim(-1.5 * mm,  1.5 * mm)
    ax.set_ylim(-1.5 * mm,  1.5 * mm)
    ax.set_zlim(-1.5 * mm,  1.5 * mm)

    ax.grid(False); plt.tight_layout(); plt.show()

# =====================================================================
