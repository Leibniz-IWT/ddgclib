#!/usr/bin/env python3
# ===================== droplet_to_sphere_modular_forces_EOS.py =====================

# -------- easy knobs --------
N_STEPS    = 1000          # how many steps to run
SAVE_EVERY = 100           # save outputs every N steps (1 = every step)

# physical constants
RHO0   = 1000.0            # kg/m^3
P0_ATM = 101325.0          # Pa
SIGMA  = 0.072             # N/m  (surface tension)
MU     = 1.0e-3            # Pa·s (surface-proxy viscosity)
RHO_MIN, RHO_MAX = 400.0, 2001.0   # plotting range (not used to clamp)
DT_CAP   = 1e-9
H_MIN    = 5e-4
TN_MAX   = 1e12

# -------- stabilizers (RELAXED for more motion) --------
ALPHA_FLOOR = 0.20          # Vi floor: Vi >= ALPHA_FLOOR * Astar * h0
RHO_CLIP    = (0.95, 1.05)  # rho in [0.95,1.05]*rho0
REMOVE_MEAN_DP = False      # allow bulk motion
C_DAMP      = 1.0e-4        # linear drag coeff in Fdamp = -C_DAMP * Astar * U
A_CLAMP     = 2e8           # m/s^2 hard clamp on |a|
U_CLAMP     = 8             # m/s  hard clamp on |U|
CFL_EDGE_FRAC = 100         # fraction of min edge per step
N_SUBSTEPS_PRESSURE = 1     # pressure substeps (keep 1 for more motion)

# -------- curved-volume behavior --------
STRICT_CV      = True          # fail hard if _curved_volume.py can’t run/parse
TIMEOUT_CV_S   = 60            # wall-clock timeout per CV attempt (seconds)
PLANE_REL_TOL  = 5e-3          # <- locked to 0.005 as requested

import sys, platform, numpy as np, os, csv, meshio, matplotlib, io, tempfile, importlib.util, shutil
import subprocess, textwrap, json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from contextlib import redirect_stdout, redirect_stderr
import warnings
import time
import matplotlib.ticker as mticker  # <-- added

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

STYLE = Path("neatplot-main/standard.mplstyle")
if STYLE.exists():
    plt.style.use(STYLE)

print(sys.executable); print(platform.python_version())

# --------------------- Tait–Murnaghan EOS ---------------------
def P_tait_murnaghan(rho, rho0=1000.0, P0=101325, K=2.15e9, n=7.15):
    rho = np.asarray(rho, float)
    st = (K / n) * ((rho / rho0)**n - 1.0)
    return P0 + st

# --------------------- ddg mesh ---------------------
from ddgclib import *
from ddgclib._complex import Complex

HC = Complex(3)
HC.triangulate()
for _ in range(2):
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

# --------------------- helpers (geometry, areas, normals) ---------------------
def _vertex_array(HC):
    verts = list(HC.V)
    xyz = np.array([np.asarray(v.x_a, float) for v in verts])
    return xyz, verts

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
        if Atri == 0: continue
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
    p = np.isclose(z, xmax, atol=ztol); add(idx[p], np.c_[x[p], y[p]])
    return np.asarray(faces, int)

def _min_edge_length(xyz, faces):
    m = np.inf
    for (i,j,k) in faces:
        for a,b in ((i,j),(j,k),(k,i)):
            m = min(m, np.linalg.norm(xyz[a]-xyz[b]))
    return m if np.isfinite(m) else 0.0

# --------------------- cotangent stuff ---------------------
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
    return Hn, N, Astar, W, Lx

# --------------------- control volumes from O ---------------------
def _barycentric_dual_volumes_from_surface_with_O(xyz, faces):
    nV = xyz.shape[0]
    Vi = np.zeros(nV, dtype=float)
    O = xyz.mean(axis=0)
    for (i, j, k) in faces:
        a, b, c = xyz[i], xyz[j], xyz[k]
        V_tet = abs(np.dot(a - O, np.cross(b - O, c - O))) / 6.0
        share = 1/3 * V_tet
        Vi[i] += share; Vi[j] += share; Vi[k] += share
    return Vi

# --------------------- forces ---------------------
def F_surface_tension(sigma, Hn, N, Astar):
    return (sigma * Hn * Astar)[:, None] * N

def F_viscous(mu, W_cotan, U, Astar):
    Lu = np.zeros_like(U)
    for (i, j), w in W_cotan.items():
        Lu[i] += w * (U[j] - U[i])
    DeltaU = (Lu.T / (2.0 * Astar)).T
    return mu * (Astar[:, None] * DeltaU)

def F_pressure_from_dual_cells(
    Vi, m_i_fixed, N, Astar, rho0=1000.0, P0=101325.0,
    alpha_floor=ALPHA_FLOOR, rho_clip=RHO_CLIP, remove_mean=REMOVE_MEAN_DP
):
    Vi      = np.asarray(Vi, float)
    Astar   = np.asarray(Astar, float)
    m_i     = np.asarray(m_i_fixed, float)

    h0      = REF_VOLUME / max(REF_AREA, 1e-30)
    v_floor = alpha_floor * Astar * h0
    Vi_eff  = np.maximum(Vi, v_floor)

    rho_i   = m_i / Vi_eff
    rho_min = rho0 * rho_clip[0]; rho_max = rho0 * rho_clip[1]
    rho_i   = np.clip(rho_i, rho_min, rho_max)

    P_abs_i = P_tait_murnaghan(rho_i, rho0=rho0, P0=P0)
    dP_i    = P_abs_i - P0
    if remove_mean:
        dP_i = dP_i - np.mean(dP_i)

    Fp      = (dP_i * Astar)[:, None] * N
    return Fp, rho_i, P_abs_i, dP_i

def accel_from_forces(F_total, m_i_fixed):
    m = np.asarray(m_i_fixed, float)
    m_safe = np.where(m > 1e-30, m, 1e-30)
    a = (F_total.T / m_safe).T
    return a

# =====================================================================
# --------------------- curved-volume helpers (FIXED) -----------------
# =====================================================================
_CV_LAST_GOOD_TIDS: set[int] | None = None

def _find_transformed_volume_csv(workdir: Path, mesh_stem: str) -> Path | None:
    cand = workdir / f"{mesh_stem}_COEFFS_Transformed_Volume.csv"
    if cand.exists():
        return cand
    for p in workdir.glob("*_COEFFS_Transformed_Volume.csv"):
        return p
    return None

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

# === NEW: read point-wise DualVolume + DualArea ======================
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
# =====================================================================

# === direct in-process call to _curved_volume.py (no subprocess) ===
_CURVED_VOL_MOD = None

class _CVTempDir(tempfile.TemporaryDirectory):
    def __init__(self, *a, **kw):
        # Windows-safe temp root. Avoid hardcoding "/tmp".
        if "dir" not in kw or kw["dir"] in (None, "", "/tmp"):
            kw["dir"] = tempfile.gettempdir()
        super().__init__(*a, **kw)

    def cleanup(self):
        pass

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
        return float(Vcorr), float(sumA), sum_A_flat, workdir
    else:
        last_err = RuntimeError("CSV parsed but missing finite Vcorr/sumA")
        pflush(f"[CV] invalid CSV tol={tol:g}")
        if STRICT_CV:
            raise RuntimeError(f"[STRICT_CV] _curved_volume.py produced no usable CSV (last_err={last_err}).")
        pflush(f"[warn] CSV missing/invalid (last_err={last_err}). Using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None

# --------------------- initial conditions & OUTPUT DIR ---------------------
script_dir = Path(__file__).resolve().parent

# NEW: output folder name = script filename stem
# e.g. droplet_to_sphere_modular_forces_EOS.py -> droplet_to_sphere_modular_forces_EOS/
OUTPUT_DIR = script_dir / Path(__file__).stem
OUTPUT_DIR.mkdir(exist_ok=True)

xyz0, verts0 = _vertex_array(HC)

# scale to a 1 mm cube (centered)
c0 = np.mean(xyz0, axis=0)
xyz0 = xyz0 - c0
xyz0 *= 1.0e-3
for i, v in enumerate(list(HC.V)):
    HC.V.move(v, tuple(xyz0[i]))
xyz0, _ = _vertex_array(HC)
faces0 = _build_cube_surface_faces(xyz0)

mesh0 = meshio.Mesh(points=xyz0, cells=[("triangle", faces0.astype(np.int32))])
with silence_all():
    plt.close('all')
    meshio.write(OUTPUT_DIR / "surface_iter_0000.msh", mesh0, file_format="gmsh22", binary=False)

REF_VOLUME = float(_enclosed_volume_PL(xyz0, faces0))
REF_AREA   = float(_surface_area_PL(xyz0, faces0))

def _sphere_area_from_volume(V):
    return 4.0 * np.pi * ((3.0 * V / (4.0 * np.pi)) ** (2.0 / 3.0))
A_TARGET = float(_sphere_area_from_volume(REF_VOLUME))

VI_REF       = _barycentric_dual_volumes_from_surface_with_O(xyz0, faces0)
MASS_I_VEC   = RHO0 * VI_REF
MASS_TOTAL   = float(np.sum(MASS_I_VEC))

_prev_U = np.zeros_like(xyz0)

def step_forces_dualvolume(HC, faces, step_idx: int):
    global _prev_U
    xyz, _ = _vertex_array(HC)

    mesh_iter_path = (OUTPUT_DIR / f"surface_iter_{step_idx:04d}.msh").resolve()
    mesh_iter = meshio.Mesh(points=xyz, cells=[("triangle", np.asarray(faces, np.int32))])
    with silence_all():
        meshio.write(mesh_iter_path, mesh_iter, file_format="gmsh22", binary=False)

    # per-triangle flat areas (same ordering as 'faces')
    A_flat_list_local = _per_triangle_area(xyz, faces)

    Aprev_PL = _surface_area_PL(xyz, faces)
    V_tet    = _enclosed_volume_PL(xyz, faces)

    V_corr, sum_A_curved, sum_A_flat, tmpdir = _curved_volume_and_sumAcurved_with_flat_fallback(
        xyz, faces, plane_rel_tol=PLANE_REL_TOL, step_idx=step_idx, mesh_path=mesh_iter_path
    )

    V_total = V_tet + V_corr*1

    # ---------- per-vertex Aflat and Ac (triangle shares) ----------
    nV = xyz.shape[0]
    Aflat_pt  = np.zeros(nV, dtype=float)
    Ac_pt_raw = np.zeros(nV, dtype=float)

    good_tids = _load_good_tids_from_coeffs(OUTPUT_DIR, mesh_iter_path.stem) or set()
    volcsv    = _find_transformed_volume_csv(OUTPUT_DIR, mesh_iter_path.stem)
    Amap_tri  = _read_Acurved_map(volcsv, good_tids=good_tids) if volcsv else None

    for tid, (i, j, k) in enumerate(faces):
        Atri_flat = float(A_flat_list_local[tid])
        if Amap_tri is not None:
            val = Amap_tri.get(tid, np.nan)
            Atri_curv = float(val) if np.isfinite(val) else Atri_flat
        else:
            Atri_curv = Atri_flat

        share_flat = Atri_flat / 3.0
        share_curv = Atri_curv / 3.0

        Aflat_pt[i]  += share_flat;  Aflat_pt[j]  += share_flat;  Aflat_pt[k]  += share_flat
        Ac_pt_raw[i] += share_curv;  Ac_pt_raw[j] += share_curv;  Ac_pt_raw[k] += share_curv

    # final per-vertex Ac uses individual max with Aflat
    Ac_pt = np.maximum(Ac_pt_raw, Aflat_pt)
    # ----------------------------------------------------------------

    # Geometry & differential operators (keep original computation)
    Hn, N, _Astar_unused, W_cotan, Lx = mean_curvature_scalar(xyz, faces)

    # === ENFORCE YOUR RULE: Astar_safe = max(Aflat, Ac) ===
    Astar_safe = np.maximum(Aflat_pt, Ac_pt)

    # apply per-point DualArea/Volume from CSV if present
    dv_map = _load_point_dualvolume_map(OUTPUT_DIR)

    # --- Vi_true arrays & DualVolume vector -------------------------
    Vi_true_base = _barycentric_dual_volumes_from_surface_with_O(xyz, faces)
    Vi_true_plus = Vi_true_base.copy()
    DualVolume_vec = np.zeros(nV, dtype=float)
    if dv_map:
        for pid, (dv, _da) in dv_map.items():
            if 0 <= pid < nV:
                Vi_true_plus[pid] += dv
                DualVolume_vec[pid] = dv
    Vi_true = Vi_true_plus
    # ----------------------------------------------------------------

    # Pressure force uses Astar_safe
    Fp, rho_i, P_abs_i, dP_i = F_pressure_from_dual_cells(
        Vi=Vi_true, m_i_fixed=MASS_I_VEC, N=N, Astar=Astar_safe, rho0=RHO0, P0=P0_ATM
    )

    A_sum = float(np.sum(Astar_safe) + 1e-30)
    tau_bar = float(np.sum((-SIGMA * Hn) * Astar_safe) / A_sum)
    dP_bar  = float(np.sum(dP_i * Astar_safe) / A_sum)
    if step_idx == 0:
        shift  = tau_bar - dP_bar
        dP_i   = dP_i + shift
        P_abs_i = P0_ATM + dP_i
        Fp      = (dP_i * Astar_safe)[:, None] * N

    pflush(f"*⟨P_abs⟩ (Pa): {float(np.mean(P_abs_i)):.5f}  *⟨rho⟩ {float(np.mean(rho_i)):.3f}")

    # Surface tension from cotan Laplacian (kept)
    Fs = SIGMA * Lx

    P_inside_mean        = float(np.sum(P_abs_i * Astar_safe) / A_sum)
    dP_mean              = float(np.sum(dP_i * Astar_safe) / A_sum)
    surface_tension_mean = tau_bar

    h_eff = V_total / max(Aprev_PL, 1e-18)
    mu_eff = MU * h_eff

    U_for_visc = _prev_U if step_idx > 0 else np.zeros_like(_prev_U)
    Fv = F_viscous(mu_eff, W_cotan, U_for_visc, Astar_safe)

    Fdrag = -C_DAMP * (Astar_safe[:, None] * U_for_visc)

    F_total = Fp + Fs + Fv + Fdrag

    a_vec = accel_from_forces(F_total, MASS_I_VEC)
    a_norm = np.linalg.norm(a_vec, axis=1)
    mask_a = a_norm > A_CLAMP
    if np.any(mask_a):
        a_vec[mask_a] *= (A_CLAMP / a_norm[mask_a])[:, None]

    U_prov = _prev_U + DT_CAP * a_vec
    v_sq_area = np.sum((np.linalg.norm(U_prov, axis=1) ** 2) * Astar_safe) + 1e-18
    A_drop    = 5e-3 * Aprev_PL
    dt        = min(DT_CAP, max(5e-6, A_drop / v_sq_area))

    umax = float(np.max(np.linalg.norm(U_prov, axis=1)) + 1e-30)
    L_min = _min_edge_length(xyz, faces)
    if L_min > 0 and umax > 0:
        dt = min(dt, CFL_EDGE_FRAC * L_min / umax)

    nsub = max(1, int(N_SUBSTEPS_PRESSURE))
    sub_dt = dt / nsub
    U = _prev_U.copy()
    for _ in range(nsub):
        U = U + sub_dt * a_vec
        U_norm = np.linalg.norm(U, axis=1)
        mask_u = U_norm > U_CLAMP
        if np.any(mask_u):
            U[mask_u] *= (U_CLAMP / U_norm[mask_u])[:, None]

    disp = dt * U if step_idx == 0 else dt * (1.5 * U - 0.5 * _prev_U)

    for i, v in enumerate(list(HC.V)):
        HC.V.move(v, tuple((xyz + disp)[i]))
    _prev_U = U.copy()

    # after move
    xyz2, _ = _vertex_array(HC)

    # --- radial projection blend toward sphere of volume REF_VOLUME ---
    r_target = (3.0 * REF_VOLUME / (4.0 * np.pi)) ** (1.0 / 3.0)
    r2 = np.linalg.norm(xyz2, axis=1)
    lambda_proj = 0.02  # full radial projection each step
    xyz2 = xyz2 + ((r_target - r2) / np.maximum(r2, 1e-30))[:, None] * xyz2 * lambda_proj
    for i, v in enumerate(list(HC.V)):
        HC.V.move(v, tuple(xyz2[i]))
    xyz2, _ = _vertex_array(HC)
    # -----------------------------------------------------------------

    A_PL = _surface_area_PL(xyz2, faces)
    V_PL = _enclosed_volume_PL(xyz2, faces)

    # === average distance between neighboring points (edges of triangles) ===
    edge_set = set()
    for (i, j, k) in faces:
        if i < j: edge_set.add((i, j))
        else:     edge_set.add((j, i))
        if j < k: edge_set.add((j, k))
        else:     edge_set.add((k, j))
        if k < i: edge_set.add((k, i))
        else:     edge_set.add((i, k))
    total_edge_len = 0.0
    for (a, b) in edge_set:
        pa = xyz2[a]; pb = xyz2[b]
        total_edge_len += float(np.linalg.norm(pb - pa))
    ave_dist_nnpt = total_edge_len / max(len(edge_set), 1)

    # === average distance to origin (0,0,0) for this iteration ===
    dist_O = np.linalg.norm(xyz2, axis=1)
    ave_dist_O = float(np.mean(dist_O))

    Fp_n = np.linalg.norm(Fp, axis=1); Fs_n = np.linalg.norm(Fs, axis=1); Fv_n = np.linalg.norm(Fv, axis=1)
    Fs_n_along_n = np.sum(Fs * N, axis=1)
    total_Fs_outward = float(np.sum(Fs_n_along_n[Fs_n_along_n > 0.0]))
    total_Fs_inward  = float(-np.sum(Fs_n_along_n[Fs_n_along_n < 0.0]))
    pflush(f"   [Fs only] total outward={total_Fs_outward:.6e}   total inward={total_Fs_inward:.6e}")
    pflush(f"   A_init(PL)={REF_AREA:.6e}  V_init(PL)={REF_VOLUME:.6e}")

    dA_PL_total_rel = (A_PL - REF_AREA) / max(REF_AREA, 1e-300)
    dV_total_rel    = (V_total - REF_VOLUME) / max(REF_VOLUME, 1e-300)
    pflush(f"   [TOTAL] ΔA_PL_total/A0={dA_PL_total_rel:+.6e} ({dA_PL_total_rel*100:+.3f}%)   "
           f"ΔV_total/V0={dV_total_rel:+.6e} ({dV_total_rel*100:+.3f}%)")

    rho_mean  = float(np.mean(rho_i))
    dP_mean_print = float(np.mean(dP_i))
    pflush(
        f"k={step_idx:04d}  A_curved≈{sum_A_curved:.6e}  A_curved_sum={sum_A_curved:.6e}  A_PL→{A_PL:.6e}  "
        f"V_total(prev)={V_total:.6e}  V_corr_sum={V_corr:.6e}  V_PL→{V_PL:.6e}  ⟨ΔP⟩={dP_mean_print:.3e} Pa  "
        f"max|dx|={np.linalg.norm(disp,axis=1).max():.3e}  mean|dx|={np.linalg.norm(disp,axis=1).mean():.3e}  dt={dt:.3e}"
    )
    pflush(
        f"   [CV] V_corr={V_corr:.6e}  A_curved_sum={sum_A_curved:.6e}  A_flat_sum={sum_A_flat:.6e}"
    )
    pflush(
        f"   Fp: max={Fp_n.max():.3e} mean={Fp_n.mean():.3e}   "
        f"Fs: max={Fs_n.max():.3e} mean={Fs_n.mean():.3e}   "
        f"Fv: max={Fv_n.max():.3e} mean={Fv_n.mean():.3e}"
    )

    # ---- WRITE PER-VERTEX CSV + NEW PNG/TRI CSV ----
    if step_idx % SAVE_EVERY == 0:
        out_csv = OUTPUT_DIR / f"iter_{step_idx:04d}_per_vertex.csv"
        with open(out_csv, "w", newline="") as f:
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
                "V_dual","A_dual",
                "Vi_true_base","Vi_true_plus_total","DualVolume",
                "U_x","U_y","U_z","|U|",
                "dx","dy","dz","|dx|",
                "dt",
                "m_i",
                "a_x","a_y","a_z","|a|",
                "Aflat","Ac","Astar_safe",
            ])
            U_mag     = np.linalg.norm(U, axis=1)
            a_mag     = np.linalg.norm(a_vec, axis=1)
            Ftot_mag  = np.linalg.norm(F_total, axis=1)
            disp_mag  = np.linalg.norm(disp, axis=1)
            for pid in range(xyz2.shape[0]):
                dist_to_O = float(np.linalg.norm(xyz2[pid]))
                w.writerow([
                    pid,
                    xyz2[pid, 0], xyz2[pid, 1], xyz2[pid, 2],
                    dist_to_O,
                    Fs[pid, 0], Fs[pid, 1], Fs[pid, 2],
                    Fv[pid, 0], Fv[pid, 1], Fv[pid, 2],
                    Fp[pid, 0], Fp[pid, 1], Fp[pid, 2],
                    F_total[pid, 0], F_total[pid, 1], F_total[pid, 2], Ftot_mag[pid],
                    P_abs_i[pid], dP_i[pid], rho_i[pid],
                    float(Vi_true[pid]), float(Astar_safe[pid]),
                    float(Vi_true_base[pid]), float(Vi_true_plus[pid]), float(DualVolume_vec[pid]),
                    U[pid, 0], U[pid, 1], U[pid, 2], U_mag[pid],
                    disp[pid, 0], disp[pid, 1], disp[pid, 2], disp_mag[pid],
                    dt,
                    MASS_I_VEC[pid],
                    a_vec[pid, 0], a_vec[pid, 1], a_vec[pid, 2], a_mag[pid],
                    float(Aflat_pt[pid]), float(Ac_pt[pid]), float(Astar_safe[pid])
                ])
        pflush(f"[write] per-vertex data -> {out_csv}")

        tri_csv = OUTPUT_DIR / f"iter_{step_idx:04d}_tri_stats.csv"
        with open(tri_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["triangle_id","area_PL","signed_volume_about_centroid",
                        "Ax","Ay","Az","Bx","By","Bz","Cx","Cy","Cz"])
            r0 = xyz2.mean(axis=0)
            for tid, (i, j, k) in enumerate(faces):
                a_p, b_p, c_p = xyz2[i], xyz2[j], xyz2[k]
                area_pl = 0.5 * np.linalg.norm(np.cross(b_p - a_p, c_p - a_p))
                vol  = np.dot(a_p - r0, np.cross(b_p - r0, c_p - r0)) / 6.0
                w.writerow([tid, area_pl, vol, *a_p, *b_p, *c_p])
        pflush(f"[write] tri stats -> {tri_csv}")

        tris_plot = [xyz2[list(t)] for t in faces]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris_plot, alpha=0.9, linewidth=0.3, edgecolor="k"))
        ax.set_box_aspect((1, 1, 1))

        mm = 1.0e-3
        ax.set_xlim(-0.8 * mm,  0.8 * mm)
        ax.set_ylim(-0.8 * mm,  0.8 * mm)
        ax.set_zlim(-0.8 * mm,  0.8 * mm)

        ax.grid(False)
        plt.tight_layout()
        with silence_all():
            plt.savefig(OUTPUT_DIR / f"org_surface_iter_{step_idx:04d}.png", dpi=150)
        plt.close(fig)

    Vi_true_base_total = float(np.sum(Vi_true_base))
    Vi_true_plus_total = float(np.sum(Vi_true_plus))

    return (
        _surface_area_PL(xyz2, faces),    # A_PL
        sum_A_curved,                     # A_curved
        _enclosed_volume_PL(xyz2, faces), # V_PL
        V_total,                          # V_total
        Vi_true_base_total,               # Vi_true_base_total
        Vi_true_plus_total,               # Vi_true_plus_total
        float(np.mean(rho_i)),            # rho_mean
        float(np.sum(P_abs_i * Astar_safe) / (np.sum(Astar_safe)+1e-30)),  # P_inside_mean
        float(np.sum(dP_i * Astar_safe) / (np.sum(Astar_safe)+1e-30)),     # dP_mean
        float(np.sum((-SIGMA * Hn) * Astar_safe) / (np.sum(Astar_safe)+1e-30)),  # surface_tension_mean
        dt,
        float(np.sum(Astar_safe)),        # A_sum_vertices
        ave_dist_nnpt,
        ave_dist_O,
    )

# ---------------- main loop ----------------
iters, Apl_list, Vtotal_list, Vi_true_base_total_list, Vi_true_plus_total_list, rhobar_list = [], [], [], [], [], []
Pbar_list, dPbar_list, taubar_list = [], [], []
dt_list = []
Avertexsum_list = []
ave_dist_nnpt_list = []
ave_dist_O_list = []

for k in range(N_STEPS):
    (
        A_PL,
        A_curved,
        V_PL,
        V_total,
        Vi_true_base_total,
        Vi_true_plus_total,
        rho_mean,
        Pbar,
        dPbar,
        taubar,
        dt,
        A_sum_vertices,
        ave_dist_nnpt,
        ave_dist_O,
    ) = step_forces_dualvolume(HC, faces0, step_idx=k)
    iters.append(k)
    Apl_list.append(A_PL)
    Vtotal_list.append(V_total)
    Vi_true_base_total_list.append(Vi_true_base_total)
    Vi_true_plus_total_list.append(Vi_true_plus_total)
    rhobar_list.append(rho_mean)
    Pbar_list.append(Pbar)
    dPbar_list.append(dPbar)
    taubar_list.append(taubar)
    dt_list.append(dt)
    Avertexsum_list.append(A_sum_vertices)
    ave_dist_nnpt_list.append(ave_dist_nnpt)
    ave_dist_O_list.append(ave_dist_O)

summary_csv = OUTPUT_DIR / "_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "iter", "A_PL_total", "Astar_safe_total", "A_target",
        "V_total", "Vi_true_base_total", "Vi_true_plus_total",
        "rho_mean", "mass_total",
        "P_inside_mean", "dP_mean", "surface_tension_mean", "P_outside_mean", "dt",
        "rel.error.V%", "rel.error.A%", "ave_dist_nnpt", "ave_dist_O"
    ])
    for k, A_PL, V_total, Vi_base_tot, Vi_plus_tot, rbar, Pbar, dPbar, taubar, dt, A_sum_vertices, ave_dist_nnpt, ave_dist_O in zip(
        iters, Apl_list, Vtotal_list, Vi_true_base_total_list, Vi_true_plus_total_list,
        rhobar_list, Pbar_list, dPbar_list, taubar_list, dt_list,
        Avertexsum_list, ave_dist_nnpt_list, ave_dist_O_list
    ):
        Astar_safe_total = A_sum_vertices
        rel_error_V = ((V_total - REF_VOLUME) / max(REF_VOLUME, 1e-300)) * 100.0
        rel_error_A = ((Astar_safe_total - A_TARGET) / max(A_TARGET, 1e-300)) * 100.0
        w.writerow([
            k, f"{A_PL:.16e}", f"{Astar_safe_total:.16e}", f"{A_TARGET:.16e}",
            f"{V_total:.16e}", f"{Vi_base_tot:.16e}", f"{Vi_plus_tot:.16e}",
            f"{rbar:.6e}", f"{MASS_TOTAL:.16e}",
            f"{Pbar:.6e}", f"{dPbar:.6e}", f"{taubar:.6e}", f"{P0_ATM:.6e}", f"{dt:.6e}",
            f"{rel_error_V:.6e}", f"{rel_error_A:.6e}", f"{ave_dist_nnpt:.16e}", f"{ave_dist_O:.16e}"
        ])

plt.close('all')

# ---------------- plotting (from _summary.csv) ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

iters_csv, rel_err_A, rel_err_V = [], [], []
with open(summary_csv, "r", newline="") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if not row:
            continue
        try:
            iters_csv.append(float(row[0]))
            rel_err_V.append(float(row[14]))  # rel.error.V%
            rel_err_A.append(float(row[15]))  # rel.error.A%
        except Exception:
            continue

mask   = [x >= 150 for x in iters_csv]
x_150  = [x for x, m in zip(iters_csv, mask) if m]
a_150  = [y for y, m in zip(rel_err_A, mask)   if m]
v_150  = [y for y, m in zip(rel_err_V, mask)   if m]

eps = 1e-12
a_150_abs = np.maximum(np.abs(a_150), eps)
v_150_abs = np.maximum(np.abs(v_150), eps)

ax[0].plot(x_150, a_150_abs)
ax[0].set_title("|rel.error.A%|")
ax[0].set_xlabel("iter")
ax[0].set_ylabel("|%|")
ax[0].set_yscale('log')
ax[0].set_xlim(left=150)
ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax[1].plot(x_150, v_150_abs)
ax[1].set_title("|rel.error.V%|")
ax[1].set_xlabel("iter")
ax[1].set_ylabel("|%|")
ax[1].set_yscale('log')
ax[1].set_xlim(left=150)
ax[1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
ax[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "manifold_cotan_progress.png", dpi=150)

if INTERACTIVE:
    xyz_final, _ = _vertex_array(HC)
    tris = [xyz_final[list(t)] for t in faces0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, alpha=0.9, linewidth=0.3, edgecolor="k"))
    ax.set_box_aspect((1, 1, 1))
    mins = xyz_final.min(axis=0); maxs = xyz_final.max(axis=0)
    ctr  = 0.5 * (mins + maxs); half = 0.55 * float(np.max(maxs - mins))
    half = half if (np.isfinite(half) and half > 0) else 1e-3
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.grid(False); plt.tight_layout(); plt.show()

# =====================================================================
