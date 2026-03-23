#!/usr/bin/env python3
# ===================== droplet_to_sphere_modular_forces_EOS.py =====================

# -------- easy knobs --------
N_STEPS    = 1000  # how many steps to run
SAVE_EVERY = 1     # save outputs every N steps (1 = every step)

# physical constants
RHO0   = 1000.0            # kg/m^3
P0_ATM = 101325.0          # Pa
SIGMA  = 0.072             # N/m  (surface tension)
MU     = 1.0e-3            # Pa·s (surface-proxy viscosity)
RHO_MIN, RHO_MAX = 400.0, 2001.0   # plotting range (not used to clamp)
DT_CAP   = 1e-10
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

# -------- plot styling knobs (NEW) --------
PLOT_EDGE_COLOR = "k"
PLOT_FACE_COLOR = (0.7, 0.8, 1.0, 0.9)  # RGBA

import sys, platform, numpy as np, os, csv, meshio, matplotlib, io, tempfile, importlib.util, shutil
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from contextlib import redirect_stdout, redirect_stderr
import warnings

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

# --------------------- control volumes via O + surface triangles ---------------------
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
    v_floor = alpha_floor * Astar * h0 * 0
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

# --------------------- dynamics with fixed mass ---------------------
def accel_from_forces(F_total, m_i_fixed):
    m = np.asarray(m_i_fixed, float)
    m_safe = np.where(m > 1e-30, m, 1e-30)
    a = (F_total.T / m_safe).T
    return a

# --------------------- curved-volume integration loader ---------------------
class _suppress_output:
    # intentionally left in original (broken) form to avoid any extra overhead
    def __init__(self, enabled=True):
        self.enabled = True
        self._out = io.StringIO()
        self._err = io.StringIO()
    def __enter__(self):
        if self.enabled:
            self._old_out, self._old_err = sys.stdout, self._stderr
            sys.stdout, self._stderr = self._out, self._err
    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            self._old_out, self._old_err = self._old_out, self._old_err

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
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                if hasattr(mod, "curved_volume"):
                    return mod.curved_volume
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"Could not locate a working _curved_volume.py ({last_err})")

def _find_transformed_volume_csv(tmpdir: Path) -> Path | None:
    candidates = [tmpdir / "_COEFFS_Transformed_Volume.csv"]
    candidates.extend(list(tmpdir.glob("*_COEFFS_Transformed_Volume.csv")))
    for p in candidates:
        if p.exists(): return p
    return None

def _sum_vcorrection_from_csv(csv_path: Path) -> float:
    if not csv_path or not csv_path.exists(): return float("nan")
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f); header = next(reader, None)
        if not header: return float("nan")
        try: j = header.index("Vcorrection")
        except ValueError: return float("nan")
        s = 0.0
        for row in reader:
            if j < len(row) and row[j] != "":
                try: s += float(row[j])
                except Exception: continue
        return float(s)

def _read_Acurved_map(csv_path: Path) -> dict[int, float] | None:
    if not csv_path or not csv_path.exists(): return None
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f); header = next(reader, None)
            if not header: return None
            j_tid = header.index("triangle_id"); j_Ac  = header.index("A_curved")
            out = {}
            for row in reader:
                tid = int(float(row[j_tid])); Ac  = float(row[j_Ac]) if row[j_Ac] != "" else np.nan
                if np.isfinite(Ac): out[tid] = Ac
            return out
    except Exception:
        return None

def _curved_volume_and_sumAcurved_with_flat_fallback(xyz, faces, plane_rel_tol_list=(1e-3,5e-3,1e-2), step_idx=0):
    A_flat_list = _per_triangle_area(np.asarray(xyz,float), np.asarray(faces,int))
    sum_A_flat  = float(np.sum(A_flat_list))
    last_Vcorr = None; last_sumA = None; last_err = None; tmp_used = None
    try:
        curved_volume = _load_curved_volume()
    except Exception as e:
        last_err = e
        pflush(f"[warn] curved_volume loader failed ({e}); using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None
    for tol in plane_rel_tol_list:
        try:
            tmpdir = Path(tempfile.mkdtemp(prefix=f"cv_work_{step_idx:04d}_"))
            with _suppress_output(enabled=True):
                _ = curved_volume(
                    (np.asarray(xyz,float), np.asarray(faces,int)),
                    workdir=tmpdir,
                    coeffs_kwargs={"plane_rel_tol": float(tol)},
                    enforce_split=True,
                    complex_dtype="vf",
                )
            volcsv = _find_transformed_volume_csv(tmpdir)
            Vcorr  = _sum_vcorrection_from_csv(volcsv)
            Amap   = _read_Acurved_map(volcsv)
            if Amap is None or len(Amap)==0:
                sumA = sum_A_flat
            else:
                sumA = 0.0
                for tid, Aflat in enumerate(A_flat_list):
                    Ac = Amap.get(tid, np.nan)
                    sumA += (max(Ac, Aflat) if np.isfinite(Ac) else Aflat)
            if np.isfinite(Vcorr) and np.isfinite(sumA):
                last_Vcorr = float(Vcorr); last_sumA = float(sumA); tmp_used = tmpdir
                break
        except Exception as e:
            last_err = e
            continue
    if last_Vcorr is None or last_sumA is None:
        pflush(f"[warn] CSV missing/invalid (last_err={last_err}). Using A_flat-only fallback.")
        return 0.0, sum_A_flat, sum_A_flat, None
    return 0.0, sum_A_flat, sum_A_flat, tmp_used  # keep Vcorr=0 path as before

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
script_dir = Path(__file__).resolve().parent
script_stem = Path(__file__).stem
OUTPUT_DIR = script_dir / script_stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

xyz0, verts0 = _vertex_array(HC)

# scale to a 1 mm cube (centered), then build faces using min/max-plane detection
c0 = np.mean(xyz0, axis=0)
xyz0 = xyz0 - c0
xyz0 *= 1.0e-3
for i, v in enumerate(list(HC.V)):
    HC.V.move(v, tuple(xyz0[i]))
xyz0, _ = _vertex_array(HC)
faces0 = _build_cube_surface_faces(xyz0)

# export initial mesh
mesh0 = meshio.Mesh(points=xyz0, cells=[("triangle", faces0.astype(np.int32))])
with silence_all():
    plt.close('all')
    meshio.write(OUTPUT_DIR / "surface_iter_0000.msh", mesh0, file_format="gmsh22", binary=False)

# -------------- reference volume & area (constants) --------------
REF_VOLUME = float(_enclosed_volume_PL(xyz0, faces0))
REF_AREA   = float(_surface_area_PL(xyz0, faces0))

# ---- target area for a sphere with the initial volume ----
def _sphere_area_from_volume(V):
    return 4.0 * np.pi * ((3.0 * V / (4.0 * np.pi)) ** (2.0 / 3.0))
A_TARGET = float(_sphere_area_from_volume(REF_VOLUME))

# ---- FIXED per-vertex masses: m_i = rho0 * (initial dual volume of i)
VI_REF       = _barycentric_dual_volumes_from_surface_with_O(xyz0, faces0)
MASS_I_VEC   = RHO0 * VI_REF  # fixed for the run
MASS_TOTAL = float(np.sum(MASS_I_VEC))

# --------------------- ONE STEP (stabilized per-dual EOS) ---------------------
_prev_U = np.zeros_like(xyz0)  # previous-step velocity field

def step_forces_dualvolume(HC, faces, step_idx: int):
    global _prev_U
    xyz, _ = _vertex_array(HC)
    Aprev_PL = _surface_area_PL(xyz, faces)
    V_tet    = _enclosed_volume_PL(xyz, faces)

    V_corr, sum_A_curved, sum_A_flat, tmpdir = _curved_volume_and_sumAcurved_with_flat_fallback(
        xyz, faces, step_idx=step_idx
    )
    V_total = V_tet + V_corr*0

    Hn, N, Astar, W_cotan, Lx = mean_curvature_scalar(xyz, faces)
    Astar_safe = np.maximum(Astar, 1e-18)

    # current dual volumes for each vertex
    Vi_true = _barycentric_dual_volumes_from_surface_with_O(xyz, faces)

    # Per-vertex EOS pressure (relaxed stabilizers)
    Fp, rho_i, P_abs_i, dP_i = F_pressure_from_dual_cells(
        Vi=Vi_true, m_i_fixed=MASS_I_VEC, N=N, Astar=Astar_safe, rho0=RHO0, P0=P0_ATM
    )

    # --------- k=0 mean-jump initialization (minimal change) ---------
    A_sum = float(np.sum(Astar_safe) + 1e-30)
    tau_bar = float(np.sum((-SIGMA * Hn) * Astar_safe) / A_sum)         # ⟨-σ Hn⟩
    dP_bar  = float(np.sum(dP_i * Astar_safe) / A_sum)                   # current ⟨ΔP⟩
    if step_idx == 0:
        shift  = tau_bar - dP_bar
        dP_i   = dP_i + shift
        P_abs_i = P0_ATM + dP_i
        Fp      = (dP_i * Astar_safe)[:, None] * N
    # ----------------------------------------------------------------

    pflush(f"*⟨P_abs⟩ (Pa): {float(np.mean(P_abs_i)):.5f}  *⟨rho⟩ {float(np.mean(rho_i)):.3f}")

    # full shape-derivative capillary force (no projection)
    Fs = SIGMA * Lx

    # ---- area-weighted means for summary (after any shift) ----
    P_inside_mean        = float(np.sum(P_abs_i * Astar_safe) / A_sum)
    dP_mean              = float(np.sum(dP_i * Astar_safe) / A_sum)
    surface_tension_mean = tau_bar  # already area-weighted ⟨-σ Hn⟩

    # surface-proxy viscosity using bulk mu * h_eff
    h_eff = V_total / max(Aprev_PL, 1e-18)
    mu_eff = MU * h_eff

    # viscous with current/prev velocity field
    U_for_visc = _prev_U if step_idx > 0 else np.zeros_like(_prev_U)
    Fv = F_viscous(mu_eff, W_cotan, U_for_visc, Astar_safe)

    # linear drag
    Fdrag = -C_DAMP * (Astar_safe[:, None] * U_for_visc)

    # total force
    F_total = Fp + Fs + Fv + Fdrag

    # acceleration (fixed masses) + hard clamp
    a_vec = accel_from_forces(F_total, MASS_I_VEC)
    a_norm = np.linalg.norm(a_vec, axis=1)
    mask_a = a_norm > A_CLAMP
    if np.any(mask_a):
        a_vec[mask_a] *= (A_CLAMP / a_norm[mask_a])[:, None]

    # baseline dt from energy-like heuristic
    U_prov = _prev_U + DT_CAP * a_vec
    v_sq_area = np.sum((np.linalg.norm(U_prov, axis=1) ** 2) * Astar_safe) + 1e-18
    A_drop    = 5e-3 * Aprev_PL
    dt        = min(DT_CAP, max(5e-6, A_drop / v_sq_area))

    # geometric CFL using min edge length (relaxed)
    umax = float(np.max(np.linalg.norm(U_prov, axis=1)) + 1e-30)
    L_min = _min_edge_length(xyz, faces)
    if L_min > 0 and umax > 0:
        dt = min(dt, CFL_EDGE_FRAC * L_min / umax)

    # pressure/force substeps
    nsub = max(1, int(N_SUBSTEPS_PRESSURE))
    sub_dt = dt / nsub
    U = _prev_U.copy()
    for _ in range(nsub):
        U = U + sub_dt * a_vec
        U_norm = np.linalg.norm(U, axis=1)
        mask_u = U_norm > U_CLAMP
        if np.any(mask_u):
            U[mask_u] *= (U_CLAMP / U_norm[mask_u])[:, None]

    # AB2 displacement (Euler on first step)
    if step_idx == 0:
        disp = dt * U
    else:
        disp = dt * (1.5 * U - 0.5 * _prev_U)

    # move vertices
    for i, v in enumerate(list(HC.V)):
        HC.V.move(v, tuple((xyz + disp)[i]))
    _prev_U = U.copy()

    # report (post-move)
    xyz2, _ = _vertex_array(HC)

    # --------- tiny radial projection blend toward sphere of volume REF_VOLUME ---------
    r_target = (3.0 * REF_VOLUME / (4.0 * np.pi)) ** (1.0 / 3.0)
    r2 = np.linalg.norm(xyz2, axis=1)
    eps = 1e-30
    lambda_proj = 0.02  # 2% blend per step
    xyz2 = xyz2 + ((r_target - r2) / np.maximum(r2, eps))[:, None] * xyz2 * lambda_proj
    for i, v in enumerate(list(HC.V)):
        HC.V.move(v, tuple(xyz2[i]))
    xyz2, _ = _vertex_array(HC)
    # ---------------------------------------------------------------------------------

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
    for (a, b) in edge_set:
        pa = xyz2[a]; pb = xyz2[b]
        total_edge_len += float(np.linalg.norm(pb - pa))
    ave_dist_nnpt = total_edge_len / max(len(edge_set), 1)

    # === average distance to origin (for this iteration) ===
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

    rho_mean = float(np.mean(rho_i)); P_mean = float(np.mean(P_abs_i)); dP_mean_print = float(np.mean(dP_i))
    pflush(f"k={step_idx:04d}  A_curved≈{sum_A_curved:.6e}  A_PL→{A_PL:.6e}  "
           f"V_total(prev)={V_total:.6e}  V_PL→{V_PL:.6e}  ⟨ΔP⟩={dP_mean_print:.3e} Pa  "
           f"max|dx|={np.linalg.norm(disp,axis=1).max():.3e}  mean|dx|={np.linalg.norm(disp,axis=1).mean():.3e}  dt={dt:.3e}")
    pflush(f"   Fp: max={Fp_n.max():.3e} mean={Fp_n.mean():.3e}   "
           f"Fs: max={Fs_n.max():.3e} mean={Fs_n.mean():.3e}   "
           f"Fv: max={Fv_n.max():.3e} mean={Fv_n.mean():.3e}")

    # Persist artifacts
    if (step_idx % SAVE_EVERY) == 0:
        write_mesh_topology_csvs(xyz2, faces, OUTPUT_DIR, step_idx)

        tri_csv = OUTPUT_DIR / f"iter_{step_idx:04d}_tri_stats.csv"
        with open(tri_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["triangle_id","area_PL","signed_volume_about_centroid",
                        "Ax","Ay","Az","Bx","By","Bz","Cx","Cy","Cz"])
            r0 = xyz2.mean(0)
            for tid, (i, j, k) in enumerate(faces):
                a, b, c = xyz2[i], xyz2[j], xyz2[k]
                area_pl = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
                vol  = np.dot(a - r0, np.cross(b - r0, c - r0)) / 6
                w.writerow([tid, area_pl, vol, *a, *b, *c])

        np.savetxt(OUTPUT_DIR / f"iter_{step_idx:04d}_coords.csv", xyz2, delimiter=",", fmt="%.8e")
        with open(OUTPUT_DIR / f"iter_{step_idx:04d}_displacement.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["dx","dy","dz","|dx|"])
            for i in range(disp.shape[0]):
                w.writerow([disp[i,0], disp[i,1], disp[i,2], np.linalg.norm(disp[i])])

        mesh = meshio.Mesh(points=xyz2, cells=[("triangle", faces.astype(np.int32))])
        with silence_all():
            meshio.write(OUTPUT_DIR / f"surface_iter_{step_idx:04d}.msh", mesh, file_format="gmsh22", binary=False)

        # -------------------- per-iter surface PNG (UPDATED) --------------------
        # (single figure, using your coll.set_edgecolor + coll.set_facecolor)
        tris_plot = [xyz2[list(t)] for t in faces]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        coll = Poly3DCollection(tris_plot, linewidths=0.3)
        coll.set_edgecolor(PLOT_EDGE_COLOR)
        coll.set_facecolor(PLOT_FACE_COLOR)
        ax.add_collection3d(coll)

        ax.set_box_aspect((1, 1, 1))
        mm = 1.0e-3
        ax.set_xlim(-0.6 * mm,  0.6 * mm)
        ax.set_ylim(-0.6 * mm,  0.6 * mm)
        ax.set_zlim(-0.6 * mm,  0.6 * mm)

        ax.grid(False)
        ax.set_axis_off()
        plt.tight_layout()
        with silence_all():
            fig.savefig(OUTPUT_DIR / f"org_surface_iter_{step_idx:04d}.png", dpi=150)
        plt.close(fig)
        # ----------------------------------------------------------------------

        vlog = OUTPUT_DIR / f"iter_{step_idx:04d}_volume_area.csv"
        with open(vlog, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter","V_PL_tet","V_corr(from CSV)","V_total(=V_tet+V_corr)",
                        "A_PL","A_flat_sum","A_curved_sum(max(Ac,Af))",
                        "rho_mean","P_abs_mean","dP_mean","dt"])
            w.writerow([step_idx, f"{V_tet:.16e}", f"{V_corr:.16e}", f"{V_total:.16e}",
                        f"{A_PL:.16e}", f"{sum_A_flat:.16e}", f"{sum_A_curved:.16e}",
                        f"{rho_mean:.6e}", f"{P_mean:.6e}", f"{dP_mean_print:.6e}", f"{dt:.6e}"])

        # ---- per-vertex dump
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
                "V_dual","A_dual",
                "U_x","U_y","U_z","|U|",
                "dx","dy","dz","|dx|",
                "dt",
                "m_i",
                "a_x","a_y","a_z","|a|"
            ])
            U_mag = np.linalg.norm(U, axis=1)
            a_mag = np.linalg.norm(a_vec, axis=1)
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
                    Vi_true[i], Astar_safe[i],
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
        float(np.mean(rho_i)),
        P_inside_mean,
        dP_mean,
        surface_tension_mean,
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
for k in range(N_STEPS):
    (
        A_PL,
        A_curved,
        V_PL,
        V_total,
        rho_mean,
        Pbar,
        dPbar,
        taubar,
        dt,
        ave_dist_nnpt,
        ave_dist_O,
    ) = step_forces_dualvolume(HC, faces0, step_idx=k)
    iters.append(k)
    Apl_list.append(A_PL)
    Vtotal_list.append(V_total)
    rhobar_list.append(rho_mean)
    Pbar_list.append(Pbar)
    dPbar_list.append(dPbar)
    taubar_list.append(taubar)
    dt_list.append(dt)
    ave_dist_nnpt_list.append(ave_dist_nnpt)
    ave_dist_O_list.append(ave_dist_O)

# --------------------- save per-step summary ---------------------
summary_csv = OUTPUT_DIR / "_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "iter", "A_PL_total", "A_target", "V_total", "rho_mean", "mass_total",
        "P_inside_mean", "dP_mean", "surface_tension_mean", "P_outside_mean", "dt",
        "rel.error.V%", "rel.error.A%", "ave_dist_nnpt", "ave_dist_O"
    ])
    for k, A_PL, V_total, rbar, Pbar, dPbar, taubar, dt, ave_dist_nnpt, ave_dist_O in zip(
        iters, Apl_list, Vtotal_list, rhobar_list, Pbar_list, dPbar_list, taubar_list, dt_list,
        ave_dist_nnpt_list, ave_dist_O_list
    ):
        rel_error_V = ((V_total - REF_VOLUME) / max(REF_VOLUME, 1e-300)) * 100.0
        rel_error_A = ((A_PL - A_TARGET) / max(A_TARGET, 1e-300)) * 100.0
        w.writerow([
            k, f"{A_PL:.16e}", f"{A_TARGET:.16e}", f"{V_total:.16e}",
            f"{rbar:.6e}", f"{MASS_TOTAL:.16e}",
            f"{Pbar:.6e}", f"{dPbar:.6e}", f"{taubar:.6e}", f"{P0_ATM:.6e}", f"{dt:.6e}",
            f"{rel_error_V:.6e}", f"{rel_error_A:.6e}",
            f"{ave_dist_nnpt:.16e}", f"{ave_dist_O:.16e}"
        ])

# --------------------- plots → PNG (ONE ROW) ---------------------
plt.close('all')

import pandas as pd
import matplotlib.ticker as mticker

df = pd.read_csv(summary_csv)

iters_csv = df["iter"].to_numpy()
rel_err_A = df["rel.error.A%"].to_numpy()
rel_err_V = df["rel.error.V%"].to_numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(iters_csv, rel_err_A)
ax[0].set_title("rel.error.A%")
ax[0].set_xlabel("iter")
ax[0].set_ylabel("%")
ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax[0].grid(True)

ax[1].plot(iters_csv, rel_err_V)
ax[1].set_title("rel.error.V%")
ax[1].set_xlabel("iter")
ax[1].set_ylabel("%")
ax[1].grid(True)
plt.tight_layout()
with silence_all():
    plt.savefig(OUTPUT_DIR / "manifold_cotan_progress.png", dpi=150)

# --------------------- final view ---------------------
if INTERACTIVE:
    xyz_final, _ = _vertex_array(HC)
    tris = [xyz_final[list(t)] for t in faces0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    coll = Poly3DCollection(tris, linewidths=0.3)
    coll.set_edgecolor(PLOT_EDGE_COLOR)
    coll.set_facecolor(PLOT_FACE_COLOR)
    ax.add_collection3d(coll)

    ax.set_box_aspect((1, 1, 1))
    mins = xyz_final.min(axis=0); maxs = xyz_final.max(axis=0)
    ctr  = 0.5 * (mins + maxs); half = 0.55 * float(np.max(maxs - mins)); half = half if (np.isfinite(half) and half>0) else 1e-3
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half); ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.grid(False); plt.tight_layout(); plt.show()

# =====================================================================
