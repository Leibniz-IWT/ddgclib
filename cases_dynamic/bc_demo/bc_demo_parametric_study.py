"""
Parametric Study: Periodic Inlet + Open Outlet BC Balance
with per-case animations (GIF + MP4).

Sweeps n_refine and n_steps.  For each case produces:
  fig/parametric/nr{N}_ns{S}/
      _frames/        PNG frames (one per FRAME_EVERY steps)
      animation.gif
      animation.mp4
  fig/
      bc_parametric_balance.png
      bc_parametric_rates.png
      bc_parametric_surplus.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, QhullError

# ============================================================
# Global typography & quality settings  (single place to tune)
# ============================================================
FRAME_DPI        = 200      # PNG frame resolution (↑ = sharper video)
FRAME_FIG_W      = 20       # frame figure width  [inches]
FRAME_FIG_H      = 6        # frame figure height [inches]
SUMMARY_DPI      = 180      # summary plot resolution
MP4_CRF          = 18       # libx264 CRF: 0=lossless, 18=high quality, 23=default
MP4_PRESET       = 'slow'   # encoding effort: slow = better compression at same quality
MP4_FPS          = 15       # frames per second for MP4
GIF_DURATION_MS  = 80       # ms per GIF frame

matplotlib.rcParams.update({
    'font.size':         20,
    'axes.titlesize':    22,
    'axes.labelsize':    20,
    'xtick.labelsize':   18,
    'ytick.labelsize':   18,
    'legend.fontsize':   18,
    'figure.titlesize':  24,
    'lines.linewidth':   3.0,
    'axes.linewidth':    1.8,
    'grid.linewidth':    1.2,
    'scatter.edgecolors': 'none',
})

from hyperct import Complex
from ddgclib._boundary_conditions import (
    PeriodicInletBC,
    OutletDeleteBC,
    PositionalNoSlipWallBC,
    identify_boundary_vertices,
)
from ddgclib.initial_conditions import (
    CompositeIC,
    UniformVelocity,
    UniformMass,
)

# ============================================================
# Fixed parameters
# ============================================================
H         = 1.0
L_period  = 1.0
L_domain  = 3.0
U         = 0.5
dt        = 0.05
d         = 2

TRANSIT_TIME  = L_domain / U
TRANSIT_STEPS = int(TRANSIT_TIME / dt)

wall_tol = 1e-10

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIG  = os.path.join(_HERE, 'fig')
os.makedirs(_FIG, exist_ok=True)

# ============================================================
# Sweep parameters
# ============================================================
N_REFINE_LIST = [0, 1, 2]
N_STEPS_LIST  = [
    TRANSIT_STEPS,        # 120 steps, 6 s
    2 * TRANSIT_STEPS,    # 240 steps, 12 s
    3 * TRANSIT_STEPS,    # 360 steps, 18 s
]
FRAME_EVERY = 2   # render a frame every N simulation steps


# ============================================================
# Shared helpers
# ============================================================

def wall_criterion(v):
    return abs(v.x_a[1]) < wall_tol or abs(v.x_a[1] - H) < wall_tol


def get_edges_delaunay(mesh):
    """Edges via Delaunay so newly-injected vertices are always connected."""
    verts = list(mesh.V)
    if len(verts) < 3:
        return []
    coords = np.array([v.x_a[:2] for v in verts])
    _, uid = np.unique(coords, axis=0, return_index=True)
    cu = coords[uid]
    if len(cu) < 3:
        return []
    try:
        tri = Delaunay(cu)
    except QhullError:
        return []
    seen, segs = set(), []
    for s in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(s[i]), int(s[j])
                k = (min(a, b), max(a, b))
                if k not in seen:
                    seen.add(k)
                    segs.append([cu[a], cu[b]])
    return segs


def compile_gif(frame_paths, out_path, duration_ms=GIF_DURATION_MS):
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in frame_paths]
        if imgs:
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                         duration=duration_ms, loop=0)
            print(f"    GIF  -> {out_path}")
        return True
    except ImportError:
        print("    PIL not available -- GIF skipped")
        return False


def compile_mp4(frame_paths, out_path, fps=MP4_FPS):
    """Try cv2 first (high quality), then ffmpeg subprocess."""
    # --- cv2 ---
    try:
        import cv2
        if not frame_paths:
            return False
        img0 = cv2.imread(frame_paths[0])
        if img0 is None:
            raise ValueError("cv2 could not read first frame")
        h, w = img0.shape[:2]
        # Use MJPG inside AVI as cv2 mp4v often has quality limits;
        # prefer ffmpeg path for best quality when available.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for p in frame_paths:
            img = cv2.imread(p)
            if img is not None:
                vw.write(img)
        vw.release()
        print(f"    MP4  -> {out_path}  (cv2, fps={fps})")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"    cv2 MP4 failed ({e}), trying ffmpeg ...")

    # --- ffmpeg (preferred: libx264, high quality) ---
    try:
        import subprocess, shutil
        if shutil.which('ffmpeg') is None:
            print("    ffmpeg not found -- MP4 skipped")
            return False
        list_path = out_path + '_filelist.txt'
        with open(list_path, 'w') as f:
            for p in frame_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
                f.write(f"duration {1.0/fps:.6f}\n")
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', list_path,
            # Ensure even dimensions (required by yuv420p)
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-crf', str(MP4_CRF),          # quality: lower = better
            '-preset', MP4_PRESET,         # encoding effort
            '-pix_fmt', 'yuv420p',         # broad compatibility
            '-r', str(fps),
            '-movflags', '+faststart',     # web-friendly
            out_path,
        ]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(list_path)
        print(f"    MP4  -> {out_path}  (ffmpeg crf={MP4_CRF}, preset={MP4_PRESET}, fps={fps})")
        return True
    except Exception as e:
        print(f"    MP4 failed: {e}")
        return False


# ============================================================
# Per-case frame renderer  (mirrors bc_demo.plot_frame exactly)
# ============================================================

def plot_frame(step, t, HC, ghost_mesh, fig_dir,
               cum_injected, cum_deleted, n_refine, n_steps):
    fig, (ax_main, ax_ghost) = plt.subplots(
        1, 2, figsize=(FRAME_FIG_W, FRAME_FIG_H),
        gridspec_kw={'width_ratios': [3, 1.5]}
    )

    # --- Main domain ---
    verts_main = list(HC.V)
    coords = (np.array([v.x_a[:2] for v in verts_main])
              if verts_main else np.empty((0, 2)))

    if len(coords) > 0:
        sc = ax_main.scatter(coords[:, 0], coords[:, 1],
                             c=coords[:, 0], cmap='coolwarm', s=40, zorder=3,
                             vmin=-L_period, vmax=L_domain + L_period)
        cbar = fig.colorbar(sc, ax=ax_main, fraction=0.03, pad=0.02)
        cbar.set_label('x position', fontsize=12)
        cbar.ax.tick_params(labelsize=11)

    edges = get_edges_delaunay(HC)
    if edges:
        lc = LineCollection(edges, colors='gray', linewidths=0.5, alpha=0.4)
        ax_main.add_collection(lc)

    ax_main.axvline(x=0.0,      color='blue', lw=2.5, ls='--', label='Inlet (x=0)')
    ax_main.axvline(x=L_domain, color='red',  lw=2.5, ls='--', label='Outlet (x=L)')

    wpts = [(v.x_a[0], v.x_a[1]) for v in HC.V if wall_criterion(v)]
    if wpts:
        wx, wy = zip(*wpts)
        ax_main.scatter(wx, wy, c='green', s=50, marker='s', zorder=4, label='Wall')

    n_int  = sum(1 for v in HC.V if not wall_criterion(v))
    n_wall = len(verts_main) - n_int
    net    = cum_injected - cum_deleted

    ax_main.set_xlim(-0.3, L_domain + 0.3)
    ax_main.set_ylim(-0.2, H + 0.2)
    ax_main.set_xlabel('x', fontsize=14)
    ax_main.set_ylabel('y', fontsize=14)
    ax_main.tick_params(labelsize=12)
    ax_main.set_title(
        f'n_refine={n_refine},  n_steps={n_steps}  |  step {step:04d},  t = {t:.3f} s\n'
        f'total={len(verts_main)}  (interior={n_int}, wall={n_wall})   '
        f'$\\Sigma$inj={cum_injected}   $\\Sigma$del={cum_deleted}   net={net:+d}',
        fontsize=13, pad=8
    )
    ax_main.set_aspect('equal')
    ax_main.legend(loc='upper right', fontsize=12, framealpha=0.85)
    ax_main.grid(True, alpha=0.25)

    # --- Ghost mesh ---
    ghost_verts = list(ghost_mesh.V)
    if ghost_verts:
        gc = np.array([v.x_a[:2] for v in ghost_verts])
        ax_ghost.scatter(gc[:, 0], gc[:, 1], c='orange', s=40, zorder=3,
                         label='Ghost vertices')
        ges = get_edges_delaunay(ghost_mesh)
        if ges:
            lc_g = LineCollection(ges, colors='darkorange', lw=0.8, alpha=0.6)
            ax_ghost.add_collection(lc_g)

    ax_ghost.axvline(x=0.0, color='blue', lw=2.5, ls='--', label='Inlet (x=0)')
    ax_ghost.set_xlim(-L_period - 0.3, 0.5)
    ax_ghost.set_ylim(-0.2, H + 0.2)
    ax_ghost.set_xlabel('x', fontsize=14)
    ax_ghost.set_ylabel('y', fontsize=14)
    ax_ghost.tick_params(labelsize=12)
    ax_ghost.set_title(f'Ghost Mesh\n{len(ghost_verts)} vertices remaining',
                       fontsize=13, pad=8)
    ax_ghost.set_aspect('equal')
    ax_ghost.legend(loc='upper right', fontsize=12, framealpha=0.85)
    ax_ghost.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(fig_dir, f'frame_{step:04d}.png')
    plt.savefig(path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# ============================================================
# BC application (wall-safe outlet + inlet wall-purge + invariants)
# ============================================================

def _apply_bcs(HC, bV, inlet_bc, wall_bc, dt, cum_inj, cum_del):
    wall_ids_pre = {id(v) for v in HC.V if wall_criterion(v)}

    # 1. Wall BC
    wall_bc.apply(HC, dt)

    # 2. Outlet -- exclude wall vertices (invariant I1)
    n0 = sum(1 for _ in HC.V)
    for v in [v for v in list(HC.V)
              if v.x_a[0] >= L_domain and not wall_criterion(v)]:
        bV.discard(v)
        HC.V.remove(v)
    n_del = max(0, n0 - sum(1 for _ in HC.V))
    cum_del += n_del

    # 3. Inlet -- purge injected wall vertices (invariant I2)
    existing_ids = {id(v) for v in HC.V}
    n2 = sum(1 for _ in HC.V)
    inlet_bc.apply(HC, dt)
    n_purged = 0
    for v in list(HC.V):
        if id(v) not in existing_ids and wall_criterion(v):
            HC.V.remove(v)
            bV.discard(v)
            n_purged += 1
    n_inj = max(0, sum(1 for _ in HC.V) - n2)
    cum_inj += n_inj

    # Invariant I1: wall count must not decrease
    wall_ids_post = {id(v) for v in HC.V if wall_criterion(v)}
    if len(wall_ids_post) < len(wall_ids_pre):
        raise RuntimeError(
            f"I1 violated: {len(wall_ids_pre) - len(wall_ids_post)} "
            f"wall vertices lost"
        )

    diag = dict(
        n_deleted=n_del, n_injected=n_inj, n_wall_purged=n_purged,
        n_wall=len(wall_ids_post),
        n_interior=sum(1 for _ in HC.V) - len(wall_ids_post),
    )
    return cum_inj, cum_del, diag


# ============================================================
# Build tiled main mesh from unit mesh copies
# ============================================================

def build_tiled_mesh(unit_mesh, n_tiles):
    HC  = Complex(d, domain=[(0.0, L_domain), (0.0, H)])
    seen = {}

    def _add(pos_tuple, src_v):
        key = tuple(round(x, 12) for x in pos_tuple)
        if key not in seen:
            v = HC.V[pos_tuple]
            for f in ['u', 'p', 'm']:
                val = getattr(src_v, f, None)
                if val is not None:
                    setattr(v, f, val.copy() if isinstance(val, np.ndarray) else val)
            seen[key] = v

    for k in range(n_tiles):
        for uv in unit_mesh.V:
            pos = uv.x_a.copy()
            pos[0] += k * L_period
            _add(tuple(pos), uv)

    bV = identify_boundary_vertices(HC, lambda v: (
        abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L_domain) < 1e-14 or
        abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
    ))
    for v in HC.V:
        v.boundary = v in bV
    return HC, bV


# ============================================================
# Run one (n_refine, n_steps) case
# ============================================================

def run_case(n_refine, n_steps, case_fig_dir):
    os.makedirs(case_fig_dir, exist_ok=True)
    frame_dir = os.path.join(case_fig_dir, '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    # Build unit mesh
    unit_mesh = Complex(d, domain=[(0.0, L_period), (0.0, H)])
    unit_mesh.triangulate()
    for _ in range(n_refine):
        unit_mesh.refine_all()
    unit_bV = identify_boundary_vertices(unit_mesh, lambda v: (
        abs(v.x_a[0]) < 1e-14 or abs(v.x_a[0] - L_period) < 1e-14 or
        abs(v.x_a[1]) < 1e-14 or abs(v.x_a[1] - H) < 1e-14
    ))
    CompositeIC(
        UniformVelocity(u_vec=np.array([U, 0.0])),
        UniformMass(total_volume=L_period * H, rho=1.0),
    ).apply(unit_mesh, unit_bV)
    for v in unit_mesh.V:
        v.p = 0.0

    n_unit   = sum(1 for _ in unit_mesh.V)
    n_wall_u = sum(1 for v in unit_mesh.V if wall_criterion(v))
    n_int_u  = n_unit - n_wall_u

    # Build tiled main mesh
    n_tiles = int(round(L_domain / L_period))
    HC, bV  = build_tiled_mesh(unit_mesh, n_tiles)
    n_main  = sum(1 for _ in HC.V)

    # BCs
    inlet_bc = PeriodicInletBC(
        unit_mesh=unit_mesh, velocity=U, axis=0,
        inlet_pos=0.0, cdist=1e-10,
        fields=['u', 'p', 'm'], period=L_period,
    )
    wall_bc = PositionalNoSlipWallBC(criterion_fn=wall_criterion, dim=d, bV=bV)

    # Time-series storage
    ts = dict(
        t=[], n_verts=[], n_interior=[], n_wall=[],
        cum_injected=[], cum_deleted=[],
        n_injected=[], n_deleted=[], n_wall_purged=[],
        n_main=n_main, n_unit=n_unit,
        n_wall_unit=n_wall_u, n_int_unit=n_int_u,
        n_refine=n_refine, n_steps=n_steps,
    )

    cum_inj = cum_del = 0
    frame_paths = []

    # Frame t=0
    fp = plot_frame(0, 0.0, HC, inlet_bc.ghost, frame_dir,
                    cum_inj, cum_del, n_refine, n_steps)
    frame_paths.append(fp)

    for step in range(1, n_steps + 1):
        # Advect interior vertices only
        for v in list(HC.V):
            if not wall_criterion(v):
                pos = v.x_a.copy()
                pos[0] += U * dt
                if v in bV:
                    bV.remove(v)
                    HC.V.move(v, tuple(pos))
                    bV.add(v)
                else:
                    HC.V.move(v, tuple(pos))

        cum_inj, cum_del, diag = _apply_bcs(
            HC, bV, inlet_bc, wall_bc, dt, cum_inj, cum_del
        )
        t = step * dt

        ts['t'].append(t)
        ts['n_verts'].append(sum(1 for _ in HC.V))
        ts['n_interior'].append(diag['n_interior'])
        ts['n_wall'].append(diag['n_wall'])
        ts['cum_injected'].append(cum_inj)
        ts['cum_deleted'].append(cum_del)
        ts['n_injected'].append(diag['n_injected'])
        ts['n_deleted'].append(diag['n_deleted'])
        ts['n_wall_purged'].append(diag['n_wall_purged'])

        if step % FRAME_EVERY == 0:
            fp = plot_frame(step, t, HC, inlet_bc.ghost, frame_dir,
                            cum_inj, cum_del, n_refine, n_steps)
            frame_paths.append(fp)

    # Final frame (avoid duplicate if n_steps % FRAME_EVERY == 0)
    t_final = n_steps * dt
    fp = plot_frame(n_steps, t_final, HC, inlet_bc.ghost, frame_dir,
                    cum_inj, cum_del, n_refine, n_steps)
    if fp not in frame_paths:
        frame_paths.append(fp)

    ts['final_verts']   = ts['n_verts'][-1]
    ts['net']           = cum_inj - cum_del
    ts['cum_inj_final'] = cum_inj
    ts['cum_del_final'] = cum_del
    ts['frame_paths']   = frame_paths

    gif_name = f"bc_balance_nr{n_refine}_ns{n_steps}.gif"
    mp4_name = f"bc_balance_nr{n_refine}_ns{n_steps}.mp4"

    compile_gif(frame_paths, os.path.join(case_fig_dir, gif_name))
    compile_mp4(frame_paths, os.path.join(case_fig_dir, mp4_name))

    return ts


# ============================================================
# Run all cases
# ============================================================
results = {}

print(f"\n{'='*72}")
print(f"{'n_ref':>6}  {'n_steps':>8}  {'t_max':>6}  "
      f"{'n_unit':>7}  {'n_int':>6}  "
      f"{'Sinj':>6}  {'Sdel':>6}  {'net':>6}  {'final_V':>8}")
print(f"{'-'*72}")

for nr in N_REFINE_LIST:
    for ns in N_STEPS_LIST:
        label    = f"nr{nr}_ns{ns}"
        case_dir = os.path.join(_FIG, 'parametric', label)
        print(f"\n  [n_refine={nr}, n_steps={ns}]  -> {case_dir}")
        ts = run_case(nr, ns, case_dir)
        results[(nr, ns)] = ts
        net_str = f"net={ts['net']:+d}"
        ok      = "OK" if ts['net'] == 0 else "!! IMBALANCE"
        print(f"  {nr:>6}  {ns:>8}  {ns*dt:>6.1f}  "
              f"{ts['n_unit']:>7}  {ts['n_int_unit']:>6}  "
              f"{ts['cum_inj_final']:>6}  {ts['cum_del_final']:>6}  "
              f"{ts['net']:>+6}  {ts['final_verts']:>8}  {ok}")

print(f"\n{'='*72}")

# ============================================================
# Theoretical analysis
# ============================================================
print(f"\nTheoretical analysis:")
print(f"  Transit time        = L/U = {L_domain}/{U} = {TRANSIT_TIME:.1f} s "
      f"({TRANSIT_STEPS} steps)")
print(f"  Period crossing time = L_period/U = {L_period}/{U} = "
      f"{L_period/U:.1f} s\n")
for nr in N_REFINE_LIST:
    ts0 = results[(nr, N_STEPS_LIST[0])]
    print(f"  n_refine={nr}:  n_unit={ts0['n_unit']}, "
          f"n_wall={ts0['n_wall_unit']}, "
          f"n_interior/period={ts0['n_int_unit']}")
    for ns in N_STEPS_LIST:
        ts  = results[(nr, ns)]
        ok  = "OK" if ts['net'] == 0 else f"net={ts['net']:+d} !!"
        print(f"    n_steps={ns:4d} ({ns*dt:5.1f}s):  "
              f"Sinj={ts['cum_inj_final']:5d}  "
              f"Sdel={ts['cum_del_final']:5d}  {ok}")
    print()

# ============================================================
# Summary plots
# ============================================================

# --- Plot 1: vertex count vs time ---
fig, axes = plt.subplots(
    len(N_REFINE_LIST), len(N_STEPS_LIST),
    figsize=(7 * len(N_STEPS_LIST), 5 * len(N_REFINE_LIST)),
    sharey='row'
)
for ri, nr in enumerate(N_REFINE_LIST):
    for si, ns in enumerate(N_STEPS_LIST):
        ax = axes[ri][si]
        ts = results[(nr, ns)]
        t_arr = np.array(ts['t'])
        ax.plot(t_arr, ts['n_verts'],    'k-',  lw=2.0, label='total')
        ax.plot(t_arr, ts['n_interior'], 'b-',  lw=1.5, label='interior')
        ax.plot(t_arr, ts['n_wall'],     'g--', lw=1.5, label='wall')
        ax.axhline(ts['n_main'], color='steelblue', ls=':', lw=1.5,
                   label=f"n_main ({ts['n_main']})")
        ax.axhline(ts['n_unit'], color='orange',    ls=':', lw=1.5,
                   label=f"n_unit ({ts['n_unit']})")
        for k in range(1, 7):
            xt = k * TRANSIT_TIME
            if xt <= t_arr[-1]:
                ax.axvline(xt, color='red', ls=':', lw=1.0, alpha=0.5)
        ok = 'OK' if ts['net'] == 0 else f"net={ts['net']:+d}!!"
        ax.set_title(f"n_refine={nr},  n_steps={ns}  [{ok}]", fontsize=14, pad=6)
        ax.set_xlabel('t  [s]', fontsize=13)
        ax.set_ylabel('vertex count', fontsize=13)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=10, loc='upper left', ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.3)
plt.suptitle('Vertex count vs time\n(red dotted lines = transit time multiples)',
             fontsize=17, y=1.01)
plt.tight_layout()
p1 = os.path.join(_FIG, 'bc_parametric_balance.png')
plt.savefig(p1, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p1}")

# --- Plot 2: cumulative injected/deleted ---
fig, axes = plt.subplots(
    len(N_REFINE_LIST), len(N_STEPS_LIST),
    figsize=(7 * len(N_STEPS_LIST), 5 * len(N_REFINE_LIST)),
    sharey='row'
)
for ri, nr in enumerate(N_REFINE_LIST):
    for si, ns in enumerate(N_STEPS_LIST):
        ax  = axes[ri][si]
        ts  = results[(nr, ns)]
        t_arr = np.array(ts['t'])
        inj   = np.array(ts['cum_injected'])
        delt  = np.array(ts['cum_deleted'])
        ax.plot(t_arr, inj,         'g-',  lw=2.0, label='$\\Sigma$ injected')
        ax.plot(t_arr, delt,        'r-',  lw=2.0, label='$\\Sigma$ deleted')
        ax.plot(t_arr, inj - delt,  'k--', lw=1.5, label='net surplus')
        ax.axhline(0, color='gray', lw=1.0)
        for k in range(1, 7):
            xt = k * TRANSIT_TIME
            if xt <= t_arr[-1]:
                ax.axvline(xt, color='red', ls=':', lw=1.0, alpha=0.5)
        ax.set_title(f"n_refine={nr},  n_steps={ns}", fontsize=14, pad=6)
        ax.set_xlabel('t  [s]', fontsize=13)
        ax.set_ylabel('cumulative count', fontsize=13)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.3)
plt.suptitle('Cumulative injected vs deleted\n'
             '(balance achieved when $\\Sigma$inj and $\\Sigma$del are parallel)',
             fontsize=17, y=1.01)
plt.tight_layout()
p2 = os.path.join(_FIG, 'bc_parametric_rates.png')
plt.savefig(p2, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p2}")

# --- Plot 3: normalised surplus ---
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', '^']
for ri, nr in enumerate(N_REFINE_LIST):
    ts0   = results[(nr, N_STEPS_LIST[0])]
    n_int = max(ts0['n_int_unit'], 1)
    t_maxs = [ns * dt for ns in N_STEPS_LIST]
    nets   = [results[(nr, ns)]['net'] / n_int for ns in N_STEPS_LIST]
    ax.plot(t_maxs, nets,
            color=colors[ri], marker=markers[ri], lw=2.0, ms=10,
            label=f"n_refine={nr}  (n_int/period = {ts0['n_int_unit']})")
ax.axhline(0,  color='k',    lw=1.2, ls='--', label='perfect balance  (net = 0)')
ax.axhline(1,  color='gray', lw=0.8, ls=':', alpha=0.7)
ax.axhline(-1, color='gray', lw=0.8, ls=':', alpha=0.7)
for k in range(1, 4):
    ax.axvline(k * TRANSIT_TIME, color='red', ls=':', lw=1.0, alpha=0.55)
    ax.text(k * TRANSIT_TIME + 0.15, ax.get_ylim()[0] if ax.get_ylim()[0] > -2 else -1.8,
            f'{k}×transit', color='red', fontsize=11, va='bottom')
ax.set_xlabel('Simulation length  $t_{max}$  [s]', fontsize=14)
ax.set_ylabel('Net surplus / $n_{int,period}$', fontsize=14)
ax.set_title('Normalised vertex surplus vs simulation length\n'
             '(all curves at $y=0$ confirms BC balance is robust across refinements)',
             fontsize=15)
ax.tick_params(labelsize=13)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
p3 = os.path.join(_FIG, 'bc_parametric_surplus.png')
plt.savefig(p3, dpi=SUMMARY_DPI, bbox_inches='tight')
plt.close()
print(f"Saved {p3}")

print("\nDone.")