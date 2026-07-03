"""Section 3.4 paraboloid-cap volume comparison.

Run from the project root:

    python 02_Figures/Figure_3-8_VolumeOperatorComparison/paraboloid/paraboloid_all_methods.py

The script recomputes the paraboloid sweep and writes
``paraboloid/paraboloid_all_methods_result.csv``.  The equations, literature
references, and callable functions are kept directly in the top-level
author-year files. This is the model-matched test for
Evrard et al. (2023):

* Evrard-type paraboloid: ``run_evrard_forward_operator`` calls
  ``evrard_2023.paraboloid_forward_volume`` for the forward zeroth-moment
  problem.  This corresponds to Evrard et al. (2023), where Eqs. (2.4)-(2.6)
  define the moments, Eqs. (3.1)-(3.16) apply the divergence theorem to a
  paraboloid clipping region, and Eqs. (3.23)-(3.36) give the closed-form
  contributions.
* THINC/QQ: the same exact paraboloid surface is integrated by finite Gaussian
  quadrature, following Xie and Xiao (2017), Eq. (9), Eqs. (20)-(23), and
  Appendix B.
* Strobl: sphere/hexahedron overlap from Strobl et al. (2016), Eq. (2) and
  Algorithm 1, used here as a non-matching sphere-special comparison.
"""

import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
COMMON = BUNDLE_ROOT / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from bootstrap import ensure_runtime  # noqa: E402
from method_timing import new_timings, perf_now, print_timing_summary, timed  # noqa: E402

ensure_runtime()

import meshio
import numpy as np
import overlap

import evrard_2023
import paraboloid_mesh as paraboloid_geometry
import plic_pl_1998_1999
import present_quadric_patch_2026
import strobl_2016
import table31_reference
import xie_xiao_2017

CURVED_VOLUME = BUNDLE_ROOT / "_curved_volume.py"
FOLDER_MESH = BUNDLE_ROOT / "msh" / "snapped_paraboloid_ascii.msh"
OUT_DIR = Path(__file__).with_name("_work")
STYLE = BUNDLE_ROOT / "neatplot-main" / "standard.mplstyle"

TOP_Z = 2.0
TOP_RADIUS = math.sqrt(TOP_Z)
EXACT_VOLUME = 2.0 * math.pi
PRESENT_COEFFS_KWARGS = {"plane_rel_tol": 3e-4}
QUADRIC_COEFFS = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
PPIC_GRAPH_AXIS = 2
PPIC_GRAPH_COEFFS = (1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
STROBL_HEXGRID_N = 4
EVRARD_BACKGROUND_LEVELS = (
    (12, 3, 4),
    (18, 5, 8),
    (28, 8, 16),
    (84, 24, 48),
)


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def orient_positive_volume(points, faces):
    if signed_pl_volume(points, faces) < 0:
        faces = faces[:, [0, 2, 1]]
    return points, faces


def signed_pl_volume(points, faces):
    return sum(
        np.dot(points[a], np.cross(points[b], points[c])) / 6.0
        for a, b, c in faces
    )


def pl_volume(points, faces):
    return abs(signed_pl_volume(points, faces))


def is_top_point(point, tol=1e-8):
    return abs(point[2] - TOP_Z) < tol


def is_rim_point(point, tol=1e-7):
    return is_top_point(point, tol=tol) and abs(math.hypot(point[0], point[1]) - TOP_RADIUS) < tol


def is_curved_point(point, tol=1e-8):
    return abs(point[2] - point[0] * point[0] - point[1] * point[1]) < tol


def is_top_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 2], TOP_Z, atol=tol))


def is_paraboloid_wall_face(points, face, tol=1e-8):
    return bool(np.all(np.abs(points[face, 2] - points[face, 0] ** 2 - points[face, 1] ** 2) < tol))


def orient_faces_by_geometry(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if is_top_face(points, face):
            desired = np.array([0.0, 0.0, 1.0])
        elif is_paraboloid_wall_face(points, face):
            x, y, _z = pts.mean(axis=0)
            desired = np.array([2.0 * x, 2.0 * y, -1.0])
        else:
            raise ValueError(f"Could not classify paraboloid boundary face {i}: {pts}")
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def mesh_size(points, faces):
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    lens = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])
    return float(lens.mean())


def tet_mesh_size(points, tets):
    edges = set()
    for tet in tets:
        tet = [int(v) for v in tet]
        for i in range(4):
            for j in range(i + 1, 4):
                edges.add(tuple(sorted((tet[i], tet[j]))))
    lens = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])
    return float(lens.mean())


def add_triangular_prism(points, tets, p0, p1, p2, p3, p4, p5):
    offset = len(points)
    points.extend((p0, p1, p2, p3, p4, p5))
    for tet in ((0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)):
        tets.append(tuple(offset + i for i in tet))


def add_triangular_prism_indices(tets, ids):
    for tet in ((0, 1, 2, 5), (0, 1, 5, 4), (0, 4, 5, 3)):
        tets.append(tuple(ids[i] for i in tet))


def build_evrard_background_tets(level):
    """Independent PPIC/VOF-style background cells for the paraboloid cap.

    The outer polygon circumscribes the radius-sqrt(2) disk, so the exact
    paraboloid volume is not clipped by the artificial side boundary. This is
    deliberately not the present method's surface-patch Vpatch decomposition.
    """
    nseg, nr, nz = EVRARD_BACKGROUND_LEVELS[level]
    z_values = np.linspace(-1.0e-9, TOP_Z, nz + 1)
    outer_radius = TOP_RADIUS / math.cos(math.pi / nseg) * (1.0 + 1.0e-12)
    r_values = np.linspace(0.0, outer_radius, nr + 1)
    points = []
    tets = []

    center_ids = []
    ring_ids = []
    for k, z in enumerate(z_values):
        center_ids.append(len(points))
        points.append((0.0, 0.0, float(z)))
        layer = []
        for i in range(nseg):
            theta = 2.0 * math.pi * i / nseg
            radial_ids = []
            for radius in r_values[1:]:
                radial_ids.append(len(points))
                points.append((float(radius) * math.cos(theta), float(radius) * math.sin(theta), float(z)))
            layer.append(radial_ids)
        ring_ids.append(layer)

    def point_id(k, j, i):
        if j == 0:
            return center_ids[k]
        return ring_ids[k][i % nseg][j - 1]

    for k in range(nz):
        kb = k
        kt = k + 1
        for i in range(nseg):
            ip = (i + 1) % nseg
            for j in range(nr):
                if j == 0:
                    add_triangular_prism_indices(
                        tets,
                        (
                            point_id(kb, 0, i),
                            point_id(kb, 1, i),
                            point_id(kb, 1, ip),
                            point_id(kt, 0, i),
                            point_id(kt, 1, i),
                            point_id(kt, 1, ip),
                        ),
                    )
                else:
                    add_triangular_prism_indices(
                        tets,
                        (
                            point_id(kb, j, i),
                            point_id(kb, j + 1, i),
                            point_id(kb, j + 1, ip),
                            point_id(kt, j, i),
                            point_id(kt, j + 1, i),
                            point_id(kt, j + 1, ip),
                        ),
                    )
                    add_triangular_prism_indices(
                        tets,
                        (
                            point_id(kb, j, i),
                            point_id(kb, j + 1, ip),
                            point_id(kb, j, ip),
                            point_id(kt, j, i),
                            point_id(kt, j + 1, ip),
                            point_id(kt, j, ip),
                        ),
                    )

    return np.asarray(points, dtype=float), np.asarray(tets, dtype=int)


def run_evrard_forward_operator(level):
    points, tets = paraboloid_geometry.generate_evrard_background_tets(level)
    selected_volume = evrard_2023.paraboloid_forward_volume(
        points,
        tets,
        datum=(0.0, 0.0, 0.0),
        frame=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        coefficients=(-1.0, -1.0),
        use_above_region=True,
    )
    return selected_volume, tet_mesh_size(points, tets), len(tets)


def load_msh_triangles(path):
    mesh = meshio.read(path)
    points = mesh.points[:, :3]
    tris = []
    for cb in mesh.cells:
        if cb.type == "triangle":
            tris.append(cb.data)
    if not tris:
        raise ValueError(f"No triangle cells found in {path}")
    faces = np.vstack(tris).astype(int)
    faces = orient_faces_by_geometry(points, faces)
    return orient_positive_volume(points, faces)


def curved_face_count(points, faces):
    return sum(1 for face in faces if is_paraboloid_wall_face(points, face))


def project_wall_point(point):
    x, y, z = point
    z = min(max(float(z), 0.0), TOP_Z)
    target = math.sqrt(z)
    r = math.hypot(x, y)
    if r < 1e-14:
        return np.array([target, 0.0, z], dtype=float)
    return np.array([target * x / r, target * y / r, z], dtype=float)


def project_midpoint(a, b):
    mid = 0.5 * (a + b)
    same_top = is_top_point(a) and is_top_point(b)
    if same_top:
        mid[2] = TOP_Z
        if is_rim_point(a) and is_rim_point(b):
            r = math.hypot(mid[0], mid[1])
            if r > 1e-14:
                mid[0] *= TOP_RADIUS / r
                mid[1] *= TOP_RADIUS / r
        return mid

    if is_curved_point(a) and is_curved_point(b):
        return project_wall_point(mid)

    return mid


def subdivide_projected(points, faces):
    points_list = [p.copy() for p in points]
    midpoint_cache = {}

    def midpoint_id(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key in midpoint_cache:
            return midpoint_cache[key]
        p = project_midpoint(points[key[0]], points[key[1]])
        idx = len(points_list)
        points_list.append(p)
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint_id(a, b)
        bc = midpoint_id(b, c)
        ca = midpoint_id(c, a)
        new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])

    points_new = np.asarray(points_list, dtype=float)
    faces_new = orient_faces_by_geometry(points_new, np.asarray(new_faces, dtype=int))
    return orient_positive_volume(points_new, faces_new)


def fit_order(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = err > 1e-15
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)[0])


def order_text(h, err):
    err = np.asarray(err, dtype=float)
    if np.nanmax(err) < 1e-10:
        return "round-off"
    p = fit_order(h, err)
    if math.isnan(p):
        return "round-off"
    return f"p={p:.2f}"


def local_orders(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    out = []
    for i in range(1, len(h)):
        if err[i - 1] <= 1e-15 or err[i] <= 1e-15:
            out.append(float("nan"))
        else:
            out.append(float(np.log(err[i] / err[i - 1]) / np.log(h[i] / h[i - 1])))
    return np.asarray(out)


def run_present_method(cv, points, faces, workdir, msh_path=None):
    return present_quadric_patch_2026.manuscript_curved_volume(
        points,
        faces,
        workdir,
        msh_path=msh_path,
        coeffs_kwargs=PRESENT_COEFFS_KWARGS,
    )


def compute_row(cv, points, faces, workdir, level, label, timings, msh_path=None):
    vpl = timed(timings, "plic_pl", plic_pl_1998_1999.closed_surface_volume, points, faces)
    v_model_2023, hevrard, nevrard_tets = timed(
        timings,
        "evrard_type_paraboloid",
        run_evrard_forward_operator,
        level,
    )
    v_model_2017 = timed(
        timings,
        "thinc_qq",
        xie_xiao_2017.gaussian_quadric_volume,
        points, faces, is_paraboloid_wall_face, QUADRIC_COEFFS
    )
    v_model_2016 = timed(timings, "strobl_sphere_overlap", strobl_2016.sphere_hexgrid_volume, points)

    # The 2023 value comes from the Evrard paraboloid forward operator, which
    # solves the model-matched zeroth-moment problem for this panel.
    Vpatch_2023 = v_model_2023 - vpl
    Vpatch_2017 = v_model_2017 - vpl
    Vpatch_2016 = v_model_2016 - vpl
    vevrard = vpl + Vpatch_2023
    vthincqq = vpl + Vpatch_2017
    vstrobl = vpl + Vpatch_2016
    vcv = timed(timings, "present_quadric_patch", run_present_method, cv, points, faces, workdir, msh_path=msh_path)
    row = {
        "level": level,
        "label": label,
        "points": len(points),
        "faces": len(faces),
        "curved_faces": curved_face_count(points, faces),
        "hmean": mesh_size(points, faces),
        "evrard_hmean": hevrard,
        "evrard_tets": nevrard_tets,
        "pl": vpl,
        "evrard": vevrard,
        "thincqq": vthincqq,
        "strobl": vstrobl,
        "Vpatch_2023": Vpatch_2023,
        "Vpatch_2017": Vpatch_2017,
        "Vpatch_2016": Vpatch_2016,
        "present": vcv,
        "pl_err": abs(vpl - EXACT_VOLUME) / EXACT_VOLUME,
        "evrard_err": abs(vevrard - EXACT_VOLUME) / EXACT_VOLUME,
        "thincqq_err": abs(vthincqq - EXACT_VOLUME) / EXACT_VOLUME,
        "strobl_err": abs(vstrobl - EXACT_VOLUME) / EXACT_VOLUME,
        "present_err": abs(vcv - EXACT_VOLUME) / EXACT_VOLUME,
    }
    return table31_reference.apply_table31_reference(row, "paraboloid")


def main():
    cv = import_from_path("curved_volume_wrapper", CURVED_VOLUME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    timings = new_timings()
    case_start = perf_now()
    for level in range(4):
        print(f"paraboloid: computing level {level + 1}/4", flush=True)
        points, faces = paraboloid_geometry.generate_mesh(level)
        rows.append(
            compute_row(
                cv,
                points,
                faces,
                OUT_DIR / f"level_{level:02d}",
                level,
                "Table 3-1 mesh" if level == 0 else f"projected subdivision level {level}",
                timings,
                msh_path=FOLDER_MESH if level == 0 else None,
            )
        )
    print("Paraboloid cap volume test: Table 3-1 mesh with projected subdivisions")
    print(f"V_exact = 2*pi = {EXACT_VOLUME:.16f}")
    print(
        "level curved_faces total_faces h_mean      PL_err_%       IRL_Evrard_err_% "
        "THINCQQ_err_%  sphere_err_%    present_err_%"
    )
    for r in rows:
        source = f" ({r['label']})"
        print(
            f"{r['level']:>2d} {r['curved_faces']:>6d} {r['faces']:>10d} {r['hmean']:.6e} "
            f"{100.0*r['pl_err']:.16e} {100.0*r['evrard_err']:.16e} "
            f"{100.0*r['thincqq_err']:.16e} {100.0*r['strobl_err']:.16e} "
            f"{100.0*r['present_err']:.16e}{source}"
        )
    print()
    h = [r["hmean"] for r in rows]
    h_evrard = [r["evrard_hmean"] for r in rows]
    print(f"Observed PL order:      {fit_order(h, [r['pl_err'] for r in rows]):.3f}")
    print(f"Observed Evrard forward-operator order: {order_text(h_evrard, [r['evrard_err'] for r in rows])}")
    print(f"Observed THINC/QQ order:{fit_order(h, [r['thincqq_err'] for r in rows]):.3f}")
    print(f"Observed sphere order:  {fit_order(h, [r['strobl_err'] for r in rows]):.3f}")
    print(f"Observed present order: {fit_order(h, [r['present_err'] for r in rows]):.3f}")
    print_timing_summary("paraboloid", timings, perf_now() - case_start)

    save_plot(rows)


def save_plot(rows):
    import matplotlib.pyplot as plt

    plt.style.use(str(STYLE))

    exact_floor = 1e-16
    h = np.array([r["hmean"] for r in rows])
    h_evrard = np.array([r["evrard_hmean"] for r in rows])
    pl_err = np.array([r["pl_err"] for r in rows])
    evrard_err = np.array([r["evrard_err"] for r in rows])
    thincqq_err = np.array([r["thincqq_err"] for r in rows])
    strobl_err = np.array([r["strobl_err"] for r in rows])
    present_err = np.array([r["present_err"] for r in rows])
    pl_pct = 100.0 * pl_err
    evrard_pct = 100.0 * np.maximum(evrard_err, exact_floor)
    thincqq_pct = 100.0 * np.maximum(thincqq_err, exact_floor)
    strobl_pct = 100.0 * np.maximum(strobl_err, exact_floor)
    present_pct = 100.0 * np.maximum(present_err, exact_floor)

    out = Path(__file__).with_name("paraboloid.png")
    fig, ax = plt.subplots()

    ax.loglog(
        h,
        pl_pct,
        "o-",
        label="PLIC / PL (Rider & Kothe, 1998; Scardovelli & Zaleski, 1999)",
    )
    ax.loglog(
        h_evrard,
        evrard_pct,
        "^-",
        label="Evrard-type paraboloid $V_{\\mathrm{patch}}$ (Evrard et al., 2023)",
    )
    ax.loglog(
        h,
        thincqq_pct,
        "s--",
        markerfacecolor="none",
        label="THINC/QQ quadratic $V_{\\mathrm{patch}}$ (Xie & Xiao, 2017)",
    )
    ax.loglog(
        h,
        strobl_pct,
        "x:",
        label="sphere/hex overlap $V_{\\mathrm{patch}}$ (Strobl et al., 2016)",
    )
    ax.loglog(
        h,
        present_pct,
        "D-",
        label="Curved (Quadric Patch) Volume",
    )
    table_rows = [i for i, r in enumerate(rows) if r["label"] == "Table 3-1 mesh"]
    if table_rows:
        idx = table_rows[0]
        ax.loglog(
            h[idx],
            present_pct[idx],
            "o",
            markerfacecolor="none",
            markeredgecolor="k",
            label="_nolegend_",
        )
    ref = pl_pct[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "k--", label="_nolegend_")
    ax.invert_xaxis()
    ax.set_xlabel("mean edge length h")
    ax.set_ylabel("relative volume error (%)")
    ax.set_title("Paraboloid cap volume error from $V_{\\mathrm{patch}}$ models")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
