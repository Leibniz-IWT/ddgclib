"""Section 3.4 hyperbolic-cylinder volume comparison.

Run from the project root:

    python 02_Figures/Figure_3-8_VolumeOperatorComparison/hyperbolic_cylinder/hyperbolic_cylinder_all_methods.py

The script recomputes the hyperbolic-cylinder sweep and writes
``hyperbolic_cylinder/hyperbolic_cylinder_all_methods_result.csv``.  The
equations, literature references, and callable functions are kept directly in
the top-level author-year files. The comparison operators are:

* Evrard-type paraboloid: local osculating-paraboloid ``Vpatch`` from the
  paraboloid moment framework of Evrard et al. (2023), Eqs. (2.4)-(3.36), used
  here as a non-matching paraboloid surrogate.
* THINC/QQ: implicit quadratic surface evaluated by finite Gaussian quadrature,
  following Xie and Xiao (2017), Eq. (9), Eqs. (20)-(23), and Appendix B.
* Strobl: fitted-sphere/hexahedron overlap via Strobl et al. (2016), Eq. (2)
  and Algorithm 1; this is a sphere-special non-matching comparison.
"""

import contextlib
import csv
import importlib.util
import io
import math
import subprocess
import sys
from pathlib import Path

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
import hyperbolic_cylinder_mesh as hyperbolic_cylinder_geometry
import plic_pl_1998_1999
import strobl_2016
import table31_reference
import xie_xiao_2017

CURVED_VOLUME = BUNDLE_ROOT / "_curved_volume.py"
TRANSLATION_ENGINE = BUNDLE_ROOT / "curved_volume" / "All_Translation_Volume_Transformed.py"
FOLDER_MESH = BUNDLE_ROOT / "msh" / "snapped_x2_minus_y2_ascii.msh"
OUT_DIR = Path(__file__).with_name("_work")
STYLE = BUNDLE_ROOT / "neatplot-main" / "standard.mplstyle"

X_MAX = 3.0
Y_MAX = 2.0
Z_MIN = 0.0
Z_MAX = 2.0
EXACT_VOLUME = (Z_MAX - Z_MIN) * (
    2.0 * X_MAX * Y_MAX
    - (Y_MAX * math.sqrt(1.0 + Y_MAX * Y_MAX) + math.asinh(Y_MAX))
)
PRESENT_COEFFS_KWARGS = {"plane_rel_tol": 1e-3}
QUADRIC_COEFFS = (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
STROBL_HEXGRID_N = 4


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def signed_pl_volume(points, faces):
    return sum(
        np.dot(points[a], np.cross(points[b], points[c])) / 6.0
        for a, b, c in faces
    )


def pl_volume(points, faces):
    return abs(signed_pl_volume(points, faces))


def orient_positive_volume(points, faces):
    if signed_pl_volume(points, faces) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return points, faces


def is_curved_point(point, tol=1e-8):
    x, y, _z = point
    return abs(x * x - y * y - 1.0) < tol and x > 0.0


def is_curved_face(points, face, tol=1e-8):
    return bool(all(is_curved_point(p, tol=tol) for p in points[face]))


def is_xright_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 0], X_MAX, atol=tol))


def is_ytop_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 1], Y_MAX, atol=tol))


def is_ybottom_face(points, face, tol=1e-8):
    return bool(np.allclose(points[face, 1], -Y_MAX, atol=tol))


def is_zcap_face(points, face, tol=1e-8):
    z = points[face, 2]
    return bool(np.allclose(z, Z_MAX, atol=tol) or np.allclose(z, Z_MIN, atol=tol))


def analytical_outward_normal(point):
    x, y, _z = point
    return np.array([-2.0 * x, 2.0 * y, 0.0], dtype=float)


def orient_faces_by_geometry(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if is_xright_face(points, face):
            desired = np.array([1.0, 0.0, 0.0])
        elif is_ytop_face(points, face):
            desired = np.array([0.0, 1.0, 0.0])
        elif is_ybottom_face(points, face):
            desired = np.array([0.0, -1.0, 0.0])
        elif is_zcap_face(points, face):
            desired = np.array([0.0, 0.0, 1.0 if pts[:, 2].mean() > 0.5 * (Z_MIN + Z_MAX) else -1.0])
        elif is_curved_face(points, face):
            desired = analytical_outward_normal(pts.mean(axis=0))
        else:
            raise ValueError(f"Could not classify boundary face {i}: {pts}")
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def compact_mesh(points, faces):
    used = np.unique(faces.ravel())
    remap = -np.ones(len(points), dtype=int)
    remap[used] = np.arange(len(used))
    return points[used], remap[faces]


def load_outer_msh_triangles(path):
    mesh = meshio.read(path)
    points = mesh.points[:, :3]
    z1_id = int(mesh.field_data["Z1Plane"][0]) if "Z1Plane" in mesh.field_data else None
    tris = []
    for block_i, cb in enumerate(mesh.cells):
        if cb.type != "triangle":
            continue
        data = cb.data.astype(int)
        if z1_id is not None and "gmsh:physical" in mesh.cell_data:
            phys = mesh.cell_data["gmsh:physical"][block_i]
            data = data[phys != z1_id]
        if len(data):
            tris.append(data)
    if not tris:
        raise ValueError(f"No exterior triangle cells found in {path}")
    faces = np.vstack(tris).astype(int)
    points, faces = compact_mesh(points, faces)
    faces = orient_faces_by_geometry(points, faces)
    return orient_positive_volume(points, faces)


def mesh_size(points, faces):
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    lens = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])
    return float(lens.mean())


def curved_face_count(points, faces):
    return sum(1 for face in faces if is_curved_face(points, face))


def project_curved_point(point):
    out = np.array(point, dtype=float)
    out[0] = math.sqrt(1.0 + out[1] * out[1])
    return out


def project_midpoint(a, b):
    mid = 0.5 * (a + b)

    both_curved = is_curved_point(a) and is_curved_point(b)
    same_xright = abs(a[0] - X_MAX) < 1e-8 and abs(b[0] - X_MAX) < 1e-8
    same_ytop = abs(a[1] - Y_MAX) < 1e-8 and abs(b[1] - Y_MAX) < 1e-8
    same_ybottom = abs(a[1] + Y_MAX) < 1e-8 and abs(b[1] + Y_MAX) < 1e-8
    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8

    if same_ytop:
        mid[1] = Y_MAX
    elif same_ybottom:
        mid[1] = -Y_MAX

    if both_curved:
        mid = project_curved_point(mid)
    elif same_xright:
        mid[0] = X_MAX

    if same_top:
        mid[2] = Z_MAX
    elif same_bottom:
        mid[2] = Z_MIN

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


def face_volume_contribution(points, face):
    a, b, c = face
    return np.dot(points[a], np.cross(points[b], points[c])) / 6.0


def write_translation_coeffs_csv(points, faces, path):
    columns = [
        "triangle_id",
        "A_id",
        "B_id",
        "C_id",
        "scale_factors1_x",
        "scale_factors1_y",
        "scale_factors1_z",
        "A_transformed_x",
        "A_transformed_y",
        "A_transformed_z",
        "B_transformed_x",
        "B_transformed_y",
        "B_transformed_z",
        "C_transformed_x",
        "C_transformed_y",
        "C_transformed_z",
        "ABC_new_A",
        "ABC_new_B",
        "ABC_new_C",
        "ABC_new_D",
        "ABC_new_E",
        "ABC_new_F",
        "ABC_new_G",
        "ABC_new_H",
        "ABC_new_I",
        "ABC_new_J",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for tri_id, face in enumerate(faces):
            if not is_curved_face(points, face):
                continue
            a, b, c = [int(v) for v in face]
            pa, pb, pc = points[[a, b, c]]
            writer.writerow(
                {
                    "triangle_id": tri_id,
                    "A_id": a,
                    "B_id": b,
                    "C_id": c,
                    "scale_factors1_x": 1.0,
                    "scale_factors1_y": 1.0,
                    "scale_factors1_z": 1.0,
                    "A_transformed_x": pa[0],
                    "A_transformed_y": pa[1],
                    "A_transformed_z": pa[2],
                    "B_transformed_x": pb[0],
                    "B_transformed_y": pb[1],
                    "B_transformed_z": pb[2],
                    "C_transformed_x": pc[0],
                    "C_transformed_y": pc[1],
                    "C_transformed_z": pc[2],
                    "ABC_new_A": 1.0,
                    "ABC_new_B": -1.0,
                    "ABC_new_C": 0.0,
                    "ABC_new_D": 0.0,
                    "ABC_new_E": 0.0,
                    "ABC_new_F": 0.0,
                    "ABC_new_G": 0.0,
                    "ABC_new_H": 0.0,
                    "ABC_new_I": 0.0,
                    "ABC_new_J": -1.0,
                }
            )


def sum_volume_corrections(path):
    total = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total += float(row["Vcorrection"])
    return total


def run_present_method(_cv, points, faces, workdir, msh_path=None):
    workdir.mkdir(parents=True, exist_ok=True)
    if msh_path == FOLDER_MESH:
        vol_csv = FOLDER_MESH.with_name(f"{FOLDER_MESH.stem}_COEFFS_Transformed_Volume.csv")
        if vol_csv.exists():
            return pl_volume(points, faces) + sum_volume_corrections(vol_csv)

    coeffs_csv = workdir / "hyperbolic_cylinder_COEFFS_Transformed.csv"
    write_translation_coeffs_csv(points, faces, coeffs_csv)
    proc = subprocess.run(
        [sys.executable, str(TRANSLATION_ENGINE), str(coeffs_csv)],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    vol_csv = workdir / f"{coeffs_csv.stem}_Volume.csv"
    return pl_volume(points, faces) + sum_volume_corrections(vol_csv)


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


def compute_row(cv, points, faces, workdir, level, label, timings, msh_path=None):
    vpl = timed(timings, "plic_pl", plic_pl_1998_1999.closed_surface_volume, points, faces)
    v_model_2023 = timed(
        timings,
        "evrard_type_paraboloid",
        evrard_2023.paraboloid_taylor_volume,
        points, faces, is_curved_face, QUADRIC_COEFFS
    )
    v_model_2017 = timed(
        timings,
        "thinc_qq",
        xie_xiao_2017.gaussian_quadric_volume,
        points, faces, is_curved_face, QUADRIC_COEFFS
    )
    v_model_2016 = timed(timings, "strobl_sphere_overlap", strobl_2016.sphere_hexgrid_volume, points)

    # Make the comparison quantity explicit: each literature model is plotted
    # as V_PL plus its own curved-volume correction Vpatch_year.
    Vpatch_2023 = v_model_2023 - vpl
    Vpatch_2017 = v_model_2017 - vpl
    Vpatch_2016 = v_model_2016 - vpl
    vppic = vpl + Vpatch_2023
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
        "pl": vpl,
        "ppic": vppic,
        "thincqq": vthincqq,
        "strobl": vstrobl,
        "Vpatch_2023": Vpatch_2023,
        "Vpatch_2017": Vpatch_2017,
        "Vpatch_2016": Vpatch_2016,
        "present": vcv,
        "pl_err": abs(vpl - EXACT_VOLUME) / EXACT_VOLUME,
        "ppic_err": abs(vppic - EXACT_VOLUME) / EXACT_VOLUME,
        "thincqq_err": abs(vthincqq - EXACT_VOLUME) / EXACT_VOLUME,
        "strobl_err": abs(vstrobl - EXACT_VOLUME) / EXACT_VOLUME,
        "present_err": abs(vcv - EXACT_VOLUME) / EXACT_VOLUME,
    }
    return table31_reference.apply_table31_reference(row, "hyperbolic_cylinder")


def main():
    cv = import_from_path("curved_volume_wrapper", CURVED_VOLUME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    timings = new_timings()
    case_start = perf_now()
    for level in range(4):
        print(f"hyperbolic_cylinder: computing level {level + 1}/4", flush=True)
        points, faces = hyperbolic_cylinder_geometry.generate_mesh(level)
        rows.append(
            compute_row(
                cv,
                points,
                faces,
                OUT_DIR / f"level_{level:02d}",
                level,
                "Table 3-1 mesh" if level == 0 else f"subdivision level {level}",
                timings,
                msh_path=FOLDER_MESH if level == 0 else None,
            )
        )
    print("Hyperbolic-cylinder volume test: x^2-y^2=1, x in [sqrt(1+y^2), 3], y in [-2, 2], z in [0, 2]")
    print(f"V_exact = {EXACT_VOLUME:.16f}")
    print(
        "level curved_faces total_faces h_mean      PL_err_%       PPIC_err_%     "
        "THINCQQ_err_%  sphere_err_%    present_err_%"
    )
    for r in rows:
        source = f" ({r['label']})"
        print(
            f"{r['level']:>2d} {r['curved_faces']:>6d} {r['faces']:>10d} {r['hmean']:.6e} "
            f"{100.0*r['pl_err']:.16e} {100.0*r['ppic_err']:.16e} "
            f"{100.0*r['thincqq_err']:.16e} {100.0*r['strobl_err']:.16e} "
            f"{100.0*r['present_err']:.16e}{source}"
        )

    h = [r["hmean"] for r in rows]
    print()
    print(f"Observed PL order:      {fit_order(h, [r['pl_err'] for r in rows]):.3f}")
    print(f"Observed Evrard paraboloid Vpatch surrogate order: {order_text(h, [r['ppic_err'] for r in rows])}")
    print(f"Observed THINC/QQ order:{order_text(h, [r['thincqq_err'] for r in rows])}")
    print(f"Observed sphere order:  {order_text(h, [r['strobl_err'] for r in rows])}")
    print(f"Observed present order: {order_text(h, [r['present_err'] for r in rows])}")
    print_timing_summary("hyperbolic_cylinder", timings, perf_now() - case_start)

    save_plot(rows)


def save_plot(rows):
    import matplotlib.pyplot as plt

    plt.style.use(str(STYLE))

    exact_floor = 1e-16
    h = np.array([r["hmean"] for r in rows])
    pl_err = np.array([r["pl_err"] for r in rows])
    ppic_err = np.array([r["ppic_err"] for r in rows])
    thincqq_err = np.array([r["thincqq_err"] for r in rows])
    strobl_err = np.array([r["strobl_err"] for r in rows])
    present_err = np.array([r["present_err"] for r in rows])

    pl_pct = 100.0 * pl_err
    ppic_pct = 100.0 * np.maximum(ppic_err, exact_floor)
    thincqq_pct = 100.0 * np.maximum(thincqq_err, exact_floor)
    strobl_pct = 100.0 * np.maximum(strobl_err, exact_floor)
    present_pct = 100.0 * np.maximum(present_err, exact_floor)

    out = Path(__file__).with_name("hyperbolic_cylinder.png")
    fig, ax = plt.subplots()

    ax.loglog(
        h,
        pl_pct,
        "o-",
        label="PLIC / PL (Rider & Kothe, 1998; Scardovelli & Zaleski, 1999)",
    )
    ax.loglog(
        h,
        ppic_pct,
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
    ax.set_title("Hyperbolic-cylinder volume error from $V_{\\mathrm{patch}}$ models")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
