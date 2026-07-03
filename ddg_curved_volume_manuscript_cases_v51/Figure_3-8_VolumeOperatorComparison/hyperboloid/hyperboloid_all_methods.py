"""Section 3.4 hyperboloid volume comparison.

Run from the project root:

    python 02_Figures/Figure_3-8_VolumeOperatorComparison/hyperboloid/hyperboloid_all_methods.py

The script recomputes the hyperboloid sweep and writes
``hyperboloid/hyperboloid_all_methods_result.csv``.  The equations, literature
references, and callable functions are kept directly in the top-level
author-year files. The literature-model lines are:

* Evrard-type paraboloid: a local osculating-paraboloid ``Vpatch`` surrogate
  based on the paraboloid moment framework of Evrard et al. (2023, Eqs.
  (2.4)-(3.36)); it is intentionally not an exact hyperboloid formula.
* THINC/QQ: ``xie2017_appendix_ab_volume`` supplies the benchmark hyperboloid
  as the quadratic surface of Xie and Xiao (2017), Eq. (9), then uses their
  Appendix-B style triangular Gaussian quadrature.  This tests a model-matched
  quadratic surface with finite quadrature, not closed-form hyperboloid volume.
* Strobl: fitted-sphere/hexahedron overlap via Strobl et al. (2016), Eq. (2)
  and Algorithm 1, included as a sphere-special non-matching baseline.
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
import hyperboloid_mesh as hyperboloid_geometry
import plic_pl_1998_1999
import present_quadric_patch_2026
import strobl_2016
import table31_reference
import xie_xiao_2017

CURVED_VOLUME = BUNDLE_ROOT / "_curved_volume.py"
FOLDER_MESH = BUNDLE_ROOT / "msh" / "coarse_hyperboloid.msh"
OUT_DIR = Path(__file__).with_name("_work")
STYLE = BUNDLE_ROOT / "neatplot-main" / "standard.mplstyle"

Z_MIN = -1.0
Z_MAX = 1.0
EXACT_VOLUME = math.pi * (2.0 + 8.0 / 12.0)
RIM_RADIUS = math.sqrt(2.0)
PRESENT_COEFFS_KWARGS = {"plane_rel_tol": 3e-4}
QUADRIC_COEFFS = (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
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


def analytical_outward_normal(point):
    x, y, z = point
    return np.array([x, y, -z], dtype=float)


def orient_faces_by_geometry(points, faces):
    out = faces.copy()
    for i, face in enumerate(out):
        pts = points[face]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        z = pts[:, 2]
        if np.allclose(z, Z_MAX, atol=1e-8):
            desired = np.array([0.0, 0.0, 1.0])
        elif np.allclose(z, Z_MIN, atol=1e-8):
            desired = np.array([0.0, 0.0, -1.0])
        else:
            desired = analytical_outward_normal(pts.mean(axis=0))
        if np.dot(normal, desired) < 0.0:
            out[i] = face[[0, 2, 1]]
    return out


def orient_positive_volume(points, faces):
    if signed_pl_volume(points, faces) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return points, faces


def generate_hyperboloid_mesh(nz, nth):
    """Closed mesh for x^2+y^2-z^2=1, z in [-1, 1]."""
    points = []
    side_ids = []
    for z in np.linspace(Z_MIN, Z_MAX, nz):
        r = math.sqrt(1.0 + z * z)
        row = []
        for j in range(nth):
            th = 2.0 * math.pi * j / nth
            row.append(len(points))
            points.append([r * math.cos(th), r * math.sin(th), z])
        side_ids.append(row)

    top_center = len(points)
    points.append([0.0, 0.0, Z_MAX])
    bottom_center = len(points)
    points.append([0.0, 0.0, Z_MIN])

    faces = []
    for i in range(nz - 1):
        for j in range(nth):
            p00 = side_ids[i][j]
            p01 = side_ids[i][(j + 1) % nth]
            p10 = side_ids[i + 1][j]
            p11 = side_ids[i + 1][(j + 1) % nth]
            faces.append([p00, p10, p11])
            faces.append([p00, p11, p01])

    top_ring = side_ids[-1]
    bottom_ring = side_ids[0]
    for j in range(nth):
        faces.append([top_center, top_ring[j], top_ring[(j + 1) % nth]])
        faces.append([bottom_center, bottom_ring[(j + 1) % nth], bottom_ring[j]])

    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    faces = orient_faces_by_geometry(points, faces)
    return orient_positive_volume(points, faces)


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


def mesh_size(points, faces):
    edges = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(i), int(j)))))
    lens = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])
    return float(lens.mean())


def is_cap_face(points, face):
    z = points[face, 2]
    return bool(
        np.allclose(z, Z_MAX, atol=1e-8) or np.allclose(z, Z_MIN, atol=1e-8)
    )


def is_curved_side_face(points, face):
    return not is_cap_face(points, face)


def curved_face_count(points, faces):
    return sum(1 for face in faces if not is_cap_face(points, face))


def is_cap_point(point):
    return abs(point[2] - Z_MAX) < 1e-8 or abs(point[2] - Z_MIN) < 1e-8


def is_rim_point(point):
    if not is_cap_point(point):
        return False
    r = math.hypot(point[0], point[1])
    return abs(r - math.sqrt(1.0 + point[2] * point[2])) < 1e-7


def project_side_point(point):
    x, y, z = point
    r = math.hypot(x, y)
    target = math.sqrt(1.0 + z * z)
    if r < 1e-14:
        return np.array([target, 0.0, z], dtype=float)
    return np.array([target * x / r, target * y / r, z], dtype=float)


def project_midpoint(a, b):
    mid = 0.5 * (a + b)

    same_top = abs(a[2] - Z_MAX) < 1e-8 and abs(b[2] - Z_MAX) < 1e-8
    same_bottom = abs(a[2] - Z_MIN) < 1e-8 and abs(b[2] - Z_MIN) < 1e-8
    if same_top or same_bottom:
        zcap = Z_MAX if same_top else Z_MIN
        mid[2] = zcap
        if is_rim_point(a) and is_rim_point(b):
            r = math.hypot(mid[0], mid[1])
            if r > 1e-14:
                mid[0] *= RIM_RADIUS / r
                mid[1] *= RIM_RADIUS / r
        return mid

    return project_side_point(mid)


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
        new_faces.extend(
            [
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca],
            ]
        )

    points_new = np.asarray(points_list, dtype=float)
    faces_new = np.asarray(new_faces, dtype=int)
    faces_new = orient_faces_by_geometry(points_new, faces_new)
    return orient_positive_volume(points_new, faces_new)


def run_present_method(cv, points, faces, workdir, msh_path=None):
    return present_quadric_patch_2026.manuscript_curved_volume(
        points,
        faces,
        workdir,
        msh_path=msh_path,
        coeffs_kwargs=PRESENT_COEFFS_KWARGS,
    )


def fit_order(h, err):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = err > 1e-15
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)[0])


def tail_order_text(h, err, tail=2):
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.nanmax(err) < 1e-10:
        return "round-off"
    if len(h) >= tail:
        h = h[-tail:]
        err = err[-tail:]
    p = fit_order(h, err)
    if math.isnan(p):
        return "round-off"
    return f"p={p:.2f}"


def xie2017_appendix_b_triangle_quadrature_points():
    """Six-point triangular Gaussian rule used for THINC/QQ face integrals.

    Xie & Xiao (2017) use Appendix A to construct the quadratic interface
    representation and Appendix B for Gaussian quadrature. In this surface
    Vpatch benchmark, the benchmark quadric supplies the Appendix-A quadratic
    exactly, and the Appendix-B triangular face rule is used for the integral.
    """
    a = 0.445948490915965
    b = 0.091576213509771
    w1 = 0.223381589678011
    w2 = 0.109951743655322
    return [
        (w1, (a, a, 1.0 - 2.0 * a)),
        (w1, (a, 1.0 - 2.0 * a, a)),
        (w1, (1.0 - 2.0 * a, a, a)),
        (w2, (b, b, 1.0 - 2.0 * b)),
        (w2, (b, 1.0 - 2.0 * b, b)),
        (w2, (1.0 - 2.0 * b, b, b)),
    ]


def compute_row(cv, points, faces, workdir, level, label, timings, msh_path=None):
    vpl = timed(timings, "plic_pl", plic_pl_1998_1999.closed_surface_volume, points, faces)
    v_model_2023 = timed(
        timings,
        "evrard_type_paraboloid",
        evrard_2023.paraboloid_taylor_volume,
        points, faces, is_curved_side_face, QUADRIC_COEFFS
    )
    v_model_2017 = timed(
        timings,
        "thinc_qq",
        xie_xiao_2017.gaussian_quadric_volume,
        points,
        faces,
        is_curved_side_face,
        QUADRIC_COEFFS,
        quadrature_points=xie2017_appendix_b_triangle_quadrature_points,
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
    return table31_reference.apply_table31_reference(row, "hyperboloid")


def main():
    cv = import_from_path("curved_volume_wrapper", CURVED_VOLUME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    timings = new_timings()
    case_start = perf_now()
    for level in range(4):
        print(f"hyperboloid: computing level {level + 1}/4", flush=True)
        points, faces = hyperboloid_geometry.generate_mesh(level)
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
    print("Hyperboloid volume test: x^2+y^2-z^2=1, z in [-1, 1]")
    print(f"V_exact = 8*pi/3 = {EXACT_VOLUME:.16f}")
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
    print(f"Observed Evrard paraboloid Vpatch surrogate order: {fit_order(h, [r['ppic_err'] for r in rows]):.3f}")
    print(f"Observed THINC/QQ Appendix A/B order:{fit_order(h, [r['thincqq_err'] for r in rows]):.3f}")
    print(f"Observed sphere order:  {fit_order(h, [r['strobl_err'] for r in rows]):.3f}")
    print(f"Observed present order: {fit_order(h, [r['present_err'] for r in rows]):.3f}")
    print(f"Observed present tail order: {tail_order_text(h, [r['present_err'] for r in rows])}")
    print_timing_summary("hyperboloid", timings, perf_now() - case_start)

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

    out = Path(__file__).with_name("hyperboloid.png")
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
    ax.set_title("Hyperboloid volume error from $V_{\\mathrm{patch}}$ models")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
