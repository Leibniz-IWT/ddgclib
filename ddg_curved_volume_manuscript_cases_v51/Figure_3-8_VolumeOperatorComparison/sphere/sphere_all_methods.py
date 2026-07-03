"""Section 3.4 sphere volume comparison.

Run from the project root:

    python 02_Figures/Figure_3-8_VolumeOperatorComparison/sphere/sphere_all_methods.py

The script recomputes the unit-sphere volume errors and writes
``sphere/sphere_all_methods_result.csv``.  The equations, literature references,
and callable functions are kept directly in the top-level author-year files. The
literature comparison lines are:

* Evrard-type paraboloid: ``evrard_2023.paraboloid_taylor_volume`` uses the
  local second-order paraboloid model associated with Evrard et al. (2023,
  Eqs. (2.4)-(3.36)).  On a sphere
  this is a paraboloid surrogate, not an exact sphere formula.
* THINC/QQ: ``xie_xiao_2017.gaussian_quadric_volume`` uses the benchmark quadratic
  surface and finite Gaussian quadrature, following Xie and Xiao (2017),
  especially Eq. (9), Eqs. (20)-(23), and Appendix B.
* Strobl: ``strobl_2016.sphere_hexgrid_volume`` calls the public ``overlap`` package
  for sphere/hexahedron overlaps, corresponding to Strobl et al. (2016),
  Eq. (2) and Algorithm 1.  This is the model-matched sphere-special method.
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

import numpy as np
import overlap

import evrard_2023
import plic_pl_1998_1999
import present_quadric_patch_2026
import sphere_mesh as sphere_geometry
import strobl_2016
import xie_xiao_2017

PYTHON_HELPERS = BUNDLE_ROOT / "common" / "sphere_paraboloid_order_test.py"
CURVED_VOLUME = BUNDLE_ROOT / "_curved_volume.py"
OUT_DIR = Path(__file__).with_name("_work")
STYLE = BUNDLE_ROOT / "neatplot-main" / "standard.mplstyle"
PRESENT_COEFFS_KWARGS = {"plane_rel_tol": 3e-4}
QUADRIC_COEFFS = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
STROBL_HEXGRID_N = 4


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    return np.array(out)


def all_faces_curved(_points, _face):
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    true_volume = 4.0 * math.pi / 3.0
    rows = []
    timings = new_timings()
    case_start = perf_now()

    for level in range(4):
        print(f"sphere: computing level {level + 1}/4", flush=True)
        verts, faces = sphere_geometry.generate_mesh(level)
        hmax, hmean = sphere_geometry.mesh_size(verts, faces)
        vpl = timed(timings, "plic_pl", plic_pl_1998_1999.closed_surface_volume, verts, faces)
        v_model_2023 = timed(
            timings,
            "evrard_type_paraboloid",
            evrard_2023.paraboloid_taylor_volume,
            verts, faces, all_faces_curved, QUADRIC_COEFFS
        )
        v_model_2017 = timed(
            timings,
            "thinc_qq",
            xie_xiao_2017.gaussian_quadric_volume,
            verts, faces, all_faces_curved, QUADRIC_COEFFS
        )
        v_model_2016 = timed(timings, "strobl_sphere_overlap", strobl_2016.sphere_hexgrid_volume, verts)

        # Make the comparison quantity explicit: each literature model is plotted
        # as V_PL plus its own curved-volume correction Vpatch_year.
        Vpatch_2023 = v_model_2023 - vpl
        Vpatch_2017 = v_model_2017 - vpl
        Vpatch_2016 = v_model_2016 - vpl
        vppic = vpl + Vpatch_2023
        vthincqq = vpl + Vpatch_2017
        vstrobl = vpl + Vpatch_2016
        workdir = OUT_DIR / f"level_{level:02d}"
        workdir.mkdir(parents=True, exist_ok=True)

        vcv = timed(
            timings,
            "present_quadric_patch",
            present_quadric_patch_2026.manuscript_curved_volume,
            verts,
            faces,
            workdir,
            coeffs_kwargs=PRESENT_COEFFS_KWARGS,
        )

        rows.append(
            {
                "level": level,
                "faces": len(faces),
                "hmean": hmean,
                "pl": vpl,
                "ppic": vppic,
                "thincqq": vthincqq,
                "strobl": vstrobl,
                "Vpatch_2023": Vpatch_2023,
                "Vpatch_2017": Vpatch_2017,
                "Vpatch_2016": Vpatch_2016,
                "curved_volume": vcv,
                "pl_err": abs(vpl - true_volume) / true_volume,
                "ppic_err": abs(vppic - true_volume) / true_volume,
                "thincqq_err": abs(vthincqq - true_volume) / true_volume,
                "strobl_err": abs(vstrobl - true_volume) / true_volume,
                "curved_volume_err": abs(vcv - true_volume) / true_volume,
            }
        )

    print("Unit sphere volume test using the manuscript curved-volume pipeline")
    print(f"V_exact = {true_volume:.16f}")
    print(
        "level faces h_mean      PL_err_%       PPIC_err_%     "
        "THINCQQ_err_%  Strobl_err_%    present_err_%"
    )
    for r in rows:
        print(
            f"{r['level']:>2d} {r['faces']:>6d} {r['hmean']:.6e} "
            f"{100.0*r['pl_err']:.16e} {100.0*r['ppic_err']:.16e} "
            f"{100.0*r['thincqq_err']:.16e} {100.0*r['strobl_err']:.16e} "
            f"{100.0*r['curved_volume_err']:.16e}"
        )

    h = [r["hmean"] for r in rows]
    pl_err = [r["pl_err"] for r in rows]
    ppic_err = [r["ppic_err"] for r in rows]
    thincqq_err = [r["thincqq_err"] for r in rows]
    strobl_err = [r["strobl_err"] for r in rows]
    cv_err = [r["curved_volume_err"] for r in rows]
    print()
    print("Observed order:")
    print(f"  PL enclosed volume:            {fit_order(h, pl_err):.3f}")
    print(f"  Evrard paraboloid Vpatch surrogate: {fit_order(h, ppic_err):.3f}")
    print(f"  THINC/QQ quadratic Vpatch approx:  {fit_order(h, thincqq_err):.3f}")
    print(f"  sphere/hex overlap Vpatch:       {fit_order(h, strobl_err):.3f} (not meaningful when error is round-off)")
    print(f"  Curved (Quadric Patch) Volume: {fit_order(h, cv_err):.3f} (not meaningful when error is round-off)")
    print_timing_summary("sphere", timings, perf_now() - case_start)

    save_plot(rows)


def save_plot(rows):
    import matplotlib.pyplot as plt

    plt.style.use(str(STYLE))

    out = Path(__file__).with_name("sphere.png")
    h = np.array([r["hmean"] for r in rows])
    pl_err = np.array([r["pl_err"] for r in rows])
    ppic_err = np.array([r["ppic_err"] for r in rows])
    thincqq_err = np.array([r["thincqq_err"] for r in rows])
    strobl_err = np.array([r["strobl_err"] for r in rows])
    cv_err = np.array([r["curved_volume_err"] for r in rows])

    eps_floor = 1e-16
    pl_pct = 100.0 * pl_err
    ppic_pct = 100.0 * np.maximum(ppic_err, eps_floor)
    thincqq_pct = 100.0 * np.maximum(thincqq_err, eps_floor)
    strobl_pct = 100.0 * np.maximum(strobl_err, eps_floor)
    cv_plot = 100.0 * np.maximum(cv_err, eps_floor)

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
        cv_plot,
        "D-",
        label="Curved (Quadric Patch) Volume",
    )

    ref = pl_pct[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "k--", label="_nolegend_")
    ax.invert_xaxis()
    ax.set_xlabel("mean edge length h")
    ax.set_ylabel("relative volume error (%)")
    ax.set_title("Sphere volume error from $V_{\\mathrm{patch}}$ models")
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.47), borderaxespad=0.0)

    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()
