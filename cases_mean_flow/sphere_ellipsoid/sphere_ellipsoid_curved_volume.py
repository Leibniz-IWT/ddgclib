# cases_mean_flow/ellipsoid_curved_volume.py
from pathlib import Path
import argparse
import numpy as np

# ddgclib method wrappers (your updated registries)
import ddgclib._method_wrappers as mw


def _read_gmsh_tri(msh_path: str):
    """
    Return (points, tris) from a Gmsh .msh file (surface triangles).
    Prefer project helper; fallback to meshio.
    """
    # 1) Project helper (consistent with benchmarks)
    try:
        from benchmarks._benchmark_plotting_utils import read_gmsh_tri
        pts, tris = read_gmsh_tri(msh_path)
        return np.asarray(pts, float), np.asarray(tris, int)
    except Exception:
        pass

    # 2) Fallback: meshio
    try:
        import meshio
    except Exception as e:
        raise ImportError(
            "Could not import benchmarks._benchmark_plotting_utils.read_gmsh_tri "
            "and meshio is not available. Install meshio."
        ) from e

    mesh = meshio.read(msh_path)
    pts = np.asarray(mesh.points[:, :3], float)

    tris = None
    for cb in mesh.cells:
        if cb.type in ("triangle", "triangle3"):
            tris = np.asarray(cb.data, int)
            break
    if tris is None:
        raise ValueError(f"No triangle cells found in msh: {msh_path}")

    return pts, tris


def _read_gmsh_tets(msh_path: str):
    """
    Return (points, tets) from a Gmsh .msh file (volume tetrahedra).
    Uses meshio (recommended).
    """
    try:
        import meshio
    except Exception as e:
        raise ImportError(
            "meshio is required to read tetrahedra from .msh. Install meshio."
        ) from e

    mesh = meshio.read(msh_path)
    pts = np.asarray(mesh.points[:, :3], float)

    tets = None
    for cb in mesh.cells:
        if cb.type in ("tetra", "tetra4"):
            tets = np.asarray(cb.data, int)
            break

    return pts, tets


def _volume_from_tets(points: np.ndarray, tets: np.ndarray) -> float:
    """
    Sum volumes of tetrahedra:
        V = |det(b-a, c-a, d-a)| / 6
    """
    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    vols = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0
    return float(np.sum(vols))


def run_case(
    msh_path: str,
    ax: float,
    ay: float,
    az: float,
    out_dir: str,
):
    msh_path = str(msh_path)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load surface mesh for curved-volume pipeline
    points_surf, tris = _read_gmsh_tri(msh_path)

    # ---- load tetra mesh for V_flat (sum of all tets)
    points_tet, tets = _read_gmsh_tets(msh_path)
    if tets is None or len(tets) == 0:
        raise ValueError(
            f"No tetra elements found in {msh_path}. "
            f"Cannot compute V_flat as sum of all tets."
        )

    # ---- theory
    V_theory = (4.0 / 3.0) * np.pi * float(ax) * float(ay) * float(az)

    # ---- V_flat = sum of tetra volumes (your requirement)
    V_flat = _volume_from_tets(points_tet, tets)

    # ---- curved total volume V (surface-based curved-volume pipeline)
    V_curved = float(
        mw.Volume(method="curved_volume")(
            (points_surf, tris),
            complex_dtype="vf",
            workdir=str(out_dir),
            msh_path=msh_path,
        )
    )

    # ---- curved dual volumes Vi (per-vertex, surface-based pipeline)
    Vi = mw.Volume_i(method="curved_volume")(
        (points_surf, tris),
        complex_dtype="vf",
        workdir=str(out_dir),
        msh_path=msh_path,
    )
    Vi = np.asarray(Vi, float)
    V_from_Vi = float(np.nansum(Vi))

    # ---- diagnostics
    abs_diff = abs(V_curved - V_from_Vi)
    rel_diff = abs_diff / max(abs(V_curved), 1e-16) * 100.0

    rel_err_flat = (V_flat - V_theory) / V_theory * 100.0
    rel_err_curved = (V_curved - V_theory) / V_theory * 100.0

    # ---- save outputs
    np.save(out_dir / "points_surface.npy", points_surf)
    np.save(out_dir / "tris_surface.npy", tris)
    np.save(out_dir / "points_tet.npy", points_tet)
    np.save(out_dir / "tets.npy", tets)
    np.save(out_dir / "Vi_curved.npy", Vi)

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Ellipsoid curved-volume case\n")
        f.write(f"msh_path: {msh_path}\n")
        f.write(f"ax, ay, az: {ax}, {ay}, {az}\n")
        f.write(f"V_theory: {V_theory:.12f}\n")
        f.write(f"V_flat (sum of tets): {V_flat:.12f}\n")
        f.write(f"V_curved (curved_volume on surface): {V_curved:.12f}\n")
        f.write(f"sum(Vi_curved): {V_from_Vi:.12f}\n")
        f.write(f"|V_curved - sum(Vi)|: {abs_diff:.12e}  ({rel_diff:.6f}%)\n")
        f.write(f"rel_err_flat%: {rel_err_flat:.6f}\n")
        f.write(f"rel_err_curved%: {rel_err_curved:.6f}\n")

    print("=== Ellipsoid curved-volume case ===")
    print(f"msh_path: {msh_path}")
    print(f"ax, ay, az = {ax}, {ay}, {az}")
    print(f"V_theory        = {V_theory:.12f}")
    print(f"V_flat (tets)   = {V_flat:.12f}   (rel_err {rel_err_flat:.6f}%)")
    print(f"V_curved        = {V_curved:.12f} (rel_err {rel_err_curved:.6f}%)")
    print(f"sum(Vi_curved)  = {V_from_Vi:.12f}")
    print(f"|V_curved - sum(Vi)| = {abs_diff:.12e} ({rel_diff:.6f}%)")
    print(f"Outputs written to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msh_path",
        type=str,
        default="test_cases/Ellip_0_sub0_full.msh",
        help="Path to ellipsoid .msh (must contain tetra elements for V_flat).",
    )
    parser.add_argument("--ax", type=float, default=1.5)
    parser.add_argument("--ay", type=float, default=1.0)
    parser.add_argument("--az", type=float, default=0.8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "ellipsoid_curved_volume"),
        help="Output directory for this case run.",
    )
    args = parser.parse_args()

    run_case(
        msh_path=args.msh_path,
        ax=args.ax,
        ay=args.ay,
        az=args.az,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
