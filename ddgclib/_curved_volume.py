"""
ddgclib._curved_volume
----------------------

Facade that computes enclosed volume using the curved-volume pipeline:

1) Part-1: local quadric coefficients per triangle
2) Part-2: transform coeffs & geometry to canonical space
3) Part-3: per-face curved patch volumes (ellipsoid / translation / rotation)
4) Enforce thickness-weighted split identity
5) Sum Vcorrection and add to a baseline (flat-tet) closed-surface volume

Returns a single float (total volume) so it can be registered in _method_wrappers.py
as the "curved_volume" entry of _volume_methods.

Usage (through the registry):
    from ddgclib._method_wrappers import Volume
    vol = Volume(method="curved_volume")((points, tris), complex_dtype="vf")

You can also call curved_volume(...) directly.
"""

from __future__ import annotations
import os
import sys
import csv
import math
import tempfile
import importlib.util
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# Local helpers: package paths & dynamic imports

_THIS_DIR = Path(__file__).resolve().parent
_CURVED_DIR = _THIS_DIR / "curved_volume"


if str(_CURVED_DIR) not in sys.path:
    sys.path.insert(0, str(_CURVED_DIR))


def _pkg_file(name: str) -> Path:
    p = _CURVED_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Required module not found: {p}")
    return p

def _import_module(script_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Part-1 & Part-2 entry points (library versions)
#   - compute_rimsafe_alltri(msh_path, out_csv, **kwargs) -> (df, total_tris, skipped_flat)
#   - process_file(in_csv) -> (out_csv, err_csv, n_errors)

# Part 1: coeffs
_mod_coeffs = _import_module(_pkg_file("_1_coeffs_computing.py"), "cv_coeffs")
compute_rimsafe_alltri = getattr(_mod_coeffs, "compute_rimsafe_alltri")

# Part 2: transformer
_mod_xform = _import_module(_pkg_file("_2_quadric_transformer.py"), "cv_transform")
process_transform_file = getattr(_mod_xform, "process_file")


# Part-3 engines (ellipsoid / translation / rotation) + split enforcer

_mod_ell = _import_module(_pkg_file("_3_5_volume_ellipsoid.py"), "cv_vol_ell")
process_coeffs_transformed_csv = getattr(_mod_ell, "process_coeffs_transformed_csv")

_mod_trn = _import_module(_pkg_file("All_Translation_Volume_Transformed.py"), "cv_vol_trn")
_mod_rot = _import_module(_pkg_file("All_Rotation_Volume_Transformed.py"), "cv_vol_rot")

_mod_split = _import_module(_pkg_file("_4_dualvolume_split_patch_volume_thickness_weighted.py"), "cv_split")
split_patch_volume_thickness_weighted = getattr(_mod_split, "split_patch_volume_thickness_weighted")


# Utilities

def _write_gmsh2(points: np.ndarray, tris: np.ndarray, out_path: Path) -> None:
    """
    Minimal ASCII Gmsh v2.2 writer for triangle surfaces.
    """
    points = np.asarray(points, float)
    tris = np.asarray(tris, int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("tris must be (M,3)")

    N = points.shape[0]
    M = tris.shape[0]

    with open(out_path, "w", newline="") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{N}\n")
        for i, (x, y, z) in enumerate(points, start=1):
            f.write(f"{i} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{M}\n")
        # elementType 2 = 3-node triangle
        for i, (a, b, c) in enumerate(tris, start=1):
            f.write(f"{i} 2 0 {a+1} {b+1} {c+1}\n")
        f.write("$EndElements\n")


def _baseline_volume(points: np.ndarray, tris: np.ndarray) -> float:
    """
    Closed-surface signed volume (positive for outward orientation).
    """
    P = points
    T = tris
    v = 0.0
    for a, b, c in T:
        v += np.dot(P[a], np.cross(P[b], P[c]))
    return abs(v) / 6.0


def _pick(df: pd.DataFrame, names) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _build_volume_csv_by_routing(tr_csv_path: Path) -> Path:
    """
    Route transformed rows to the proper engines and concatenate results into
    <stem>_Volume.csv next to the input.
    """
    tr_csv_path = Path(tr_csv_path).resolve()
    df = pd.read_csv(tr_csv_path)

    # Identify coefficient columns (new or legacy)
    Acol = _pick(df, ["ABC_new_A", "ABC_A"]); Bcol = _pick(df, ["ABC_new_B", "ABC_B"]); Ccol = _pick(df, ["ABC_new_C", "ABC_C"])
    Ecol = _pick(df, ["ABC_new_E", "ABC_E"]); Fcol = _pick(df, ["ABC_new_F", "ABC_F"]); Icol = _pick(df, ["ABC_new_I", "ABC_I"])
    Jcol = _pick(df, ["ABC_new_J", "ABC_J"])
    needed = [Acol, Bcol, Ccol, Ecol, Fcol, Icol, Jcol]
    if any(c is None for c in needed):
        miss = [n for c, n in zip(needed, ["A", "B", "C", "E", "F", "I", "J"]) if c is None]
        raise ValueError(f"Missing coefficient columns in transformed CSV: {miss}")

    A = df[Acol].to_numpy()
    B = df[Bcol].to_numpy()
    C = df[Ccol].to_numpy()
    E = df[Ecol].to_numpy()
    F = df[Fcol].to_numpy()
    I = df[Icol].to_numpy()
    J = df[Jcol].to_numpy()

    TOL = 1e-9
    # Heuristic routing (same as used interactively):
    ell_mask = (A > 0) & (B > 0) & (C > 0) & (np.abs(J) > 0)           # ellipsoid-like
    noz_mask = (np.abs(C) < TOL) & (np.abs(E) < TOL) & (np.abs(F) < TOL) & (np.abs(I) < TOL)  # no z dependence -> translation
    axis_mask = ~(ell_mask | noz_mask)                                 # axisymmetric/rotation

    parts = []
    with tempfile.TemporaryDirectory() as td:
        TMP = Path(td)

        def write_subset(mask, fname):
            if mask.any():
                p = TMP / fname
                df.loc[mask].to_csv(p, index=False)
                return p
            return None

        p_ell = write_subset(ell_mask, "ell.csv")
        p_trn = write_subset(noz_mask, "trn.csv")
        p_rot = write_subset(axis_mask, "rot.csv")

        # ellipsoid engine (pure function style)
        if p_ell is not None:
            out_csv_ell, _, _, _ = process_coeffs_transformed_csv(str(p_ell))
            parts.append(Path(out_csv_ell))

        # translation engine (script-style main, writes to CWD)
        if p_trn is not None:
            argv_bak, cwd_bak = sys.argv[:], os.getcwd()
            try:
                os.chdir(TMP)
                sys.argv = ["All_Translation_Volume_Transformed.py", str(p_trn)]
                _mod_trn.main()
            finally:
                sys.argv = argv_bak
                os.chdir(cwd_bak)
            parts.append(TMP / f"{p_trn.stem}_Volume.csv")

        # rotation engine (script-style main, writes to CWD)
        if p_rot is not None:
            argv_bak, cwd_bak = sys.argv[:], os.getcwd()
            try:
                os.chdir(TMP)
                sys.argv = ["All_Rotation_Volume_Transformed.py", str(p_rot)]
                _mod_rot.main()
            finally:
                sys.argv = argv_bak
                os.chdir(cwd_bak)
            parts.append(TMP / f"{p_rot.stem}_Volume.csv")

        # Concatenate
        dfs = []
        for p in parts:
            if not p.exists():
                raise FileNotFoundError(f"Expected temp volume file missing: {p}")
            dfs.append(pd.read_csv(p))

        OUT = tr_csv_path.with_name(tr_csv_path.stem + "_Volume.csv")
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(OUT, index=False)
        else:
            # No curved faces -> write empty CSV with required headers
            pd.DataFrame({"triangle_id": [], "Vcorrection": []}).to_csv(OUT, index=False)

    return OUT


def _enforce_split_identity(vol_csv: Path, tr_csv: Path) -> None:
    """
    Enforce: Vcorrection_scal * scaling_factor == Vcorrection
    by recomputing the per-vertex split with a thickness-weighted rule.
    Works in-place on vol_csv.
    """
    vol_csv = Path(vol_csv).resolve()
    tr_csv = Path(tr_csv).resolve()

    df_vol = pd.read_csv(vol_csv)
    df_tr = pd.read_csv(tr_csv)

    # needed geometry & coeff columns
    coef_cols = {
        "A": _pick(df_tr, ["ABC_new_A","ABC_A"]),
        "B": _pick(df_tr, ["ABC_new_B","ABC_B"]),
        "C": _pick(df_tr, ["ABC_new_C","ABC_C"]),
        "D": _pick(df_tr, ["ABC_new_D","ABC_D"]),
        "E": _pick(df_tr, ["ABC_new_E","ABC_E"]),
        "F": _pick(df_tr, ["ABC_new_F","ABC_F"]),
        "G": _pick(df_tr, ["ABC_new_G","ABC_G"]),
        "H": _pick(df_tr, ["ABC_new_H","ABC_H"]),
        "I": _pick(df_tr, ["ABC_new_I","ABC_I"]),
        "J": _pick(df_tr, ["ABC_new_J","ABC_J"]),
    }
    if any(v is None for v in coef_cols.values()):
        return  # nothing to do

    pt_cols = [
        "A_transformed_x","A_transformed_y","A_transformed_z",
        "B_transformed_x","B_transformed_y","B_transformed_z",
        "C_transformed_x","C_transformed_y","C_transformed_z",
    ]
    if not all(c in df_tr.columns for c in pt_cols):
        return

    sc_col = _pick(df_tr, ["scaling_factor","scale_factor","jacobian","scale"])
    if sc_col is None:
        if all(c in df_tr.columns for c in ["scale_factors1_x","scale_factors1_y","scale_factors1_z"]):
            df_tr["__scaling__"] = (
                pd.to_numeric(df_tr["scale_factors1_x"], errors="coerce") *
                pd.to_numeric(df_tr["scale_factors1_y"], errors="coerce") *
                pd.to_numeric(df_tr["scale_factors1_z"], errors="coerce")
            ).fillna(1.0)
            sc_col = "__scaling__"
        else:
            df_tr["__scaling__"] = 1.0
            sc_col = "__scaling__"

    # Merge geometry
    df_vol["triangle_id"] = pd.to_numeric(df_vol["triangle_id"], errors="raise").astype(int)
    df_tr["triangle_id"] = pd.to_numeric(df_tr["triangle_id"], errors="raise").astype(int)

    geom_cols = ["triangle_id", sc_col] + pt_cols + list(coef_cols.values())
    df = df_vol.merge(df_tr[geom_cols], on="triangle_id", how="left", suffixes=("", "_geom"))

    # Work only on rows with full geometry
    row_has_geom = ~df[pt_cols].isna().any(axis=1)
    if not bool(row_has_geom.any()):
        return

    vA_list, vB_list, vC_list = [], [], []
    vAs_list, vBs_list, vCs_list = [], [], []
    tri_list = []

    for _, row in df.loc[row_has_geom].iterrows():
        V = float(row.get("Vcorrection", 0.0))
        Axyz = np.array([row["A_transformed_x"], row["A_transformed_y"], row["A_transformed_z"]], float)
        Bxyz = np.array([row["B_transformed_x"], row["B_transformed_y"], row["B_transformed_z"]], float)
        Cxyz = np.array([row["C_transformed_x"], row["C_transformed_y"], row["C_transformed_z"]], float)
        coeffs = tuple(float(row[coef_cols[k]]) for k in ["A","B","C","D","E","F","G","H","I","J"])
        s = float(row[sc_col]) if np.isfinite(row[sc_col]) else 1.0
        if not np.isfinite(s) or abs(s) < 1e-30:
            s = 1.0

        try:
            vA, vB, vC = split_patch_volume_thickness_weighted(V, Axyz, Bxyz, Cxyz, coeffs, eps=1e-12)
            S = vA + vB + vC
            if not np.isfinite(S) or abs(S) < 1e-30:
                raise ValueError("ill-conditioned sum")
            scale = V / S if S != 0 else 0.0
            vA *= scale; vB *= scale; vC = V - vA - vB
        except Exception:
            vA = vB = vC = V / 3.0

        tri_list.append(int(row["triangle_id"]))
        # *_scal must be the unscaled values: divide by s
        vA_list.append(vA); vB_list.append(vB); vC_list.append(vC)
        vAs_list.append(vA / s); vBs_list.append(vB / s); vCs_list.append(vC / s)

    if tri_list:
        df_vol = df_vol.set_index("triangle_id")
        df_tmp = pd.DataFrame({
            "triangle_id": tri_list,
            "V_patch_A": vA_list, "V_patch_B": vB_list, "V_patch_C": vC_list,
            "V_patch_A_scal": vAs_list, "V_patch_B_scal": vBs_list, "V_patch_C_scal": vCs_list,
        }).set_index("triangle_id")
        df_vol.update(df_tmp)
        df_vol = df_vol.reset_index()

    # Always recompute sums from current patches
    if {"V_patch_A","V_patch_B","V_patch_C"}.issubset(df_vol.columns):
        df_vol["Vcorrection"] = df_vol[["V_patch_A","V_patch_B","V_patch_C"]].sum(axis=1)
    if {"V_patch_A_scal","V_patch_B_scal","V_patch_C_scal"}.issubset(df_vol.columns):
        df_vol["Vcorrection_scal"] = df_vol[["V_patch_A_scal","V_patch_B_scal","V_patch_C_scal"]].sum(axis=1)

    df_vol.to_csv(vol_csv, index=False)


# Public API

def curved_volume(HC, complex_dtype: str = "vf", **kwargs) -> float:
    """
    Compute enclosed volume using the curved-volume pipeline.

    Parameters
    ----------
    HC : tuple | str
        If complex_dtype == 'vf': (points: (N,3), triangles: (M,3)).
        Alternatively, you may pass a path to an existing .msh file via
        kwargs['msh_path'] (recommended for large meshes).
    complex_dtype : {'vf'}
        Vertex/face mesh expected.
    kwargs :
        coeffs_kwargs : dict
            Tweaks for Part-1 (min_pts, theta_max_deg, plane_*_tol, irls_iters, etc.)
        enforce_split : bool
            Re-run thickness-weighted split with scaling identity (default True).
        workdir : str or Path
            Where to write temporary CSVs (defaults next to the .msh or a tempdir).

    Returns
    -------
    float
        Total enclosed volume (baseline flat-tet volume + sum of curved corrections).
    """
    if complex_dtype != "vf":
        raise NotImplementedError("curved_volume currently supports 'vf' meshes only")

    # resolve mesh path
    msh_path = kwargs.get("msh_path", None)
    tmpdir = None

    if msh_path is None:
        # Expect HC = (points, tris), write a temporary .msh
        if not isinstance(HC, tuple) or len(HC) != 2:
            raise ValueError("HC must be (points, triangles) when no 'msh_path' is provided")
        points, tris = HC
        points = np.asarray(points, float)
        tris = np.asarray(tris, int)
        tmpdir = tempfile.TemporaryDirectory()
        msh_path = Path(tmpdir.name) / "mesh.msh"
        _write_gmsh2(points, tris, msh_path)
        baseline = _baseline_volume(points, tris)
    else:
        # If msh_path is given we still want a baseline (readable only if HC provided as well)
        if isinstance(HC, tuple) and len(HC) == 2:
            baseline = _baseline_volume(np.asarray(HC[0], float), np.asarray(HC[1], int))
        else:
            baseline = 0.0  # unknown; you can pass your own baseline later if desired

    msh_path = Path(msh_path).resolve()

    # choose work directory for CSVs
    workdir = Path(kwargs.get("workdir", msh_path.parent)).resolve()

    # Part 1: coeffs
    coeffs_kwargs = dict(kwargs.get("coeffs_kwargs", {}))
    coeffs_csv = workdir / f"{msh_path.stem}_COEFFS.csv"
    df_coeffs, total_tris, skipped_flat = compute_rimsafe_alltri(str(msh_path), str(coeffs_csv), **coeffs_kwargs)

    # Part 2: transform
    tr_out_csv, err_csv, n_err = process_transform_file(str(coeffs_csv))

    # Part 3: volumes by routing
    vol_csv = _build_volume_csv_by_routing(Path(tr_out_csv))

    # enforce scaling identity for the per-vertex split (optional)
    if bool(kwargs.get("enforce_split", True)):
        _enforce_split_identity(vol_csv, Path(tr_out_csv))

    # read final volume csv and sum Vcorrection
    df_vol = pd.read_csv(vol_csv)
    V_sum = float(np.nansum(df_vol.get("Vcorrection", pd.Series(dtype=float)).values))

    # total = baseline + curved correction
    total = baseline + V_sum

    # cleanup tempdir if used
    if tmpdir is not None:
        tmpdir.cleanup()

    return float(total)
