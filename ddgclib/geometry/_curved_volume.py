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
import time            # timers
import shutil          # kept
import tempfile
import importlib.util
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
import io


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
    # silence noisy imports (those !!!!debug: ... lines)
    spec = importlib.util.spec_from_file_location(mod_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    # silence stdout/stderr produced DURING import
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)  # type: ignore
    return mod


# Part-1 & Part-2 entry points (library versions)

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

def _silence_ctx():
    # local context to swallow noisy per-triangle prints from engines
    return redirect_stdout(io.StringIO())

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
    # Heuristic routing:
    ell_mask = (A > 0) & (B > 0) & (C > 0) & (np.abs(J) > 0)           # ellipsoid-like
    noz_mask = (np.abs(C) < TOL) & (np.abs(E) < TOL) & (np.abs(F) < TOL) & (np.abs(I) < TOL)  # no z -> translation
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

        # ellipsoid engine
        if p_ell is not None:
            with _silence_ctx(), redirect_stderr(io.StringIO()):
                out_csv_ell, _, _, _ = process_coeffs_transformed_csv(str(p_ell))
            sub_df = pd.read_csv(out_csv_ell)
            sub_df["Type"] = "ellipsiod"
            tmp_ell = Path(out_csv_ell).with_name(Path(out_csv_ell).stem + "_with_type.csv")
            sub_df.to_csv(tmp_ell, index=False)
            parts.append(tmp_ell)

        # translation engine
        if p_trn is not None:
            argv_bak, cwd_bak = sys.argv[:], os.getcwd()
            try:
                os.chdir(TMP)
                sys.argv = ["All_Translation_Volume_Transformed.py", str(p_trn)]
                with _silence_ctx(), redirect_stderr(io.StringIO()):
                    _mod_trn.main()
            finally:
                sys.argv = argv_bak
                os.chdir(cwd_bak)
            trn_out = TMP / f"{p_trn.stem}_Volume.csv"
            sub_df = pd.read_csv(trn_out)
            sub_df["Type"] = "translation"
            tmp_trn = trn_out.with_name(trn_out.stem + "_with_type.csv")
            sub_df.to_csv(tmp_trn, index=False)
            parts.append(tmp_trn)

        # rotation engine
        if p_rot is not None:
            argv_bak, cwd_bak = sys.argv[:], os.getcwd()
            try:
                os.chdir(TMP)
                sys.argv = ["All_Rotation_Volume_Transformed.py", str(p_rot)]
                with _silence_ctx(), redirect_stderr(io.StringIO()):
                    _mod_rot.main()
            finally:
                sys.argv = argv_bak
                os.chdir(cwd_bak)
            rot_out = TMP / f"{p_rot.stem}_Volume.csv"
            sub_df = pd.read_csv(rot_out)
            sub_df["Type"] = "rotation"
            tmp_rot = rot_out.with_name(rot_out.stem + "_with_type.csv")
            sub_df.to_csv(tmp_rot, index=False)
            parts.append(tmp_rot)

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
            pd.DataFrame({"triangle_id": [], "Vcorrection": [], "Type": []}).to_csv(OUT, index=False)

    # add A_id, B_id, C_id from the transformed CSV (if present)
    out_df = pd.read_csv(OUT)
    merge_cols = ["triangle_id"]
    extra_cols = []
    for col in ["A_id", "B_id", "C_id"]:
        if col in df.columns:
            extra_cols.append(col)
    if extra_cols:
        df_ids = df[merge_cols + extra_cols].copy()
        # FIX: avoid A_id_x / A_id_y by forcing suffixes and then coalescing
        out_df = out_df.merge(df_ids, on="triangle_id", how="left", suffixes=("", "_y"))
        for col in extra_cols:
            ycol = f"{col}_y"
            if ycol in out_df.columns:
                out_df[col] = out_df[col].fillna(out_df[ycol])
                out_df.drop(columns=[ycol], inplace=True)
        out_df.to_csv(OUT, index=False)

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

    _t_split0 = time.time()
    for _, row in df.loc[row_has_geom].iterrows():
        V = float(row.get("Vcorrection", 0.0))
        Axyz = np.array([row["A_transformed_x"], row["A_transformed_y"], row["A_transformed_z"]], float)
        Bxyz = np.array([row["B_transformed_x"], row["B_transformed_y"], row["B_transformed_z"]], float)
        Cxyz = np.array([row["C_transformed_x"], row["C_transformed_y"], row["C_transformed_z"]], float)
        coeffs = tuple(float(row[c]) for c in [coef_cols[k] for k in ["A","B","C","D","E","F","G","H","I","J"]])
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

    _t_split1 = time.time()
    print(f"[CV] split-enforce(thickness) loop: {(_t_split1 - _t_split0):.3f}s for {len(tri_list)} tris", flush=True)

    df_vol.to_csv(vol_csv, index=False)


def _write_dualvolume_csv_from_vol_csv(vol_csv: Path) -> Path:
    """
    From *_COEFFS_Transformed_Volume.csv build *_COEFFS_Transformed_DualVolume.csv
    as: PointID, DualVolume, DualArea, Type
    """
    vol_csv = Path(vol_csv).resolve()
    df = pd.read_csv(vol_csv)

    # target name: replace trailing "_Volume" with "_DualVolume"
    stem = vol_csv.stem
    if stem.endswith("_Volume"):
        dual_stem = stem[:-len("_Volume")] + "_DualVolume"
    else:
        dual_stem = stem + "_DualVolume"
    out_path = vol_csv.with_name(dual_stem + ".csv")

    needed_ids = ["A_id", "B_id", "C_id"]
    needed_vols = ["V_patch_A", "V_patch_B", "V_patch_C"]

    # Area columns may or may not be present
    area_cols = ["A_patch_A", "A_patch_B", "A_patch_C"]
    has_area = all(col in df.columns for col in area_cols)

    has_type_col = "Type" in df.columns

    # load *_COEFFS.csv to know which triangles to ignore
    ignore_tris = set()
    coeffs_csv_name = vol_csv.name.replace("_COEFFS_Transformed_Volume.csv", "_COEFFS.csv")
    coeffs_csv_path = vol_csv.with_name(coeffs_csv_name)
    if coeffs_csv_path.exists():
        try:
            df_coeffs = pd.read_csv(coeffs_csv_path)
            if (
                "triangle_id" in df_coeffs.columns
                and "Max_Residual_ABC" in df_coeffs.columns
                and "Residual_Threshold" in df_coeffs.columns
            ):
                for _, crow in df_coeffs.iterrows():
                    try:
                        tid = int(crow["triangle_id"])
                        mres = float(crow["Max_Residual_ABC"])
                        thr = float(crow["Residual_Threshold"])
                        if np.isfinite(mres) and np.isfinite(thr) and (mres > thr):
                            ignore_tris.add(tid)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[CV] dual-volume: could not read coeffs for residual filtering: {e}", flush=True)

    # if we don't have the columns, just emit empty file (but keep DualArea column!)
    if not all(col in df.columns for col in needed_ids) or not all(col in df.columns for col in needed_vols):
        pd.DataFrame({"PointID": [], "DualVolume": [], "DualArea": [], "Type": []}).to_csv(out_path, index=False)
        print(f"[CV] dual-volume: missing columns in {vol_csv.name}, wrote empty {out_path.name}", flush=True)
        return out_path

    acc_vol: dict[int, float] = {}
    acc_area: dict[int, float] = {}
    acc_type: dict[int, str] = {}

    has_tri_col = "triangle_id" in df.columns

    for _, row in df.iterrows():
        tri_bad = False
        tri_type = None
        if has_tri_col:
            try:
                tri_id = int(row["triangle_id"])
                if tri_id in ignore_tris:
                    tri_bad = True
            except Exception:
                tri_bad = False

        if has_type_col:
            tval = row["Type"]
            if isinstance(tval, str) and tval.strip():
                tri_type = tval.strip()
            else:
                tri_type = None
        else:
            tri_type = None

        # A vertex
        pid = row["A_id"]
        if pd.notna(pid):
            pid_i = int(pid)
            if not tri_bad:
                v = row["V_patch_A"]
                if pd.notna(v):
                    acc_vol[pid_i] = acc_vol.get(pid_i, 0.0) + float(v)
                if has_area:
                    a = row["A_patch_A"]
                    if pd.notna(a):
                        acc_area[pid_i] = acc_area.get(pid_i, 0.0) + float(a)
                if tri_type is not None and pid_i not in acc_type:
                    acc_type[pid_i] = tri_type

        # B vertex
        pid = row["B_id"]
        if pd.notna(pid):
            pid_i = int(pid)
            if not tri_bad:
                v = row["V_patch_B"]
                if pd.notna(v):
                    acc_vol[pid_i] = acc_vol.get(pid_i, 0.0) + float(v)
                if has_area:
                    a = row["A_patch_B"]
                    if pd.notna(a):
                        acc_area[pid_i] = acc_area.get(pid_i, 0.0) + float(a)
                if tri_type is not None and pid_i not in acc_type:
                    acc_type[pid_i] = tri_type

        # C vertex
        pid = row["C_id"]
        if pd.notna(pid):
            pid_i = int(pid)
            if not tri_bad:
                v = row["V_patch_C"]
                if pd.notna(v):
                    acc_vol[pid_i] = acc_vol.get(pid_i, 0.0) + float(v)
                if has_area:
                    a = row["A_patch_C"]
                    if pd.notna(a):
                        acc_area[pid_i] = acc_area.get(pid_i, 0.0) + float(a)
                if tri_type is not None and pid_i not in acc_type:
                    acc_type[pid_i] = tri_type

    # unify keys and enforce DualArea==0 -> DualVolume=0
    all_pids = sorted(set(acc_vol.keys()) | set(acc_area.keys()) | set(acc_type.keys()))
    rows = []
    for pid in all_pids:
        dual_area = acc_area.get(pid, 0.0) if has_area else 0.0
        dual_vol  = acc_vol.get(pid, 0.0)

        # >>> ADDED RULE: if DualArea == 0, force DualVolume = 0 <<<
        # if has_area and (dual_area == 0.0 or not np.isfinite(dual_area)):
        #     dual_vol = 0.0

        rows.append({
            "PointID": pid,
            "DualVolume": dual_vol,
            "DualArea": dual_area,
            "Type": acc_type.get(pid, ""),
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[CV] dual-volume: wrote {out_path.name} ({len(rows)} points), ignored {len(ignore_tris)} bad tris", flush=True)
    return out_path


# Public API

def curved_volume(HC, complex_dtype: str = "vf", **kwargs) -> float:
    """
    Compute enclosed volume using the curved-volume pipeline.
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
        if isinstance(HC, tuple) and len(HC) == 2:
            baseline = _baseline_volume(np.asarray(HC[0], float), np.asarray(HC[1], int))
        else:
            baseline = 0.0

    msh_path = Path(msh_path).resolve()

    # choose work directory for CSVs
    workdir = Path(kwargs.get("workdir", msh_path.parent)).resolve()

    # Part 1: coeffs
    t1 = time.time()
    coeffs_kwargs = dict(kwargs.get("coeffs_kwargs", {}))
    coeffs_csv = workdir / f"{msh_path.stem}_COEFFS.csv"
    df_coeffs, total_tris, skipped_flat = compute_rimsafe_alltri(str(msh_path), str(coeffs_csv), **coeffs_kwargs)
    print(f"[CV] part-1 done in {time.time()-t1:.3f}s → {coeffs_csv} (tris={total_tris}, skipped_flat={skipped_flat})", flush=True)

    # EARLY OUT FOR ALL-PLANAR CASE
    if skipped_flat == total_tris:
        empty_vol_csv = workdir / f"{msh_path.stem}_COEFFS_Transformed_Volume.csv"
        pd.DataFrame({
            "triangle_id": [],
            "Vcorrection": [],
            "A_curved": [],
            "A_id": [],
            "B_id": [],
            "C_id": [],
            "Type": [],
        }).to_csv(empty_vol_csv, index=False)
        # also write empty dualvolume csv (WITH DualArea now, and Type)
        empty_dual_csv = workdir / f"{msh_path.stem}_COEFFS_Transformed_DualVolume.csv"
        pd.DataFrame({"PointID": [], "DualVolume": [], "DualArea": [], "Type": []}).to_csv(empty_dual_csv, index=False)
        print(f"[CV] all {total_tris} triangles planar/too-small; skipping part-2/3, wrote {empty_vol_csv.name} and {empty_dual_csv.name}", flush=True)
        if tmpdir is not None:
            tmpdir.cleanup()
        return float(baseline)

    # Part 2: transform (SILENCED)
    t2 = time.time()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        tr_out_csv, err_csv, n_err = process_transform_file(str(coeffs_csv))
    print(f"[CV] part-2 done in {time.time()-t2:.3f}s → {tr_out_csv} (err_csv={err_csv}, n_err={n_err})", flush=True)

    # Part 3: volumes by routing
    t3 = time.time()
    vol_csv = _build_volume_csv_by_routing(Path(tr_out_csv))
    print(f"[CV] part-3 done in {time.time()-t3:.3f}s → {vol_csv}", flush=True)

    # enforce scaling identity
    if bool(kwargs.get("enforce_split", True)):
        t4 = time.time()
        _enforce_split_identity(vol_csv, Path(tr_out_csv))
        print(f"[CV] split-enforce done in {time.time()-t4:.3f}s", flush=True)

    # dual-volume per point (now also dual-area, with residual filter, and Type)
    _write_dualvolume_csv_from_vol_csv(Path(vol_csv))

    # read final volume csv and sum Vcorrection
    t5 = time.time()
    df_vol = pd.read_csv(vol_csv)
    V_sum = float(np.nansum(df_vol.get("Vcorrection", pd.Series(dtype=float)).values))
    print(f"[CV] sum+cleanup done in {time.time()-t5:.3f}s", flush=True)

    total = baseline + V_sum

    if tmpdir is not None:
        tmpdir.cleanup()

    return float(total)
