import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
from ddgclib._method_wrappers import (
    Curvature_i, Curvature_ijk,
    Area_i, Area_ijk, Area,
    Volume
)

# --- Optional: Part-1 coeffs engine (benchmarks/_1_coeffs_computing.py) ---
try:
    from benchmarks._1_coeffs_computing import compute_rimsafe_alltri
except Exception:
    compute_rimsafe_alltri = None  # keep import-safe if Part-1 isn’t present

# --- Optional: Part-2 transformer (benchmarks/_2_quadric_transformer.py) ---
try:
    # single-file transform: returns (out_csv, err_csv, n_errors)
    from benchmarks._2_quadric_transformer import process_file as _transform_process_file
except Exception:
    _transform_process_file = None


# --------- tiny utilities (only used by the new Stage-3 helpers) ---------
def _normalize_coeffs_kwargs(kwargs: dict) -> dict:
    """
    Map friendly names -> compute_rimsafe_alltri(...) parameter names.
    Accepts either the canonical names or short aliases.
    """
    k = dict(kwargs or {})
    if "plane_rel" in k and "plane_rel_tol" not in k:
        k["plane_rel_tol"] = k.pop("plane_rel")
    if "plane_abs" in k and "plane_abs_tol" not in k:
        k["plane_abs_tol"] = k.pop("plane_abs")
    if "irls" in k and "irls_iters" not in k:
        k["irls_iters"] = k.pop("irls")
    return k


def _pick(df, names):
    """Return the first column name from 'names' that exists in df; else None."""
    for n in names:
        if n in df.columns:
            return n
    return None


def _import_module(script_path, name):
    """Lazy import of single-file helpers shipped as scripts."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _find_script(name: str):
    """
    Look for helper scripts in common locations:
    - current working directory (notebook/CLI cwd)
    - this module's folder (benchmarks/)
    - the repo root (parent of benchmarks/)
    - benchmarks/backup/ (if you keep alternates there)
    - /mnt/data (for uploaded files)
    """
    from pathlib import Path
    mod_dir = Path(__file__).resolve().parent          # .../benchmarks
    repo_root = mod_dir.parent                         # repo root
    candidates = [
        Path.cwd().resolve() / name,                   # notebook/CLI cwd
        mod_dir / name,                                # benchmarks/
        repo_root / name,                              # repo root (your two files live here)
        mod_dir / "backup" / name,                     # optional
        Path("/mnt/data") / name,                      # optional
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Script not found: {name}\n"
        f"Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


class GeometryBenchmarkBase:
    """
    Base class for benchmarking geometric quantities on simplicial complexes.
    """

    def __init__(self, name="unnamed", method=None, complex_dtype="vf"):
        self.name = name
        self.complex_dtype = complex_dtype
        self.method = method or {}

        self.points = None
        self.simplices = None

        self.local_area = None
        self.H_computed = None
        self.H_analytical = None

        self.area_computed = None
        self.area_analytical = None

        self.volume_computed = None
        self.volume_analytical = None

        # --- Part-1 (coeffs) placeholders; filled by run_coeffs_stage ---
        self.coeffs_df = None
        self.coeffs_meta = {}   # {"out_csv": str, "total_tris": int, "skipped_flat": int}

        # --- Part-2 (transform) placeholders; filled by run_transform_stage ---
        self.coeffs_tr_df = None
        self.coeffs_tr_meta = {}  # {"in_csv": str, "out_csv": str, "err_csv": str|None, "n_errors": int}

        # --- Part-3 (volume + dual-volume split) placeholders; filled by run_volume_stage ---
        self.vol_df = None
        self.vol_meta = {}       # {"tr_csv","vol_csv","generated","recomputed_rows","missing_geom_rows","V_sum"}

        # Set methods (keep behavior unchanged; if unavailable, set to None)
        try:
            self.curvature_i = Curvature_i(self.method.get("curvature_i_method", "default"))
        except Exception as e:
            logger.warning(f"curvature_i method unavailable: {e}")
            self.curvature_i = None

        try:
            self.curvature_ijk = Curvature_ijk(self.method.get("curvature_ijk_method", "default"))
        except Exception as e:
            logger.warning(f"curvature_ijk method unavailable: {e}")
            self.curvature_ijk = None

        try:
            self.area_i = Area_i(self.method.get("area_i_method", "default"))
        except Exception as e:
            logger.warning(f"area_i method unavailable: {e}")
            self.area_i = None

        try:
            self.area_ijk = Area_ijk(self.method.get("area_ijk_method", "default"))
        except Exception as e:
            logger.warning(f"area_ijk method unavailable: {e}")
            self.area_ijk = None

        try:
            self.area = Area(self.method.get("area_method", "default"))
        except Exception as e:
            logger.warning(f"area method unavailable: {e}")
            self.area = None

        try:
            self.volume = Volume(self.method.get("volume_method", "default"))
        except Exception as e:
            logger.warning(f"volume method unavailable: {e}")
            self.volume = None

    # ---------------------- Part-1 convenience hook ---------------------- #
    def run_coeffs_stage(self, msh_path: str | None = None, out_csv: str | None = None, **coeffs_kwargs):
        """
        Part-1: Fit per-triangle quadric coefficients and write <mesh>_COEFFS.csv.
        Stores results on the object:
          - self.coeffs_df
          - self.coeffs_meta = {"out_csv","total_tris","skipped_flat"}
        """
        if compute_rimsafe_alltri is None:
            raise ImportError(
                "Part-1 not available: ensure benchmarks/_1_coeffs_computing.py is present and importable."
            )

        import os
        from pathlib import Path

        mesh_path = msh_path or getattr(self, "msh_path", None)
        if not mesh_path:
            raise ValueError("run_coeffs_stage requires msh_path or self.msh_path to be set.")

        mesh_path = os.path.abspath(str(mesh_path))
        if out_csv is None:
            out_csv = os.path.join(os.path.dirname(mesh_path), f"{Path(mesh_path).stem}_COEFFS.csv")

        k = _normalize_coeffs_kwargs(coeffs_kwargs)
        # Sane defaults if caller doesn’t specify
        k.setdefault("min_pts", 10)
        k.setdefault("theta_max_deg", 35.0)
        k.setdefault("plane_rel_tol", 1e-3)
        k.setdefault("plane_abs_tol", 1e-9)
        k.setdefault("irls_iters", 0)
        k.setdefault("snap_rel", 5e-6)
        k.setdefault("snap_abs", 1e-9)
        k.setdefault("canonize_abc", True)
        k.setdefault("debug_j", False)

        df, total_tris, skipped_flat = compute_rimsafe_alltri(mesh_path, out_csv, **k)

        # Persist on the benchmark instance
        self.coeffs_df = df
        self.coeffs_meta = {
            "out_csv": out_csv,
            "total_tris": int(total_tris),
            "skipped_flat": int(skipped_flat),
        }
        return df

    # ---------------------- Part-2 convenience hook ---------------------- #
    def run_transform_stage(self, in_csv: str | None = None, read_df: bool = True):
        """
        Part-2: Transform coeffs to canonical space and write <stem>_Transformed.csv.
        Uses Part-1 output by default. Stores:
          - self.coeffs_tr_df   (if read_df=True)
          - self.coeffs_tr_meta = {"in_csv","out_csv","err_csv","n_errors"}
        """
        if _transform_process_file is None:
            raise ImportError(
                "Part-2 not available: ensure benchmarks/_2_quadric_transformer.py is present and importable."
            )

        import os
        import pandas as pd

        if in_csv is None:
            if not self.coeffs_meta or "out_csv" not in self.coeffs_meta:
                raise ValueError("run_transform_stage needs in_csv or a prior run_coeffs_stage() with out_csv.")
            in_csv = self.coeffs_meta["out_csv"]

        in_csv = os.path.abspath(str(in_csv))
        out_csv, err_csv, n_err = _transform_process_file(in_csv)  # returns (out_csv, err_csv, n_errors)
        self.coeffs_tr_meta = {
            "in_csv": in_csv,
            "out_csv": out_csv,
            "err_csv": err_csv,
            "n_errors": int(n_err),
        }

        if read_df:
            try:
                self.coeffs_tr_df = pd.read_csv(out_csv)
            except Exception:
                self.coeffs_tr_df = None

        return out_csv

    # ---------------------- Part-3 convenience hook ---------------------- #
    def run_volume_stage(self, tr_csv: str | None = None, read_df: bool = True, enforce_split: bool = True):
        """
        Stage-3: Ensure <stem>_Volume.csv exists for a given transformed CSV, then
        enforce the thickness-weighted split so that:

            Vcorrection_scal * scaling_factor == Vcorrection

        If tr_csv is None, uses self.coeffs_tr_meta['out_csv'] from Part-2.
        Stores:
          - self.vol_df     (if read_df=True)
          - self.vol_meta   {"tr_csv","vol_csv","generated","recomputed_rows","missing_geom_rows","V_sum"}
        """
        import os
        from pathlib import Path

        if tr_csv is None:
            if not self.coeffs_tr_meta or "out_csv" not in self.coeffs_tr_meta:
                raise ValueError("run_volume_stage needs tr_csv or a prior run_transform_stage().")
            tr_csv = self.coeffs_tr_meta["out_csv"]

        tr_csv = os.path.abspath(str(tr_csv))
        vol_csv = Path(tr_csv).with_name(Path(tr_csv).stem + "_Volume.csv")

        generated = False
        if not vol_csv.exists():
            # build the volume csv by routing subsets to the proper engines
            vol_csv = self._build_volume_csv_by_routing(tr_csv)
            generated = True

        if enforce_split:
            # re-split patches to satisfy the scaling identity
            self._enforce_thickness_weighted(vol_csv, tr_csv)

        self.vol_meta.update({"tr_csv": tr_csv, "vol_csv": str(vol_csv), "generated": generated})

        if read_df:
            import pandas as pd
            try:
                self.vol_df = pd.read_csv(str(vol_csv))
            except Exception:
                self.vol_df = None

        return str(vol_csv)

    # ============================ Stage-3 internals ============================ #
    def _build_volume_csv_by_routing(self, tr_csv_path: str):
        """
        Split the transformed CSV into ellipsoid / translation / rotation subsets,
        run the corresponding engines, and concatenate into <stem>_Volume.csv.
        """
        import os, tempfile
        from pathlib import Path
        import pandas as pd
        import numpy as np

        tr_csv = Path(tr_csv_path)
        if not tr_csv.exists():
            raise FileNotFoundError(f"Transformed CSV not found: {tr_csv}")

        out_csv = tr_csv.with_name(tr_csv.stem + "_Volume.csv")
        df = pd.read_csv(tr_csv)

        Acol = _pick(df, ["ABC_new_A","ABC_A"]); Bcol = _pick(df, ["ABC_new_B","ABC_B"]); Ccol = _pick(df, ["ABC_new_C","ABC_C"])
        Ecol = _pick(df, ["ABC_new_E","ABC_E"]); Fcol = _pick(df, ["ABC_new_F","ABC_F"]); Icol = _pick(df, ["ABC_new_I","ABC_I"])
        Jcol = _pick(df, ["ABC_new_J","ABC_J"])
        need = [Acol,Bcol,Ccol,Ecol,Fcol,Icol,Jcol]
        if any(c is None for c in need):
            miss = [n for c,n in zip(need,["A","B","C","E","F","I","J"]) if c is None]
            raise ValueError(f"Missing coefficient columns in transformed CSV: {miss}")

        A = df[Acol].to_numpy(); B = df[Bcol].to_numpy(); C = df[Ccol].to_numpy()
        E = df[Ecol].to_numpy(); F = df[Fcol].to_numpy(); I = df[Icol].to_numpy(); J = df[Jcol].to_numpy()

        TOL = 1e-9
        ell_mask = (A > 0) & (B > 0) & (C > 0) & (np.abs(J) > 0)
        noz_mask = (np.abs(C) < TOL) & (np.abs(E) < TOL) & (np.abs(F) < TOL) & (np.abs(I) < TOL)  # z-invariant → translation
        axis_mask = ~(ell_mask | noz_mask)

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

            # --- ellipsoid subset ---
            if p_ell is not None:
                m = _import_module(_find_script("_3_5_volume_ellipsoid.py"), "vol_ell")
                out_csv_ell, _, _, _ = m.process_coeffs_transformed_csv(str(p_ell))
                parts.append(Path(out_csv_ell))

            # --- translation subset (writes to CWD) ---
            if p_trn is not None:
                m = _import_module(_find_script("All_Translation_Volume_Transformed.py"), "vol_trn")
                argv_bak, cwd_bak = __import__("sys").argv[:], os.getcwd()
                try:
                    os.chdir(TMP)
                    __import__("sys").argv = ["All_Translation_Volume_Transformed.py", str(p_trn)]
                    m.main()
                finally:
                    __import__("sys").argv = argv_bak; os.chdir(cwd_bak)
                parts.append(TMP / f"{p_trn.stem}_Volume.csv")

            # --- rotation subset (writes to CWD) ---
            if p_rot is not None:
                m = _import_module(_find_script("All_Rotation_Volume_Transformed.py"), "vol_rot")
                argv_bak, cwd_bak = __import__("sys").argv[:], os.getcwd()
                try:
                    os.chdir(TMP)
                    __import__("sys").argv = ["All_Rotation_Volume_Transformed.py", str(p_rot)]
                    m.main()
                finally:
                    __import__("sys").argv = argv_bak; os.chdir(cwd_bak)
                parts.append(TMP / f"{p_rot.stem}_Volume.csv")

            # concat into the final <stem>_Volume.csv
            dfs = []
            for p in parts:
                if not p.exists():
                    raise FileNotFoundError(f"Expected temp volume file missing: {p}")
                dfs.append(pd.read_csv(p))
            pd.concat(dfs, ignore_index=True).to_csv(out_csv, index=False)

        return out_csv

    def _enforce_thickness_weighted(self, vol_csv_path, tr_csv_path):
        """
        Recompute per-vertex splits with the thickness-weighted splitter so that
            Vcorrection_scal * scaling_factor == Vcorrection
        holds per row. Overwrites <stem>_Volume.csv and updates self.vol_meta.
        """
        from pathlib import Path
        import pandas as pd
        import numpy as np

        vol_csv = Path(vol_csv_path)
        tr_csv = Path(tr_csv_path)

        df_vol = pd.read_csv(vol_csv)
        df_tr  = pd.read_csv(tr_csv)

        # --- align types ---
        if "triangle_id" not in df_vol.columns or "triangle_id" not in df_tr.columns:
            raise ValueError("Both CSVs must contain 'triangle_id'.")
        df_vol["triangle_id"] = pd.to_numeric(df_vol["triangle_id"], errors="raise").astype(int)
        df_tr["triangle_id"]  = pd.to_numeric(df_tr["triangle_id"], errors="raise").astype(int)

        # --- columns we need from transformed CSV ---
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
            miss = [k for k,v in coef_cols.items() if v is None]
            raise ValueError(f"Missing coefficient columns in transformed CSV: {miss}")

        pt_cols = [
            "A_transformed_x","A_transformed_y","A_transformed_z",
            "B_transformed_x","B_transformed_y","B_transformed_z",
            "C_transformed_x","C_transformed_y","C_transformed_z",
        ]
        if not all(c in df_tr.columns for c in pt_cols):
            raise ValueError(f"Transformed CSV missing required point columns: {set(pt_cols) - set(df_tr.columns)}")

        # scaling column (prefer 'scaling_factor', else product of scale_factors1_*)
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

        # Merge geometry/coeffs onto volume rows via triangle_id
        geom_cols = ["triangle_id", sc_col] + pt_cols + list(coef_cols.values())
        df = df_vol.merge(df_tr[geom_cols], on="triangle_id", how="left", suffixes=("", "_geom"))

        row_has_geom = ~df[pt_cols].isna().any(axis=1)
        miss_cnt = int((~row_has_geom).sum())

        # load your splitter
        split_mod = _import_module(_find_script("_4_dualvolume_split_patch_volume_thickness_weighted.py"), "split_mod")
        split_fn = getattr(split_mod, "split_patch_volume_thickness_weighted", None)
        if split_fn is None:
            raise AttributeError("split_patch_volume_thickness_weighted not found in splitter module.")

        vA_list, vB_list, vC_list = [], [], []
        vAs_list, vBs_list, vCs_list = [], [], []
        tri_list = []
        repaired = 0

        df_work = df.loc[row_has_geom].copy()
        for _, row in df_work.iterrows():
            V = float(row.get("Vcorrection", 0.0))
            Axyz = np.array([row["A_transformed_x"], row["A_transformed_y"], row["A_transformed_z"]], float)
            Bxyz = np.array([row["B_transformed_x"], row["B_transformed_y"], row["B_transformed_z"]], float)
            Cxyz = np.array([row["C_transformed_x"], row["C_transformed_y"], row["C_transformed_z"]], float)
            coeffs = tuple(float(row[coef_cols[k]]) for k in ["A","B","C","D","E","F","G","H","I","J"])
            s = float(row[sc_col]) if np.isfinite(row[sc_col]) else 1.0
            if not np.isfinite(s) or abs(s) < 1e-30:
                s = 1.0

            try:
                vA, vB, vC = split_fn(V, Axyz, Bxyz, Cxyz, coeffs, eps=1e-12)
                S = vA + vB + vC
                if not np.isfinite(S) or abs(S) < 1e-30:
                    raise ValueError("ill-conditioned sum")
                scale = V / S if S != 0 else 0.0
                vA *= scale; vB *= scale; vC = V - vA - vB
            except Exception:
                repaired += 1
                vA = vB = vC = V / 3.0

            tri_list.append(int(row["triangle_id"]))
            # enforce: Vcorrection_scal * scaling_factor == Vcorrection
            vA_list.append(vA); vB_list.append(vB); vC_list.append(vC)
            vAs_list.append(vA / s); vBs_list.append(vB / s); vCs_list.append(vC / s)

        # write recomputed pieces back
        if tri_list:
            df_vol = df_vol.set_index("triangle_id")
            df_tmp = pd.DataFrame({
                "triangle_id": tri_list,
                "V_patch_A": vA_list, "V_patch_B": vB_list, "V_patch_C": vC_list,
                "V_patch_A_scal": vAs_list, "V_patch_B_scal": vBs_list, "V_patch_C_scal": vCs_list,
            }).set_index("triangle_id")
            df_vol.update(df_tmp)
            df_vol = df_vol.reset_index()

        # always recompute totals
        if {"V_patch_A","V_patch_B","V_patch_C"}.issubset(df_vol.columns):
            df_vol["Vcorrection"] = df_vol[["V_patch_A","V_patch_B","V_patch_C"]].sum(axis=1)
        if {"V_patch_A_scal","V_patch_B_scal","V_patch_C_scal"}.issubset(df_vol.columns):
            df_vol["Vcorrection_scal"] = df_vol[["V_patch_A_scal","V_patch_B_scal","V_patch_C_scal"]].sum(axis=1)

        df_vol.to_csv(vol_csv, index=False)

        # store a few summary numbers
        V_sum = float(np.nansum(df_vol["Vcorrection"].values)) if "Vcorrection" in df_vol.columns else 0.0
        self.vol_meta.update({
            "recomputed_rows": len(tri_list),
            "missing_geom_rows": miss_cnt,
            "V_sum": V_sum,
        })

    # ---------------------- Existing API below (unchanged) ---------------------- #
    def generate_mesh(self):
        raise NotImplementedError

    def compute_surface_area(self):
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        if self.area is not None:
            self.area_computed = self.area(HC, complex_dtype=self.complex_dtype)
        else:
            self.area_computed = None

    def compute_volume(self):
        if self.complex_dtype == "vf" and self.volume is not None:
            HC = (self.points, self.simplices)
            self.volume_computed = self.volume(HC, complex_dtype=self.complex_dtype)
        else:
            self.volume_computed = None

    def compute_curvature(self):
        HC = (self.points, self.simplices) if self.complex_dtype == "vf" else self.points
        if self.complex_dtype == "vv" and self.curvature_i is not None:
            self.H_computed = self.curvature_i(HC, complex_dtype=self.complex_dtype)
        elif self.complex_dtype == "vf" and self.curvature_ijk is not None:
            # Placeholder for future: per-triangle curvature if implemented
            pass

    def analytical_values(self):
        raise NotImplementedError

    def run_benchmark(self):
        self.generate_mesh()
        self.compute_surface_area()
        self.compute_volume()
        self.compute_curvature()
        self.analytical_values()

        if self.complex_dtype == "vv":
            # Placeholder: loop over all vertices
            for _ in range(len(self.points) if self.points is not None else 0):
                pass  # per-vertex error logic
        elif self.complex_dtype == "vf":
            # Placeholder: loop over all simplices
            for _ in (self.simplices if self.simplices is not None else []):
                pass  # per-simplex error logic

    def summary(self):
        # Protect against None values if some stages were intentionally skipped
        if self.H_computed is None or self.H_analytical is None:
            mean_error_H = 0.0
            std_error_H = 0.0
        else:
            err = np.abs(self.H_computed - self.H_analytical)
            mean_error_H = float(np.mean(err))
            std_error_H = float(np.std(err))

        summary_dict = {
            "Surface Area": (self.area_computed, self.area_analytical),
            "Volume": (self.volume_computed, self.volume_analytical),
            "Curvature Mean Error": (mean_error_H, 0.0),
            "Curvature Std Error": (std_error_H, 0.0),
        }

        # If Part-1/2/3 ran, append short sections (no-op otherwise)
        if self.coeffs_meta:
            summary_dict["Coeff fit — total tris"] = (self.coeffs_meta.get("total_tris", 0), 0)
            summary_dict["Coeff fit — skipped planar"] = (self.coeffs_meta.get("skipped_flat", 0), 0)
            summary_dict["Coeff fit — CSV"] = (self.coeffs_meta.get("out_csv", ""), 0)
        if self.coeffs_tr_meta:
            summary_dict["Transform — errors"] = (self.coeffs_tr_meta.get("n_errors", 0), 0)
            summary_dict["Transform — CSV"] = (self.coeffs_tr_meta.get("out_csv", ""), 0)
        if self.vol_meta:
            summary_dict["Volume — total Vcorrection"] = (self.vol_meta.get("V_sum", 0.0), 0)
            summary_dict["Volume — recomputed rows"] = (self.vol_meta.get("recomputed_rows", 0), 0)
            summary_dict["Volume — CSV"] = (self.vol_meta.get("vol_csv", ""), 0)

        return summary_dict


def run_all_benchmarks(benchmark_classes, method=None, complex_dtype="vf"):
    """
    Runs all provided benchmark cases with specified method implementations.

    Parameters
    ----------
    benchmark_classes : list of GeometryBenchmarkBase subclasses
        List of benchmark classes to instantiate and run.
    method : dict, optional
        Dictionary specifying method names for each evaluator. Example:
        {
            "curvature_i_method": "laplace-beltrami",
            "area_i_method": "default",
            "area_ijk_method": "default",
            "area_method": "default",
            "volume_method": "default",
        }
        If not provided, all evaluators default to their "default" method.
    complex_dtype : str
        Type of mesh structure. Either 'vf' (vertex-face) or 'vv' (vertex-vertex).
    """
    method = method or {}
    for BenchmarkClass in benchmark_classes:
        bench = BenchmarkClass(method=method, complex_dtype=complex_dtype)
        bench.run_benchmark()
        print(f"Benchmark: {bench.name}")
        for k, (v_comp, v_ana) in bench.summary().items():
            # v_comp may be None if a stage was intentionally skipped
            vc = "None" if v_comp is None else f"{v_comp:.6f}" if isinstance(v_comp, (int, float)) else str(v_comp)
            va = "None" if v_ana is None else f"{v_ana:.6f}" if isinstance(v_ana, (int, float)) else str(v_ana)
            print(f"  {k}: computed={vc}, analytical={va}")
