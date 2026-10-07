"""
datasets.py
==========

Clean CSV schema + loader for binary electrolyte-water activity/osmotic data
used to fit the e-NRTL model in ``enrtl_binary.py``.

CSV schema (one row per molality point):

    system,m,gamma_pm,phi,a_w,T,source,sigma_m
    KCl,0.1,0.770,,,298.15,RobinsonStokes1959,0.004
    ...

Columns
-------
    system   : salt label (KCl, KOH, H2SO4, ...)         [required]
    m        : molality, mol/kg water                     [required]
    gamma_pm : mean ionic activity coefficient (or blank) [optional]
    phi      : osmotic coefficient (or blank)             [optional]
    a_w      : water activity (or blank)                  [optional]
    T        : temperature, K (default 298.15)            [optional]
    source   : citation key (see data/SOURCES.md)         [recommended]
    sigma_m  : 1-sigma uncertainty on the fitted obs.     [optional]

At least one of {gamma_pm, phi, a_w} must be present per row.  Blank cells load
as NaN and are skipped by the residual function.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Registry of known dataset files per system.  Files are OPTIONAL -- only those
# present on disk are loaded (see ``available_datasets``), so new systems (e.g.
# KOH once extracted) are picked up automatically without code changes.
DATASET_FILES = {
    "KOH":   "koh_hamer_wu.csv",          # Hamer & Wu 1972, Table 31 (to 20 m)
    "H2SO4": "h2so4_que2011_fig10.csv",   # Que2011 Fig.10 (digitised)
    "KCl":   "kcl_hamer_wu.csv",          # Hamer & Wu 1972, Table 28 (to sat.)
}


def available_datasets() -> dict:
    """Return {system: absolute_path} for the dataset files that exist."""
    out = {}
    for system, fn in DATASET_FILES.items():
        p = os.path.join(_DATA_DIR, fn)
        if os.path.exists(p):
            out[system] = p
    return out


def _f(x):
    x = (x or "").strip()
    return float(x) if x not in ("", "nan", "NaN", "NA") else np.nan


def load_csv(path: str, system: str | None = None) -> dict:
    """
    Load a schema CSV into a dict of arrays, optionally filtered to one system.

    Returns {'m','gamma_pm','phi','a_w','T','source'(list),'sigma_m'} with the
    numeric fields as float arrays, sorted by ascending m.  If ``system`` is
    None and the file holds several, returns a dict-of-dicts keyed by system.
    """
    by_sys: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            s = (row.get("system") or "").strip()
            if not s or s.startswith("#"):
                continue
            d = by_sys[s]
            d["m"].append(_f(row.get("m")))
            d["gamma_pm"].append(_f(row.get("gamma_pm")))
            d["phi"].append(_f(row.get("phi")))
            d["a_w"].append(_f(row.get("a_w")))
            d["T"].append(_f(row.get("T")) if row.get("T") else 298.15)
            d["sigma_m"].append(_f(row.get("sigma_m")))
            d["source"].append((row.get("source") or "").strip())

    def _finish(d):
        order = np.argsort(np.asarray(d["m"], float))
        out = {}
        for k in ("m", "gamma_pm", "phi", "a_w", "T", "sigma_m"):
            out[k] = np.asarray(d[k], float)[order]
        out["source"] = [d["source"][i] for i in order]
        return out

    if system is not None:
        if system not in by_sys:
            raise KeyError(f"system {system!r} not in {path}; "
                           f"have {sorted(by_sys)}")
        return _finish(by_sys[system])
    if len(by_sys) == 1:
        return _finish(next(iter(by_sys.values())))
    return {s: _finish(d) for s, d in by_sys.items()}
