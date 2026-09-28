"""Load experimental/fitted reference data from Lunowa et al. (2022).

Data source: https://github.com/Lunowa/dynamic-capillary-rise
Local path:  data/caprise/dynamic-capillary-rise-main/data/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "caprise"
_LUNOWA_DATA = _DATA_ROOT / "dynamic-capillary-rise-main" / "data"

# Available fluids and tube radii (mm)
_FLUID_RADII: dict[str, list[float]] = {
    "water": [0.375, 0.5, 0.65],
    "soltrol": [0.375, 0.5, 0.65],
    "glycerol": [0.25, 0.5, 1.0],
}


def _radius_label(R_mm: float) -> str:
    """Format radius for filename lookup, e.g. 0.375 -> '0.375mm'.

    Preserves at least one decimal place: 1.0 -> '1.0mm', not '1mm'.
    """
    s = f"{R_mm:g}"
    if "." not in s:
        s += ".0"
    return s + "mm"


def load_sample_data(fluid: str, R_mm: float) -> dict[str, np.ndarray]:
    """Load sample CSV data for a fluid/radius combination.

    Returns dict with keys:
        't_s':    time in seconds (ndarray)
        'h_m':    rise height in metres (ndarray)
        'h_cm':   rise height in cm (ndarray)
        'ca_deg': contact angle in degrees (ndarray, NaN where missing)
        'ca_rad': contact angle in radians (ndarray, NaN where missing)
    """
    fname = f"{fluid}_R{_radius_label(R_mm)}0.csv"
    path = _LUNOWA_DATA / fname
    if not path.exists():
        raise FileNotFoundError(f"Sample data not found: {path}")

    # Skip the comment header line; columns are rise,time,CA
    raw = np.genfromtxt(path, delimiter=",", skip_header=2, filling_values=np.nan)

    h_cm = raw[:, 0]
    t_s = raw[:, 1]
    ca_deg = raw[:, 2]

    return {
        "t_s": t_s,
        "h_m": h_cm * 1e-2,
        "h_cm": h_cm,
        "ca_deg": ca_deg,
        "ca_rad": np.deg2rad(ca_deg),
    }


def load_extended_model(fluid: str, R_mm: float) -> dict[str, np.ndarray] | None:
    """Load extended model solutions (nondimensional).

    Returns None if the file does not exist (e.g. glycerol has no extended
    model data).  Otherwise returns a dict whose keys mirror the CSV
    columns: 't', 'h_h0', 'v_h0', 'ca_h0', ... , 'h_b1', 'v_b1', 'ca_b1'.
    """
    fname = f"{fluid}_R{_radius_label(R_mm)}_ext.csv"
    path = _LUNOWA_DATA / fname
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {col: raw[:, i] for i, col in enumerate(header)}


def load_fluid_properties() -> dict:
    """Load fluid_data.json.  Returns dict keyed by fluid name."""
    path = _DATA_ROOT / "fluid_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Fluid properties not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_fitted_summary() -> dict:
    """Load fitted_data.json summary of fitted parameters.

    Handles JSON files with ``//``-style line comments.
    """
    path = _DATA_ROOT / "fitted_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Fitted summary not found: {path}")
    with open(path, encoding="utf-8") as f:
        # Strip JS-style line comments before parsing
        lines = [
            line for line in f if not line.lstrip().startswith("//")
        ]
    return json.loads("".join(lines))


def load_all_sample_data() -> dict[str, dict[float, dict[str, np.ndarray]]]:
    """Load sample data for ALL fluids and radii.

    Returns nested dict ``{fluid: {R_mm: data_dict}}``.
    """
    result: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    for fluid, radii in _FLUID_RADII.items():
        result[fluid] = {}
        for R_mm in radii:
            result[fluid][R_mm] = load_sample_data(fluid, R_mm)
    return result
