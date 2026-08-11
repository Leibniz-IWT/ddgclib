from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
AXISYM_CHART_PATH = PACKAGE_ROOT / "reference_data" / "pitois_fig5_chart.py"
NEATPLOT_STYLE = PACKAGE_ROOT / "reference_data" / "neatplot-main" / "standard.mplstyle"
PITOIS_RADIUS_MM = 4.0
PITOIS_FIG5_GAP_RATE_MM_S = 5.0e-3
DEFAULT_MAX_STEP = 5000

CASE_DIRS = {
    f"case{i}": f"v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib_case{i}"
    for i in range(1, 13)
}


def _load_axisym_chart_module():
    spec = importlib.util.spec_from_file_location("axisym_chart", AXISYM_CHART_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {AXISYM_CHART_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_time_series() -> tuple[np.ndarray, np.ndarray]:
    chart = _load_axisym_chart_module()
    exp_d_over_r = np.asarray(chart.EXP_D_OVER_R, dtype=float)
    exp_force_mn = np.asarray(chart.EXP_FORCE_MN, dtype=float)
    exp_t_s = (
        PITOIS_RADIUS_MM
        * (exp_d_over_r - float(exp_d_over_r[0]))
        / PITOIS_FIG5_GAP_RATE_MM_S
    )
    return exp_t_s, exp_force_mn


def _load_case_history(case_dir: Path) -> list[dict]:
    history_path = case_dir / "separation_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")
    payload = json.loads(history_path.read_text())
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{history_path} does not contain a non-empty history list")
    return history


def _case_time_force(history: list[dict], max_step: int) -> tuple[np.ndarray, np.ndarray, float]:
    filtered = [row for row in history if int(row.get("step", 0)) <= int(max_step)]
    if not filtered:
        raise RuntimeError(f"No history samples found through step {max_step}")
    limit_candidates = [row for row in filtered if int(row.get("step", 0)) == int(max_step)]
    if limit_candidates:
        t_limit_s = float(limit_candidates[-1]["t"])
    else:
        t_limit_s = float(filtered[-1]["t"])

    time_s = np.asarray([float(row["t"]) for row in filtered], dtype=float)
    force_mn = np.asarray(
        [abs(float(row["fixed_force_axial"])) * 1.0e3 for row in filtered],
        dtype=float,
    )
    finite = np.isfinite(time_s) & np.isfinite(force_mn)
    if not np.any(finite):
        raise RuntimeError("No finite time/force samples in history")
    return time_s[finite], force_mn[finite], t_limit_s


def _case_label(case_key: str, case_dir: Path) -> str:
    return f"{case_key}: {case_dir.name}"


def write_chart_vs_t(case_key: str, case_dir: Path, max_step: int) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(str(NEATPLOT_STYLE))

    exp_t_s, exp_force_mn = _experiment_time_series()
    history = _load_case_history(case_dir)
    sim_t_s, sim_force_mn, t_limit_s = _case_time_force(history, max_step=max_step)
    exp_visible = exp_t_s <= t_limit_s

    fig, ax = plt.subplots()
    ax.scatter(
        exp_t_s[exp_visible],
        exp_force_mn[exp_visible],
        s=16,
        color="#111111",
        label="Pitois et al. (2000)",
        zorder=3,
    )
    ax.plot(
        sim_t_s,
        sim_force_mn,
        color="#d62828",
        linewidth=1.4,
        label="Raw ddgclib simulation",
        zorder=5,
    )

    ax.set_xlim(0.0, max(t_limit_s, 1.0e-12))
    visible_exp_force = exp_force_mn[exp_visible] if np.any(exp_visible) else exp_force_mn
    upper = max(
        1.7,
        float(np.nanmax(visible_exp_force)) * 1.05,
        float(np.nanmax(sim_force_mn)) * 1.05,
    )
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"$|F|$ [mN]")
    ax.set_title(f"Pitois 2000 Fig. 5: time comparison, {_case_label(case_key, case_dir)}")
    ax.grid(True, which="major", alpha=0.28)
    ax.legend()

    out_path = case_dir / "chart_vs_t.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _selected_cases(tokens: list[str] | None) -> list[tuple[str, Path]]:
    selected = list(CASE_DIRS.items())
    if tokens:
        wanted = set(tokens)
        selected = [
            (case_key, case_dir_name)
            for case_key, case_dir_name in selected
            if case_key in wanted or case_dir_name in wanted
        ]
        if not selected:
            raise RuntimeError(f"No cases matched: {', '.join(tokens)}")
    return [(case_key, ROOT / case_dir_name) for case_key, case_dir_name in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create chart_vs_t.png in numbered Pitois case output folders."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Case keys or output folder names, e.g. case3 case12.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
        help="Maximum iteration to show on the time axis.",
    )
    args = parser.parse_args()

    for case_key, case_dir in _selected_cases(args.only):
        out_path = write_chart_vs_t(case_key, case_dir, max_step=args.max_step)
        print(out_path)


if __name__ == "__main__":
    main()
