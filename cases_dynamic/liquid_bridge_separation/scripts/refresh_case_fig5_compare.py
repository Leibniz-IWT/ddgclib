from __future__ import annotations

import argparse
import json
from pathlib import Path

import plot_case_chart_vs_t as cases
import v5Conti_vd_Dimentionless_VolFlux0_speed_FHeron_cu0_ddgclib as base_solver
import _ddgclib_case_core as core_solver


CORE_CASES = {"case9", "case10", "case11", "case12"}


def _load_history(case_dir: Path) -> list[dict]:
    history_path = case_dir / "separation_history.json"
    payload = json.loads(history_path.read_text())
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"{history_path} does not contain a non-empty history list")
    return history


def _selected_cases(tokens: list[str] | None) -> list[tuple[str, Path]]:
    selected = list(cases.CASE_DIRS.items())
    if tokens:
        wanted = set(tokens)
        selected = [
            (case_key, case_dir_name)
            for case_key, case_dir_name in selected
            if case_key in wanted or case_dir_name in wanted
        ]
        if not selected:
            raise RuntimeError(f"No cases matched: {', '.join(tokens)}")
    return [(case_key, cases.ROOT / case_dir_name) for case_key, case_dir_name in selected]


def refresh_case(case_key: str, case_dir: Path) -> None:
    history = _load_history(case_dir)
    solver = core_solver if case_key in CORE_CASES else base_solver
    solver._write_pitois_fig5_comparison(
        separation_history=history,
        out_dir=case_dir,
    )
    print(case_dir / solver.FIG5_COMPARE_PNG_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh saved Pitois Fig. 5 comparison artifacts from existing histories."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Case keys or output folder names, e.g. case3 case12.",
    )
    args = parser.parse_args()
    for case_key, case_dir in _selected_cases(args.only):
        refresh_case(case_key, case_dir)


if __name__ == "__main__":
    main()
