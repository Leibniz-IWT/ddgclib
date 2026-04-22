"""Quantitative metrics for the oscillating-droplet regression harness.

Two scores gate every phase of the fix plan:

- ``equilibrium_score``: a static droplet with epsilon=0 should stay
  at rest. Small values → good equilibrium.
- ``oscillation_score``: a perturbed droplet should decay along the
  Rayleigh–Lamb envelope. Small values → good agreement.

Both consume a list of per-frame diagnostic dicts (as produced by
:func:`_plot_helpers.compute_diagnostics` + ``{'t': t}``) and emit a
JSON-serializable summary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ._analytical import radius_perturbation


def _as_arr(diags: Iterable[dict], key: str) -> np.ndarray:
    return np.array([float(d[key]) for d in diags])


def equilibrium_score(
    diags: list[dict],
    M0: float,
    c_s: float,
    R0: float,
) -> dict:
    """Equilibrium regression score for a static (epsilon=0) droplet.

    The three normalized pieces should all be ≪ 1 in a healthy run.
    The summary scalar is the max of the three.
    """
    t = _as_arr(diags, 't')
    KE = _as_arr(diags, 'KE')
    total_mass = _as_arr(diags, 'total_mass')
    R_max = _as_arr(diags, 'R_max')
    R_min = _as_arr(diags, 'R_min')

    KE_scale = 0.5 * M0 * c_s * c_s
    max_KE_norm = float(np.max(KE) / KE_scale) if KE_scale > 0 else float('inf')
    mass_drift = float(np.max(np.abs(total_mass - total_mass[0])) / total_mass[0])
    R_drift = float(np.max(np.abs(np.concatenate([R_max, R_min]) - R0)) / R0)

    summary = max(max_KE_norm, mass_drift, R_drift)

    return {
        'kind': 'equilibrium',
        'n_frames': int(len(t)),
        't_start': float(t[0]) if len(t) else 0.0,
        't_end': float(t[-1]) if len(t) else 0.0,
        'max_KE_normalized': max_KE_norm,
        'mass_drift': mass_drift,
        'interface_radius_drift': R_drift,
        'summary': summary,
        'inputs': {'M0': float(M0), 'c_s': float(c_s), 'R0': float(R0)},
    }


def oscillation_score(
    diags: list[dict],
    R0: float,
    epsilon: float,
    l: int,
    omega: float,
    beta: float,
) -> dict:
    """Oscillation regression score.

    L2 error of r_apex(t) against Rayleigh–Lamb evaluated at the actual
    tracked apex angle theta_apex(t) (per-frame, not fixed to 0). Also
    returns mass drift and a flag for KE monotonic decay in the tail.
    """
    t = _as_arr(diags, 't')
    r_apex = _as_arr(diags, 'r_apex')
    theta_apex = _as_arr(diags, 'theta_apex')
    KE = _as_arr(diags, 'KE')
    total_mass = _as_arr(diags, 'total_mass')

    r_analytical = np.array([
        float(radius_perturbation(ti, thi, R0, epsilon, l, omega, beta))
        for ti, thi in zip(t, theta_apex)
    ])

    l2_err = float(np.sqrt(np.mean((r_apex - r_analytical) ** 2)) / (epsilon * R0))
    linf_err = float(np.max(np.abs(r_apex - r_analytical)) / (epsilon * R0))

    # KE should be non-growing after the first relaxation; flag the
    # ratio of tail-max to overall-max. 1.0 means KE kept growing;
    # near 0 means it decayed properly.
    if len(KE) >= 4:
        tail = KE[len(KE) // 2:]
        head_max = float(np.max(KE[: len(KE) // 2]))
        tail_max = float(np.max(tail))
        tail_growth = tail_max / head_max if head_max > 0 else float('inf')
    else:
        tail_growth = float('nan')

    mass_drift = float(np.max(np.abs(total_mass - total_mass[0])) / total_mass[0])

    summary = max(l2_err, mass_drift, max(0.0, tail_growth - 1.0))

    return {
        'kind': 'oscillation',
        'n_frames': int(len(t)),
        't_start': float(t[0]) if len(t) else 0.0,
        't_end': float(t[-1]) if len(t) else 0.0,
        'l2_error_normalized': l2_err,
        'linf_error_normalized': linf_err,
        'tail_growth': tail_growth,
        'mass_drift': mass_drift,
        'summary': summary,
        'inputs': {
            'R0': float(R0), 'epsilon': float(epsilon), 'l': int(l),
            'omega': float(omega), 'beta': float(beta),
        },
    }


def save_score(path: str | Path, score: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(score, f, indent=2, sort_keys=True)


def load_score(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def diff_baselines(baseline_path: str | Path, current_path: str | Path) -> dict:
    """Print and return a comparison between a baseline and a current run.

    Positive delta on ``summary`` means the current run is WORSE.
    """
    base = load_score(baseline_path)
    curr = load_score(current_path)
    assert base['kind'] == curr['kind'], (
        f"kind mismatch: {base['kind']} vs {curr['kind']}"
    )

    keys = sorted(k for k in base if isinstance(base[k], (int, float)))
    rows = []
    for k in keys:
        b = float(base[k])
        c = float(curr[k])
        delta = c - b
        rel = (delta / b) if b != 0 else float('inf')
        rows.append((k, b, c, delta, rel))

    print(f"\n=== {base['kind']} score: {baseline_path} -> {current_path} ===")
    print(f"{'metric':<28} {'baseline':>14} {'current':>14} "
          f"{'delta':>14} {'rel':>10}")
    for k, b, c, d, r in rows:
        marker = '  '
        if k == 'summary':
            marker = '!!' if d > 0 else '  '
        print(f"{marker}{k:<26} {b:>14.4e} {c:>14.4e} {d:>+14.4e} {r:>+10.2%}")

    return {
        'kind': base['kind'],
        'baseline': base,
        'current': curr,
        'improved': float(curr['summary']) <= float(base['summary']),
    }
