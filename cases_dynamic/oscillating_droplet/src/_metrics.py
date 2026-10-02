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

from ._analytical import (
    max_radius_envelope, radius_perturbation, radius_perturbation_two_fluid,
)


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


def add_two_fluid_reference(
    score: dict,
    diags: list[dict],
    R0: float,
    epsilon: float,
    l: int,
    omega_two_fluid: float,
    beta_two_fluid: float,
    beta_energy: float | None = None,
) -> dict:
    """Attach the two-fluid-reference error to an oscillation score.

    Computes the same apex-tracked L2/Linf construction as
    :func:`oscillation_score` but against
    :func:`~._analytical.radius_perturbation_two_fluid` with the
    two-fluid dispersion ``(omega, beta)``
    (:func:`~._analytical.two_fluid_omega_beta_2d`).  The pinned
    single-fluid fields (``l2_error_normalized``, ``summary``, ...)
    are left untouched — regression continuity — and all new keys are
    suffixed ``_two_fluid``.  ``beta_energy`` optionally records the
    closed-form Lamb-method estimate
    (:func:`~._analytical.lamb_damping_rate_two_fluid`) for
    side-by-side reporting.  Mutates and returns *score*.
    """
    t = _as_arr(diags, 't')
    r_apex = _as_arr(diags, 'r_apex')
    theta_apex = _as_arr(diags, 'theta_apex')

    r_ref = np.array([
        float(radius_perturbation_two_fluid(
            ti, thi, R0, epsilon, l, omega_two_fluid, beta_two_fluid))
        for ti, thi in zip(t, theta_apex)
    ])
    score['l2_error_normalized_two_fluid'] = float(
        np.sqrt(np.mean((r_apex - r_ref) ** 2)) / (epsilon * R0))
    score['linf_error_normalized_two_fluid'] = float(
        np.max(np.abs(r_apex - r_ref)) / (epsilon * R0))
    score['inputs_two_fluid'] = {
        'omega': float(omega_two_fluid),
        'beta': float(beta_two_fluid),
        'beta_energy': None if beta_energy is None else float(beta_energy),
    }
    return score


def oscillation_score_3d(
    diags: list[dict],
    R0: float,
    epsilon: float,
    l: int,
    omega: float,
    beta: float,
    r_boundary: float | None = None,
    saturation_frac: float = 0.9,
) -> dict:
    """3D oscillation regression score (Tier 3B harness, 06 §1.6).

    Primary metric: L2 error of R_max(t) against the Rayleigh–Lamb
    ``max_radius_envelope`` (theta=0), normalized by ``epsilon * R0``
    exactly like the 2D ``oscillation_score``.  ``omega``/``beta`` are
    the 3D values from ``_analytical`` (Miller–Scriven outer-density
    correction when the runner passes ``rho_outer``; Lamb ``beta_3d``).

    When per-frame apex tracking is recorded (``r_apex`` +
    ``theta_apex``, with theta measured from the perturbation axis —
    the 3D runner uses ``compute_diagnostics(..., polar_axis='z')``),
    a secondary apex score against ``radius_perturbation`` at the
    tracked angle is included (``apex_l2_error_normalized`` /
    ``apex_linf_error_normalized``); otherwise those keys are None.

    Boundary-saturation guard: the documented 3D artefact
    (DEVELOPMENT.md Tier 3B note) is R_max inflating until interface
    vertices pile up at the outer mesh boundary (~5·R0).  When
    *r_boundary* is given (the runner passes ``L_domain``), any frame
    with ``R_max >= saturation_frac * r_boundary`` sets
    ``boundary_saturation: True`` — the numeric scores are still
    reported but must NOT be read as a physics comparison.

    Optional per-frame fields consumed when present on every frame
    (A/B bookkeeping checks for the dual-only retopo policy):

    - ``total_dual_vol``: reports ``dual_vol_step0_jump`` (the known
      frame-0→1 boundary-cell-zeroing + mps.refresh transition) and
      ``dual_vol_drift_post`` (max relative drift from frame 1 on).
    - ``n_interface``: reports start/end/min/max interface counts.
    """
    t = _as_arr(diags, 't')
    R_max = _as_arr(diags, 'R_max')
    KE = _as_arr(diags, 'KE')
    total_mass = _as_arr(diags, 'total_mass')

    R_env = np.asarray(
        max_radius_envelope(t, R0, epsilon, omega, beta, l=l), dtype=float,
    )
    l2_err = float(np.sqrt(np.mean((R_max - R_env) ** 2)) / (epsilon * R0))
    linf_err = float(np.max(np.abs(R_max - R_env)) / (epsilon * R0))

    # Secondary apex score (same construction as the 2D score).
    apex_l2 = apex_linf = None
    if all(('r_apex' in d and 'theta_apex' in d) for d in diags):
        r_apex = _as_arr(diags, 'r_apex')
        theta_apex = _as_arr(diags, 'theta_apex')
        r_analytical = np.array([
            float(radius_perturbation(ti, thi, R0, epsilon, l, omega, beta))
            for ti, thi in zip(t, theta_apex)
        ])
        apex_l2 = float(
            np.sqrt(np.mean((r_apex - r_analytical) ** 2)) / (epsilon * R0)
        )
        apex_linf = float(
            np.max(np.abs(r_apex - r_analytical)) / (epsilon * R0)
        )

    # KE tail flag — identical to the 2D score.
    if len(KE) >= 4:
        tail = KE[len(KE) // 2:]
        head_max = float(np.max(KE[: len(KE) // 2]))
        tail_max = float(np.max(tail))
        tail_growth = tail_max / head_max if head_max > 0 else float('inf')
    else:
        tail_growth = float('nan')

    mass_drift = float(np.max(np.abs(total_mass - total_mass[0])) / total_mass[0])

    # Boundary-saturation artefact detection.
    R_max_peak = float(np.max(R_max))
    boundary_saturation = False
    n_saturated_frames = 0
    if r_boundary is not None:
        thresh = saturation_frac * float(r_boundary)
        n_saturated_frames = int(np.sum(R_max >= thresh))
        boundary_saturation = bool(n_saturated_frames > 0)

    summary = max(l2_err, mass_drift, max(0.0, tail_growth - 1.0))

    score = {
        'kind': 'oscillation_3d',
        'n_frames': int(len(t)),
        't_start': float(t[0]) if len(t) else 0.0,
        't_end': float(t[-1]) if len(t) else 0.0,
        'l2_error_normalized': l2_err,
        'linf_error_normalized': linf_err,
        'apex_l2_error_normalized': apex_l2,
        'apex_linf_error_normalized': apex_linf,
        'tail_growth': tail_growth,
        'mass_drift': mass_drift,
        'R_max_peak': R_max_peak,
        'boundary_saturation': boundary_saturation,
        'n_saturated_frames': n_saturated_frames,
        'summary': summary,
        'inputs': {
            'R0': float(R0), 'epsilon': float(epsilon), 'l': int(l),
            'omega': float(omega), 'beta': float(beta),
            'r_boundary': None if r_boundary is None else float(r_boundary),
            'saturation_frac': float(saturation_frac),
        },
    }

    # Optional A/B bookkeeping diagnostics.
    if all('total_dual_vol' in d for d in diags):
        V = _as_arr(diags, 'total_dual_vol')
        if len(V) >= 2 and V[0] > 0:
            score['dual_vol_step0_jump'] = float(abs(V[1] - V[0]) / V[0])
        if len(V) >= 2 and V[1] > 0:
            score['dual_vol_drift_post'] = float(
                np.max(np.abs(V[1:] - V[1])) / V[1]
            )
    if all('n_interface' in d for d in diags):
        n_if = _as_arr(diags, 'n_interface').astype(int)
        score['n_interface_start'] = int(n_if[0])
        score['n_interface_end'] = int(n_if[-1])
        score['n_interface_min'] = int(np.min(n_if))
        score['n_interface_max'] = int(np.max(n_if))

    return score


def save_score(path: str | Path, score: dict, methods=None) -> None:
    """Write *score* as JSON.

    *methods* (a ``ddgclib.methods.SolverMethods``) is embedded under the
    ``'methods'`` key so the score is self-describing: a baseline pins a
    number AND the configuration that produced it, and
    :func:`diff_baselines` flags a configuration drift before comparing
    numbers.  Reproducibility rule of the campaign (debugging_plan.md,
    2026-09-25): every scored run passes its config here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if methods is not None:
        score = dict(score)
        score['methods'] = methods.to_dict()
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

    # Configuration drift check (reproducibility rule, 2026-09-25): the
    # numbers are only comparable if the solver methods match.
    mb, mc = base.get('methods'), curr.get('methods')
    _doc_fields = ('label', 'notes')   # documentation, not method choices
    if mb is not None and mc is not None:
        # A score written before an axis existed reads as that axis'
        # default (laneL: else every new axis reports a difference).
        from ddgclib.methods import AXES

        def _get(m, k):
            return m[k] if k in m else (AXES[k].default if k in AXES
                                        else None)

        changed = sorted(k for k in set(mb) | set(mc)
                         if k not in _doc_fields
                         and _get(mb, k) != _get(mc, k))
        if changed:
            print(f"\n!! SOLVER METHODS DIFFER from the baseline on: "
                  + ', '.join(f"{k}: {_get(mb, k)!r} -> {_get(mc, k)!r}"
                              for k in changed))
            print("!! (a different method, not a regression of the same one)")
    elif mb is None or mc is None:
        print("\n?? one of the scores carries no 'methods' block; comparison "
              "is not configuration-checked")

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
