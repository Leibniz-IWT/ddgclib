# Lane 1 — zero-gauge-pressure sentinel + multiphase momentum fix

> 2026-07-02, sequential debugging workflow (first lane). Verdict: **LANDED** — all
> oscillation/equilibrium metrics improved, floors unchanged, no new suite failures.

## Audits addressed

- `docs_temp/audit/zero-gauge-pressure.md` (CONFIRMED high)
- `docs_temp/audit/multiphase-momentum.md` (CONFIRMED high, fallback; latent skip branch)

Root cause: multiphase code keyed "phase present at vertex" on the stored pressure
float being exactly `0.0`. At the case default gauge `P0 = 0`, quiescent vertices
legitimately store `p_phase[k] = 0.0` bitwise; the sentinel misread deleted the
pressure-difference flux one-sidedly (the two ends of a dual face booked different
face pressures), breaking gauge invariance and Newton's third law.

## Changes (file:line, post-edit numbering)

1. **`ddgclib/operators/multiphase_stress.py:83-113`** — replaced the
   `val == 0.0 -> fallback` body of `_phase_pressure` with a geometric presence
   test; added helper `_phase_present_at(v, k)`: phase k is present iff
   `dual_vol_phase[k] > 1e-30` (falls back to "stores a p_phase entry" when no
   per-phase dual volume is cached). Exactly the audit-verified monkeypatch.
   Signature of `_phase_pressure` unchanged (imported by
   `cases_dynamic/oscillating_droplet/diagnose_static.py` and
   `diagnose_a5_dissect_2d.py`).
2. **`ddgclib/operators/multiphase_stress.py:193-224`** — rewrote the per-phase
   sub-face loop of `multiphase_stress_force` so absent-phase handling is
   symmetric on both face ends:
   - presence at BOTH ends keyed on `_phase_present_at` (geometry), not tags /
     pressure values;
   - phase absent at both ends -> skip from both sides (old one-sided skip branch
     removed; its comment had misidentified the trigger per the audit);
   - phase present only at the neighbour -> mirror the neighbour's one-sided
     extrapolation (face pressure = p_j at both ends), the exact counterpart of
     the existing `fallback=p_i_k` branch, so `F_ij = -F_ji` holds under
     stale/inconsistent interface tags too;
   - `has_p_phase == False` path (vertices without `p_phase`) keeps the previous
     tag-based behaviour.
3. **`ddgclib/multiphase.py:555-568`** — interface `v.p` average filter changed
   from `v.p_phase[k] != 0.0` to `v.dual_vol_phase[k] > 1e-30 and
   v.m_phase[k] > 1e-30` (mirrors the write-side gate at :543 and the
   `mass_redistribution.py:304-313` pattern). Pre-fix this reported the full
   droplet pressure (5 Pa) instead of the two-phase mean (2.5 Pa) at P0=0,
   gauge-dependently. `v.p` is diagnostics/visualization only in the multiphase
   pipeline (verified: dynamics reads `p_phase`; scores read KE/mass/radius).

Sweep for other pressure-keyed presence tests: only these two existed
(`grep '== 0.0|!= 0.0'` over `multiphase.py` / `multiphase_stress.py`;
`mass_redistribution.py:312` `p_k_before < 1e-30` is a legacy fallback that only
fires when no dual-volume snapshot exists — left alone, also DO-NOT-listed).

## Regression tests added

`ddgclib/tests/test_multiphase_gauge_invariance.py` (8 tests, all passing, ~0.5 s
after fixture build; fixture: small 2D droplet, `refinement_outer=1,
refinement_droplet=2`, P0=0, +1 % outer-phase mass inside r < 1.3 R0 — the
audit-probe3 compression-front state, with the front confined so all free-frozen
faces stay quiescent and the free-vertex momentum sum measures pure pairwise
antisymmetry):

- `TestGaugeInvariance::test_gauge_offset_invariance` — max_v |F(p+1000) − F(p)|
  < 1e-10 N (measured 8.9e-15; pre-fix 1.17 N).
- `TestGaugeInvariance::test_state_exercises_the_bug` — guards >= 1 present entry
  stored exactly 0.0 next to O(100 Pa) pressures, max|F| > 0.1 N.
- `TestNetMomentum::test_net_momentum_free_vertices` — |ΣF|/max|F| < 1e-10
  (measured 1.5e-13; pre-fix sentinel semantics: 0.70).
- `TestNetMomentum::test_net_momentum_gauge_shifted` — same in the +1000 Pa gauge.
- `TestAbsentPhaseFallback::test_unit_semantics` — present phase with stored 0.0
  returns 0.0 (the fix); geometrically absent phase still returns the fallback;
  no-p_phase / out-of-range -> fallback; no-geometry -> trust stored value.
- `TestAbsentPhaseFallback::test_bulk_vertex_far_from_interface` — genuinely
  absent phase behaviour preserved on the real fixture.
- `TestInterfacePressureAverage::test_average_includes_zero_gauge_phase` —
  deterministic bitwise-0.0 present phase (power-of-two dual volume so
  rho == rho0 exactly) must enter the interface v.p mean (pre-fix: dropped).
- `TestInterfacePressureAverage::test_fixture_interface_average_consistent` —
  v.p equals an independent recomputation of the geometric-presence mean.

## Probe evidence (scratchpad `wf3/lane1_{probe,pairwise,closure,momentum}.py`)

Small fixture, compression front r < 2.2 R0 (probe3 replica), real
`multiphase_stress_force`:

| quantity | pre-fix | post-fix |
|---|---|---|
| `_phase_pressure` value-replaced firings | 7 / 378 calls | 0 |
| gauge: max |F(p+1000) − F(p)| | 1.172895e+00 N | 8.881784e-15 N |
| max pairwise defect |f_ij + f_ji| | 1.1729e+00 N | 0.0 (bitwise) |
| confined front (r < 1.3 R0): |ΣF|/max|F| free | 7.001115e-01 | 1.507403e-13 |

Note: on this coarse fixture (refinement_outer=1) the 2.2 R0 front touches
wall-adjacent vertices, so |ΣF| over free vertices contains a *physical* wall
reaction term (3.5003 N) in both gauges — that is why the momentum test uses the
confined 1.3 R0 front. All free dual cells verified closed (max |Σ_j A_ij| =
1.8e-18), so the gauge test needs no closure caveat.

## Measurement battery (vs baselines)

Commands run from repo root with `/home/endres/anaconda3/envs/ddg/bin/python`.

| metric | baseline | after lane 1 | delta |
|---|---|---|---|
| pinned floor tests (`test_case_oscillating_droplet.py -m ""`) | 12 pass | **12 passed in 13.71s** | unchanged (floors did NOT move, as predicted: at static equilibrium the fallback coincides with truth) |
| fast suite (`-m "not slow"`) | 796 passed, 1 pre-existing failure | **804 passed, 1 failed, 12 skipped, 2 xfailed** (same pre-existing `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`) | +8 new tests, no new failures |
| equil summary (static_droplet_2D) | 1.1636e-03 | **1.1351e-03** (max_KE_norm 1.0735e-08, mass_drift 0.0, R_drift 1.1351e-03) | −2.4 % (improved) |
| osc l2_error_normalized | 2.9519 | **0.7090740204580823** | −76 % (improved) |
| osc linf_error_normalized | 5.9145 | **1.05384362884825** | −82 % (improved) |
| osc tail_growth | 2.3530 | **2.2052688800146534** | −6.3 % (improved) |
| osc summary | 2.9519 | **1.2052688800146534** | −59 % (improved) |
| osc mass_drift | — | 7.847970997579543e-15 | machine precision |

Score files copied to scratchpad `wf3/lane1_osc_score.json`,
`wf3/lane1_equil_score.json`.

## Verdict / notes for later lanes

- **LANDED.** No re-pin needed (floors bitwise-stable at the pinned constants).
- The oscillation L2/Linf improved ~4-6x: the spurious wave-launch momentum
  injection during the initial transient was a first-order contaminant of the
  Rayleigh–Lamb fit, exactly as the audit predicted. `tail_growth` (2.3530 →
  2.2053) improved only mildly — the KE tail growth is therefore mostly a
  DIFFERENT mechanism (retopo/EOS noise? the recurring ~0.9 N |ΣF| spikes at
  steps 15/18 with n_sub=0 noted in the multiphase-momentum skeptic review are a
  candidate) — that is the remaining target for the next lanes.
- The interface `v.p` average changed for P0=0 states (was up to +100 % of the
  Laplace jump too high on some interface vertices). Any diagnostic/plot that
  consumed interface `v.p` at gauge zero will show different (correct) values.
- The one-sided skip branch is gone; under corrupted/stale `interface_phases`
  tags the operator now books a symmetric neighbour-extrapolated flux instead of
  silently dropping one side. Tag consistency itself (flag 19, swallowed closure
  validation at `multiphase.py:379-382`) is still an open item.
- Changed files: `ddgclib/operators/multiphase_stress.py`,
  `ddgclib/multiphase.py`, `ddgclib/tests/test_multiphase_gauge_invariance.py`
  (new). No hyperct edits. No git operations performed.
