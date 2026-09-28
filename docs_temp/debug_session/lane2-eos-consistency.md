# Lane 2 — EOS consistency (lane key: lane2-eos-consistency)

> 2026-07-02. Audit basis: `docs_temp/audit/eos-formulas.md`,
> `docs_temp/audit/multiphase-eos-interface.md`.
> Verdict: **LANDED** (no new failures, all metrics bit-identical to the
> post-lane-1 state; 14 new regression tests).

## What was changed (file:line)

1. **`ddgclib/eos/_tait_murnaghan.py`** — `rho_clip` is now applied
   consistently in all three of `pressure()` / `density()` /
   `sound_speed()` via a shared `_clip_rho()` helper (was: `pressure()`
   only), making the clipped EOS a coherent *saturating* model:
   - `density()` (now ~:117-137) clamps its result to the same band, so
     `density(pressure(rho))` is idempotent (band edge for out-of-band
     states) and `pressure(density(P))` saturates at the representable
     window edge. Previously `density(P)` below the Tait domain limit
     silently returned the cavitation-clamp value (`rho ~ 6e-2` for the
     soft droplet EOS) — 50–100 kg/m³ round-trip errors at band edges.
   - `sound_speed()` (now ~:141-162) evaluates on the clipped density:
     out-of-band states report the band-edge stiffness instead of e.g.
     3285 m/s where the clipped law is exactly flat.
   - **Saturation is never silent**: per-method engagement counters in
     `self.clip_count = {'pressure','density','sound_speed'}`
     (`__init__` ~:69-75) + one `RuntimeWarning` per instance on first
     engagement (`_clip_rho`, ~:79-99). Chose the audit §4.1 "coherent
     saturating model + visible engagement" combination (both options at
     once) because the skeptic review shows the clip is load-bearing at
     frozen-wall vertices (see A/B below) — removing it is not safe.
2. **`ddgclib/eos/_multiphase_eos.py`**
   - New module-level shared helper `interface_mean_pressure(v, n_phases)`
     (~:33-61): mean of `v.p_phase[k]` over phases *geometrically*
     present (`dual_vol_phase[k] > 1e-30 and m_phase[k] > 1e-30`,
     filtered on `v.interface_phases`) — THE interface `v.p` convention,
     shared with `compute_phase_pressures` so the two can never diverge.
   - `MultiphaseEOS.__call__` tail (~:133-147): guarded with
     `v.phase >= 0`. Interface vertices (sentinel -1) now get the mean
     convention instead of the silent numpy `p_phase[-1]` last-phase
     wrap; `v.rho` is set to the mixture density for interface vertices.
   - Fallback path B (no per-phase arrays, ~:112-123): raises
     `ValueError` for `v.phase < 0` (mixture density through
     `eos_list[-1]` produced ~740x errors in the audit probe; the path
     had 0 production invocations in the audit's 60-step probe, so the
     raise is safe and enforces the documented sentinel contract,
     `multiphase.py:59-66`).
   - Module/`__call__` docstrings updated to state the interface
     convention.
3. **`ddgclib/multiphase.py`**
   - `compute_phase_pressures` (~:528-565): stale "**inner-phase**
     pressure" docstring fixed (it averages); the inline interface-mean
     block (lane 1's volume/mass-keyed version) replaced by a call to
     the shared `interface_mean_pressure` — behaviourally identical
     (verified: all metrics bit-identical), now a single source of truth.
   - New import at :56-57 (`from ddgclib.eos._multiphase_eos import
     interface_mean_pressure`; no cycle — `_multiphase_eos` only imports
     `ddgclib.eos._base` + numpy).
4. **`ddgclib/eos/_ideal_gas.py`** — `density()` floored at 0
   (`np.maximum(P, 0.0) / (R T)`, ~:51-58); `P0`/`rho0` defaults
   documented (ISA 15 degC density; default P0 = 103085 Pa is 1.7%
   above 1 atm).
5. **`ddgclib/eos/_base.py`** — "isentropic speed of sound" docstrings
   corrected to "speed of sound of the implemented P(rho) law"
   (IdealGas is isothermal).
6. **NEW `ddgclib/tests/test_eos_consistency.py`** — 14 regression
   tests: round-trip bijection inside the band (soft droplet EOS,
   rtol 1e-12), idempotent two-sided saturation, sound_speed/clip
   consistency, engagement counter + warn-once visibility, in-band
   no-op, `rho_clip=None` unchanged, IdealGas floor, MultiphaseEOS
   interface mean (incl. exactly-0.0 gauge pressure inclusion,
   one-sided presence, path-B ValueError, bulk dispatch unchanged), and
   a mesh-level test asserting `meos(v) == compute_phase_pressures`'
   `v.p` on every vertex of a real two-phase interface mesh.

Not changed: `mass_redistribution.py` guards (DO-NOT list), the
`(0.8, 1.2)` clip band in the droplet setups (measured — see below), no
hyperct edits.

## Probe / measurement evidence

### Clip engagement during the shipped 2D oscillating run
(`scratchpad/wf3/probe_clip_engagement.py`, exact mirror of the shipped
script: K_d=800, K_o=1000, P0=0, dt=6.2167e-05, n_steps=1839;
droplet-EOS representable window **[-89.1959, +300.1429] Pa**, outer
**[-111.4948, +375.1786] Pa** — confirming the task's window numbers):

- `eos_outer.clip_count` run-only: **pressure 25661, density 0,
  sound_speed 0** (~14 element-engagements/step).
- `eos_drop.clip_count` run-only: **pressure 5, density 0, sound_speed 0**.
- Setup engages the clip 0 times.
- Classification of out-of-band (vertex,phase) pairs (103 samples,
  every 18 steps): min 0 / mean 6.68 / max 15 per sample; summed
  **wall 680, interface 0, bulk 8** — saturation is a frozen-wall-mass
  phenomenon; **interface vertices never flat-line** under shipped
  defaults (`redistribute_mass=True` heals them each retopo), exactly
  as the eos-formulas skeptic review predicted.
- `density()` clip: **0 engagements** → the production inversion
  (`redistribute_mass_multiphase`) only ever sees in-window snapshot
  pressures, so the density-side clip fix is a latent-trap closure, not
  a production behaviour change (consistent with bit-identical metrics).

### Clip band A/B (is (0.8, 1.2) appropriate for the soft K?)
(`scratchpad/wf3/probe_wide_band_ab.py`: identical run, band widened to
(0.5, 2.0) post-setup on both EOS; outer window becomes
[-138.88, +19723.76] Pa):

| metric | band (0.8,1.2) (shipped) | band (0.5,2.0) |
|---|---|---|
| summary | 1.2052688800146534 | 3.0917885097 |
| l2_error_normalized | 0.7090740204580823 | 3.0917885097 |
| linf_error_normalized | 1.05384362884825 | 5.9207138300 |
| tail_growth | 2.2052688800146534 | 1.9761960448 |

**Widening the band makes the case 4.4x WORSE on l2 / 5.6x on linf**
(tail_growth marginally better, -10%): the wider window lets the
frozen-wall bookkeeping artifact inject up to ~20 kPa (vs capped
~375 Pa) spurious near-wall pressure. The (0.8, 1.2) band is
load-bearing for the shipped case — **kept unchanged**. The real fix
target is the frozen-wall-mass artifact itself (out of this lane's
scope; now permanently visible via `clip_count` and the warning).

## Measurement battery (vs baselines / post-lane-1 state)

1. **Floor tests** `pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`:
   **12 passed** (both pinned floors intact: 2.3748568e-3 / 2.2716938e-3
   in 2D, 6.0153e-5 / 7.3768e-5 in 3D). The new RuntimeWarning fires
   once in the 3D floor test — the clip was already engaging there,
   previously silently.
2. **Fast suite** `pytest ddgclib/tests/ -m "not slow" -q`:
   **1 failed, 818 passed, 12 skipped, 17 deselected, 2 xfailed** in
   52.98s. The single failure is the pre-existing
   `test_simplex_aware_duals.py::TestBoundaryFromSimplices::test_raises_unsupported_dim`.
   818 = 804 (post-lane-1) + 14 new tests. **No new failures.**
3. **Equilibrium score** (`static_droplet_2D.py`): summary =
   **1.1351e-03** (post-lane-1: 1.1351e-03; original baseline
   1.1636e-03). Unchanged.
4. **Oscillation score** (`oscillating_droplet_2D.py`):
   summary **1.2052688800146534**, l2 **0.7090740204580823**,
   linf **1.05384362884825**, tail_growth **2.2052688800146534**,
   mass_drift 7.847970997579543e-15 — **bit-identical to lane 1's
   landed values** (post-lane-1: l2 0.7090740204580823,
   tail 2.2052688800146534; original baseline l2 2.9519, tail 2.353).
   Score files copied to `scratchpad/wf3/lane2-eos-consistency_{osc,equil}_score.json`.

## Answer to the lane's MEASURE question

Clip saturation was **not** rate-limiting the Laplace-jump dynamics:
the saturated states are exclusively frozen wall vertices (0 interface
engagements in 1839 steps), production `density()` inversions never see
saturated-window-exceeding pressures, and both making the clip
consistent and (experimentally) widening it left tail_growth
essentially unchanged (widening trades -10% tail for +340% l2). The
KE-tail mechanism lane 1 flagged lives elsewhere (retopo noise /
near-wall flux of the capped-but-still-wrong wall pressures — see the
recurring |sum F| spikes noted in the multiphase-momentum review).

## Caveats for future lanes

- The clip warning (`RuntimeWarning`, warn-once-per-instance) now
  appears in any run whose wall vertices drift out of band — including
  the 3D floor test and both droplet cases. This is intentional. Do not
  silence it globally; read `eos.clip_count` instead.
- `clip_count['density']` can pick up float-epsilon re-saturations when
  round-tripping exactly-band-edge pressures; in the shipped runs it
  stayed 0.
- `MultiphaseEOS.__call__` path B (no per-phase arrays) now RAISES for
  interface vertices. Any future wiring that feeds fresh vertices
  (e.g. adaptive-remesh `edge_split_2d` inserts) to a force pass before
  an `mps.refresh` will now fail loudly instead of silently applying
  the last phase's EOS to a mixture density. That is the desired
  tripwire, but expect it to surface when A.4 (adaptive remesh) work
  starts.
- The interface `v.p` convention is now defined in ONE place:
  `ddgclib.eos._multiphase_eos.interface_mean_pressure`. Change it
  there only.
