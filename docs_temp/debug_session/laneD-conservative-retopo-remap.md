# laneD-conservative-retopo-remap — conservative retopology remap for large-deformation multiphase flow

Date: 2026-07-29. Lane D of the wf4 session (parallel to laneA/B/C).
Structural fix lane that lane 5 explicitly left open (lane5 log §8):
dam break / detaching bubbles cannot freeze connectivity (`dual_only`),
so the per-Delaunay-rewire jolt mechanism needs a real fix, not the
benchmark-configuration workaround.

## 1. DESIGN (written before implementation)

### 1.1 Where the existing snapshot/redistribute machinery falls short

Read first: `ddgclib/operators/mass_redistribution.py`,
`_retopologize_multiphase` (`_integrators_dynamic.py:440-524`),
`MultiphaseSystem.refresh/compute_phase_pressures` (`multiphase.py`),
and the p_ref reference (`cases_dynamic/oscillating_droplet_p_ref/
scripts/pr33_operators.py`, esp. `active_retopology_tet_remap` and the
"key pressure-reference choice" comment block at :892-899).

The existing machinery ALREADY solves per-phase masses on the NEW duals
to reproduce the old per-vertex pressure field, with exact per-phase
mass conservation enforced by a **global multiplicative scale**
`scale_k = M_k / sum_i rho_k(p_old_i) * dvp_new_i[k]`
(`redistribute_mass_multiphase`).  After the final
`compute_phase_pressures`, every vertex reads

    p_new_i = eos_k(scale_k * rho_k(p_old_i))  ~=  p_old_i + K_k*(scale_k - 1)

so the per-vertex field is preserved EXACTLY up to a **uniform per-phase
pressure offset K_k*(scale_k - 1)**.  Crucially, because redistribution
overwrites the local density signal every step, this offset is the ONLY
channel through which pressure evolves at all under
`redistribute_mass=True`: the entire compressible physics flows through
`scale_k` (global phase compression -> scale_k > 1 -> uniform pressure
rise; this is why lane 5 called it a per-step quasi-projection).

`scale_k` is the ratio of conserved phase mass to the phase volume
measured on the *new* duals weighted by target densities.  Under
`dual_only` the measured phase volume changes only smoothly (vertex
motion at frozen connectivity) => `scale_k` carries a clean physical
strain signal.  Under per-step Delaunay the measured phase volume ALSO
jumps discontinuously with every reconnection (the p_ref-documented
"connectivity change read as compression"), so

    scale_k = (physical strain) x (connectivity measurement artifact)

and the artifact multiplies through K into a per-phase-uniform pressure
jolt every step.  A per-phase-uniform offset produces ~zero net force on
interior bulk vertices (closed dual cells, sum_j A_ij = 0) but a DIRECT
force jolt at the interface, where phase-d and phase-o offsets differ
(the per-phase force assembly books p_phase[k] on phase-k sub-faces).
With K = 800/1000 Pa and Laplace jump gamma/R0 = 5 Pa, a 1e-4 relative
volume-measurement jump is already a 2% Laplace-jump jolt, every step.
Probe P1 (below) measures this before implementation.

Secondary channels NOT addressed by any pressure remap (kept in view for
the failure analysis in case the win condition is missed):
- per-vertex mass reshuffling at fixed u (KE bookkeeping jitter,
  acceleration weight a = F/m_i jitter);
- interface edge-structure churn feeding the surface-tension curvature
  stencil;
- dual-area-vector (A_ij) tessellation change at fixed p (consistent
  discretization of a smooth field — expected small).

### 1.2 Chosen mechanism: (c) two-stage pressure-preserving remap

Candidates from the task: (a) p_ref-style simplex-volume closure —
replaces the EOS-on-dual-volumes pressure operator wholesale; too
invasive for the library pipeline, and its local-strain reset (targets
:= current volumes) wipes local pressure structure, which would break
hydrostatic cases (dam break needs pressure gradients).  (b) old->new
dual overlap remap — geometrically conservative but requires polygon/
polyhedron intersection machinery per vertex per step; heavy, and it
remaps MASS while the jolt enters through the *pressure read*, so it
still needs a closure for m/V -> p.  (c) generalise the existing
re-projection — chosen: minimal new code, uses only validated building
blocks, and follows directly from the §1.1 diagnosis.

Key invariant exploited: inside one `_retopologize_multiphase` call the
vertex positions are FROZEN.  Everything the Delaunay rebuild changes at
fixed positions is measurement/connectivity artifact, and the remap must
make the *pressure field* (what forces read) exactly invariant across
the rebuild, while total per-phase mass stays exactly conserved.  The
physical strain signal is exactly what `dual_only` would have produced
at the same positions.  So, with `retopo_remap='conservative'`:

Stage 1 — physical update on the OLD connectivity (== the validated
`dual_only` per-step sequence):
  1. `_retopologize(..., skip_triangulation=True)` — refresh duals at
     the new positions, old connectivity;
  2. `mps.refresh(reset_mass=False, split_method)` — dvp_mid;
  3. `redistribute_mass_multiphase` with the pre-call snapshot p_old;
  4. `compute_phase_pressures` -> p_phys (this scale factor is the
     PHYSICAL one; identical to what dual_only produces).
  5. Snapshot (p_phys, dvp_mid) via `snapshot_geometry_multiphase`.

Stage 2 — connectivity rebuild, forced pressure-neutral:
  6. `_retopologize(..., skip_triangulation=False)` — full Delaunay;
  7. `mps.refresh(reset_mass=False)` -> dvp_new;
  8. `redistribute_mass_multiphase` with snapshot (p_phys, dvp_mid):
     masses m_i = scale2_k * rho(p_phys_i) * dvp_new_i keep per-vertex
     inertia consistent with the new dual cells and conserve M_k
     exactly.  The leftover scale2_k = sum rho*dvp_mid / sum rho*dvp_new
     is now PURELY the connectivity measurement artifact (positions
     frozen since stage 1).
  9. `restore_pressure_multiphase` (NEW helper): overwrite
     `v.p_phase[k]` with the stage-1 snapshot wherever phase k persists
     across the rebuild (present in both dvp_mid and dvp_new), and
     recompute `v.p` under the shared interface-mean convention.  This
     cancels the K*(scale2_k - 1) jolt exactly instead of injecting it.
     Equivalent to reading the EOS with a per-phase volume-correction
     factor C_k = scale2_k, but implemented without touching
     `compute_phase_pressures` or the EOS classes; the transient
     p != eos(m/dvp) inconsistency is immaterial because the next
     step's redistribution regenerates m from p anyway (m is only the
     mass ledger + inertia between retopos; forces read p_phase).

Opt-in: `retopo_remap=None` default everywhere (bit-identical
behaviour); `'conservative'` requires `mps` + `redistribute_mass=True`
and is a no-op when `skip_triangulation=True`.  Cost: one extra
compute_vd + batch_e_star + refresh + redistribute per step (~2x retopo
cost) — acceptable for a structural prototype.

### 1.3 Validation plan

- P1 probe (before implementing): instrumented 200-step full_delaunay
  vs dual_only runs recording per-step redistribution scale factors,
  KE, and total measured phase volumes -> confirm the scale-factor
  noise hypothesis.
- A/B on the lane-5 instrumented config (driver rebuilt in scratchpad
  wf4/laneD — the wf3 copy was tmp-cleaned; validated by reproducing
  the full_delaunay baseline l2 0.48992 / tail 1.72505 bit-exactly).
  WIN: with remap ON under per-step full Delaunay, KE_max down >= 10x
  toward the 8.3e-7 J physical envelope and tail < 1.2.
- Dam break smoke ON vs OFF (KE trajectory, mass drift, alpha_art
  outlook).  No dam-break default changes this lane.
- Full measurement battery; defaults unchanged => baseline numbers
  bit-stable.  Regression test on the remap path if the win condition
  is met.

## 2. RESULTS — WIN CONDITION MET (this IS major progress)

Full 1839-step oscillating-droplet run, **per-step FULL Delaunay
reconnection active the whole run**, refine 3/3, c_s=1 (identical
parameters to the lane-5 battery; driver validated by reproducing the
lane-5 full_delaunay baseline bit-exactly first: l2 0.48991833470391266
/ tail 1.7250489596305962 / KE_max 0.04071648634239194):

| metric | full_delaunay (baseline to beat) | full_delaunay + remap (final) | dual_only (official) | win condition |
|---|---|---|---|---|
| l2_error_normalized | 0.48991833470391266 | **0.17479361640597058** | 0.1785660454150319 | — (beats dual_only) |
| tail_growth | 1.7250489596305962 | **0.9998967874595965** | 0.9992507831101141 | < 1.2 ✓ |
| linf | 0.8503543927096853 | 0.32364245955409165 | 0.32842491275938274 | — |
| KE_max [J] | 4.0716e-02 (still growing) | **8.31727774219008e-07** | 8.317e-07 | ≥10x drop ✓ (**48,954x**) |
| t @ KE_max | 0.1136 (t_end) | 0.0560 (analytical 0.0598) | 0.0549 | physical envelope ✓ |
| KE_final/KE_max | 0.968 | 0.7055 | 0.6955 | decaying ✓ |
| mass_drift | 1.43e-14 | 2.41e-14 | 2.68e-14 | machine ✓ |

The Delaunay-churn KE pump is eliminated at its thermodynamic root
while connectivity reconnects every step — the structural fix for
large-deformation cases that dual_only could not provide.

### 2.1 What shipped (three pieces, all opt-in via `retopo_remap='conservative'`)

1. **Structure restore** (`restore_pressure_multiphase`,
   `operators/mass_redistribution.py`): after the rebuild +
   redistribution, `p_phase[k]` is restored bit-exactly from the
   pre-call snapshot wherever phase presence persists (one-call
   neutrality on the coarse test fixture: max|dp| 3.8e-13 Pa vs
   1.7e2 Pa without the remap).
2. **Volume-gauge factor** (`MultiphaseSystem.vol_corr`,
   `multiphase.py`): per-phase EOS volume gauge
   `rho_k = m_k/(vol_corr[k]*dual_vol_phase[k])`, set to the stage-2
   redistribution scale factor.  Keeps the mass ledger and the pressure
   field self-consistent so the NEXT redistribution does not bounce the
   artifact back (v1 dead end below).  Exactly 1.0 unless the remap is
   active (default path bit-identical, verified).
3. **Level anchor** (`anchor_phase_pressure_levels`): the per-phase
   pressure LEVEL is pinned to the volume strain relative to
   connectivity-artifact-corrected volume targets — `_remap_vol_tar`
   multiplied by `V_new/V_mid` at every rebuild (both measured at the
   same frozen positions ⇒ pure measurement artifact), reference
   density `_remap_rho_ref = eos.density(L_k(0))` from the INITIAL
   pressure level.  This is the p_ref pattern (rebuild targets after
   every retopo; pressure from strain vs rebuilt targets — a state
   function, not an integral of noisy increments).

`_retopologize_multiphase` gained `retopo_remap=None` (default OFF
everywhere; `'conservative'` requires mps + redistribute_mass=True,
no-op under skip_triangulation).  Remap sequence per call: stage-1
measurement pass (dual-only refresh at frozen positions on the old
connectivity → V_mid), then the normal rebuild + refresh +
redistribution, then gauge update + restore + anchor.  Cost ~2x retopo
(one extra compute_vd + split per step).

### 2.2 Evidence chain / mechanism (probes in scratchpad wf4/laneD-*)

- **P1 (mechanism confirmation)**: 200-step instrumented runs. The
  redistribution scale factor (whose deviation from 1 maps through K
  into a uniform per-phase pressure jolt) has outer-phase noise
  max|s−1| = 1.43e-2 / rms 2.3e-3 under per-step Delaunay vs 1.44e-5 /
  1.0e-6 under dual_only (~1000x) — while the droplet-phase physical
  signal (3.76e-4) is identical in both.  K_o=1000 ⇒ up to 14 Pa jolts
  vs the 5 Pa Laplace jump.  The §1.1 diagnosis is measured fact.
- **v1 dead end (restore only)**: KE_max 5.77e-3 → 5.59e-3 only.  The
  restored p is inconsistent with the redistributed m, so the NEXT
  step's redistribution scale re-applies the artifact:
  corr(stage1[n+1]−1, stage2[n]−1) = **+0.9998**.  Restore alone just
  delays the jolt one step.  Do not retry.
- **v2 dead end (restore + gauge, no anchor)**: per-rebuild jolt gone,
  trajectory error collapses at 200 steps (l2 0.402 → 0.0116 ≈
  dual_only 0.0063) but KE stays pumped (3.9e-3).  Region-decomposed
  KE: **entirely far-field** (r > 2.5R0: 3.163e-3 of 3.164e-3 J;
  interface/droplet/near at physical levels).  Cause: the pressure
  LEVEL is an integral of per-step scale factors; reconnection noise
  rectifies (quantities drifting smoothly within a connectivity
  segment and resetting at rebuilds are read as physical strain) and
  self-amplifies through real motion → runaway outer-phase tension
  (p_far: −0.06 at step 25 → −9.56 Pa at step 275, ACCELERATING;
  boundary set stable 31, no bV churn — that was ruled out).  Full-run
  v2: l2 0.4228 / tail 1.368 / KE_max 2.71e-2.  Do not retry an
  incremental level under active reconnection.
- **v3 anchor**: p_far pinned within ±0.01 Pa over 300 steps, p_drop
  rises smoothly to the Laplace value ~5.2 Pa, KE on the physical
  envelope (200-step KE 3.456e-7 vs dual_only 3.438e-7).
- **v3 init fix**: anchor reference from the mass ledger (M_k/V_k)
  jolted the coarse-fixture droplet level by −5.39 Pa at the first
  call — setup's Young–Laplace mass loading reads pre-perturbation
  dual volumes, so the ledger is not volume-consistent at t=0.
  Reference density now comes from the initial pressure level
  (`eos.density(L_k(0))`); first-call shift = 0 by construction.

### 2.3 Dam break smoke (no case files changed)

**Discovery**: `dam_break_2D.py` passes `skip_triangulation=True` to
`symplectic_euler`, but `_do_retopologize` does NOT forward that flag
to a callable `retopologize_fn` (only remesh_mode/kwargs) — the
shipped dam break actually runs **per-step full Delaunay**, its
README/comment notwithstanding.  Currently harmless-by-accident: at
the shipped smoke horizon the collapse is stalled (|u|max 0.019–0.028
m/s; displacement << dx even at t_end×5 = 0.1 s), Delaunay never flips
(shipped == true-dual_only trajectories to all printed digits), so the
A/B cannot discriminate the remap at these parameters.  Measured, all
variants (shipped / remap / dual_only, plus alpha_art 0.5 and t_end
0.1 probes): no aborts, mass drift ≤ 5e-15, zero EOS clips, interface
count stable (9).  KE_liq plateau: shipped 8.9e-5 J, remap 2.0e-4 J
(2.3x — the level anchor holds the phase mean while the incremental
scheme slowly relaxes it; both tiny vs a real collapse, flag for the
lane that unsticks the case).  alpha_art 2.0 → 0.5: both variants
remain stable (KE ~15x higher, still bounded, no clips) — the crutch
addresses the corner-vertex force defect, which is orthogonal to
retopo churn; **no evidence at this horizon that the remap allows
reducing it, and none against** — re-test once the collapse actually
runs (larger t_end / weaker crutches / finer mesh), where reconnection
will fire and the remap should be load-bearing.

### 2.4 Measurement battery (final code state; ddg env, repo root)

1. Floor battery `pytest ddgclib/tests/test_case_oscillating_droplet.py -v -m ""`:
   **17 passed** (14 pinned + 3 new `TestConservativeRetopoRemap2D`),
   no re-pins — floors 2.3748568e-3 / 2.2716938e-3 / 6.0153e-5 /
   7.274172e-5 untouched (Delaunay floor path exercised post-edit).
2. Fast suite `pytest ddgclib/tests/ -m "not slow" -q`:
   **865 passed, 12 skipped, 17 deselected, 2 xfailed in 53.77s — 0
   failures** (= lane-C 862 + 3 new).
3. `static_droplet_2D.py`: summary **1.1847e-03** (mass_drift 0.0) —
   bit-stable.
4. `oscillating_droplet_2D.py` (default dual_only): l2
   **0.1785660454150319** / tail **0.9992507831101141** / linf
   0.32842491275938274 / mass 2.676988154993443e-14 — **bit-identical
   to 16 digits** (score.json diffed).
5. No-remap per-step-Delaunay path: 200-step driver replica
   bit-identical pre/post edit (l2 0.4022819035367851, KE_max
   0.005770763455074671).
6. hyperct: untouched this lane (no suite run required).

### 2.5 Changed files

- `ddgclib/dynamic_integrators/_integrators_dynamic.py` —
  `_retopologize_multiphase` retopo_remap kwarg + remap stages.
- `ddgclib/operators/mass_redistribution.py` — NEW
  `restore_pressure_multiphase`, `phase_volume_totals`,
  `anchor_phase_pressure_levels`.
- `ddgclib/multiphase.py` — `MultiphaseSystem.vol_corr` (ones;
  bit-neutral default) + gauge in `compute_phase_pressures`.
- `ddgclib/tests/test_case_oscillating_droplet.py` — NEW
  `TestConservativeRetopoRemap2D` (3 tests: one-call pressure
  neutrality 1e-9 bound, 40-step full-Delaunay KE < 1e-5 with active
  rewiring asserted, arg validation).
- `docs_temp/debug_session/laneD-conservative-retopo-remap.md` (this
  log), `debugging_plan.md` (status entry).

### 2.6 Caveats / proposals for future lanes (do not apply blind)

1. **Proposal**: flip the 2D droplet runner default `retopo_policy_2d`
   from `'dual_only'` to delaunay + `retopo_remap='conservative'` once
   long-run stability is proven (multi-t_end horizons, refine 4/4,
   c_s=5): it already scores slightly better (l2 0.17479 vs 0.17857)
   AND keeps retopology active — the more honest benchmark.  Would
   re-pin `baseline_oscillation.json`.
2. **Proposal**: fix `dam_break_2D.py`'s dead `skip_triangulation=True`
   flag (either bind it into setup's retopo_fn partial or forward the
   flag to callables in `_do_retopologize`) — currently the comment
   lies about what runs.
3. The anchor holds the volume-weighted MEAN level per phase; in
   gravity-stratified phases (dam break) this interacts with the
   hydrostatic structure only through the restore (structure
   preserved), but the 2.3x KE-plateau difference on the stalled smoke
   is unexplained — probe before relying on remap there.
4. `MultiphaseEOS.__call__` (fallback pressure path) does not read
   `vol_corr`; irrelevant for the production pipeline (forces read
   `v.p_phase` populated by `compute_phase_pressures`) but a consumer
   that calls the EOS directly during a remap run would see unguaged
   densities.
5. Gauge/ledger magnitudes on the full run stayed benign
   (|vol_corr−1| ≤ 1.1e-2 outer, 3.8e-4 droplet; anchored levels
   bounded), but any future 3D application must re-verify — the 3D
   skip-triangulation measurement pass has different boundary-volume
   bookkeeping (lane-B verified it clean for the 3D droplet).

