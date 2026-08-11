# Interface Stress Rewrite — Per-Phase Summed Force

## Context

The oscillating-droplet case at
`cases_dynamic/oscillating_droplet/` exhibits catastrophic kinetic-energy
growth (baseline oscillation score = 5.38 — L2 error of r_apex is 538%
of ε·R0; tail KE grows 2.7× across the run). Five parallel
investigations (summarized in
`/home/stefan_endres/.claude/plans/hidden-baking-thacker.md`) ranked
the root causes; after Phases 0–5 the remaining dominant issue is the
interface stress computation.

The current `multiphase_stress.py` treats interface vertices with a
single `own_phase = v.phase`, using only that phase's pressure and
viscosity on every edge (including edges that straddle the other
phase). The user's stated design requires instead that interface
vertices sum per-phase contributions, each using that phase's own EOS
and viscosity on its own sub-face of the dual cell. This spec captures
the target behaviour and the machinery needed to implement it.

## Current behaviour

`ddgclib/operators/multiphase_stress.py`:

- Lines 77-82 / 119-123: pressure flux uses `p_i = v.p_phase[own_phase]`
  and `p_j = v_j.p_phase[own_phase]` — own-phase pressure on both
  ends of every edge.
- Lines 105-111: edges to bulk neighbours in a different phase are
  silently skipped (`continue`).
- Line 90: `mu = mps.get_mu(own_phase)` applied uniformly on all
  edges, including cross-phase interface–interface edges. No
  harmonic mean at the viscosity jump.
- Lines 134-136: surface tension added as a separate `F_st` force
  only on interface vertices (already correct; keep).

The companion doc `cases_dynamic/oscillating_droplet/INTERFACE_STRESS.md`
describes this as the design intent ("no harmonic mean"), but the user
has now decided the own-phase-only scheme is incorrect for a two-sided
Lagrangian parcel: the phase-0 sub-cell contribution is missing
entirely, and at a μ-discontinuity the interior face flux should be
harmonic-mean.

## Target behaviour

For every vertex `v` with `phases_present(v) = {k : m_phase[k] > 0}`,
compute

    F_i = sum_{k in phases_present(v)} F_i^{(k)}

where `F_i^{(k)}` uses:

- `p_phase[k]` on both sides of each sub-face (never mixing phases
  across a face),
- viscosity `μ_k` on the phase-k sub-face (harmonic mean at μ-jumps
  on edges that straddle the interface),
- dual area vector `A_ij^{(k)}` from the phase-k sub-polygon (2D) /
  sub-polyhedron (3D) of the dual cell.

Bulk vertices have `phases_present = {v.phase}` only — the sum
collapses to the current single-phase formula, so bulk physics is
unchanged.

Surface tension remains a separate `F_st` force on interface vertices.
Do NOT add γκ to the pressure field.

## Required new machinery

1. **Exact 2D dual-volume split.** Intersect each interface vertex's
   barycentric dual polygon with the piecewise-linear interface curve
   to obtain per-phase sub-polygons and the sub-edge length in each
   phase. Location: new module `ddgclib/geometry/_dual_split_2d.py`.
2. **Exact 3D dual-volume split.** Dual polyhedron × triangulated
   interface surface → per-phase sub-polyhedra. May defer to a
   follow-up window.
3. **Split-method flag on `MultiphaseSystem.split_dual_volumes`** —
   `method={'exact', 'neighbour_count'}`, default `'neighbour_count'`
   so existing cases (dam break, etc.) are unaffected until callers
   opt in.
4. **Per-phase dual_area_vector.** Needs an `A_ij^{(k)}` accessor
   derived from the sub-polygon edge between `v_i` and `v_j`.
5. **Harmonic-mean μ utility** at cross-phase faces:
   `μ_face = 2·μ_i·μ_j / (μ_i + μ_j)`.

## Verification plan

Use the Phase-0 regression harness from
`cases_dynamic/oscillating_droplet/src/_metrics.py`. Baselines are
checked into `cases_dynamic/oscillating_droplet/baselines/`.

- **Flat interface, uniform pressure, two phases.** A periodic or
  large-domain 2D setup with a horizontal interface y=0 and uniform
  p on each side (no γκ because κ=0). Expect `F_i ≈ 0` to machine
  precision on every interface vertex. This is the fundamental
  check that the per-phase summed stress cancels correctly.
- **Static circular droplet with Young-Laplace equilibrium.**
  Re-run `static_droplet_2D.py`. Target `equilibrium_score.summary`
  to drop by ≥1 order of magnitude vs the Delaunay baseline
  (7.41e-3). Mass drift must stay ~1e-16.
- **Oscillating droplet decay.** Re-run `oscillating_droplet_2D.py`.
  Target `oscillation_score.tail_growth < 1.0` (KE decays) and
  `l2_error_normalized < 0.2` (r_apex(t) tracks Rayleigh-Lamb to
  within 20%).
- **Dam break.** Run `cases_dynamic/dam_break/` and confirm no
  regression — the 2D and 3D variants must still produce physically
  plausible output.

## Files to modify

- `ddgclib/operators/multiphase_stress.py` — rewrite
  `multiphase_stress_force` for the per-phase summed scheme.
- `ddgclib/multiphase.py` — extend `split_dual_volumes(method=...)`.
- `ddgclib/operators/curvature_2d.py` — verify sub-edge consistency
  with the new dual split (the tangent-vector identity depends on
  the sub-edge endpoints).
- `ddgclib/_curvatures_heron.py` — 3D surface tension on interface
  sub-mesh (unchanged in scheme, but verify).
- NEW: `ddgclib/geometry/_dual_split_2d.py` — polygon × polyline
  intersection for exact per-phase sub-polygons.
- Tests: `ddgclib/tests/test_multiphase_stress_per_phase.py`.

## Sibling windows

This work is coordinated with two companion tasks that should be done
in their own context windows:

- **Exact geometric dual-volume split** (window C in the master
  plan) — the prerequisite for item (1)/(2) above. Can land before
  this rewrite as a standalone feature gated on `method='exact'`.
- **2D Lamb damping formula literature review** (window B) — fixes
  the non-standard `(2l²-1)` prefactor in
  `cases_dynamic/oscillating_droplet/src/_analytical.py`. Independent
  of this rewrite.

## Upstream blockers discovered during Phase 4

Attempting to enable `remesh_mode='adaptive'` in the oscillating
droplet revealed two hyperct issues that must be fixed before
adaptive remesh is viable for any multiphase case:

- `hyperct/remesh/_operations_2d.py:198` — `edge_split_2d` sets the
  new midpoint mass to the arithmetic mean of its endpoints, silently
  inflating total mass on every split (static-droplet test: mass
  went 9.7 → 187 over 100 steps).
- The adaptive driver uses a single global `h_local` for its
  auto-scaled L_min / L_max thresholds, so a mixed fine-droplet /
  coarse-outer mesh triggers unbounded splits in the outer region
  when L_max is sized to the droplet.

Both are tracked here because they block the interface-preserving
retopology needed to cleanly validate the per-phase stress rewrite
on the oscillating case.
