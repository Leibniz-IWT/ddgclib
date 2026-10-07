# Audit: 2D curvature operator consistency (curvature_2d.py + _curvatures_heron.py)
> Sources checked: docs_temp/02_physics_foundations.md §3.4/§6, docs_temp/code_map/multiphase_surface_tension.md §5/§7 (flags 13, 14), docs_temp/sources/debugging_plan_distilled.md (Tier 2B step 1), ddgclib/operators/curvature_2d.py, ddgclib/operators/multiphase_stress.py, ddgclib/_curvatures_heron.py, ddgclib/geometry/_interface_subcomplex.py, ddgclib/geometry/_dual_split_2d.py, ddgclib/dynamic_integrators/_integrators_dynamic.py, cases_dynamic/oscillating_droplet/src/_setup.py | Written 2026-07-02 by physics-audit workflow

Probe scripts + raw outputs: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/curvature-2d-consistency/` (probe_ngon.py, probe_heuristic_failure.py, probe_real_droplet.py, probe_post_retopo.py, probe_force_eval_time.py, probe_floor_decompose.py).

## 1. What the physics requires

The 2D surface-tension force on an interface vertex is the integrated curvature normal over the dual portion of the interface curve (02_physics_foundations.md §3.4, item 1):

    F_st_i = gamma * ∫_{Gamma_i} kappa N ds = gamma * (t_next - t_prev)     (FTC identity)

exact on a piecewise-linear curve. Two requirements follow: (i) `t_prev`/`t_next` must be the tangents of the *true* polyline edges at `v` (correct curve-neighbour identification); (ii) the estimator used by the force must be consistent with the neighbour identification used by the dual-volume splitter, or the pressure side and the tension side of the Young–Laplace balance see different interface geometry.

## 2. What the code does

- `integrated_curvature_normal_2d` (`ddgclib/operators/curvature_2d.py:91-149`) never receives `HC`. Curve neighbours come from `_select_curve_neighbours` (`curvature_2d.py:50-88`): sort *all* 1-ring interface neighbours by polar angle, pick the pair bracketing the largest angular gap.
- The production caller `_interface_surface_tension` builds the neighbour set as the raw 1-ring `{nb for nb in v.nn if nb.is_interface}` (`ddgclib/operators/multiphase_stress.py:234`) — including flag-complex chords that are NOT interface edges — and routes 2D to `surface_tension_force_2d` (`multiphase_stress.py:262` and `:277`).
- The dual splitter uses the exact subcomplex lookup instead: `split_dual_polygon_2d` calls `curve_neighbours(v, interface)` which tests `frozenset({v.x, nb.x}) in HC.interface_edges` (`ddgclib/geometry/_dual_split_2d.py:131-139`, `ddgclib/geometry/_interface_subcomplex.py:145-166`). So the inconsistency named in the item is real at the source level.
- `hndA_i_interface` (`ddgclib/_curvatures_heron.py:310-393`) is the cotangent-Heron 3D stencil. Its docstring claims "Works in both 2D and 3D by padding 2D vectors to 3D" (`:317-318`). In 2D its apex enumeration needs interface *triangles*: `_apex_via_interface_triangles` (`:59-90`) reads `HC.interface_triangles`, which `extract_interface` sets to the **empty set** in 2D (`_interface_subcomplex.py:87,112-114` — `iface_tris` only filled when `dim==3`); the legacy fallback `vi.nn ∩ vj.nn ∩ interface_set` (`:351`) is empty on a clean polyline. It is only ever called with `dim==3` in production (`multiphase_stress.py:271-274`, `:300-305`).

## 3. Probe design and OUTPUT

### Probe 1 — N-gon path comparison (probe_ngon.py), N = 16…256, uniform and 40%-angle-jittered, vertices on the unit circle

Paths: P1 = shipped heuristic `integrated_curvature_normal_2d`; P2 = ground-truth polygon FTC from known connectivity; P3 = FTC via exact `curve_neighbours`; P4/P5 = `hndA_i_interface` with `HC=None` / with a 2D-extracted mock HC (`interface_triangles=set()`). References: smooth-circle dual-arc integral (integral of kappa N ds between the chord-tangency arc midpoints), uniform 2π/N magnitude, and pointwise kappa = |ΔT|/ℓ_chord vs 1/R.

```
uniform N=16..256:  P1 vs P2 max = 0.0 (all N);  P3 vs P2 max = 0.0 (all N)
                    P2 vs smooth dual-arc integral: 8.7e-16 … 7.4e-15  (machine precision)
                    |P2| vs 2pi/N: 2.52e-03 → 6.16e-07, order +3.00
                    kappa=|DT|/l_chord vs 1/R: 5.6e-16 … 2.7e-13  (machine-exact)
                    P4, P5 max|HNdA| = 0.0 exactly (all N)   <-- hndA_i_interface returns ZERO in 2D
jittered N=16..256: P1 vs P2 = 0.0; P3 vs P2 = 0.0
                    P2 vs smooth dual-arc integral: 7.0e-16 … 9.4e-15  (machine precision)
                    |P2| vs 2pi/N: 7.46e-02 → 4.79e-03, order ≈ +1.0   (partition mismatch, not curvature error)
                    kappa vs 1/R: 2.18e-03 → 9.78e-06, order ≈ +2.0
Gauss-Bonnet: |sum DT| ≤ 1.1e-15 all cases; total turning − 2π = O(h²)
```

Key exactness fact: for vertices ON a circle the chord tangent equals the smooth tangent at the arc midpoint, so the FTC per-vertex value equals the smooth-circle integral over the chord-tangency arc partition to machine precision **even on 40%-jittered polygons**. The apparent "first-order" error (7.46e-2 → order 1.0) appears only when the reference uses a *different* dual partition (equal arcs) — i.e. it is a dual-partition bookkeeping mismatch, not curvature truncation.

### Probe 2 — heuristic failure modes (probe_heuristic_failure.py)

```
CASE 1 convex N-gon + all v_i–v_{i+2} chords (4 interface nbs each):
  N=16/32/64: wrong-pair count 0, max|heuristic−truth| = 0.0
  cotan hndA_i_interface (now fed K3 apexes): max|HNdA| = 5.8e-2 / 7.5e-3 / 9.4e-4
     = 0.15 / 0.04 / 0.01 × the FTC magnitude → NOT the FTC value (garbage, decaying)
CASE 2 locally concave polyline vertex + one flag-complex chord:
  1-ring iface nbs {(-1,1),(1,1),(2,1.2)}; heuristic picks ((-1,1),(2,1.2)) — the CHORD;
  exact picks ((1,1),(-1,1)).  |DT| exact 1.4142 vs heuristic 1.2308:
  abs error 0.2444 (17.28% relative), direction error 7.0°  — O(1), does not vanish with h.
CASE 3 convex vertex, chord inside the (prev,next) span: heuristic correct, error 0.0.
```

### Probes 3/3b/4 — the real droplet case (probe_real_droplet.py, probe_post_retopo.py, probe_force_eval_time.py)

`setup_oscillating_droplet(dim=2)` (defaults, 32 interface vertices), epsilon ∈ {0, 0.05 l=2, 0.3 l=4}:

```
all three setups: chord-contaminated 1-rings = 0/32, heuristic pick != exact pick = 0,
                  max |DT_heur − DT_exact| = 0.0
after 1x _retopologize_multiphase: identical (all zeros)
after 20 dynamic steps (positions moved, no refresh since):
                  curve_neighbours degenerate = 32/32  <-- coordinate-keyed
                  HC.interface_edges is STALE the moment any vertex moves
50-step production run (retopo every step), monkeypatched at force-eval time:
  calls=1600, stale_or_degenerate=0, ring_ne_exact=0, pick_differs=0, max_dDT=0.0
```

The shipped heuristic and the exact lookup are **bit-identical at every production force evaluation** of the oscillating/static droplet case. And the "exact" lookup is unusable outside a freshly-refreshed state: its `frozenset({v.x, nb.x})` keys break on any vertex motion (probe 3b: 32/32 degenerate), in which case `curve_neighbours` returns `(None, None)` and a naive "fix" routing the force through it would silently zero the surface tension whenever the displacement gate (`_integrators_dynamic.py:352-355`) skips a refresh.

### Probe 5 — floor decomposition (probe_floor_decompose.py), static droplet step 0

```
step-0 max|F| over interface verts = 2.3749e-03   (reproduces pinned peak exactly)
max |F_st| per vertex               = 1.2218e-02
ideal uniform N-gon (N=32, R0=0.01, gamma=0.05):
  F_ST = 2γ sin(π/N) = 9.801714e-03 ;  F_p = (γ/R0)|m_next−m_prev| = 9.754516e-03
  ideal curvature-side imbalance |F_ST − F_p| = 4.72e-05  =  2.0% of the observed peak
```

So the polygon-vs-smooth-circle truncation of the curvature normal accounts for only **~2%** of the pinned 2.37e-3 step-0 peak; ~98% of the floor comes from the pressure-flux / dual-volume-split side of the balance on the irregular mesh. This refutes the annotation "(100% curvature-stencil O(h) truncation)" at `docs_temp/02_physics_foundations.md:135` and confirms the completeness-critic note in `docs_temp/sources/debugging_plan_distilled.md` (Tier 2B routing already landed, floor unchanged): **a Tier 2B 2D curvature-normal rewrite has essentially nothing left to gain** — the operator is already exact for the PL interface and machine-precision-consistent with the smooth circle under the correct (chord-tangency) dual partition. The remaining lever on the 2.27e-3 floor is making the *pressure-side* dual-face geometry consistent with the interface arc (e.g. bulge/arc reconstruction via `reconstruct_arc_length_and_bulge_area`, `curvature_2d.py:184-231`, or arc-aware dual faces), not the curvature stencil.

## 4. Verdicts and reasoning

**(a) Angular-gap heuristic vs exact `HC.interface_edges` — DESIGN_LIMITATION (with a documented latent failure mode).**
The inconsistency is real at source level (force path `multiphase_stress.py:234` + `curvature_2d.py:50-88` heuristic vs splitter `_dual_split_2d.py:131-139` exact). A concave interface vertex with a flag-complex chord makes the heuristic pick the chord (probe 2 case 2: 17.3% magnitude / 7.0° direction error, O(1)). However: (i) on every droplet configuration tested — static, standard perturbation, aggressive l=4 ε=0.3, and 1600 production force evaluations over 50 dynamic steps with per-step retopo — heuristic and exact picks are bit-identical (all diffs 0.0); (ii) the "exact" lookup cannot simply replace the heuristic because its coordinate-tuple keys go stale on any vertex motion (32/32 degenerate after 20 steps), which would silently zero the tension force under the displacement gate. The heuristic is a deliberate motion-robust choice; its convexity assumption is even stated in its docstring ("robust for convex droplets", `curvature_2d.py:56-57`). It is a genuine hazard only for locally concave interfaces with interface-vertex chords (contact lines, large-amplitude modes, post-Delaunay slivers).

**(b) `hndA_i_interface` with dim=2 — genuinely different (cotangent) estimator that does NOT reduce to the FTC form; in practice it returns identically zero in 2D.** With a 2D-extracted HC (`interface_triangles == set()`) the apex cache is empty and every edge is skipped; with `HC=None` the legacy nn-intersection is empty on a clean polyline — probe 1: max|HNdA| = 0.0 exactly, all N. When spurious K3 chords exist it returns nonzero cotangent values that are 0.15×–0.01× the FTC magnitude (probe 2 case 1) — not a curvature normal. The docstring claim "Works in both 2D and 3D by padding 2D vectors to 3D" (`_curvatures_heron.py:317-318`) is false in the physics sense (padding only fixes `np.cross` shapes). Not a production bug — `_interface_surface_tension` only calls it for `dim==3` — but it is a documented trap: any caller trusting the docstring in 2D gets silent zero surface tension.

**(c) Bonus finding — stale doc attribution of the 2D floor.** The "100% curvature-stencil O(h) truncation" attribution of the 2.2717e-3 floor (`02_physics_foundations.md:135`; also `multiphase_stress.py:216-219` "first-order … static-droplet residual") is quantitatively wrong: ideal curvature-side truncation is 4.7e-5 ≈ 2% of the observed 2.3749e-3 step-0 peak (probe 5). The first-order signature belongs to the dual-partition mismatch between the tension integral (chord-tangency arcs) and the pressure flux (dual-polygon chords), plus the `neighbour_count` volume split.

## 5. Suggested fixes (in priority order)

1. Documentation/plan fix (cheap, high value): update `02_physics_foundations.md:135` and `multiphase_stress.py:216-219` — the 2D floor is NOT curvature-stencil truncation; close debugging-plan item "Tier 2B step 1" as "routing landed, curvature side exact; remaining lever is pressure-side/dual-partition consistency (arc/bulge reconstruction)".
2. Fix the `hndA_i_interface` docstring (`_curvatures_heron.py:317-318`): state that in 2D it returns zeros (no interface triangles) and that 2D callers must use `surface_tension_force_2d`; optionally add an early `if len(v.x_a) < 3 or not HC.interface_triangles: raise/warn`.
3. Harden the heuristic instead of replacing it: in `_interface_surface_tension` (`multiphase_stress.py:234`), when `HC.interface_edges` is fresh (e.g. an epoch counter set by `extract_interface` and bumped by `_move`), filter the 1-ring set through `interface_nn(v, HC)` before calling `surface_tension_force_2d`, keeping the angular-gap heuristic as the fallback for stale/absent connectivity. Equivalently: make `extract_interface` store neighbour references (object identity) rather than coordinate-frozenset keys so the exact lookup survives vertex motion.
4. Add a regression test with a locally concave interface + chord (probe 2 case 2 geometry) pinning the neighbour selection.
