# laneQ: exact simplex-based dual face areas in 3D (axis `edge_area_source`)

Date: 2026-10-05. Library lane (hyperct + ddgclib), follows laneJ (measurement,
2026-09-25) and laneT's hull-edge finding (2026-10-02). Repo state at the
start: ddgclib `8c9cc97`, hyperct `476c289`, both on master, env `ddg`, cwd
repo root. Python `/home/endres/anaconda3/envs/ddg/bin/python`.

## Verdict up front

1. **Kernel.** `hyperct.ddg.simplex_dual_face_areas(HC, dim)` computes the
   oriented barycentric dual face area vector of every directed edge of every
   vertex (hull included) from the top-simplex cache in one vectorised pass:
   per simplex `A_ij^T = |T| (grad phi_j - grad phi_i) / (dim + 1)`, which is
   the sum of the two barycentric-subdivision triangles (edge midpoint, face
   barycentre, cell barycentre) at the edge. The sign of a simplex is its
   combinatorial orientation propagated over the facet adjacency, not
   `sign(det)`: the droplet mesh carries 44 exactly flat tetrahedra (of 2540)
   and a Delaunay rebuild of the refinement 2 box 13, and a flat tetrahedron's
   dual face piece lies in its own plane, so no geometric rule can orient it.
   Measured on every mesh tried: antisymmetric to the bit, every interior cell
   closes to 1.6e-16 (flat tetrahedra included), linear precision (tensor
   identity) 2.0e-15, hull half cells closed by their hull facets to 3.3e-16
   (against the box-face normals beside flat hull tetrahedra: 1.7e-16 on the
   droplet, 1.2e-16 on the Delaunay-rebuilt box), equal to laneJ's per-edge
   polygon to 2.1e-15 on every link and to laneT's per-tetrahedron quads to
   1.3e-15 on every link without a flat tetrahedron (on the hull-hull links
   that carry one, 179 of the droplet's 6220, the quads' per-piece sign rule
   `quad . d_ij > 0` is undefined and the two differ by up to 163 %; the
   kernel is the one that closes the half cell there). 11 ms on the 2540
   tetrahedra of the droplet against 187 ms for `batch_e_star`.
2. **Axis.** `edge_area_source` is an explicit 3D `SolverMethods` field
   (default `None` = the legacy source of the path) with the values
   `e_star_cache` (the `batch_e_star` fan cache, validated, every droplet
   pin), `p_ij_simplex` (the kernel cache, validated: default of
   `hydrostatic_3D`), `p_ij` (the same face built per edge, opt-in) and
   `p_ij_ring` (the legacy ring walk with its face heuristic, status broken,
   kept so that every pre-laneQ number of a cache-less path and laneJ's
   `p_ij` arm reproduce). Every retopology function that builds duals fills
   `HC._edge_area_cache` from the chosen source and records the value on
   `HC._edge_area_source`; every operator that reads a dual face follows it;
   `effective_methods` reports the recorded value. The wrapper rejects the
   combinations that would be silent no-ops.
3. **laneJ's F4b and laneT's tie are fixed.** The per-edge construction of
   the value `p_ij` (`stress._dual_area_vector_3d_simplex`) reads the ring
   order and the face vertices from the tetrahedra around the edge and runs
   an open chain through the edge midpoint on a hull edge: on the 56
   free-surface edges of the refinement 2 column it is exact to 1e-14 where
   the ring walk was off by up to 37 % on 30 of them (the strict xfail of
   laneT is a passing test now). The heuristic is not gone: it is the
   registered value `p_ij_ring` and the default construction of a mesh no
   retopology has tagged (so every step-0 number is bit-identical).
4. **Default decision (protocol rule 5).** The exact faces ARE adopted as the
   default of `hydrostatic_3D` (re-pinned: the refinement 1 pins move in
   round-off only, the refinement 2 remap arm by -2.2 % / -2.6 %, which is
   the removed hull-edge error; a 1e-15 shift now moves the refinement 2
   column by 9.7e-13 instead of 2e-6; 2.5 to 3.5x faster per step) and are
   NOT adopted as the default of the 3D droplet presets: on the main
   benchmark every exact-area arm fails the better-l2-AND-tail rule. The
   full table is in section 4; in short, the exact faces cut the early bump
   (q2 +0.245 -> +0.093) and halve the final inflation (1.87 % -> 0.83 % R0),
   which exposes laneG's genuine over-decay, so l2 rises from 0.24811 to
   0.28653 (+15.5 %) and the tail from 0.08417 to 0.08825 (+4.8 %). With the
   redistribution lever off the error is one-signed (-0.38 .. -0.20 over the
   four quarters, no overshoot, mass drift 1e-16, tail 0.0096) and l2 is
   0.32625; projection cadence 2 inflates in 3D (l2 0.34581). The
   `oscillating_droplet_3D`, `oscillating_droplet_3D_delaunay`,
   `static_droplet_floor_3D` and `dam_break_3D` presets keep the fan cache.
5. **Cost.** With the exact cache the 3D retopology of the 2/2 droplet takes
   231 ms instead of 407 ms (no fan walk), the force evaluation is unchanged
   (cache lookups, 70 ms for 377 vertices); the A.5.b floor runs at 0.36
   against 0.56 s/step, the full droplet at 0.34 against 0.55 s/step, the
   column at 22 against 55 ms per step (refinement 1) and 159 against 550
   (refinement 2). The uncached `p_ij` costs 603 ms per force evaluation
   (1.07 s/step on the floor), the ring walk 1790 ms (2.2 s/step).

## 1. What changed and where

hyperct (`/home/endres/projects/hyperct`):

- `hyperct/ddg/_dual_volume.py`: `simplex_dual_face_areas(HC, dim)` (2D and
  3D, one branch), `_orientation_signs(idx, det)` (combinatorial orientation:
  the facet relation `s_1 o_1 = -s_2 o_2` propagated as connected components
  of a two-layer graph, each component signed so that its summed signed
  volume is positive; a flat simplex takes the orientation of its
  neighbours, an inconsistent one falls back to `sign(det)`),
  `_facet_parity`. Exported from `hyperct.ddg`.
- `hyperct/tests/test_dual_face_areas.py` (14): orientation = `sign(det)`
  on every simplex of positive measure, antisymmetry, closure and linear
  precision at interior vertices on a random cloud and on the cube lattice
  with its flat tetrahedra, hull half cells closed by the hull facets (cloud)
  and against the box-face normals beside the lattice's flat hull tetrahedra
  (36 of its 98 hull vertices; added in fix round 1), equality with an
  independent per-edge `p_ij` polygon, 2D parity with the dual segment, the
  two error paths.

ddgclib:

- `ddgclib/operators/stress.py`: `dual_area_vector(..., source=None)` (3D:
  `source` or `HC._edge_area_source` picks the construction),
  `_dual_area_vector_3d_simplex` (per-edge exact polygon from the tetrahedra,
  open chain on hull edges, ring-walk fallback for a non-manifold link),
  `edge_area_vector` (cache entry, else `dual_area_vector`: the one lookup
  every reader of a dual face uses), `_EXACT_EDGE_SOURCES`.
- `ddgclib/operators/gradient.py` (`velocity_laplacian`),
  `operators/stabilisation.py` (`density_diffusion_step`),
  `operators/multiphase_stress.py` (`_csf_dual_surface_tension`): read the
  face through `edge_area_vector` (they bypassed the cache before, audit
  A11). `stress_force`, `multiphase_stress_force`,
  `velocity_difference_tensor` and `scalar_gradient_integrated` already read
  the cache first and follow the axis through it; the `simplex_gradient`
  fluxes read no dual face.
- `ddgclib/dynamic_integrators/_integrators_dynamic.py`:
  `_retopologize(..., edge_area_source=None)` step 5b fills the cache from
  the source (`p_ij_simplex`: the kernel, no fan walk, so no fan-failure
  promotion; `p_ij` / `p_ij_ring`: the fan walk still runs for its failed
  set and volumes, then the cache is cleared, which is laneJ's `no_cache`
  wrapper) and records `HC._edge_area_source`; `_do_retopologize` forwards
  the kwarg (by name to custom callables); `_retopologize_multiphase` passes
  it to both of its `_retopologize` calls; the five integrators take it.
- `ddgclib/methods/_retopo.py`: `_set_edge_area_source` for the retopologies
  that build no fan cache; `bare_dual_refresh` and
  `retopologize_material_delaunay` take `edge_area_source` (`None` = the
  legacy ring walk, `'e_star_cache'` raises).
- `ddgclib/methods/_axes.py`: the axis is explicit (`default=None`); values
  `None`, `e_star_cache`, `p_ij_simplex` (validated), `p_ij` (opt-in),
  `p_ij_ring` (broken), the 2D values reported only; evidence on the
  `dual_only_bare` and `pressure_flux=centred` entries.
- `ddgclib/methods/_config.py`: field `edge_area_source`; validation (3D
  only; applied by `delaunay`, `dual_only`, `dual_only_bare`,
  `delaunay_material`; `e_star_cache` only where the fan cache is built;
  `backend` only with `e_star_cache`); `integrator_kwargs` carries it.
- `ddgclib/methods/_effective.py`: reports `HC._edge_area_source`, else
  infers (`e_star_cache` with a cache, `p_ij_ring` without). The keys
  `batch_e_star_cache` / `p_ij_ring_3d` are gone.
- `ddgclib/methods/_presets.py`: `hydrostatic_3D` reads
  `edge_area_source='p_ij_simplex'`; notes on the three 3D droplet presets.
- Tests: `ddgclib/tests/test_edge_area_source.py` (26), the two converted
  tests of `test_determinism.py` (the strict xfail of laneT is `p_ij_ring`'s
  defect now, and the exact sources are tested on the same 56 edges),
  `test_methods.py` / `test_case_hydrostatic.py` key renames, the four
  hydrostatic 3D re-pins.
- Drivers: `cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py`
  rewritten around the axis (arms = `preset.replace(...)`, the custom
  wrappers and the scratch cache builder are gone; new sub-commands
  `hydrostatic`, `dambreak`, `--out`, `--snapshots`);
  `cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d` runs the
  ring, the exact cache and the per-edge arm through the axis.
- `METHODS.md` regenerated, section 4 rows of the 3D cases, `DEVELOPMENT.md`,
  `debugging_plan.md`.

No behaviour changed for a configuration that does not set the axis except
`hydrostatic_3D` (flipped) and `velocity_laplacian` / `density_diffusion_step`
/ the `csf_dual` stencil in 3D when a cache exists (they read it now; no
pinned case reaches that combination: `density_diffusion` is 2D, the other
two have no 3D caller).

## 2. Kernel invariants (hyperct tests, scratch check on the lane's meshes)

| mesh | simplices (flat) | kernel | antisym | interior closure max (n) | hull half-cell closure max (n) | tensor max | vs laneJ polygon (n closed links) | vs laneT quads (links without a flat tet) |
|---|---|---|---|---|---|---|---|---|
| box r2 builder | 768 (0) | 3.8 ms | 0 | 4.1e-17 (91) | 1.0e-16 (98) | 1.1e-15 | 8.3e-16 (1528) | 1.4e-16 |
| box r2 after Delaunay | 781 (13) | 3.3 ms | 0 | 8.6e-17 (91) | 1.2e-16 (98, box-face normals; 36 vertices beside a flat hull tet) | 5.6e-16 | 8.3e-16 (1554) | 1.4e-16 |
| ball r2 builder / Delaunay | 768 (0) | 18 / 3.2 ms | 0 | 6.5e-17 / 8.3e-17 | 1.1e-16 / 1.7e-16 | 8.5e-16 / 5.7e-16 | 8.8e-16 / 7.5e-16 | 1.0e-15 |
| rectangle r3, disk r2 (2D) | 256, 64 | 1.2, 0.9 ms | 0 | 4.5e-17, 5.3e-17 | 4.5e-17, 1.7e-16 | 5.0e-16, 3.9e-16 | (2D: equal to the dual segment, test) | |
| random cloud 300 | 1754 (0) | 10 ms | 0 | 1.3e-16 (249) | 3.3e-16 (51) | 7.0e-16 | 7.2e-15 (3910) | 1.1e-14 |
| droplet 2/2 setup | 2540 (44) | 10.5 ms | 0 | 1.6e-16 (377) | 1.7e-16 (98, box-face normals; 82 beside a flat hull tet) | 2.0e-15 | 2.1e-15 (5644) | 1.3e-15 (6041 of 6220) |

The combinatorial sign equals `sign(det)` on every simplex of positive
measure of every mesh (tested); the 44 and 13 flat tetrahedra are oriented
by their neighbours, which is what closes the cells there (dropping them,
as the laneO reference `simplex_area_vectors` does, opens the cells at
their vertices by the flat piece).

The last column is qualified (fix round 1, reviewer's finding, re-measured):
laneT's reference `test_determinism._per_tetrahedron_area` signs every
quadrilateral on its own by `quad . d_ij > 0`, which is undefined on a flat
tetrahedron (its piece is perpendicular to the edge), so on the 179 of 6220
directed links of the droplet mesh that are hull-hull edges with 1 or 2 flat
tetrahedra (link sizes 1, 3, 4, 5) the two differ by up to 163 %; on every
other link they agree to 1.3e-15. The kernel is the one that is right
there: the hull half cells close against the box-face normals at every one
of the 82 hull vertices beside a flat tetrahedron (1.7e-16; 1.2e-16 for the
36 of the Delaunay-rebuilt box), which is now a hyperct test
(`test_hull_half_cell_closes_beside_flat_hull_tetrahedra`). The droplet
count is a one-off measurement (scratch script on the `static_droplet_floor_3D`
setup after one retopology, the mesh of section 3); the lattice case is the
regression lock.

## 3. Static accuracy on the lane's meshes (`laneQ_static.json`)

Same metrics as laneJ (closure = |sum_j A_ij| / sum_j |A_ij|; tensor =
max |M_i - Vol_i I| / Vol_i with `M_i = 1/2 sum_j d_ij (x) A_ij`; force =
|F_i + g Vol_i| / (|g| Vol_i) for the centred force of the point-valued
pressure 5 + g.x, g = (1, 2, 3), mu = 0; the force metric mixes closure
times the offset 5 with linear precision). Three paths: `cache` = the fan
cache as the force reads it, `p_ij` = the exact per-edge construction (=
the kernel to 1e-15), `p_ij_ring` = the legacy ring walk. Median / max.

| mesh | class (n) | cache: closure, tensor, force | p_ij: closure, tensor, force | p_ij_ring: closure, tensor, force |
|---|---|---|---|---|
| box r2 (cache vs p_ij per edge 0.125 / 0.625) | interior (11) | 0 / 0; 4.5e-2 / 7.1e-2; 3.0e-2 / 3.9e-2 | 1.9e-17 / 5.3e-17; 1.2e-15 / 1.3e-15; 1.3e-15 / 3.0e-15 | = p_ij (ring vs p_ij 5.6e-17 / 3.3e-16) |
| | bnd-adjacent (80) | 0 / 0; 6.3e-2 / 1.2e-1; 3.8e-2 / 1.6e-1 | 2.5e-17 / 8.1e-17; 1.1e-15 / 1.4e-15; 1.6e-15 / 4.2e-15 | = p_ij |
| ball r2 (0.127 / 0.250) | interior (9) | 2.6e-17 / 4.4e-17; 1.5e-3; 2.3e-3 / 2.6e-3 | 3.2e-17 / 5.0e-17; 1.4e-16; 4.7e-16 / 1.9e-15 | = p_ij |
| | bnd-adjacent (82) | 3.1e-17 / 8.1e-17; 3.2e-2 / 5.8e-2; 3.1e-2 / 9.6e-2 | 4.9e-17 / 1.3e-16; 2.0e-16 / 5.5e-16; 1.1e-15 / 4.5e-15 | = p_ij |
| droplet eps 0 (static-floor preset, 2/2, one retopology; cache vs p_ij 0.159 / 0.499; ring vs p_ij mean 1.3e-3, max 0.056) | interior (295) | 7.0e-17 / 1.7e-2; 8.4e-2 / 2.5e-1; 2.5e-1 / 4.3e+1 | 4.6e-17 / 1.6e-16; 4.5e-16 / 1.7e-15; 8.8e-14 / 4.0e-13 | 6.8e-17 / 1.5e-2; 7.5e-16 / 1.5e-2; 1.7e-13 / 2.2e+1 |
| | interface (98) | 1.1e-2 / 1.7e-2; 8.4e-2 / 2.5e-1; 2.2e+1 / 4.3e+1 | 4.7e-17 / 1.6e-16; 3.9e-16 / 1.3e-15; 1.1e-13 / 4.0e-13 | 6.2e-17 / 9.1e-3; 5.4e-16 / 1.3e-2; 1.3e-13 / 2.2e+1 |
| | bulk droplet (91) | 3.2e-17 / 6.9e-17; 3.2e-2 / 5.8e-2; 2.8e-2 / 9.6e-2 | 5.0e-17 / 1.3e-16; 5.7e-16 / 1.7e-15; 1.3e-13 / 3.8e-13 | = p_ij |
| | bulk outer interior (106) | 3.6e-3 / 8.0e-3; 1.5e-1 / 2.0e-1; 2.6 / 1.2e+1 | 3.9e-17 / 1.0e-16; 4.4e-16 / 1.7e-15; 5.5e-14 / 1.9e-13 | 4.4e-3 / 1.5e-2; 2.1e-3 / 1.5e-2; 5.0 / 2.0e+1 |
| droplet eps 0.05 (dual_only preset; ring vs p_ij max 0.5) | interior (295) | 8.0e-4 / 3.4e-2; 9.6e-2 / 2.5e-1; 6.7e-1 / 6.2e+1 | 4.8e-17 / 2.7e-16; 4.3e-16 / 1.9e-15; 9.9e-14 / 9.6e-13 | 6.4e-17 / 2.3e-2; 5.7e-16 / 1.1e-1; 1.4e-13 / 4.7e+1 |
| | interface (98) | 1.5e-2 / 3.4e-2; 9.6e-2 / 2.5e-1; 3.3e+1 / 6.2e+1 | 5.1e-17 / 1.2e-16; 4.3e-16 / 1.9e-15; 1.3e-13 / 3.1e-13 | 5.4e-17 / 2.3e-2; 4.1e-16 / 8.3e-2; 1.3e-13 / 4.7e+1 |

laneJ's picture holds on the full outer mesh (the fan cache is 3 to 25 % off
linear precision and fails closure at every interface vertex by about 1 %;
the ring walk fails closure on the outer and interface cells by up to
2.3 %), and the exact construction closes every class to round-off.

## 4. Decision table: 3D droplet, refinement 2/2, `dual_only`, 872 steps

Every arm is `PRESETS['oscillating_droplet_3D'].replace(...)` run by
`diagnose_3d_edge_area_source.py dynamic --arm <name>` (results
`results_3d/score_laneQ_<arm>.json`, `methods_laneQ_<arm>.json`,
`diags_laneQ_<arm>.json`). Quarter means of `(R_max - envelope) / (eps R0)`
(+ = above the envelope) as in laneG section 2. Pin (laneB mesh):
`baselines/baseline_oscillation_3d.json`.

| arm (`replace`) | l2 | tail | linf | mass drift | R_max_peak | R_max(t_end) | KE_max [J] | q1 / q2 / q3 / q4 | s/step |
|---|---|---|---|---|---|---|---|---|---|
| `cache` (= preset, pin) | **0.24811443136179492** | **0.08417379628161890** | 0.59927 | 4.8e-14 | 0.010790238 | 0.010187 | 1.7430e-06 | +0.325 / +0.245 / -0.061 / -0.140 | 0.554 |
| `pij_simplex` (`edge_area_source='p_ij_simplex'`) | 0.28653087631979274 (+15.5 %) | 0.08825446509925354 (+4.8 %) | 0.56246 | 8.6e-14 | 0.010771833 | 0.010083 | 1.7047e-06 | +0.279 / +0.093 / -0.257 / -0.346 | 0.338 |
| `pij` (`'p_ij'`, uncached) | 0.28653087622018630 | 0.08825446501866843 | 0.56246 | 4.4e-14 | 0.010771833 | 0.010083 | 1.7047e-06 | +0.279 / +0.093 / -0.257 / -0.346 | 1.065 |
| `pij_ring` (`'p_ij_ring'`, laneJ's p_ij arm) | 0.28712884337228660 | 0.08839011881475957 | 0.56251 | 4.5e-14 | 0.010771858 | 0.010082 | 1.7050e-06 | +0.279 / +0.093 / -0.258 / -0.347 | 2.219 |
| `cache_noredis` (`redistribute_mass=False`) | 0.26658458963801102 | 0.01013388376375447 | 0.60340 | 1.1e-16 | 0.010500000 | 0.010224 | 6.5232e-06 | -0.371 / -0.213 / -0.208 / -0.121 | 0.536 |
| `pij_simplex_noredis` | 0.32625367445774917 | 0.00955295466699528 | 0.62052 | 1.1e-16 | 0.010500000 | 0.010179 | 6.4088e-06 | -0.384 / -0.341 / -0.280 / -0.201 | 0.324 |
| `cache_p2` (`projection_every=2`) | 0.28897894112151279 | 0.21086526376709805 | 0.58777 | 1.8e-14 | 0.010784486 | 0.010389 | 1.6343e-06 | +0.347 / +0.245 / +0.238 / +0.256 | 0.533 |
| `pij_simplex_p2` | 0.34581427827269451 | 0.14696420712984348 | 0.58418 | 2.8e-14 | 0.010780829 | 0.010464 | 1.5965e-06 | +0.360 / +0.261 / +0.308 / +0.392 | 0.332 |

Reading, channel by channel:

- The `cache` arm reproduces the pin in every key (l2, tail, linf, mass,
  R_max_peak, the step-0 dual-volume jump 0.3515625): the driver and the
  library default are validated.
- The `pij_ring` arm (laneJ's `p_ij` arm through the axis, 2.2 s/step;
  finished after the review, fix round 1) lands on laneJ's l2 0.28713 to
  five digits and within 0.2 % of the two exact arms in every key (l2
  0.28713 against 0.28653, tail 0.08839 against 0.08825, the same quarter
  signs): on the full outer mesh the ring walk's face heuristic (up to
  5.6 % per edge, section 3) is not what separates the exact faces from
  the pin; the fan cache is.
- The exact faces act on the bump: q2 +0.245 -> +0.093, the final inflation
  1.87 % -> 0.83 % R0, KE_max -2.2 %, the dual-volume drift after step 0
  halved (1.06e-4 -> 5.5e-5). They barely touch the overshoot (R_max_peak
  -1.7e-4 relative): that is the redistribution pump, which the `noredis`
  arms remove entirely (R_max_peak = 0.0105 = the initial radius).
- What remains with exact faces and no redistribution is a one-signed
  over-decay of -0.38 / -0.34 / -0.28 / -0.20: laneG's genuine over-decay
  (lever (a), the interface triangulation / curvature stencil pairing), not
  the faces and not the pump. Its l2 (0.326) is the worst of the four
  physics-clean arms because nothing cancels it any more.
- Projection cadence 2 inflates in 3D on both sources (every quarter
  positive, R_max(t_end) +3.9 % / +4.6 % R0, tail 0.21 / 0.15): laneH's 2D
  lever does not transfer.
- No arm has both l2 and tail at least as good as the pin. By protocol rule
  5 the droplet default stays `edge_area_source=None` (the fan cache) and
  the 3D baseline is NOT re-pinned. The l2 score of this benchmark cannot be
  used to adopt a linearly precise operator until the over-decay is
  addressed; laneG's guidance 3 stands.

A.5.b static floor (`static_droplet_floor_3D`, 2/2, 20 steps, u := 0 every
step; `laneQ_a5b.json`):

| arm | step 0 | peak | end | excess over A.5.a 6.0153201140e-05 | mass drift | s/step (excl. callback) |
|---|---|---|---|---|---|---|
| harness `run_a5b` (cache) | 6.0153201140e-05 | 7.274133897024673e-05 (= pin 7.274134e-05) | 7.274133896826632e-05 | 1.2588e-05 | 1.7e-15 | |
| `cache` | 6.0153201140e-05 | 7.2741338970e-05 | 7.2741338968e-05 | 1.2588e-05 | 1.7e-15 | 0.558 |
| `pij_simplex` | 6.0153201140e-05 | 6.2838104071e-05 | 6.2838104068e-05 | 2.685e-06 (-78.7 %) | 1.7e-15 | 0.361 |
| `pij` | 6.0153201140e-05 | 6.2838104071e-05 | 6.2838104068e-05 | 2.685e-06 | 1.7e-15 | 1.074 |
| `pij_ring` | 6.0153201140e-05 | 6.2838085762e-05 | 6.2838085760e-05 | 2.685e-06 | 1.7e-15 | 2.199 |

(laneJ measured 6.283858835e-05 for its p_ij arm on the lossy pre-laneB
mesh.) Step 0 is the untagged setup mesh (ring walk) in every arm.

## 5. The other 3D cases

### 5.1 Hydrostatic column (`hydrostatic_3D`, `dual_only_bare`; `--arm remap` = `delaunay_material` + conservative remap)

`diagnose_3d_edge_area_source.py hydrostatic --refine N --n-tac T`
(`laneQ_hydrostatic_r{N}_t{T}.json`). `dual_only_bare` builds no fan
cache, so the sub-command runs the three explicit sources (`HYDRO_ARMS` =
`pij_ring`, `pij_simplex`, `pij`; keys `preset/<arm>` and `remap/<arm>`):
`p_ij_ring` is the preset as it was before this lane (it read the ring
walk), `p_ij_simplex` the preset now. The kept `r1_t40` and `r2_t2` JSONs
were regenerated with these keys in fix round 1 (the first version of the
driver ran the preset itself as the arm `cache`, which after the flip would
have been `p_ij_simplex` twice); every number in those two files is
bit-identical between the pre-flip run, the reviewer's re-run and the
regenerated file. The `r2_t10` regeneration was still running when the fix
round was cut, so `laneQ_hydrostatic_r2_t10.json` still carries the
pre-flip keys (`preset/cache` and `remap/cache` are its `p_ij_ring` rows);
re-run the section 9 command to refresh it. The ms/step column gives the
pre-flip run and the regenerated run where one exists.

| configuration | arm | max\|u\| peak | max\|u\| end | KE end [J] | integrated L2 [Pa] | ms/step (pre-flip run / regenerated) |
|---|---|---|---|---|---|---|
| refinement 1, 40 t_ac (the `PIN_3D_*` run) | `p_ij_ring` (pre-laneQ preset) | 0.08910127097486757 | 2.423297e-04 | 6.206365156298652e-06 | 414.63 | 54.9 / 52.1 |
| | `p_ij_simplex` (preset now) | 0.08910127097486756 | 2.423297e-04 | 6.2063651562987255e-06 | 414.63 | 21.5 / 21.1 |
| | `p_ij` | 0.08910127097486757 | 2.423297e-04 | 6.206365156313909e-06 | 414.63 | 32.5 / 30.7 |
| refinement 1, 40 t_ac, remap arm | `p_ij_ring` | 0.0971664601801251 | 1.591e-03 | 1.0457479044750727e-04 | 2120.6 | 56.8 / 54.1 |
| | `p_ij_simplex` | 0.0971664601801251 | 1.476e-03 | 8.751127871519907e-05 | 2177.9 | 22.2 / 21.2 |
| | `p_ij` | 0.09716646018012509 | 1.469e-03 | 8.713063663848083e-05 | 2177.3 | 33.1 / 31.3 |
| refinement 2, 2 t_ac | `p_ij_ring` | 0.1438594356688898 | 8.620e-02 | 0.581253746837281 | 1862.2 | 549.6 / 509.3 |
| | `p_ij_simplex` | 0.1434040426640146 | 8.648e-02 | 0.5733219073980406 | 1884.8 | 159.4 / 150.2 |
| | `p_ij` | 0.1434040426640146 | 8.648e-02 | 0.5733219073980402 | 1884.8 | 279.3 / 268.1 |
| refinement 2, 2 t_ac, remap arm (the `PIN_3D_REMAP_*` run) | `p_ij_ring` | 0.15658060026054665 | 8.905e-02 | 0.5431445985762776 | 1690.1 | 551.6 / 532.1 |
| | `p_ij_simplex` | 0.15305813130485327 | 8.706e-02 | 0.528851067635385 | 1723.8 | 137.6 / 138.7 |
| | `p_ij` | 0.1530673072965945 | 8.699e-02 | 0.5290890472250437 | 1722.3 | 273.3 / 265.8 |
| refinement 2, 10 t_ac | `p_ij_ring` | 0.1438594 | 1.310e-03 | 1.393756988775e-04 | 258.8 | 544 |
| | `p_ij_simplex` | 0.1434040 | 1.287e-03 | 1.422876148906e-04 | 261.8 | 163 |
| | `p_ij` | 0.1434040 | 1.287e-03 | 1.422876148906e-04 | 261.8 | 279 |
| refinement 2, 10 t_ac, remap arm | `p_ij_ring` | 0.1565806 | 1.379e-03 | 1.314794793836e-04 | 304.0 | 542 |
| | `p_ij_simplex` | 0.1530581 | 1.416e-03 | 1.379677649929e-04 | 312.6 | 143 |
| | `p_ij` | 0.1530673 | 1.433e-03 | 1.415116878954e-04 | 311.9 | 268 |

Perturbation sensitivity of the flipped preset (protocol rule 8,
`diagnose_determinism.py sweep <case> --procs 1 --perturb 1e-15 --n-perturb 8`,
`cases_dynamic/Hydrostatic_column/results/laneQ/sweep_*.json`), largest
relative shift over 8 seeds against the plain run; laneT's values on the
ring walk in brackets:

| case | steps | max\|u\| peak | max\|u\| tail | KE end |
|---|---|---|---|---|
| `pin_hydro3d` (refinement 1, 40 t_ac) | 370 | 1.7e-13 (1.7e-13) | 7.6e-12 | 6.1e-12 (6.1e-12) |
| `hydro3d` (refinement 2) | 150 | 8.6e-14 | 2.7e-12 | **9.7e-13 (2e-06)** |
| `pin_hydro3d_remap` (refinement 2, 2 t_ac) | 37 | 3.0e-03 (1.25e-03) | 2.4e-03 | 6.4e-04 (6.9e-04) |

The fixed-connectivity refinement 2 column is no longer decided by the
hull-edge tie (2e-06 -> 1e-12, the refinement 1 level); the remap arm keeps
its Delaunay-tie sensitivity (the cospherical lattice reconnects differently
under a 1e-15 shift), so its two pins remain locks on the exact arithmetic
at rel 1e-6, as laneT pinned them.

Re-pins (`ddgclib/tests/test_case_hydrostatic.py`), old -> new:

| pin | old (ring walk) | new (`p_ij_simplex`) | move |
|---|---|---|---|
| `PIN_3D_UMAX_PEAK` | 0.08910127097486757 | 0.08910127097486756 | 1.1e-16 (round-off) |
| `PIN_3D_KE_40` | 6.206365156298652e-06 | 6.2063651562987255e-06 | 1.2e-11 (round-off amplified; the run's own 1e-15 sensitivity is 6.1e-12) |
| `PIN_3D_REMAP_UMAX_PEAK` | 0.15658060026054665 | 0.15305813130485327 | -2.2 % (beyond the 1.25e-3 perturbation range: the removed 37 % hull-edge areas) |
| `PIN_3D_REMAP_KE_END` | 0.5431445985762776 | 0.528851067635385 | -2.6 % (same) |

`PRESETS['hydrostatic_3D'].replace(edge_area_source='p_ij_ring')` reproduces
every old value to the bit (the `p_ij_ring` rows above).

### 5.2 Hagen-Poiseuille 3D, `pressure_flux='centred'` (reads the faces)

`diagnose_poiseuille.py arms3d --only centred --tag laneQprobe` (refinement
1, L 3, mu 0.1, 600 steps; `results/laneH/arms3d_laneQprobe.json`). The
preset itself (both fluxes `simplex_gradient`) reads no dual face and is
unchanged (l2 0.0566, radial velocity 2.6e-18, laneH).

| arm | l2 end | tail mean / max | u_max (0.2) | radial velocity | wall |
|---|---|---|---|---|---|
| fan cache (laneH / laneT value) | 0.08211494566330206 | 8.872e-02 / 1.657e-01 | 0.20434 | 6.3e-03 | 58 s |
| `p_ij_ring` (= laneH's custom ring wrapper) | 0.05656136676066496 | 5.946e-02 / 6.739e-02 | 0.19858 | 1.26e-05 | 152 s |
| `p_ij_simplex` | 0.05665809341912639 | 5.968e-02 / 6.767e-02 | 0.19856 | **2.5e-17** | 44 s |
| `p_ij` | 0.05661195172315967 | 5.955e-02 / 6.749e-02 | 0.19855 | 4.8e-17 | 83 s |

With the exact faces the centred flux gives the result of the volume form
(`simplex_gradient`): no radial drift, l2 0.0567. The arm is chaotic
(laneT: a 1e-15 shift moves the cache arm between 0.079 and 0.169), so the
two exact arms differing in the 4th digit is round-off amplified, not a
method difference.

### 5.3 Dam break 3D (smoke, `dam_break_3D`)

`diagnose_3d_edge_area_source.py dambreak --n-steps N` (`laneQ_dambreak.json`,
`laneQ_dambreak_pij_simplex_793.json`; dt 2.524e-4, the full horizon is 793
steps). The shipped preset (fan cache) aborts with NaN after 17 steps
(KE_liq 5.1e+46), on the HEAD library too (`git archive` copy of both
repositories, same 17 steps and KE); `p_ij_simplex` runs 100 steps (KE_liq
1.401e-02 J, |u|max 3.46 m/s, mass drift 3.5e-15) and aborts after 156 of
793; `p_ij` aborts after 91. The case blows up on every source (laneF's
air-sliver ejection on the laneS setup); not re-pinned, preset unchanged.

### 5.4 Not measured

`electrolysis_bubble_3D` (unstable, gas phase lost; not in the brief) and
the periodic 3D shearing plate (the periodic retopology does not forward
the axis; `SolverMethods` raises).

## 6. Measured DO-NOTs

- Do not orient a flat tetrahedron's dual face piece by its geometry
  (`sign(det)` is 0 and the piece is perpendicular to the edge): on the
  droplet mesh 44 of 2540 tetrahedra are exactly flat and the cells at
  their vertices close only with the combinatorial orientation.
- Do not drop flat tetrahedra from a dual face sum (the laneO reference
  `simplex_area_vectors` does): the cell opens by the flat piece.
- Do not flip the 3D droplet default to the exact faces on the l2 score,
  with or without the redistribution lever or the projection cadence:
  every arm fails the flip rule (section 4). Do not read the exact-face l2
  0.28653 as a regression of physics: mass, dual-volume drift, final
  inflation and the early bump all improve.
- Do not apply `projection_every=2` in 3D: it inflates on both sources.
- Do not run `edge_area_source='p_ij'` or `'p_ij_ring'` for cost reasons:
  the per-edge Python constructions cost 8.6x and 26x the force evaluation
  of a cached source.
- Do not expect `backend` to do anything under `p_ij_simplex` (the wrapper
  raises).
- Do not read a dam-break 3D smoke run as evidence for any source: the case
  aborts on all of them.

## 7. Known limits

- `p_ij_simplex` runs no fan walk, so the fan-failure promotion of
  `batch_e_star` (a vertex whose dual fan walk fails is tagged boundary)
  does not happen under it; 0 occurrences were ever measured on the droplet
  and the jittered box (laneL), none in this lane's runs.
- An inverted tetrahedron (frozen connectivity after a vertex crossed a
  face) is oriented by the combinatorial propagation, i.e. with the sign
  its neighbours induce: the cell still closes, the tensor identity counts
  its volume with a negative sign while `simplex_dual_volumes` uses |T|.
  No run of this lane produced one.
- `frozen`, `custom` and `periodic` connectivities do not apply the axis
  (`SolverMethods` raises); a mesh no retopology has tagged (the setup mesh,
  step 0 of every run) reads the legacy ring walk, so every step-0 number
  is unchanged.
- The hydrostatic remap arm is still tie-decided (Delaunay flips of the
  cospherical lattice, 3e-3 under a 1e-15 shift); the exact faces fix the
  area tie only.
- The 2D sources are not selectable (reported only); 2D needs no kernel (the
  segment is exact since laneO).
- The oscillating-droplet 3D over-decay (laneG lever (a)) remains the
  reason the exact faces cannot be adopted on the l2 score.

## 8. Battery

- ddgclib fast: 1225 passed, 12 skipped, 2 xfailed, 0 failed in 131 s (fix
  round 1 run, after every change of this log; before the lane 1197 passed,
  12 skipped, 3 xfailed: +26 new tests, the laneT strict xfail converted
  into 2 passing tests). The review had found 2 failures in the first
  version (a stale status expectation in the new test and a METHODS.md one
  edit behind the registry), both fixed at the cause (section 10).
- ddgclib slow: 32 passed, 1 xfailed in 240 s (32 passed, 1 xfailed before;
  the four hydrostatic 3D pins re-pinned), confirmed by the reviewer's own
  run (32 passed, 1 xfailed in 244 s) on the same pinned values. The fix
  round's re-run was at 20 of 33 tests with no failure when the round was
  cut; nothing numeric changed in the round (an error text, a group label,
  a test expectation and docs), so the slow pins are unaffected (unverified
  by a complete run of this round).
- hyperct (`pytest hyperct/tests -k "not benchmark"`): 340 passed, 38
  skipped, 6 xfailed in 8 s (326 before, +14; the 14th is the lattice hull
  test of fix round 1).

## 9. How to reproduce

```
PY=/home/endres/anaconda3/envs/ddg/bin/python
cd /home/endres/projects/ddgclib
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py static
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py a5b
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dynamic --arm cache          # also pij_simplex, pij, pij_ring, cache_noredis, pij_simplex_noredis, cache_p2, pij_simplex_p2
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py hydrostatic --refine 1 --n-tac 40
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py hydrostatic --refine 2 --n-tac 2
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py hydrostatic --refine 2 --n-tac 10
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dambreak --n-steps 100
$PY cases_dynamic/oscillating_droplet/diagnose_3d_edge_area_source.py dambreak --arm pij_simplex --n-steps 793
$PY cases_dynamic/Hagen_Poiseuile/diagnose_poiseuille.py arms3d --only centred --tag laneQprobe
$PY cases_dynamic/diagnose_determinism.py sweep pin_hydro3d --procs 1 --perturb 1e-15 --n-perturb 8
$PY cases_dynamic/diagnose_determinism.py sweep hydro3d --procs 1 --perturb 1e-15 --n-perturb 8
$PY cases_dynamic/diagnose_determinism.py sweep pin_hydro3d_remap --procs 1 --perturb 1e-15 --n-perturb 8
$PY -m pytest ddgclib/tests/test_edge_area_source.py ddgclib/tests/test_determinism.py -q -m "not slow"
cd /home/endres/projects/hyperct && $PY -m pytest hyperct/tests/test_dual_face_areas.py -q
```

Every droplet arm writes `score_laneQ_<arm>.json` (with the `methods` block),
`methods_laneQ_<arm>.json` and `diags_laneQ_<arm>.json` into
`results_3d/` (`--out DIR` elsewhere; the arms of this log ran under
`--out <scratch>` with `OMP_NUM_THREADS=1`, eight concurrently, and were
copied into `results_3d/`). Timing numbers come from the same concurrent
runs, so the per-step ratios are the comparable figures: the reviewer's
single re-runs gave 0.537 s/step for the `cache` arm (0.554 here) and
0.3376 for `pij_simplex` (0.338), hydrostatic refinement 1 51 against 21
ms/step (55 against 22 here). Absolute times are indicative only; every
number that is not a time is bit-identical between the runs.

## 10. Fix round 1 (2026-10-05, after independent review)

Blocking findings, resolved at the cause:

- `test_edge_area_source.py::test_integrator_kwargs_carry_the_field`
  asserted the status `experimental` for `p_ij_simplex`, which the registry
  had promoted to `validated` after the test was written: the expectation
  now reads the registry's values (`validated` for `p_ij_simplex`, `opt-in`
  for `p_ij`); no registry status was changed.
- `METHODS.md` regenerated from the registry (the `p_ij` evidence row was
  one edit behind); `test_methods_md_matches_registry` passes.
- The `pij_ring` droplet arm finished after the review (872 steps, 2.219
  s/step): the row of section 4 is filled from its score JSON and its three
  JSONs are in `results_3d/` (l2 0.28712884337228660, tail
  0.08839011881475957, R_max(t_end) 0.010082, KE_max 1.7050e-06, quarters
  +0.279 / +0.093 / -0.258 / -0.347). It changes no decision (the reference
  arm sits within 0.2 % of the two exact arms).
- The fast-suite line of section 8 quotes the green run of this round.

Non-blocking findings, addressed:

- The over-stated equality with laneT's per-tetrahedron quads is qualified
  in the verdict, section 2, `_axes.py` and `debugging_plan.md` (equal on
  every link without a flat tetrahedron; undefined reference on the 179
  hull-hull links with one; re-measured, section 2), and the hull half cell
  beside flat hull tetrahedra is a hyperct test now
  (`test_hull_half_cell_closes_beside_flat_hull_tetrahedra`, lattice r2: 13
  flat tetrahedra, 36 hull vertices beside one, closure 1.2e-16 against the
  box-face normals; hyperct 340 passed).
- `diagnose_3d_edge_area_source.py hydrostatic` runs the three explicit
  sources (`HYDRO_ARMS`: `pij_ring`, `pij_simplex`, `pij`) instead of the
  preset-as-`cache` arm, so the section 9 command reproduces the section
  5.1 table after the flip; the kept `laneQ_hydrostatic_r1_t40.json` and
  `r2_t2.json` were regenerated with the keys `preset/<arm>` and
  `remap/<arm>` (every value bit-identical to the pre-flip files and the
  pins, section 5.1; the `r2_t10` file was still regenerating at the cut
  and keeps the pre-flip keys). The
  `cache` arm keeps its name where it is the fan cache (droplet, dam break)
  and the module docstring says so.
- The axis group reads `dual geometry` (it held an explicit axis under the
  label `(reported)`); the `SolverMethods` error for a connectivity that
  does not apply the axis names `adaptive` (2D only) as well.
- Timings: section 5.1 carries the ms/step of the pre-flip and the
  regenerated run side by side; section 9 quotes the reviewer's single
  re-runs.

Not changed on review: the rejection of `connectivity='adaptive'` with a
non-None `edge_area_source` stays (the axis is 3D and `adaptive` is 2D
only, so the combination cannot run; the error text says so now).
