# Audit: retopology edge-case bugs (skip_triangulation+filtered bV; <dim+1 early return; boundary dual_vol=0 / DualVolumeMass / mass-relaxation BCs)
> Sources checked | Written 2026-07-02 by physics-audit workflow

Sources read: `docs_temp/02_physics_foundations.md` (dual-cell tiling §, boundary-flux caveat L148, conservation caveats L138), `docs_temp/04_solver_pipeline.md` §3 (retopo steps, freshness table), `docs_temp/code_map/integrators_and_bcs.md` (line-anchored `_retopologize` table, BC table, IC table).
Code read: `ddgclib/dynamic_integrators/_integrators_dynamic.py:48-247` (full `_retopologize`), `:289-388` (`_do_retopologize`), `:391-475` (`_retopologize_multiphase`), `ddgclib/initial_conditions.py:300-326` (`DualVolumeMass`), `ddgclib/_boundary_conditions.py:839-1030` (mass-relaxation BCs), `ddgclib/operators/stress.py:52-230, 311-408, 634-675`, `hyperct/ddg/_compute_dual.py:145-300, 303-400`, `hyperct/ddg/_operators.py:358-420`.
Probes: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/retopo-edge-cases/probe_{a_skip_tri_filtered_bv,a2_which_path,a3_3d,a4_spurious_force,b_early_return,c_dualvolmass,d_dambreak_path}.py`, run with `/home/endres/anaconda3/envs/ddg/bin/python`.

---

## Key load-bearing lines (exact quotes)

`ddgclib/dynamic_integrators/_integrators_dynamic.py`:

```python
135    verts = list(HC.V)
136    if len(verts) < dim + 1:
137        return  # not enough vertices for a simplex
...
152            if len(verts) < dim + 1:      # second early return, after merge_all
153                return
...
199    else:
200        # skip_triangulation: keep existing connectivity,
201        # use current bV as the boundary set
202        dV = set(bV)
203
204    # 4. Tag v.boundary on ALL topological boundary vertices.
205    #    compute_vd needs the full boundary to build correct half-cells.
206    for v in HC.V:
207        v.boundary = v in dV
...
224        for v in HC.V:
225            v.dual_vol = vols.get(id(v), 0.0) if v not in dV else 0.0   # boundary zeroing
226        HC._edge_area_cache = edge_areas
227    except (ImportError, NotImplementedError):
228        from ddgclib.operators.stress import cache_dual_volumes
229        cache_dual_volumes(HC, dim)        # fallback: boundary verts get REAL half-volumes
...
235    if boundary_filter is not None:
236        dV = {v for v in dV if boundary_filter(v)}
237    bV.clear()
238    bV.update(dV)                          # bV wholly rewritten to the FILTERED set
```

`ddgclib/initial_conditions.py:320-326` (`DualVolumeMass.apply`):

```python
320    def apply(self, HC, bV: set) -> None:
321        for v in HC.V:
322            dual_vol = getattr(v, 'dual_vol', None)
323            if dual_vol is None or dual_vol < 1e-30:
324                v.m = self.rho * 1e-30
325            else:
326                v.m = self.rho * dual_vol
```

`ddgclib/_boundary_conditions.py:846-848` (same guard at `:911`, `:998`, `:1020`):

```python
846            vol = getattr(v, 'dual_vol', 0.0)
847            if vol < 1e-30:
848                continue
```

Dimensional split that governs everything below — `hyperct/ddg/_operators.py:403-404`:

```python
403    if dim != 3:
404        raise NotImplementedError("batch_e_star only supports dim=3")
```

So **in 2D the `except NotImplementedError` fallback (`cache_dual_volumes`, half-cell path) ALWAYS runs** and boundary vertices get real half-cell volumes; **only in 3D does the `batch_e_star` path run** and zero boundary `dual_vol` (`:225`). Confirmed by probe A2: `batch_e_star import: OK` yet `after baseline: HC._edge_area_cache is None? True` on a 2D mesh.

---

## Claim (a): `skip_triangulation=True` with a filtered `bV` mis-tags open-boundary vertices as interior

**What the physics requires.** Every dual cell used by the FVM operators must be a closed control volume (dual-face closure `sum_j A_ij = 0`, validated to 1e-12 in `test_stress.py`; 02_physics_foundations.md L122). Vertices on the topological boundary cannot have closed cells; the code's own contract (`:204-205`) is that ALL topological boundary vertices are flagged before `compute_vd`.

**What the code does.** With `boundary_filter`, step 6 rewrites `bV` to the filtered subset (`:235-238`). On the next step with `skip_triangulation=True`, `dV = set(bV)` (`:202`) is that filtered subset, so step 4 (`:206-207`) sets `v.boundary = False` on every open-boundary (non-wall) vertex. The narrowing is persistent (the filtered `bV` feeds every subsequent skip step).

**Probe A (2D, `probe_a_skip_tri_filtered_bv.py`, 145-vertex unit square, walls = y-faces, open = x-faces).** Output:

```
[step1 delaunay + boundary_filter=walls] n_verts=145 |bV|=18 n(v.boundary=True)=32
[step2 skip_triangulation=True (dV = filtered bV)] n_verts=145 |bV|=18 n(v.boundary=True)=18
[step2] side verts: n=14, boundary-flagged=0, dual_vol min/max = 3.906250e-03/3.906250e-03
[step2] sum(dual_vol) over ALL verts = 0.992187500  (domain area = 1.0)
volume inflation vs baseline at step2: +0.000000e+00 (+0.000%)
```

The mis-tagging happens (14/14 open-boundary verts flip to `boundary=False`) but is **numerically inert in 2D**: `_compute_vd_2d_simplex_aware` (`hyperct/ddg/_compute_dual.py:228-300`) determines boundary edges topologically (1 vs 2 adjacent simplices in `HC._simplices`) and never reads `v.boundary`; the 2D `dual_area_vector` (`stress.py:158-166`) uses `v_i.vd ∩ v_j.vd`, also flag-free; and the dual-volume cache is the flag-free `cache_dual_volumes` fallback. Dual volumes identical to machine precision. (The flag WOULD matter on the legacy nn-intersection 2D path at `_compute_dual.py:170`, which only runs when `HC._simplices` is empty — e.g. after `adaptive_remesh` + `invalidate_simplex_cache` followed by a skip step.)

**Probe A3 (3D, `probe_a3_3d.py`, 189-vertex unit box, refinement 2, walls = z-faces, open = x/y-faces; per-step call replicates `symplectic_euler` forwarding both kwargs).** Output:

```
[baseline full-boundary]           open-face verts: n=48 flagged=48 dual_vol==0: 48
[baseline full-boundary]           sum(dual_vol) ALL = 0.640625000 (domain vol = 1.0)
[step2 skip_tri+filter]            open-face verts: n=48 flagged=2  dual_vol==0: 2   min/max=0.0000e+00/5.4253e-03
[step2 skip_tri+filter]            sum(dual_vol) ALL = 0.830064562
step2 vs baseline: +1.894396e-01 (+29.571%)
  open-face v (0.0, 0.0, 0.5): boundary=False dual_vol=0.003078884548611111
```

In 3D the flag flip is **live**: 46 of 48 open-boundary vertices are passed to `batch_e_star` as interior (only 2 fan-walks fail and self-heal via the `failed` promotion at `:221-223`), acquire **fake positive dual volumes from non-closed fans** (total dual volume inflates +29.6%), and — since an EOS `pressure_model` computes `rho = m / dual_vol` (`stress.py:669-671`) — the `vol < 1e-30` reference-pressure guard (`stress.py:667-668`) that previously protected these vertices no longer fires: a bogus density → bogus pressure is written to `v.p` and read by every interior neighbour's pressure flux.

**Probe A4 (`probe_a4_spurious_force.py`, uniform p=1000, mu=0):**

```
[step1 delaunay+filter] interior: max|F|=0.000e+00
[step1 delaunay+filter] open-face verts (flagged bnd): max|F|=8.168e+01
[step2 skip+filter]     open-face verts (now 'interior'): max|F|=8.333e+01
open-face verts in bV (frozen): 0/48 -> the rest are advected by the spurious force
```

Note the spurious ~O(80) force on open-boundary vertices exists even on the *intended* path (step1) — that is the separately documented "no boundary-flux closure" limitation (02 L148). What claim (a) adds is the dual-volume/EOS corruption and the loss of the `dual_vol=0` pressure pinning.

**Verdict (a): CONFIRMED_BUG** (3D live; 2D mechanism present but inert on the current simplex-aware path).

**Production reachability:** no shipped case hits it today. The path needs `skip_triangulation=True` together with `boundary_filter` (or a hand-narrowed `bV`) and `retopologize_fn=None`:
- `Hagen_Poiseuile_2D.py:281` uses `boundary_filter=wall_criterion` with full Delaunay (boundary re-derived topologically each step → no narrowing propagates), and is 2D anyway.
- `dam_break_2D.py:119` / `dam_break_3D.py:112` pass `skip_triangulation=True` — but **it is silently dropped**: they also pass a custom `retopologize_fn` (`src/_setup.py:188` binds only `mps`/`redistribute_mass`), and `_do_retopologize:360-374` forwards only `remesh_mode`/`remesh_kwargs` to callables (documented at `:316-319` "Ignored when retopologize_fn is a callable"). Probe D output: `kwargs received by custom retopologize_fn: ['remesh_kwargs', 'remesh_mode']; skip_triangulation forwarded? False; inner _retopologize called with skip_triangulation = False`. So dam_break actually runs **full Delaunay every step**, contradicting its own comment at `dam_break_2D.py:111-114` — a case-level bug worth fixing in its own right (it re-exposes dam_break to the cross-phase-edge instability the comment claims to avoid).
- `dam_break_*_no_air.py` use `boundary_filter` only (full Delaunay); oscillating_droplet and electrolysis_bubble use `_retopologize_multiphase` partials with neither `boundary_filter` nor (forwarded) `skip_triangulation`; `static_droplet_2D.py:114-123`'s `_dual_only_retopo` re-derives the full topological boundary via `HC.boundary()` each call, so no narrowing.

## Claim (b): early return with `< dim+1` vertices leaves duals stale

**What the code does.** `:136-137` (and the post-merge duplicate at `:152-153`) return before boundary tagging, `compute_vd`, and volume caching — while the docstring (`:55-58`) promises "All vertices have valid dual cells". No warning, no flag.

**Probe B (`probe_b_early_return.py`).** Output:

```
3 verts: dual_vols = {'(0.0, 0.0)': 0.041667, '(1.0, 0.0)': 0.041667, '(0.5, 1.0)': 0.041667}
after removal: n_verts = 2
retopologize returned: None
dual_vol untouched (stale marker survives): True
v.vd untouched: True
merge path: n_verts after merge = 2, returned=None, stale marker survives: True
euler ran 2 steps on the 2-vertex stale-dual mesh: no exception, no warning
post-euler dual_vols: [777.0, 777.0]
```

Both early returns leave a poison marker (`dual_vol=777.0`) and stale `v.vd` untouched, and `euler` then integrates on that stale geometry silently.

**Verdict (b): CONFIRMED_BUG, but severity low** — it deviates from the function's documented contract and is silent, yet it is only reachable when the whole mesh has ≤ dim vertices (2 in 2D, 3 in 3D). None of oscillating_droplet / dam_break / electrolysis_bubble can reach it short of total mesh annihilation (they run 10²–10⁴ vertices with no outlet deletion in the closed-box cases); `OutletDeleteBC` pipelines (Hagen-Poiseuille) could in principle drain to that state only after the simulation is already physically meaningless. A one-line `warnings.warn` (or raise) would close the gap.

## Claim (c): boundary `dual_vol=0` → `DualVolumeMass` gives `m = rho*1e-30`; mass-relaxation BCs skip those vertices

**What the code does.** In 3D (batch path) all boundary vertices get `dual_vol = 0.0` (`:225`). `DualVolumeMass.apply` maps that to `m = rho*1e-30` (`initial_conditions.py:323-324`). All three mass-relaxation BCs (`PressureReservoirBC:847`, `AbsorbingPressureBC:911/:921`, `ExpandingDomainBC:998/:1020`) silently `continue` on `vol < 1e-30`.

**Probe C (`probe_c_dualvolmass.py`, 3D unit box refinement 2, rho=1000).** Output:

```
=== 3D (batch_e_star path) ===
setup path (cache_dual_volumes): boundary dual_vol min=2.1159e-03 max=7.4870e-03 n_zero=0
DualVolumeMass at setup: boundary m min=2.1159e+00 max=7.4870e+00; total M=937.500000
after _retopologize: boundary dual_vol n_zero=98/98
DualVolumeMass re-applied after retopo: boundary m min=1.0000e-27 max=1.0000e-27
  boundary verts with m == rho*1e-30 = 1e-27: 98/98
  total M after=640.625000 vs setup 937.500000 (-31.67%)
PressureReservoirBC on 98 boundary verts: acted on 0
PressureReservoirBC on 91 interior verts: acted on 91
=== 2D (fallback cache_dual_volumes inside _retopologize) ===
2D after _retopologize: boundary dual_vol min=6.5104e-04 max=3.9063e-03 n_zero=0/32
2D DualVolumeMass after retopo: boundary m min=6.5104e-01 (no 1e-27 masses: True)
```

So: mechanism fully reproduced in 3D (98/98 boundary vertices annihilated to m=1e-27, -31.7% of total mass; `PressureReservoirBC` acts on **0** of 98 boundary targets, silently), and **cannot occur in 2D** (fallback half-cell path, no zeros).

**Verdict (c): DESIGN_LIMITATION (documented), not currently hit in production.**
- `DualVolumeMass` docstring `initial_conditions.py:308-309` explicitly requires "`compute_vd` and `cache_dual_volumes` to have been called before applying this IC", and every shipped user (`Hydrostatic_column/src/_setup.py:318-329`, `Hydrostatic_2D/3D/1D/2D_periodic`) applies it at setup immediately after `compute_vd` + `cache_dual_volumes` — the correct half-cell path. No case applies it after an integrator retopo.
- The mass-relaxation BCs are used only by `cases_dynamic/cube2droplet/*` (2D → fallback path → no zero volumes → guard never trips) and their code-map entry already documents the trap ("these BCs only act on vertices that are interior per the current dual build").
- The zeroing itself is catalogued in 02_physics_foundations.md L138 as a known artefact ("one-shot |dV/V0|≈0.3 after the first 3D retopo is a boundary-shell dual_vol zeroing artefact — never assert 3D volume conservation from step 0") — my probe's baseline (interior sum 0.6406 for a coarse box; boundary shell 36% of volume) quantifies exactly that.
- Residual physics consequence for 3D production cases (dam_break_3D, oscillating_droplet_3D, electrolysis_bubble_3D): wall vertices with `dual_vol=0` have their EOS pressure pinned to the reference `eos.pressure(rho0)` (`stress.py:667-668`), so wall-adjacent pressure fluxes use a reference `p_j` instead of the local hydrostatic/compressed value. This is a discretization approximation at frozen walls, consistent with the documented "no boundary-flux closure" limitation, not a new defect.

---

## Impact on the named production cases

| Case | (a) skip+filter mis-tag | (b) <dim+1 early return | (c) dual_vol=0 / DualVolumeMass / BC guard |
|---|---|---|---|
| oscillating droplet (2D main) | not hit (no `boundary_filter`, no `skip_triangulation`; 2D inert anyway) | not reachable (~10³ verts) | not hit (2D → fallback half-cells) |
| oscillating droplet (3D) | not hit (no filter) | not reachable | outer-box wall verts get `dual_vol=0` each retopo → reference-pinned wall pressure (documented artefact); interface verts interior, unaffected |
| dam break 2D/3D | not hit — **but `skip_triangulation=True` is silently dropped** (`_do_retopologize:319`; probe D) so full Delaunay runs every step, contradicting `dam_break_2D.py:111-114` | not reachable | 3D setup (`src/_setup.py:168-169`: `_retopologize` then `mps.refresh(reset_mass=True)`) gives zero-mass frozen wall verts; documented artefact |
| electrolysis bubble 2D/3D | not hit (multiphase retopo partial, no filter/skip) | not reachable | same 3D wall artefact; its `dudt` already guards `v.m < 1e-30` (`src/_setup.py:465-467`) |

## Suggested fixes

1. **(a)** In the `skip_triangulation` branch, derive the boundary from topology instead of the (possibly filtered) `bV`: `dV = boundary_from_simplices(HC, dim) if getattr(HC, '_simplices', None) else HC.boundary()` — connectivity is unchanged in this branch, so the cached simplices are exactly valid, and the cost is small. Alternatively keep a separate `HC._topological_boundary` set from the last full pass and use that, never `bV`, at `:202`.
2. **(dam_break wiring)** Either forward `skip_triangulation` to custom callables that accept it (same inspect pattern as `remesh_mode`, `_do_retopologize:360-374` — `_retopologize_multiphase` already has the parameter at `:393`), or fix the two case files to bind it in the partial: `partial(_retopologize_multiphase, mps=mps, skip_triangulation=True, ...)`.
3. **(b)** Replace the bare `return` at `:137` and `:153` with `warnings.warn("retopologize skipped: %d vertices < dim+1; duals are stale" % len(verts))` (or raise for the merge case, which indicates `merge_cdist` ate the mesh).
4. **(c)** Make `DualVolumeMass.apply` fail loudly (raise or warn) when a nonzero fraction of vertices hits the `1e-30` fallback, and/or have `_retopologize` store the half-cell volume for boundary vertices in a separate attribute (`v.dual_vol_half`) that the mass-relaxation BCs read — keeps the FVM invariant (`dual_vol=0` ⇒ no closed cell) while letting the reservoir BCs act at walls.
