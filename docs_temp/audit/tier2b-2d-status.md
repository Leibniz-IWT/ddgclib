# Audit: Tier 2B step 1 TRUE status — is `surface_tension_force_2d` wired for dim=2?
> Sources checked | Written 2026-07-02 by physics-audit workflow

**Item key:** `tier2b-2d-status`
**Verdict:** CORRECT_AS_INTENDED (code) — the exact 2D integrated FTC surface-tension operator IS the production path for dim=2. The debugging plan's Tier 2B step 1 "rewrite" recommendation (2026-05-28 entry) is **stale/moot**: the rewrite it asks for has been in place since the first multiphase commit (2026-04-07). The remaining 2.27e-3 floor is O(h) polygon-discretization error, not a wiring defect.

## 1. What the physics requires

`docs_temp/02_physics_foundations.md:65-69` (§3.4): the surface-tension force on an interface vertex is
F_γ,i = ∫_{Γ_i} γ κ N dS, and for the 2D multiphase default (`curvature_path='integrated'`) the intended
discrete form is the exact FTC identity on piecewise-linear curves:
∫ κ N ds = t_next − t_prev, i.e. `F_st = γ (t_next − t_prev)` via
`surface_tension_force_2d` / `integrated_curvature_normal_2d` (`ddgclib/operators/curvature_2d.py:91-181`).
The code map (`docs_temp/code_map/multiphase_surface_tension.md:101-104`) documents the same dispatch.

The suspicion to resolve (from `debugging_plan.md:84`, entry 2026-05-28):

> "The 2D residual is now provably 100% pointwise-curvature truncation: `hndA_i_interface` on a polygon
> approximation of the circle has O(h) error. Replace it with the existing `surface_tension_force_2d`
> ... verified called from `_interface_surface_tension` for the 2D dim path"

i.e. the plan implies 2D might still route through the pointwise/cotangent `hndA_i_interface`.

## 2. What the code does (static reading)

`ddgclib/operators/multiphase_stress.py`, `_interface_surface_tension` (lines 196-278):

- Line 66: module-top import `from ddgclib.operators.curvature_2d import surface_tension_force_2d`.
- `curvature_path='integrated'` (production default, threaded from `multiphase_stress_force:113` and
  `multiphase_stress_acceleration:351`):
  - `dim == 3` → `hndA_i_interface` cotangent-Heron (lines 271-274).
  - **`else` (dim=2) → `F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)` (lines 276-278).**
- `curvature_path='stokes'`, dim=2 → same FTC delegate (lines 261-263).
- `curvature_path='csf_dual'` → `_csf_dual_surface_tension`, which in 2D takes its magnitude from
  `integrated_curvature_normal_2d` (lines 306-311) — still the FTC quantity, redirected along S_inner.

**There is no dim=2 branch anywhere in `_interface_surface_tension` that calls `hndA_i_interface`.**

Git history: the dim=2 FTC wiring is not recent. `git show 1690ce0:ddgclib/operators/multiphase_stress.py`
(commit 1690ce0, "ENH: Multiphase simulation infrastructure with capillary rise case", **2026-04-07** —
the first commit containing this file) already reads:

```python
    if dim == 3:
        from ddgclib._curvatures_heron import hndA_i_interface
        ...
    else:
        F = np.zeros(dim)
        F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)
```

So the 2D path has used the exact FTC operator for its entire existence; it never went through the
cotangent path.

## 3. Probe design

Scripts in `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/tier2b-2d-status/probe_dispatch.py`, run with `/home/endres/anaconda3/envs/ddg/bin/python`, cwd `/home/endres/projects/ddgclib`. No source edits — counting wrappers monkeypatched onto:

- `mstress.surface_tension_force_2d` (the **bound name** the dispatch actually uses, since the import is
  module-top at multiphase_stress.py:66 — patching only `curvature_2d.surface_tension_force_2d` would miss it),
- `curvature_2d.integrated_curvature_normal_2d`,
- `_curvatures_heron.hndA_i_interface` and `integrated_hndA_i_interface` (function-local imports at
  multiphase_stress.py:272/256 resolve at call time, so module-attribute patching intercepts them),
- `mstress._csf_dual_surface_tension`, `mstress._interface_surface_tension`.

Mesh: identical to the A.5 harness (`setup_oscillating_droplet`, dim=2, refinement_outer=3,
refinement_droplet=3 → 311 vertices, 32 interface). Then: (1) full `multiphase_stress_force` sweep over all
interface vertices per curvature_path; (2) per-vertex value check dispatch output vs independently
recomputed `γ(t_next − t_prev)`; (3) A.5.a refinement sweep 32/64/128 interface vertices.

Plus the exact harness command from the task:
`/home/endres/anaconda3/envs/ddg/bin/python cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py --skip-3d --n-steps 50 --redistribute-mass`.

## 4. Probe OUTPUT

### 4a. Harness run (production defaults, `curvature_path='integrated'`)

```
A.5.a (2D) — frozen mesh:   Interface max |F| = 2.3749e-03   (×0.625 of baseline 3.8e-3)
A.5.b (2D) — 50 steps, retopo ON, u=0, redistribute_mass=True:
  t=0:      max|F| = 2.3749e-03
  step 1:   max|F| = 2.2717e-03
  steps 5..50: max|F| = 2.2717e-03 (bit-identical at every printed step)
  |dM/M0| = 2.555e-15   |dV/V0| = 3.542e-16   vertices 311->311, interface 32->32
```

**Both floors match the pinned regression values exactly** (pinned: A.5.a 2.3748568012e-3, A.5.b
post-retopo 2.2716937802e-3 — `debugging_plan.md:47-49`; the probe's full-precision A.5.a value below is
2.3748568012e-03, bit-identical to the pin).

### 4b. Dispatch instrumentation (dim=2, 32 interface vertices, one full sweep each)

```
curvature_path='integrated'  (PRODUCTION DEFAULT)   max|F| = 2.3748568012e-03
     0  _curvatures_heron.hndA_i_interface            [pointwise/cotangent 3D path]
     0  _curvatures_heron.integrated_hndA_i_interface [3D 'stokes' path]
    32  curvature_2d.integrated_curvature_normal_2d   [FTC tangent-difference]
     0  multiphase_stress._csf_dual_surface_tension
    32  multiphase_stress._interface_surface_tension  [dispatch entry]
    32  multiphase_stress.surface_tension_force_2d    [BOUND NAME USED BY DISPATCH]

Per-vertex value check: max |F_dispatch − γ(t_next − t_prev)| over 32 vertices = 0.000e+00

curvature_path='stokes'   : max|F| = 2.3748568012e-03, same counters (32 FTC, 0 heron) — 2D delegates to FTC
curvature_path='csf_dual' : max|F| = 2.3748568012e-03, 32 csf_dual + 32 integrated_curvature_normal_2d, 0 heron
```

Every one of the 32 interface vertices routes through `surface_tension_force_2d`; `hndA_i_interface` is
called **zero** times for dim=2 on any path, and the dispatched force is bitwise equal to the FTC identity.

### 4c. Refinement sweep — nature of the residual floor

```
refinement_droplet=3: n_iface= 32  max|F|=2.374857e-03  ftc_calls=32   heron_calls=0
refinement_droplet=4: n_iface= 64  max|F|=1.110493e-03  ftc_calls=64   heron_calls=0
refinement_droplet=5: n_iface=128  max|F|=5.375644e-04  ftc_calls=128  heron_calls=0
observed order: 32->64 p=1.10, 64->128 p=1.05
```

This reproduces, to all quoted digits, the numbers in the 2026-05-06 audit note embedded in the harness CLI
help (`cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py:411-416`: "32→2.37e-3, 64→1.11e-3,
128→5.4e-4"). The floor is clean **first-order (O(h)) polygon-discretization error**, present even though
the surface-tension operator itself is exact on the piecewise-linear curve.

## 5. Verdict and reasoning

**CORRECT_AS_INTENDED.** The suspicion that "2D still routes through the pointwise/cotangent
`hndA_i_interface`" is refuted by static reading (multiphase_stress.py:276-278), by call-count
instrumentation (32/32 FTC calls, 0 heron calls), by a bitwise value check, and by git archaeology (the
wiring dates to the first multiphase commit 1690ce0, 2026-04-07).

**The debugging plan is internally inconsistent about this and its Tier 2B step 1 recommendation is stale:**

- `debugging_plan.md:84` (2026-05-28 next-prompt list) claims the 2D residual is "100% pointwise-curvature
  truncation: `hndA_i_interface` ... has O(h) error. Replace it with ... `surface_tension_force_2d`" — a
  false premise; 2D never used `hndA_i_interface`, so there is nothing to replace and no wiring change that
  can move the floor.
- The plan's own earlier 2026-05-06 audit note (quoted in `diagnose_a5_bisection.py:411-416`) and the newer
  2026-06-02 entry (`debugging_plan.md:39`: "higher-order 2D curvature stencil (research push, the only
  lever on the 2D O(h) floor)") already state the correct status: the FTC path is in place and the floor is
  O(h) polygon geometry.
- Older entries carried the confusion, e.g. `debugging_plan.md:313`/`:341` ("requires replacing pointwise
  `hndA_i_interface`/`surface_tension_force_2d` with an integrated γ-flux form") and `:179` ("worth checking
  but not the active blocker" — this audit is that check).

**Physical interpretation of the remaining floor.** `surface_tension_force_2d` is *exact* for the polygon
(the actual PL interface): γ(t_next − t_prev) has magnitude 2γ sin(θ/2) per vertex. The 2.27e-3 residual is
the Young–Laplace imbalance between that exact PL surface-tension force and the discrete per-phase pressure
flux on the dual faces of a *polygonal approximation of a circle* — geometric modelling error of the
interface representation itself, O(h) by the probe (p ≈ 1.05–1.10). Driving it toward machine precision
requires a higher-order interface representation (curved-arc reconstruction, e.g. via
`reconstruct_arc_length_and_bulge_area`, curvature_2d.py:184-231) or a γ-flux form made exactly consistent
with the discrete pressure flux — i.e. a research-level "higher-order 2D curvature stencil", exactly as the
2026-06-02 plan entry reframes it. This is a DESIGN_LIMITATION of the PL interface, not a code bug.

## 6. Suggested fix

No code change. Documentation hygiene only:

1. In `debugging_plan.md`, mark the 2026-05-28 "Tier 2B step 1 for 2D — integrated γ-flux curvature
   rewrite" recommendation (line 84) as **already-satisfied-by-construction / superseded**, pointing to the
   2026-06-02 reframing ("higher-order 2D curvature stencil is the only lever on the 2D O(h) floor") and to
   this audit. Same for the stale phrasing at lines 313 and 341.
2. If the O(h) floor is to be attacked, the correct next item is a *new* one (curved-arc / higher-order
   interface reconstruction consistent with the pressure flux), not a dispatch change — the dispatch needs
   no modification anywhere.
3. Optional: close the "worth checking but not the active blocker" aliasing question from
   `debugging_plan.md:179` with a pointer to §4b (they do not alias; 2D never touches the cotangent path).

## 7. Re-verification (2026-07-02, second independent pass)

A second, independent audit pass re-ran both probes from a fresh session and reproduced every number:

- **Harness** (`diagnose_a5_bisection.py --skip-3d --n-steps 50 --redistribute-mass`, wall 5.5s for A.5.b):
  A.5.a max|F| = 2.3749e-03 (×0.625 of the 3.8e-3 full-dynamic baseline); A.5.b t=0 = 2.3749e-03,
  steps 1..50 = 2.2717e-03 at every printed step; |dM/M0| = 2.555e-15, |dV/V0| = 3.542e-16,
  311→311 vertices, 32→32 interface. **Both floors identical to the pinned regression values
  (2.3748568012e-3 / 2.2716937802e-3, debugging_plan.md:47-49) — no drift since the 2026-05-28 pin.**
- **Instrumentation** (`probe_dispatch.py`): `curvature_path='integrated'`, dim=2 → 32/32 calls to
  `surface_tension_force_2d` + `integrated_curvature_normal_2d`, **0** calls to `hndA_i_interface`
  or `integrated_hndA_i_interface`; per-vertex `|F_dispatch − γ(t_next − t_prev)| = 0.000e+00` (bitwise).
  `'stokes'` dim=2 delegates to the same FTC path (32/0); `'csf_dual'` uses 32× `_csf_dual_surface_tension`
  + 32× `integrated_curvature_normal_2d`, still 0 heron calls.
- **Refinement sweep**: n_iface 32/64/128 → max|F| 2.374857e-03 / 1.110493e-03 / 5.375644e-04,
  observed order p = 1.10 and 1.05 — clean O(h), matching the 2026-05-06 audit note verbatim
  (`diagnose_a5_bisection.py` CLI help, lines ~405-417).
- **Git archaeology re-confirmed**: `git log --diff-filter=A -- ddgclib/operators/multiphase_stress.py`
  → single adding commit `1690ce0 2026-04-07`; `git show 1690ce0:...` lines 128-134 already contain
  `else: F[:2] = surface_tension_force_2d(v, gamma, interface_nbs)`.
- **Current dispatch line numbers re-checked**: module-top import at `multiphase_stress.py:66`;
  `'integrated'` dim==3 branch at :271-274 (`hndA_i_interface`, the ONLY route to the cotangent path,
  unreachable for dim=2); dim=2 `else` at :275-278; `'stokes'` 2D delegate at :261-263; unknown-path
  ValueError at :265-269.

Verdict unchanged: **CORRECT_AS_INTENDED** — Tier 2B step 1's "wire in `surface_tension_force_2d`"
rewrite is already satisfied by construction and has been since the first multiphase commit; the
2.37e-3/2.27e-3 floors are O(h) polygon-geometry error (DESIGN_LIMITATION of the PL interface), and
the only remaining lever is a higher-order interface representation, exactly as the 2026-06-02
debugging_plan entry ("higher-order 2D curvature stencil ... the only lever on the 2D O(h) floor",
debugging_plan.md:39) already reframes it.

## 8. Files referenced

- `/home/endres/projects/ddgclib/ddgclib/operators/multiphase_stress.py:66,196-278` (dispatch)
- `/home/endres/projects/ddgclib/ddgclib/operators/curvature_2d.py:91-181` (FTC operator)
- `/home/endres/projects/ddgclib/debugging_plan.md:39,47-49,76,84,179,313,341`
- `/home/endres/projects/ddgclib/cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py:400-417`
- `/home/endres/projects/ddgclib/docs_temp/02_physics_foundations.md:65-75,135`
- `/home/endres/projects/ddgclib/docs_temp/code_map/multiphase_surface_tension.md:98-116`
- Probe: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/tier2b-2d-status/probe_dispatch.py`
