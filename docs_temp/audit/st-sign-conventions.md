# Audit: surface-tension sign conventions (`e_ij = -e_ij # WHY???` flip + `csf_dual` inner-phase-by-index)
> Sources checked | Written 2026-07-02 by physics-audit workflow (probe re-run and all file:line citations re-verified 2026-07-02)

## Item
1. **(a)** Empirical sign flip `e_ij = -e_ij  # WHY???` at `ddgclib/_curvatures_heron.py:241` (propagated to `hndA_i_interface` at `:361` as `e_ij = -e_ij  # Sign convention (matches hndA_i)`) — underived but load-bearing for the surface-tension direction.
2. **(b)** The `csf_dual` curvature path assumes "inner phase = higher index" (`ddgclib/operators/multiphase_stress.py:318-325`) — if a droplet were phase 0 inside phase 1, the ST force sign would flip.

## What the physics requires
`docs_temp/02_physics_foundations.md` §3.4 (:65-75) and `docs_temp/sources/fundamentals.md` §1.5 (:62-66): the surface-tension force on an interface parcel is

    F_st_i = ∫_{Γ_i} γ κ N dS

which on a **convex droplet points radially INWARD** (toward the centre of curvature) with Young–Laplace magnitude γ·κ·(dual interface measure): κ = 1/R in 2D, κ = 2H = 2/R in 3D. The direction is a purely **geometric** property of the interface — it must NOT depend on which integer labels the two phases carry (γ lookup is already symmetric: `MultiphaseSystem.get_gamma_pair`, `ddgclib/multiphase.py:583-588`, canonicalises to `(min,max)`).

## What the code does

### (a) The `e_ij = -e_ij` flip
`HNdC_ijk` (`_curvatures_heron.py:93-122`) computes `hnda_ijk = w_ij * e_ij` with `w_ij = (1/8)(l_jk² + l_ik² − l_ij²)/A = ½ cot θ_apex` — the standard cotan weight, and is **exactly linear in `e_ij`** (the lengths enter only through norms). `hndA_i` (:213-298) and `hndA_i_interface` (:310-393) accumulate `Σ_j ½(cot α + cot β) e_ij` but first flip `e_ij = -(x_j − x_i) = x_i − x_j` (:241, :361).

Derivation the `# WHY???` comment is missing: the discrete Laplace–Beltrami of position is `(Δ_S x)_i = Σ_j ½(cotα+cotβ)(x_j − x_i) = ∫_{Γ_i} 2H N dA` with **N the inward normal on a convex surface** (mean-curvature vector points toward the centre of curvature). With the flip, the code instead returns

    HNdA_i = Σ_j ½(cotα+cotβ)(x_i − x_j) = +∫_{Γ_i} 2H N_out dA   (OUTWARD-oriented)

The consumer `_interface_surface_tension` (`ddgclib/operators/multiphase_stress.py:271-274`) then applies `F = -gamma * HNdA[:dim]` (same in `ddgclib/operators/surface_tension.py`, `F = -γ·HNdA`, per 02 §3.4 item 3). The two negations **cancel**: F = −γ·(outward 2H A_i N) = inward Young–Laplace pull. So the flip is one half of a self-consistent sign-convention *pair*; removing either negation alone flips the ST force outward (droplet explodes), removing/adding both is a no-op.

### (b) `csf_dual` inner-phase convention
`_csf_dual_surface_tension` (`multiphase_stress.py:281-342`) redirects the FTC/Heron magnitude along `S_inner = Σ_j frac_inner(i,j)·A_ij` and picks `inner_phase = phases[-1]` — literally the **higher phase index** (:322-325, comment :318-319 "'interior side' is the higher-phase index by convention"). `dual_area_vector` (`ddgclib/operators/stress.py:52+`) returns the dual-face area vector oriented outward *from v_i toward v_j*, so `S_inner` points toward whichever phase carries the higher index — into the droplet only if the droplet happens to be phase 1. The default `'integrated'` path (2D FTC `γ(t_next − t_prev)`, `ddgclib/operators/curvature_2d.py:91-181`; 3D `−γ·HNdA` per above) and the `'stokes'` path (`integrated_hndA_i_interface`, `_curvatures_heron.py:396-522`, conormal boundary integral) use **no phase information at all** for direction.

## Probe design
`/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/st-sign-conventions/probe_st_signs.py`, run with `/home/endres/anaconda3/envs/ddg/bin/python` (cwd = repo root).

- Production fixtures `droplet_in_box_2d(R=0.01, L=0.05, ref=2/2)` (16 interface vertices) and `droplet_in_box_3d(R=0.01, L=0.05, ref=1/1)` (26 interface vertices), which build the full multiphase pipeline (`compute_vd` barycentric duals, `assign_simplex_phases`, `identify_interface_from_subcomplex`). γ = 0.05 set via `mps._gamma = {(0,1): 0.05}`.
- For every interface vertex, evaluate the **production assembly** `_interface_surface_tension(v, dim, mps, HC, curvature_path=p)` for p ∈ {`integrated`, `stokes`, `csf_dual`}; record cos(F, r̂_out) and the ratio |F| / (γ·κ·dual-measure), with dual-measure = ½(l_prev+l_next) from `HC.interface_edges` in 2D and the stencil dual area C_i in 3D; κ = 1/R (2D), 2/R (3D).
- **Phase swap**: relabel via `mps.assign_simplex_phases(HC, dim, criterion_fn=swapped)` (droplet→0, outer→1) + `mps.identify_interface_from_subcomplex(HC, dim)`; assert the interface vertex set is unchanged; repeat all three paths.
- **Flip check (a)**: sign of HNdA·r̂_out on the sphere; |HNdA|/((2/R)·C_i); exact linearity of `HNdC_ijk` in `e_ij` (so the no-flip variant is exactly −HNdA).

## Probe OUTPUT (verbatim, re-run 2026-07-02)

```
===== 2D circle R=0.01, droplet=phase 1 (default convention) (dim=2, n_interface=16) =====
  path=integrated : n=16 inward=16 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-0.9994 mean=-0.9997
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 0.000e+00
  path=stokes     : n=16 inward=16 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-0.9994 mean=-0.9997
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 0.000e+00
  path=csf_dual   : n=16 inward=16 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-0.9984 mean=-0.9992
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 6.939e-18

===== 2D circle R=0.01, SWAPPED: droplet=phase 0 (dim=2, n_interface=16) =====
  path=integrated : n=16 inward=16 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-0.9994 mean=-0.9997
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 0.000e+00
  path=stokes     : n=16 inward=16 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-0.9994 mean=-0.9997
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 0.000e+00
  path=csf_dual   : n=16 inward=0 outward=16 zero=0
     cos(F, r_out): min=+0.9984 max=+1.0000 mean=+0.9992      <-- SIGN FLIPPED
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0006 mean=1.0003
     |sum F| = 6.939e-18

[flip check 3D] HNdA (with e_ij=-e_ij flip) outward on 26/26 vertices;
  |HNdA| / (2/R * C_i): min=1.0000 max=1.0000 mean=1.0000
  linearity: HNdC_ijk(-e) + HNdC_ijk(e) = [0. 0. 0.] (dual areas equal: True)

===== 3D sphere R=0.01, droplet=phase 1 (default convention) (dim=3, n_interface=26) =====
  path=integrated : n=26 inward=26 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-1.0000 mean=-1.0000
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 1.626e-19
  path=stokes     : n=26 inward=26 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-1.0000 mean=-1.0000
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 2.724e-19
  path=csf_dual   : n=26 inward=26 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-1.0000 mean=-1.0000
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 1.626e-19

===== 3D sphere R=0.01, SWAPPED: droplet=phase 0 (dim=3, n_interface=26) =====
  path=integrated : n=26 inward=26 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-1.0000 mean=-1.0000
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 1.626e-19
  path=stokes     : n=26 inward=26 outward=0 zero=0
     cos(F, r_out): min=-1.0000 max=-1.0000 mean=-1.0000
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 2.724e-19
  path=csf_dual   : n=26 inward=0 outward=26 zero=0
     cos(F, r_out): min=+1.0000 max=+1.0000 mean=+1.0000      <-- SIGN FLIPPED
     |F|/(gamma*kappa*dual_len): min=1.0000 max=1.0000 mean=1.0000
     |sum F| = 2.861e-19
```

## Verdict and reasoning

### (a) `e_ij = -e_ij` flip — **CORRECT_AS_INTENDED** (self-consistent convention pair; comment is debt, not a defect)
- With the flip, HNdA points **outward** on 26/26 sphere vertices with |HNdA| = (2/R)·C_i to 4+ digits; the production `-gamma * HNdA` at `multiphase_stress.py:274` then gives the inward Young–Laplace pull with ratio 1.0000 on every vertex, and |ΣF| ≈ 1.6e-19 (Newton-3rd-law closure on the closed sphere).
- The 2D FTC default is exact by construction (the two edge tensions at a polyline vertex sum to exactly γ(t_next − t_prev)); measured ratio 1.0000–1.0006 (the 0.06% is chord-vs-arc, the documented O(h) polygon floor).
- `HNdC_ijk` is exactly linear in `e_ij` (verified: `HNdC_ijk(-e) + HNdC_ijk(e) = [0,0,0]`), so deleting the flip alone would exactly negate HNdA and turn ST into an outward (unstable, explosive) force. The flip and the `-γ` multiplier must only ever be changed **together**. The independently-derived `'stokes'` path agrees in sign and magnitude with the flip+(−γ) pair, which is a second, stencil-independent confirmation.
- Derivation (should replace the `# WHY???` comment): flipping makes `HNdA_i = Σ_j ½(cotα+cotβ)(x_i − x_j) = −(Δ_S x)_i·A_i = +∫_{Γ_i} 2H N_out dA`, the *outward-oriented* integrated mean-curvature normal; consumers then apply the physical form `F_st = −γ ∫ 2H N_out dA`.

### (b) `csf_dual` inner-phase-by-index — **CONFIRMED_BUG** (in an experimental, non-default path)
- Swapping labels (droplet = phase 0 inside phase 1) flips the `csf_dual` force from cos(F, r̂_out) ≈ −1 to **+1 on every interface vertex, in both 2D (16/16) and 3D (26/26)**, with unchanged magnitude — surface tension then pushes the droplet **outward**, the opposite of γκN. The interface geometry and vertex set are identical under the swap (asserted in the probe); only the arbitrary integer labels changed, so this is a demonstrable physics violation of the label-invariance requirement.
- Root cause: `inner_phase = phases[-1]` at `multiphase_stress.py:325`; `S_inner = Σ frac_inner·A_ij` then points toward the *higher-indexed* phase, not toward the centre of curvature.
- `'integrated'` (default) and `'stokes'` are bit-identical under the swap — fully label-invariant, as required.

### Reachability / severity
`csf_dual` is reachable only via the explicit `--curvature-path csf_dual` CLI flag of `cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py:401-406` (repo-wide grep: no other call site passes it); the production default everywhere (`multiphase_stress_force`, `multiphase_stress.py:113`; `_interface_surface_tension`, `:197`; A5B regression tests, all case studies) is `'integrated'`. Moreover both `droplet_in_box_2d/3d` builders hardcode droplet = phase 1 (`_multiphase_droplet.py:134` "phase 0 (outer) or phase 1 (droplet)"; docstrings `:200` and `:323` "Phase 0 = outer fluid, Phase 1 = droplet"), which matches the `csf_dual` convention — so no currently existing case is mis-signed, including the oscillating-droplet diagnostics that expose the flag. The bug bites only a future user who runs the A/B probe on a mesh whose interior phase carries the lower index (e.g. a *bubble* labelled 0 inside liquid 1 with the labels of the toy builders inverted).

## Suggested fix
1. **(b)** Make `_csf_dual_surface_tension` orientation geometric instead of label-based: keep `S = Σ_j frac_k·A_ij` for either phase k, but orient it against the label-invariant curvature normal it already computes —
   `inward_ref = delta_t` (2D FTC vector, points toward centre of curvature) or `inward_ref = -HNdA` (3D); then `if np.dot(S, inward_ref) < 0: S = -S` before normalising. This removes the `phases[-1]` convention entirely. (Alternatively at minimum assert/document the "interior = higher index" requirement at the call site.)
2. **(a)** Replace `# WHY???` at `_curvatures_heron.py:241` (and the echo at `:158`, `:585`) with the derivation above, and add a comment at `multiphase_stress.py:274` / `surface_tension.py` that the leading minus is the partner of the `e_ij` flip (change both or neither). A cheap guard: a unit test asserting `dot(F_integrated, x_v - centre) < 0` on a coarse sphere already exists in spirit (`test_simplex_aware_curvature.py:299-353` for `'stokes'` vs cotangent); extend it to run the swapped-label variant for every `curvature_path` — the probe in this audit is a ready template.

## Files
- Probe: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/st-sign-conventions/probe_st_signs.py`
- Code audited: `ddgclib/_curvatures_heron.py:93-122,213-298,310-393,396-522`; `ddgclib/operators/multiphase_stress.py:107-342`; `ddgclib/operators/curvature_2d.py:91-181`; `ddgclib/geometry/_dual_split_2d.py:545-612`; `ddgclib/multiphase.py:131-165,289-396,573-588`; `ddgclib/geometry/domains/_multiphase_droplet.py:130-137,200,323`.
