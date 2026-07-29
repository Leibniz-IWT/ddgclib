# 05 — Test Cases and Validation
> Sources: code_map/cases_dynamic_inventory.md, code_map/validation_and_tests.md, sources/debugging_plan_distilled.md, sources/library_audit_and_features.md | Written: 2026-07-02 by understand-and-document workflow

How to run every dynamic case, what "success" means analytically, the sanctioned validation utilities, and the pytest recipes. For the *status* of each case (passing/failing/broken) see [06_known_issues_and_debug_history.md](06_known_issues_and_debug_history.md).

---

## 0. Environment rules (apply to everything below)

- Use **`/home/endres/anaconda3/envs/ddg/bin/python`** from repo root `/home/endres/projects/ddgclib`. System/base python lacks matplotlib (figures silently skipped) and pytest.
- Exception: `oscillating_droplet_p_ref` scripts run **from inside that folder** (`python3 scripts/...` — sibling bare imports).
- Common multiphase pipeline: `setup_*()` in `<case>/src/_setup.py` returns `(HC, bV, mps, bc_set, dudt_fn, retopo_fn, params)` → `symplectic_euler(HC, bV, dudt_fn, dt, n_steps, dim, bc_set, callback, retopologize_fn=retopo_fn, remesh_mode=..., remesh_kwargs=...)` → record with `StateHistory(fields=['u','p','phase','is_interface'], save_dir=<case>/results/snapshots)` → animate with `dynamic_plot_fluid(..., phase_field='phase', interface_field='is_interface')`. Outputs land in `<case>/fig/` and `<case>/results/`.

## 1. Fast feedback loops, fastest first (for debugging workflows)

1. **`cases_dynamic/oscillating_droplet/static_droplet_2D.py`** — 100 steps hardcoded, ε=0 circular droplet equilibrium. Target: KE stays at machine precision; **currently FAILS** (KE grows 0 → 4.4e-3 in 100 steps under full Delaunay retopo, per `INVESTIGATION_PROMPT.md`). The best bug-reproduction loop for the multiphase interface-stress/retopo problem. Contrast: `cube2droplet/diagnostic_no_retopo.py` (fixed connectivity, dual-only recompute, 5000 steps) is stable (circularity 0.71 → 0.95).
2. **`cases_dynamic/oscillating_droplet/diagnose_a5_bisection.py`** — the A.5 harness (n=100 default; flags `--skip-3d --n-steps --split-method --redistribute-mass --results-suffix`). Pins against the regression floors in §5. **Footgun:** without `--redistribute-mass` you exercise the legacy guard path and reproduce the pre-fix 3D 1.44e-3 — expected.
3. **`cube2droplet/cube_to_droplet_2D_adaptive.py`** — 30-step smoke (fix the `Cube2droplet` capital-C import first; all cube2droplet scripts have it).
4. **`oscillating_droplet/oscillating_droplet_2D_adaptive.py`** (no flags) — Delaunay-vs-adaptive side-by-side, 200 steps/mode at refine 2 (~20 s); `--full` ≈ 80 min; `--n-steps`, `--refine` overrides.
5. **`dam_break/dam_break_2D.py`** — deliberate smoke test, t_end = 0.02 s, O(200–400) steps.
6. **`oscillating_droplet_p_ref` cases 11/12** — 81 samples × 5 substeps, minutes; quantitative a2(t) vs Rayleigh with active retopology. README dev note: "USE ONLY case 11 and 12 for dev purposes".
7. **`liquid_bridge_equilibrium/Case_1`** — 100 dynamic steps per refinement; machine-precision numbers to regress against (the healthiest quantitative suite).
8. **`oscillating_droplet/oscillating_droplet_2D.py`** — ~460 steps, ~2 min, emits `results/score.json`.
9. Slow: `electrolysis_bubble_2D.py` (O(5–10k) steps), `cube_to_droplet_2D.py` (5000 steps), any 3D/`--full` variant.

## 2. Case inventory (detail)

### 2.1 oscillating_droplet/ — Rayleigh–Lamb flagship (2D validation currently FAILING)

- **Run:** `python cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py` (~2 min), `oscillating_droplet_3D.py` (~5 min), `mesh_convergence_2D.py`.
- **Physics** (`src/_params.py`, default = *overdamped* oil droplet): ρ_d=800, μ_d=0.5, ρ_o=1000, μ_o=0.1, γ=0.05 N/m, R0=0.01 m, ε=0.05 mode-l=2 perturbation, L_domain=5·R0, refinement 3/3. Weakly-compressible EOS heavily softened: c_s = max(10·u_scale, 1.0) = **1.0 m/s** → K_d=800 Pa, K_o=1000 Pa. An `UnderdampedParams` set (water droplet in air, γ=0.072) exists but is NOT wired into the runners.
- **Timestep** (computed at runtime, `oscillating_droplet_2D.py:79-91`): dt = min(0.25·dx_min/c_s, 0.5·√(ρ_d·dx_min³/γ)) ≈ 2.5e-4; t_end ≈ 0.114 s → ~460 steps.
- **Analytical reference** (`src/_analytical.py`): Rayleigh frequency **ω²_2D = (l³−l)·γ/(ρ_d·R0³)**, Lamb damping **β_2D = (2l²−1)·μ_d/(ρ_d·R0²)**; compare R_max(t) against `max_radius_envelope`. 3D (l=2): ω² = 8γ/(ρR0³), β = 5μ/(ρR0²); Young–Laplace ΔP = γ/R (2D), 2γ/R (3D) — these formulas are unit-tested in `ddgclib/tests/test_multiphase.py`.
- **Scoring:** `src/_metrics.py:oscillation_score` → `results/score.json`; `summary = max(l2_err, mass_drift, tail_growth−1)`; baselines in `baselines/baseline_oscillation.json` / `baseline_equilibrium.json`.
- **Current state:** `results/score.json` = l2_error_normalized **5.62** (target < 0.2), tail_growth **1.68** (target < 1.0), mass_drift 3.7e-16 (machine precision). Mass conserved; trajectory far off analytical. README presents it as working — stale.
- **3D:** runs (872 steps, mass drift 1e-14) but emits **no oscillation_score**; no checked-in 3D baseline; R_max saturates at ~5·R0 (mesh boundary) — 3D validation cannot be asserted.
- **Extra scripts:** `static_droplet_2D.py` (§1.1), 10× `diagnose_*.py` forensic scripts (verified 2026-07-02: a5_bisection, a5_dissect_2d, a5_step1_diff, balance, discrete_dp, dual_only, eos_ic, retopo_effect, split_methods, static).

### 2.2 oscillating_droplet_p_ref/ — pressure-reference benchmark (tet-volume closure)

- Standalone 12-case benchmark; benchmark-only operators in `scripts/pr33_operators.py` (VolumeGradientPressureState, tet_volume_matrix, compressible EOS correction, incompressible projection, active_retopology_tet_remap) — deliberately kept OUT of core `ddgclib/operators/`.
- **Run (from inside the folder):**
  ```bash
  python3 scripts/sphere_fheron_dynamic_integrator_p_ref.py --case-id 11 \
    --subdivision 1 --steps 81 --substeps-per-sample 5 --t-final 0.016 \
    --shape-mode-ar 1.05 --inertia-scale 1.08 \
    --out-dir out/dynamic_integrator_p_ref_case11_multiphase
  ```
  `--case-id 12` = incompressible-projection repeat. Cases 5/6 via `scripts/sphere_fheron_flux_fv_ddgclib_benchmark.py --closure compressible|incompressible`.
- **Physics:** l=2 perturbation (AR 1.05) of R=1 mm sphere, γ=0.072, ρ=1000; fit a2(t) vs **ω_l² = l(l−1)(l+2)γ/(ρR³)**; Heron surface force F_i = −γ(HN dA)_i; tet-volume continuity B_{t,i}=∂V_t/∂x_i.
- **Why it exists:** documents and works around the core artifact — "after a Delaunay flip, the local dual cell can change even when the liquid has not compressed; the code reads a connectivity change as physical compression" → pressure/continuity rebuilt from **tet volumes** after every retopo. Achieved a2(t_final) = 0.0283 (cases 11/12, ~1060–1084 changed tets) vs Rayleigh 0.02989 — still ~5–6% low. `out/` is fully gitignored; the 33 MB pptx is the only versioned artifact.

### 2.3 electrolysis_bubble/ — H2 bubble on electrode (gravity + gas injection)

- **Run:** `python cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py`, `..._3D.py`; replay `view_polyscope.py --dim 2`. `electrolysis_bubble_fritz_2D.py` = geometry-only smoke (axisymmetric Young–Laplace ODE γκ = P0 + ρ_diff·g·z → Fritz sessile profile; no time integration).
- **Physics** (`src/_params.py`): γ=0.072; ρ_liq=1000, μ_liq=0.1 (100× real, "for stability"); ρ_gas=10 (softened H2), μ_gas=0.01; **K_liq = K_gas = 1e5 Pa matched** — README: "the single biggest stability decision" (physical K_gas<K_liq drives runaway inflation). g=9.81, R0=1 mm, box 8×8 mm, P0=0. Gas injection placeholder `src/_reaction.py:inject_gas_mass` (dm_dt_2d=2e-2 kg/(s·m)).
- **Numerics:** dt = min(cfl_safety·dx_min/c_s, 0.4·√(ρ_liq·dx_min³/γ)), cfl_safety=0.05 (README claims 0.025 for 3D; `_params.py` has only 0.05), c_s=100 m/s gas side; t_end_2d=2e-4 s → O(5–10k) steps. Requires per-step **NaN sanitisation** (3D barycentric duals produce NaN volumes at box corners). Uses full `_retopologize_multiphase` — dual-only retopo was *less* stable here (opposite of static_droplet_2D; contradictory retopo guidance across cases).
- **Analytical reference** (`src/_analytical.py`): capillary length λ=√(γ/(ρ_diff·g))≈2.7 mm; **Fritz detachment radius R_det=(3·R0·λ²/2)^{1/3}≈2.23 mm**; success = R_eq climbs toward it (2D: ~1.4–1.5 mm; 3D: 1→1.9 mm); detachment heuristic = gas COM rises 0.5·R0. Fritz physics comes from the `cases_mean_flow/equil_bubble/` thermodynamics toy.
- **Known limitation (verbatim):** Delaunay retopology "does not preserve a thin detaching interface indefinitely; bubble mass is slowly lost once the interface starts to neck".

### 2.4 dam_break/ — Martin–Moyce smoke test

- **Run:** `python cases_dynamic/dam_break/dam_break_2D.py` (water+air, exercises `multiphase_dudt_i` — primary surface-tension operator test), `dam_break_2D_no_air.py` (single-phase free surface via Tait–Murnaghan), `dam_break_3D*.py`.
- **Physics:** a=0.05 m; tank 4a×2a=0.2×0.1 m; column a×2a. Water 1000/1e-3, air 1.225/1.81e-5, γ=0.072, g=9.81. c_s≈14 m/s → K_l≈1.96e5. **alpha_art=2.0 artificial viscosity** (SPH-style μ_art=α·ρ·c_s·dx). **t_end=0.02 s deliberately short**, cfl=0.1, `skip_triangulation=True` (full Delaunay creates cross-phase edges that destabilise the interface). Integrator wrapped in try/except so partial snapshots survive divergence.
- **Success:** qualitative only (collapse, KE_liq history, phase plot); no quantitative Martin–Moyce front comparison in-script. README documents spurious accelerations at truncated corner dual cells as the known limiter.

### 2.5 cube2droplet/ — square→circle relaxation (**imports broken**)

- **All scripts unrunnable as-is**: they import `cases_dynamic.Cube2droplet.src._setup` (capital C) but the dir is `cases_dynamic/cube2droplet/` → verified ModuleNotFoundError. Fix imports (or rename dir) before use. No README in the dir.
- Physics: square half-side R=0.01 m relaxes to circle R_eq=2R/√π under γ. **Two disagreeing parameter sources:** `src/_params.py` (μ_d=0.5, μ_o=0.2, γ=0.01, K_d=800, dt=1e-5) vs constants hardcoded in `cube_to_droplet_2D.py:33-48` (MU_D=2.0, MU_O=1.0, K_D=100, DT=2e-4, N_STEPS=5000). Success: circularity R_min/R_max→1, ΔP→γ/R_eq, KE decays.
- `diagnostic_no_retopo.py` (5000 steps, fixed connectivity) is the documented STABLE reference (circularity 0.71→0.95).

### 2.6 liquid_bridge_equilibrium/ — catenoid benchmarks (machine-precision reference)

- **Run:** `python cases_dynamic/liquid_bridge_equilibrium/Case_1_equilibrium_particle_particle_bridge_benchmark.py` … Case_5. Self-contained; outputs → `out/Case_N`.
- Constants (Case_1): GAMMA=0.0728, RADIUS=1.0, THETA_P=20°, REFINEMENTS=(2,3,4,5) = 8/16/32/64 boundary vertices, DYNAMIC_DT=2e-6, DYNAMIC_STEPS=100. Case 5 = volumetric `stress.py` path on a thickened catenoid (up to n_total 8385; `DDGCLIB_CASE5_WORKERS` env override).
- **Reference:** Endres (2024) catenoid values embedded in scripts. Current: DDG capillary force error ~1e-15–1e-16 % (Cases 1/2/5, machine precision), ~1e-10 % (Case 3), converging (Case 4). Case 5 checks the volumetric Cauchy residual is numerically zero at equilibrium.

### 2.7 template/ and Hagen_Poiseuile / Hydrostatic_column

- `template/template.py` — canonical single-phase 5-step workflow on 2D Poiseuille (mock dudt, `euler_velocity_only`, dt=1e-3, 500 steps, seconds). Use `oscillating_droplet/src/_setup.py` as the real multiphase template.
- `cases_dynamic/Hagen_Poiseuile/` (note misspelling) and `cases_dynamic/Hydrostatic_column/` back the machine-precision equilibrium pytest fixtures (§4).

## 3. Validation utilities (the ONLY sanctioned error metrics)

FVM doctrine: `v.p` is a volume average; compare `|p_i·Vol_i − ∫P dV|`, never `|v.p − P(x_vertex)|`.

From `ddgclib/analytical/_integrated_comparison.py` (re-exported from `ddgclib.analytical`):
- `volume_averaged_scalar(f, v, dim=2, polygon_method="barycentric_dual_p_ij", n_gauss=7)` (L166) — correct value to ASSIGN to `v.p`
- `integrated_pressure_error(HC, interior_vertices, P_analytical, dim=2, ...)` (L222) — per-vertex `|p_i·Vol_i − ∫P dV|`
- `integrated_l2_norm(HC, interior_vertices, P_analytical, dim=2, ...)` (L282) — volume-weighted L2
- `compare_stress_force(HC, interior_vertices, dim=2, mu=1e-3)` (L350) — `{'max_F','mean_F','median_F','F_norms'}` equilibrium check

**LIMITATION:** all `dim==3` branches silently fall back to **point-wise** `P(x)·Vol` ("exact 3D dual cell integration via faces is deferred") — 3D comparisons are not truly integrated. The quadrature (`integrated_gradient_3d` in `_divergence_theorem.py`) exists; 3D dual-cell *face extraction* is the gap.

Recording/diagnostics (`ddgclib/data/`):
- `StateHistory(fields=('u','p'), record_every=1, record_every_t=None, save_dir=None, conservation=False, dim=None)` — pass `.callback` to the integrator; `save_dir` auto-writes `state_{idx:06d}_t{t:.6f}.json`; `conservation=True` merges `compute_conservation` per snapshot. Query: `.query_vertex(key, field)`, `.query_field_at_time(t, field)`, `.query_diagnostics()`.
- `compute_conservation(HC, dim)` → ke, momentum, mass_total, volume_total, h_min/h_max, u/p extrema, per-phase variants when `v.m_phase` exists; `drift_fractions(initial, current)`.
- `save_state`/`load_state` — JSON `ddgclib_state_v1`; **duals are NOT persisted** (recompute `compute_vd` after load); domain bounds inferred from vertex bbox.

Benchmark runner: `python benchmarks/run_integrated_benchmarks.py [--linear-only | --comparison --dim 2 | --convergence --dim 2]` — sweeps barycentric/circumcentric × p_ij/bary polygons × symmetric/jittered (seed 42); machine-precision gate err < 1e-13 applies to the PRODUCTION method barycentric/p_ij (all dims/fields/meshes).  The non-production variants have known, pre-existing nonzero floors and print FAIL by design of the gate: `bary` integration polygons ~5.6e-2..1.2e-1 even on symmetric meshes; circumcentric duals ~2.7e-3..5.7e-3 on jittered meshes (measured 2026-07-02, lane 7 — these floors predate the 2026-07 debug session; see debug_session/lane7-cleanup-regression-lockin.md).

## 4. Pytest recipes

```bash
# from /home/endres/projects/ddgclib, ddg env
pytest ddgclib/tests/ -v -m "not slow"                    # full fast suite, ~18-31 s (808 pass @ audit)
pytest ddgclib/tests/test_stress.py -v -m "not slow"       # stress ops: 2D machine precision
pytest ddgclib/tests/test_stress.py -v -m "slow"           # 3D channel/hydrostatic (directional only)
pytest ddgclib/tests/test_integrated_validation.py -v -m "not slow"
pytest ddgclib/tests/test_multiphase.py ddgclib/tests/test_multiphase_flat_interface.py \
       ddgclib/tests/test_multiphase_stress_per_phase.py -v      # multiphase, all fast
pytest ddgclib/tests/test_a5b_longrun_regression.py -v -m "not slow"  # 2D pinned floor
pytest ddgclib/tests/test_case_oscillating_droplet.py -v   # incl. TestStaticDroplet{2D,3D}RetopologyFloor
pytest ddgclib/tests/test_dem_*.py -v -m "not slow"        # DEM 124 fast (+1 xfail liquid bridge)
```

Key regression gatekeepers:
- `test_multiphase_flat_interface.py` — "Tier 2A gatekeeper": flat-interface zero force, ATOL=1e-12, 2D and 3D (3D via **Kuhn-decomposed cube** — default `Complex(3).triangulate()` does NOT give a planar z=0 interface).
- `test_a5b_longrun_regression.py` + `test_case_oscillating_droplet.py::TestStaticDroplet{2D,3D}RetopologyFloor` — pinned static-droplet floors (2D 2.3748568e-3 frozen / 2.2716938e-3 post-retopo; 3D 6.0153e-05 / 7.3768e-05), 1% trip wire, mass/volume drift < 1e-10.
- `test_stress.py::TestHagenPoiseuilleStress2D` — Poiseuille equilibrium median residual < 1e-13.
- Empty tombstone: `test_hagen_poiseuille_equilibrium.py` (0 tests, points at `cases_dynamic/Hagen_Poiseuile_equilibrium/test_equilibrium.py`, not collected).

## 5. What is machine-precision-verified vs qualitative

**Machine precision (≤1e-12..1e-13):** dual-face closure Σ_j A_ij = 0 and antisymmetry (2D+3D); integrated gradients of linear fields (all dual methods, symmetric+jittered, 1D/2D; 3D p_ij linear precision incl. jittered in `TestEdgeAreaCache3D`); quadratic fields on symmetric meshes; 2D Poiseuille equilibrium and hydrostatic linear-pressure force; flat-interface multiphase zero force (2D+3D, with/without γ); catenoid capillary force (liquid_bridge Cases 1/2/5).

**Bounded/directional/qualitative only:** jittered-mesh Poiseuille residual (< 0.1 — diffusion-form O(h) truncation); ALL 3D channel/hydrostatic stress cases (@slow, direction+profile only); sphere curvature (finite/non-divergent only); multiphase long-run = pinned empirical floors, not zero.

**Untested gaps:** true 3D integrated pressure comparison; curved-interface multiphase curvature accuracy (flat-interface gatekeeper deliberately excludes it); 3D convergence rates; 3D oscillation_score baseline; `load_state` vs domains larger than the vertex hull.
