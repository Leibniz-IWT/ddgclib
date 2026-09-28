# Physics Foundations: the Lagrangian DDG Finite-Volume Method
> Sources: sources/fundamentals.md, code_map/operators_stress.md, code_map/multiphase_surface_tension.md, code_map/validation_and_tests.md, sources/debugging_plan_distilled.md, sources/library_audit_and_features.md | Written: 2026-07-02 by understand-and-document workflow

Orientation doc for the mathematical method. Detail lives in
[sources/fundamentals.md](sources/fundamentals.md) (derivations + formulation history),
[code_map/operators_stress.md](code_map/operators_stress.md) (line-level code map),
[code_map/multiphase_surface_tension.md](code_map/multiphase_surface_tension.md) (multiphase/EOS/surface tension),
[code_map/validation_and_tests.md](code_map/validation_and_tests.md) (what is actually verified and to what tolerance).

---

## 1. Continuum starting point

Integral Cauchy momentum equation on a **material (Lagrangian) control volume** $V(t)$ with closed boundary $S(t)$, outward normal $\mathbf{n}$:

$$\frac{d}{dt}\int_{V(t)}\rho\mathbf{v}\,dV = \int_{S(t)}\boldsymbol{\sigma}\cdot\mathbf{n}\,dS + \int_{V(t)}\rho\mathbf{b}\,dV$$

There is **no convective flux term** — the control volume moves with the fluid, so the LHS is a pure material derivative. Newtonian constitutive decomposition:

$$\boldsymbol{\sigma} = -p\,\mathbf{I} + \boldsymbol{\tau},\qquad \boldsymbol{\tau}=2\mu\boldsymbol{\varepsilon},\qquad \boldsymbol{\varepsilon}=\tfrac12(\nabla\mathbf{v}+(\nabla\mathbf{v})^\top)$$

Sign convention: $p$ positive in compression (fluid convention, $-p\mathbf{I}$).

**No continuity equation is solved.** Mass conservation holds by construction (§2). Compressibility enters only through the EOS (§5): pressure responds to dual-cell volume change.

## 2. Discrete Lagrangian parcels

- Each primal vertex `v` in `HC.V` is a material parcel with **fixed mass** `v.m` = $m_i = \rho_i\,\mathrm{Vol}_i^{\mathrm{dual}}$ = const. `v.m` is never modified by the physics operators (only by explicit mass-redistribution utilities and reservoir BCs — see [04_solver_pipeline.md](04_solver_pipeline.md)).
- The parcel occupies the **barycentric dual cell** $V_i$ built by `hyperct.ddg.compute_vd(HC, method="barycentric")`. Its boundary tiles exactly into one dual face per primal edge $ij$, with oriented area vector $\mathbf{A}_{ij}$ (outward from $i$). $\mathbf{A}_{ji}=-\mathbf{A}_{ij}$ exactly → Newton's 3rd law pairwise.
  - 2D: $\mathbf{A}_{ij}$ = 90° rotation of the shared dual edge (`dual_area_vector`, `ddgclib/operators/stress.py:158-173`).
  - 3D: DEC p_ij polygon interleaving tet barycenters with primal-face barycenters $(\mathbf{x}_i+\mathbf{x}_j+\mathbf{x}_k)/3$ — this interleaving is what gives **linear precision to machine eps** for barycentric duals (`_dual_area_vector_3d_p_ij`, `stress.py:182`; legacy `e_star` fallback at `stress.py:280` is NOT linearly precise).
- Discrete momentum ODE per parcel:

$$m_i\frac{d\mathbf{u}_i}{dt} = \mathbf{F}_{\mathrm{stress},i} + \mathbf{F}_{\mathrm{body},i} + \mathbf{F}_{\gamma,i},\qquad \mathbf{F}_{\mathrm{stress},i}=\sum_{j\in N(i)}\big(\mathbf{F}_{p,ij}+\mathbf{F}_{v,ij}\big)$$

Entry points (all in `ddgclib/operators/stress.py`): `stress_force` (:702, default `mu=8.9e-4` Pa·s = water 25 °C), `stress_acceleration` = `F/v.m` (:778), `dudt_i` alias (:829) — the canonical integrator `dudt_fn`, bound via `functools.partial(dudt_i, dim=2, mu=..., HC=HC)`.

## 3. Force operators (discrete formulas + code locations)

### 3.1 Pressure force

$$\mathbf{F}_{p,ij}=-\tfrac12(p_i+p_j)\,\mathbf{A}_{ij}$$

`pressure_flux(p_i, p_j, A_ij)` at `stress.py:680-682`, used in `stress_force` at `:711`.
Properties: pairwise momentum-conserving; recovers $-\int_{V_i}\nabla p\,dV$ **exactly for linear $p$** (discrete divergence theorem on the barycentric dual). For interior vertices, where the closure $\sum_j\mathbf{A}_{ij}=\mathbf{0}$ holds, the face-average form is algebraically identical to the half-difference form $-\tfrac12(p_j-p_i)\mathbf{A}_{ij}$ (the $p_i$ terms cancel).

### 3.2 Viscous force — diffusion form ONLY (load-bearing design decision)

Face-centered rank-1 velocity gradient $(\nabla\mathbf{u})_f = (\mathbf{u}_j-\mathbf{u}_i)\otimes\hat{\mathbf{d}}_{ij}/|\mathbf{d}_{ij}|$ contracted with the face:

$$\mathbf{F}_{v,ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,(\mathbf{u}_j-\mathbf{u}_i)\,(\hat{\mathbf{d}}_{ij}\cdot\mathbf{A}_{ij}),\qquad \mathbf{d}_{ij}=\mathbf{x}_j-\mathbf{x}_i$$

`viscous_flux(mu, delta_u, d_ij, A_ij)` at `stress.py:685-699`.

**The transpose term $(\nabla\mathbf{v})^\top$ is deliberately omitted.** The original vertex-centered formulation (`sigma_f = 0.5(sigma_i+sigma_j)`, full symmetric $\boldsymbol\tau$, preserved in `ddgclib/operators/stress_pointwise.py`) produced **O(1) acceleration residuals (~0.34) at Poiseuille equilibrium** because the rank-1 face gradient has spurious discrete compressibility: $\mathrm{tr}((\nabla\mathbf{u})_f)=\Delta\mathbf{u}\cdot\hat{\mathbf{d}}/|\mathbf{d}|\ne 0$ on diagonal edges even for divergence-free fields, so the symmetric form picks up a spurious $\mu\nabla(\nabla\cdot\mathbf{u})$ term. The diffusion form restores machine-precision equilibrium. Full history: [sources/fundamentals.md §2](sources/fundamentals.md). Recorded open trade-off: compressible or full-tensor constitutive models will need a divergence correction or higher-order gradient reconstruction.

Accuracy: exact integrated Laplacian for quadratic $u$ on well-formed duals (midpoint-derivative argument); **O(h) truncation on jittered/non-symmetric meshes** (jittered Poiseuille residual only bounded < 0.1 in tests).

### 3.3 Body force

$$\mathbf{F}_{\mathrm{body},i}=m_i\,\mathbf{b}_i \quad(\text{exact for uniform } \mathbf{b})$$

**Not implemented inside `stress_acceleration`** — there is no gravity term in the operator (audit flag, [sources/library_audit_and_features.md](sources/library_audit_and_features.md)). Hydrostatic cases verify equilibrium via the pressure field IC (`HydrostaticPressure`), not an in-operator body force. Cases add gravity in their own `dudt_fn` wrappers.

### 3.4 Surface tension (as implemented — Fundamentals docs are stale here)

Physical form $\mathbf{F}_{\gamma,i}=\int_{\Gamma_i}\gamma\kappa\mathbf{N}\,dS$ over the interface piece inside parcel $i$. Three implementations:

1. **2D multiphase (default, `curvature_path='integrated'`)** — exact FTC identity on piecewise-linear curves: $\int_{\Gamma_i}\kappa\mathbf{N}\,ds = \mathbf{t}_{next}-\mathbf{t}_{prev}$, so $\mathbf{F}_{\gamma}= \gamma(\mathbf{t}_{next}-\mathbf{t}_{prev})$, magnitude $2\sin(\theta/2)$. `surface_tension_force_2d` / `integrated_curvature_normal_2d` in `ddgclib/operators/curvature_2d.py:91-181`. Exact for PL curves; **first-order** static-droplet residual on polygonal approximations of smooth interfaces (this is the 2D droplet floor, see §6).
2. **3D multiphase** — cotangent-Heron integrated mean-curvature normal on the interface sub-mesh: $\mathbf{F}=-\gamma\,\mathrm{HNdA}_i$ via `hndA_i_interface` (`ddgclib/_curvatures_heron.py:310-393`); alternative `'stokes'` path `integrated_hndA_i_interface` (conormal boundary integral $\gamma\oint_{\partial\Gamma_i}\boldsymbol{\nu}\,dl$, exactly zero on planar interfaces) kept as a regression-tested A/B probe — **do not delete it as dead code**.
3. **Thin-film / surface meshes (no dual mesh)** — `ddgclib/operators/surface_tension.py:28-61`: `surface_tension_force(v, gamma=0.072, ...)` returns $-\gamma\,\mathrm{HNdA}_i$ from the full-mesh cotan stencil.

Assembly for interface vertices happens in `_interface_surface_tension` (`ddgclib/operators/multiphase_stress.py:196-278`). Design rule (docstring :29-33): surface tension is a **separate force on sharp-interface vertices — never add $\gamma\kappa$ to the pressure field**.

Sign-convention debt: an empirical `e_ij = -e_ij  # WHY???` flip at `_curvatures_heron.py:241` (propagated to :361) is load-bearing for the surface-tension direction and has never been derived.

### 3.5 Multiphase per-phase stress

Sharp interface = **primal subcomplex** (edges 2D / triangles 3D between top-simplices of differing phase); `MultiphaseSystem.simplex_phase` is authoritative; interface vertices carry `v.phase = -1` and per-phase arrays `m_phase/p_phase/rho_phase/dual_vol_phase`. Force (`multiphase_stress_force`, `ddgclib/operators/multiphase_stress.py:107-193`):

$$\mathbf{F}_i = \sum_{k\in\text{phases present}}\sum_j\Big[-\tfrac12\big(p_i^{(k)}+p_j^{(k)}\big)\mathbf{A}_{ij}^{(k)} + \frac{\mu_k}{|\mathbf{d}_{ij}|}\Delta\mathbf{u}\,(\hat{\mathbf{d}}\cdot\mathbf{A}_{ij}^{(k)})\Big] + \mathbf{F}_{\gamma,i}$$

with phase sub-face areas $\mathbf{A}_{ij}^{(k)}=\mathrm{frac}_k\,\mathbf{A}_{ij}$ from `edge_phase_area_fractions` (`ddgclib/geometry/_dual_split_2d.py:545-612`). Viscosity is **exact per-phase $\mu_k$, no harmonic mean** (stale docstrings claim otherwise). Bulk vertices collapse exactly to single-phase `stress_force`. Known defects (details in [code_map/multiphase_surface_tension.md §7](code_map/multiphase_surface_tension.md)): 50/50 hardwired split on interface-polyline edges even under `split_method='exact'`; `p==0.0` treated as "missing"; unguarded `p_phase[v.phase]` wrap on interface vertices in `MultiphaseEOS.__call__`; two-phase assumptions hardcoded (no triple junctions).

## 4. FVM volume-averaged field convention

All scalar vertex fields are **dual-cell volume averages, never point samples**:

$$v.p = \frac{1}{\mathrm{Vol}_i}\int_{V_i} P(\mathbf{x})\,dV \quad\Rightarrow\quad v.p\cdot\mathrm{Vol}_i = \int_{V_i} P\,dV \text{ to machine precision for polynomial } P$$

- NEVER assign `v.p = P(x_vertex)`. Use `volume_averaged_scalar` (`ddgclib/analytical/_integrated_comparison.py:166`) or the IC classes (`HydrostaticPressure`, `LinearPressureGradient`) which branch to the averaged form when duals exist.
- Validation must use integrated comparisons: `integrated_pressure_error` (:222), `integrated_l2_norm` (:282), `compare_stress_force` (:350). Never `abs(v.p - P(x_vertex))`.
- **Caveat**: all `dim==3` branches of these utilities silently fall back to point-wise $P(x)\cdot\mathrm{Vol}$ ("exact 3D dual cell integration deferred") — 3D comparisons are not truly integrated.

Velocity ICs are point-wise (the convention is stated for scalars).

## 5. EOS coupling (weakly compressible pressure closure)

Pressure is resolved inside the force pass by `_resolve_pressure` (`stress.py:634-673`), three modes:
1. `pressure_model=None` → read `v.p` as-is (incompressible / prescribed field).
2. Plain callable → `p = fn(v)`.
3. `EquationOfState` object → $\rho_i = m_i/\mathrm{Vol}_i^{\mathrm{dual}}$, $p_i = P(\rho_i)$; **side effect: writes `v.p` and `v.rho` in place**. Zero-volume guard: `vol < 1e-30` → `eos.pressure(eos.rho0)`.

EOS classes (`ddgclib/eos/` package — note `ddgclib/_eos.py` is 9 lines of dead code):

| Class | Law | Defaults | Caveats |
|---|---|---|---|
| `TaitMurnaghan` | $P = P_0 + \frac{K}{n}\big[(\rho/\rho_0)^n - 1\big]$ | $\rho_0{=}1000$, $P_0{=}101325$, $K{=}2.15\times10^9$, $n{=}7.15$ | `rho_clip=(0.9,1.1)` clips density **inside `pressure()` only** — outside ±10% the EOS silently flat-lines (zero effective sound speed) while `density()`/`sound_speed()` ignore the clip; $c(\rho_0)\approx1466$ m/s |
| `IdealGas` | $P=\rho R_{sp} T$ (isothermal) | $\rho_0{=}1.225$, $T{=}293.15$, $R{=}287.058$ | `sound_speed` is isothermal $\sqrt{RT}\approx290$ m/s (ABC docstring says "isentropic" — stale); default $P_0\approx103\,093$ Pa ≠ 1 atm |
| `MultiphaseEOS` | per-phase dispatch, $\rho_k=m_k/\mathrm{Vol}_k$ | — | unguarded `p_phase[v.phase]` on interface vertices (phase=-1) wraps to the last phase |

Dynamic cases **soften** the EOS (artificial sound speed ~1 m/s, e.g. $K_d=800$ Pa in the oscillating droplet) to keep the acoustic CFL tractable — see [code_map/cases_dynamic_inventory.md](code_map/cases_dynamic_inventory.md). Stiff EOS + crude dual-volume split at interfaces is the dominant spurious-force generator (1% volume-split error × $K/n\approx3\times10^8$ Pa → ~3 MPa spurious pressure).

Key stability coupling: pressure comes from the **dual volume**, so any retopology-induced dual-volume jump (3D Delaunay non-uniqueness on cospherical clouds flips edges, shifting `dual_vol` by ~1e-8 against bit-frozen mass) is misread by the EOS as physical compression — the root cause of the historical 3D retopology blow-up (fixed via the `redistribute_mass_multiphase` guard + snapshotting; see [sources/debugging_plan_distilled.md](sources/debugging_plan_distilled.md)).

## 6. Physics invariants that MUST hold (verification checklist)

Machine-precision invariants — pinned by tests, any violation means a real regression:

| Invariant | Statement | Verified tolerance | Test / gate |
|---|---|---|---|
| Dual-face closure | $\sum_j \mathbf{A}_{ij} = \mathbf{0}$ on every interior dual cell | atol 1e-12 (2D & 3D) | `test_stress.py::TestDualAreaVector{2D,3D}` |
| Momentum antisymmetry | $\mathbf{A}_{ij} = -\mathbf{A}_{ji}$ → $\mathbf{F}_{p,ij}=-\mathbf{F}_{p,ji}$ | atol 1e-12 | same |
| Linear-field exactness | integrated gradient of any linear field exact on barycentric AND circumcentric duals, symmetric AND jittered meshes | < 1e-13 | `benchmarks/run_integrated_benchmarks.py --linear-only`; `test_integrated_validation.py` |
| Uniform-field zero force | uniform $p$ / uniform $\mathbf{u}$ → $\mathbf{F}=0$ | atol 1e-10 | `TestStressForce2D/3D` |
| Poiseuille equilibrium | median residual of `stress_acceleration` at analytical Poiseuille | < 1e-13 (2D symmetric) | `test_stress.py::TestHagenPoiseuilleStress2D` |
| Hydrostatic linear-P force | net pressure force at hydrostatic equilibrium | < 1e-12 (incl. jittered) | `test_integrated_validation.py::TestIntegratedStress2D` |
| Flat-interface multiphase | `multiphase_stress_force` = 0 on a flat interface, with and without γ | ATOL 1e-12 (2D & 3D Kuhn mesh) | `test_multiphase_flat_interface.py` ("Tier 2A gatekeeper") |
| Mass per parcel | $m_i$ = const (single-phase, no reservoir BCs, frozen mesh) | exact by construction | `test_conservation.py` |

Empirical pinned floors (NOT zero — bit-stable regression trip wires, >1% shift fails):

| Quantity | Value | Test |
|---|---|---|
| Static droplet 2D max\|F\| — step-0 peak / post-retopo steady state | 2.3749e-3 / **2.2717e-3** (100% curvature-stencil O(h) truncation) | `test_a5b_longrun_regression.py` (`A5B_2D_PEAK`/`A5B_2D_END`, verified 2026-07-02) |
| Static droplet 3D max\|F\| — post-retopo steady state | **7.3768e-05** (frozen-mesh floor is 6.0153e-05, pinned in `test_case_oscillating_droplet.py`) | same file (@slow) |

Conservation caveats (do not mistake for bugs): a one-shot $|dV/V_0|\approx0.3$ "volume drift" after the first 3D retopo is a boundary-shell dual_vol zeroing artefact — never assert 3D volume from step 0. Real open leaks: single-phase retopology dual-volume-refresh leaks ~2-4% volume; adaptive remesh loses ~164% mass by step 80 (upstream `hyperct/remesh/_operations_2d.py:198` mass-averaging bug).

Only bounded/directional (NOT machine precision): jittered-mesh Poiseuille (< 0.1, O(h) diffusion truncation); all 3D channel/hydrostatic cases (@slow, direction/profile only); curved-interface curvature accuracy (untested by the flat gatekeeper); dynamic multiphase Rayleigh-Lamb (oscillating droplet 2D currently FAILS spec: tail_growth 1.68-1.85 vs target <1.0).

## 7. Known contradictions / stale claims to not trip on

- Fundamentals docs call surface tension "future"/"zero" — it is implemented (§3.4).
- Multiple docstrings promise harmonic-mean cross-phase viscosity — code uses exact per-phase $\mu_k$.
- `dudt_i` docstring (`stress.py:838-840`) shows passing `HC` via integrator kwargs → `TypeError`; always `functools.partial`.
- Test counts in CLAUDE.md/ARCHITECTURE.md/Fundamentals (415/488/604) are all stale; ~813 test functions in 37 files as of the 2026-06-08 audit.
- Boundary edges contribute **zero flux** (`A_ij = zeros`) — boundary vertices have incomplete surface integrals; no boundary-flux closure exists in `stress.py`.
