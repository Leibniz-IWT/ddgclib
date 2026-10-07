# Audit: body-force (gravity) handling
> Sources checked | Written 2026-07-02 by physics-audit workflow
> Independently re-verified 2026-07-02 (second pass): all probe numbers below reproduced
> exactly on re-run (injection identity 1.776e-15; point-value equilibrium 2.233e-12;
> EOS-path equilibrium 1.5967e-06; vol-avg-IC failure max|a|=6.029e+01, mean a_y=+9.810;
> `volume_averaged_scalar(1)`=2.000000 on both meshes; quadrature weight sum=1.000000;
> shoelace/dual_vol=1.0000), and all quoted file:line references re-checked against source.

Sources: `docs_temp/02_physics_foundations.md` §2, §3.3; `docs_temp/sources/fundamentals.md` §1.4, §1.6;
`docs_temp/code_map/operators_stress.md`; code as quoted below.
Probes: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/gravity-bodyforce/`
(`probe_hydrostatic.py`, `probe_diagnose.py`, `probe_va_factor2.py`, `probe_va2.py`, `probe_final.py`),
run with `/home/endres/anaconda3/envs/ddg/bin/python` from the repo root.

## 1. What the physics requires

Fundamentals (`docs_temp/sources/fundamentals.md:56-58,70`):

    F_body,i = ∫_{V_i} rho b dV = m_i b_i   (exact for uniform b; F_body = v.m * b)
    a_i = (F_stress,i + F_body,i + F_gamma,i) / m_i

For uniform gravity `b = g_vec` the body acceleration is `F_body/m_i = g_vec`
independent of the parcel mass, so adding a constant `g_vec` to the stress
acceleration is *exactly* the required Lagrangian body-force term — including
on interface vertices with per-phase masses.

`docs_temp/02_physics_foundations.md:59-63` (§3.3) states the intended design:
body force is **not implemented inside `stress_acceleration`**; "Cases add
gravity in their own `dudt_fn` wrappers."

## 2. What the code does

### 2.1 Operator: no body-force term (as documented)

`ddgclib/operators/stress.py:778-825` — `stress_acceleration` returns
`stress_force(...) / v.m` only. There is literally no gravity/body term in the
operator (the only match for "body" in the file is the docstring at
`stress.py:789`). `dudt_i` is an alias at `stress.py:829`.

Minor doc nit: the docstring writes Newton's law with the body force,
`m_i * dv_i/dt = F_stress_i + F_body` (`stress.py:789`), then returns
`a_i = F_stress_i / m_i` (`stress.py:790,824-825`) without saying F_body is
intentionally external. Confusing but consistent with 02 §3.3.

### 2.2 Injection points in gravity-driven cases (all the same pattern)

- Hydrostatic column: `make_gravity_dudt` at
  `cases_dynamic/Hydrostatic_column/src/_setup.py:118-168`;
  `dudt_with_gravity(v) = dudt_stress(v) + g_vec` (`:165-166`),
  `g_vec[gravity_axis] = -g` (`:157-158`).
- Dam break, multiphase: `cases_dynamic/dam_break/src/_setup.py:182-186`
  (`dudt_fn(v) = _stress_fn(v) + g_vec` on top of `multiphase_dudt_i`);
  single-phase variant `:332-336`.
- Electrolysis bubble: `cases_dynamic/electrolysis_bubble/src/_setup.py:461-472`
  (`return a + gravity_vec` on top of `multiphase_dudt_i`); Fritz variant
  `electrolysis_bubble_fritz_2D.py:758-767`.

Consistency checks:
- **Same substep, same vertices**: every integrator evaluates the *whole*
  user `dudt_fn` per vertex per RHS evaluation via `_compute_accel`
  (`ddgclib/dynamic_integrators/_integrators_dynamic.py:543`, call sites
  `:733` euler, `:823` symplectic_euler, `:956` rk45 — each RK stage, `:1053`
  euler_velocity_only, `:1166` euler_adaptive). Gravity therefore enters
  every stage/substep exactly once, on exactly the integrated vertex set.
- **No double counting**: `grep -rn gravity|g_vec|body_force` over
  `ddgclib/operators/*.py` and `ddgclib/dynamic_integrators/*.py` finds no
  gravity term anywhere in the operator/integrator layer.
  `HydrostaticEOSMass` (`ddgclib/initial_conditions.py:329-412`) is a mass/
  pressure IC only, no force.
- **Interface vertices**: the uniform `+ g_vec` is mass-independent, hence
  exact for interface vertices (`v.phase = -1`, per-phase masses). Gravity is
  NOT skipped on interface vertices in dam_break/electrolysis.
- **Frozen walls**: vertices in `bV` are excluded from integration entirely
  (stress AND gravity) — intended no-slip wall semantics. In free-surface
  mode the top-face vertices are not in `bV` and do receive gravity.
- Small caveat: the electrolysis wrapper's defensive guard
  (`electrolysis_bubble/src/_setup.py:467-471`) returns zeros (dropping
  gravity too) for vertices with non-finite mass or non-finite stress
  acceleration. This is a deliberate NaN guard for degenerate 3D corner
  duals, not double counting; acceptable.

## 3. Probe design and OUTPUT

2D hydrostatic column, `setup_hydrostatic_column(dim=2, n_refine=3, H=1,
rho=1000, g=9.81)`, 145 vertices / 113 interior. Equilibrium requires
`dudt_fn(v) = a_stress + g_vec ≈ 0` on interior vertices.

**(a) Injection exactness** (`probe_hydrostatic.py` [3],[4]):

    (dudt_fn - stress_only - g_vec) max abs component = 1.776e-15   (interior)
    boundary sample:                                    1.776e-15

Gravity is injected exactly once, uniformly, on every vertex the wrapper is
called on.

**(b) Equilibrium with point-value linear pressure** (`probe_diagnose.py`):

    With POINT-VALUE p_i = P(x_i):
      max |a_net| interior = 2.233e-12   (rel to g: 2.276e-13)

Machine precision — the `+ g_vec` body force exactly balances the discrete
pressure gradient. This is the same floor as the linear-precision benchmark
(`test_integrated_validation.py:807-822`, which assigns point values at
`benchmarks/_integrated_benchmark_cases.py:509`).

**(c) Equilibrium through the EOS path the dynamic case actually integrates**
(`probe_final.py` [A]; replicates Hydrostatic_2D.py Section 2b:
`HydrostaticEOSMass` + `pressure_model=TaitMurnaghan`):

    max|a| = 1.5967e-06   mean = 1.0541e-06   (rel to g: 1.628e-07)

Near-perfect balance (small residual = nonlinearity of the compressible
profile). Gravity handling through the EOS pipeline is correct.

**(d) Equilibrium through the case's DEFAULT static path — FAILS**
(`probe_hydrostatic.py` [2], `probe_diagnose.py`; this is exactly the
"Section 2 static residual" that `Hydrostatic_2D.py:55-56` prints):

    case dudt_fn with HydrostaticPressure vol-avg IC, interior vertices:
      max |a| = 6.029e+01 m/s^2   (6.1 g);  mean a_y = +9.810e+00 (= +g)

Diagnosis (`probe_diagnose.py`): dual-face closure is exact (|ΣA_ij| = 0),
mass is exact (m = rho·dual_vol), but `v.p` is **exactly 2× the pressure**:
dp = v.p − P(x_i) equals P(x_i) at every vertex (e.g. +9197 Pa at y=0.0625
where P=9197 Pa). Doubled pressure ⇒ a_stress ≈ +2g ⇒ net ≈ +g upward.

**(e) Root cause — factor-2 quadrature bug** (`probe_va_factor2.py`,
`probe_va2.py`):

    volume_averaged_scalar(f≡1) over interior dual cells:
      rectangle() case mesh:  min=2.000000 max=2.000000
      raw Complex mesh:       min=2.000000 max=2.000000
    quadrature weight sum (n=7) = 1.000000
    dual polygon shoelace / dual_vol = 1.0000  (polygon and dual_vol correct)

`_dual_cell_pressure_integral_2d_simple`
(`ddgclib/analytical/_integrated_comparison.py:116-163`) computes the true
triangle area (`tri_area = 0.5*|cross|`, `:154-156`) and then accumulates
`w * P * 2.0 * tri_area` (`:161`). The Dunavant weights returned by
`_triangle_quadrature_points` (`ddgclib/analytical/_divergence_theorem.py:
271-324`) sum to **1**, so the correct formula is `w * P * tri_area`; the
`2.0*Area` convention in the docstring (`_divergence_theorem.py:281`) matches
weights summing to 1/2, which is NOT what the table contains. The 3D helpers
use the correct convention (`_triangle_integral_scalar`,
`_divergence_theorem.py:352-359`: `w * f * n_vec * 0.5` with |n_vec| = 2·Area)
— which is why the 3D benchmarks pass at 1e-12.

Blast radius of the factor 2 (all 2D, only when duals exist):
- `volume_averaged_scalar` (`_integrated_comparison.py:208-212`) returns 2×
  the volume average ⇒ `HydrostaticPressure` and `LinearPressureGradient`
  ICs assign `v.p = 2×⟨P⟩` (`ddgclib/initial_conditions.py:106,144`) —
  despite CLAUDE.md's FVM doctrine mandating these very utilities.
- The **sanctioned metrics** `integrated_pressure_error` (2D branch,
  `_integrated_comparison.py:271-275`) and `integrated_l2_norm` compare
  against a doubled ∫P dV. With a *correct* pressure field they report a
  false O(∫P dV) error; with the doubled IC they report ~0 (self-consistent
  cancellation). The Hydrostatic_2D dynamic loop's reported "Integrated P
  err" values (`Hydrostatic_2D.py:117-125,182-193`) are therefore doubled on
  the analytical side.
- Bonus defect (unused main paths): `_dual_cell_pressure_integral_2d`
  (`_integrated_comparison.py:68-114`) both keeps the `2.0*tri_area` factor
  and skips valid quadrature points with `if ti + tj > 1.0: continue`
  (`:105-106`) even though its collapsed-square mapping already keeps points
  inside the triangle.

Why never caught: the machine-precision force benchmark assigns point values
(`_integrated_benchmark_cases.py:509`); the IC unit tests apply ICs on meshes
**without duals** so they take the point-value fallback and assert point
values at atol 1e-12 (`ddgclib/tests/test_initial_conditions.py:86-103`);
`test_stress.py:1170-1181` applies the IC *before* `compute_vd` (fallback
again); no test asserts `volume_averaged_scalar(1) == 1`.

## 4. Verdict and reasoning

**Gravity/body-force handling itself: CORRECT AS INTENDED.**
The distillation claim is accurate — `stress_acceleration` has no body-force
term by design; every gravity case injects `a += g_vec` in a case-level
wrapper, which is the exact discrete `F_body/m_i` for uniform gravity, applied
at every integrator substep to the same non-frozen vertex set, never double
counted, and not skipped on interface vertices. Evidence: injection identity
1.8e-15; hydrostatic balance 2.2e-12 (point-value P) and 1.6e-6 (EOS path).

**Item verdict: CONFIRMED_BUG (collateral, high severity)** because the
mandated probe — "net force ~0 at equilibrium through whichever path the case
uses" — *fails* through the case's documented pressure-IC path (max |a| =
60 m/s² ≈ 6 g, mean +g upward), and the cause is a demonstrable factor-2
quadrature bug at `ddgclib/analytical/_integrated_comparison.py:161` that
doubles all 2D volume-averaged pressure ICs and corrupts the sanctioned 2D
integrated error metrics. The bug is in the validation/IC layer, not in the
gravity term.

## 5. Suggested fix

1. `_integrated_comparison.py:161`: change `w * P_analytical(x) * 2.0 *
   tri_area` → `w * P_analytical(x) * tri_area` (weights sum to 1).
2. `_dual_cell_pressure_integral_2d` (`:68-114`): apply the same factor fix
   and delete the erroneous `if ti + tj > 1.0: continue` (the collapsed
   mapping already stays inside the triangle), or delete the function in
   favour of `_simple`.
3. Fix the stale docstring `_divergence_theorem.py:279-281` to
   `∫_T f dA ≈ Area * Σ w_i f(x_i)`.
4. Add regression tests: `volume_averaged_scalar(lambda x: 1.0, v, dim=2)
   == 1.0` on a mesh WITH duals; `HydrostaticPressure` applied after
   `compute_vd` must reproduce `⟨P⟩` (not 2×); the Hydrostatic_2D "Section 2"
   static residual should then be re-baselined (expect ≈ point-value floor on
   symmetric meshes; note boundary-neighbour vol-avg cells remain less
   accurate when `P_ref` is large, since `dual_cell_polygon_2d` on truncated
   boundary duals is approximate).
5. Cosmetic: clarify `stress.py:787-790` docstring that `F_body` is
   intentionally excluded and must be added by the caller (cite 02 §3.3
   pattern `dudt_fn(v) = dudt_i(v) + g_vec`).

## 6. Droplet / bubble impact

The oscillating droplet cases run without gravity; `electrolysis_bubble` uses
the same (verified-correct) `a + gravity_vec` wrapper. None of
`cases_dynamic/oscillating_droplet*`, `cases_dynamic/electrolysis_bubble`, or
`cases_mean_flow/equil_bubble` import `volume_averaged_scalar`,
`HydrostaticPressure`, `LinearPressureGradient`, or
`integrated_pressure_error` (grep: zero hits), and their pressures come from
the EOS on `rho = m/Vol`, so the factor-2 bug does not enter their dynamics.
Indirect impact only: any future 2D droplet diagnostic built on
`integrated_pressure_error`/`integrated_l2_norm` would report doubled
analytical integrals until fixed.

## Skeptic review

> Adversarial re-verification 2026-07-02 (independent probes, not the auditor's
> scripts). Attempted to refute the claim on five fronts; all attempts failed.
> **Verdict upheld: CONFIRMED_BUG, severity high.**

Refutation attempts and results:

1. **"The 2.0 compensates for a half-area polygon convention"** — refuted.
   Independently recomputed `shoelace(dual_cell_polygon_2d(v)) / dual_vol` on
   the `rectangle()` case mesh: exactly 1.000000 (min = max) on all 113
   interior vertices, and `sum(dual_vol)` over all vertices = 0.992 ≈ domain
   area 1.0 (deficit is only boundary-cell truncation). The polygon is the
   full dual cell; nothing cancels the extra 2.0.
2. **"The failing IC path is not production-reachable"** — refuted.
   `setup_hydrostatic_column` (`cases_dynamic/Hydrostatic_column/src/_setup.py`)
   calls `compute_vd` + `cache_dual_volumes` *before*
   `HydrostaticPressure(...).apply(...)`, so `initial_conditions.py:102-106`
   takes the `volume_averaged_scalar` branch in the shipped case, not the
   point-value fallback. Reproduced `v.p / P(x_i)` = 2.000000 on every
   interior vertex of the real case mesh.
3. **"The probe misconfigured the physics"** — refuted by counterfactuals run
   through the *same* case `dudt_fn`: (a) default IC path max|a| = 6.029e+01,
   mean a_y = +9.810 (reproduces the auditor exactly); (b) overwrite `v.p`
   with point values → max|a| = 2.233e-12 (gravity term itself balances to
   machine precision); (c) overwrite with 0.5x the library's vol-avg integral
   → mean a_y drops to +1.6e-05 (the +g bias is entirely the factor 2; the
   remaining boundary-adjacent residual is the separate cell-average-vs-vertex
   offset noted in §5.4).
4. **"The 2x is an intentional convention"** — refuted. The
   `volume_averaged_scalar` docstring promises `(1/Vol_i) * ∫ f dV`; `f≡1`
   returns 2.000000, and a linear field returns exactly 2x the exact polygon
   centroid value (verified against shoelace centroid). The 7-point weight
   table in `_divergence_theorem.py:316-324` sums to 1.0 (0.225 +
   3·0.132394152788506 + 3·0.125939180544827), and the 3D helper in the *same
   module* (`_triangle_integral_scalar:357`, `w*f*n_vec*0.5` with
   |n_vec| = 2·Area) uses the weights-sum-1 convention. The two 2D helpers
   cannot both be consistent with it while carrying `2.0*tri_area`.
5. **"Existing tests validate current behaviour"** — refuted. Ran
   `test_initial_conditions.py` (20 passed): its fixtures never compute duals,
   and the `test_stress.py:1170-1181` fixture applies the IC *before*
   `compute_vd`, so only the point-value fallback is ever tested. No test
   asserts `volume_averaged_scalar(1) == 1`.

Strongest evidence found (new): the **self-cancellation trap**. With the
doubled IC in place, `integrated_pressure_error` reports max err = 2.8e-14
(both sides doubled, looks perfect) while the same field produces 6g spurious
accelerations in the dynamics; conversely a *correct* point-value pressure
field is reported as max err = 8.9e+01 by the same metric. The sanctioned 2D
validation metric therefore inverts good and bad fields — false confidence in
the broken state, false alarm on the correct one.

Severity assessment: **high stands**. Gravity/body-force handling itself is
correct (confirmed at 2.2e-12), but the collateral factor-2 corrupts (a) the
CLAUDE.md-mandated FVM IC pattern for every 2D case that applies
`HydrostaticPressure`/`LinearPressureGradient` after `compute_vd`, (b) the
sanctioned `integrated_pressure_error`/`integrated_l2_norm` 2D metrics used in
`Hydrostatic_2D.py`'s dynamic loop and final diagnostics (their printed
values are meaningless), and (c) does so silently and self-consistently.
Scope limits confirmed: 1D (Gauss–Legendre) and 3D (point-value branch) paths
are unaffected, as are all EOS-driven pressures (droplet/bubble cases).
