# Fundamentals: Lagrangian FVM Mathematical Formulation (all versions distilled)
> Sources: Fundamentals.md (CURRENT), Fundamentals_v1.md, Fundamentals_pointwise.md, Fundamentals_old.md | Written: 2026-07-02 by understand-and-document workflow

Precedence: `Fundamentals.md` is the declared "single source of truth". `Fundamentals_v1.md` is an expanded/formal rendering of the same content (notation table, conservation table) — it ALSO claims to be "the single source of truth"; treat Fundamentals.md as authoritative where they differ. `Fundamentals_old.md` is an earlier compact version of the same face-centered formulation (contains two derivations absent from the current doc — see §1.3 and §1.2 notes). `Fundamentals_pointwise.md` is the ARCHIVED, superseded vertex-centered formulation, kept because it documents a real failure mode (§2).

---

## 1. CURRENT FORMULATION (face-centered, integrated)

### 1.0 Continuum starting point

Integral Cauchy momentum equation over a material (Lagrangian) control volume $V(t)$ with closed boundary $S(t)=\partial V(t)$, outward normal $\mathbf{n}$:

$$\frac{d}{dt}\int_{V(t)}\rho\mathbf{v}\,dV = \int_{S(t)}\boldsymbol{\sigma}\cdot\mathbf{n}\,dS + \int_{V(t)}\rho\mathbf{b}\,dV$$

No convective flux term: $V(t)$ moves with the material, so the LHS is a pure material derivative. Newtonian constitutive decomposition:

$$\boldsymbol{\sigma} = -p\,\mathbf{I} + \boldsymbol{\tau},\qquad \boldsymbol{\tau}=2\mu\boldsymbol{\varepsilon},\qquad \boldsymbol{\varepsilon}=\tfrac12(\nabla\mathbf{v}+(\nabla\mathbf{v})^\top)$$

Sign convention: $p$ positive in compression (fluids). ($-p\mathbf{I}$; the pointwise doc notes the $+p\mathbf{I}$ solid-mechanics convention flips only the sign of $p$.)

### 1.1 Discrete Lagrangian parcels

- Each primal vertex $v_i$ (`v in HC.V`) is a material parcel with **fixed mass** $m_i=\rho_i\,\mathrm{Vol}_i^{\mathrm{dual}}=\mathrm{const}$ (stored `v.m`, never modified). Mass conservation is identical by construction — **no continuity equation is solved**; $\rho_i$ and $\mathrm{Vol}_i^{\mathrm{dual}}$ change in tandem.
- Parcel occupies the barycentric dual cell $V_i$ (built by `compute_vd` in `hyperct.ddg`). $S_i=\partial V_i$ tiles exactly into one planar dual face per primal edge $ij$, with exact oriented area vector $\mathbf{A}_{ij}$ (outward from $i$; $\mathbf{A}_{ji}=-\mathbf{A}_{ij}$ → exact Newton's 3rd law). Computed by `dual_area_vector()` in `ddgclib/operators/stress.py` / `e_star()` in `hyperct.ddg`. In 3D a dual face is a fan of triangles (`A_ijk_arr` from `e_star`); in 2D it is a line segment whose area vector is its 90°-rotation.
- Discrete momentum ODE (exact, since dual faces tile $S_i$ exactly):

$$m_i\frac{d\mathbf{u}_i}{dt} = \mathbf{F}_{\mathrm{stress},i} + \mathbf{F}_{\mathrm{body},i} + \mathbf{F}_{\gamma,i},\qquad \mathbf{F}_{\mathrm{stress},i}=\int_{S_i}\boldsymbol{\sigma}\cdot\mathbf{n}\,dS$$

### 1.2 Pressure force (from $-p\mathbf{I}$)

$$\mathbf{F}_{p,i} = -\int_{S_i}p\,\mathbf{n}\,dS = \sum_{j\in N(i)}\mathbf{F}_{p,ij},\qquad \mathbf{F}_{p,ij}=-\tfrac12(p_i+p_j)\,\mathbf{A}_{ij}$$

Properties: pairwise momentum-conserving ($\mathbf{F}_{p,ij}=-\mathbf{F}_{p,ji}$); recovers $-\int_{V_i}\nabla p\,dV$ **exactly for linear $p$** (discrete divergence theorem on barycentric dual). From Fundamentals_old.md: for interior vertices where $\sum_j\mathbf{A}_{ij}=\mathbf{0}$ (closure), the face-average form is algebraically equivalent to the half-difference form $-\tfrac12(p_j-p_i)\mathbf{A}_{ij}$ — the $p_i$ terms cancel because $\sum\mathbf{A}=\mathbf{0}$.

Code (verified): `ddgclib/operators/stress.py:681-682` helper and inside `stress_force()` at `:711`:
```python
F_p_ij = -0.5 * (p_i + p_j) * A_ij
```

### 1.3 Viscous force — diffusion (vector-Laplacian) form

For liquids whose compressibility lives in the EOS (Tait–Murnaghan etc.). Face-centered velocity gradient from the discrete exterior derivative (rank-1 outer product; formula from Fundamentals_old.md):

$$(\nabla\mathbf{u})_f = \frac{(\mathbf{u}_j-\mathbf{u}_i)\otimes\hat{\mathbf{d}}_{ij}}{|\mathbf{d}_{ij}|},\qquad \mathbf{d}_{ij}=\mathbf{x}_j-\mathbf{x}_i$$

$$\mathbf{F}_{v,ij} = \mu(\nabla\mathbf{u})_f\cdot\mathbf{A}_{ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij}\cdot\mathbf{A}_{ij}),\qquad \Delta\mathbf{u}=\mathbf{u}_j-\mathbf{u}_i$$

**The transpose term $(\nabla\mathbf{v})^\top$ is deliberately omitted** (see §2 for why). Code (verified): `ddgclib/operators/stress.py:699` helper, used in `stress_force()`:
```python
F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)
```

Quadratic exactness argument (Fundamentals_old.md only): for quadratic $u(y)$, $(u_j-u_i)/|\mathbf{d}_{ij}|$ is the **exact midpoint derivative** along $\hat{\mathbf{d}}$; $\hat{\mathbf{d}}\cdot\mathbf{A}$ is the effective face area for that directional derivative; on a well-formed (barycentric or circumcentric) dual the face sum recovers the exact integrated Laplacian — no non-orthogonality correction needed for the scalar Laplacian.

### 1.4 Body force

$$\mathbf{F}_{\mathrm{body},i}=\int_{V_i}\rho\mathbf{b}\,dV = m_i\mathbf{b}_i \quad(\text{exact for uniform }\mathbf{b};\ \texttt{F\_body = v.m * b})$$

Spatially varying $\mathbf{b}$: quadrature over the dual cell using exact geometry.

### 1.5 Surface tension (documented as "future"; PARTIALLY STALE — see §3)

$$\mathbf{F}_{\gamma,i}=\int_{\Gamma_i}\gamma\kappa\mathbf{n}\,dS$$

$\Gamma_i$ = interface portion inside parcel $i$, $\gamma$ = surface-tension coefficient, $\kappa$ = mean curvature. Docs say it will use the Laplace–Beltrami estimator `Curvature_i` in `ddgclib/operators/curvature.py` and is "zero in the current implementation" — but `ddgclib/operators/surface_tension.py` now exists with `surface_tension_force(v, gamma=0.072, dim=3, HC=None)` returning `-gamma * HNdA[:dim]` (integrated mean-curvature normal), plus `surface_tension_acceleration` and `dual_area_heron`. `multiphase_stress.py` also exists. The Fundamentals docs have not been updated for this.

### 1.6 Total force, acceleration, entry points

$$\mathbf{F}_{\mathrm{stress},i}=\sum_{j\in N(i)}(\mathbf{F}_{p,ij}+\mathbf{F}_{v,ij}),\qquad \mathbf{a}_i=\frac{\mathbf{F}_{\mathrm{stress},i}+\mathbf{F}_{\mathrm{body},i}+\mathbf{F}_{\gamma,i}}{m_i}$$

Verified code locations in `ddgclib/operators/stress.py`:
- `stress_force(v, dim=3, mu=8.9e-4, HC=None, ...)` — line 702 (default `mu` = water at 25 °C, Pa·s)
- `stress_acceleration(...)` — line 778
- `dudt_i = stress_acceleration` — line 829 (canonical `dudt_fn` for integrators; bind params with `functools.partial(dudt_i, dim=2, mu=0.1, HC=HC)`)
- Backward-compat wrappers in `ddgclib/operators/gradient.py`: `pressure_gradient` (:34), `velocity_laplacian` (:62), `acceleration` (:104) — thin forwards to the stress operators.

### 1.7 Per-timestep pipeline (Lagrangian FVM)

1. **State** on each vertex: cell-averaged pressure $p_i$ (`v.p`), parcel velocity $\mathbf{u}_i$ (`v.u`), fixed mass $m_i$ (`v.m`).
2. **Mesh construction (every timestep)**: primal Delaunay triangulation → barycentric dual via `compute_vd` (`hyperct.ddg`) → exact dual flux planes $\mathbf{A}_{ij}$ via `dual_area_vector()`/`e_star()` → cache dual volumes `v.dual_vol`.
3. **Force computation**: Stokes'/divergence theorem on each dual cell (operators §1.2–1.5).
4. **Time integration**: $\mathbf{a}_i = \mathbf{F}_{\mathrm{total},i}/m_i$; advance velocity and position with an integrator (Euler, symplectic Euler, RK45, adaptive CFL). Docs cite `ddgclib/dynamic_integrators.py`; the real path is the package `ddgclib/dynamic_integrators/` (`_integrators_dynamic.py`).
5. **EOS update**: recompute $p_i$ from new $\mathrm{Vol}_i^{\mathrm{dual}}$ (e.g. Tait–Murnaghan).
6. **Retopologize** and rebuild duals for the next step.

### 1.8 Conservation / accuracy properties (claimed)

| Property | Status |
|---|---|
| Mass per parcel | exact ($m_i$ const by construction) |
| Pairwise momentum | exact ($\mathbf{F}_{p,ij}+\mathbf{F}_{p,ji}=\mathbf{0}$) |
| Linear pressure recovery | exact (discrete divergence theorem, barycentric dual) |
| Poiseuille equilibrium residual | machine precision ("verified by 488 regression tests" — stale count, see §3) |

Design notes: compact two-point stencil (only the two vertices of each primal edge); never divides by volume inside the force loop; all geometric quantities ($\mathbf{A}_{ij}$, $\mathrm{Vol}_i$) exact for barycentric dual; the ONLY modelling approximation is the edge-based reconstruction of $\nabla\mathbf{u}$. Extensions (viscoelastic, non-Newtonian, bulk viscosity $\tfrac23\mu(\nabla\cdot\mathbf{v})\mathbf{I}$, surface tension) plug in by replacing/augmenting `stress_force()`; dual-geometry foundation unchanged.

---

## 2. EVOLUTION — what changed between versions and why

Inferred lineage: **Fundamentals_pointwise.md** (original refactoring plan + post-mortem) → **Fundamentals_old.md** (first face-centered doc) → **Fundamentals.md** (current condensed) ≈ **Fundamentals_v1.md** (expanded formal rendering of the current one).

### 2.1 Abandoned: vertex-centered point-wise formulation (Fundamentals_pointwise.md)

The original generalization from a pressure-only integrator to full Cauchy stress:
1. Compute full tensor at each vertex: `sigma_i = -p_i*np.eye(3) + tau_i`, with $\boldsymbol{\tau}_i$ from a **vertex-centered** discrete gradient $\nabla\mathbf{v}\big|_i \approx \frac{1}{\mathrm{Vol}_i}\sum_j \mathbf{v}_{ij}\otimes\mathbf{A}_{ij}$.
2. Face-average the tensor: `sigma_f = 0.5*(sigma_i + sigma_j)` (second-order consistent).
3. Contract: `F_from_ij_on_i = sigma_f @ A_ij`; sum over neighbors.
4. Orientation via sign test: `if np.dot(wedge, v_i.x_a - vc_12.x_a) > 0: wedge = -wedge` (flip small dual-triangle normals to point outward from $i$).

This was **implemented** (`ddgclib/operators/stress_pointwise.py`, still present) and tested on Poiseuille flow.

**Failure mode (the key lesson):** the full symmetric stress $\boldsymbol{\tau}=\mu(\nabla\mathbf{u}+(\nabla\mathbf{u})^\top)$ produced **O(1) acceleration residuals (~0.34) at equilibrium** instead of machine-precision zero. Cause: the rank-1 face gradient $(\nabla\mathbf{u})_f=\Delta\mathbf{u}\otimes\hat{\mathbf{d}}/|\mathbf{d}|$ has **spurious discrete compressibility** — its trace $\mathrm{tr}((\nabla\mathbf{u})_f)=\Delta\mathbf{u}\cdot\hat{\mathbf{d}}/|\mathbf{d}|$ is nonzero on diagonal (non-orthogonal) edges even for divergence-free fields. In the continuum, $\nabla\cdot[\mu(\nabla\mathbf{u}+(\nabla\mathbf{u})^\top)]=\mu\nabla^2\mathbf{u}+\mu\nabla(\nabla\cdot\mathbf{u})$, and the second term vanishes for incompressible flow; discretely, the symmetric form picks up the spurious $\mu\nabla(\nabla\cdot\mathbf{u})$ term while the diffusion form avoids it entirely.

### 2.2 The fix: face-centered diffusion form (Fundamentals_old.md → current)

Replace vertex tensors + tensor averaging with direct face fluxes:
- pressure: $-\tfrac12(p_i+p_j)\mathbf{A}_{ij}$ (unchanged in substance — the pointwise doc already noted $p_f\mathbf{A}$ reduces to the old scalar method),
- viscous: **drop the transpose term**, use $\mu(\nabla\mathbf{u})_f\cdot\mathbf{A}_{ij}$ = the two-point diffusion form.

Result: machine-precision zero residual on Poiseuille equilibrium. Trade-off explicitly recorded (Fundamentals_old.md Notes): for compressible flow or constitutive relations needing the full stress tensor, the symmetric form would require a **divergence correction or higher-order gradient reconstruction** — an open design item.

### 2.3 Fundamentals_old.md → Fundamentals.md / Fundamentals_v1.md

Content essentially preserved; changes are editorial + additions:
- Added body-force operator (§1.4) and the planned surface-tension term (§1.5) — absent in _old.
- Added explicit code excerpts with "exact line" markers, backward-compat wrapper list, extension table.
- v1 adds full notation table, orientation convention ($\mathbf{A}_{ji}=-\mathbf{A}_{ij}$), conservation-properties table, integrator table, and a figure (`lagrangian_fvm_distorted`, GitHub-hosted).
- **Dropped from current docs** (only in _old): the half-difference algebraic-equivalence note for the pressure flux and the quadratic-exactness/midpoint-derivative argument for the viscous form (both restated in §1.2/§1.3 above).

---

## 3. Contradictions and stale claims (verified against code, 2026-07-02)

1. **Surface tension is not purely "future"**: `ddgclib/operators/surface_tension.py` implements `surface_tension_force` (returns $-\gamma\,\mathrm{HNdA}_i$, default $\gamma=0.072$ N/m) and `surface_tension_acceleration`; `multiphase_stress.py` also exists. Docs' claim "this term is zero in the current implementation" and the plan to use `Curvature_i` Laplace–Beltrami are stale.
2. **Test count**: docs claim "488 regression tests"; project CLAUDE.md says ~415 tests (407 pass, 8 skipped). Both counts are snapshots; do not trust either without running pytest.
3. **Integrator module path**: docs say `ddgclib/dynamic_integrators.py`; actual is the package `ddgclib/dynamic_integrators/` (`_integrators_dynamic.py`, `_simulation.py`). Actual integrators (per CLAUDE.md): `euler`, `symplectic_euler`, `rk45`, `euler_velocity_only` (Eulerian, validation only), `euler_adaptive`.
4. **Duelling "single source of truth"**: both Fundamentals.md and Fundamentals_v1.md claim the title. Fundamentals.md is authoritative per workflow instruction.
5. **v1's wrapper flags** (`stress_force(pressure_only=True)` etc.) are a schematic; actual `gradient.py` wrappers are separate functions (`pressure_gradient:34`, `velocity_laplacian:62`, `acceleration:104`).
6. Verified line anchors in `ddgclib/operators/stress.py`: pressure-flux helper :681-682, viscous-flux helper :699, `stress_force` :702 (uses helpers, `F_p_ij` at :711), `stress_acceleration` :778, `dudt_i` alias :829. `stress_pointwise.py` exists as the archived formulation, as claimed.
