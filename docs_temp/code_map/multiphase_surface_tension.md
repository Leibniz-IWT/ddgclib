# Multiphase Layer: Sharp Interface, Surface Tension, Phase-Aware Stress, EOS
> Sources: ddgclib/multiphase.py, ddgclib/operators/multiphase_stress.py, ddgclib/operators/surface_tension.py, ddgclib/operators/curvature_2d.py, ddgclib/eos/ (package: _base.py, _tait_murnaghan.py, _ideal_gas.py, _multiphase_eos.py, _update.py), supporting: ddgclib/geometry/_dual_split_2d.py, ddgclib/geometry/_interface_subcomplex.py, ddgclib/_curvatures_heron.py | Written: 2026-07-02 by understand-and-document workflow

NOTE ON PATHS: the task named `ddgclib/eos.py` — **that file does not exist**. The EOS lives in the package `ddgclib/eos/`. A legacy `ddgclib/_eos.py` (9 lines) is dead code: a truncated CoolProp `PropsSI` wrapper with the import commented out and an incomplete `IAPWS(T)` function body — calling either raises `NameError`/`SyntaxError`-adjacent failures. Ignore `_eos.py` in the physics audit.

---

## 1. Sharp-interface data model (`ddgclib/multiphase.py`)

Sharp interface = **primal subcomplex** (closed polyline of primal edges in 2D; closed triangulated 2-manifold of primal faces in 3D). Not a diffuse phase field, no color function.

Per-vertex attributes (docstring multiphase.py:7–47):
- `v.phase` — int phase ID `0..n_phases-1` for bulk; `INTERFACE_PHASE = -1` for interface vertices (multiphase.py:66). Sentinel deliberately chosen so `v.p_phase[v.phase]` on an unguarded interface vertex wraps to the *last* phase via numpy negative indexing (comment 59–65: callers MUST gate on `v.is_interface` / `v.phase >= 0`).
- `v.is_interface` — True only for vertices of the interface subcomplex.
- `v.interface_phases` — frozenset of phase IDs of incident top-simplices (bulk vertex: `{v.phase}`).
- Per-phase arrays (length n_phases, created by `init_phase_fields`, multiphase.py:400–413): `v.m_phase[k]`, `v.p_phase[k]`, `v.rho_phase[k]`, `v.dual_vol_phase[k]`. Interface vertices are still *bulk fluid volume carriers* — their dual cell straddles phases and the arrays store the split contributions.

### Classes
- `PhaseProperties` dataclass (multiphase.py:110–128): `eos: EquationOfState`, `mu` [Pa·s], `rho0` [kg/m³], `name`.
- `MultiphaseSystem(phases, gamma=None)` (multiphase.py:131): `gamma` is `dict[(i,j)] -> float` [N/m], keys canonicalized to `(min,max)` (152–154). `self.simplex_phase: dict[frozenset(v.x tuples) -> int]` is the **authoritative source of truth** (155–160); vertex labels and the interface subcomplex are derived caches. `_simplex_criterion_fn` cached for `refresh`.

### Phase assignment
- `assign_phases(HC, criterion_fn)` (169–179): legacy per-vertex `v.phase = criterion_fn(v.x_a)`.
- `assign_simplex_phases(HC, dim, criterion_fn)` (183–235): evaluates `criterion_fn(centroid)` per top-simplex (triangle 2D / tet 3D via `iter_top_simplices`, multiphase.py:69–98 — 2D delegates to `hyperct.remesh._quality.iter_triangles_2d`, 3D needs `HC._simplices`, auto-populated via `connect_and_cache_simplices` at 220–226). Key = `_simplex_key` = frozenset of vertex `v.x` tuples (101–107) — order-independent, stable across retriangulation.
- `assign_simplex_phases_from_vertices(HC, dim)` (237–287): post-retopologization path — simplex label = majority vote of bulk vertex phases (`Counter.most_common`, tie → lower phase ID implicitly via Counter ordering); all-interface simplices fall back to `min(interface_phases)` else 0 (286).
- `assign_vertex_phases_from_simplices(HC, dim)` (289–329): vertex phase = unique incident-simplex phase, else `INTERFACE_PHASE`; sets `v.interface_phases`. Isolated vertices default to phase 0 (319–323).

### Interface detection
`identify_interface_from_subcomplex(HC, dim, strict_closure=True)` (333–396):
1. requires `simplex_phase` populated (raises ValueError otherwise, 367–372);
2. calls `extract_interface(HC, simplex_phase, dim)` (`ddgclib/geometry/_interface_subcomplex.py:50–114`): a primal face (edge in 2D, triangle in 3D) is on the interface **iff shared by top-simplices of ≥2 different phases** (line 92: `len(phases) < 2: continue`); populates `HC.interface_vertices`, `HC.interface_edges`, `HC.interface_triangles` (in 3D interface edges = edges of interface triangles, 104–110);
3. `validate_closure` — hard error at init, swallowed (`except ValueError: pass`, 379–382) during runtime retopologization;
4. tags `v.is_interface`; interface vertices get `v.phase = INTERFACE_PHASE` (389–394).

`interface_nn(v, HC)` / `curve_neighbours(v, HC)` (`_interface_subcomplex.py:122–166`): edge-membership lookup in `HC.interface_edges` (exact) with legacy fallback `v.nn & interface_vertices` — legacy is documented as susceptible to spurious K₃ cliques near contact lines (129–132). `curve_neighbours` returns `(None, None)` unless exactly 2 polyline neighbours (164–165).

### `refresh(HC, dim, reset_mass=True, split_method='neighbour_count', criterion_fn=None)` (592–674)
One-call pipeline: relabel simplices (spatial criterion if `reset_mass and fn`, else vertex majority vote — the Lagrangian interface-tracking path, 642–652) → `identify_interface_from_subcomplex(strict_closure=reset_mass)` → `init_phase_fields` (reset) or `_reinit_geometry_fields` (676–690: preserves `m_phase`/`v.m`, zeroes `dual_vol_phase`, `p_phase`, `rho_phase`) → `split_dual_volumes` → `compute_phase_masses` (only if reset) → `compute_phase_pressures`. **`reset_mass=False` during simulation preserves Lagrangian mass.**

---

## 2. Dual-volume splitting at the interface

`MultiphaseSystem.split_dual_volumes(HC, dim, method='neighbour_count')` (multiphase.py:417–503). Bulk vertex: full `v.dual_vol` → own phase (474–476). Interface vertex, two methods:

**`'neighbour_count'` (default)** (477–503): fraction of 1-ring neighbours in each phase, restricted to phases in `v.interface_phases`; interface-to-interface neighbours excluded; if all neighbours are interface → equal split across active phases. O(h) accurate at best — see audit flags.

**`'exact'`** (443–466):
- 2D → `split_dual_polygon_2d(v, n_phases, interface=HC)` (`ddgclib/geometry/_dual_split_2d.py:101–232`): builds the barycentric dual polygon *with edge midpoints* (`_build_typed_polygon`, 52–86, CCW-sorted by polar angle; area check via `hyperct.ddg._dual_cell.dual_cell_area_2d(include_edge_midpoints=True)`, line 98). Interface polyline enters at midpoint(v, v_prev), passes through v, exits at midpoint(v, v_next) — both midpoints are dual-polygon vertices, so the clip is **exact for piecewise-linear interfaces**. Curve neighbours from `HC.interface_edges` (exact) or angular-gap heuristic fallback (132–139). Side phases identified from first bulk midpoint-neighbour on each arc (198–208); 2-phase fixups at 210–223 (e.g. `phase_B = 1 - phase_A`) **hardcode the two-phase assumption**.
- 3D → `split_dual_polyhedron_3d(v, HC, n_phases)` (`_dual_split_2d.py:384–512`; re-exported from `_dual_split_3d.py:16`, which is what multiphase.py:437 docstring cites — actual implementation lives in `_dual_split_2d.py`, the import at multiphase.py:458–461 `from ddgclib.geometry._dual_split_2d import split_dual_polyhedron_3d` is correct): clips the DEC `p_ij` dual polyhedron (from `hyperct.ddg._dual_cell.dual_cell_faces_3d`) against a **local interface tangent plane** at v (normal = smallest-eigenvalue eigenvector of the 1-ring interface neighbour covariance, `_interface_plane_at_3d`, 239+). Plane orientation fixed by voting with bulk-neighbour signed distances (470–482). Fan-tetrahedralized, each tet clipped by the plane (`_clip_tet_by_plane`). **Result rescaled** so the two sub-volumes sum to the authoritative `v.dual_vol` (491–507) because the p_ij polyhedron volume ≠ `v.dual_vol` (v_star) on non-symmetric meshes — ratio preserved, magnitudes adjusted. Exact on locally planar interfaces, O(h²) on curved (multiphase.py:437–439). Multiple degenerate-geometry fallbacks → equal split among active phases (417–428, 434–443, 447–456).

### Per-edge dual-face phase fractions
`edge_phase_area_fractions(v_i, v_j, dim=2, interface=None)` (`_dual_split_2d.py:545–612`) — used by the multiphase stress force. Rules (docstring 557–565):
- bulk–bulk: `{v_i.phase: 1.0}`;
- interface–bulk(k): `{k: 1.0}` (face lies entirely in bulk phase k);
- interface–interface AND primal edge ∈ interface (`_is_curve_adjacent`, 515–542, exact via `frozenset({v_i.x, v_j.x}) in HC.interface_edges`): **fixed 50/50 split** between the two phases (585–593);
- interface–interface interior chord: phase = majority bulk phase of shared 1-ring `v_i.nn & v_j.nn` (600–606), fallback equal split over `v_i.interface_phases`.

---

## 3. Per-phase mass and pressure

- `compute_phase_masses(HC)` (multiphase.py:507–524): `v.m_phase[k] = rho0_k * v.dual_vol_phase[k]` (threshold `vol_k > 1e-30`); `v.m = sum(m_phase)`. Initialization only (mass is Lagrangian afterwards).
- `compute_phase_pressures(HC)` (528–563): `rho_k = m_k / dual_vol_phase_k`; `p_k = eos_k.pressure(rho_k)` for `vol_k, m_k > 1e-30`, else 0. Representative `v.p`: bulk → `p_phase[v.phase]`; interface → **arithmetic mean of nonzero active phase pressures** (555–561). NB the docstring (536–537) says `v.p` stores the "**inner-phase** pressure" for interface vertices — the code averages instead. Stale docstring.
- Lookups: `get_mu(phase_id)` (567–569); `get_gamma(v_i, v_j)` returns 0 for same-phase pairs (573–581); `get_gamma_pair(a, b)` (583–588).

### Mass redistribution: `mass_conserving_merge(HC, cdist=1e-10)` (multiphase.py:711–788)
Merges near-duplicate vertices (cKDTree `query_pairs(cdist)` + union-find). Survivor gets: summed mass (767), mass-weighted momentum → `survivor.u = total_momentum / total_mass` (768–776), plain-arithmetic-mean pressure (772). Removed vertices' neighbours re-wired to survivor (779–785). Returns count removed. **Does NOT touch `m_phase`/`p_phase` arrays** — per-phase state must be rebuilt by `refresh` afterwards.

---

## 4. Phase-aware stress force (`ddgclib/operators/multiphase_stress.py`)

Model (module docstring, multiphase_stress.py:1–55): per vertex,

$$F_i = \sum_{k \in \text{phases\_present}(v)} F_i^{(k)} + F_{st,i}$$

Each phase-k sub-force uses `p_phase[k]` on both face ends (never mixing phases across a face), the phase-k sub-face area `A_ij^{(k)} = \text{frac}_k \cdot A_{ij}`, and viscosity μ_k. Bulk vertices: `phases_present = {v.phase}` → collapses to single-phase `stress_force`, so bulk physics is unchanged (24–27).

`multiphase_stress_force(v, dim=3, mps=None, HC=None, pressure_model=None, curvature_path='integrated')` (107–193):
- `_phases_present(v, n)` (70–80): sorted `v.interface_phases` clipped to `[0, n)`, fallback `[int(v.phase)]`.
- `p_i_by_phase = {k: float(v.p_phase[k])}` (141–142); if no `p_phase`, single fallback via `stress.py:_resolve_pressure` (634–673: None → `v.p`; callable → `fn(v)`; EOS → `P(m/dual_vol)`, updates `v.p`, `v.rho` in place).
- Per neighbour j: `A_ij = dual_area_vector(v, v_j, HC, dim)` (stress.py:52), with optional `HC._edge_area_cache` lookup keyed by `id(v)`/`id(v_j)` (150–159).
- `fractions = edge_phase_area_fractions(v, v_j, dim=dim, interface=HC)` (167–169); phases not in `phases_present` are **skipped** ("bulk-bulk cross-phase edges, a mesh artefact", 171–177).
- Flux primitives (shared with single-phase; stress.py:680–699):
  - pressure: `F_p_ij = -0.5 (p_i^k + p_j^k) A_ij^k` (`pressure_flux`, stress.py:680–682) with `p_j_k = _phase_pressure(v_j, k, fallback=p_i_k)` — **treats a stored value of exactly 0.0 as "missing"** and substitutes the fallback (multiphase_stress.py:83–91).
  - viscous diffusion form: `F_v_ij = (μ_k/|d_ij|) Δu (d̂·A_ij^k)` (`viscous_flux`, stress.py:685–699). Face viscosity `_face_viscosity_for_phase` (94–104) returns plain `mps.get_mu(k)` — module docstring 21–23 promises a harmonic mean `2μ_iμ_j/(μ_i+μ_j)` at μ-jumps that the code **does not implement** (each per-phase sub-face is argued to lie entirely in one phase, so no blending needed; docstring is stale).
- Surface tension added only for `v.is_interface` (187–191).

`multiphase_stress_acceleration = multiphase_dudt_i` (345–363): `a = F/m`, zero when `v.m < 1e-30`. Canonical integrator usage: `partial(multiphase_dudt_i, dim=2, mps=mps, HC=HC, pressure_model=meos)` (49–54). Note the `pressure_model` arg is **not called in the per-phase loop** (130–132) — it is only a fallback when `v.p_phase` is absent; `mps.refresh()` is expected to have populated `p_phase`.

Design rule (docstring 29–33): surface tension stays a separate force on sharp-interface vertices — **do NOT add γκ to the pressure field** (it's already integrated over the dual edge/area).

---

## 5. Surface tension assembly

### Interface vertices (multiphase path): `_interface_surface_tension(v, dim, mps, HC, curvature_path)` (multiphase_stress.py:196–278)
Physical form: `F_st_i = ∫_{Γ_i} γ κ N dS` over the interface portion inside the dual cell. Guards: needs ≥2 interface neighbours and ≥2 phases at v, else zero (234–240). `gamma = mps.get_gamma_pair(sorted(phases)[0], sorted(phases)[1])` (242–243) — **only the first two sorted phases**; triple junctions ignore other pairs.

Three `curvature_path` options:
1. **`'integrated'` (default)**:
   - 2D → `surface_tension_force_2d` (below): FTC on the tangent, `F_st = γ (t_next − t_prev)`. Exact for piecewise-linear curves; static-droplet residual converges **first-order** in mesh spacing on a polygonal approximation of a smooth interface (comment 216–219, plan note 2026-05-06).
   - 3D → `hndA_i_interface(v, interface_nbs ∪ {v}, HC=HC)` then `F = −γ · HNdA[:dim]` (271–274).
2. **`'stokes'`** (250–263): 3D → `integrated_hndA_i_interface` (`_curvatures_heron.py:396+`): direct Stokes boundary integral `F_st_i = γ ∮_{∂Γ_i} ν dl` over the barycentric dual-cell boundary inside interface triangles (segments midpoint→centroid→midpoint per triangle); exact zero on planar interfaces by conormal cancellation (no symmetry needed); on a uniformly refined sphere gives `F_st = −(2γ/R) A_i N` (Young–Laplace). Requires `HC.interface_triangles`; returns zeros if absent (461–463). 2D delegates to the FTC form (already an exact Stokes discretisation).
3. **`'csf_dual'`** (experimental A/B probe, `_csf_dual_surface_tension`, 281–342): FTC/Heron *magnitude* redirected along `S_inner = Σ_j frac_inner(i,j) A_ij` (the dual face-area vector the pressure flux integrates against): `F = γ |Δt| / |S_inner| · S_inner` (338–342). "Interior side" = **higher phase index by convention** (318–325) — arbitrary. Explicitly "not a replacement" (231–232): does not converge to γκN at the same order.
Unknown path → ValueError (265–269).

### 2D curvature estimator (`ddgclib/operators/curvature_2d.py`)
Central identity (docstring 1–38): for a piecewise-linear interface curve,
$$\int_{\Gamma_i} \kappa N\, ds = T(\text{end}) - T(\text{start}) = t_{next} - t_{prev}$$
exact because the tangent is piecewise constant; magnitude `2 sin(θ/2)` (θ = exterior angle, line 102). No area/length prefactor, no pointwise κ sampling. Reconstructs circle arc length/area to machine precision (`dev_notebooks/2D_machine_precision_area_from_curvatures`).
- `_select_curve_neighbours(v, interface_nbs)` (50–88): sorts interface neighbours by polar angle around v, picks the pair bracketing the **largest angular gap**. Heuristic; robust for convex droplets. (The exact subcomplex lookup `curve_neighbours` in `_interface_subcomplex.py` supersedes this where `HC.interface_edges` exists — used by `split_dual_polygon_2d` but **`integrated_curvature_normal_2d` still uses the heuristic**, it never receives HC.)
- `integrated_curvature_normal_2d(v, interface_nbs=None)` (91–149): returns `t_next − t_prev` (2-vector, points toward centre of curvature); zeros on degenerate edges (`< 1e-30`) or unidentifiable neighbours.
- `surface_tension_force_2d(v, gamma, interface_nbs=None)` (152–181): `γ (t_next − t_prev)`.
- `reconstruct_arc_length_and_bulge_area(v_i, v_j, Delta_T)` (184–231): constant-curvature closed form — `c=|v_j−v_i|`, `d=|ΔT|=2sin(θ/2)`, `θ=arccos(1−d²/2)`, `r=c/d`, `L=rθ`, `A_bulge=½r²(θ−sinθ)`. Returns `(c, 0.0, inf, c)` for straight edges.

### 3D curvature estimator (`ddgclib/_curvatures_heron.py`, supporting)
`HNdC_ijk(e_ij, l_ij, l_jk, l_ik)` (93–122): floating-point-stable Heron area `A = ¼√((a+(b+c))(c−(a−b))(c+(a−b))(a+(b−c)))` with sorted lengths; cotan weight `w_ij = ⅛ (l_jk² + l_ik² − l_ij²)/A` (= ½cot θ per edge per triangle); `hnda_ijk = w_ij e_ij`; dual area `c_ijk = ½ |w_ij| l_ij · ½ l_ij`. `hndA_i_interface` (310–393): same cotan stencil restricted to interface neighbours; apex enumeration via `HC.interface_triangles` (`_apex_via_interface_triangles`) or legacy `vi.nn ∩ vj.nn ∩ interface_set` (documented spurious-K₃ hazard, 332–334); 2D inputs zero-padded to 3D (`_pad3`). Contains the inherited `e_ij = -e_ij  # Sign convention (matches hndA_i)` flip (361) — in `hndA_i` (241) it is annotated `# WHY???` — an empirically fixed sign nobody has derived.

### Primal-mesh (thin-film) surface tension (`ddgclib/operators/surface_tension.py`)
For surface meshes (spherical shells, capillary bridges) where `compute_vd` cannot run (docstring 1–21). No dual mesh, no phases:
- `surface_tension_force(v, gamma=0.072, dim=3, HC=None)` (28–61): `F_st = −γ · HNdA_i` from full-mesh `hndA_i`. Default `gamma=0.072` N/m (water–air at ~25 °C).
- `surface_tension_acceleration(v, gamma=0.072, damping=0.0, dim=3, HC=None, **kwargs)` (64–99): `a = (F_st − damping·u)/m`. Drop-in `dudt_fn`; use with `retopologize_fn=False`.
- `dual_area_heron(v, HC=None)` (102–120): returns `C_i` from `hndA_i`.

---

## 6. EOS classes (`ddgclib/eos/`)

Public API (`__init__.py`): `EquationOfState`, `TaitMurnaghan`, `IdealGas`, `MultiphaseEOS`, `eos_pressure_update`.

### `EquationOfState` ABC (`_base.py:9–27`)
Abstract: `pressure(rho)`, `density(P)`, `sound_speed(rho)` ("isentropic speed of sound c = sqrt(dP/drho)"). NB `rho0` is not part of the ABC but is assumed by `_resolve_pressure` (stress.py:667) and `eos_pressure_update` (_update.py:34) for the zero-volume fallback.

### `TaitMurnaghan` (`_tait_murnaghan.py:24–91`)
$$P(\rho) = P_0 + \frac{K}{n}\left[\left(\frac{\rho}{\rho_0}\right)^{n} - 1\right]$$
Defaults: `rho0=1000.0`, `P0=101325.0`, `K=2.15e9` Pa, `n=7.15` (water; B = K/n ≈ 3.007e8 Pa — standard), `rho_clip=(0.9, 1.1)` — **density clipped to ±10 % of ρ0 inside `pressure()` by default** (61–66). Inverse (71–75): `rho = rho0 [(n/K)(P−P0) + 1]^{1/n}` with ratio floored at 1e-30 — **not clipped**, so `density(pressure(rho)) ≠ rho` outside the clip band. `sound_speed` (79–83): `c = √((K/ρ0)(ρ/ρ0)^{n−1})` — also unclipped (c(ρ0) ≈ 1466 m/s for defaults).

### `IdealGas` (`_ideal_gas.py:15–58`)
$$P(\rho) = \rho R_{specific} T,\qquad \rho(P) = P/(R_{specific}T)$$
Defaults: `rho0=1.225`, `T=293.15` K, `R_specific=287.058` J/(kg·K) (dry air), `P0 = rho0·R·T` if None (≈ 103 093 Pa, i.e. ~1.7 % above 1 atm — mind this when mixing with TaitMurnaghan's `P0=101325`). `sound_speed = √(R_specific T)` — explicitly the **isothermal** speed (≈290 m/s), consistent with the isothermal P(ρ) used but contradicting the ABC docstring ("isentropic"; adiabatic air would be √(γRT) ≈ 343 m/s).

### `MultiphaseEOS` (`_multiphase_eos.py:33–96`)
Wrapper dispatching to `eos_list[k]` per phase; implements callable protocol `__call__(v) -> float` for `stress_force(..., pressure_model=...)`. In `__call__` (46–89): if `v.dual_vol_phase` and `v.m_phase` exist, sets `v.rho_phase[k] = m_k/vol_k`, `v.p_phase[k] = eos_k.pressure(rho_k)` for `vol_k, m_k > 1e-30` else zeros; fallback single-phase path uses `v.m / v.dual_vol` with `eos_list[v.phase]` (76–84). Returns and stores own-phase pressure: `own_p = v.p_phase[v.phase]; v.p = own_p` (86–87); `v.rho` similarly (88). `pressure_for_phase(phase_id, rho)` (91–93). **No `v.phase >= 0` guard anywhere** — see audit flags.

### `eos_pressure_update(HC, eos, dim=3)` (`_update.py:7–38`)
Single-phase whole-mesh update: `v.rho = v.m / dual_vol_i`, `v.p = eos.pressure(v.rho)`; `dual_vol` via `stress._get_dual_vol`; zero-volume vertices get `rho0` state. Requires `compute_vd` + cached dual volumes.

---

## 7. Physics-audit flags (suspect / fragile)

1. **`MultiphaseEOS.__call__` violates the INTERFACE_PHASE contract** (`_multiphase_eos.py:81,86–88`): `own_p = v.p_phase[v.phase]` and `self.eos_list[v.phase]` are evaluated unguarded. For an interface vertex (`v.phase == -1`) numpy/list negative indexing silently returns the **last phase's** pressure/EOS — exactly the failure mode the sentinel comment (multiphase.py:59–65) says should be gated. `v.p` on interface vertices then disagrees with `MultiphaseSystem.compute_phase_pressures` (which averages, multiphase.py:555–561). Two different `v.p` conventions for interface vertices coexist.
2. **`_phase_pressure` conflates "0.0" with "missing"** (multiphase_stress.py:83–91): a legitimately zero gauge pressure in phase k is replaced by the fallback `p_i_k`, silently zeroing the pressure-difference flux on that face. Any simulation with gauge pressures crossing 0 gets wrong forces. Same pattern in `compute_phase_pressures`'s interface averaging (`v.p_phase[k] != 0.0` filter, multiphase.py:559).
3. **Stale docstring vs code — face viscosity** (multiphase_stress.py:21–23 vs 94–104): harmonic-mean μ at interface-straddling edges is documented but not implemented; `_face_viscosity_for_phase` returns bare `μ_k` (with a rationale, but the module header still promises the harmonic mean).
4. **Stale docstring — interface `v.p`** (multiphase.py:536–537 says "inner-phase pressure"; code 555–561 computes the mean of active phase pressures).
5. **Momentum non-conservation at skipped fractions** (multiphase_stress.py:171–177): phase-k flux skipped when `k ∉ phases_present(v_i)`, but the neighbour may still include its side of that face → pairwise action–reaction broken on bulk–bulk cross-phase edges. Called a "mesh artefact" but no diagnostic counts occurrences.
6. **Fallback pressure breaks face-value symmetry**: with `p_j_k = fallback = p_i_k`, i computes flux with face pressure `p_i_k`, while j (bulk phase k) computes the shared face using `0.5(p_j + p_i_k)` — antisymmetry holds only if both sides resolve the same pair; the code paths differ (single-phase `stress_force` on bulk j vs multiphase on interface i) and the momentum defect is unquantified.
7. **50/50 interface-edge face split is heuristic even under `method='exact'`** (`_dual_split_2d.py:585–593`): dual *volumes* can be split exactly, but the per-edge *area fractions* the force actually uses are hardwired to 0.5/0.5 for interface-polyline edges — inconsistent with the exact volume clip on asymmetric geometry.
8. **`neighbour_count` volume split (default) is low-order**: per-phase densities ρ_k = m_k/vol_k inherit O(1)-ish volume errors at curved/asymmetric interfaces → spurious per-phase pressure via stiff EOS. With TaitMurnaghan defaults (K/n ≈ 3e8 Pa), a 1 % volume-split error → ~3e6 Pa spurious pressure. Stiff EOS + crude split is the dominant spurious-force generator; the default `rho_clip=(0.9,1.1)` bounds it but then flat-lines dP/dρ.
9. **`TaitMurnaghan.rho_clip` silently saturates the EOS** (`_tait_murnaghan.py:61–66`): outside ±10 % of ρ0, `pressure()` is constant → zero effective sound speed / no restoring force there, while `sound_speed()` and `density()` ignore the clip → mutually inconsistent thermodynamics. A run that hits the clip will look stable but has switched off compressibility physics without warning.
10. **Two-phase assumptions hardcoded**: `split_dual_polygon_2d` uses `1 - phase` fixups (`_dual_split_2d.py:212–223`); `_interface_surface_tension` uses only `sorted(phases)[:2]` for γ (multiphase_stress.py:242–243); `split_dual_polyhedron_3d` uses `active[0]`/`active[1]` (460–466). Triple junctions/3-phase contact lines are outside the modeled physics.
11. **`csf_dual` "inner = higher phase index" convention** (multiphase_stress.py:318–325) is arbitrary — sign of the ST force flips if the droplet is phase 0. Experimental path only, but nothing prevents production use.
12. **Sign-convention debt in the Heron stencil**: `e_ij = -e_ij  # WHY???` (`_curvatures_heron.py:241`), propagated into `hndA_i_interface` (361) as "matches hndA_i". Empirical sign, never derived; any refactor of edge orientation can silently flip surface-tension direction. 3D 'integrated' path applies a further `-gamma` (multiphase_stress.py:274).
13. **Angular-gap curve-neighbour heuristic** (`curvature_2d.py:50–88`) fails for locally concave interfaces or reflex configurations (>π turning) and for >2 interface neighbours picks "the most colinear pair"; `integrated_curvature_normal_2d` never uses the exact `HC.interface_edges` lookup even when available (unlike the dual splitter). Inconsistent neighbour selection between force (heuristic) and volume split (exact).
14. **First-order static-droplet residual** for the 2D FTC surface tension on polygonal approximations of smooth interfaces (multiphase_stress.py:216–219) — parasitic currents shrink only ~O(h).
15. **`mass_conserving_merge` field handling** (multiphase.py:759–777): pressure averaged un-volume-weighted; momentum only summed over vertices that *have* `u` (a merged vertex without `u` silently drops its momentum share... conversely if the survivor lacks `u` the summed momentum is discarded); per-phase arrays (`m_phase` etc.) not merged at all — must call `refresh(reset_mass=False)` afterwards or per-phase mass is stale/inconsistent with `v.m`.
16. **`split_dual_polyhedron_3d` rescale step** (`_dual_split_2d.py:491–507`): the p_ij-polyhedron partition is rescaled to `v.dual_vol`; preserves ratio but means the "exact" 3D split is exact only up to the mismatch between the two dual-volume definitions on non-symmetric meshes.
17. **Interface `v.p` averaging hides the Laplace jump**: for any downstream consumer of `v.p` (visualization, single-phase operators accidentally run on multiphase meshes), the interface pressure is a two-phase average, not either physical side; the actual jump is carried only in `p_phase`.
18. **`_phases_present` fallback can return `[-1]`** (multiphase_stress.py:78–79): an interface vertex lacking `interface_phases` yields `phases_present=[-1]`, then `p_i_by_phase = {-1: v.p_phase[-1]}` — wrap-around read; fractions dict never contains -1 so the flux loop contributes nothing (a silent zero-force vertex), and surface tension still requires ≥2 phases so it is also zero.
19. **Closure-validation swallowing during retopologization** (multiphase.py:379–382): a non-manifold interface after Delaunay is silently accepted (`except ValueError: pass` — not even a warning despite docstring claiming "log a warning", multiphase.py:353–356). Subsequent curvature/split calls on a broken interface degrade without diagnostics.

## 8. Call-graph summary (who calls what)

```
MultiphaseSystem.refresh
 ├─ assign_simplex_phases / assign_simplex_phases_from_vertices
 ├─ identify_interface_from_subcomplex ─ extract_interface, validate_closure  (_interface_subcomplex.py)
 ├─ init_phase_fields / _reinit_geometry_fields
 ├─ split_dual_volumes ─ split_dual_polygon_2d / split_dual_polyhedron_3d  (_dual_split_2d.py)
 ├─ compute_phase_masses            (init only)
 └─ compute_phase_pressures         (per-phase EOS → v.p_phase, v.p)

integrator (symplectic_euler, ...) with dudt_fn = partial(multiphase_dudt_i, dim, mps, HC)
 └─ multiphase_stress_force
     ├─ dual_area_vector, pressure_flux, viscous_flux, _resolve_pressure  (operators/stress.py)
     ├─ edge_phase_area_fractions   (_dual_split_2d.py)
     └─ _interface_surface_tension
         ├─ 2D: surface_tension_force_2d ← integrated_curvature_normal_2d  (curvature_2d.py)
         └─ 3D: hndA_i_interface / integrated_hndA_i_interface  (_curvatures_heron.py)

thin-film (no duals): surface_tension_acceleration ← hndA_i  (operators/surface_tension.py)
```
