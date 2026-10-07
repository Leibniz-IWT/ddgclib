# Code Map: Core Stress Operators (`ddgclib/operators/`)
> Sources: ddgclib/operators/stress.py, ddgclib/operators/gradient.py, ddgclib/operators/__init__.py, ddgclib/operators/_registry.py | Written: 2026-07-02 by understand-and-document workflow

Physics core of the Lagrangian FVM: forces on each Lagrangian parcel (dual cell of vertex `v`)
are surface integrals over dual flux planes (Stokes' theorem):

$$F_i = \sum_j (F^p_{ij} + F^v_{ij}), \qquad
F^p_{ij} = -\tfrac{1}{2}(p_i + p_j)\,A_{ij}, \qquad
F^v_{ij} = \frac{\mu}{|d_{ij}|}\,\Delta u\,(\hat d \cdot A_{ij})$$

with $\Delta u = u_j - u_i$, $d_{ij} = x_j - x_i$. The viscous term is the **diffusion form**
($\mu \nabla^2 u$); the transpose term of $\nabla\cdot[\mu(\nabla u + \nabla u^T)]$ is
deliberately omitted because the rank-1 face gradient has spurious discrete compressibility
on non-orthogonal edges (stress.py:17-24, 720-724).

Vertex attribute prerequisites: `v.p` (pressure), `v.u` (velocity ndarray), `v.m` (mass),
`v.nn` (1-ring), `v.vd` (dual vertices, set by `hyperct.ddg.compute_vd`), optionally
`v.dual_vol` (cached) and `v.rho`.

---

## stress.py — public API

File: `/home/endres/projects/ddgclib/ddgclib/operators/stress.py` (847 lines)

### Geometry

#### `dual_area_vector(v_i, v_j, HC, dim: int = 3) -> np.ndarray` (stress.py:52)
Oriented dual area vector $A_{ij} = \int_{S_{ij}} n\,dS$ of the dual face between parcels i
and j, outward from `v_i`. Shape `(dim,)`. Branches:

- **dim=1** (stress.py:91-95): signed direction only —
  `return np.array([np.sign(v_j.x_a[0] - v_i.x_a[0])])`.
- **dim=2, periodic path** (stress.py:97-156): if `getattr(HC, '_periodic_axes', None)` is
  truthy, **ignores `v.vd` entirely** and rebuilds the dual edge from primal geometry with
  minimum-image coordinates (`round(delta/p)*p` wrap, stress.py:107-113), because
  "compute_vd's dual vertex positions are wrong for any triangle that includes a
  periodic-face vertex with wrapped neighbors" (stress.py:98-101).
  - `common = v_i.nn.intersection(v_j.nn)`; if `len(common) < 1` → `np.zeros(2)`;
    if exactly 1 (boundary edge) → dual edge from triangle barycenter to primal edge
    midpoint (stress.py:118-130).
  - Interior edge: shared-neighbor count can be >2 under ghost resolution, so one neighbor
    is picked on each side of the edge via cross-product sign (stress.py:131-147);
    if either side missing → `np.zeros(2)`.
  - Normal = 90° rotation of dual edge: `A_ij = np.array([-dual_edge[1], dual_edge[0]])`
    (stress.py:126, 151), flipped if `np.dot(A_ij, vec_to_i) > 0`.
- **dim=2, standard path** (stress.py:158-173): `vdnn = v_i.vd.intersection(v_j.vd)`;
  if `< 2` shared dual vertices (boundary edge) → **`np.zeros(2)`** (stress.py:161-163) —
  boundary edges contribute zero flux. Otherwise
  `dual_edge = vd2.x_a[:2] - vd1.x_a[:2]`; `A_ij = [-dual_edge[1], dual_edge[0]]`;
  orient outward by flipping when `dot(A_ij, x_i - centroid) > 0` (stress.py:169-172).
- **dim=3** (stress.py:175-176): delegates to `_dual_area_vector_3d_p_ij`.
- **else**: `NotImplementedError` (stress.py:179).

TODO at stress.py:49, 89: move `dual_area_vector`/`dual_volume` to `hyperct.ddg._operators`
(pure geometry, no physics).

#### `_dual_area_vector_3d_p_ij(v_i, v_j, HC)` (stress.py:182) — private, primary 3D path
DEC p_ij face construction: polygon interleaving **tet barycenters** (shared dual vertices)
with **face barycenters** `(x_i + x_j + x_k)/3`. This gives **linear precision (machine eps)
for barycentric duals on any tet mesh**; a polygon of tet barycenters alone does not,
because the non-planar face triangulates to a wrong area vector (stress.py:189-193).

Traversal of hyperct dual structures:
1. `shared_vd = v_i.vd.intersection(v_j.vd)`; if `< 3` → fall back to
   `_dual_area_vector_3d_e_star` (stress.py:199-202).
2. Ring-walk over `shared_vd` using **dual vertex connectivity** `curr.nn`
   (stress.py:204-217). If the walk is incomplete (boundary-adjacent edges with truncated
   dual connectivity), retry with `hyperct.ddg._dual_cell._angular_sort_3d(shared_list)`
   (stress.py:220-229); if still `< 3` ring points → e_star fallback.
3. Face barycenters come from `common_nbs = v_i.nn.intersection(v_j.nn)` (opposite vertices
   of shared triangular faces); for each consecutive tet-barycenter pair the chosen
   $x_k$ is the common neighbor whose face barycenter is nearest the midpoint of the two
   tet barycenters (stress.py:246-260) — a heuristic nearest-match, not a topological lookup.
4. Area vector by centroid-fan triangulation:
   `A_ij += 0.5 * np.cross(p1 - centroid, p2 - centroid)` (stress.py:264-271).
5. Orientation: flip if `np.dot(A_ij, x_j - x_i) < 0` (stress.py:274-276) — outward = along
   the primal edge direction (note: differs from the 2D orientation test which uses the
   dual-face centroid).

#### `_dual_area_vector_3d_e_star(v_i, v_j, HC)` (stress.py:280) — legacy 3D fallback
Uses `hyperct.ddg.e_star(v_i, v_j, HC, dim=3)` fan-walk from the primal edge midpoint.
Explicitly documented as **NOT linearly precise** for barycentric duals on non-symmetric
meshes (stress.py:285-289). Kept only for boundary/degenerate topologies. Details:
- `e_star` failure (`IndexError`, `KeyError`) or empty array → `np.zeros(3)` (stress.py:294-299).
- Looks up the dual **edge-midpoint vertex** directly in the dual cache by coordinate key:
  `vc_12 = HC.Vd[tuple(0.5*(v_j.x_a - v_i.x_a) + v_i.x_a)]` (stress.py:300-301) — this is the
  only place `HC.Vd` (dual vertex cache keyed by coordinate tuple) is touched in this module.
- Per-facet orientation: flip each `A_ijk` when `dot(A_ijk, v_i.x_a - vc_12.x_a) > 0`, then sum.

#### `dual_volume(v, HC, dim: int = 3) -> float` (stress.py:311)
Dual cell measure Vol_i (2D: area, 3D: volume). Branches:
- **dim=1** (stress.py:339-349): interval `max - min` over `v.vd` positions; boundary vertex
  with single dual vertex → half-edge `0.5*abs(v_j.x - v.x)` using an arbitrary
  `next(iter(v.nn))` neighbor; no duals → `0.0`.
- **dim=2** (stress.py:351-353): delegates to
  `hyperct.ddg.dual_cell_area_2d(v, include_edge_midpoints=True)`.
- **dim=3** (stress.py:355-369): sums per-edge wedge volumes via
  `hyperct.ddg.v_star(v, v_j, HC, dim=3)` over all `v_j in v.nn`; expects a
  `(_, V_ij)` tuple and accumulates `np.sum(np.abs(V_ij))`; scalar returns are cast to
  float ("shouldn't happen in 3D", stress.py:364-366); per-edge `KeyError/IndexError/
  ValueError` are silently skipped with `continue` (stress.py:367-368) — degenerate edges
  simply under-count the volume.
- **else**: `NotImplementedError` (stress.py:372).

#### `cache_dual_volumes(HC, dim: int = 3) -> None` (stress.py:379)
Sets `v.dual_vol = dual_volume(v, HC, dim)` for every `v in HC.V`. Meant to be called after
`compute_vd` (e.g. inside `_retopologize`). Degenerate vertices (raising
`ValueError/IndexError`, e.g. domain corners) get `v.dual_vol = 0.0` (stress.py:394-399).

#### `_get_dual_vol(v, HC, dim=3) -> float` (stress.py:402) — private
Returns `v.dual_vol` if present, else computes and caches it lazily (AttributeError path).

### Physics

#### `velocity_difference_tensor(v, HC, dim: int = 3) -> np.ndarray` (stress.py:415)
Integrated (NOT volume-divided) velocity difference tensor, shape `(dim, dim)`:
$$Du_i = \tfrac{1}{2}\sum_j (u_j - u_i)\otimes A_{ij} \approx \int_{V_i}\nabla u\,dV$$
Component convention `Du_i[a, b] = 0.5 * sum_j (u_j^a - u_i^a) * A_ij^b` (stress.py:443).
Implementation: `Du_i += np.outer(delta_u, A_ij)` then `Du_i *= 0.5` (stress.py:455-456).
Uses the **edge-area cache** `HC._edge_area_cache[id(v)][id(v_j)]` when present
(stress.py:445-453), falling back to `dual_area_vector` per edge. Cache is populated by
`batch_e_star(..., orient=True)` during retopologization (per comment at stress.py:757-758)
and is keyed by Python `id()` — stale after vertex objects are replaced unless the cache is
rebuilt.

#### `velocity_difference_tensor_pointwise(v, HC, dim=3) -> np.ndarray` (stress.py:460)
`Du_i / Vol_i` (volume-averaged pointwise gradient). Returns `np.zeros((dim, dim))` if
`Vol_i < 1e-30` (stress.py:481-482). Diagnostic / analytical comparison only.

#### `scalar_gradient_integrated(v, HC, dim=3, field_attr='f') -> np.ndarray` (stress.py:486)
Scalar analog: $Df_i = \tfrac12\sum_j (f_j - f_i)\,A_{ij} \approx \int_{V_i}\nabla f\,dV$
(stress.py:523-530). Field read via `getattr(v, field_attr)`. Same edge-area cache logic.
**Note:** exported from neither `operators/__init__.py` `__all__` nor its import list —
callers must import from `ddgclib.operators.stress` directly.

#### `strain_rate(du: np.ndarray) -> np.ndarray` (stress.py:534)
$\varepsilon = \tfrac12(du + du^T)$; `return 0.5 * (du + du.T)` (stress.py:555). Pure ndarray
helper, no mesh.

#### `cauchy_stress(p, du, mu, dim=3) -> np.ndarray` (stress.py:558)
Pointwise Newtonian constitutive relation
$\sigma = -pI + 2\mu\varepsilon$: `return -p * np.eye(dim) + 2.0 * mu * strain_rate(du)`
(stress.py:592).

#### `integrated_cauchy_stress(p, Du, mu, Vol_i, dim=3) -> np.ndarray` (stress.py:595)
Volume-integrated stress $\Sigma = -p\,Vol_i\,I + 2\mu\,\varepsilon(Du)$:
`return -p * Vol_i * np.eye(dim) + 2.0 * mu * strain_rate(Du)` (stress.py:631). Pointwise
`cauchy_stress` recovered by dividing by `Vol_i`. Diagnostic only — **`stress_force` does
not assemble either of these tensors**; it uses the factored face fluxes directly.

#### `_resolve_pressure(v, pressure_model, HC, dim)` (stress.py:634) — private
Three pressure modes:
- `None`: read `v.p` as-is; handles 0-d scalars and length-1 arrays:
  `float(p) if np.ndim(p) == 0 else float(p[0])` (stress.py:656-657).
- plain callable (detected as `callable(...) and not hasattr(pressure_model, 'pressure')`,
  stress.py:659): `float(pressure_model(v))`.
- `EquationOfState` (has `.pressure`): weakly compressible;
  `rho = v.m / _get_dual_vol(v, HC, dim)`; `p = eos.pressure(rho)`;
  **side effect**: writes `v.rho` and `v.p` in-place (stress.py:670-672).
  Guard: `vol < 1e-30` → `eos.pressure(eos.rho0)` (stress.py:666-667).

#### `pressure_flux(p_i, p_j, A_ij) -> np.ndarray` (stress.py:680)
One-liner: `return -0.5 * (p_i + p_j) * A_ij` (stress.py:682). Face-average, conservative
(equal-and-opposite across each face by construction).

#### `viscous_flux(mu, delta_u, d_ij, A_ij) -> np.ndarray` (stress.py:685)
`return (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)` (stress.py:699); coincident vertices
(`d_norm < 1e-30`) → zeros (stress.py:696-697). These two primitives are factored out to be
shared with `multiphase_stress` (stress.py:676-677 section comment).

#### `stress_force(v, dim=3, mu=8.9e-4, HC=None, pressure_model=None) -> np.ndarray` (stress.py:702)
The core force loop (stress.py:753-775):
```python
p_i = _resolve_pressure(v, pressure_model, HC, dim)
...
for v_j in v.nn:
    A_ij = _cache[...] or dual_area_vector(v, v_j, HC, dim)
    p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
    delta_u = v_j.u[:dim] - u_i
    d_ij = v_j.x_a[:dim] - x_i
    F += pressure_flux(p_i, p_j, A_ij)
    F += viscous_flux(mu, delta_u, d_ij, A_ij)
```
Default `mu=8.9e-4` Pa·s (water at 25 °C). Note: with `_periodic_axes` set, `d_ij` here is
**not** minimum-imaged (only `A_ij` is, inside `dual_area_vector`) — wrapped-neighbor viscous
flux uses the raw coordinate difference. Same applies to the periodic boundary-edge case.

#### `stress_acceleration(v, dim=3, mu=8.9e-4, HC=None, pressure_model=None) -> np.ndarray` (stress.py:778)
`return stress_force(...) / v.m` (stress.py:824-825). Newton's 2nd law per parcel. Canonical
`dudt_fn` — bind with `functools.partial(stress_acceleration, dim=2, mu=1e-3, HC=HC)`
(binding via `**dudt_kwargs` causes "multiple values for HC" per project CLAUDE.md).

#### `dudt_i = stress_acceleration` (stress.py:829) — alias
Module docstring at stress.py:838-840 shows passing `HC=HC` via integrator-forwarded kwargs;
this contradicts the CLAUDE.md instruction to always use `partial` — treat the docstring
example as stale.

---

## gradient.py — thin legacy wrappers

File: `/home/endres/projects/ddgclib/ddgclib/operators/gradient.py` (136 lines).
Header (gradient.py:4-10): old scalar-area API kept as special cases of the tensor pipeline.

- `pressure_gradient(v, dim=3, HC=None)` (gradient.py:34):
  `return stress_force(v, dim=dim, mu=0.0, HC=HC)` (gradient.py:59). Despite the name it
  returns the integrated pressure **force** (sign/scale: $-\sum_j \tfrac12(p_i+p_j)A_{ij}$,
  i.e. $\approx -\int \nabla p\,dV$), not a gradient.
- `velocity_laplacian(v, dim=3, HC=None)` (gradient.py:62): reimplements the viscous loop
  inline with `mu=1` (gradient.py:89-99), i.e. $\sum_j \frac{1}{|d_{ij}|}\Delta u\,(\hat d\cdot A_{ij})
  \approx \int \nabla^2 u\,dV$. Does NOT use the `_edge_area_cache` (unlike `stress_force`) —
  slower on cached meshes, but numerically identical per-edge formula.
- `acceleration(v, dim=3, mu=8.9e-4, HC=None)` (gradient.py:104):
  `return stress_acceleration(v, dim=dim, mu=mu, HC=HC)` (gradient.py:135). No
  `pressure_model` passthrough — EOS flows must call `stress_acceleration` directly.

---

## __init__.py and _registry.py (skim)

`/home/endres/projects/ddgclib/ddgclib/operators/__init__.py` (79 lines): re-exports the
whole operator surface. From `stress`: `dual_area_vector, dual_volume, cache_dual_volumes,
velocity_difference_tensor, velocity_difference_tensor_pointwise, strain_rate,
cauchy_stress, integrated_cauchy_stress, stress_force, stress_acceleration, dudt_i`
(lines 20-32). From `gradient`: `pressure_gradient, velocity_laplacian, acceleration`
(lines 33-37). Also exports sibling modules relevant to the stress pipeline:
`surface_tension` (`surface_tension_force`, `surface_tension_acceleration`,
`dual_area_heron`), `curvature_2d` (`integrated_curvature_normal_2d`,
`surface_tension_force_2d`, `reconstruct_arc_length_and_bulge_area`), `multiphase_stress`
(`multiphase_stress_force`, `multiphase_stress_acceleration`, `multiphase_dudt_i`),
`mass_redistribution` (snapshot/redistribute functions), plus registry-based
`Curvature_i/ijk`, `Area_i/ijk/Area/DualArea_i`, `Volume/Volume_i`.
`scalar_gradient_integrated` is missing from both the import and `__all__` (lines 61-78).

`/home/endres/projects/ddgclib/ddgclib/operators/_registry.py` (45 lines):
`MethodRegistry(name)` — a plain `dict[str, Callable]` wrapper with `register(key, fn)`,
`__getitem__` (KeyError listing available methods), `__contains__`, `available()`. Used by
the curvature/area/volume method-wrapper machinery, **not** by the stress pipeline (stress
functions are called directly, no registry indirection).

---

## Special-casing summary (quick reference)

| Situation | Behavior | Location |
|---|---|---|
| dim=1 area vector | signed ±1 scalar direction | stress.py:91-95 |
| 2D boundary edge (<2 shared `vd`) | `A_ij = zeros(2)` (zero flux) | stress.py:161-163 |
| 2D periodic mesh (`HC._periodic_axes`) | bypass `v.vd`, min-image primal reconstruction | stress.py:102-156 |
| 2D periodic, <1 common nn | `zeros(2)` | stress.py:119-120 |
| 2D periodic, missing left/right triangle | `zeros(2)` | stress.py:146-147 |
| 3D <3 shared `vd` | e_star fallback | stress.py:200-202 |
| 3D incomplete ring-walk | `_angular_sort_3d` retry, then e_star fallback | stress.py:220-229 |
| 3D no common primal neighbors | e_star fallback | stress.py:232-234 |
| 3D e_star raises / empty | `zeros(3)` | stress.py:294-299 |
| 1D dual_volume, 1 dual vertex | half-edge to arbitrary neighbor | stress.py:343-346 |
| 3D dual_volume per-edge exception | silently `continue` (under-count) | stress.py:367-368 |
| `cache_dual_volumes` degenerate vertex | `v.dual_vol = 0.0` | stress.py:397-399 |
| `Vol_i < 1e-30` (pointwise tensor) | zeros | stress.py:481-482 |
| `vol < 1e-30` (EOS pressure) | `eos.pressure(eos.rho0)` | stress.py:666-667 |
| `d_norm < 1e-30` (viscous flux) | zeros | stress.py:696-697 |
| dim not in {1,2,3} | `NotImplementedError` | stress.py:179, 372 |

## TODOs / known-problem markers

- stress.py:32-41: constitutive TODOs — viscoelastic (Maxwell/Oldroyd-B), non-Newtonian
  (power-law/Carreau `mu_eff = K*|strain_rate|^(n-1)`), Hookean elastic solid, surface
  tension stress `sigma += gamma*(I - n⊗n)*kappa` (surface tension currently lives in the
  separate `operators/surface_tension.py` / `curvature_2d.py` instead).
- stress.py:49, 89, 337: move `dual_area_vector`/`dual_volume` to `hyperct.ddg._operators`.
- stress.py:98-101: acknowledged bug in `compute_vd` for periodic meshes (dual positions
  wrong near periodic faces) — worked around locally, not fixed upstream.
- stress.py:285-289: e_star 3D path acknowledged as not linearly precise (why p_ij exists).
- Transpose viscous term intentionally omitted (stress.py:19-24) — full symmetric stress
  divergence is only recovered in the incompressible limit.
- No commented-out dead code found in any of the four files.
