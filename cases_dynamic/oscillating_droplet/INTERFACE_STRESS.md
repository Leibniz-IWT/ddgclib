# Interface Stress Computation — Primal Subcomplex Model

## Vertex classification

The sharp interface is an explicit **primal subcomplex**: a closed
polyline of primal edges (2D) or a closed 2-manifold of primal
triangles (3D).  Top-simplices (triangles in 2D, tets in 3D) carry
per-simplex phase labels (`MultiphaseSystem.simplex_phase`); the
interface subcomplex is derived deterministically from these labels.

Every vertex `v` in `HC.V` has exactly one of three roles:

| Role | `v.phase` | `v.is_interface` | Location |
|------|-----------|-------------------|----------|
| **Bulk droplet** | 1 | `False` | Interior of droplet (r << R) |
| **Sharp interface** | -1 (`INTERFACE_PHASE`) | `True` | On the interface subcomplex at r = R |
| **Bulk gas** | 0 | `False` | Exterior (r > R) |

Interface vertices carry `v.phase = INTERFACE_PHASE = -1` — they have
no bulk phase.  `v.interface_phases` gives the frozenset of phases of
their incident top-simplices (typically `{0, 1}`).

## Per-phase vertex data model

Every vertex carries per-phase arrays of length `n_phases`:

| Field | Meaning |
|-------|---------|
| `v.m_phase[k]` | Mass of phase *k* in this dual cell |
| `v.p_phase[k]` | Pressure of phase *k* |
| `v.rho_phase[k]` | Density of phase *k* |
| `v.dual_vol_phase[k]` | Dual cell sub-volume belonging to phase *k* |

For **bulk vertices** only `v.*_phase[v.phase]` is non-zero.
For **interface vertices** multiple entries are non-zero (the dual cell
straddles two phases).

Scalar shortcuts: `v.m = sum(v.m_phase)`, `v.p = v.p_phase[v.phase]`.

## Dual volume splitting

Each dual cell is split among phases.  Currently uses a **neighbour-count
approximation**: fraction of 1-ring neighbours in each phase gives the
volume fraction.  See `MultiphaseSystem.split_dual_volumes`.

Future: exact geometric split by intersecting the dual polygon/polyhedron
with the interface curve/surface (see `FEATURES.md`).

## How `multiphase_stress_force(v)` computes the force

### All vertex types (pressure + viscous)

```
F_i = sum_j [ -0.5*(p_i + p_j)*A_ij  +  (mu/|d_ij|)*du*(d_hat . A_ij) ]
```

- **`p_i`**: own-phase pressure `v.p_phase[v.phase]`.
  For interface vertices this is the inner (droplet) pressure including
  the Young-Laplace jump.
- **`mu`**: own-phase viscosity `mps.phases[v.phase].mu`.
  Each dual cell belongs to a single phase — the viscosity is that phase's
  viscosity, applied uniformly to all edges (including cross-phase edges).
  There is **no harmonic mean**.
- **`A_ij`**: oriented dual area vector (from `dual_area_vector`).

### Sharp interface vertices (additional surface tension)

Surface tension acts **only** on vertices with `v.is_interface == True`:

```
F_st = gamma * kappa * n * dA      (integrated over the dual edge)
```

- **3D**: cotangent-weight Heron curvature restricted to the interface
  sub-mesh (`hndA_i_interface` in `_curvatures_heron.py`).
  `F_st = -gamma * HNdA_i`.
- **2D**: integrated dual curvature from the Fundamental Theorem of
  Calculus applied to the tangent vector of the interface curve
  (`surface_tension_force_2d` in `ddgclib/operators/curvature_2d.py`).
  `F_st = gamma * (t_next - t_prev)` — exact integral of `kappa * N ds`
  over the dual edge for any piecewise linear interface (machine
  precision for constant curvature).

### Total acceleration

```
a_i = F_total / m_i
    = (F_pressure + F_viscous + F_surface_tension) / sum(m_phase)
```

## Per-phase pressure computation (`MultiphaseEOS`)

For each phase *k* present at a vertex:
```
rho_k = m_phase[k] / dual_vol_phase[k]
p_k   = eos_k.pressure(rho_k)
```

**No blending or weighted averages.**  Each phase's pressure is computed
independently from its own density and its own EOS.  Stored in
`v.p_phase[k]`.

The `MultiphaseEOS.__call__(v)` populates all `v.p_phase[k]` and returns
`v.p_phase[v.phase]` for the `_resolve_pressure` protocol.

## Mass on interface vertices

Interface dual cells straddle two phases.  The per-phase mass is:
```
m_phase[k] = rho0_k * dual_vol_phase[k]
```

The total mass `v.m = sum(m_phase)` is constant (Lagrangian).  When the
dual volume changes (mesh deformation), the sub-volumes change and the
densities `rho_k = m_k / vol_k` change, but the masses `m_phase[k]`
remain fixed.

## Initialisation sequence (`_setup.py`)

1. Build mesh (`droplet_in_box_2d/3d`).
2. `ZeroVelocity`.
3. `mps.refresh(HC, dim, reset_mass=True)` — identify interface, init
   per-phase fields, split dual volumes, compute per-phase mass and
   pressure.
4. Apply ellipsoidal perturbation (move interface vertices).
5. `mass_conserving_merge(HC, cdist=1e-10)` — collapse any
   bit-identical duplicates produced by the perturbation.
6. Young-Laplace equilibrium by mass pre-load: compute
   `rho_d_eq = eos_drop.density(p_outer + gamma * kappa)` and set
   `v.m_phase[1] = rho_d_eq * v.dual_vol_phase[1]` for every droplet
   / interface vertex.  This is equivalent to baking the Laplace
   jump into the droplet density; the EOS then produces
   `p_phase[1] = p_outer + gamma * kappa` at equilibrium.  Surface
   tension is still applied as a force every step (see
   `multiphase_stress.py:134-136`), and the compressed-droplet
   pressure balances it — NOT double-counting of the force.
7. `mps.refresh(HC, dim, reset_mass=False)` — recompute per-phase
   pressures from the adjusted densities.

Note: a prior revision of this doc described step 6 as
`v.p_phase[1] += gamma * kappa` (a direct additive pressure bump).
That code path does not exist; the mass-based approach above is what
is actually executed.

## File locations

| Component | File | Key function |
|-----------|------|-------------|
| Phase data model | `ddgclib/multiphase.py` | `MultiphaseSystem`, `PhaseProperties` |
| Per-phase EOS | `ddgclib/eos/_multiphase_eos.py` | `MultiphaseEOS.__call__` |
| Stress + surface tension | `ddgclib/operators/multiphase_stress.py` | `multiphase_stress_force` |
| 2D surface tension | `ddgclib/operators/curvature_2d.py` | `surface_tension_force_2d` |
| 3D surface tension | `ddgclib/_curvatures_heron.py` | `hndA_i_interface` |
| Mesh builder | `ddgclib/geometry/domains/_multiphase_droplet.py` | `droplet_in_box_2d/3d` |
| Case setup | `cases_dynamic/oscillating_droplet/src/_setup.py` | `setup_oscillating_droplet` |
| Retopologization | `ddgclib/dynamic_integrators/_integrators_dynamic.py` | `_retopologize_multiphase` |
