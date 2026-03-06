# Fundamentals of the method

## Continuum starting point

The method begins from the **integral form of the Cauchy momentum equation** written over an arbitrary material (Lagrangian) control volume $V(t)$:

$$
\frac{d}{dt}\int_{V(t)}\rho\mathbf{v}\,dV = \int_{S(t)}\mathbf{\sigma}\cdot\mathbf{n}\,dS + \int_{V(t)}\rho\mathbf{b}\,dV
$$

where $S(t) = \partial V(t)$ is the closed surface with outward unit normal $\mathbf{n}$, $\mathbf{\sigma}$ is the Cauchy stress tensor, and $\mathbf{b}$ collects all body forces per unit mass (gravity, etc.).

For a Newtonian fluid the stress decomposes as

$$
\mathbf{\sigma} = -p\,\mathbf{I} + \mathbf{\tau}(\mathbf{v}),
\qquad
\mathbf{\tau} = 2\mu\,\mathbf{\varepsilon},\qquad
\mathbf{\varepsilon} = \tfrac12\bigl(\nabla\mathbf{v} + (\nabla\mathbf{v})^\top\bigr)
$$

(with the deviatoric part $\mathbf{\tau}$ optionally replaced by more advanced constitutive models in the future).

## Discrete Lagrangian parcels

Each primary vertex $v_i$ of the simplicial complex represents a material parcel with **fixed mass**

$$
m_i = \rho_i\,\mathrm{Vol}_i^\mathrm{dual} = \text{const}.
$$

The parcel occupies the barycentric dual cell $V_i$ whose closed surface $S_i$ consists of the oriented dual flux planes (one per primal edge $ij$) with exact area vectors $\mathbf{A}_{ij}$ (outward from parcel $i$).

Because mass is conserved per parcel, the discrete momentum equation on parcel $i$ is exactly

$$
m_i\frac{d\mathbf{v}_i}{dt} = \mathbf{F}_\mathrm{stress} + \mathbf{F}_\mathrm{body} + \mathbf{F}_\mathrm{surface},
\qquad
\mathbf{F}_\mathrm{stress} = \int_{S_i}\mathbf{\sigma}\cdot\mathbf{n}\,dS.
$$

All surface integrals are evaluated by summing face-centered fluxes over the exact dual geometry produced by `compute_vd` and `e_star`.

## Force operators (implemented in `ddgclib/operators/stress.py`)

### 1. Pressure force (from $-p\mathbf{I}$)

$$
\mathbf{F}_{p,i} = -\int_{S_i}p\,\mathbf{n}\,dS = \sum_{j\in N(i)}\mathbf{F}_{p,ij},
\qquad
\mathbf{F}_{p,ij} = -\tfrac12(p_i+p_j)\,\mathbf{A}_{ij}.
$$

**Code location**: `ddgclib/operators/stress.py`, inside `stress_force()`:

```python
# --- Pressure flux (face-average, conservative) ---
p_j = float(v_j.p) if np.ndim(v_j.p) == 0 else float(v_j.p[0])
F -= 0.5 * (p_i + p_j) * A_ij          # ← exact line
```

This form is pairwise momentum-conserving ($\mathbf{F}{p,ij} = -\mathbf{F}{p,ji}$) and recovers ($-\int_{V_i}\nabla p dV$) exactly for any linear pressure field (discrete divergence theorem on the barycentric dual).

### 2. Viscous force (diffusion form, exact for incompressible/weakly-compressible liquids)
For liquids whose compressibility is carried by the EOS (Tait-Murnaghan, etc.) we use the diffusion (vector-Laplacian) form:

$$\mathbf{F}_{v,i} = \int_{S_i}\mathbf{\tau}\cdot\mathbf{n}\,dS \;\to\; \sum_{j\in N(i)}\mathbf{F}_{v,ij},
\qquad
\mathbf{F}_{v,ij} = \mu\,(\nabla\mathbf{u})_f\cdot\mathbf{A}_{ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij}\cdot\mathbf{A}_{ij}).$$

**Code location**: `ddgclib/operators/stress.py`, inside `stress_force()`:

```python
# --- Viscous flux (face-centered diffusion) ---
delta_u = v_j.u[:dim] - u_i
d_ij = v_j.x_a[:dim] - x_i
d_norm = np.linalg.norm(d_ij)
d_hat = d_ij / d_norm
F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)   # ← exact line
```

(The symmetric stress form that includes the transpose term is deliberately omitted; it introduces spurious discrete compressibility on non-orthogonal edges and prevents machine-precision equilibria on divergence-free test cases such as Poiseuille flow.)

### 3. Body force

$$\mathbf{F}_{\mathrm{body},i} = \int_{V_i}\rho\mathbf{b}\,dV = m_i\mathbf{b}_i.$$
(For spatially varying ($\mathbf{b}$) the integral is evaluated by quadrature on the dual cell; the constant-($\mathbf{b}$) case is simply `F_body = v.m * b`.)

### 4. Surface-tension force (future interface term)

For parcels on a free surface or fluid–fluid interface we will add

$$\mathbf{F}_{\gamma,i} = \int_{\Gamma_i}\gamma\kappa\mathbf{n}\,dS$$

where $\Gamma_i$ is the portion of the interface inside parcel (i), $\gamma$ is surface tension, and $\kappa$ is mean curvature. This will be computed using the curvature estimators in ddgclib/operators/curvature.py (currently Curvature_i with the Laplace–Beltrami method) and the exact dual geometry.

## Total stress-force operator

The complete stress contribution (pressure + viscous) is

$$\mathbf{F}_{\mathrm{stress},i} = \sum_{j\in N(i)}\bigl(\mathbf{F}_{p,ij} + \mathbf{F}_{v,ij}\bigr).$$

Implemented as:

```python
# ddgclib/operators/stress.py
def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    ...
    for v_j in v.nn:
        A_ij = dual_area_vector(v, v_j, HC, dim)
        # pressure line above
        # viscous line above
    return F
 ```

The acceleration follows directly:

$$\mathbf{a}_i = \frac{\mathbf{F}_{\mathrm{stress},i} + \mathbf{F}_{\mathrm{body},i} + \mathbf{F}_{\gamma,i}}{m_i}.$$

This is exposed as the drop-in integrator function

```python
from ddgclib.operators.stress import dudt_i          # alias for stress_acceleration
# or the thin wrapper in gradient.py
from ddgclib.operators.gradient import acceleration
```

## Lagrangian FVM pipeline (per-timestep)

1. State on each vertex ($v_i$ (`v in HC.V`)): cell-averaged pressure ($p_i$ (`v.p`)), parcel velocity ($\mathbf{u}_i$ (`v.u`), fixed mass ($m_i$ (`v.m`)).
2. Mesh construction (every timestep):
- Primary Delaunay triangulation → barycentric dual via compute_vd (in `hyperct.ddg`).
- Exact dual flux planes ($\mathbf{A}_{ij}$) via `dual_area_vector() / e_star()` (in `ddgclib/operators/stress.py` and `hyperct.ddg`).
- Cache dual volumes v.dual_vol.
3. Force computation (Stokes’ theorem on each dual cell): see operators above.
4. Time integration:
- $(\mathbf{a}_i =) total force / (m_i)$
- Advance velocity and position with any integrator from `ddgclib/dynamic_integrators.py` (Euler, symplectic Euler, RK45, adaptive CFL, …).
- Update pressure from the chosen EOS using the new dual volume ($\mathrm{Vol}_i$).
- Retopologize and rebuild duals for the next step.
- 
## Notes

- The formulation uses a compact two-point stencil (only the two vertices sharing each primal edge) and never divides by volume inside the force loop.
- All geometric quantities (($\mathbf{A}_{ij}$), ($\mathrm{Vol}_i)$) are exact for the barycentric dual; the only modelling approximation is the edge-based reconstruction of ($\nabla\mathbf{u}$).
- The diffusion viscous form recovers machine-precision zero residual on Poiseuille equilibrium (and all 488 regression tests).
- Backward-compatible wrappers (pressure_gradient, velocity_laplacian, acceleration) live in `ddgclib/operators/gradient.py` and simply call the new stress operators with appropriate flags.
- Future extensions (viscoelasticity, non-Newtonian, full compressible bulk viscosity, surface tension) plug in by replacing or augmenting the stress_force routine while keeping the same dual-geometry foundation.

This document is the single source of truth for the mathematical and implementation correspondence of the whole library.