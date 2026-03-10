# Fundamentals of the Lagrangian FVM Method

> **Scope.** This document is the single source of truth for the mathematical foundations and implementation of the Lagrangian finite-volume method used throughout `ddgclib`. It describes the governing equations, the discrete operators that implement them, and how they correspond to the source code. Every symbol is defined on first use; no prior knowledge of the codebase is assumed.

---

## Table of Contents

1. [Notation and Conventions](#1-notation-and-conventions)
2. [Continuum Starting Point](#2-continuum-starting-point)
3. [Discrete Lagrangian Parcels](#3-discrete-lagrangian-parcels)
4. [Force Operators](#4-force-operators)
   - [4.1 Pressure Force](#41-pressure-force)
   - [4.2 Viscous Force](#42-viscous-force)
   - [4.3 Body Force](#43-body-force)
   - [4.4 Surface-Tension Force (Future)](#44-surface-tension-force-future)
5. [Total Stress-Force and Acceleration](#5-total-stress-force-and-acceleration)
6. [Per-Timestep Pipeline](#6-per-timestep-pipeline)
7. [Design Notes and Future Extensions](#7-design-notes-and-future-extensions)

---

## 1. Notation and Conventions

| Symbol | Type | Meaning |
|---|---|---|
| $V(t)$ | 3-D region | Lagrangian (material) control volume at time $t$ |
| $S(t) = \partial V(t)$ | closed surface | Boundary of $V(t)$ |
| $\mathbf{n}$ | unit vector | Outward-pointing normal on $S(t)$ |
| $\rho$ | scalar field | Mass density |
| $\mathbf{v}$ | vector field | Continuum velocity |
| $\boldsymbol{\sigma}$ | rank-2 tensor | Cauchy stress tensor |
| $\mathbf{b}$ | vector field | Body force per unit mass (e.g. gravity $\mathbf{g}$) |
| $p$ | scalar field | Thermodynamic pressure |
| $\mathbf{I}$ | rank-2 tensor | Identity tensor |
| $\boldsymbol{\tau}$ | rank-2 tensor | Viscous (deviatoric) stress tensor |
| $\boldsymbol{\varepsilon}$ | rank-2 tensor | Symmetric rate-of-strain tensor |
| $\mu$ | scalar | Dynamic viscosity |
| $v\_i$ | vertex | $i$-th primary vertex of the simplicial complex |
| $V\_i$ | 3-D cell | Barycentric dual cell associated with $v\_i$ |
| $S\_i = \partial V\_i$ | closed surface | Boundary of the dual cell $V\_i$ |
| $m\_i$ | scalar | Fixed mass of parcel $i$ |
| $\rho\_i$ | scalar | Density of parcel $i$ |
| $\mathrm{Vol}\_i^{\mathrm{dual}}$ | scalar | Volume of the barycentric dual cell $V\_i$ |
| $\mathbf{x}\_i$ | vector | Position of vertex $i$ |
| $\mathbf{u}\_i$ | vector | Velocity of parcel $i$ (stored as `v.u`) |
| $p\_i$ | scalar | Pressure of parcel $i$ (stored as `v.p`) |
| $N(i)$ | index set | Set of vertices $j$ sharing a primal edge with $i$ |
| $\mathbf{A}\_{ij}$ | vector | Exact dual-face area vector on the face between parcels $i$ and $j$, pointing outward from $i$ |
| $\mathbf{d}\_{ij}$ | vector | Vector from $\mathbf{x}\_i$ to $\mathbf{x}\_j$; $\mathbf{d}\_{ij} = \mathbf{x}\_j - \mathbf{x}\_i$ |
| $\hat{\mathbf{d}}\_{ij}$ | unit vector | $\mathbf{d}\_{ij} / \lvert\mathbf{d}\_{ij}\rvert$ |
| $\mathbf{F}\_{\star,i}$ | vector | Discrete force of type $\star$ acting on parcel $i$ |

> **Orientation convention.** The area vector $\mathbf{A}_{ij}$ always points *outward* from parcel $i$, so $\mathbf{A}_{ji} = -\mathbf{A}_{ij}$. This guarantees exact pairwise cancellation (Newton's third law) in the discrete setting.

---

## 2. Continuum Starting Point

The method discretises the **integral (weak) form** of the Cauchy momentum equation. For an arbitrary material control volume $V(t)$ that moves with the fluid, Newton's second law reads:

$$
\frac{d}{dt}\int_{V(t)}\rho\mathbf{v}\mathrm{d}V = \underbrace{\int_{S(t)}\boldsymbol{\sigma}\cdot\mathbf{n}\mathrm{d}S}_{\text{surface forces}} + \underbrace{\int_{V(t)}\rho\mathbf{b}\mathrm{d}V}_{\text{body forces}}
$$

Because $V(t)$ moves with the material, there is no convective flux through $S(t)$; the left-hand side is a pure material (Lagrangian) time derivative of momentum.

### Constitutive model — Newtonian fluid

The Cauchy stress tensor is decomposed into an isotropic pressure part and a viscous deviatoric part:

$$
\boldsymbol{\sigma} = -p\mathbf{I} + \boldsymbol{\tau}
$$

For a Newtonian fluid the viscous stress is proportional to the symmetric rate-of-strain tensor:

$$
\boldsymbol{\tau} = 2\mu\boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} = \tfrac{1}{2}\bigl(\nabla\mathbf{v} + (\nabla\mathbf{v})^{\top}\bigr)
$$

where $\nabla\mathbf{v}$ is the velocity gradient tensor and the superscript $\top$ denotes transposition. The factor of $\tfrac{1}{2}$ ensures $\boldsymbol{\varepsilon}$ is the physical strain rate (not twice it). More advanced constitutive models (viscoelastic, non-Newtonian) can replace $\boldsymbol{\tau}$ without altering the rest of the framework.

---

## 3. Discrete Lagrangian Parcels

### Mesh and dual construction

The spatial domain is triangulated as a **simplicial complex** (a Delaunay triangulation in practice). Each primary vertex $v_i$ is associated with a **barycentric dual cell** $V_i$: the region closer to $v_i$ than to any other vertex when distances are measured using barycentric weights. The dual is computed by `compute_vd` (in `hyperct.ddg`).

The boundary $S_i = \partial V_i$ is a closed polyhedral surface composed of one planar **dual face** per primal edge $ij \in N(i)$. The exact area vector of that face,

$$
\mathbf{A}_{ij} = \text{(signed area vector of the dual face between } V_i \text{ and } V_j\text{)},
$$

is computed by `dual_area_vector()` / `e_star()` (in `ddgclib/operators/stress.py` and `hyperct.ddg`). Because the dual faces tile $S_i$ exactly, surface integrals over $S_i$ become exact sums over $N(i)$.

### Fixed-mass parcel

Each parcel $i$ carries a **fixed mass**

$$
m_i = \rho_i\mathrm{Vol}_i^{\mathrm{dual}} = \text{const},
$$

so mass conservation is satisfied identically throughout the simulation — there are no continuity equations to solve. As the parcel moves and deforms, $\rho_i$ and $\mathrm{Vol}_i^{\mathrm{dual}}$ change in tandem while their product stays fixed.

### Discrete momentum equation

Applying the integral momentum balance to the dual cell $V_i$ and using fixed mass gives the **parcel equation of motion**:

$$
m_i\frac{d\mathbf{u}_i}{dt} = \underbrace{\int_{S_i}\boldsymbol{\sigma}\cdot\mathbf{n}\mathrm{d}S}_{\mathbf{F}_{\mathrm{stress},i}} + \mathbf{F}_{\mathrm{body},i} + \mathbf{F}_{\gamma,i}
$$

where $\mathbf{F}_{\gamma,i}$ is the surface-tension contribution (Section 4.4). All surface integrals are evaluated by summing face-centred fluxes over the exact dual geometry.

---

## 4. Force Operators

All operators are implemented in `ddgclib/operators/stress.py`.

### 4.1 Pressure Force

The pressure contribution to the surface integral comes from the $-p\mathbf{I}$ term in the stress:

$$
\mathbf{F}_{p,i}
= -\int_{S_i} p\mathbf{n}\mathrm{d}S = \sum_{j \in N(i)} \mathbf{F}_{p,ij},
\qquad
\mathbf{F}_{p,ij} = -\tfrac{1}{2}(p_i + p_j)\mathbf{A}_{ij}
$$

The face-average pressure $\tfrac{1}{2}(p_i + p_j)$ is a linear interpolation of the two vertex values, which is exact for any linear pressure field. The resulting force is **pairwise momentum-conserving**:

$$
\mathbf{F}_{p,ij} = -\mathbf{F}_{p,ji}
\quad \Longleftrightarrow \quad
\text{Newton's third law holds exactly.}
$$

By the discrete divergence theorem on the barycentric dual, this scheme recovers $-\int_{V_i}\nabla p\mathrm{d}V$ exactly for linear $p$.

**Source code** (inside `stress_force()`):

```python
# --- Pressure flux (face-average, conservative) ---
p_j = float(v_j.p) if np.ndim(v_j.p) == 0 else float(v_j.p[0])
F -= 0.5 * (p_i + p_j) * A_ij
```

---

### 4.2 Viscous Force

For liquids whose compressibility is handled entirely by an equation of state (e.g. Tait–Murnaghan), the deviatoric stress integral reduces to a **vector-Laplacian diffusion** form. The viscous flux across the dual face between parcels $i$ and $j$ is approximated using a two-point gradient:

$$
\mathbf{F}_{v,ij} = \mu(\nabla\mathbf{u})_f \cdot \mathbf{A}_{ij} = \frac{\mu}{\lvert\mathbf{d}_{ij}\rvert}
  \underbrace{(\mathbf{u}_j - \mathbf{u}_i)}_{\Delta\mathbf{u}}
  \underbrace{\bigl(\hat{\mathbf{d}}_{ij}\cdot\mathbf{A}_{ij}\bigr)}_{\text{projected area}}
$$

**Symbol glossary for this expression:**

| Symbol | Meaning |
|---|---|
| $\mu$ | Dynamic viscosity of the fluid |
| $(\nabla\mathbf{u})\_f$ | Face-centred approximation of the velocity gradient |
| $\mathbf{d}\_{ij} = \mathbf{x}\_j - \mathbf{x}\_i$ | Edge vector pointing from parcel $i$ to parcel $j$ |
| $\lvert\mathbf{d}\_{ij}\rvert$ | Euclidean length of the edge |
| $\hat{\mathbf{d}}\_{ij} = \mathbf{d}\_{ij}/\lvert\mathbf{d}\_{ij}\rvert$ | Unit vector along the edge |
| $\Delta\mathbf{u} = \mathbf{u}\_j - \mathbf{u}\_i$ | Velocity difference across the edge |
| $\hat{\mathbf{d}}\_{ij}\cdot\mathbf{A}\_{ij}$ | Projection of the dual-face area onto the edge direction |

> **Why not the full symmetric form?** The complete Newtonian viscous stress includes a transpose term $(\nabla\mathbf{v})^{\top}$. Including it on non-orthogonal edges introduces a spurious discrete compressibility and prevents machine-precision equilibria on divergence-free flows such as Poiseuille flow. The diffusion form above recovers exact zero residual on all such benchmarks (verified by 488 regression tests).

**Source code** (inside `stress_force()`):

```python
# --- Viscous flux (face-centred diffusion) ---
delta_u  = v_j.u[:dim] - u_i
d_ij     = v_j.x_a[:dim] - x_i
d_norm   = np.linalg.norm(d_ij)
d_hat    = d_ij / d_norm
F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)
```

The full viscous force on parcel $i$ is:

$$
\mathbf{F}_{v,i} = \sum_{j \in N(i)} \mathbf{F}_{v,ij}
$$

---

### 4.3 Body Force

For a body force field $\mathbf{b}$ (acceleration per unit mass, e.g. $\mathbf{b} = \mathbf{g}$ for gravity), the volume integral over the dual cell gives:

$$
\mathbf{F}_{\mathrm{body},i} = \int_{V_i} \rho\mathbf{b}\mathrm{d}V = m_i\mathbf{b}_i
$$

The last equality holds exactly when $\mathbf{b}$ is uniform over $V_i$ (the constant-body-force case):

```python
F_body = v.m * b   # v.m is the fixed parcel mass; b is the body-force vector
```

For a spatially varying body force, the integral is evaluated by quadrature over the dual cell using the exact geometry.

---

### 4.4 Surface-Tension Force (Future Extension)

For parcels lying on a free surface or a fluid–fluid interface, a surface-tension term will be added:

$$
\mathbf{F}_{\gamma,i} = \int_{\Gamma_i} \gamma\kappa\mathbf{n}\mathrm{d}S
$$

| Symbol | Meaning |
|---|---|
| $\Gamma\_i$ | Portion of the interface contained within parcel $V\_i$ |
| $\gamma$ | Surface-tension coefficient |
| $\kappa$ | Mean curvature of the interface (positive for a convex surface) |
| $\mathbf{n}$ | Interface outward normal |

Curvature $\kappa$ will be estimated using the Laplace–Beltrami method in `ddgclib/operators/curvature.py` (`Curvature_i`), combined with the exact dual geometry. This term is zero in the current implementation.

---

## 5. Total Stress-Force and Acceleration

The complete stress contribution on parcel $i$ combines pressure and viscous parts:

$$
\mathbf{F}_{\mathrm{stress},i} = \sum_{j \in N(i)} \bigl(\mathbf{F}_{p,ij} + \mathbf{F}_{v,ij}\bigr)
$$

The total force is:

$$
\mathbf{F}_{\mathrm{total},i} = \mathbf{F}_{\mathrm{stress},i} + \mathbf{F}_{\mathrm{body},i} + \mathbf{F}_{\gamma,i}
$$

Dividing by the fixed parcel mass gives the acceleration:

$$
\mathbf{a}_i = \frac{\mathbf{F}_{\mathrm{total},i}}{m_i}
$$

**Full operator skeleton** (`ddgclib/operators/stress.py`):

```python
def stress_force(v, dim: int = 3, mu: float = 8.9e-4, HC=None) -> np.ndarray:
    """Return the total stress force vector on parcel v."""
    F   = np.zeros(dim)
    x_i = v.x_a[:dim]
    u_i = v.u[:dim]
    p_i = float(v.p) if np.ndim(v.p) == 0 else float(v.p[0])

    for v_j in v.nn:                          # loop over neighbours j ∈ N(i)
        A_ij = dual_area_vector(v, v_j, HC, dim)

        # 4.1 Pressure flux
        p_j = float(v_j.p) if np.ndim(v_j.p) == 0 else float(v_j.p[0])
        F  -= 0.5 * (p_i + p_j) * A_ij

        # 4.2 Viscous flux (diffusion form)
        delta_u = v_j.u[:dim] - u_i
        d_ij    = v_j.x_a[:dim] - x_i
        d_norm  = np.linalg.norm(d_ij)
        d_hat   = d_ij / d_norm
        F      += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)

    return F
```

The parcel acceleration is exposed through two convenience aliases:

```python
from ddgclib.operators.stress   import dudt_i       # alias: stress_acceleration
from ddgclib.operators.gradient import acceleration  # thin wrapper, same result
```

---

## 6. Per-Timestep Pipeline

The following sequence is executed once per timestep for every parcel $v_i$ (`v in HC.V`).

### Step 1 — State variables (beginning of timestep)

Each vertex `v` stores:

| Attribute | Symbol | Description |
|---|---|---|
| `v.p` | $p\_i$ | Cell-averaged pressure |
| `v.u` | $\mathbf{u}\_i$ | Parcel velocity vector |
| `v.m` | $m\_i$ | Fixed parcel mass (never modified) |

### Step 2 — Mesh construction

1. Compute the primary Delaunay triangulation.
2. Build the barycentric dual via `compute_vd` (`hyperct.ddg`).
3. Compute exact dual-face area vectors $\mathbf{A}_{ij}$ via `dual_area_vector()` / `e_star()`.
4. Cache dual cell volumes as `v.dual_vol` ($\mathrm{Vol}_i^{\mathrm{dual}}$).

### Step 3 — Force computation

Apply Stokes' theorem on each dual cell as described in Section 4 to obtain $\mathbf{F}_{\mathrm{total},i}$ for every parcel.

### Step 4 — Time integration

$$
\mathbf{a}_i = \frac{\mathbf{F}_{\mathrm{total},i}}{m_i}
$$

Advance velocity and position using any integrator from `ddgclib/dynamic_integrators.py`:

| Integrator | Notes |
|---|---|
| Forward Euler | First-order; simple but dissipative |
| Symplectic Euler | First-order; energy-conserving for Hamiltonian systems |
| RK45 | Fourth/fifth-order; adaptive step recommended |
| Adaptive CFL | Timestep chosen to satisfy the CFL stability condition |

### Step 5 — Equation of state update

Recompute $p_i$ from the new dual volume $\mathrm{Vol}_i^{\mathrm{dual}}$ using the chosen equation of state (e.g. Tait–Murnaghan for liquids).

### Step 6 — Retopologisation

Rebuild the Delaunay triangulation and dual geometry for the updated positions, ready for the next timestep.

---

## 7. Design Notes and Future Extensions

### Compact two-point stencil

The force loop uses only the two vertices sharing each primal edge. This keeps the stencil compact, avoids division by volume inside the loop, and ensures that all geometric quantities ($\mathbf{A}_{ij}$, $\mathrm{Vol}_i^{\mathrm{dual}}$) are exact for the barycentric dual. The only modelling approximation is the edge-based reconstruction of $\nabla\mathbf{u}$ in the viscous term.

### Conservation properties

| Property | Status |
|---|---|
| Mass conservation (per parcel) | Exact — $m\_i = \text{const}$ by construction |
| Momentum conservation (pairwise) | Exact — $\mathbf{F}\_{p,ij} + \mathbf{F}\_{p,ji} = \mathbf{0}$ |
| Linear pressure recovery | Exact — by the discrete divergence theorem on the barycentric dual |
| Poiseuille equilibrium residual | Machine precision — verified by 488 regression tests |

### Backward-compatible API

Legacy callers are supported via thin wrappers in `ddgclib/operators/gradient.py` that simply forward to the new stress operators with appropriate flags:

```python
from ddgclib.operators.gradient import pressure_gradient    # → stress_force(pressure_only=True)
from ddgclib.operators.gradient import velocity_laplacian   # → stress_force(viscous_only=True)
from ddgclib.operators.gradient import acceleration         # → stress_force() / m
```

### Planned extensions

| Feature | Entry point |
|---|---|
| Surface tension | Replace `F_gamma = 0` with the integral in Section 4.4 |
| Non-Newtonian / viscoelastic fluids | Replace or augment the viscous flux $\mathbf{F}\_{v,ij}$ |
| Full compressible bulk viscosity | Add the $\tfrac{2}{3}\mu(\nabla\cdot\mathbf{v})\mathbf{I}$ term to $\boldsymbol{\tau}$ |

All extensions plug in by modifying `stress_force()` while the dual-geometry foundation (Steps 1–2 of the pipeline) remains unchanged.
