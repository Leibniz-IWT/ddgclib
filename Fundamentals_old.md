
# Fundamentals of the method

## Lagrangian FVM pipeline

### Per-timestep computation cycle

1. **State**: Each vertex $v$ stores pressure $p_i$ (average in FVM) and velocity $\mathbf{u}_i$ (exact velocity of the finite volume element dual to the vertex), as well as mass $m_i$.

2. **Mesh construction** (every timestep):
   - 2.1: Primary Delaunay triangulation (or existing triangulation). Every vertex $v$ is connected via primal edges to a set of neighbours $v.\text{nn}$.
   - 2.2: Barycentric dual via `compute_vd`. Produces dual vertices and dual hyperplanes/flux planes dual to the primary edges.
   - 2.3: Dual flux planes $\mathbf{A}_{ij}$ (area vectors) for each primal edge — in 3D: a connected set of triangles (`A_ijk_arr` from `e_star`); in 2D: a line segment. Each flux plane has an exact vector area (plane normal $\times$ scalar area).
   - 2.4: The sum of all dual flux planes around $v$ forms the boundary/surface of the Hodge dual of $v$ (a simplicial complex).
   - 2.5: This dual cell is the **finite volume element** (FVM) around $v$, with volume $\text{Vol}_i$ (cached as `v.dual_vol`).

3. **Force computation** (Stokes' theorem on each FVM):
   - For each dual flux plane (face between parcels $i$ and $j$):
     - Pressure flux: $\mathbf{F}_{p,ij} = -\tfrac{1}{2}(p_i + p_j)\,\mathbf{A}_{ij}$
     - Viscous flux: $\mathbf{F}_{v,ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij} \cdot \mathbf{A}_{ij})$
   - Total force on FVM $i$: $\mathbf{F}_i = \sum_j (\mathbf{F}_{p,ij} + \mathbf{F}_{v,ij})$

4. **Time integration**: $\mathbf{a}_i = \mathbf{F}_i / m_i$, update velocity via integrator, update pressure from EOS using $\text{Vol}_i$, return to step 1.

---

## Relevant equations (continuum level)

Cauchy momentum equation (integral form over a material parcel $V(t)$, the Lagrangian control volume):

$$\frac{d}{dt} \int_{V(t)} \rho \mathbf{v}\, dV = \int_{S(t)} \mathbf{\sigma} \cdot \mathbf{n}\, dS + \int_{V(t)} \rho \mathbf{b}\, dV$$

For a discrete parcel $i$ with fixed mass $m_i = \rho_i \, \text{Vol}_i^{\text{dual}}$:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_{\text{stress}} + \mathbf{F}_{\text{body}}, \quad \mathbf{F}_{\text{stress}} = \int_{S_i} \mathbf{\sigma} \cdot \mathbf{n}\, dS$$

where $S_i$ is the closed surface of the dual cell around primary vertex $i$.

### Constitutive relation (Newtonian fluid)

$$\mathbf{\sigma} = -p\,\mathbf{I} + \mathbf{\tau}(\mathbf{v})$$

- $p$ is the scalar pressure (positive in compression)
- $\mathbf{\tau} = 2\mu\,\mathbf{\varepsilon}$ is the deviatoric stress, $\mathbf{\varepsilon} = \frac{1}{2}(\nabla\mathbf{v} + (\nabla\mathbf{v})^\top)$

---

## Discrete formulation (face-centered integrated)

### Pressure flux (face-average, conservative)

For each primal edge $ij$ with dual flux plane area vector $\mathbf{A}_{ij}$ (outward from $i$):

$$\mathbf{F}_{p,ij} = -\frac{1}{2}(p_i + p_j)\,\mathbf{A}_{ij}$$

The face-averaged pressure $p_f = \frac{1}{2}(p_i + p_j)$ pushes on the oriented face area $\mathbf{A}_{ij}$. This is the conservative form: each face contribution is self-contained and Newton's 3rd law is manifest ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$ since $\mathbf{A}_{ij} = -\mathbf{A}_{ji}$), guaranteeing global momentum conservation.

For interior vertices (where $\sum_j \mathbf{A}_{ij} = \mathbf{0}$, closure), this is algebraically equivalent to the half-difference form $-\frac{1}{2}(p_j - p_i)\,\mathbf{A}_{ij}$.

### Viscous flux (face-centered diffusion)

The face-centered velocity gradient is constructed from the discrete exterior derivative (velocity difference along the primal edge):

$$(\nabla\mathbf{u})_f = \frac{(\mathbf{u}_j - \mathbf{u}_i) \otimes \hat{\mathbf{d}}_{ij}}{|\mathbf{d}_{ij}|}$$

where $\mathbf{d}_{ij} = \mathbf{x}_j - \mathbf{x}_i$ and $\hat{\mathbf{d}}_{ij} = \mathbf{d}_{ij}/|\mathbf{d}_{ij}|$.

The viscous flux uses the **diffusion form** (only $\nabla\mathbf{u}$, not the symmetric $\nabla\mathbf{u} + (\nabla\mathbf{u})^\top$):

$$\mathbf{F}_{v,ij} = \mu\,(\nabla\mathbf{u})_f \cdot \mathbf{A}_{ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij} \cdot \mathbf{A}_{ij})$$

where $\Delta\mathbf{u} = \mathbf{u}_j - \mathbf{u}_i$.

**Why diffusion form, not symmetric stress form?** For incompressible flow ($\nabla \cdot \mathbf{u} = 0$), the stress divergence simplifies:

$$\nabla \cdot \bigl[\mu(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top)\bigr] = \mu\nabla^2\mathbf{u} + \mu\nabla(\nabla \cdot \mathbf{u}) = \mu\nabla^2\mathbf{u}$$

Both forms give $\mu\nabla^2\mathbf{u}$ in the continuum. However, the rank-1 face gradient $(\nabla\mathbf{u})_f = \Delta\mathbf{u} \otimes \hat{\mathbf{d}}/|\mathbf{d}|$ has **spurious discrete compressibility**: its trace $\text{tr}((\nabla\mathbf{u})_f) = \Delta\mathbf{u} \cdot \hat{\mathbf{d}}/|\mathbf{d}|$ is nonzero on diagonal edges even for divergence-free fields. The symmetric form picks up this spurious $\nabla(\nabla \cdot \mathbf{u})$ term, while the diffusion form avoids it. On the Poiseuille test case, the diffusion form gives exact zero residual at machine precision.

### Why the diffusion form is exact for quadratic fields

For a quadratic velocity field $u(y)$, the finite difference $(u_j - u_i)/|\mathbf{d}_{ij}|$ gives the **exact midpoint derivative** in the $\hat{\mathbf{d}}$ direction. The dot product $\hat{\mathbf{d}} \cdot \mathbf{A}$ projects the area vector onto the edge direction, giving the effective face area for this directional derivative. On a well-formed dual mesh (barycentric or circumcentric), the sum over all faces recovers the exact integrated Laplacian via the divergence theorem.

### Total force and acceleration

$$\mathbf{F}_i = \sum_{j \in N(i)} (\mathbf{F}_{p,ij} + \mathbf{F}_{v,ij})$$

$$\mathbf{a}_i = \mathbf{F}_i / m_i$$

---

## Notes

- The old point-wise formulation (vertex-centered gradients, face-averaged stresses) is preserved in `Fundamentals_pointwise.md` and `ddgclib/operators/stress_pointwise.py` for reference.
- The face-centered approach uses a compact stencil (only the two vertices sharing each edge) and avoids division by volume.
- The diffusion form $\Delta\mathbf{u}\,(\hat{\mathbf{d}} \cdot \mathbf{A})$ is exact for quadratic velocity fields on well-formed meshes — no non-orthogonality correction needed for the scalar Laplacian.
- For compressible flow or non-Newtonian constitutive relations requiring the full stress tensor, the symmetric form $\mu(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top) \cdot \mathbf{A}$ would need a divergence correction or a higher-order gradient reconstruction.
