
# Fundamentals of the method (ARCHIVED — vertex-centered point-wise formulation)

> **Superseded** by the face-centered integrated formulation in `Fundamentals.md`.
> This document is kept for reference only.
>
> **Key issue discovered**: The symmetric stress tensor $\boldsymbol{\tau} = \mu(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top)$ introduces a spurious $\mu\nabla(\nabla \cdot \mathbf{u})$ term in the discrete setting. The rank-1 face gradient $(\nabla\mathbf{u})_f = \Delta\mathbf{u} \otimes \hat{\mathbf{d}}/|\mathbf{d}|$ has nonzero discrete divergence on diagonal edges even for divergence-free fields, causing O(1) acceleration residuals at equilibrium instead of machine-precision zero. The current formulation uses the **diffusion form** $\mu(\nabla\mathbf{u})_f \cdot \mathbf{A}$ which avoids this issue entirely. See `Fundamentals.md` for details.

## Relevant equations (continuum level)

Cauchy momentum equation (integral form over a material parcel $  V(t)  $, the Lagrangian control volume):

$$\frac{d}{dt} \int_{V(t)} \rho \mathbf{v}\, dV = \int_{S(t)} \mathbf{\sigma} \cdot \mathbf{n}\, dS + \int_{V(t)} \rho \mathbf{b}\, dV$$

For a discrete parcel $  i  $ with fixed mass $  m_i = \rho_i \, \mathrm{Vol}_i^\mathrm{dual}  $ (or variable density if compressible) this becomes the ODE for the parcel velocity:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_\mathrm{stress} + \mathbf{F}_\mathrm{body}, \quad \mathbf{F}_\mathrm{stress} = \int_{S_i} \mathbf{\sigma} \cdot \mathbf{n}\, dS$$

where $  S_i  $ is the closed surface of the dual cell around primary vertex $  i  $.

In code, we now want to general the stress tensor $\mathbf{\sigma}$ to accept any constitutive relation, but we will start with the one we had before for fluids leading to the Navier-Stokes equations:

## Constitutive relation (Newtonian fluid, your starting point):

$$\mathbf{\sigma} = -p\,\mathbf{I} + \mathbf{\tau}(\mathbf{v})$$

- $  p  $ is the scalar pressure (positive in compression for fluids; you wrote $  +p\mathbf{I}  $, which is the opposite sign convention – common in solid mechanics where $  p  $ is tension – but the mathematics is identical, only the sign of $  p  $ changes).
- $  \mathbf{\tau}  $ is the deviatoric stress (shear). For an incompressible Newtonian fluid $  \mathbf{\tau} = 2\mu\,\mathbf{\varepsilon}  $, $  \mathbf{\varepsilon} = \frac12(\nabla\mathbf{v} + (\nabla\mathbf{v})^\top)  $.

The pressure part of the traction on any oriented surface element is simply $  -p\,\mathbf{A}  $ (or $  +p\,\mathbf{A}  $ in your sign convention), where $  \mathbf{A} = \mathbf{n}\,dA  $ is the outward area vector.
Navier–Stokes is exactly the above with the Newtonian $  \mathbf{\tau}  $ inserted and the continuity equation added.


### Discrete setting

- Primary mesh: vertices = fluid parcels (each stores $  \mathbf{v}_i  $, $  p_i  $, mass $  m_i  $).
- Dual mesh: control volume around each primary vertex = polyhedron (3D) / polygon (2D) whose boundary consists of flat simplices (triangles in 3D, line segments in 2D) that are the duals to the primal edges.
- For every primal edge $  ij  $ the dual surface that “separates” parcels $  i  $ and $  j  $ is already triangulated in your code (see v_star / the loop that builds A_ij lists of $  \frac12  $ cross products).

### Refactoring idea (completed)

This refactoring was completed — but the vertex-centered tensor approach described below was found to have a spurious discrete compressibility issue. The final production formulation uses face-centered fluxes instead (see `stress.py` and `Fundamentals.md`).

## Step-by-step computation

### 1. Compute oriented area vectors per interface
For each primal edge $  ij  $, when you call the dual-surface routine from vertex $  i  $:
- Build the list of small triangle area vectors exactly as you already do (wedge = ½ cross(...)).
- Orient each small triangle so that its normal points outward from parcel $  i  $.

A robust way (already almost in code):

```python
vec_to_i = v_i.x_a - vc_12.x_a          # from dual triangle toward parcel i (inside)
if np.dot(wedge, vec_to_i) > 0:         # normal points inside → flip
    wedge = -wedge
```

(or use the signed volume of the pyramid from $  v_i  $ to the triangle; if negative, flip).
- Sum the small vectors → total outward area vector $  \mathbf{A}_{ij}  $ for the whole dual surface of edge $  ij  $.

In 2D the same logic applies: the “dual face” is the line segment between the two barycenters (or midpoint on boundary); the area vector is the 90° rotation of that segment with length = segment length and sign chosen so the normal points outward from $  i  $.

### 2. Stress at the interface
At each vertex compute the full Cauchy tensor:

```python
sigma_i = -p_i * np.eye(3) + tau_i          # tau_i from local velocity gradient (or your old du routine generalised)
```

On the shared face:

```python
sigma_f = 0.5 * (sigma_i + sigma_j)         # arithmetic average – second-order consistent
```

### 3. Force contribution from each interface

```python
F_from_ij_on_i = sigma_f @ A_ij             # 3×3 matrix × 3-vector → 3-force
```
(In your sign convention `sigma = p*I + tau` it becomes `+p_f * A_ij` – the mathematics is identical.)

### 4. Total stress force on parcel $  i  $

```python
F_stress_i = np.zeros(3)
for j in neighbors_of_i:
    A_ij = compute_oriented_dual_area(i, j)   # outward from i
    sigma_f = 0.5 * (sigma_i + sigma_j)
    F_stress_i += sigma_f @ A_ij
```
Then
```python
m_i * dv_i/dt = F_stress_i + body_forces
```

### 5. Pressure part alone (to see it reduces to old method)

The pressure contribution is exactly $  -p_f \mathbf{A}_{ij}  $ (or $  +p_f \mathbf{A}_{ij}  $ in your convention).
With the average $  p_f = (p_i + p_j)/2  $ the global sum over a closed surface automatically gives a consistent discretisation of $  -\nabla p  $ (the $  p_i  $ terms cancel because $  \sum \mathbf{A} = \mathbf{0}  $).
Your old scalar version was the 1-D / projected special case of this.

### Why this is the natural generalisation

- The old code already computed the geometric ingredients (A_ij lists, dual volumes, etc.).
- The only new step is to keep the vectors instead of taking norms, orient them outward, and do the matrix–vector product.
- The pressure is automatically “distributed in the trace” because $  p\mathbf{I} \cdot \mathbf{A} = p\mathbf{A}  $.
- The shear part $  \boldsymbol{\tau} \cdot \mathbf{A}  $ is added in the same loop – no extra data structures needed.
- The whole thing is still exactly Newton’s second law on the Lagrangian parcels (integrated Cauchy equation).

### Small implementation notes for your code base

- Extend v_star / e_star (or write dual_area_vector(i, j)) to return the summed, oriented $  \mathbf{A}_{ij}  $.
- When you call it from $  i  $ use the sign test above (or set n = v_i.x_a - vc_12.x_a and flip if needed).
- For 2D: A = np.array([-dy, dx]) (or the opposite rotation) with the same inside/outside test.
- Compute $  \boldsymbol{\tau}_i  $ at vertices from a discrete gradient operator (you can reuse the same dual-area machinery: the discrete gradient at vertex $  i  $ is $  \frac{1}{\mathrm{Vol}_i} \sum_j \mathbf{v}_{ij} \otimes \mathbf{A}_{ij}  $, etc.).
- The old dP and du become special cases of the new loop (pressure-only or Laplacian-only).

This is the clean, dimension-independent way to go from your current pressure-only integrator to a full Cauchy-stress / Navier–Stokes integrator on the same dual mesh. Your intuition about “use the area vector and matrix math” is precisely the standard control-volume / mimetic discretisation used in modern unstructured Lagrangian and vertex-centered finite-volume codes. You already have all the geometric machinery – you just need to keep the vectors and do the tensor contraction.

---

## Post-implementation note

This vertex-centered approach was implemented in `stress_pointwise.py` and tested on Poiseuille flow. The symmetric stress form $\boldsymbol{\tau} \cdot \mathbf{A} = \mu(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top) \cdot \mathbf{A}$ produced O(1) acceleration residuals (~0.34) at equilibrium due to spurious discrete compressibility from the rank-1 face gradient.

The production code now uses the **face-centered diffusion form**:

$$\mathbf{F}_{v,ij} = \frac{\mu}{|\mathbf{d}_{ij}|}\,\Delta\mathbf{u}\,(\hat{\mathbf{d}}_{ij} \cdot \mathbf{A}_{ij})$$

which gives machine-precision zero residual for Poiseuille flow. See `Fundamentals.md` and `ddgclib/operators/stress.py` for the current formulation.