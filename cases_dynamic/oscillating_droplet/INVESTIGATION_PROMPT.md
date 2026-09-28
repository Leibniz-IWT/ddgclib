# Investigation prompt: static droplet instability vs cube2droplet stability

**For a fresh Claude Code session.**

## Context

We completed a multiphase sharp-interface refactor (primal subcomplex
model, exact per-phase viscosity, factored flux helpers).  Running the
case studies revealed an important puzzle:

- `cases_dynamic/Cube2droplet/diagnostic_no_retopo.py` (square droplet,
  fixed connectivity, dual-only recomputation) **works correctly**:
  circularity 0.71 → 0.95 over 5000 steps, velocity decays, interface
  preserved.  The surface tension correctly pulls the square toward a
  circle.

- `cases_dynamic/oscillating_droplet/static_droplet_2D.py` (circular
  droplet, epsilon=0, full Delaunay retopo) **is unstable**: KE grows
  from ~0 to 4.4e-3 in 100 steps, the interface deforms even though
  epsilon=0 means no perturbation was applied.  A static circular
  droplet at Young-Laplace equilibrium should have zero net force on
  every vertex — KE should remain at machine precision.

## What to investigate

Read the following files first:
- `Fundamentals.md` (continuum equations, stress operators)
- `CLAUDE.md` (test commands, architecture)
- `ddgclib/operators/stress.py` — `pressure_flux`, `viscous_flux`, `stress_force`
- `ddgclib/operators/multiphase_stress.py` — `multiphase_stress_force`, `_interface_surface_tension`
- `ddgclib/operators/curvature_2d.py` — `surface_tension_force_2d`
- `cases_dynamic/oscillating_droplet/src/_setup.py` — Young-Laplace initialisation
- `cases_dynamic/oscillating_droplet/static_droplet_2D.py` — the failing equilibrium test
- `cases_dynamic/Cube2droplet/diagnostic_no_retopo.py` — the working case
- Memory file: `project_multiphase_interface_model.md`

Then diagnose:

1. **Force balance at t=0 on the static circular droplet.**
   For an unperturbed circular droplet at Young-Laplace equilibrium,
   the net force on every vertex should be zero:
   - Pressure gradient force (from the Laplace jump across the
     interface) should exactly balance surface tension.
   - Viscous force should be zero (zero velocity field).
   
   Compute `multiphase_stress_force(v)` at t=0 for a few interface
   vertices and a few bulk vertices.  Print the force components
   (pressure, viscous, surface tension separately).  Is the residual
   zero?  If not, which component is non-zero and why?

2. **Initial condition consistency.**
   The setup applies Young-Laplace pre-loading at line ~160 of
   `_setup.py`:
   ```python
   rho_d_eq = float(eos_drop.density(p_outer + delta_p))
   v.m_phase[1] = rho_d_eq * vol_d
   ```
   Then calls `mps.refresh(HC, dim, reset_mass=False)` which
   recomputes dual volumes and pressures.  Check:
   - Does `v.p_phase[1]` equal `p_outer + gamma*kappa` after refresh?
   - Does `v.p_phase[0]` equal `p_outer` for the outer-phase sub-cell
     of interface vertices?
   - Is there a pressure gradient within the bulk droplet (should be
     uniform)?

3. **Why does cube2droplet work but static droplet doesn't?**
   Key differences:
   - Cube2droplet starts at P0=0 everywhere (no Laplace pre-loading)
     and lets the EOS evolve pressure dynamically.
   - Static droplet pre-loads the Laplace jump into the mass field.
   - Cube2droplet uses `dual_only_retopo` (fixed connectivity).
   - Static droplet uses full Delaunay retopo.
   
   Try running `static_droplet_2D.py` with `dual_only_retopo` instead
   of full Delaunay — does it stabilise?  If so, the instability is
   from retopologisation, not from the force balance.

4. **Mesh-conformity of the initial droplet.**
   The domain builder `droplet_in_box_2d` constructs an
   interface-conforming mesh (vertices placed on the circle).  After
   `mps.refresh(HC, dim, reset_mass=True)`, check:
   - Are all interface vertices exactly on the circle (r == R0)?
   - Does `validate_closure(HC, 2)` pass?  (It should, since the
     initial mesh is conforming.)
   - After the first Delaunay retopo step, does closure still hold?

5. **Surface tension force direction.**
   For a convex interface (circle), the integrated surface tension
   force `gamma * (t_next - t_prev)` should point INWARD (toward
   the center).  Verify the sign for a few interface vertices.
   If it points outward, the surface tension is driving expansion
   instead of compression.

## Expected outcome

A diagnostic that pinpoints the root cause:
- Incorrect Young-Laplace pre-loading (pressure imbalance at t=0), OR
- Retopologisation destroying the interface geometry, OR
- Surface tension force sign error, OR
- Phase-fraction error in `edge_phase_area_fractions` for the circular
  droplet case.

Then propose a targeted fix.  Do NOT attempt a broad refactor — isolate
the specific bug and fix it with a minimal change.  Confirm the fix by
running `static_droplet_2D.py` and showing KE remains < 1e-10 over
100 steps.
