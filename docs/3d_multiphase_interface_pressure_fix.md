# Multiphase Interface Pressure-Flux Fix

**Date:** 2026-04-15
**Scope:** `ddgclib/operators/multiphase_stress.py`
**Affects:** both 2D and 3D multiphase simulations (oscillating droplet,
dam break, and any case with surface tension)

## Problem

The 3D (and 2D) oscillating droplet simulations showed **unbounded
growth** even in the overdamped regime:

- KE jumps instantly from ~1e-30 (zero initial velocity) to ~1e-5 in
  the first time step
- R_max grows from R0=0.01 to ~0.035 (3D) or ~0.036 (2D) over the
  simulation window
- Analytical solution predicts R_max stays near R0 (overdamped)

The kinetic-energy jump in a single time step from zero IC is the
signature of a spurious initial impulse — physical forces cannot do
this.

## Root Cause

The old `multiphase_stress_force` computed the pressure flux using each
vertex's **own-phase pressure**:

```python
p_i = _resolve_pressure(v, ...)       # v.p_phase[v.phase]
for v_j in v.nn:
    p_j = _resolve_pressure(v_j, ...) # v_j.p_phase[v_j.phase]  ← WRONG
    F -= 0.5 * (p_i + p_j) * A_ij
```

For an **outer-phase bulk vertex** `v_i` whose neighbour `v_j` is an
**interface vertex**:

- `v_i.phase = 0` (outer), so `p_i = v_i.p_phase[0] = 0 Pa`
- `v_j.phase = 1` (droplet, the "home" phase of interface vertices),
  so `p_j = v_j.p_phase[1] = 10 Pa` (Young-Laplace droplet pressure)

The flux sees a spurious jump of `~γκ` (~10 Pa here) across every
bulk-to-interface edge, producing inward force on the outer phase and
outward reaction on the droplet. Once velocity accumulates, the
instability grows.

### Why the 3D case showed it first

In 3D the Laplace pressure is `γ·(2/R)` vs `γ·(1/R)` in 2D, so the
spurious jump is twice as large. The 3D mesh also has many more
bulk-to-interface adjacencies (482 such edges in the test case),
compounding the impulse.

## Quantitative Evidence — Static Equilibrium

On an **equilibrium sphere/circle** (no perturbation → should be
perfectly static):

### 3D (472 vertices, 98 interface)

| Vertex category | Before fix | After fix | Improvement |
|----------------|-----------|-----------|-------------|
| interior_droplet (91) | 7.1e-19 | 7.1e-19 | — |
| **interior_outer (194)** | **1.9e-04** | **2.7e-17** | **13 orders** |
| interface (98) | 9.2e-05 | 8.5e-05 | — (separate issue) |
| boundary_wall (89) | 1.5e-17 | 1.5e-17 | — |

### 2D (311 vertices, 32 interface)

| Vertex category | After fix |
|----------------|-----------|
| interior_droplet (113) | **2.5e-16** (machine eps) |
| **interior_outer (55)** | **2.6e-15** (machine eps) |
| interface (32) | 3.8e-03 |
| boundary_wall (15) | 2.7e-15 |

The fix delivers **machine-precision static equilibrium on all bulk
vertices**, both inside and outside the droplet.

## Fix

Applied in [`ddgclib/operators/multiphase_stress.py`](../ddgclib/operators/multiphase_stress.py)
`multiphase_stress_force`:

### 1. Use same-phase pressure at both ends of each flux

```python
own_phase = v.phase
p_i = float(v.p_phase[own_phase])
for v_j in v.nn:
    ...
    p_j = float(v_j.p_phase[own_phase])   # ← v_i's phase, not v_j's
    F -= 0.5 * (p_i + p_j) * A_ij
```

Interface vertices store pressures for *both* phases in `v_j.p_phase`,
so reading the right index gives the correct same-phase value.

### 2. Skip cross-phase edges that double-count the interface jump

```python
if v_j.phase != own_phase and not is_interface_j:
    continue   # other-phase bulk — not in v_i's dual cell
```

The interface vertex's dual cell lives inside its own phase; the
surface tension term already accounts for the pressure jump across
the sharp interface.

## Simulation Results

### 3D Oscillating Droplet

Configuration: R0=0.01, γ=0.05, μ_d=0.5, μ_o=0.1, mode l=2, ε=0.05,
472 vertices, 98 interface, dt=1.84e-4 s, 872 steps, t_end=0.16 s.

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Final R_max | **0.0345** (3.45× R0, runaway) | **0.0134** (1.34× R0, bounded) |
| KE(t) | monotone growth to 4.4e-3 | plateau at 2-5e-3 |
| Mass conservation \|ΔM/M₀\| | 1.05e-14 | 1.08e-14 (unchanged) |

Figures in [`cases_dynamic/oscillating_droplet/fig/`](../cases_dynamic/oscillating_droplet/fig/):
- `oscillating_droplet_3D_radius.png` / `_after_fix.png` (same image; named both for clarity)
- `oscillating_droplet_3D_energy.png` / `_after_fix.png`

### 2D Oscillating Droplet

Configuration: R0=0.01, γ=0.05, μ_d=0.5, μ_o=0.1, mode l=2, ε=0.05,
311 vertices, 32 interface, dt=6.22e-5 s, 1839 steps, t_end=0.11 s.

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Final R_max | **0.0356** (3.56× R0, runaway) | **0.0206** (2.06× R0, improved but still drifting) |
| Final KE | 4e-1 J | 2.4e-1 J |
| Mass conservation \|ΔM/M₀\| | 0 | 0 |

Figures:
- `oscillating_droplet_2D_radius_before_fix.png` / `_after_fix.png`
- `oscillating_droplet_2D_energy_before_fix.png` / `_after_fix.png`

The 2D case improves by 1.7× in final radius error but is NOT yet
stable. The remaining instability comes from the **~3.8e-3 residual
force on interface vertices** (see table above), not from the
bulk-phase spurious forces we fixed. See "Remaining Issue" below.

## Remaining Issue (Separate)

The interface vertices still exhibit non-zero force at equilibrium:

- **3D**: ~8.5e-5 max (≈ 0.01% of nominal pressure)
- **2D**: ~3.8e-3 max (≈ 0.4% of nominal pressure)

This is the **discrete curvature / pressure-jump discretization
imbalance**: the surface tension force `F_st = -γ ∫ κ N dS` computed
from the discrete mesh geometry doesn't exactly cancel the pressure
flux from `v_i.p_phase[own_phase]` neighbours across interface-to-
interface edges.

### Why 2D is worse than 3D here

Counter-intuitive at first. Hypothesis: the 2D interface has 32
vertices on the circle, while the 3D interface has 98 vertices on
the sphere.  Per-vertex the 2D case has larger arc-length (~2πR/32
vs ~R/3 for 3D sphere ≈ R/3 per vertex face), so the curvature
imbalance per vertex is larger.

This is a separate fix, deferred to follow-up work. Options:

1. Use the **integrated** 2D curvature operator `surface_tension_force_2d`
   (already exists in `ddgclib/operators/curvature_2d.py`) everywhere,
   and apply the corresponding 3D Heron integrated operator.
2. Apply **curvature-weighted pressure** at interface flux edges so
   the pressure jump and curvature terms are discretized consistently.
3. **Sub-cell pressure splitting** at interface vertices: compute
   separate pressure gradients on each side of the interface and
   combine properly.

## Test Suite

`pytest ddgclib/tests/ -m "not slow"`: **702 passed, 0 failed** —
no regression.

## Related Work

This bug is distinct from (but compounds with) the 3D boundary sliver-
tet bug fixed earlier in [`docs/3d_simplex_aware_dual_fix.md`](3d_simplex_aware_dual_fix.md).
Both affect multiphase simulations:

1. **Sliver-tet bug** (fixed in earlier commit) — breaks dual
   connectivity near domain boundaries, causing O(h) errors in the
   linear precision identity.
2. **Interface pressure-flux bug** (this fix) — causes O(γκ)
   impulsive forces on bulk vertices adjacent to the interface.
3. **Interface curvature imbalance** (remaining, documented above) —
   causes smaller residual forces on interface vertices themselves.

## Files Modified

- [`ddgclib/operators/multiphase_stress.py`](../ddgclib/operators/multiphase_stress.py)
  lines 30–125 — rewrote `multiphase_stress_force` with same-phase
  pressure flux and cross-phase edge skip.
