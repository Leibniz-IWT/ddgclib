# laneK: single-phase EOS instability (template box, Hydrostatic_column 2D)

Date: 2026-09-25. Measurement lane: no library, test, baseline, case or
registry file was changed. New files only:

- `cases_dynamic/template/diagnose_single_phase_eos.py`: the matrix runner,
  energy ledger, remap prototype and Hydrostatic cross-check
- this log

Scratch outputs (session scratchpad, regenerable with the commands in §8):
`.../scratchpad/laneK/matrix2/<arm>.{methods.json,npz,summary.json}`,
`.../laneK/hydro2/`, `fd_volume_gradient.py`, `fd_variational.py`.

## 0. Verdict

The blow-up is not an acoustic time-step instability. It has two separate
causes, and both come from the dual volume the EOS reads not being
consistent with the pressure flux the force applies:

1. **Reconnection (the D1 mechanism), which drives the template box.** A
   Delaunay flip changes the barycentric dual volume discontinuously, by
   |T|/3 per affected vertex, so dV/V = 0.33 to 1.0 even when the four
   points are exactly co-circular. With masses held, the EOS reads this as
   compression: Δp ≈ K·ΔV/V = 3e4 to 5e4 Pa at K = 1e5, against a dynamic
   pressure ρu² = 10 Pa. The energy ledger shows every step without a flip
   is dissipative (dE_dyn < 0). Each flip step injects ±2e3 to 5e3 J, while
   the initial energy E0 is 2.2 J. The structured grid is the worst case:
   its squares are co-circular, so the first 1e-5 relative motion already
   flips them back and forth.
2. **The 2D fallback dual volume, which drives Hydrostatic_column 2D with
   zero reconnection.** On a builder mesh without `HC._simplices`,
   `cache_dual_volumes` uses `dual_cell_area_2d`. That volume is wrong at
   boundary vertices that are not on a straight segment: corners are
   undercounted 4x, and when a free-surface vertex moves, only 1/4 of its
   own volume change is credited to it. The shipped Hydrostatic 2D loop
   runs on this path. It amplifies a 1e-6 m/s seed exponentially to
   blow-up (about 13 t_ac at g = 0, about 6 t_ac with gravity). The same
   loop with exact simplex volumes damps the seed by 8 orders of magnitude
   and settles over 40 t_ac.

With fixed connectivity and exact volumes (`dual_only`), the discrete
pressure force is the exact gradient of Σ p_k V_k for every closed fan
(finite-difference check: 1e-10 relative). The scheme is therefore
variational, and symplectic Euler is stable up to dt·c_s/dx_min = 1.5 for
every c_s and n tested. A **single-phase conservative remap** (fresh
old-connectivity pressure snapshot, then rebuild, then re-target masses on
ALL vertices including the frozen walls, then one mass scale) removes
mechanism 1. The EOS box stays stable for 1000 steps, at c_s = 100, at
n = 7.15 and up to CFL 1.5. It reproduces the fixed-connectivity KE to
0.1 % at the end of the horizon.

## 1. Setup

This is the probe setup (`scratchpad/probe_template.py`, reused verbatim):

- rectangle L = h = 1, refinement 2, 41 vertices, 16 frozen hull vertices
- ρ0 = 1000, μ = 50, divergence-free swirl u0 = 0.1 m/s
- TaitMurnaghan(P0 = 0, K = ρ0·c_s², rho_clip = (0.5, 2))
- dt = CFL·dx_min/c_s with dx_min = 0.17678
- same physical horizon for every arm: t_end = 0.884 s (200 steps at CFL 0.25 and c_s 10), or 5x that for `_long` arms
- every arm aborts when max|u| > c_s

Every arm is a `SolverMethods`, recorded with `record_methods()`. Fields
that are the same in every arm, from `methods.to_dict()`:

`{dim: 2, phases: 'single', remap: None, projection_every: 1, split_method: 'neighbour_count', curvature_path: 'integrated', displacement_eps: None, merge_cdist: None, remesh_kwargs: None, periodic_axes: None, backend: None, workers: None, label: 'laneK <arm>'}`

`notes` holds the JSON of the non-method knobs `{eos, cfl, c_s, n, setup_volumes, custom, custom_kw}`.
The table below gives the varying fields (integrator / connectivity / redistribute_mass).

Setup variants:

- `builder` is the probe: builder mesh, no simplex cache, so the 2D fallback volumes apply. Total volume is 0.96875, with the 4 corners at 0.0026 instead of 0.0104.
- `exact` rebuilds the simplex cache on the builder mesh.
- `settled` runs one library Delaunay retopology before the ICs, so the per-step rebuild starts from the mesh the masses were computed on (no step-0 flips).

Probe artefact (kept, so the numbers reproduce): `walls` is the top/bottom
group only. The six side vertices are frozen by bV but keep their initial
tangential swirl velocity, so they act as moving lids. That is why KE
plateaus around 0.55 to 0.8 J instead of decaying to zero. Arms marked
`wallsall` zero every hull velocity; the conclusions do not change.

Energy ledger. E = KE + Σ m_i e(m_i/V_i), where e is the exact Tait
specific energy. Each step is split into:

- dE_retopo = PE on the new duals and masses minus PE on the old connectivity at the same positions
- dE_dyn = the rest

In arms without the EOS in dudt, the PE column is not seen by the solver
and is not meaningful.

## 2. Matrix

Columns: CFL = dt·c_s/dx_min. u2 = first step with max|u| > 2u0. Flips =
step-0 flips + later flips. ΣdE_re+ = sum of the positive reconnection
energy injections (J). ΣdE_dyn = the dynamics part (J). V = total dual
volume at setup / at the end.

| arm | integrator / connectivity / redist | CFL | steps | KE0 -> KE_end | max\|u\| | u2 | flips | ΣdE_re+ | ΣdE_dyn | p interior | p walls | ρ range | V setup/end | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 probe as-is | symp / delaunay / T | 0.25 | 80/200 | 2.08 -> 2.8e3 | 18.4 | 0 | 24+294 | 1.3e4 | -2.4e3 | -5e4..1e5 | -5e4..1e5 | 250..6777 | 0.969/1.044 | BLOW-UP |
| A2 | symp / delaunay / F | 0.25 | 32/200 | 2.08 -> 7.9e3 | 10 | 0 | 24+170 | 3.4e4 | 2.4e3 | -5e4..1e5 | -5e4..1e5 | 59..3815 | 0.969/1.069 | BLOW-UP |
| A3 no EOS | symp / delaunay / F | 0.25 | 200 | 2.08 -> 0.785 | 0.1 | - | 24+10 | n/a | n/a | 0 | 0 | 238..2010 | 0.969/1.000 | stable |
| A4 | symp / dual_only / T | 0.25 | 200 | 2.08 -> 0.678 | 0.1 | - | 0 | 0 | -1.27 | -1.9..0 | -209..215 | 998..1002 | 0.969/0.969 | stable (p projected) |
| A5 | symp / dual_only / F | 0.25 | 200 | 2.08 -> 0.648 | 0.1 | - | 0 | 0 | -1.43 | -134..140 | -204..209 | 998..1002 | 0.969/0.969 | stable |
| A6 no EOS | symp / dual_only / F | 0.25 | 200 | 2.08 -> 0.795 | 0.1 | - | 0 | n/a | n/a | 0 | 0 | 793..1288 | 0.969 | stable |
| A7 | symp / frozen / F | 0.25 | 200 | 2.08 -> 0.794 | 0.1 | - | 0 | 0 | -1.29 | 0 | 0 | 1000 | 0.969 | stable, EOS INERT (V never refreshed) |
| A8 no EOS | symp / frozen / F | 0.25 | 200 | 2.08 -> 0.794 | 0.1 | - | 0 | - | - | 0 | 0 | 1000 | 0.969 | stable |
| A9 | euler_velocity_only / delaunay / F | 0.25 | 14/200 | 2.08 -> 6.8e3 | 10.4 | 0 | 24+0 | 1.5e4 (step 0) | 6.8e3 | -5e4..1e5 | -5e4..3e4 | 250..2000 | 0.969/1.000 | BLOW-UP (step-0 jump frozen in) |
| A10 no EOS | euler_velocity_only / delaunay / F | 0.25 | 200 | 2.08 -> 0.774 | 0.1 | - | 24+0 | n/a | n/a | 0 | 0 | - | - | stable |
| B1 | euler / delaunay / F | 0.25 | 23/200 | 2.08 -> 9.4e3 | 14.6 | 0 | 24+66 | 2.2e4 | 9.3e3 | | | | | BLOW-UP |
| B2 | euler / dual_only / F | 0.25 | 200 | 2.08 -> 1.5e3 | 5.8 | 108 | 0 | 0 | +2.2e3 | -2e4..4e4 | -5e4..1e5 | 206..1.8e4 | 0.969/0.970 | BLOW-UP (forward Euler) |
| C1 | symp / delaunay / F | 0.05 | 350/1000 | -> 1.2e4 | 10.2 | 1 | 24+510 | 6.4e4 | -3.4e3 | | | | | BLOW-UP |
| C2 | symp / delaunay / F | 0.01 | 328/5000 | -> 8.5e3 | 10.1 | 7 | 24+248 | 2.9e4 | 6.8e3 | | | | | BLOW-UP |
| C3 | symp / dual_only / F | 0.05 | 1000 | -> 0.649 | 0.1 | - | 0 | 0 | -1.43 | -134..140 | -204..209 | 998..1002 | | stable |
| C4 | symp / dual_only / F | 0.01 | 5000 | -> 0.649 | 0.1 | - | 0 | 0 | -1.43 | same | same | same | | stable |
| C5 | symp / delaunay / T | 0.05 | 1000 | -> 5.5e3 | 8.8 | 1 | 24+781 | 2.8e4 | 2.2e3 | | | | | BLOW-UP (no abort, saturated) |
| D1 c_s 100 | symp / delaunay / F | 0.25 | 72/2000 | -> 1.4e6 | 100 | 0 | 24+435 | 5.7e6 | -4.2e5 | -5e6..1e7 | | | | BLOW-UP |
| D2 c_s 100 | symp / dual_only / F | 0.25 | 2000 | -> 0.649 | 0.1 | - | 0 | 0 | -1.43 | -1.4e3..1.4e3 | -2.3e3..2.3e3 | 999.8..1000 | | stable |
| D3 n 7.15 | symp / delaunay / F | 0.25 | 1/200 | -> 1.1e5 | 32 | 0 | 24 | 6.2e4 | 9.6e4 | ..2e6 | | | | BLOW-UP at step 0 |
| D4 n 7.15 | symp / dual_only / F | 0.25 | 200 | -> 0.648 | 0.1 | - | 0 | 0 | -1.43 | -133..140 | | | | stable |
| E1 exact setup | symp / delaunay / F | 0.25 | 35/200 | -> 9.2e3 | 10.6 | 0 | 24+168 | 4.1e4 | -781 | | | 500..3776 | 1.000/1.000 | BLOW-UP |
| E2 exact setup | symp / delaunay / T | 0.25 | 88/200 | -> 3.8e3 | 13.9 | 0 | 24+162 | 7.7e3 | 707 | | | | | BLOW-UP |
| E3 exact setup | symp / dual_only / F | 0.25 | 200 | -> 0.648 | 0.1 | - | 0 | 0 | -1.43 | | | 998..1002 | 1.000 | stable |
| **F1 settled** | symp / delaunay / F | 0.25 | 71/200 | 2.19 -> 7.6e3 | 11 | 1 | 0+313 | 5.0e4 | -2.9e3 | -5e4..1e5 | -5e4..1e5 | 419..3311 | 1.000/1.078 | BLOW-UP |
| F2 settled | symp / delaunay / T | 0.25 | 76/200 | -> 5.4e3 | 18.6 | 1 | 0+279 | 1.1e4 | 2.5e3 | | | | | BLOW-UP |
| **F3 settled** | symp / delaunay / F | **0.01** | 2087/5000 | -> 1.3e4 | 10 | 7 | 0+442 | 9.0e4 | 3.7e3 | | | | | BLOW-UP |
| F4 settled c_s 100 | symp / delaunay / F | 0.25 | 68/2000 | -> 1.1e6 | 102 | 1 | 0+344 | 3.3e6 | -5.7e4 | | | | | BLOW-UP |
| F5 settled wallsall | symp / delaunay / F | 0.25 | 200 | 1.88 -> 6.6e3 | 8.8 | 1 | 0+746 | 9.6e4 | -2.3e4 | | | | | BLOW-UP |
| F6 settled no EOS | symp / delaunay / F | 0.25 | 200 | 2.19 -> 0.751 | 0.1 | - | 0+14 | n/a | n/a | 0 | 0 | | | stable |
| F7 settled | symp / dual_only / F | 0.25 | 200 | 2.19 -> 0.553 | 0.1 | - | 0 | 0 | -1.61 | -118..111 | -293..298 | 997..1003 | 1.000 | stable |
| F8 settled long | symp / dual_only / F | 0.25 | 1000 | -> 0.335 | 0.1 | - | 0 | 0 | -1.25 | -567..384 | | 994..1004 | | stable |
| F9 wallsall long | symp / dual_only / F | 0.25 | 1000 | 1.88 -> 6.3e-5 | 0.086 | - | 0 | 0 | -1.87 | -94..101 | -276..280 | 997..1003 | | stable, decays |
| F10 wallsall long | symp / delaunay / F | 0.25 | 358/1000 | 1.88 -> 1.0e4 | 10.6 | 1 | 0+1464 | 2.0e5 | -5.9e4 | | | | | BLOW-UP |
| P8 exact setup long | symp / dual_only / F | 0.25 | 1000 | 2.08 -> 0.572 | 0.22 | 919 | 0 | 0 | -1.34 | -778..939 | | 981..1013 | | bounded (builder connectivity tangles late) |
| G cfl 0.5..1.5 (6 arms) | symp / dual_only / F, wallsall | 0.5, 0.75, 1.0, 1.25, 1.5 | full | 1.88 -> 0.038..0.041 | ≤0.084 | - | 0 | 0 | -1.83 | about ±100..130 | | | | stable |
| G cfl 2.0 | symp / dual_only / F, wallsall | 2.0 | 11/25 | -> 8.8e4 | 41 | 6 | 0 | 0 | +1.2e5 | | | | | BLOW-UP (acoustic limit) |
| G euler cfl 0.05 | euler / dual_only / F | 0.05 | 1000 | -> 0.044 | 0.086 | - | 0 | 0 | -1.82 | | | | | stable |
| G euler cfl 0.25 | euler / dual_only / F | 0.25 | 200 | -> 16.4 | 0.37 | 166 | 0 | 0 | +20.4 | | | | | GROWING |

Answers to the task's questions:

- **(a) Frozen connectivity with the EOS is stable but inert.** With
  `retopologize_fn=False`, V is never refreshed, so m/V and therefore p are
  constant (p = 0 to 1e-11). That arm tests nothing about EOS dynamics.
  Eulerian `euler_velocity_only` is also inert: its only volume change is
  the step-0 rebuild. That rebuild freezes a ±5e4 Pa field into the fixed
  mesh, which is why A9 blows up.
- **(b) `dual_only` with the EOS is stable in every arm.** This holds for
  c_s 10 and 100, n 1 and 7.15, CFL 0.01 to 1.5, redistribution on or off,
  and builder, exact or settled setup. Wall and interior pressures stay at
  about ±100 to 300 Pa, ρ stays within ±0.3 % of ρ0, dE_dyn < 0, and
  dE_retopo = 0.
- **(c) The wall half-cells do not drift under fixed connectivity.** Wall
  pressures stay the same order as interior pressures (A5, F7, F9). The
  walls matter only under reconnection: the library's single-phase
  redistribution excludes bV (`_is_redistributable`), so wall-cell volume
  jumps from flips survive. See §3.3 and prototype P2.

## 3. First divergence

### 3.1 Settled setup (F1, F3): the jump is at the first reconnection

In F3 (CFL 0.01), steps without a flip have dE_dyn between -0.09 and
-0.17 J per step (dissipative). Flip steps inject about ±2.4e3 to
4.9e3 J:

| step | flips | dE_retopo | dE_dyn | max\|dV/V\| from reconnection | where | max\|dp\| |
|---|---|---|---|---|---|---|
| 1 | 8 | +1.96e3 | -0.086 | 0.333 | interior (0.75, 0.75) | 3.3e4 (interior) |
| 2 | 8 | +4.88e3 | -0.148 | 0.500 | interior (0.5, 0.5), degree-8 centre | 5.0e4 |
| 3 | 0 | -9e-13 | -0.136 | 0 | - | 7.7 (wall) |
| 4 | 8 | -4.88e3 | -0.088 | 1.000 | (0.5, 0.5) | 5.0e4 |
| 7 | 8 | +4.88e3 | -0.166 | 0.500 | (0.5, 0.5) | 5.0e4 |

At step 1 the mean relative volume change from physical motion is 1.6e-5.
The reconnection change is 0.33, which is about 2e4 times larger. The first
vertices hit are interior fan centres of the structured grid's co-circular
squares (the degree-8 vertices and their neighbours), not specifically
boundary vertices. Walls (0, 0.5) and (0.75, 1) enter from step 3 or 5 at
CFL 0.25. The flips flutter back and forth: at (0.5, 0.5), dV/V goes
+0.5 / -1.0 / +0.5, so the energy injected and removed does not cancel,
because the EOS responds nonlinearly and the velocity field changes in
between. KE ratchets up about 2.2 -> 38 J in 30 steps at CFL 0.01, and
F3 reaches |u| > c_s at step 2086.

Static check (`diagnose ... flipjump`): four exactly co-circular points
with the diagonal switched give barycentric V_a = 0.33367 on one side and
0.66633 on the other. Barycentric volumes are discontinuous even at a
continuous Delaunay flip.

### 3.2 Probe setup (A1, A2, A9): an extra step-0 jump

The builder mesh is not Delaunay-canonical: qhull picks other diagonals, so
there are 24 flips at step 0. The 2D fallback volume also gives the
corners 0.0026 instead of the exact 0.0104. At the first rebuild, corner
density drops to 0.25ρ0, which is clipped to 0.5ρ0, giving p = -5e4 Pa.
This is also the "2-4 % single-phase volume leak localised to the
dual-volume refresh" from the debugging_plan A.3/S-lane notes: the
builder total is 0.96875 and the exact total is 1.0, a 3.1 % difference.
It accounts for the step-0 dE_retopo of 1.5e4 J in A2 and A9, but it is
not the root cause: E1 (exact setup) and F1 (settled setup) still blow
up.

### 3.3 Library redistribution (A1, F2)

`redistribute_mass_single_phase` snapshots the stale `v.p` from the last
force evaluation, which is taken before the move. Every rebuild therefore
projects the interior pressure back to its previous value and erases the
physical compression. What evolves is only the uniform scale and the wall
pressures, because walls are excluded from redistribution. In F2 every
max|dp| after the first flips is at wall vertices (2.5e4 Pa at steps 1
and 2). KE then grows by draining that injected wall PE (dE_dyn is about
-30 to +5 J per step, while KE rises 12 -> 900 J).

## 4. Hydrostatic_column 2D cross-check (no reconnection)

The Section 3 loop of `Hydrostatic_2D.py` is copied verbatim into
`diagnose hydro`. It is `_recompute_duals` + `cache_dual_volumes` on the
builder connectivity, 145 vertices, K = 9.81e5, c0 = 31.3, mu_art = 1591,
CFL 0.25. Flips = 0 in every variant.

| variant | outcome |
|---|---|
| case (shipped, fallback V) | KE grows from about 3 t_ac, 6.1 m/s at 4.4 t_ac, 21 m/s at 5.1 t_ac, 313 m/s = 10 c0 at 6.9 t_ac (matches the audit) |
| nogravity (fallback, exact equilibrium, roundoff seed) | exponential growth from roundoff, umax 1e-12 -> 1.6e-8 over 8 t_ac, located at the free-surface vertices (0.25..0.75, 1) |
| nogravity_seed (fallback, 1e-6 seed) | 3.4e-6 at 4 t_ac, 0.039 at 9.9 t_ac, blow-up at about 13 t_ac |
| nogravity_exact_seed (exact V, same seed) | decays: 1.3e-7 at 1.3 t_ac, 2e-10 at 14.8 t_ac |
| exact (with gravity, 40 t_ac) | settles: KE 1e-5 J, umax 6e-4 at 40 t_ac, interior p 598..9180 Pa |

Monitor: |V_fb - V_ex|/V_ex = 0.75 at the frozen corners from t = 0.
Interior vertices agree to 1e-13.

Root cause, from finite differences (`fd_volume_gradient.py`, refinement
3). Move the free-surface vertex (0.5, 1) outward by ds:

- fallback: dV_i/ds = 0.0104 and d(ΣV_nbrs)/ds = 0.1146
- exact: dV_i/ds = 0.0417 and d(ΣV_nbrs)/ds = 0.0833
- both totals: 0.125

The fallback polygon leaves out the vertex's own point, so the vertex does
not feel its own outward motion as expansion.

Variational check (`fd_variational.py`, jittered mesh, random p):

- closed fans: F_flux = Σ p_k ∂V_k/∂x_i to 1.3e-10 to 3.5e-10 relative
- open (free-surface) fans, exact V: 0.30 to 0.45 relative mismatch in the normal component
- open fans, fallback V: 0.75 to 1.12 relative mismatch

The exact-volume open-fan mismatch is proportional to the surface-vertex
pressure, which is about 0 at a free surface (p_top between -60 and 6 Pa in
the exact run). That is consistent with it being stable in practice, but
it is not proven energy-stable. It is the next thing to check for the
capillary-rise free surface.

**This rules out D1 as the sole cause:** Hydrostatic blows up without a
single reconnection, through mechanism 2.

## 5. Prototype: single-phase conservative remap (diagnose script only)

`remap_retopo` works in three stages:

1. Snapshot p_i = eos(m_i / V_i^old(x^{n+1})), using `simplex_dual_volumes` on the old `HC._simplices` at the new positions (`fresh`), or the stale `v.p` (`stale`, the library's semantics).
2. Run `_retopologize` (plain Delaunay, no redistribution).
3. Set m_i = ρ(p_i)·V_i^new on all vertices with V > 0 (`include_walls`) or on interior vertices only, then apply one global mass scale (`gauge='scale'`) or none.

It runs as `SolverMethods(connectivity='custom')` + `custom=partial(remap_retopo, eos=...)`.

| arm | snapshot / scope | result |
|---|---|---|
| P1 (exact setup), P9 (settled) | fresh / all incl. walls | stable. KE 2.19 -> 0.552, versus 0.553 for dual_only F7 (end within 0.1 %, max 11 % over the transient). ΣdE_re+ = 0.008 J |
| P13 | fresh / all, CFL 0.01 | stable, KE -> 0.559 |
| P7 | fresh / all, 5x horizon | stable, KE -> 0.454 |
| P10 | fresh / all, wallsall, 5x horizon | stable, KE 1.88 -> 5.9e-6 (dual_only F9: 6.3e-5) |
| P5 / P6 | fresh / all, c_s 100 / n 7.15 | stable (KE -> 0.397 / 0.552) |
| G_remap cfl 0.5 / 1.0 / 1.5 | fresh / all, wallsall | stable, same acoustic headroom as dual_only |
| P11 | fresh / all, gauge none | same as P9 (KE 0.552). The mass scale is a no-op in practice |
| **P2** | fresh / interior only | **BLOW-UP**: KE -> 743, \|u\| 5.4. Wall jumps survive |
| P3 / P12 | stale / all | stable but p ≡ 0 (±7e-10). The projection freezes the IC pressure, and KE (0.753) equals the no-EOS arm (0.751). Not compressible |
| P4 | Delaunay + circumcentric duals, barycentric setup | BLOW-UP: the step-0 volume-source jump plus unbounded circumcentric cells, V total up to 1.53 |
| P14 / P15 | circumcentric setup | setup fails: zero-volume boundary cells (right-angle corner triangles), ZeroDivisionError |

What is left after the library's scale, as the task asked: two things.
First, the snapshot must be the fresh old-connectivity pressure at the new
positions, not the stale v.p (otherwise the compression is erased).
Second, the walls must be included. There is no uniform offset worth
correcting (P11 = P9), and no separate treatment of the boundary
half-cells is needed beyond including them.

## 6. Acoustic stability boundary

Symplectic Euler on fixed connectivity (`dual_only`, settled, wallsall) is
stable for dt·c_s/dx_min ≤ 1.5 and blows up at 2.0 (u2 at step 6). This
is consistent with ω_max·dt < 2 for a symplectic oscillator. The probe's
0.25 has a 6x margin, so the template instability is not acoustic.
Forward Euler (`euler`) grows at CFL 0.25 (u2 at step 166; B2 is on the
same path) and survives the horizon at 0.05. Forward Euler is
non-dissipative-unstable for undamped acoustics.

## 7. Recommendation

Nothing below is implemented; this lane only measured.

1. **Single-phase conservative remap.** This is the fix for mechanism 1.
   - *Where:* `ddgclib/operators/mass_redistribution.py`. Add a
     `restore_pressure_single_phase` (or a `snapshot='fresh'`,
     `include_frozen=True` path in `redistribute_mass_single_phase`). In
     `ddgclib/dynamic_integrators/_integrators_dynamic.py:_retopologize`,
     when remapping, take the snapshot as eos(m / simplex_dual_volumes(old
     `HC._simplices`, current x)) BEFORE the rebuild, instead of
     `snapshot_pressure(HC)` (stale v.p). Re-target masses on every vertex
     with dual_vol > 0 including bV, then keep the existing global scale.
   - *Registry:* extend the explicit `remap` axis (`_axes.py`, currently
     `applies_to='multi'`, listed in `_MULTI_ONLY` in `_config.py`) to
     `phases='single'`. `remap='conservative'` requires
     `redistribute_mass=True`, a reconnecting connectivity, and
     `pressure_model` at integrator level. The existing validation rules
     already cover this.
   - *Evidence to cite:* P9 / P10 / P5 / P6 / P13 / G_remap.
2. **Mark single-phase `redistribute_mass=True` alone (bare Delaunay + EOS)
   as `broken`** in the `redistribute_mass` axis evidence, citing
   A1 / C5 / E2 / F2. `SolverMethods` should raise or warn for single-phase
   `connectivity in ('delaunay', 'adaptive')` + EOS in dudt without the
   remap. Registry status for single-phase bare Delaunay + EOS:
   measured-worse / DO-NOT (A2, B1, C1, C2, D1, D3, E1, F1, F3, F4, F5, F10).
3. **Setup volume consistency.** This fixes mechanism 2 and the step-0 jump.
   - Domain builders (`ddgclib/geometry/domains/`) or the setup helpers
     should populate `HC._simplices` (hyperct
     `rebuild_simplex_cache_2d(HC)`) before `compute_vd` /
     `DualVolumeMass`, so the ICs, hand-rolled loops and `dual_only`
     (which never builds the cache) all read `simplex_exact`.
   - The reported axis `dual_volume` value `dual_cell_area_2d`
     (`effective_methods`) should get status `broken` for moving or
     corner boundary vertices, with evidence from §4 (corner 4x,
     free-surface self-derivative 4x, Hydrostatic 2D blow-up vs the exact
     variant settling over 40 t_ac).
   - Longer term: fix or retire the fallback in
     `ddgclib/operators/stress.py:dual_volume` (dim == 2), or in
     hyperct `dual_cell_area_2d`, by adding the vertex point to the
     boundary polygon.
   - For `cases_dynamic/Hydrostatic_column/Hydrostatic_2D.py`, a one-line
     candidate is `rebuild_simplex_cache_2d(HC)` after
     `setup_hydrostatic_column` and before `DualVolumeMass` (variant
     `exact`). Not applied: it is a shipped case.
4. **Integrator:** with an EOS, `euler` needs a warning in `SolverMethods`
   (or a status note in the `integrator` axis). Symplectic Euler is fine up
   to CFL 1.5.
5. **Template:** once (1) exists, the template can demonstrate the EOS
   with `remap='conservative'`. Until then its docstring warning is
   correct.
6. **Capillary rise (free surface):** also needs (3). The remaining open
   question is the open-fan normal mismatch at free-surface vertices (§4,
   30 to 45 % with exact V). Test it on a sloshing free-surface box with
   the remap before trusting long runs.

## 8. DO-NOTs (measured)

- Do not shrink dt to fix the template instability: CFL 0.05 and 0.01 still blow up (C1, C2, F3), because flips occur per unit physical time.
- Do not stiffen the EOS (c_s 100, n 7.15) under reconnection: it blows up faster (D1, F4, D3 at step 0). Δp is proportional to K.
- Do not use the library single-phase `redistribute_mass=True` as the cure: A1, C5, E2 and F2 all blow up (stale snapshot, walls excluded).
- Do not remap interior only: P2 blows up.
- Do not use a stale-v.p snapshot, even with walls: it is stable but freezes p at the IC (P3, P12). It is a projection, not compressible flow.
- Do not read frozen or Eulerian EOS arms as evidence of EOS stability: the EOS is inert there (A7) or carries a frozen step-0 jump (A9).
- Do not switch to circumcentric duals as a quick fix: P4 blows up, and P14/P15 fail at setup with zero-volume boundary cells.
- Do not use `euler` with an EOS at CFL ≥ 0.25 (B2, G euler 0.25).
- Do not seed the IC masses from builder-mesh fallback volumes and then run the simplex-aware retopology: that gives the corner -5e4 Pa jump (A-arms, step 0).

## 9. Reproduce

```bash
PY=/home/endres/anaconda3/envs/ddg/bin/python
S=<scratchpad>/laneK
$PY cases_dynamic/template/diagnose_single_phase_eos.py flipjump
$PY cases_dynamic/template/diagnose_single_phase_eos.py matrix --out $S/matrix2 --procs 32   # about 5 min, 55 arms
$PY cases_dynamic/template/diagnose_single_phase_eos.py matrix --out $S/matrix2 --only F1 P9  # subset
$PY cases_dynamic/template/diagnose_single_phase_eos.py hydro --out $S/hydro2 --variant case|exact|nogravity|nogravity_seed|nogravity_exact_seed --n-tac 8
```

The default `--out` is `cases_dynamic/template/results/laneK`; this lane
passed the scratchpad instead. Arms that aborted at |u| > c_s report the
state at abort.
