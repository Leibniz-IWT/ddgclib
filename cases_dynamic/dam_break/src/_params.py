"""Physical parameters for the dam break test case.

Classic Martin–Moyce (1952) dam break geometry:

    y=H      +-----------------------+
             |        air            |
             |                       |
    y=a      +-----+                 |
             |     |                 |
             |water|      air        |
             |     |                 |
    y=0      +-----+-----------------+
             x=0   x=a              x=L

- Water column: ``a × a`` (square) in the lower-left corner
- Tank:         ``L × H``
- Gravity in -y direction

The column is released at t=0 (the virtual "dam" on its right side
vanishes) and collapses under gravity.  Surface tension at the
water–air interface opposes the collapse on small length scales.
"""
import numpy as np


# =====================================================================
# Geometry
# =====================================================================

# Reference length scale of the dam column (m)
a = 0.05

# Tank dimensions
L = 4.0 * a     # tank width along flow axis (x)
H = 2.0 * a     # tank height along gravity axis (y)

# Water column dimensions (lower-left corner of tank)
#
# NOTE(laneF-geometry): col_h was 2.0*a, which equals the tank height H
# — the "column" filled the tank lid-to-floor with NO air above it (the
# docstring diagram shows the column top at y=a with air above).  A
# full-height slab pinned between the frozen floor and lid rows cannot
# collapse as a dam break.  col_h = a restores the documented square
# Martin–Moyce column with headspace.
col_w = a       # width  (x direction)
col_h = a       # height (y direction)  -- square column, top at y = a < H

# 3D depth (out-of-plane)
W = 2.0 * a     # depth of the tank in z (3D cases)
col_d = a       # depth of the water column in z (3D cases)


# =====================================================================
# Fluids
# =====================================================================

# Liquid (water)
rho_l = 1000.0      # kg/m^3
mu_l = 1.0e-3       # Pa s

# Gas (air)
rho_g = 1.225       # kg/m^3
mu_g = 1.81e-5      # Pa s

# Surface tension (water–air)
gamma = 0.072       # N/m

# Gravity
g = 9.81            # m/s^2
gravity_axis = 1    # -y in 2D, -y in 3D (index 1)

# Atmospheric reference pressure
P_atm = 101325.0    # Pa


# =====================================================================
# EOS (weakly compressible)
# =====================================================================

# Sound speed: use 10x the expected max velocity (gravity driven dam break)
u_ref = np.sqrt(2.0 * g * col_h)     # ~ 1.4 m/s for col_h=0.1 m
c_s = max(10.0 * u_ref, 5.0)         # floor
K_l = rho_l * c_s**2                 # bulk modulus (liquid)
K_g = rho_g * c_s**2                 # bulk modulus (gas) -- stiff enough to
                                     # keep air weakly compressible


# =====================================================================
# Artificial viscosity
# =====================================================================
#
# Free-surface / interface corner vertices in the DDG FVM have
# truncated dual cells; the resulting force imbalance leads to
# spurious large accelerations at those corners.  We follow the
# Hydrostatic_column case and add an SPH-style artificial viscosity
# ``mu_art = alpha * rho * c_s * dx`` on top of the physical viscosity
# to damp those modes.  ``alpha = 0`` disables artificial viscosity.
#
# NOTE(laneF-alpha): default 2.0 -> 0.3 on the 2026-07-30 sweep.  At
# alpha=2.0 the effective liquid viscosity is ~276 Pa s (276,000x
# water): the collapse creeps at |u| ~ 0.015 m/s, the front moves
# 1/8 of an edge length over 10x the shipped horizon, and Delaunay
# reconnection NEVER fires — the case cannot demonstrate the physics
# it exists for.  Sweep (refine 3, t_end 0.2, remap ON): 0.5 survives
# (|u| 0.059, 2 flips), 0.4 survives (0.076, 2 flips), 0.3 survives
# the full horizon with 6 reconnection events absorbed (|u| 0.112,
# front +36% of col_w, KE rise-then-fall), 0.2 aborts at t=0.16,
# 0.1 aborts at t=0.094 (air sliver-cell F/m ejection at reconnection
# — the corner-vertex defect this crutch papers over).  0.3 is the
# smallest surviving value; without the conservative retopo remap the
# same alpha=0.3 run blows up at t=0.125.
alpha_art = 0.3


# =====================================================================
# Time integration
# =====================================================================

# Characteristic timescale of a dam break
t_ref = np.sqrt(col_h / g)           # ~ 0.1 s for col_h = 0.1 m

# End time.  NOTE(laneF-horizon): 0.02 -> 0.2 (~2.8 t_ref).  The old
# 0.02 s horizon was 1/8 of the gravity ramp time — no configuration
# can show a collapse there.  0.2 s is the horizon the laneF remap
# A/B and endurance runs measured clean (KE peak at t ~ 0.05 then
# decay; mass drift ~ 6e-15; reconnection events absorbed by
# ``retopo_remap='conservative'``).  The air-side sliver-cell F/m
# ejection defect (laneF log §4) still bounds refine=4 runs and
# alpha_art <= 0.2 — fix that before pushing to ``4 * t_ref`` or
# finer meshes.
t_end = 0.2                          # ~ 2.8 * t_ref

# CFL safety factor
cfl = 0.1


# =====================================================================
# Mesh refinement
# =====================================================================

n_refine_2d = 3
n_refine_3d = 2
