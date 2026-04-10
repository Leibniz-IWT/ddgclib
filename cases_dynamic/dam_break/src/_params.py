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
col_w = a       # width  (x direction)
col_h = 2.0 * a # height (y direction)  -- classic 2:1 aspect dam

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
alpha_art = 2.0


# =====================================================================
# Time integration
# =====================================================================

# Characteristic timescale of a dam break
t_ref = np.sqrt(col_h / g)           # ~ 0.1 s for col_h = 0.1 m

# End time (target).  The full dam break traversal would be
# ``4 * t_ref ~ 0.4 s`` but the multiphase DDG pipeline currently
# develops spurious accelerations at free-surface / interface corners
# (see FEATURES.md "AMR remeshing" and "multiphase dual volume
# splitting" sections).  A short default run is used here so the case
# runs end-to-end out of the box; increase ``t_end`` and tune
# ``alpha_art`` / ``n_refine`` for production runs.
t_end = 0.02                         # ~ 0.2 * t_ref

# CFL safety factor
cfl = 0.1


# =====================================================================
# Mesh refinement
# =====================================================================

n_refine_2d = 3
n_refine_3d = 2
