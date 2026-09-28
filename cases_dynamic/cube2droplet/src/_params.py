"""Physical parameters for the cube-to-droplet relaxation case.

A square (2D) or cube (3D) droplet relaxes to a circle/sphere
under surface tension, with viscous damping.

The overdamped regime (high viscosity relative to surface tension)
is used so the relaxation is monotonic.
"""
import numpy as np

# =====================================================================
# Geometry
# =====================================================================
R = 0.01                 # half-side of the square/cube droplet [m]
L_domain = 0.03          # half-side of the outer domain [m] (3*R)

# Equivalent circular/spherical radius (same area/volume)
R_eq_2d = R * 2.0 / np.sqrt(np.pi)      # sqrt(4*R^2 / pi)
R_eq_3d = R * (6.0 / np.pi) ** (1/3)    # (8*R^3 * 3/(4*pi))^(1/3)

# =====================================================================
# Material properties
# =====================================================================
# Droplet (phase 1)
rho_d = 800.0            # kg/m^3
mu_d = 0.5               # Pa*s

# Outer fluid (phase 0)
rho_o = 1000.0           # kg/m^3
mu_o = 0.2               # Pa*s

# Interface
gamma = 0.01             # N/m (surface tension)

# =====================================================================
# EOS (soft weakly compressible)
# =====================================================================
P0 = 0.0                 # reference pressure [Pa]
K_d = 800.0              # bulk modulus (droplet) [Pa]
K_o = 1000.0             # bulk modulus (outer) [Pa]

# =====================================================================
# Mesh refinement
# =====================================================================
n_refine = 4             # 2D refinement level
n_refine_3d = 2          # 3D refinement level

# =====================================================================
# Time stepping
# =====================================================================
dt_default = 1e-5        # [s]
u_clamp = 0.5            # max velocity [m/s]
retopo_every = 500       # retopologize every N steps
