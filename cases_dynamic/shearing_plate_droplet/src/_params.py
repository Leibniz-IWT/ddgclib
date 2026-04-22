"""Physical parameters for the shearing-plate droplet test case.

Geometry
--------
An oil droplet of radius ``R0`` sits at the centre of a rectangular (2D)
or cuboidal (3D) channel of half-height ``L_y`` filled with water.

- 2D: domain ``[-L_x, L_x] x [-L_y, L_y]``.
- 3D: domain ``[-L_x, L_x] x [-L_y, L_y] x [-L_z, L_z]``.

Boundary conditions
-------------------
- Top plate (``y = +L_y``):  no-slip moving wall at velocity ``+U_wall * e_x``.
- Bottom plate (``y = -L_y``): no-slip moving wall at velocity ``-U_wall * e_x``.
- x-direction (and z in 3D): periodic.

Steady-state far-field shear rate (absent the droplet):
``gamma_dot = U_wall / L_y``  (units 1/s).

Physical regime
---------------
Two dimensionless groups govern droplet deformation in simple shear:

- Capillary number:   ``Ca = mu_o * gamma_dot * R0 / gamma``
- Viscosity ratio:    ``lambda = mu_d / mu_o``
- Reynolds number:    ``Re = rho_o * gamma_dot * R0**2 / mu_o``

For ``Ca <~ 0.1`` the droplet relaxes to a steady ellipsoidal shape
(Taylor 1934 small-deformation regime).  Larger ``Ca`` can trigger
tip-streaming or break-up (not the target of this base case).

The default parameters below sit comfortably in the steady-deformation
regime (``Ca ~ 0.1``, ``Re ~ 1``, ``lambda ~ 1``) so the droplet tilts
and stretches without breaking.
"""
from __future__ import annotations

import numpy as np


# --- Droplet (phase 1) — light oil ---
rho_d = 900.0          # kg/m^3
mu_d = 0.05            # Pa.s

# --- Outer fluid (phase 0) — water ---
rho_o = 1000.0         # kg/m^3
mu_o = 0.05            # Pa.s  (lambda = mu_d / mu_o = 1.0)

# --- Interface ---
gamma = 0.03           # N/m (oil-water-ish surface tension)
R0 = 0.005             # m (5 mm equilibrium radius)

# --- Domain half-extents ---
# Aspect ratio ~ 3:1 (streamwise) x 2:1 (wall-normal) so the droplet
# feels roughly uniform far-field shear and its periodic images stay
# well clear of it.
L_x = 3.0 * R0         # streamwise periodic half-width
L_y = 2.0 * R0         # wall-normal half-height (plates at y = ±L_y)
L_z = 2.0 * R0         # spanwise periodic half-width (3D only)

# --- Wall plate velocity ---
# ``U_wall`` is the speed of each plate; plates move in opposite
# directions so the far-field flow is pure shear about y = 0.
U_wall = 0.05          # m/s
shear_rate = U_wall / L_y   # 1/s

# --- Dimensionless numbers ---
Ca = mu_o * shear_rate * R0 / gamma       # capillary number
visc_ratio = mu_d / mu_o                   # viscosity ratio
Re = rho_o * shear_rate * R0 ** 2 / mu_o  # Reynolds based on R0

# --- Timescales ---
# Capillary (relaxation) timescale for a droplet under shear.
tau_capillary = mu_o * R0 / gamma                       # s
tau_shear = 1.0 / shear_rate if shear_rate > 0 else np.inf
t_end_default = 8.0 * tau_shear                         # s (long enough to reach near-steady shape)

# --- EOS (weakly compressible) ---
# Sound speed floor selected to be >> U_wall so compressibility is small.
u_scale = max(U_wall, np.sqrt(gamma / (rho_o * R0)))
c_s = max(10.0 * u_scale, 1.0)
K_d = rho_d * c_s ** 2
K_o = rho_o * c_s ** 2
P0 = 0.0

# --- Mesh refinement defaults ---
n_refine_outer = 3
n_refine_droplet = 3

# --- Time stepping default (driver may override via CFL) ---
dt_default = 0.1 * R0 / c_s
