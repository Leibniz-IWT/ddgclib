"""Physical parameters for the electrolysis-bubble case.

Represents a hydrogen gas bubble growing on a flat electrode at the
bottom of a column of electrolyte.  For the proof-of-concept run we:

- use water/air surface tension but softened bulk moduli so the CFL
  limit is tractable (same philosophy as ``oscillating_droplet``),
- pick a density contrast of ~100 between liquid and gas (large enough
  to drive buoyant detachment, small enough to keep the problem stiff
  but finite),
- use a linear gas-generation rate (placeholder for the electrolysis
  reaction rate -- no charge-transport / Butler-Volmer coupling yet).

Values are quoted in SI units throughout.
"""
from __future__ import annotations


# -- Phase properties ---------------------------------------------------------

# Liquid electrolyte (water-like).  ``mu_liq`` is *much* higher than
# the physical 1e-3 Pa s of water: at the mesh resolution we run here
# (dx ~ 60 um in 2D), the Reynolds number of even a slow millimetre-
# per-second interface motion is far out of the viscous-stable regime
# for an explicit Lagrangian scheme.  Oscillating-droplet and
# cube2droplet both run with mu_liq ~ 0.1 for the same reason.
rho_liq = 1000.0        # kg/m^3
mu_liq = 0.1            # Pa s  (raised from 1e-3 for numerical stability)
K_liq = 1.0e5            # Pa  (softened; real water ~2.2e9)

# Hydrogen gas phase.  rho_gas is boosted above real H2 (0.09) to keep
# the liquid/gas mass ratio bounded; behaviour of interest is still
# driven by the Bond number rather than the absolute density contrast.
# mu_gas is 10 x higher than real H2 for the same reason (it's still
# 10 x *lower* than mu_liq, preserving the physically correct ordering).
rho_gas = 10.0           # kg/m^3
mu_gas = 1.0e-2          # Pa s  (raised from 8.9e-6 for stability,
                         # still 10 x lower than mu_liq)
# K_gas is set *equal* to K_liq so both phases live in the
# weakly-compressible regime where the EOS never produces negative
# pressure.  A softer K_gas (like a real gas) would let the bubble
# inflate unbounded whenever the local pressure briefly dipped below
# the EOS reference density -- in a Lagrangian mesh that happens
# during any natural oscillation.  Keeping both K's equal also means
# we only have one sound speed to respect in the CFL limit.
K_gas = 1.0e5            # Pa  (matched to liquid for stability)

# Interfacial tension (water-air at 20 C)
gamma = 0.072            # N/m

# Gravity.  Using real g = 9.81 m/s^2 with the parameters above yields a
# capillary length lambda = sqrt(gamma/(rho_liq*g)) ~ 2.7 mm and a
# Fritz detachment radius of a few millimetres -- matching the domain.
g = 9.81                 # m/s^2


# -- Geometry -----------------------------------------------------------------

R0 = 1.0e-3              # m, initial bubble radius (1 mm)
L_domain = 4.0e-3        # m, half-side of outer box.  Box is 8 mm x 8 mm.
                         # Kept compact because slow growth over a large
                         # domain gives Delaunay retopology more chances
                         # to degrade the sharp multiphase interface.

# Nucleation offset.  Fraction of (L_domain - 2*R0) to drop the bubble
# below the domain centre at t=0.  The bubble sits at
# ``y_center = -nucleation_frac * (L_domain - 2 R0)`` with the
# electrode at ``y = -L_domain``.
nucleation_frac = 0.5

# Mesh refinement.
n_refine_outer_2d = 2
n_refine_drop_2d = 3

n_refine_outer_3d = 1
n_refine_drop_3d = 1

# Reference pressure.  Ambient at the top of the domain.
P0 = 0.0


# -- Reaction (placeholder) ---------------------------------------------------

# Linear gas mass-generation rate.  Tuned so the bubble grows from R0
# to roughly the Fritz detachment radius within the simulation window.
#
# Physical motivation: for a planar electrode at current density j,
# the molar hydrogen production is j / (2 F).  Mapping this to a
# per-electrode-area mass rate and then spreading it over the whole
# gas phase is the simplification we are using here.  We will replace
# this placeholder with a real reaction-diffusion coupling later.
dm_dt_2d = 2.0e-2        # kg / (s * m)  -- per unit depth in 2D
dm_dt_3d = 2.0e-5        # kg / s        -- per bubble in 3D


# -- Run control --------------------------------------------------------------

# Physical simulation time (seconds).  Tuned so that the bubble
# demonstrably grows AND rises under buoyancy within the window.
t_end_2d = 2.0e-4
t_end_3d = 1.5e-4

# CFL safety factor on the dt = cfl * dx / c_s estimate.
cfl_safety = 0.05
