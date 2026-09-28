"""Physical parameters for the two-particle liquid bridge case study.

All quantities in SI units.

Setup: Two identical silica glass spheres approach each other in vacuum
(no gravity, no fluid mesh). Both are wetted with water. When they come
close enough, a capillary liquid bridge forms, pulling them together.
Hertz contact repulsion prevents overlap, resulting in agglomeration at
a small equilibrium gap.
"""

import numpy as np

# Particle geometry
R = 1e-3          # particle radius [m] (1 mm)
rho_s = 2500.0    # solid density [kg/m^3] (silica glass)

# Liquid / wetting
gamma = 0.072     # surface tension [N/m] (water at 25 C)
theta = np.radians(30)   # contact angle [rad]
liquid_volume = 1e-9      # liquid volume per particle [m^3]

# Contact mechanics (Hertz model, silica glass)
E = 70e9          # Young's modulus [Pa]
nu = 0.22         # Poisson's ratio
gamma_n = 0.01    # normal viscous damping [N s/m]

# Initial conditions
v_approach = 0.005  # approach velocity per particle [m/s]
initial_sep = 0.8e-3  # initial surface-to-surface gap [m]
# Particles placed at x = -(R + initial_sep/2) and x = +(R + initial_sep/2)

# Time stepping
dt = 1e-6           # time step [s]
n_steps = 100_000   # total steps (100 ms; bridge forms ~30 ms, settling ~70 ms)

# Recording
record_every = 200  # save history snapshot every N steps

# Derived quantities
m = rho_s * (4.0 / 3.0) * np.pi * R**3   # particle mass [kg]
approach_time_est = initial_sep / (2 * v_approach)  # estimated approach time [s]
bridge_volume_fraction = 1.0  # fraction of particle liquid used per bridge
bridge_volume = bridge_volume_fraction * liquid_volume
rupture_dist = (1 + 0.5 * theta) * bridge_volume**(1.0 / 3.0)  # Lian et al. 1993
formation_dist = 0.5 * R  # default in LiquidBridgeManager


def print_params():
    """Print all case parameters."""
    print("=" * 60)
    print("  Two-Particle Liquid Bridge Case Study")
    print("=" * 60)
    print(f"Particle radius R     = {R*1e3:.1f} mm")
    print(f"Solid density rho_s   = {rho_s:.0f} kg/m^3")
    print(f"Particle mass m       = {m:.4e} kg")
    print()
    print(f"Surface tension gamma = {gamma:.3f} N/m")
    print(f"Contact angle theta   = {np.degrees(theta):.0f} deg")
    print(f"Liquid volume / part. = {liquid_volume:.2e} m^3")
    print(f"Rupture distance      = {rupture_dist:.4e} m")
    print()
    print(f"Young's modulus E     = {E:.2e} Pa")
    print(f"Poisson's ratio nu    = {nu}")
    print(f"Normal damping        = {gamma_n:.2e} N s/m")
    print()
    print(f"Approach velocity     = {v_approach} m/s (each particle)")
    print(f"Initial gap           = {initial_sep*1e3:.1f} mm")
    print(f"Est. approach time    = {approach_time_est*1e3:.1f} ms")
    print()
    print(f"Time step dt          = {dt:.2e} s")
    print(f"Total steps           = {n_steps}")
    print(f"Total sim time        = {n_steps * dt * 1e3:.1f} ms")
    print(f"Record every          = {record_every} steps")
    print("=" * 60)
