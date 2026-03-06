"""Physical parameters for the CFD-DEM liquid bridge case study.

Re-exports all DEM parameters from the pure-DEM case and adds fluid
film properties for the surface-conforming wetting film.
"""

# Re-export every DEM parameter unchanged
from cases_dynamic.liquid_bridge_dem.src._params import (  # noqa: F401
    R, rho_s, gamma, theta, liquid_volume,
    E, nu, gamma_n,
    v_approach, initial_sep,
    dt, n_steps, record_every,
    m, approach_time_est,
    bridge_volume_fraction, bridge_volume, rupture_dist, formation_dist,
)
from cases_dynamic.liquid_bridge_dem.src._params import (
    print_params as _print_params_dem,
)

# Fluid parameters (water at 25 C)
rho_f = 1000.0      # fluid density [kg/m^3]

# Fluid film parameters
film_thickness = 1e-5    # wetting film thickness [m] (10 um)
film_refinement = 2      # mesh refinement for film surfaces

# Surface mesh edge length bounds for remeshing
film_min_edge = 5e-5     # minimum edge length [m] (50 um)
film_max_edge = 5e-4     # maximum edge length [m] (500 um)

# Velocity damping for numerical stability
film_damping = 1e-3      # damping coefficient [kg/s]

# Bridge formation threshold (connect rim vertices within this distance)
bridge_threshold = 2e-4  # [m] (200 um)

# Integration sub-stepping
n_fluid_sub = 10         # fluid sub-steps per DEM step
dt_fluid = dt / n_fluid_sub  # fluid time step

# Macro-stepping
n_macro_steps = n_steps  # total macro steps (same as DEM)


def print_params():
    """Print all case parameters (DEM + fluid film)."""
    _print_params_dem()
    print()
    print(f"Fluid density rho_f   = {rho_f:.0f} kg/m^3")
    print(f"Film thickness        = {film_thickness*1e6:.0f} um")
    print(f"Film refinement       = {film_refinement}")
    print(f"Film edge range       = [{film_min_edge*1e6:.0f}, "
          f"{film_max_edge*1e6:.0f}] um")
    print(f"Film damping          = {film_damping:.2e}")
    print(f"Bridge threshold      = {bridge_threshold*1e6:.0f} um")
    print(f"Fluid sub-steps       = {n_fluid_sub}")
    print(f"Fluid dt              = {dt_fluid:.2e} s")
