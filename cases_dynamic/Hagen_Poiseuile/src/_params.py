import numpy as np

# Geometry (SI units)
r = 0.5   # tube radius [m]
D = 2 * r   # tube diameter [m]
L = 15.0   # tube length [m]

# Flow conditions (SI units)
Re_D = 100   # Reynolds number based on diameter and average velocity (dimensionless)
U_avg = 0.1   # average inlet velocity (uniform plug) [m/s]
U_max = 2 * U_avg   # expected fully developed centerline velocity [m/s]

# Fluid properties (SI units)
rho = 1.0   # fluid density [kg/m³]
mu = (rho * U_avg * D) / Re_D   # dynamic viscosity [Pa·s]

# Derived quantities (SI units)
L_e_approx = 0.06 * Re_D * D   # theoretical entrance length (99% developed) [m]
G = 8 * mu * U_avg / r**2   # axial pressure gradient for fully developed flow [Pa/m]

# Mesh & numerical parameters
refinements = 3   # refinement levels in cube_to_tube
cdist = 1e-10   # vertex merging tolerance [m]
inlet_layer_thickness = 0.15   # axial thickness of each new inlet layer [m]
outlet_buffer = 0.5   # remove vertices beyond L + outlet_buffer [m]

# Time stepping & inlet/outlet control
add_inlet_every = 20   # add new inlet layer every N timesteps
CFL_target = 0.4   # target CFL number (dimensionless)

def print_params():
    print("=== Hagen–Poiseuille Developing Flow Configuration ===")
    print(f"Radius r          = {r} m")
    print(f"Diameter D        = {D} m")
    print(f"Tube length L     = {L} m  (L/D = {L/D:.1f})")
    print(f"Re_D              = {Re_D}")
    print(f"Entrance length   ≈ {L_e_approx:.2f} m  (L_e/D ≈ {L_e_approx/D:.1f})")
    print(f"Inlet U_avg       = {U_avg} m/s")
    print(f"Developed U_max   = {U_max} m/s")
    print(f"Density ρ         = {rho} kg/m³")
    print(f"Viscosity μ       = {mu:.5f} Pa·s")
    print(f"Pressure gradient G = {G:.5f} Pa/m")

