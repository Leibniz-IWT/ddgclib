"""Physical parameters for the oscillating droplet test case.

Two parameter sets are provided:

- **Overdamped (high viscosity)**: Oil-like droplet in a viscous outer fluid.
  Monotonic decay of perturbation — easiest to validate.
- **Underdamped (water-like)**: Lower viscosity, oscillatory decay.
  Harder but shows frequency validation.

References
----------
- Lamb, H. (1932). *Hydrodynamics*, Cambridge University Press, §275.
- Rayleigh, Lord (1879). "On the capillary phenomena of jets".
  Proc. R. Soc. Lond. 29, 71–97.
- Prosperetti, A. (1980). "Free oscillations of drops and bubbles:
  the initial-value problem". J. Fluid Mech. 100(2), 333–347.
"""
import numpy as np


# =====================================================================
# Overdamped case (high viscosity — monotonic decay)
# =====================================================================

# Droplet (phase 1) — viscous oil
rho_d = 800.0          # kg/m³
mu_d = 0.5             # Pa·s  (high viscosity)

# Outer fluid (phase 0)
rho_o = 1000.0         # kg/m³
mu_o = 0.1             # Pa·s

# Interface
gamma = 0.05           # N/m (surface tension)
R0 = 0.01              # m (equilibrium radius)

# Perturbation (mode l=2 ellipsoidal deformation)
epsilon = 0.05         # relative amplitude: 5% of R0
l = 2                  # oscillation mode number

# Analytical (3D, mode l=2)
omega_sq_3d = l * (l - 1) * (l + 2) * gamma / (rho_d * R0**3)
beta_3d = (l - 1) * (2 * l + 1) * mu_d / (rho_d * R0**2)

# Analytical (2D, mode l=2)
omega_sq_2d = (l**3 - l) * gamma / (rho_d * R0**3)
beta_2d = (2 * l**2 - 1) * mu_d / (rho_d * R0**2)

# Timescales
T_osc_3d = 2 * np.pi / np.sqrt(omega_sq_3d) if omega_sq_3d > 0 else np.inf
tau_damp_3d = 1.0 / beta_3d if beta_3d > 0 else np.inf
T_osc_2d = 2 * np.pi / np.sqrt(omega_sq_2d) if omega_sq_2d > 0 else np.inf
tau_damp_2d = 1.0 / beta_2d if beta_2d > 0 else np.inf

# Simulation time (5 damping times)
t_end_3d = 5 * tau_damp_3d
t_end_2d = 5 * tau_damp_2d

# EOS (weakly compressible)
# Sound speed: 10× the max expected velocity scale.
# NOTE: raising the floor to 50 m/s cleans up acoustic reflections but
# tightens dt by ~10x — net effect on the oscillation score is weak
# once the real KE-growth bugs (interface stress) dominate.  Revisit
# this after Phase 6 (per-phase interface stress rewrite) lands.
u_scale = epsilon * R0 * max(np.sqrt(abs(omega_sq_3d)), beta_3d)
c_s = max(10.0 * u_scale, 1.0)   # floor at 1 m/s
K_d = rho_d * c_s**2             # bulk modulus (droplet)
K_o = rho_o * c_s**2             # bulk modulus (outer)

# Retopology policy for the 2D dynamic oscillation runner
# (oscillating_droplet_2D.py).  Lane-5 dynamic configuration sweep
# (2026-07-02, docs_temp/debug_session/lane5-dynamic-config-sweep.md)
# on the full 1839-step run at identical parameters:
#
#   policy                     l2_error  tail_growth  KE_max [J]
#   per-step Delaunay (old)    0.48992   1.72505      4.07e-02 (still growing)
#   dual_only (new default)    0.17857   0.99925      8.32e-07 (physical decay)
#
# 'dual_only' keeps the interface-conforming builder connectivity for
# the whole run (skip_triangulation=True in _retopologize_multiphase)
# while still refreshing duals, per-phase splits, mass redistribution
# and EOS pressures every step.  Per-step global Delaunay reconnection
# on the moving mesh injects spurious KE ~5e4x the physical level and
# is the dominant l2/tail error source; with dual_only the KE(t)
# envelope quantitatively matches the analytical overdamped decay
# (peak t 0.055 vs 0.060 s; tail decay rate 6.79 vs 7.18 1/s).
# Displacement-gated Delaunay (eps in {0.01,0.05,0.2}*h_min) and
# hybrid gate+dual-refresh policies were all worse than either
# extreme; see the lane log for the full sweep table.
# The pinned floor tests (test_case_oscillating_droplet.py) keep
# exercising the per-step Delaunay path via setup's default retopo_fn.
retopo_policy_2d = 'dual_only'

# Retopology policy for the 3D dynamic oscillation runner
# (oscillating_droplet_3D.py).  First 3D A/B with the Tier 3B score
# harness (2026-07-29, docs_temp/debug_session/laneB-3d-score-harness.md)
# on the full 872-step run at identical parameters (refine 2/2):
#
#   policy                     l2_error  tail_growth  mass_drift
#   per-step Delaunay (old)    1.52446   0.47067      3.87e-14
#   dual_only (new default)    0.24811   0.08410      1.91e-14
#
# Lane-5's caveat (3D skip-triangulation has different boundary-volume
# bookkeeping, batch_e_star zeroing) was verified BEFORE trusting the
# score: step-granular probe shows identical boundary count (96),
# identical |dV/V0| step-0 boundary-zeroing artefact (0.357050), exact
# simplex volumes engaged in both paths, machine-precision mass; the
# harness's dual_vol_*/n_interface_* fields confirm it on the full run
# (interface 98 constant, no boundary saturation in either policy).
# The pinned 3D floor tests keep exercising the per-step Delaunay path
# via setup's default retopo_fn.
retopo_policy_3d = 'dual_only'

# Domain size (outer box should be ≥ 5× droplet radius)
L_domain = 5.0 * R0

# Default mesh refinement
n_refine_outer = 3
n_refine_droplet = 3

# Time stepping
dt_default = 0.1 * R0 / c_s     # CFL-based default dt
n_steps_default = int(t_end_3d / dt_default) + 1
record_every = max(1, n_steps_default // 100)

# Young-Laplace pressure jump at equilibrium
delta_P_2d = gamma / R0           # 2D: ΔP = γ/R
delta_P_3d = 2 * gamma / R0       # 3D: ΔP = 2γ/R


# =====================================================================
# Underdamped case (water-like — oscillatory decay)
# =====================================================================

class UnderdampedParams:
    """Lower-viscosity parameter set for oscillatory validation."""
    rho_d = 1000.0       # kg/m³
    mu_d = 0.001         # Pa·s (water)
    rho_o = 1.225        # kg/m³ (air)
    mu_o = 1.81e-5       # Pa·s (air)
    gamma = 0.072        # N/m (water-air)
    R0 = 0.001           # m (1 mm droplet)
    epsilon = 0.05
    l = 2
    # omega_sq_3d ≈ 5.76e8 → ω ≈ 24000 rad/s → T ≈ 0.26 ms
    # beta_3d ≈ 5.0 → very underdamped
