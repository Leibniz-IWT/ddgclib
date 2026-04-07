"""Physical parameters for dynamic capillary rise.

References
----------
Lunowa, S. B., Bringedal, C., & Pop, I. S. (2022). On an averaged model
    for the 2-fluid immiscible flow with surface tension in a thin
    cylindrical tube. IMA J. Appl. Math., 87(3), 387-413.
Heshmati, M., & Piri, M. (2014). Experimental investigation of dynamic
    contact angle and capillary rise in tubes with circular and
    noncircular cross sections. Langmuir, 30(47), 14151-14162.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Fluid property tables (SI units)
# ---------------------------------------------------------------------------
#   rho     : density [kg/m^3]
#   mu      : dynamic viscosity [Pa s]
#   gamma   : surface tension [N/m]
#   theta_s_deg : static contact angle of the liquid [deg]
#   radii_mm : tube radii tested experimentally [mm]

FLUIDS: dict[str, dict] = {
    "water": {
        "rho": 997.0,
        "mu": 0.0011,
        "gamma": 0.0728,
        "theta_s_deg": 9.99,
        "radii_mm": [0.375, 0.5, 0.65],
    },
    "soltrol": {
        "rho": 774.0,
        "mu": 0.0026,
        "gamma": 0.02483,
        "theta_s_deg": 9.79,
        "radii_mm": [0.375, 0.5, 0.65],
    },
    "glycerol": {
        "rho": 1260.0,
        "mu": 1.011,
        "gamma": 0.06346,
        "theta_s_deg": 5.63,
        "radii_mm": [0.25, 0.5, 1.0],
    },
}

g: float = 9.81  # gravitational acceleration [m/s^2]

# ---------------------------------------------------------------------------
# Default test case: water in a 0.5 mm radius tube
# ---------------------------------------------------------------------------
DEFAULT_FLUID: str = "water"
DEFAULT_RADIUS_MM: float = 0.5
DEFAULT_RADIUS: float = DEFAULT_RADIUS_MM * 1e-3  # [m]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_fluid(name: str) -> dict:
    """Return the fluid property dictionary for *name*.

    Raises ``KeyError`` if the fluid is not in the table.
    """
    return FLUIDS[name.lower()]


def jurin_height(
    r: float,
    gamma: float,
    theta_deg: float,
    rho: float,
    g: float = 9.81,
    dim: int = 3,
) -> float:
    """Equilibrium capillary rise height (Jurin's law).

    Parameters
    ----------
    r : float
        Tube radius [m].
    gamma : float
        Surface tension [N/m].
    theta_deg : float
        Static contact angle of the liquid [deg].
    rho : float
        Liquid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    dim : int
        Spatial dimension.  3 = cylindrical tube (two curvature
        directions), 2 = slit channel (one curvature direction).

    Returns
    -------
    float
        Equilibrium meniscus height *h* [m]:
        3D: ``h = 2 gamma cos(theta) / (rho g r)``
        2D: ``h = gamma cos(theta) / (rho g r)``
    """
    theta = math.radians(theta_deg)
    if dim == 3:
        return 2.0 * gamma * math.cos(theta) / (rho * g * r)
    elif dim == 2:
        return gamma * math.cos(theta) / (rho * g * r)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def capillary_pressure(
    r: float,
    gamma: float,
    theta_deg: float,
    dim: int = 3,
) -> float:
    """Young-Laplace capillary pressure jump across the meniscus.

    Parameters
    ----------
    r : float
        Tube radius [m].
    gamma : float
        Surface tension [N/m].
    theta_deg : float
        Static contact angle [deg].
    dim : int
        Spatial dimension (3 = cylindrical tube, 2 = slit channel).

    Returns
    -------
    float
        Pressure difference [Pa]:
        3D: ``DP = 2 gamma cos(theta) / r``,
        2D: ``DP = gamma cos(theta) / r``.
    """
    theta = math.radians(theta_deg)
    if dim == 3:
        return 2.0 * gamma * math.cos(theta) / r
    elif dim == 2:
        return gamma * math.cos(theta) / r
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def nondim_scales(
    r: float,
    gamma: float,
    theta_deg: float,
    mu: float,
    rho: float,
    g: float = 9.81,
) -> dict[str, float]:
    """Non-dimensional scales following Lunowa et al. (2022).

    Lunowa's convention measures the contact angle theta_s in the
    *receding* (non-wetting) fluid, so ``cos(theta_s) = -cos(theta_liquid)``.
    The characteristic length is defined so that it is positive for a
    rising meniscus:

        L = 2 sigma |cos(theta_s)| / (rho g R)
          = 2 sigma cos(theta_liquid) / (rho g R)

    Parameters
    ----------
    r : float
        Tube radius *R* [m].
    gamma : float
        Surface tension *sigma* [N/m].
    theta_deg : float
        Static contact angle of the **liquid** [deg].
    mu : float
        Dynamic viscosity [Pa s].
    rho : float
        Liquid density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    dict with keys:
        L       : characteristic length [m]
        U       : characteristic velocity [m/s]
        T       : characteristic time [s]
        P       : characteristic pressure [Pa]
        epsilon : aspect ratio R/L
        Re      : Reynolds number rho U L / mu
        I       : inertia parameter epsilon^2 Re
        Ca      : capillary number mu U / (sigma epsilon)
    """
    theta = math.radians(theta_deg)
    cos_theta = math.cos(theta)

    # Characteristic scales
    L = 2.0 * gamma * cos_theta / (rho * g * r)        # length [m]
    U = r**2 * rho * g / mu                              # velocity [m/s]
    T = L / U                                             # time [s]
    P = rho * g * L                                       # pressure [Pa]

    # Dimensionless groups
    epsilon = r / L                                       # aspect ratio
    Re = rho * U * L / mu                                 # Reynolds number
    I = epsilon**2 * Re                                   # inertia parameter
    Ca = mu * U / (gamma * epsilon)                       # capillary number

    return {
        "L": L,
        "U": U,
        "T": T,
        "P": P,
        "epsilon": epsilon,
        "Re": Re,
        "I": I,
        "Ca": Ca,
    }


def eos_params(
    r: float,
    fluid: str = "water",
) -> dict[str, float]:
    """Suggested TaitMurnaghan EOS parameters for weakly compressible flow.

    The bulk modulus *K* is chosen so that the artificial speed of sound
    ``c_s = sqrt(K / rho)`` is at least 10x the maximum expected velocity
    (capillary-gravity velocity scale), ensuring density variations below 1%.

    Parameters
    ----------
    r : float
        Tube radius [m].
    fluid : str
        Fluid name (key into ``FLUIDS``).

    Returns
    -------
    dict with keys ``K`` (bulk modulus [Pa]) and ``c_s`` (speed of sound [m/s]).
    """
    f = get_fluid(fluid)
    rho = f["rho"]
    mu = f["mu"]
    gamma = f["gamma"]

    # Velocity scale: Washburn / gravity-viscous scale R^2 rho g / mu
    U_ref = r**2 * rho * g / mu
    # Target Mach ~ 0.1 => c_s ~ 10 U_ref
    c_s = max(10.0 * U_ref, 1.0)  # at least 1 m/s floor
    K = rho * c_s**2

    return {"K": K, "c_s": c_s}
