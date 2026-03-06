"""Setup function for the two-particle liquid bridge case study."""

import numpy as np

from ddgclib.dem import (
    Particle,
    ParticleSystem,
    ContactDetector,
    HertzContact,
    LiquidBridgeManager,
)
from . import _params as P


def setup_liquid_bridge(
    R=None, rho_s=None, gamma=None, theta=None, v_approach=None,
    liquid_volume=None, E=None, nu=None, gamma_n=None, initial_sep=None,
):
    """Set up the two-particle liquid bridge scenario.

    All parameters default to values in ``_params.py``. Pass overrides
    to explore parameter sensitivity.

    Returns
    -------
    ps : ParticleSystem
    detector : ContactDetector
    contact_model : HertzContact
    bridge_mgr : LiquidBridgeManager
    """
    R = R if R is not None else P.R
    rho_s = rho_s if rho_s is not None else P.rho_s
    gamma = gamma if gamma is not None else P.gamma
    theta = theta if theta is not None else P.theta
    v_approach = v_approach if v_approach is not None else P.v_approach
    liquid_volume = liquid_volume if liquid_volume is not None else P.liquid_volume
    E = E if E is not None else P.E
    nu = nu if nu is not None else P.nu
    gamma_n = gamma_n if gamma_n is not None else P.gamma_n
    initial_sep = initial_sep if initial_sep is not None else P.initial_sep

    # Particle positions: centred at origin, separated by initial_sep
    x_offset = R + initial_sep / 2.0

    ps = ParticleSystem(dim=3, gravity=np.zeros(3))
    p1 = ps.add(Particle.sphere(
        x=[-x_offset, 0, 0], radius=R, rho_s=rho_s, dim=3,
        u=np.array([v_approach, 0, 0]),
        wetted=True, wetting_angle=theta, liquid_volume=liquid_volume,
    ))
    p2 = ps.add(Particle.sphere(
        x=[x_offset, 0, 0], radius=R, rho_s=rho_s, dim=3,
        u=np.array([-v_approach, 0, 0]),
        wetted=True, wetting_angle=theta, liquid_volume=liquid_volume,
    ))

    detector = ContactDetector(ps)
    contact_model = HertzContact(E=E, nu=nu, gamma_n=gamma_n)
    bridge_mgr = LiquidBridgeManager(
        gamma=gamma, bridge_volume_fraction=1.0,
    )

    return ps, detector, contact_model, bridge_mgr
