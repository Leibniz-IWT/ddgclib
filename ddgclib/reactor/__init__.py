"""Reduced-order electrolysis reactor models for RL control studies.

This package implements a hierarchical RL + PID control architecture
for a Mars-based alkaline water electrolyser.  The Lippmann
electrocapillarity model (Eq. C18) is the primary electrostatic
coupling mechanism, with E_cell as the RL action variable.

Modules
-------
_lippmann     : Lippmann electrocapillarity functions
_pid          : PID controller with anti-windup
_electrolyser : Reduced-order electrolyser plant model
_gym_env      : Gymnasium RL environment
train_rl      : Training script (PPO / SAC)
"""

from ddgclib.reactor._lippmann import sigma_lippmann, LippmannParams, lippmann_detachment_scaling
from ddgclib.reactor._pid import PIDController, PIDParams
from ddgclib.reactor._electrolyser import Electrolyser, ElectrolyserParams

__all__ = [
    "sigma_lippmann",
    "LippmannParams",
    "lippmann_detachment_scaling",
    "PIDController",
    "PIDParams",
    "Electrolyser",
    "ElectrolyserParams",
]
