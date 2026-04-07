"""Equation of State (EOS) module for ddgclib.

Provides thermodynamic equations of state mapping density to pressure
for weakly compressible fluid simulations.

Usage
-----
    from ddgclib.eos import TaitMurnaghan, eos_pressure_update

    eos = TaitMurnaghan(rho0=1000.0, P0=101325.0, K=2.15e9, n=7.15)
    P = eos.pressure(rho=1001.0)
    rho = eos.density(P=102000.0)

    # Update pressure on all vertices from dual volumes:
    eos_pressure_update(HC, eos, dim=2)
"""

from ddgclib.eos._base import EquationOfState
from ddgclib.eos._tait_murnaghan import TaitMurnaghan
from ddgclib.eos._ideal_gas import IdealGas
from ddgclib.eos._multiphase_eos import MultiphaseEOS
from ddgclib.eos._update import eos_pressure_update

__all__ = [
    'EquationOfState',
    'TaitMurnaghan',
    'IdealGas',
    'MultiphaseEOS',
    'eos_pressure_update',
]
