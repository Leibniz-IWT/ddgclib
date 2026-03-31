"""EOS-based pressure update for mesh vertices."""
from __future__ import annotations

from ddgclib.eos._base import EquationOfState


def eos_pressure_update(HC, eos: EquationOfState, dim: int = 3) -> None:
    """Update ``v.p`` on all vertices using the equation of state.

    For each vertex *i*:

        rho_i = m_i / dual_vol_i
        v.p   = eos.pressure(rho_i)

    Also sets ``v.rho`` for diagnostic access.

    Requires ``compute_vd`` and ``cache_dual_volumes`` to have been
    called so that ``v.dual_vol`` is available.

    Parameters
    ----------
    HC : Complex
        Simplicial complex with duals computed.
    eos : EquationOfState
        Equation of state instance.
    dim : int
        Spatial dimension.
    """
    from ddgclib.operators.stress import _get_dual_vol

    for v in HC.V:
        vol_i = _get_dual_vol(v, HC, dim)
        if vol_i < 1e-30:
            v.rho = eos.rho0
            v.p = float(eos.pressure(eos.rho0))
            continue
        v.rho = v.m / vol_i
        v.p = float(eos.pressure(v.rho))
