"""
electrocapillarity_lippmann.py
=============================

Implementation of Section 3C.9 — Electrocapillarity via the Lippmann equation.
This module provides the Lippmann-modified electrode-electrolyte interfacial
tension and integrates cleanly with the existing bubble_enrtl_toy.py pipeline.

Key equation (Eq. C18 from the document):
    sigma_elec(E_cell) = sigma_0 - (1/2) * C_dl * (E_cell - E_pzc)**2

Where:
    sigma_0      : base surface tension from Butler + e-NRTL (Layer B)
    C_dl         : electric double-layer capacitance per unit area (F/m²)
    E_cell       : applied cell voltage (V)
    E_pzc        : potential of zero charge (V)

This sigma_elec is then used in the modified Fritz detachment balance (Eq. C19)
instead of the bare sigma in the contact-line anchoring term.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Default / typical parameters (from document Section 6.5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LippmannParams:
    """Electrocapillarity parameters for Lippmann model."""
    C_dl: float = 30.0 * 1e-6 / 1e-4          # 30 μF/cm² → F/m² (baseline Pt/KOH)
    E_pzc: float = -0.075                     # V vs RHE, typical for Pt in 1M KOH
    E_cell: float = -1.0                      # Applied cell voltage (cathodic, negative)


# ---------------------------------------------------------------------------
# Core Lippmann function (3C.9)
# ---------------------------------------------------------------------------
def lippmann_sigma_elec(
    sigma_0: float,
    E_cell: float,
    params: Optional[LippmannParams] = None
) -> float:
    """
    Compute Lippmann-modified interfacial tension sigma_elec (Eq. C18).

    Parameters
    ----------
    sigma_0 : float
        Base surface tension from Butler-eNRTL model (N/m).
    E_cell : float
        Applied cell voltage (V). More negative → stronger effect at cathode.
    params : LippmannParams, optional
        Capacitance and PZC parameters.

    Returns
    -------
    sigma_elec : float
        Modified surface tension (N/m). Always ≤ sigma_0.
    """
    if params is None:
        params = LippmannParams()

    # (E_cell - E_pzc) in volts
    delta_E = E_cell - params.E_pzc
    # Energy term: ½ C_dl (ΔE)² has units (F/m²) * V² = J/m² = N/m
    electrocapillary_reduction = 0.5 * params.C_dl * (delta_E ** 2)

    sigma_elec = sigma_0 - electrocapillary_reduction

    # Physical bound: tension cannot go negative (though unrealistic values
    # would be caught by the model user)
    return max(sigma_elec, 0.0)


# ---------------------------------------------------------------------------
# Convenience wrapper for use with existing butler_surface_tension
# ---------------------------------------------------------------------------
def butler_lippmann_surface_tension(
    m: float,
    salt,
    E_cell: Optional[float] = None,
    lippmann_params: Optional[LippmannParams] = None
) -> Tuple[float, float]:
    """
    Drop-in replacement / extension for butler_surface_tension(m, salt)
    that also applies the Lippmann correction.

    Returns (sigma_elec, x_s_S) where sigma_elec includes electrocapillarity.
    """
    from bubble_enrtl_toy import butler_surface_tension  # existing Layer B

    sigma_0, x_s_S = butler_surface_tension(m, salt)

    if E_cell is None:
        # No voltage applied → fall back to pure Butler
        return sigma_0, x_s_S

    sigma_elec = lippmann_sigma_elec(sigma_0, E_cell, lippmann_params)
    return sigma_elec, x_s_S


# ---------------------------------------------------------------------------
# Modified detachment volume that uses sigma_elec in anchoring force
# ---------------------------------------------------------------------------
def detachment_volume_lippmann(
    sigma_0: float,                    # from Butler
    theta_gas: float,
    E_cell: float,
    lippmann_params: Optional[LippmannParams] = None,
    rho_l: float = 997.05,
) -> dict:
    """
    Extended detachment_volume using Lippmann-modified anchoring force.

    Uses sigma_elec in the pinning term of the Fritz balance (Eq. C19).
    """
    from bubble_enrtl_toy import young_laplace_shape

    sigma_elec = lippmann_sigma_elec(sigma_0, E_cell, lippmann_params)

    def residual(b):
        shape = young_laplace_shape(sigma_elec, b, theta_gas, rho_l=rho_l)
        if not shape.get('converged', False):
            return 1e6
        F_buoy = (rho_l - 0.0899) * 9.80665 * shape['V']
        F_pin = 2.0 * np.pi * shape['r_cl'] * sigma_elec * np.sin(theta_gas)
        return F_buoy - F_pin

    # Reuse same bracketing strategy as original
    capillary_length = np.sqrt(sigma_elec / ((rho_l - 0.0899) * 9.80665))
    bs = np.linspace(0.1 * capillary_length, 2.5 * capillary_length, 20)

    residuals = [residual(b) for b in bs]
    residuals = np.array(residuals)

    valid = np.isfinite(residuals)
    idx = np.where(valid[:-1] & valid[1:] & (residuals[:-1] * residuals[1:] < 0))[0]

    if len(idx) == 0:
        # fallback: use original sigma_elec without full optimization
        return {
            'converged': False,
            'sigma_elec': sigma_elec,
            'note': 'No root found in bracket'
        }

    i = idx[0]
    from scipy.optimize import brentq
    b_star = brentq(residual, bs[i], bs[i + 1], rtol=1e-5)

    shape = young_laplace_shape(sigma_elec, b_star, theta_gas, rho_l=rho_l)
    V_d = shape['V']
    D_d = (6.0 * V_d / np.pi) ** (1.0 / 3.0)

    return {
        'converged': True,
        'b': b_star,
        'V': V_d,
        'D': D_d,
        'r_cl': shape['r_cl'],
        'H': shape['H'],
        'shape': shape,
        'sigma_elec': sigma_elec,
        'sigma_0': sigma_0,
        'E_cell': E_cell
    }


# ---------------------------------------------------------------------------
# Demo / test function
# ---------------------------------------------------------------------------
def demo_lippmann():
    """Quick demonstration of the Lippmann effect."""
    from bubble_enrtl_toy import KOH, butler_surface_tension
    import matplotlib.pyplot as plt

    m = 1.0  # mol/kg
    sigma_0, _ = butler_surface_tension(m, KOH)

    E_cells = np.linspace(-0.5, -2.0, 50)
    sigma_elec = [lippmann_sigma_elec(sigma_0, e) for e in E_cells]

    plt.figure(figsize=(6, 4))
    plt.plot(E_cells, 1000 * np.array(sigma_elec), 'b-', label='σ_elec (Lippmann)')
    plt.axhline(1000 * sigma_0, color='k', linestyle='--', label=f'σ₀ = {1000*sigma_0:.1f} mN/m')
    plt.xlabel('E_cell (V)')
    plt.ylabel('σ_elec (mN/m)')
    plt.title(f'Electrocapillarity at m = {m} mol/kg KOH\nC_dl = 30 μF/cm², E_pzc = {LippmannParams().E_pzc} V')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"sigma_0          = {sigma_0*1000:.2f} mN/m")
    print(f"sigma_elec @ -1V = {lippmann_sigma_elec(sigma_0, -1.0)*1000:.2f} mN/m")
    print(f"Reduction        = {(sigma_0 - lippmann_sigma_elec(sigma_0, -1.0))*1000:.2f} mN/m")


if __name__ == "__main__":
    demo_lippmann()
