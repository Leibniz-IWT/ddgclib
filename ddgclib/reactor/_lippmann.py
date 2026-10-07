"""
Lippmann electrocapillarity model for electrode-electrolyte interfacial tension.

Implements Eq. C18 from the BubbleModel_Electrostatic_v0.2 modelling document:

    σ_elec(E_cell) = σ_0 − ½ · C_dl · (E_cell − E_pzc)²

This model was selected over DEP (Model 2) and GCS Coulomb (Model 1) because:

* DEP produces changes of ~10⁻⁹ % at realistic *j* (completely negligible).
* GCS Coulomb has 9× uncertainty from zeta potential and physical
  inconsistency at high ionic strength (Debye length ~0.43 nm vs mm-scale
  bulk field).
* Lippmann reduces V_d by 26.8 % at ΔV = 0.30 V, 66.8 % at ΔV = 0.50 V.

Mars applicability (modelling document §3.4):
    At ΔV = 0.40 V, Lippmann reduces Martian V_d by 45.6 %,
    partially compensating the 4.28× gravity penalty.

References
----------
[1] Lippmann, G. Ann. Chim. Phys. 5, 494 (1875).
[2] Hamelin, A. J. Electroanal. Chem. 142, 299 (1985).
[3] Modelling document §2.3, §3, §5.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LippmannParams:
    """Lippmann electrocapillarity parameters.

    Default values: Pt electrode in 1 M KOH at 298 K.
    Literature range: C_dl = 20–50 µF/cm² ; E_pzc = −0.05 to −0.10 V vs RHE.

    Attributes
    ----------
    C_dl : float
        Double-layer capacitance [F/m²].  Default 0.30 (= 30 µF/cm²).
    E_pzc : float
        Potential of zero charge [V vs RHE].  Default −0.07 V for Pt/KOH.
    delta_V_max : float
        Maximum |E_cell − E_pzc| allowed [V].  Electrowetting saturation
        occurs at ~0.5–0.6 V; the Lippmann equation predicts σ < 0 at
        |ΔV| = √(2σ₀/C_dl) ≈ 0.693 V.  Clamped to avoid unphysical values.
    """

    C_dl: float = 0.30          # F/m²  (30 µF/cm²)
    E_pzc: float = -0.07       # V vs RHE  (Pt in KOH)
    delta_V_max: float = 0.50   # V  (electrowetting saturation limit)


# ---------------------------------------------------------------------------
# Core Lippmann function  (Eq. C18)
# ---------------------------------------------------------------------------
def sigma_lippmann(
    sigma_butler: float,
    E_cell: float,
    params: LippmannParams | None = None,
) -> float:
    """Compute Lippmann-modified interfacial tension (Eq. C18).

    Parameters
    ----------
    sigma_butler : float
        Base surface tension from Butler / e-NRTL model [N/m].
    E_cell : float
        Applied cell voltage [V].
    params : LippmannParams, optional
        Capacitance and PZC parameters.  Uses Pt/KOH defaults if *None*.

    Returns
    -------
    sigma_elec : float
        Modified electrode-electrolyte interfacial tension [N/m].
        Always ≥ 0 (clipped at electrowetting saturation).

    Notes
    -----
    The modelling document (§4, L3) warns that σ_elec → 0 at
    |ΔV| = √(2σ₀/C_dl) ≈ 0.693 V.  Real systems saturate around
    0.5–0.6 V.  The *delta_V_max* parameter enforces this limit.

    Examples
    --------
    >>> sigma_lippmann(0.07197, -0.07 + 0.30)   # ΔV = 0.30 V
    0.05847...
    """
    if params is None:
        params = LippmannParams()

    delta_V = E_cell - params.E_pzc
    # Clamp to electrowetting saturation limit
    delta_V = float(np.clip(delta_V, -params.delta_V_max, params.delta_V_max))

    sigma_elec = sigma_butler - 0.5 * params.C_dl * delta_V ** 2
    return max(sigma_elec, 0.0)


# ---------------------------------------------------------------------------
# Detachment-volume / frequency scaling ratios  (modelling document §2.3)
# ---------------------------------------------------------------------------
def lippmann_detachment_scaling(
    sigma_0: float,
    sigma_elec: float,
) -> dict:
    """Compute detachment-volume and departure-frequency scaling ratios.

    From the modelling document §2.3 (Modified Fritz, Eq. 15):

        R_d(Lipp) / R_Fritz = √(σ_elec / σ₀)
        V_d(Lipp) / V_Fritz = (σ_elec / σ₀)^{3/2}
        Δf_d / f_d          = (σ₀ / σ_elec)^{3/2} − 1

    Parameters
    ----------
    sigma_0 : float
        Baseline (Butler / pure-water) surface tension [N/m].
    sigma_elec : float
        Lippmann-modified surface tension [N/m].

    Returns
    -------
    dict
        Keys: ``ratio_R``, ``ratio_V``, ``delta_f_rel``.
    """
    if sigma_0 <= 0 or sigma_elec <= 0:
        return {"ratio_R": 0.0, "ratio_V": 0.0, "delta_f_rel": np.inf}

    ratio = sigma_elec / sigma_0
    return {
        "ratio_R": float(np.sqrt(ratio)),
        "ratio_V": float(ratio ** 1.5),
        "delta_f_rel": float((1.0 / ratio) ** 1.5 - 1.0),
    }
