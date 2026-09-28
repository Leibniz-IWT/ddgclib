"""
melt_butler_sigma.py
====================

Layer-B analogue for the molten regolith: multicomponent Butler-equation
surface tension sigma(x, T) driven by the melt NRTL activities, with the
Nakamoto-Tanaka pure-oxide parameter set (model of record, avg err 4.2% vs
457 Slag Atlas points) and the Xin et al. 2020 statistical model as an
independent cross-check.  All parameters, brackets and caveats are documented
in SURFACE_PROPERTIES.md and data/butler_surface_tension.csv.

Butler (1932), per component i:
    sigma = sigma_i0(T) + (R*T / A_i) * ln( a_i^surf / a_i^bulk )

solved for the surface composition x^S (N-1 unknowns + closure) such that all
component equations give the same sigma.  Differences from the aqueous toy
(../bubble_enrtl_electrostatic_toy.butler_surface_tension):
  * the temperature argument is USED (R*T/A_i and sigma_i0(T));
  * A_i = N_A^(1/3) * V_i(T)^(2/3) with geometric factor f = 1 (Nakamoto
    convention; the aqueous toy uses 1.091);
  * fallback on non-convergence is the IDEAL (mole-fraction-weighted) sigma,
    never the aqueous 72 mN/m.

Simplification vs the full Nakamoto model (flagged): we use NRTL bulk/surface
activities directly instead of their ionic-radius pair-fraction machinery;
the published 4.2% accuracy therefore does not transfer automatically -- the
Nakamoto-vs-Xin spread is carried as the sigma model-form band (+-~30 mN/m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from melt_nrtl import OXIDES, N_OX, IDX, MeltNRTL, interim_melt

R_GAS = 8.314462618          # J/(mol K)
N_AVOGADRO = 6.02214076e23   # 1/mol

# Pure-oxide sigma_i0(T) [mN/m], T in K -- Nakamoto/NIST set (VERIFIED; FeO
# and MnO read visually from Nakamoto 2007 Table 4 during verification).
SIGMA0_NAKAMOTO = {
    "SiO2":  (243.2, +0.031),     # NIST; positive dT coefficient
    "CaO":   (791.0, -0.0935),
    "Al2O3": (1024.0, -0.177),
    "MgO":   (1770.0, -0.636),
    "FeO":   (504.0, +0.0984),    # 678 mN/m at 1773 K (HIGH end of bracket)
}
# Xin et al. 2020 statistical partials: sigma_i(T) = a + b*(T - 1773) [mN/m].
SIGMA0_XIN = {
    "SiO2":  (293.93, -0.00455),
    "Al2O3": (759.12, -0.27727),
    "CaO":   (672.09, -0.26493),
    "MgO":   (1177.56, -0.78932),
    "FeO":   (571.02, +0.13216),  # LOW end of the sigma_FeO0 bracket
}
# Molar volumes at 1773 K [cm^3/mol], x(1 + 1e-4 (T-1773)) (Nakamoto Table 2).
V_1773 = {"SiO2": 27.516, "Al2O3": 28.3, "CaO": 20.7, "FeO": 15.8, "MgO": 16.1}


def sigma0_nakamoto(oxide: str, T: float) -> float:
    a, b = SIGMA0_NAKAMOTO[oxide]
    return (a + b * T) * 1e-3            # N/m


def sigma0_xin(oxide: str, T: float) -> float:
    a, b = SIGMA0_XIN[oxide]
    return (a + b * (T - 1773.0)) * 1e-3  # N/m


def molar_volume(oxide: str, T: float) -> float:
    """V_i(T) in m^3/mol."""
    return V_1773[oxide] * 1e-6 * (1.0 + 1e-4 * (T - 1773.0))


def molar_surface_area(oxide: str, T: float, f: float = 1.0) -> float:
    """A_i = f * N_A^(1/3) * V_i^(2/3) in m^2/mol (Nakamoto f = 1)."""
    return f * N_AVOGADRO ** (1.0 / 3.0) * molar_volume(oxide, T) ** (2.0 / 3.0)


def butler_sigma(x_bulk: np.ndarray, T: float, melt: MeltNRTL,
                 sigma0_fn=sigma0_nakamoto,
                 sigma_feo0_override: float | None = None) -> tuple[float, np.ndarray]:
    """
    Solve the multicomponent Butler system for sigma(x_bulk, T) [N/m].

    sigma_feo0_override : optional sigma_FeO0 at THIS temperature [N/m] to
        propagate the 571-679 mN/m (1773 K) bracket as uncertainty.

    Returns (sigma [N/m], x_surface).  Falls back to the ideal mole-fraction-
    weighted sigma with x_surface = x_bulk when the root-find fails.
    """
    x_b = np.asarray(x_bulk, dtype=float)
    x_b = np.clip(x_b, 1e-12, None)
    x_b = x_b / x_b.sum()

    sig0 = np.array([sigma0_fn(ox, T) for ox in OXIDES])
    if sigma_feo0_override is not None:
        sig0[IDX["FeO"]] = sigma_feo0_override
    A = np.array([molar_surface_area(ox, T) for ox in OXIDES])
    a_bulk = melt.activity(x_b)

    def sigma_components(x_s):
        a_surf = melt.activity(x_s)
        return sig0 + (R_GAS * T / A) * np.log(a_surf / a_bulk)

    ideal = float(np.dot(x_b, sig0))

    # unknowns: z (N-1 log-ratio coordinates) -> x_s via softmax-like closure
    def unpack(z):
        w = np.concatenate([[0.0], z])
        e = np.exp(w - w.max())
        return e / e.sum()

    def resid(z):
        s = sigma_components(unpack(z))
        return s[1:] - s[0]

    z0 = np.log(x_b[1:] / x_b[0])
    try:
        sol = least_squares(resid, z0, method="lm", xtol=1e-14, ftol=1e-14)
        x_s = unpack(sol.x)
        s_all = sigma_components(x_s)
        spread = float(np.max(s_all) - np.min(s_all))
        sigma = float(np.mean(s_all))
        if not np.isfinite(sigma) or spread > 1e-4 or sigma <= 0:
            return ideal, x_b          # physically sensible fallback
        return sigma, x_s
    except Exception:
        return ideal, x_b


# ---------------------------------------------------------------------------
# SurfaceTensionModel adapters (pulloff_models interface)
# ---------------------------------------------------------------------------
@dataclass
class MeltHandle:
    """
    The 'salt' handle for the melt case: bundles the NRTL model and the
    composition trajectory so the pipeline's opaque scalar 'm' can be
    interpreted as the Stage-1 FeO conversion xi in [0, 1].
    """
    melt: MeltNRTL
    T: float = 1850.0

    def composition(self, xi: float) -> np.ndarray:
        from melt_nrtl import stage1_composition
        return stage1_composition(xi)


class MeltButlerSigma:
    """
    SurfaceTensionModel-compatible: sigma_lv(m, salt, T) where m = xi (FeO
    conversion along the Stage-1 trajectory) and salt is a MeltHandle.
    Duck-typed against pulloff_models.SurfaceTensionModel.
    """
    name = "melt_butler_nrtl"

    def __init__(self, sigma0_fn=sigma0_nakamoto,
                 sigma_feo0_1773: float | None = None):
        self.sigma0_fn = sigma0_fn
        # bracket handling: override specified at 1773 K, slope from Nakamoto
        self._feo_1773 = sigma_feo0_1773

    def _feo_override(self, T: float):
        if self._feo_1773 is None:
            return None
        slope = SIGMA0_NAKAMOTO["FeO"][1] * 1e-3     # N/m/K
        return self._feo_1773 + slope * (T - 1773.0)

    def sigma_lv(self, m: float, salt=None, T: float = 1850.0) -> float:
        handle: MeltHandle = salt
        x = handle.composition(m)
        sigma, _ = butler_sigma(x, T, handle.melt, sigma0_fn=self.sigma0_fn,
                                sigma_feo0_override=self._feo_override(T))
        return sigma


class XinLinearSigma:
    """
    Cross-check model: linear mixing of the Xin et al. 2020 fitted partial
    surface tensions (their binary-interaction excess terms are NOT included
    -- approximate; carries its own ~5%/30 mN/m stated accuracy).
    """
    name = "xin2020_linear"

    def sigma_lv(self, m: float, salt=None, T: float = 1850.0) -> float:
        handle: MeltHandle = salt
        x = handle.composition(m)
        return float(sum(x[IDX[ox]] * sigma0_xin(ox, T) for ox in OXIDES))


if __name__ == "__main__":
    from melt_nrtl import stage1_composition
    handle = MeltHandle(melt=interim_melt(1.70))
    T = 1850.0
    print(f"T = {T} K   (anchor band: 0.30-0.50 N/m; natural basalts 0.35-0.37)")
    for model in (MeltButlerSigma(), XinLinearSigma(),
                  MeltButlerSigma(sigma_feo0_1773=0.571)):
        vals = [model.sigma_lv(xi, handle, T) for xi in (0.0, 0.5, 1.0)]
        print(f"{model.name:24s} sigma(xi=0,0.5,1) = "
              + "  ".join(f"{v*1e3:6.1f}" for v in vals) + "  mN/m")
    # ideal-mixing reference
    x0 = stage1_composition(0.0)
    ideal = sum(x0[IDX[ox]] * sigma0_nakamoto(ox, T) for ox in OXIDES)
    print(f"{'ideal-mix nakamoto':24s} sigma(xi=0)       = {ideal*1e3:6.1f}  mN/m")
