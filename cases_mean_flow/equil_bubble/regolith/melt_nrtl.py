"""
melt_nrtl.py
============

Multicomponent symmetric NRTL activity model for molten oxide (silicate slag)
pseudo-components -- the Layer-A analogue for the Mars-regolith Stage-1 melt.

Design decisions (documented in MODEL_SELECTION.md / THERMODYNAMICS.md):
  * PLAIN symmetric NRTL, mole-fraction basis, NO Pitzer-Debye-Hueckel and NO
    Born term: a fused oxide has no molecular solvent / dielectric-continuum
    dilute reference, and (Kontogeorgis & Folas Ch. 15) essentially all
    adjustable content of e-NRTL sits in the short-range term anyway.
  * Multicomponent G_ex is built ONLY from binary tau_ij (book Table 5.3);
    alpha_ij = 0.3 fixed (tune 0.2-0.47 only for miscibility-gap binaries).
  * NO published NRTL parameterisation of molten silicates exists (verified
    negative, THERMODYNAMICS.md sec. 4) -- the taus shipped here are an
    INTERIM, anchor-calibrated set (see ``calibrate_interim``), to be replaced
    by a fit to MELTS/ThermoEngine or FactSage pseudo-data (open pipeline,
    THERMODYNAMICS.md sec. 5) or digitised Slag Atlas tables.

Interim calibration anchors (all sourced, see docs):
    gamma_FeO   = 1.70 +- 0.22 (basaltic melts, pure-liquid-FeO reference;
                  Holzheid, Palme & Chakraborty 1997; composition envelope
                  1.2-2.1 per Sossi & Fegley 2018)             [VERIFIED]
    gamma_SiO2  ~ 0.3   (Rein & Chipman 1965 CMAS a_SiO2 maps, basaltic window)
    gamma_CaO   ~ 3e-3  (strong negative deviation, CaO-SiO2 lit.)
    gamma_MgO   ~ 0.1   (moderate negative deviation)
    gamma_Al2O3 ~ 0.05  (negative deviation, aluminate formation)

Note the Butler sigma layer consumes activity RATIOS a_i(surf)/a_i(bulk), so
absolute gamma levels matter less than their composition derivatives; the
gamma_FeO envelope [1.2, 2.1] is the dominant propagated uncertainty.

The fitting scaffolding (pack/unpack/residuals) mirrors ../enrtl_fit/enrtl_binary.py
so the future pseudo-data fit reuses the same scipy.least_squares driver.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares

# Canonical component ordering for the Stage-1 five-oxide melt.
OXIDES = ("SiO2", "FeO", "Al2O3", "MgO", "CaO")
N_OX = len(OXIDES)
IDX = {ox: i for i, ox in enumerate(OXIDES)}

# Molar masses [kg/mol]
M_OXIDE = {"SiO2": 60.08e-3, "FeO": 71.85e-3, "Al2O3": 101.96e-3,
           "MgO": 40.30e-3, "CaO": 56.08e-3}

# Baseline Stage-1 melt (devolatilised Rocknest, 5-component normalisation,
# mol fractions -- data/compositions.csv, recomputation-verified).
X_BASELINE = np.array([0.504, 0.188, 0.065, 0.152, 0.092])
# End-of-stage residual (FeO -> 0), mol fractions on the same 5 slots.
X_POST_FE = np.array([0.620, 0.0, 0.080, 0.187, 0.113])


def stage1_composition(xi: float) -> np.ndarray:
    """
    Melt composition along the Stage-1 trajectory.

    xi = FeO conversion (0 = baseline feed melt, 1 = all FeO removed).
    Removes FeO from the baseline melt and renormalises -- the other oxides
    keep their mutual proportions (verified against data/compositions.csv
    endpoint: xi=1 reproduces melt_post_fe to rounding).
    """
    xi = float(np.clip(xi, 0.0, 1.0))
    x = X_BASELINE.copy()
    x[IDX["FeO"]] *= (1.0 - xi)
    return x / x.sum()


@dataclass(frozen=True)
class MeltNRTL:
    """
    Symmetric multicomponent NRTL on oxide mole fractions.

    tau : (N, N) interaction matrix, tau[i][j] = tau_ij (i != j), zeros on the
          diagonal.  The INTERIM set is symmetric (tau_ij = tau_ji), one
          effective parameter per binary -- 5 calibrated pairs, others zero.
    alpha : non-randomness factor (scalar, applied to every pair).
    provenance : free-text flag carried into results/figures.
    """
    tau: tuple = ((0.0,) * N_OX,) * N_OX
    alpha: float = 0.3
    provenance: str = "ideal"

    # ---- core -------------------------------------------------------------
    def _tau_G(self):
        tau = np.asarray(self.tau, dtype=float)
        G = np.exp(-self.alpha * tau)
        return tau, G

    def ln_gamma(self, x: np.ndarray) -> np.ndarray:
        """Standard multicomponent NRTL ln(gamma_i) at mole fractions x."""
        x = np.asarray(x, dtype=float)
        tau, G = self._tau_G()
        # S_j = sum_k x_k G_kj ;  C_j = sum_k x_k tau_kj G_kj
        S = x @ G                       # S[j]
        C = x @ (tau * G)               # C[j]
        term1 = C / S                   # evaluated at j=i below
        ln_g = np.empty(N_OX)
        for i in range(N_OX):
            s = 0.0
            for j in range(N_OX):
                s += x[j] * G[i, j] / S[j] * (tau[i, j] - C[j] / S[j])
            ln_g[i] = term1[i] + s
        return ln_g

    def gamma(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.ln_gamma(x))

    def activity(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x * self.gamma(x)

    def gamma_of(self, oxide: str, x: np.ndarray) -> float:
        return float(self.gamma(x)[IDX[oxide]])

    # ---- fitting glue (mirrors enrtl_fit/enrtl_binary.py) -----------------
    def with_pairs(self, pairs: dict) -> "MeltNRTL":
        """Return a copy with symmetric tau set from {('A','B'): tau_b}."""
        tau = np.asarray(self.tau, dtype=float).copy()
        for (a, b), t in pairs.items():
            tau[IDX[a], IDX[b]] = t
            tau[IDX[b], IDX[a]] = t
        return replace(self, tau=tuple(map(tuple, tau)))


# ---------------------------------------------------------------------------
# INTERIM calibration: 5 symmetric binary taus <- 5 gamma anchors at baseline
# ---------------------------------------------------------------------------
CAL_PAIRS = (("FeO", "SiO2"), ("CaO", "SiO2"), ("MgO", "SiO2"),
             ("Al2O3", "SiO2"), ("CaO", "Al2O3"))

GAMMA_ANCHORS = {          # at X_BASELINE, ~1850 K; sources in module docstring
    "FeO": 1.70,           # Holzheid 1997 (VERIFIED); envelope 1.2-2.1
    "SiO2": 0.30,          # Rein & Chipman 1965 (basaltic window, approx)
    "CaO": 3e-3,           # strong negative deviation (approx)
    "MgO": 0.10,           # moderate negative deviation (approx)
    "Al2O3": 0.05,         # negative deviation (approx)
}


def calibrate_interim(gamma_feo: float = 1.70,
                      anchors: dict | None = None,
                      alpha: float = 0.3) -> MeltNRTL:
    """
    Least-squares calibrate the 5 symmetric binary taus so the model
    reproduces the gamma anchors at the baseline composition.

    ``gamma_feo`` overrides the FeO anchor so the verified envelope
    [1.2, 2.1] can be propagated as thermodynamic-model uncertainty.
    """
    anch = dict(GAMMA_ANCHORS if anchors is None else anchors)
    anch["FeO"] = gamma_feo
    target = np.log([anch[ox] for ox in OXIDES])

    def resid(theta):
        m = MeltNRTL(alpha=alpha).with_pairs(dict(zip(CAL_PAIRS, theta)))
        return m.ln_gamma(X_BASELINE) - target

    theta0 = np.array([0.5, -3.0, -1.5, -2.0, -1.0])
    sol = least_squares(resid, theta0, method="lm", xtol=1e-12)
    m = MeltNRTL(alpha=alpha).with_pairs(dict(zip(CAL_PAIRS, sol.x)))
    return replace(m, provenance=(
        f"interim-anchored(gamma_FeO={gamma_feo:.2f}, "
        f"rms={np.sqrt(np.mean(sol.fun**2)):.2e})"))


_CAL_CACHE: dict = {}


def interim_melt(gamma_feo: float = 1.70) -> MeltNRTL:
    """Cached interim-calibrated model (one per gamma_FeO anchor)."""
    key = round(float(gamma_feo), 4)
    if key not in _CAL_CACHE:
        _CAL_CACHE[key] = calibrate_interim(gamma_feo=key)
    return _CAL_CACHE[key]


if __name__ == "__main__":
    for gf in (1.2, 1.70, 2.1):
        m = interim_melt(gf)
        g = m.gamma(X_BASELINE)
        print(f"gamma_FeO anchor {gf}: {m.provenance}")
        for ox in OXIDES:
            print(f"   gamma_{ox:6s} = {g[IDX[ox]]:.4g} "
                  f"(target {GAMMA_ANCHORS[ox] if ox != 'FeO' else gf})")
        print(f"   gamma_FeO at xi=0.9: {m.gamma_of('FeO', stage1_composition(0.9)):.3f}")
