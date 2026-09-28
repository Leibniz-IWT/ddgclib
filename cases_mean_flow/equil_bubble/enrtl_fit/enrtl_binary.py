"""
enrtl_binary.py
==============

Clean, self-contained, FITTABLE symmetric-reference electrolyte-NRTL (e-NRTL)
model for a single strong electrolyte in water -- the forward model ("EOS")
your fitting library evaluates.

This is a standalone extraction of the lumped-salt e-NRTL used in
``bubble_enrtl_electrostatic_toy.py`` (Layer A), rewritten in the STANDARD
literature sign convention so it ingests Que2011 / Valverde2023 tau values
directly (no relabelling):

    tau_cw = tau_{salt -> water}   (NEGATIVE, ~ -4 .. -6)     [fit target]
    tau_wc = tau_{water -> salt}   (POSITIVE, ~ +8 .. +12)    [fit target]
    alpha  = 0.2 (non-randomness; usually fixed)
    A_phi  = 0.392 (Debye-Hueckel slope, water 298 K; usually fixed)
    rho    = 14.9 (PDH closest-approach; usually fixed)

Model = Pitzer-Debye-Hueckel long-range + short-range lumped-salt NRTL, on a
mole-fraction (symmetric) reference.  Outputs the observables you fit to:

    a_w(m)        water activity
    phi(m)        osmotic coefficient          phi = -ln(a_w)/(nu m Mw)
    gamma_pm(m)   mean ionic activity coeff    (Gibbs-Duhem from phi)

WARNING (convention): the *representative* values baked into the toy's Salt
objects (KOH tau_sw=+11.5, tau_ws=-4.5) are hand-tuned and in a SWAPPED
convention. Do NOT fit those. Fit ``tau_cw`` (negative) and ``tau_wc``
(positive) here. See lit/enrtl_parameters.py and manuscript/CORRECTIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid

M_W = 18.01528e-3            # kg/mol, water


@dataclass(frozen=True)
class ENRTLParams:
    """e-NRTL parameters + salt stoichiometry.  Fit targets: tau_cw, tau_wc."""
    # salt identity (fixed, not fitted)
    z_c: int = +1               # cation charge
    z_a: int = -1               # anion charge
    nu_c: int = 1               # cations per formula unit
    nu_a: int = 1               # anions per formula unit
    # short-range NRTL (FIT THESE), standard convention
    tau_cw: float = -4.0        # tau_{salt->water}   (negative)
    tau_wc: float = +8.0        # tau_{water->salt}   (positive)
    # usually-fixed model constants
    alpha: float = 0.2
    A_phi: float = 0.392
    rho: float = 14.9

    @property
    def nu(self) -> int:
        return self.nu_c + self.nu_a


# ---------------------------------------------------------------------------
# Core: extensive excess Gibbs energy G^ex / RT  (dimensionless)
# ---------------------------------------------------------------------------
def _mole_fractions(n_w: float, n_salt: float, p: ENRTLParams):
    n_c = p.nu_c * n_salt
    n_a = p.nu_a * n_salt
    n_t = n_w + n_c + n_a
    return n_w / n_t, n_c / n_t, n_a / n_t, n_t


def g_ex_over_RT(n_w: float, n_salt: float, p: ENRTLParams) -> float:
    """Extensive G^ex/RT = PDH (long-range) + lumped-salt NRTL (short-range)."""
    x_w, x_c, x_a, n_t = _mole_fractions(n_w, n_salt, p)

    # Pitzer-Debye-Hueckel (Chen-Song symmetric-reference form)
    I_x = 0.5 * (p.z_c ** 2 * x_c + p.z_a ** 2 * x_a)
    arg = 1.0 + p.rho * np.sqrt(I_x) if I_x > 0 else 1.0
    if I_x <= 0 or arg <= 0:
        # arg<=0 only for non-physical rho excursions during a free PDH fit
        g_pdh = 0.0
    else:
        g_pdh_molar = -(1000.0 / 18.01528) ** 0.5 \
            * (4.0 * p.A_phi * I_x / p.rho) \
            * np.log(arg)
        g_pdh = n_t * g_pdh_molar

    # short-range NRTL, lumped salt.  Standard convention:
    #   water-central term  -> coefficient tau_cw (salt->water)
    #   salt-central  term  -> coefficient tau_wc (water->salt)
    X_s = x_c + x_a
    G_cw = np.exp(-p.alpha * p.tau_cw)
    G_wc = np.exp(-p.alpha * p.tau_wc)
    if X_s > 0 and x_w > 0:
        term_w = (p.tau_cw * G_cw) / (x_w + X_s * G_cw)   # water central
        term_s = (p.tau_wc * G_wc) / (X_s + x_w * G_wc)   # salt central
        g_nrtl = n_t * x_w * X_s * (term_w + term_s)
    else:
        g_nrtl = 0.0

    return g_pdh + g_nrtl


def _chemical_potentials(n_w: float, n_salt: float, p: ENRTLParams):
    """(mu_w^ex/RT, mu_salt^ex/RT) by central finite difference on G^ex."""
    h = max(1e-7, 1e-5 * (n_w + n_salt))
    mu_w = (g_ex_over_RT(n_w + h, n_salt, p)
            - g_ex_over_RT(n_w - h, n_salt, p)) / (2 * h)
    if n_salt - h < 0:
        mu_s = (g_ex_over_RT(n_w, n_salt + h, p)
                - g_ex_over_RT(n_w, n_salt, p)) / h
    else:
        mu_s = (g_ex_over_RT(n_w, n_salt + h, p)
                - g_ex_over_RT(n_w, n_salt - h, p)) / (2 * h)
    return mu_w, mu_s


# ---------------------------------------------------------------------------
# Observables you fit to
# ---------------------------------------------------------------------------
def water_activity(m: float, p: ENRTLParams) -> float:
    """a_w(m) for molality m (mol salt / kg water)."""
    n_w = 1.0 / M_W
    x_w, _, _, _ = _mole_fractions(n_w, m, p)
    mu_w_ex, _ = _chemical_potentials(n_w, m, p)
    return x_w * np.exp(mu_w_ex)


def osmotic_coefficient(m: float, p: ENRTLParams) -> float:
    """phi(m) = -ln(a_w) / (nu m Mw);  phi -> 1 as m -> 0."""
    if m <= 0:
        return 1.0
    return -np.log(water_activity(m, p)) / (M_W * p.nu * m)


def mean_ionic_activity_coeff(m_grid: Sequence[float], p: ENRTLParams) -> np.ndarray:
    """
    gamma_pm(m) on the molality basis, by Gibbs-Duhem integration of phi:

        ln gamma_pm(m) = (phi(m) - 1) + integral_0^m (phi(m')-1)/m' dm'

    Returns an array aligned with ``m_grid`` (which must be ascending, >0).
    """
    m_grid = np.asarray(m_grid, dtype=float)
    pad = m_grid[0] > 0
    m_fine = np.concatenate([[0.0], m_grid]) if pad else m_grid
    phi = np.array([osmotic_coefficient(m, p) for m in m_fine])
    integrand = np.where(m_fine <= 1e-12, 0.0, (phi - 1.0) / np.where(m_fine <= 1e-12, 1.0, m_fine))
    integral = cumulative_trapezoid(integrand, m_fine, initial=0.0)
    ln_gamma = (phi - 1.0) + integral
    return np.exp(ln_gamma[1:] if pad else ln_gamma)


def predict(m_grid: Sequence[float], p: ENRTLParams) -> dict:
    """Return {'m','a_w','phi','gamma_pm'} arrays for a molality grid."""
    m_grid = np.asarray(m_grid, dtype=float)
    return dict(
        m=m_grid,
        a_w=np.array([water_activity(m, p) for m in m_grid]),
        phi=np.array([osmotic_coefficient(m, p) for m in m_grid]),
        gamma_pm=mean_ionic_activity_coeff(m_grid, p),
    )


# ---------------------------------------------------------------------------
# Fitting glue: pack/unpack the fit vector and a weighted residual
# ---------------------------------------------------------------------------
def pack(p: ENRTLParams, fit=("tau_cw", "tau_wc")) -> np.ndarray:
    """Vector of the parameters being fitted (default tau_cw, tau_wc)."""
    return np.array([getattr(p, k) for k in fit], dtype=float)


def unpack(theta: np.ndarray, p0: ENRTLParams, fit=("tau_cw", "tau_wc")) -> ENRTLParams:
    """Rebuild an ENRTLParams from a fit vector, holding non-fit params at p0."""
    return replace(p0, **{k: float(v) for k, v in zip(fit, theta)})


def residuals(theta: np.ndarray, data: dict, p0: ENRTLParams,
              fit=("tau_cw", "tau_wc"),
              use=("gamma_pm", "phi"), weights=None) -> np.ndarray:
    """
    Weighted residual vector for a least-squares fitter (scipy.optimize.
    least_squares, lmfit, etc.).

    Parameters
    ----------
    theta   : current fit-vector (see ``fit``).
    data    : dict with 'm' and any of 'gamma_pm','phi','a_w' (experimental).
              Missing observables are skipped; NaNs are dropped.
    p0      : template params (salt stoichiometry + fixed constants).
    use     : which observables to fit against (must exist in ``data``).
    weights : optional dict {observable: array} of 1/sigma weights.

    Returns a 1-D residual array (model - data) for scipy.least_squares.
    """
    p = unpack(theta, p0, fit)
    m = np.asarray(data["m"], float)
    pred = predict(m, p)
    res = []
    for obs in use:
        if obs not in data:
            continue
        d = np.asarray(data[obs], float)
        mask = np.isfinite(d)
        r = pred[obs][mask] - d[mask]
        if weights and obs in weights:
            r = r * np.asarray(weights[obs], float)[mask]
        # relative residual keeps gamma_pm (~O(1)) and phi (~O(1)) comparable
        res.append(r / np.maximum(np.abs(d[mask]), 1e-6))
    return np.concatenate(res) if res else np.array([0.0])


# ---------------------------------------------------------------------------
# Ready-made salt templates (stoichiometry only; tau are the fit targets)
# ---------------------------------------------------------------------------
KOH_TEMPLATE = ENRTLParams(z_c=+1, z_a=-1, nu_c=1, nu_a=1)          # 1:1
H2SO4_TEMPLATE = ENRTLParams(z_c=+1, z_a=-2, nu_c=2, nu_a=1)        # 2:1 (stoich)
KCL_TEMPLATE = ENRTLParams(z_c=+1, z_a=-1, nu_c=1, nu_a=1)          # 1:1


def to_toy_salt(p: ENRTLParams, name: str, sigma0: float, A_s: float,
                V_app: float, k_sech: float):
    """
    Build a ``bubble_enrtl_electrostatic_toy.Salt`` from FITTED params so the
    detachment pipeline / systems.py use the fitted thermodynamics.

    The toy's ``g_ex_over_RT`` uses the SAME standard convention as this module
    (its ``tau_sw`` slot is the water-central coefficient = tau_cw here, and
    ``tau_ws`` = tau_wc), so the mapping is direct -- NOT the swapped
    representative convention. (See manuscript/CORRECTIONS.md.)
    """
    from bubble_enrtl_electrostatic_toy import Salt
    return Salt(name=name, z_c=p.z_c, z_a=p.z_a, nu_c=p.nu_c, nu_a=p.nu_a,
                tau_sw=p.tau_cw, tau_ws=p.tau_wc,
                sigma0=sigma0, A_s=A_s, V_app=V_app, k_sech=k_sech)
