"""
bubble_enrtl_toy.py
===================

Toy problem: how a proper thermodynamic free-energy model (symmetric-reference
e-NRTL) changes the predicted bubble shape, detachment volume, and VLE of H2
bubbles in water electrolysis, compared to the "ideal" assumption used in
Krug et al. 2022 (H2SO4) and Raman/Rivas et al. 2022 (NaHCO3).

Pipeline
    e-NRTL (G^ex)  ->  mu_i, a_w, gamma_pm
                   ->  Butler surface tension sigma(m)
                   ->  Young-Laplace axisymmetric bubble shape
                   ->  buoyancy/capillary detachment volume V_d(m)
                   ->  VLE inside bubble: y_H2, r_crit(m)

Two electrolytes for contrast (1:1 vs 1:2 -- very different ionic strength):
    KOH   (alkaline electrolysis -- headline industrial case)
    H2SO4 (matches Krug et al. operating electrolyte)

Run: python bubble_enrtl_toy.py
Output: toy_bubble/fig/*.png, toy_bubble/out/summary.csv

The numbers are physically sensible (DH limiting law recovered at m->0,
sigma matches pure water, Butler surface composition is enriched in water)
but parameters are deliberately "representative" rather than fitted to
4 sig figs -- the point is the TREND.
"""

from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants (SI unless noted)
# ---------------------------------------------------------------------------
R_GAS   = 8.314462618
N_A     = 6.02214076e23
T_REF   = 298.15
P_ATM   = 1.01325e5
G_GRAV  = 9.80665
M_W     = 18.01528e-3            # kg/mol
RHO_W   = 997.05                 # kg/m^3
RHO_H2  = 0.0899                 # kg/m^3 at 1 bar, 25 C
SIGMA_W = 0.07197                # N/m pure water, 25 C
P_SAT_W = 3169.0                 # Pa water vapour pressure, 25 C
A_PHI   = 0.392                  # Debye-Hueckel parameter, water 298 K
RHO_PDH = 14.9                   # closest-approach param in PDH
ALPHA   = 0.2                    # NRTL non-randomness

# Molar surface areas (m^2/mol) -- from f * N_A^(1/3) * V_i^(2/3), f = 1.091
A_W     = 6.30e4                 # water
A_KOH   = 8.40e4                 # approximate for "lumped" KOH salt
A_H2SO4 = 1.31e5                 # approximate for "lumped" H2SO4 salt

# Hypothetical pure-salt surface tensions (tune only to give right trend)
SIGMA0_KOH_PURE   = 0.170
SIGMA0_H2SO4_PURE = 0.155

# Henry's constant for H2 in pure water at 25 C, mol/(kg_water * Pa)
# c_aq = kH0 * p_H2. Published: ~7.8e-9 mol/(kg*Pa)
KH_H2_0 = 7.8e-9


# ---------------------------------------------------------------------------
# Salt specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Salt:
    name: str
    z_c: int        # cation charge (+1, +1, ...)
    z_a: int        # anion charge (-1, -2, ...)
    nu_c: int       # stoichiometry cation
    nu_a: int       # stoichiometry anion
    tau_sw: float   # e-NRTL param: salt-pair -> solvent
    tau_ws: float   # e-NRTL param: solvent -> salt-pair
    sigma0: float   # hypothetical pure-salt surface tension (N/m)
    A_s: float      # molar surface area (m^2/mol)
    V_app: float    # apparent molar volume at inf. dilution (m^3/mol)
    k_sech: float   # Sechenov coefficient for H2 salting-out (L/mol)

    @property
    def nu(self) -> int:
        return self.nu_c + self.nu_a


KOH   = Salt('KOH',   z_c=+1, z_a=-1, nu_c=1, nu_a=1,
             tau_sw=11.5, tau_ws=-4.5,
             sigma0=SIGMA0_KOH_PURE,   A_s=A_KOH,
             V_app=27e-6, k_sech=0.134)

H2SO4 = Salt('H2SO4', z_c=+1, z_a=-2, nu_c=2, nu_a=1,
             tau_sw=13.0, tau_ws=-5.2,
             sigma0=SIGMA0_H2SO4_PURE, A_s=A_H2SO4,
             V_app=53e-6, k_sech=0.099)


# ---------------------------------------------------------------------------
# LAYER A -- symmetric-reference e-NRTL for single strong electrolyte + water
# ---------------------------------------------------------------------------
def _mole_fractions(n_w: float, n_salt: float, salt: Salt):
    """Return (x_w, x_c, x_a, n_t) for n_salt formula-units of salt."""
    n_c = salt.nu_c * n_salt
    n_a = salt.nu_a * n_salt
    n_t = n_w + n_c + n_a
    return n_w / n_t, n_c / n_t, n_a / n_t, n_t


def g_ex_over_RT(n_w: float, n_salt: float, salt: Salt) -> float:
    """
    Extensive excess Gibbs energy G^ex / RT (dimensionless),
    symmetric-reference e-NRTL = PDH + short-range NRTL (lumped-salt).

    The PDH term carries the charge-asymmetry (z_c^2 x_c + z_a^2 x_a),
    which is why KOH (1:1) and H2SO4 (1:2) have very different curves
    even with similar tau parameters.
    """
    x_w, x_c, x_a, n_t = _mole_fractions(n_w, n_salt, salt)

    # ---- Pitzer-Debye-Hueckel (PDH), symmetric reference ----
    I_x = 0.5 * (salt.z_c**2 * x_c + salt.z_a**2 * x_a)
    if I_x <= 0:
        g_pdh = 0.0
    else:
        sqrt_Ix = np.sqrt(I_x)
        # Chen-Song form: G^ex_PDH / (n_t R T)
        #   = - (1000 / M_w_g)^0.5 * (4 A_phi I_x / rho) * ln(1 + rho sqrt(I_x))
        g_pdh_molar = -(1000.0 / 18.01528) ** 0.5 \
                      * (4.0 * A_PHI * I_x / RHO_PDH) \
                      * np.log(1.0 + RHO_PDH * sqrt_Ix)
        g_pdh = n_t * g_pdh_molar

    # ---- short-range NRTL (lumped-salt, 2 params) ----
    X_s = x_c + x_a                          # total ionic mole fraction
    G_sw = np.exp(-ALPHA * salt.tau_sw)
    G_ws = np.exp(-ALPHA * salt.tau_ws)
    if X_s > 0 and x_w > 0:
        term1 = (salt.tau_sw * G_sw) / (x_w + X_s * G_sw)
        term2 = (salt.tau_ws * G_ws) / (X_s + x_w * G_ws)
        g_nrtl_molar = x_w * X_s * (term1 + term2)
    else:
        g_nrtl_molar = 0.0
    g_nrtl = n_t * g_nrtl_molar

    return g_pdh + g_nrtl


def chemical_potentials(n_w: float, n_salt: float, salt: Salt):
    """
    Excess chemical potentials mu_i^ex / RT by central finite difference on
    the extensive G^ex.  Returns (mu_w_ex_over_RT, mu_salt_ex_over_RT).
    """
    h = max(1e-7, 1e-5 * (n_w + n_salt))
    # water derivative
    gp = g_ex_over_RT(n_w + h, n_salt, salt)
    gm = g_ex_over_RT(n_w - h, n_salt, salt)
    mu_w = (gp - gm) / (2 * h)
    # salt derivative
    gp = g_ex_over_RT(n_w, n_salt + h, salt)
    gm = g_ex_over_RT(n_w, max(n_salt - h, 0.0), salt)
    # forward difference if we clamped
    if n_salt - h < 0:
        gp = g_ex_over_RT(n_w, n_salt + h, salt)
        g0 = g_ex_over_RT(n_w, n_salt, salt)
        mu_s = (gp - g0) / h
    else:
        mu_s = (gp - gm) / (2 * h)
    return mu_w, mu_s


def water_activity(m: float, salt: Salt) -> float:
    """a_w(m) for molality m (mol salt / kg water)."""
    n_w = 1.0 / M_W                          # moles water in 1 kg
    n_s = m
    x_w, _, _, _ = _mole_fractions(n_w, n_s, salt)
    mu_w_ex, _ = chemical_potentials(n_w, n_s, salt)
    return x_w * np.exp(mu_w_ex)


def osmotic_coefficient(m: float, salt: Salt) -> float:
    """
    Osmotic coefficient phi(m).  Definition:
        phi = -ln(a_w) / (M_w_kg * nu * m)
    For m -> 0, phi -> 1.
    """
    if m <= 0:
        return 1.0
    a_w = water_activity(m, salt)
    return -np.log(a_w) / (M_W * salt.nu * m)


def mean_ionic_activity_coeff(m_grid: np.ndarray, salt: Salt) -> np.ndarray:
    """
    Mean ionic activity coefficient gamma_pm(m) on molality basis,
    from Gibbs-Duhem integration of the osmotic coefficient:
        ln gamma_pm(m) = (phi(m) - 1) + integral_0^m (phi(m')-1)/m' dm'
    """
    m_grid = np.asarray(m_grid, dtype=float)
    # evaluate phi on a fine grid starting at 0
    m_fine = np.concatenate([[0.0], m_grid]) if m_grid[0] > 0 else m_grid
    phi = np.array([osmotic_coefficient(m, salt) for m in m_fine])
    # integrand (phi - 1) / m, with L'Hopital-style limit at m=0 (phi -> 1 so (phi-1)/m -> finite)
    integrand = np.zeros_like(m_fine)
    for i, m in enumerate(m_fine):
        if m <= 1e-12:
            integrand[i] = 0.0   # (phi-1)/m tends to a finite constant -- we approximate as 0 for toy
        else:
            integrand[i] = (phi[i] - 1.0) / m
    integral = cumulative_trapezoid(integrand, m_fine, initial=0.0)
    ln_gamma_pm = (phi - 1.0) + integral
    # strip padded zero if we added it
    if m_grid[0] > 0:
        return np.exp(ln_gamma_pm[1:])
    return np.exp(ln_gamma_pm)


# ---------------------------------------------------------------------------
# LAYER B -- Butler surface tension
# ---------------------------------------------------------------------------
def butler_surface_tension(m: float, salt: Salt) -> tuple[float, float]:
    """
    Solve the Butler equation (binary form, bulk-eNRTL for surface activities).
    Returns (sigma, x_s_surface).

        sigma = sigma_w0 + (RT/A_w) * ln(a_w^S / a_w^B)
        sigma = sigma_s0 + (RT/A_s) * ln(a_s^S / a_s^B)

    We solve for the surface salt mole fraction x_s^S by matching the two.
    Activity coefficients evaluated with the same e-NRTL model at both compositions.
    """
    # --- bulk composition and activities ---
    n_w = 1.0 / M_W
    n_s = m
    x_w_B, x_c_B, x_a_B, n_t_B = _mole_fractions(n_w, n_s, salt)
    x_s_B = x_c_B + x_a_B     # lumped salt mole fraction (ions)
    mu_w_ex_B, mu_s_ex_B = chemical_potentials(n_w, n_s, salt)
    a_w_B = x_w_B * np.exp(mu_w_ex_B)
    # activity of lumped salt:  a_s = X_s * gamma_s where X_s = x_c + x_a
    # Use excess mu_salt as approximation -- acceptable for a toy Butler.
    a_s_B = max(x_s_B, 1e-15) * np.exp(mu_s_ex_B)

    # --- residual: given candidate x_s^S, build surface composition,
    #     evaluate a_w^S and a_s^S, and compare the two sigma values.
    # Convert surface mole fraction back to (n_w^S, n_s^S) of same total (say 1 mol):
    def residual(x_s_S: float) -> float:
        # We need to pick n_c^S, n_a^S summing to x_s_S * n_total_S.
        # Keep stoichiometry nu_c:nu_a; lumped x_s maps to:
        #   x_c = x_s * nu_c/nu, x_a = x_s * nu_a/nu
        x_w_S = 1.0 - x_s_S
        # Equivalent n_w, n_s so that the *lumped* x_s matches:
        # We have x_s = (nu_c + nu_a)*n_s / (n_w + nu*n_s)
        # Solve: fix n_total_S = 1 mol (arbitrary scale).
        # n_s_S * (nu_c + nu_a) = x_s_S * (n_w_S + nu * n_s_S)
        # with n_w_S + nu * n_s_S = 1  =>  n_w_S = 1 - nu * n_s_S
        # and  nu * n_s_S = x_s_S       =>  n_s_S = x_s_S / nu
        n_s_S = x_s_S / salt.nu
        n_w_S = 1.0 - salt.nu * n_s_S
        if n_w_S <= 0:
            return 1.0  # invalid
        x_w_S_check, _, _, _ = _mole_fractions(n_w_S, n_s_S, salt)
        # should match x_w_S within round-off
        mu_w_ex_S, mu_s_ex_S = chemical_potentials(n_w_S, n_s_S, salt)
        a_w_S = x_w_S_check * np.exp(mu_w_ex_S)
        a_s_S = max(x_s_S, 1e-15) * np.exp(mu_s_ex_S)
        sigma_from_w = SIGMA_W + (R_GAS * T_REF / A_W) * np.log(a_w_S / a_w_B)
        sigma_from_s = salt.sigma0 + (R_GAS * T_REF / salt.A_s) * np.log(a_s_S / a_s_B)
        return sigma_from_w - sigma_from_s

    # Bracket: surface is enriched in water so x_s^S < x_s^B, and always > 0
    lo = 1e-10
    hi = max(x_s_B, 1e-8)
    # Expand bracket if the root isn't between lo and hi (edge cases)
    f_lo = residual(lo)
    f_hi = residual(hi)
    if f_lo * f_hi > 0:
        # fall back: scan to bracket
        xs = np.logspace(-10, np.log10(max(x_s_B * 0.999, 1e-9)), 50)
        fs = np.array([residual(x) for x in xs])
        sign_change = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)[0]
        if len(sign_change) == 0:
            # Butler cannot find a root -> fall back to sigma = sigma_w (pure water)
            return SIGMA_W, 0.0
        lo, hi = xs[sign_change[0]], xs[sign_change[0] + 1]
    x_s_S = brentq(residual, lo, hi, rtol=1e-6)

    # compute sigma at the solution
    n_s_S = x_s_S / salt.nu
    n_w_S = 1.0 - salt.nu * n_s_S
    mu_w_ex_S, _ = chemical_potentials(n_w_S, n_s_S, salt)
    x_w_S_check, _, _, _ = _mole_fractions(n_w_S, n_s_S, salt)
    a_w_S = x_w_S_check * np.exp(mu_w_ex_S)
    sigma = SIGMA_W + (R_GAS * T_REF / A_W) * np.log(a_w_S / a_w_B)
    return sigma, x_s_S


# ---------------------------------------------------------------------------
# LAYER C -- axisymmetric Young-Laplace bubble shape
# ---------------------------------------------------------------------------
def young_laplace_shape(sigma: float, b: float, theta_gas: float,
                        rho_l: float = RHO_W, s_max_factor: float = 8.0):
    """
    Integrate the axisymmetric Young-Laplace equation for a sessile bubble
    (gas, light) sitting on a horizontal solid at the bottom.

    Parameters
    ----------
    sigma     : surface tension (N/m)
    b         : apex radius of curvature (m)
    theta_gas : contact angle measured through the gas (rad)
    rho_l     : liquid density
    s_max_factor : multiple of b at which to cap integration (safety)

    Returns
    -------
    dict with arrays r, z (absolute, apex at z=H>0, contact at z=0),
    scalars r_cl, H, V (volume), and converged=True/False.
    """
    beta = (rho_l - RHO_H2) * G_GRAV / sigma   # 1/m^2

    # ODE:  d r/ds  = cos phi
    #       d z_dep/ds = sin phi              (z_dep = depth below apex)
    #       d phi/ds  = 2/b - beta * z_dep - sin(phi)/r
    def rhs(s, y):
        r, z_dep, phi = y
        if r < 1e-12:
            # near-apex expansion: r ~ s, sin(phi)/r -> 1/b
            return [np.cos(phi), np.sin(phi), 2.0 / b - beta * z_dep - 1.0 / b]
        return [np.cos(phi), np.sin(phi), 2.0 / b - beta * z_dep - np.sin(phi) / r]

    # Start a tiny step from apex using spherical series (r=b sin(s/b), etc.)
    s0 = 1e-4 * b
    y0 = [b * np.sin(s0 / b), b * (1.0 - np.cos(s0 / b)), s0 / b]

    # target contact line: phi = pi - theta_gas  (measured through gas)
    phi_target = np.pi - theta_gas

    def event_phi(s, y):
        return y[2] - phi_target
    event_phi.terminal = True
    event_phi.direction = 1

    # event for r -> 0 again (pinch-off)
    def event_pinch(s, y):
        return y[0] - 1e-4 * b
    event_pinch.terminal = True
    event_pinch.direction = -1

    s_max = s_max_factor * b
    sol = solve_ivp(rhs, (s0, s_max), y0, method='RK45',
                    events=[event_phi, event_pinch],
                    rtol=1e-6, atol=1e-9, max_step=b / 20)

    if not sol.success:
        return dict(converged=False, reason='solver_failure')

    # contact event fired?
    if len(sol.t_events[0]) == 0:
        return dict(converged=False,
                    reason='pinched' if len(sol.t_events[1]) else 'no_contact')

    # extract solution up to contact event
    t_contact = sol.t_events[0][0]
    y_contact = sol.y_events[0][0]
    mask = sol.t <= t_contact
    t = np.concatenate([sol.t[mask], [t_contact]])
    r = np.concatenate([sol.y[0, mask], [y_contact[0]]])
    z_dep = np.concatenate([sol.y[1, mask], [y_contact[1]]])
    phi = np.concatenate([sol.y[2, mask], [y_contact[2]]])

    H = float(z_dep[-1])
    r_cl = float(r[-1])
    # Build absolute z with apex at top (z=H), contact at z=0
    z_abs = H - z_dep

    # Volume by disk integration (dV = pi r^2 dz = pi r^2 sin(phi) ds)
    dV_ds = np.pi * r ** 2 * np.sin(phi)
    V = float(np.trapezoid(dV_ds, t))

    return dict(converged=True, r=r, z=z_abs, phi=phi, s=t,
                r_cl=r_cl, H=H, V=V, b=b)


# ---------------------------------------------------------------------------
# LAYER C2 -- detachment volume from Fritz-type force balance
# ---------------------------------------------------------------------------
def detachment_volume(sigma: float, theta_gas: float,
                      rho_l: float = RHO_W) -> dict:
    """
    Find apex curvature b* such that buoyancy = vertical capillary pinning
    at the contact line.  Returns V_d, D_d, r_cl at detachment, and shape.

        F_buoy = (rho_l - rho_g) g V
        F_pin  = 2 pi r_cl sigma sin(theta_gas)
    """
    def residual(b):
        shape = young_laplace_shape(sigma, b, theta_gas, rho_l=rho_l)
        if not shape['converged']:
            # return a large residual so the root-finder moves away
            return 1e6
        F_buoy = (rho_l - RHO_H2) * G_GRAV * shape['V']
        F_pin  = 2.0 * np.pi * shape['r_cl'] * sigma * np.sin(theta_gas)
        return F_buoy - F_pin

    # Scan over b to bracket the root
    capillary_length = np.sqrt(sigma / ((rho_l - RHO_H2) * G_GRAV))
    bs = np.linspace(0.1 * capillary_length, 2.5 * capillary_length, 18)
    residuals = []
    for b in bs:
        try:
            residuals.append(residual(b))
        except Exception:
            residuals.append(np.nan)
    residuals = np.array(residuals)
    # find first sign change (buoyancy crosses pinning from below)
    valid = np.isfinite(residuals)
    idx = np.where(valid[:-1] & valid[1:] & (residuals[:-1] * residuals[1:] < 0))[0]
    if len(idx) == 0:
        return dict(converged=False)
    i = idx[0]
    b_star = brentq(residual, bs[i], bs[i + 1], rtol=1e-5)
    shape = young_laplace_shape(sigma, b_star, theta_gas, rho_l=rho_l)
    V_d = shape['V']
    D_d = (6.0 * V_d / np.pi) ** (1.0 / 3.0)
    return dict(converged=True, b=b_star, V=V_d, D=D_d, r_cl=shape['r_cl'],
                H=shape['H'], shape=shape)


# ---------------------------------------------------------------------------
# LAYER D -- VLE in the bubble and nucleation proxy
# ---------------------------------------------------------------------------
def henry_effective(m: float, salt: Salt) -> float:
    """
    Effective Henry's constant for H2 in electrolyte, mol/(kg_water * Pa).
    Use Sechenov salting-out: k_H,eff = k_H,0 / 10^(k_s * c_salt).
    c_salt here is approximated by molarity ~ m * rho / (1 + 1e-3 m * M_salt) ~ m.
    For the TOY we use c_salt_M ~ m.
    """
    c_salt_M = m           # mol/L approx at moderate m
    return KH_H2_0 * 10 ** (-salt.k_sech * c_salt_M)


def vle_bubble(m: float, salt: Salt, R_bub: float,
               S0: float = 2.0) -> dict:
    """
    VLE inside an H2 bubble of radius R_bub in electrolyte of molality m.

    Bubble contains H2 + water vapour.
      p_H2O = a_w * p_sat              (Raoult with activity from e-NRTL)
      P_bub = P_atm + 2 sigma / R_bub   (Laplace)
      y_H2  = (P_bub - p_H2O) / P_bub

    For nucleation we fix the *dissolved-H2 concentration* at a reference
    supersaturation S0 w.r.t. pure water (S0 = c_H2 / (k_H^0 P_atm)).
    In salty electrolyte, the effective supersaturation is
        S_eff(m) = c_H2 / (k_H,eff(m) P_atm)  = S0 * k_H^0 / k_H,eff(m)
    so salting-out *boosts* the driving force.  Critical radius:
        r* = 2 sigma(m) / ((S_eff - 1) P_atm)
    Both sigma(m) and S_eff(m) are non-linear in m, giving compound behaviour.
    """
    sigma, _ = butler_surface_tension(m, salt)
    a_w = water_activity(m, salt)
    p_H2O = a_w * P_SAT_W
    P_bub = P_ATM + 2.0 * sigma / R_bub
    p_H2 = P_bub - p_H2O
    y_H2 = p_H2 / P_bub

    kH = henry_effective(m, salt)
    S_eff = S0 * KH_H2_0 / kH
    if S_eff > 1.0:
        r_crit = 2.0 * sigma / ((S_eff - 1.0) * P_ATM)
    else:
        r_crit = np.inf

    return dict(sigma=sigma, a_w=a_w, p_H2O=p_H2O, P_bub=P_bub,
                y_H2=y_H2, S_eff=S_eff, r_crit=r_crit)


# ---------------------------------------------------------------------------
# DEMO / plotting
# ---------------------------------------------------------------------------
def _ensure_dirs(fig_name: str = 'fig', out_name: str = 'out'):
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, fig_name)
    out_dir = os.path.join(here, out_name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return fig_dir, out_dir


def _style():
    plt.rcParams.update({
        'figure.figsize': (6.2, 4.2),
        'figure.dpi': 110,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
    })


def _save_fig(fig, fig_dir: str, filename: str, citation_banner: str = '',
              existing_suptitle: str = ''):
    """Attach citation banner (if any), tidy layout, save, close."""
    if citation_banner:
        full = (existing_suptitle + ' -- ' + citation_banner
                if existing_suptitle else citation_banner)
        fig.suptitle(full, fontsize=9, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.955])
    elif existing_suptitle:
        fig.suptitle(existing_suptitle, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.955])
    else:
        plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=140)
    plt.close(fig)


def demo(salts=None, m_grids=None, colors=None,
         fig_dir_name: str = 'fig', out_dir_name: str = 'out',
         citation_banner: str = ''):
    """
    Produce the 6 slide-ready PNGs and summary CSV.

    Parameters
    ----------
    salts           : list[Salt], default [KOH, H2SO4] (representative e-NRTL)
    m_grids         : dict[str, np.ndarray] keyed by salt.name
    colors          : dict[str, str] keyed by salt.name
    fig_dir_name    : folder under ``toy_bubble/`` for PNGs (default 'fig')
    out_dir_name    : folder under ``toy_bubble/`` for CSV  (default 'out')
    citation_banner : text shown as fig.suptitle above every figure (empty
                      = omit).  Use to flag "representative" vs. "literature
                      parameters" runs on the slide.
    """
    fig_dir, out_dir = _ensure_dirs(fig_dir_name, out_dir_name)
    _style()
    if salts is None:
        salts = [KOH, H2SO4]
    if colors is None:
        default_palette = ['C0', 'C3', 'C2', 'C4', 'C5']
        colors = {s.name: default_palette[i % len(default_palette)]
                  for i, s in enumerate(salts)}
    if m_grids is None:
        # Molality ranges (H2SO4 rarely run above 3 M industrially,
        # KOH goes up to 10-12 M in alkaline electrolysers)
        m_grids = {'KOH':   np.linspace(0.01, 10.0, 40),
                   'H2SO4': np.linspace(0.01, 6.0, 40)}

    # Store derived quantities for CSV
    results = {s.name: {} for s in salts}

    # -------- Figure 1: gamma_pm and a_w --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in salts:
        m = m_grids[s.name]
        gamma_pm = mean_ionic_activity_coeff(m, s)
        a_w = np.array([water_activity(mi, s) for mi in m])
        results[s.name]['m'] = m
        results[s.name]['gamma_pm'] = gamma_pm
        results[s.name]['a_w'] = a_w
        ax1.semilogy(m, gamma_pm, color=colors[s.name], label=s.name)
        ax2.plot(m, a_w, color=colors[s.name], label=s.name)
    ax1.axhline(1.0, color='k', linestyle=':', linewidth=1)
    ax1.set_xlabel('molality  m  (mol/kg water)')
    ax1.set_ylabel(r'mean ionic activity coeff  $\gamma_\pm$')
    ax1.set_title(r'$\gamma_\pm(m)$ from symmetric e-NRTL')
    ax1.legend()
    ax2.axhline(1.0, color='k', linestyle=':', linewidth=1)
    ax2.set_xlabel('molality  m  (mol/kg water)')
    ax2.set_ylabel(r'water activity  $a_w$')
    ax2.set_title(r'$a_w(m)$ -- departs from ideal above $\sim$1 mol/kg')
    ax2.legend()
    _save_fig(fig, fig_dir, 'fig1_activities.png', citation_banner)

    # -------- Figure 2: surface tension from Butler --------
    fig, ax = plt.subplots()
    for s in salts:
        m = m_grids[s.name]
        sigma = np.array([butler_surface_tension(mi, s)[0] for mi in m])
        results[s.name]['sigma'] = sigma
        ax.plot(m, 1e3 * sigma, color=colors[s.name], label=f'{s.name} (Butler-eNRTL)')
    ax.axhline(1e3 * SIGMA_W, color='k', linestyle=':',
               label=r'$\sigma = 72$ mN/m  (Sepahi et al. 2022; Raman et al. 2022)')
    ax.set_xlabel('molality  m  (mol/kg water)')
    ax.set_ylabel(r'$\sigma$  (mN/m)')
    ax.set_title('Surface tension from Butler eq. with e-NRTL activities')
    ax.legend()
    _save_fig(fig, fig_dir, 'fig2_surface_tension.png', citation_banner)

    # -------- Figure 3: effective Henry constant (salting-out) --------
    fig, ax = plt.subplots()
    for s in salts:
        m = m_grids[s.name]
        kH_eff = np.array([henry_effective(mi, s) for mi in m])
        results[s.name]['kH_ratio'] = kH_eff / KH_H2_0
        ax.plot(m, kH_eff / KH_H2_0, color=colors[s.name], label=s.name)
    ax.axhline(1.0, color='k', linestyle=':',
               label='constant $k_H$  (Sepahi et al. 2022, Electrochim. Acta)')
    ax.set_xlabel('molality  m  (mol/kg water)')
    ax.set_ylabel(r'$k_{H}^{\mathrm{eff}} / k_{H}^{0}$  (H$_2$ solubility ratio)')
    ax.set_title(r'Salting-out of H$_2$: Sechenov-type correction')
    ax.set_yscale('log')
    ax.legend()
    _save_fig(fig, fig_dir, 'fig3_salting_out.png', citation_banner)

    # -------- Figure 4: bubble shape and detachment volume --------
    theta_gas = np.deg2rad(30.0)   # hydrophobic cavity (Rivas-like): pinched bubble
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left panel: overlay of 3 bubble profiles per salt at selected m values
    for s in salts:
        m_show = [m_grids[s.name][0], m_grids[s.name][len(m_grids[s.name]) // 2],
                  m_grids[s.name][-1]]
        for j, m in enumerate(m_show):
            sigma, _ = butler_surface_tension(m, s)
            res = detachment_volume(sigma, theta_gas)
            if not res['converged']:
                continue
            shape = res['shape']
            ls = ['-', '--', ':'][j]
            axL.plot(1e3 * shape['r'], 1e3 * shape['z'],
                     color=colors[s.name], linestyle=ls, alpha=0.9,
                     label=f'{s.name}  m={m:.2f} mol/kg')
            axL.plot(-1e3 * shape['r'], 1e3 * shape['z'],
                     color=colors[s.name], linestyle=ls, alpha=0.9)
    axL.axhline(0.0, color='gray', linewidth=1)
    axL.set_aspect('equal')
    axL.set_xlabel('r  (mm)')
    axL.set_ylabel('z  (mm)')
    axL.set_title(r'Detachment shapes  ($\theta_{\mathrm{gas}} = 30^\circ$)')
    axL.legend(loc='upper right', fontsize=8)

    # Right panel: V_d vs m, with Krug/Rivas constant-sigma reference
    for s in salts:
        m = m_grids[s.name]
        V_d = []
        D_d = []
        for mi in m:
            sigma, _ = butler_surface_tension(mi, s)
            res = detachment_volume(sigma, theta_gas)
            if res['converged']:
                V_d.append(res['V'])
                D_d.append(res['D'])
            else:
                V_d.append(np.nan)
                D_d.append(np.nan)
        V_d = np.array(V_d)
        D_d = np.array(D_d)
        results[s.name]['V_d'] = V_d
        results[s.name]['D_d'] = D_d
        axR.plot(m, 1e9 * V_d, color=colors[s.name],
                 label=f'{s.name}  (Butler-eNRTL $\\sigma(m)$)')
    # Reference: constant-sigma (Sepahi et al. 2022 / Raman et al. 2022)
    V_d_ref_info = detachment_volume(SIGMA_W, theta_gas)
    if V_d_ref_info['converged']:
        V_ref = 1e9 * V_d_ref_info['V']
        axR.axhline(V_ref, color='k', linestyle=':',
                    label=f'constant $\\sigma=72$ mN/m: $V_d=${V_ref:.2f} mm$^3$\n'
                          f'(Sepahi et al. 2022; Raman et al. 2022)')
    axR.set_xlabel('molality  m  (mol/kg water)')
    axR.set_ylabel(r'detachment volume  $V_d$  (mm$^3$)')
    axR.set_title('Detachment volume: eNRTL vs. constant-$\\sigma$ assumption')
    axR.legend(loc='best', fontsize=8)
    _save_fig(fig, fig_dir, 'fig4_detachment.png', citation_banner)

    # -------- Figure 5: VLE in bubble and critical nucleus --------
    R_bub_fixed = 50e-6   # 50 micron probe bubble for VLE
    S0 = 2.0              # reference supersaturation at pure water
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in salts:
        m = m_grids[s.name]
        y_H2 = []
        r_crit = []
        for mi in m:
            v = vle_bubble(mi, s, R_bub_fixed, S0=S0)
            y_H2.append(v['y_H2'])
            r_crit.append(v['r_crit'])
        y_H2 = np.array(y_H2)
        r_crit = np.array(r_crit)
        results[s.name]['y_H2'] = y_H2
        results[s.name]['r_crit'] = r_crit
        axL.plot(m, y_H2, color=colors[s.name], label=s.name)
        axR.plot(m, 1e9 * r_crit, color=colors[s.name], label=s.name)
    # Reference: r_crit with constant sigma and constant kH (Krug/Rivas assumption)
    r_crit_ref = 2.0 * SIGMA_W / ((S0 - 1.0) * P_ATM) * 1e9
    axR.axhline(r_crit_ref, color='k', linestyle=':',
                label=fr'const. $\sigma$ + const. $k_H$: $r^*=${r_crit_ref:.0f} nm''\n'
                      '(Sepahi et al. 2022; Raman et al. 2022)')
    axL.set_xlabel('molality  m  (mol/kg water)')
    axL.set_ylabel(r'$y_{\mathrm{H}_2}$ in bubble')
    axL.set_title(r'Bubble composition (H$_2$ + H$_2$O vapour) at $R$ = 50 $\mu$m')
    axL.legend()
    axR.set_xlabel('molality  m  (mol/kg water)')
    axR.set_ylabel(r'critical nucleus radius  $r^*$  (nm)')
    axR.set_title(r'$r^*$ from $\sigma(m)$ + salting-out  ($S_0=2$)')
    axR.legend()
    _save_fig(fig, fig_dir, 'fig5_vle_nucleation.png', citation_banner)

    # -------- Figure 6: ion identity (sigma vs molality AND vs ionic strength) --------
    # Point: for a fully-dissociated strong electrolyte, "salt molality" IS the
    # ion concentration.  But different salts with the SAME ionic strength give
    # DIFFERENT sigma because e-NRTL's short-range tau parameters are ion-pair
    # specific.  Plotting sigma(m) and sigma(I) side-by-side makes this visible:
    # the curves do not collapse on I alone -> ion-identity matters beyond I.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in salts:
        m = m_grids[s.name]
        sigma = results[s.name]['sigma']   # already computed
        # Ionic strength on mole-fraction basis converted back to molality basis:
        # I_m = 0.5 (nu_c z_c^2 + nu_a z_a^2) * m
        I_m = 0.5 * (s.nu_c * s.z_c**2 + s.nu_a * s.z_a**2) * m
        axL.plot(m, 1e3 * sigma, color=colors[s.name],
                 label=f'{s.name}  ({s.z_c}:{-s.z_a} electrolyte)')
        axR.plot(I_m, 1e3 * sigma, color=colors[s.name],
                 label=f'{s.name}  ({s.z_c}:{-s.z_a} electrolyte)')
    axL.axhline(1e3 * SIGMA_W, color='k', linestyle=':', linewidth=1)
    axR.axhline(1e3 * SIGMA_W, color='k', linestyle=':', linewidth=1)
    axL.set_xlabel('salt molality  m  (mol/kg water)')
    axL.set_ylabel(r'$\sigma$  (mN/m)')
    axL.set_title(r'(a)  $\sigma$ vs. salt molality')
    axL.legend()
    axR.set_xlabel(r'ionic strength  $I_m = \frac{1}{2}\sum_i \nu_i z_i^2\, m$  (mol/kg)')
    axR.set_ylabel(r'$\sigma$  (mN/m)')
    axR.set_title(r'(b)  $\sigma$ vs. ionic strength -- curves do NOT collapse')
    axR.legend()
    _save_fig(fig, fig_dir, 'fig6_ion_identity.png', citation_banner,
              existing_suptitle='Ion-identity beyond ionic strength: '
                                 'e-NRTL ion-pair specificity')

    # -------- summary CSV --------
    csv_path = os.path.join(out_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['salt', 'm (mol/kg)', 'gamma_pm', 'a_w',
                    'sigma (N/m)', 'kH_eff/kH0',
                    'V_d (m^3)', 'D_d (m)', 'y_H2', 'r_crit (m)'])
        for s in salts:
            n = len(results[s.name]['m'])
            for i in range(n):
                w.writerow([s.name,
                            f"{results[s.name]['m'][i]:.4g}",
                            f"{results[s.name]['gamma_pm'][i]:.4g}",
                            f"{results[s.name]['a_w'][i]:.5f}",
                            f"{results[s.name]['sigma'][i]:.5f}",
                            f"{results[s.name]['kH_ratio'][i]:.4g}",
                            f"{results[s.name]['V_d'][i]:.4g}",
                            f"{results[s.name]['D_d'][i]:.4g}",
                            f"{results[s.name]['y_H2'][i]:.4g}",
                            f"{results[s.name]['r_crit'][i]:.4g}"])

    # -------- print-to-console snapshot --------
    print(f'Wrote figures to {fig_dir}')
    print(f'Wrote {csv_path}')
    for s in salts:
        m_hi = m_grids[s.name][-1]
        sig_0 = butler_surface_tension(m_grids[s.name][0], s)[0]
        sig_hi = butler_surface_tension(m_hi, s)[0]
        print(f'  {s.name:6s}: sigma({m_grids[s.name][0]:.2f}) = {1e3*sig_0:5.2f} mN/m   '
              f'sigma({m_hi:.2f}) = {1e3*sig_hi:5.2f} mN/m')


if __name__ == '__main__':
    demo()
