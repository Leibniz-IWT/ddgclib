"""Data-driven (back-computed) contact-line forcing for capillary rise.

There is no rigorous mesh-independent three-phase contact force model in
the library yet.  Instead of modelling the dynamic contact angle, this
module *back-computes* the capillary driving force from the experimental
data of Heshmati & Piri (2014) as fitted/sampled by Lunowa et al. (2022)
(``data/caprise/``):

1. ``ExperimentalDrive`` — interpolates the measured dynamic contact
   angle CA(t) and provides the resulting meniscus driving pressure
   ``P_cap(t) = 2 gamma cos(theta_exp(t)) / R`` (tube) or the matched
   2D-slit equivalent.  The simulation then *predicts* the meniscus
   rise h(t), which is compared against the measured rise.

2. ``backcompute_pcap_from_h`` — the inverse route: given the measured
   h(t), compute the driving pressure (and implied contact angle) that
   the reduced momentum balance requires.  Comparing this against the
   measured CA(t) closes the loop on data consistency.

Matched 2D slit
---------------
A 2D slit of half-width ``a = R/2`` with viscosity ``mu_2d = (2/3) mu``
has *identical* reduced-order dynamics to the 3D tube of radius R:

    tube:  rho d/dt(h hdot) = 2 gamma cos(theta)/R - rho g h - 8 mu h hdot/R^2
    slit:  rho d/dt(h hdot) =   gamma cos(theta)/a - rho g h - 3 mu_2 h hdot/a^2

    a = R/2, mu_2 = (2/3) mu  =>  same driving pressure, same drag
    coefficient, same Jurin height.

References
----------
Heshmati & Piri (2014), Langmuir 30, 14151-14162.
Lunowa et al. (2022), Langmuir 38, DOI 10.1021/acs.langmuir.1c02680.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

from ._data import load_sample_data
from ._params import FLUIDS, g as G_DEFAULT


def matched_slit_params(R: float, mu: float) -> tuple[float, float]:
    """Return (half-width a, viscosity mu_2d) of the matched 2D slit."""
    return 0.5 * R, (2.0 / 3.0) * mu


class ExperimentalDrive:
    """Interpolated experimental drive for a fluid / tube-radius pair.

    Provides the measured rise ``h_exp(t)``, its rate ``hdot_exp(t)``,
    the measured dynamic contact angle ``theta_exp(t)`` and the
    back-computed capillary driving pressure ``P_cap(t)``.

    Outside the measured time range the contact angle is clamped to its
    first / last measured value.
    """

    def __init__(self, fluid: str, R_mm: float):
        self.fluid = fluid
        self.R_mm = R_mm
        self.R = R_mm * 1e-3
        fp = FLUIDS[fluid]
        self.rho = fp["rho"]
        self.mu = fp["mu"]
        self.gamma = fp["gamma"]
        self.theta_s_deg = fp["theta_s_deg"]

        d = load_sample_data(fluid, R_mm)
        t = d["t_s"]
        h = d["h_m"]
        ca = d["ca_deg"]

        # Rise interpolant (h is monotone -> PCHIP is shape-preserving)
        ok_h = np.isfinite(t) & np.isfinite(h)
        self.t_data = t[ok_h]
        self.h_data = h[ok_h]
        self._h_ip = PchipInterpolator(self.t_data, self.h_data)
        self._hdot_ip = self._h_ip.derivative()

        # Contact angle interpolant (first sample often NaN)
        ok_ca = np.isfinite(t) & np.isfinite(ca)
        self._t_ca = t[ok_ca]
        self._ca = ca[ok_ca]
        self._ca_ip = PchipInterpolator(self._t_ca, self._ca)

        self.t_min = float(self.t_data[0])
        self.t_max = float(self.t_data[-1])

    # -- kinematics ------------------------------------------------------
    def h_exp(self, t):
        return self._h_ip(np.clip(t, self.t_min, self.t_max))

    def hdot_exp(self, t):
        t = np.asarray(t, dtype=float)
        inside = (t >= self.t_min) & (t <= self.t_max)
        out = np.where(inside, self._hdot_ip(np.clip(t, self.t_min, self.t_max)), 0.0)
        return out if out.ndim else float(out)

    # -- contact angle / driving force -----------------------------------
    def theta_exp_deg(self, t):
        return self._ca_ip(np.clip(t, self._t_ca[0], self._t_ca[-1]))

    def cos_theta_exp(self, t):
        return np.cos(np.deg2rad(self.theta_exp_deg(t)))

    def p_cap_tube(self, t):
        """Meniscus driving pressure in the 3D tube [Pa]."""
        return 2.0 * self.gamma * self.cos_theta_exp(t) / self.R

    def p_cap_slit(self, t):
        """Driving pressure of the matched 2D slit [Pa] (== p_cap_tube)."""
        a, _ = matched_slit_params(self.R, self.mu)
        return self.gamma * self.cos_theta_exp(t) / a


def rise_ode_solve(
    drive: ExperimentalDrive,
    t0: float,
    t_end: float,
    h0: float | None = None,
    hdot0: float | None = None,
    theta_mode: str = "dynamic",
    inertia: bool = True,
    g: float = G_DEFAULT,
    n_eval: int = 400,
):
    """Integrate the reduced tube momentum balance with data-driven forcing.

        rho (h hddot + hdot^2) = P_cap(t) - rho g h - 8 mu h hdot / R^2

    Parameters
    ----------
    theta_mode : 'dynamic' | 'static'
        'dynamic' uses the measured CA(t); 'static' uses theta_s.
    inertia : bool
        If False, drop the inertial terms (classic Lucas-Washburn).

    Returns (t, h, hdot).
    """
    rho, mu, R = drive.rho, drive.mu, drive.R
    if h0 is None:
        h0 = float(drive.h_exp(t0))
    if hdot0 is None:
        hdot0 = float(drive.hdot_exp(t0))
    h0 = max(h0, 1e-6)

    if theta_mode == "dynamic":
        p_cap = drive.p_cap_tube
    elif theta_mode == "static":
        p0 = 2.0 * drive.gamma * math.cos(math.radians(drive.theta_s_deg)) / R
        p_cap = lambda t: p0
    else:
        raise ValueError(f"unknown theta_mode {theta_mode!r}")

    if inertia:
        def rhs(t, y):
            h, hd = y
            h = max(h, 1e-9)
            hdd = (p_cap(t) - rho * g * h - 8.0 * mu * h * hd / R**2
                   - rho * hd * hd) / (rho * h)
            return [hd, hdd]
        y0 = [h0, hdot0]
    else:
        def rhs(t, y):
            h = max(y[0], 1e-9)
            return [(p_cap(t) - rho * g * h) * R**2 / (8.0 * mu * h)]
        y0 = [h0]

    t_eval = np.linspace(t0, t_end, n_eval)
    sol = solve_ivp(rhs, (t0, t_end), y0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-10, max_step=(t_end - t0) / 200)
    h = sol.y[0]
    hdot = sol.y[1] if inertia else np.gradient(h, sol.t)
    return sol.t, h, hdot


def backcompute_pcap_from_h(
    drive: ExperimentalDrive,
    t: np.ndarray | None = None,
    inertia: bool = True,
    g: float = G_DEFAULT,
):
    """Back-compute the driving pressure (and implied CA) from measured h(t).

        P_cap_impl(t) = rho (h hddot + hdot^2) + rho g h + 8 mu h hdot / R^2

    Returns dict with 't', 'p_cap_impl', 'theta_impl_deg' (NaN where
    |cos| > 1), and the same quantities from the *measured* CA for
    comparison ('p_cap_meas', 'theta_meas_deg').
    """
    if t is None:
        # Interior of the data range, avoiding the endpoints where the
        # PCHIP derivative is one-sided.
        t = np.linspace(drive.t_min, drive.t_max, 400)[2:-2]
    rho, mu, R = drive.rho, drive.mu, drive.R

    h = drive.h_exp(t)
    hd = drive._hdot_ip(t)
    hdd = drive._hdot_ip.derivative()(t)

    p_impl = rho * g * h + 8.0 * mu * h * hd / R**2
    if inertia:
        p_impl = p_impl + rho * (h * hdd + hd * hd)

    cos_impl = p_impl * R / (2.0 * drive.gamma)
    theta_impl = np.full_like(cos_impl, np.nan)
    ok = np.abs(cos_impl) <= 1.0
    theta_impl[ok] = np.rad2deg(np.arccos(cos_impl[ok]))

    return {
        "t": t,
        "p_cap_impl": p_impl,
        "theta_impl_deg": theta_impl,
        "p_cap_meas": drive.p_cap_tube(t),
        "theta_meas_deg": drive.theta_exp_deg(t),
    }
