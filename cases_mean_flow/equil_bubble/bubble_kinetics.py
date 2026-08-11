"""
bubble_kinetics.py
==================

Layer D/F/G of the electrolysis bubble pipeline: the chain that turns a
detachment SIZE (V_d, from the pull-off models) into detachment RATES and
gas-production rates, plus concentration polarization and uncertainty.

    reaction rate (Faradaic)  ->  gas flux
                              ->  local composition near electrode (Nernst)
                              ->  bubble growth to V_d  ->  detachment time t_d
                              ->  detachment frequency f = 1/t_d
                              ->  gas-production partition / rate
                              ->  uncertainty (Monte Carlo, + constant-sigma bias)

Design intent
-------------
This module is deliberately a set of SMALL, SWAPPABLE pieces with clearly
labelled STUBS, so the richer physics (Butler-Volmer kinetics, nucleation-site
statistics, non-constant growth laws, 2D Nernst-Planck fields) can be dropped in
later without changing call sites.  Verified first-pass forms:

    Faradaic molar flux         N = j / (n F)                    [mol/(m^2 s)]
    volumetric gas flux         q = N R T / P                    [m^3/(m^2 s)]
    Nernst surface conc.        c_s = c_bulk +/- j delta/(n F D) [mol/m^3]
    limiting current            j_lim = n F D c_bulk / delta     [A/m^2]
    growth time to detachment   t_d = V_d / Q_bubble             [s]
    detachment frequency        f   = 1 / t_d = Q_bubble / V_d   [1/s]

See manuscript/01_MODEL.md and 03_ROADMAP.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

FARADAY = 96485.33212       # C/mol
R_GAS = 8.314462618         # J/(mol K)
P_ATM = 1.01325e5           # Pa
T_REF = 298.15              # K
RHO_W = 997.05              # kg/m^3
M_W = 18.01528e-3           # kg/mol


# ---------------------------------------------------------------------------
# Reaction rate  (STUBS -- default is simple galvanostatic / Faradaic)
# ---------------------------------------------------------------------------
@dataclass
class ReactionRate:
    """
    Faradaic reaction rate for a gas-evolving electrode.

    Default (galvanostatic): the current density ``j`` is prescribed and the
    molar gas flux is N = j/(n F).  Subclass / swap for potential-controlled
    kinetics.

    Attributes
    ----------
    n_electrons : electrons per gas molecule (H2: 2, O2: 4).
    """
    n_electrons: int = 2

    def molar_flux(self, j: float, **kw) -> float:
        """Molar gas flux per electrode area N [mol/(m^2 s)]."""
        return j / (self.n_electrons * FARADAY)


class ButlerVolmerRate(ReactionRate):
    """
    STUB -- potential-controlled Butler-Volmer / Tafel kinetics.

    j(eta) = j0 [ exp(alpha_a F eta / RT) - exp(-alpha_c F eta / RT) ]

    Provided so a case can pass an overpotential ``eta`` instead of a fixed j.
    Not exercised by the default pipeline yet; wire up when kinetics matter.
    """
    def __init__(self, n_electrons: int = 2, j0: float = 1e-3,
                 alpha_a: float = 0.5, alpha_c: float = 0.5, T: float = T_REF):
        self.n_electrons = n_electrons
        self.j0, self.alpha_a, self.alpha_c, self.T = j0, alpha_a, alpha_c, T

    def current_density(self, eta: float) -> float:
        f = FARADAY / (R_GAS * self.T)
        return self.j0 * (np.exp(self.alpha_a * f * eta)
                          - np.exp(-self.alpha_c * f * eta))

    def molar_flux(self, j: float = None, eta: float = None, **kw) -> float:
        if j is None:
            if eta is None:
                raise ValueError("ButlerVolmerRate needs j or eta")
            j = self.current_density(eta)
        return j / (self.n_electrons * FARADAY)


# ---------------------------------------------------------------------------
# Concentration polarization  (1D steady Nernst diffusion layer)
# ---------------------------------------------------------------------------
@dataclass
class NernstDiffusionLayer:
    """
    1D steady Nernst diffusion-layer estimate of the LOCAL composition at the
    electrode, which then feeds the e-NRTL surface-tension model.

        c_surf = c_bulk + sign * j * delta / (n F D)
        j_lim  = n F D c_bulk / delta

    ``sign = +1`` if the salt/relevant ion is ENRICHED at the electrode (e.g.
    OH- produced at an alkaline cathode), ``-1`` if depleted (reactant consumed).

    Parameters
    ----------
    delta : diffusion-layer thickness [m].  ~1e-4 m for natural convection,
            ~1e-5 m for vigorous stirring / gas-sparging (Bard & Faulkner Ch.1).
    D     : effective diffusivity of the polarizing species [m^2/s] (~1e-9).
    n     : electrons transferred per formula event coupling j to the flux.
    sign  : +1 enrich / -1 deplete.

    STUB: a full 2D/axisymmetric Nernst-Planck (migration + diffusion) field
    solve around the growing bubble is a later refinement; the 1D algebraic
    estimate is adequate for a first-pass magnitude (see manuscript/03_ROADMAP).
    """
    delta: float = 1e-4
    D: float = 1e-9
    n: int = 1
    sign: int = +1
    M_solute: float = 0.05846   # solute molar mass [kg/mol]; default ~ NaCl

    def limiting_current(self, c_bulk: float) -> float:
        """c_bulk in mol/m^3 -> j_lim in A/m^2."""
        return self.n * FARADAY * self.D * c_bulk / self.delta

    def surface_molality(self, m_bulk: float, j: float,
                         rho_solvent: float = RHO_W) -> float:
        """
        Local molality [mol/kg] at the electrode given bulk molality and j.

        Converts the bulk molality to a bulk molarity, applies the Nernst
        shift, and converts back.  The (1 + m*M_solute) term uses the SOLUTE
        molar mass (not water); set ``M_solute`` per salt.  Clamped at >= 0.
        """
        M = self.M_solute
        c_bulk = m_bulk * rho_solvent / (1.0 + m_bulk * M)     # mol/m^3
        dc = self.sign * j * self.delta / (self.n * FARADAY * self.D)
        c_surf = max(c_bulk + dc, 0.0)
        # convert molarity back to molality
        return c_surf / max(rho_solvent - c_surf * M, 1e-6)


# ---------------------------------------------------------------------------
# Growth -> detachment time -> frequency -> gas-production partition
# ---------------------------------------------------------------------------
@dataclass
class BubbleGrowth:
    """
    Constant-flux (reaction-limited) bubble growth to the detachment volume.

    A single bubble on one nucleation site captures a fraction ``capture_eff``
    of the Faradaic gas produced over its catchment area ``A_catch`` (= 1 /
    site density).  Growth time and detachment frequency follow directly.

    STUB: diffusion-limited growth (R ~ t^1/2) and coalescence are not modelled;
    constant-flux (R ~ t^1/3 in volume-linear-in-time) is the first-pass law.
    """
    site_density: float = 1e6      # nucleation sites per m^2 [1/m^2]
    capture_eff: float = 1.0       # fraction of local Faradaic gas -> this bubble
    T: float = T_REF
    P: float = P_ATM

    @property
    def A_catch(self) -> float:
        """Catchment area per active site [m^2]."""
        return 1.0 / self.site_density

    def bubble_gas_rate(self, N_molar: float) -> float:
        """
        Volumetric gas rate into one bubble Q_bubble [m^3/s] from the molar
        Faradaic flux N [mol/(m^2 s)] (ideal gas at T, P).
        """
        return self.capture_eff * N_molar * self.A_catch * R_GAS * self.T / self.P

    def detachment_time(self, V_d: float, N_molar: float) -> float:
        Q = self.bubble_gas_rate(N_molar)
        return V_d / Q if Q > 0 else np.inf

    def detachment_frequency(self, V_d: float, N_molar: float) -> float:
        t = self.detachment_time(V_d, N_molar)
        return 1.0 / t if np.isfinite(t) and t > 0 else 0.0


@dataclass
class DetachmentRateResult:
    """Container for the size -> rate chain outputs."""
    V_d: float
    D_d: float
    sigma_eff: float
    theta_eff: float
    m_surface: float
    N_molar: float          # mol/(m^2 s), Faradaic
    Q_bubble: float         # m^3/s per bubble
    t_detach: float         # s
    f_detach: float         # 1/s
    gas_rate_area: float    # m^3/(m^2 s) captured gas flux per electrode area


def detachment_rate(V_d: float, sigma_eff: float, theta_eff: float,
                    m_surface: float, j: float,
                    reaction: ReactionRate, growth: BubbleGrowth) -> DetachmentRateResult:
    """
    Assemble the full size -> rate result for one operating point.

    The detachment SIZE V_d is supplied by the pull-off solver; this function
    adds the reaction/growth chain.  Note f_detach ~ 1/V_d, so the mixture- and
    field-driven changes in V_d translate directly into detachment-frequency
    changes.
    """
    N = reaction.molar_flux(j=j)
    Q = growth.bubble_gas_rate(N)
    t = growth.detachment_time(V_d, N)
    f = growth.detachment_frequency(V_d, N)
    gas_area = growth.capture_eff * N * R_GAS * growth.T / growth.P
    D_d = (6.0 * V_d / np.pi) ** (1.0 / 3.0)
    return DetachmentRateResult(
        V_d=V_d, D_d=D_d, sigma_eff=sigma_eff, theta_eff=theta_eff,
        m_surface=m_surface, N_molar=N, Q_bubble=Q, t_detach=t, f_detach=f,
        gas_rate_area=gas_area)


# ---------------------------------------------------------------------------
# Uncertainty propagation  (Monte Carlo)
# ---------------------------------------------------------------------------
@dataclass
class UncertainParam:
    """A parameter with a (log)normal or uniform uncertainty."""
    name: str
    nominal: float
    rel_sigma: float = 0.0        # relative std for 'normal'/'lognormal'
    dist: str = "normal"          # 'normal' | 'lognormal' | 'uniform'
    lo: Optional[float] = None    # for 'uniform'
    hi: Optional[float] = None

    def sample(self, rng: np.random.Generator) -> float:
        if self.dist == "uniform":
            return rng.uniform(self.lo, self.hi)
        if self.dist == "lognormal":
            mu = np.log(self.nominal)
            return float(np.exp(rng.normal(mu, self.rel_sigma)))
        return float(rng.normal(self.nominal, self.rel_sigma * self.nominal))


def monte_carlo(evaluate: Callable[[dict], dict], params: list[UncertainParam],
                n: int = 300, seed: int = 0) -> dict:
    """
    Propagate parameter uncertainty through an arbitrary ``evaluate`` mapping.

    Parameters
    ----------
    evaluate : function taking a dict {name: sampled_value} and returning a dict
               of scalar outputs (e.g. {'V_d':..., 'f_detach':...}).
    params   : list of UncertainParam to sample independently.
    n        : number of Monte-Carlo samples.
    seed     : RNG seed (Date/Random are otherwise unavailable in workflows;
               callers pass an explicit seed for reproducibility).

    Returns {output_name: {'mean','std','p05','p50','p95','samples'}}.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n):
        draw = {p.name: p.sample(rng) for p in params}
        try:
            r = evaluate(draw)
        except Exception:
            continue
        if r:                       # skip empty/failed draws (e.g. non-converged)
            rows.append(r)
    if not rows:
        return {}
    # union of keys across ALL rows (not just rows[0], which may be partial)
    keys = set().union(*(r.keys() for r in rows))
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
        if len(vals) == 0:
            continue
        out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                      p05=float(np.percentile(vals, 5)),
                      p50=float(np.percentile(vals, 50)),
                      p95=float(np.percentile(vals, 95)),
                      samples=vals)
    return out
