"""
pulloff_models.py
=================

Pluggable pull-off / detachment force models for the H2 bubble toy
(``bubble_enrtl_electrostatic_toy.py``, Layer C2).

Motivation
----------
The bare Fritz balance ``F_buoy = F_pin`` in this reduced model has **no
root** for any contact angle: at the largest closable Young-Laplace shape the
capillary pinning force always exceeds buoyancy (verified numerically over
theta = 5..150 deg).  The physically meaningful detachment criterion here is
therefore the **maximum closable volume** -- the largest quasi-static pinned
shape the interface can hold before it loses a solution and pinches off.  This
reproduces the classic Fritz detachment diameter (D ~ 1.7 mm at theta = 30 deg,
sigma = 73 mN/m) to two figures.

Two ways electrostatics couple to detachment
--------------------------------------------
Every contribution declares one of two coupling modes:

  * ``FORCE``   -- adds a vertical term to the (diagnostic) force budget.
                   Under the max-volume criterion these do NOT move V_d
                   (the closure limit is geometric); they are reported for
                   comparison.  Under ``criterion='force_balance'`` they
                   enter a genuine root find.
  * ``TENSION`` -- modifies sigma *before* the shape is solved, so it reshapes
                   the whole bubble and DOES move V_d.

Physics note (why there is no naive "double-layer Maxwell force" model):
the raw Maxwell stress of the ~1e9 V/m double-layer field over the bubble foot
would give absurd forces (~tens of N).  That stress is very nearly balanced by
the osmotic pressure of the diffuse layer; the *net* interfacial effect is
exactly the Lippmann lowering of sigma.  So double-layer electrostatics enter
correctly through the ``Lippmann`` (TENSION) model, not as an additive force.

This module is deliberately dependency-light: the shape solver is injected as
``shape_fn`` so there is no import cycle with the toy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Shared constants (kept local so the module is import-cycle free)
# ---------------------------------------------------------------------------
EPS_0 = 8.854187817e-12   # F/m
RHO_W = 997.05            # kg/m^3
RHO_H2 = 0.0899           # kg/m^3
G_EARTH = 9.80665         # m/s^2
G_MARS = 3.72             # m/s^2

FORCE = "FORCE"       # adds a vertical term to the force budget
TENSION = "TENSION"   # modifies the liquid-vapour tension sigma_lv
THETA = "THETA"       # modifies the three-phase contact angle theta_gas
FARADAY = 96485.33212     # C/mol
R_GAS = 8.314462618       # J/(mol K)


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------
@dataclass
class EnvState:
    """Operating conditions shared by every pull-off term."""
    # mechanics
    g: float = G_EARTH
    rho_l: float = RHO_W
    rho_g: float = RHO_H2
    # bulk electrostatics (DEP)
    j: float = 0.0            # current density [A/m^2]
    kappa_e: float = 21.5     # ionic conductivity [S/m]
    eps_r: float = 78.4       # electrolyte relative permittivity
    eps_g: float = 1.0        # gas relative permittivity
    # double-layer electrocapillarity (Lippmann)
    E_cell: Optional[float] = None   # applied cell voltage [V]; None => off
    E_pzc: float = -0.075            # potential of zero charge [V]
    C_dl: float = 0.30               # double-layer capacitance [F/m^2]

    @property
    def E_z(self) -> float:
        """Bulk ohmic field E_z = j / kappa_e [V/m] (0 if kappa_e ~ 0)."""
        return (self.j / self.kappa_e) if self.kappa_e > 1e-12 else 0.0


@dataclass
class BubbleState:
    """Geometry of one candidate pinned shape at apex curvature b."""
    b: float
    sigma: float
    theta_gas: float
    V: float
    r_cl: float
    shape: dict

    @property
    def R_d(self) -> float:
        """Equivalent-sphere radius from the exact volume."""
        return (3.0 * self.V / (4.0 * np.pi)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Model interface + concrete terms
# ---------------------------------------------------------------------------
class PullOffModel:
    """Base class.  Subclasses set ``coupling`` and override one method."""
    coupling: str = FORCE
    name: str = "base"

    def force(self, s: BubbleState, env: EnvState) -> float:
        """Vertical force [N], +ve = detaching (upward).  FORCE models only."""
        return 0.0

    def sigma_shift(self, sigma: float, env: EnvState) -> float:
        """Return modified sigma_lv [N/m].  TENSION models only."""
        return sigma

    def theta_shift(self, theta: float, sigma: float, env: EnvState) -> float:
        """Return modified contact angle theta_gas [rad].  THETA models only."""
        return theta


class Buoyancy(PullOffModel):
    """(rho_l - rho_g) g V -- upward, detaching."""
    coupling = FORCE
    name = "buoyancy"

    def force(self, s: BubbleState, env: EnvState) -> float:
        return (env.rho_l - env.rho_g) * env.g * s.V


class CapillaryPinning(PullOffModel):
    """-2 pi r_cl sigma sin(theta_gas) -- downward, resists detachment."""
    coupling = FORCE
    name = "pinning"

    def force(self, s: BubbleState, env: EnvState) -> float:
        return -2.0 * np.pi * s.r_cl * s.sigma * np.sin(s.theta_gas)


class BulkDEP(PullOffModel):
    """
    Dielectrophoretic pull-off from the *bulk ohmic* field E_z = j/kappa_e.

        F_DEP = pi * eps * CM * R_d^2 * E_z^2 ,   CM = (eps_r-eps_g)/(eps_r+eps_g)

    ``medium_eps=False`` (default) reproduces the original toy's prefactor
    ``eps = eps_0`` exactly.  ``medium_eps=True`` uses the physically correct
    Maxwell-stress prefactor ``eps = eps_0 * eps_r`` (~78x larger), which is
    still negligible versus buoyancy -- the point of the comparison.
    """
    coupling = FORCE
    name = "dep_bulk"

    def __init__(self, medium_eps: bool = False) -> None:
        self.medium_eps = medium_eps

    def force(self, s: BubbleState, env: EnvState) -> float:
        E = env.E_z
        if E == 0.0:
            return 0.0
        CM = (env.eps_r - env.eps_g) / (env.eps_r + env.eps_g)
        eps = EPS_0 * (env.eps_r if self.medium_eps else 1.0)
        return np.pi * eps * CM * s.R_d ** 2 * E ** 2


class Lippmann(PullOffModel):
    """
    Electrocapillarity applied *directly to sigma_lv* (SIMPLIFIED / DEPRECATED):
        sigma_elec = sigma_lv - 1/2 C_dl (E_cell - E_pzc)^2.

    TENSION-coupled.  This conflates the electrode|electrolyte double layer with
    the gas|liquid interface -- the double-layer capacitive energy physically
    lowers the *solid-liquid* tension (changing the contact angle via
    electrowetting), NOT the liquid-vapour tension.  Retained only for
    backward-comparison with the earlier model; prefer ``Electrowetting`` (THETA)
    for the physically correct three-phase-line coupling.
    """
    coupling = TENSION
    name = "lippmann"

    def sigma_shift(self, sigma: float, env: EnvState) -> float:
        if env.E_cell is None:
            return sigma
        reduction = 0.5 * env.C_dl * (env.E_cell - env.E_pzc) ** 2
        return max(sigma - reduction, 1e-6)   # keep positive for the shape ODE


class Electrowetting(PullOffModel):
    """
    Electrowetting / Young-Lippmann on the THREE-PHASE CONTACT ANGLE -- the
    physically correct coupling of applied potential to a sessile bubble
    (Mugele & Baret 2005; Berthier, EWOD):

        gamma_sl(V) = gamma_sl(V_pzc) - 1/2 C_dl (V - V_pzc)^2      (Lippmann)
        cos theta(V) = cos theta_0 + eta,   eta = C_dl (V-V_pzc)^2 / (2 sigma_lv)

    The double-layer capacitive energy lowers the SOLID-LIQUID tension (NOT
    sigma_lv), so by Young's equation the liquid wets the electrode better and
    the contact angle decreases -- the liquid undercuts the bubble, shrinks its
    foot, and it detaches at a SMALLER volume (electrowetting-assisted removal).

    Convention note (verified empirically for this codebase): the Young-Laplace
    solver's ``theta_gas`` behaves as the Fritz contact angle (larger angle ->
    larger foot -> larger V_d, reproducing D_F ~ theta).  The standard
    Young-Lippmann law (angle DECREASES with |V-V_pzc|) therefore applies
    directly to ``theta_gas`` with a ``+eta`` on cos(theta).

    Contact-angle SATURATION (theta cannot fall to 0; mechanism debated --
    charge trapping / dielectric breakdown; Mugele & Baret 2005) is imposed via
    ``theta_min`` [rad].

    Caveats (see manuscript/01_MODEL.md):
    * ``env.E_cell`` should be the WORKING-electrode double-layer potential vs
      V_pzc, not the full cell voltage (which lumps iR + counter-electrode +
      overpotential).  Using the cell voltage over-states the shift.
    * C_dl and V_pzc are themselves composition-dependent; a fuller model would
      couple them to the e-NRTL layer.
    * Acts on theta only, NOT on sigma_lv -- compose with an e-NRTL sigma_lv model.
    """
    coupling = THETA
    name = "electrowetting"

    def __init__(self, theta_min: float = np.deg2rad(5.0)) -> None:
        self.theta_min = theta_min

    def theta_shift(self, theta: float, sigma: float, env: EnvState) -> float:
        if env.E_cell is None or sigma <= 0:
            return theta
        eta = 0.5 * env.C_dl * (env.E_cell - env.E_pzc) ** 2 / sigma  # dimensionless
        cos_th = np.clip(np.cos(theta) + eta, -1.0, 1.0)
        return max(np.arccos(cos_th), self.theta_min)


def default_models() -> list[PullOffModel]:
    """Baseline FORCE budget used by ``detachment_volume``."""
    return [Buoyancy(), CapillaryPinning(), BulkDEP(medium_eps=False)]


# ---------------------------------------------------------------------------
# Swappable liquid-vapour surface-tension models (Layer B abstraction)
# ---------------------------------------------------------------------------
class SurfaceTensionModel:
    """
    Interface for sigma_lv(composition).  Lets a case swap the literature
    constant-sigma assumption for a composition-driven mixture model (e-NRTL
    Butler) without touching the detachment solver.
    """
    name: str = "base"

    def sigma_lv(self, m: float, salt=None, T: float = 298.15) -> float:
        raise NotImplementedError


class ConstantSigma(SurfaceTensionModel):
    """Fixed sigma_lv -- the typical literature assumption (e.g. 72 mN/m)."""
    name = "constant"

    def __init__(self, sigma: float = 0.072) -> None:
        self.sigma = sigma

    def sigma_lv(self, m: float, salt=None, T: float = 298.15) -> float:
        return self.sigma


class CallableSigma(SurfaceTensionModel):
    """
    Wrap any ``fn(m, salt) -> sigma`` (or ``-> (sigma, ...)``) as a
    SurfaceTensionModel.  Use to inject ``butler_surface_tension`` (e-NRTL)
    from the toy without an import cycle::

        CallableSigma(butler_surface_tension, name='enrtl_butler')
    """
    def __init__(self, fn: Callable, name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def sigma_lv(self, m: float, salt=None, T: float = 298.15) -> float:
        out = self.fn(m, salt) if salt is not None else self.fn(m)
        if isinstance(out, (tuple, list)):
            return float(out[0])
        # np.ravel handles 0-d, 1-d, and higher-dim arrays as well as scalars
        return float(np.ravel(out)[0])


# ---------------------------------------------------------------------------
# Detachment solver (criterion-explicit, model-agnostic)
# ---------------------------------------------------------------------------
def _refine_closure(shape_fn: Callable, sigma: float, theta_gas: float,
                    env: EnvState, b_lo: float, b_hi: float,
                    n_iter: int = 30) -> float:
    """Bisection on the upper convergence boundary: largest b that closes."""
    for _ in range(n_iter):
        b_mid = 0.5 * (b_lo + b_hi)
        sh = shape_fn(sigma, b_mid, theta_gas, rho_l=env.rho_l, g=env.g)
        if sh.get("converged"):
            b_lo = b_mid
        else:
            b_hi = b_mid
    return b_lo


def solve_detachment(sigma0: float, theta_gas: float,
                     models: Optional[list[PullOffModel]] = None,
                     env: Optional[EnvState] = None,
                     shape_fn: Optional[Callable] = None,
                     criterion: str = "max_volume",
                     n_scan: int = 40, b_cap_factor: float = 3.0) -> dict:
    """
    Locate the detachment shape and report the full force budget.

    Parameters
    ----------
    sigma0     : base surface tension from Layer B (N/m)
    theta_gas  : contact angle through the gas (rad)
    models     : list of PullOffModel; default ``default_models()``
    env        : EnvState; default EnvState()
    shape_fn   : Young-Laplace solver, signature
                 ``shape_fn(sigma, b, theta_gas, rho_l=..., g=...) -> dict``
    criterion  : 'max_volume'   -> largest closable pinned shape (recommended;
                                   reproduces the Fritz diameter), or
                 'force_balance'-> genuine root of the summed FORCE terms,
                                   bracketed only between converged shapes.
                                   Returns converged=False if none exists.

    Returns a dict with converged, criterion, b, V, D, r_cl, H, shape,
    sigma_eff, sigma_0, theta_eff, theta_0, per-term ``forces`` dict,
    F_buoy/F_DEP/F_pin aliases, net_force, and force_imbalance (|net|/F_buoy).
    """
    if models is None:
        models = default_models()
    if env is None:
        env = EnvState()
    if shape_fn is None:
        raise ValueError("solve_detachment requires shape_fn (the YL solver)")

    tension = [m for m in models if m.coupling == TENSION]
    theta_models = [m for m in models if m.coupling == THETA]
    forces = [m for m in models if m.coupling == FORCE]

    # 1. TENSION models reshape sigma_lv first ...
    sigma = sigma0
    for m in tension:
        sigma = m.sigma_shift(sigma, env)
    # ... then THETA models reshape the contact angle (may depend on sigma)
    theta_gas0 = theta_gas
    for m in theta_models:
        theta_gas = m.theta_shift(theta_gas, sigma, env)

    # 2. scan apex curvature; collect converged shapes
    cap = np.sqrt(sigma / ((env.rho_l - env.rho_g) * env.g))
    bs = np.linspace(0.05 * cap, b_cap_factor * cap, n_scan)
    states: list[BubbleState] = []
    state_idx: list[int] = []             # grid index of each converged state
    converged_mask = []
    for i, b in enumerate(bs):
        try:
            sh = shape_fn(sigma, b, theta_gas, rho_l=env.rho_l, g=env.g)
        except Exception:
            sh = {"converged": False}
        ok = bool(sh.get("converged"))
        converged_mask.append(ok)
        if ok:
            states.append(BubbleState(b, sigma, theta_gas,
                                      sh["V"], sh["r_cl"], sh))
            state_idx.append(i)
    if not states:
        return dict(converged=False, criterion=criterion, sigma_eff=sigma,
                    sigma_0=sigma0)

    def net(s: BubbleState) -> float:
        return sum(m.force(s, env) for m in forces)

    b_star = None
    fallback_state = states[-1]           # safe default reported shape (max_volume)
    if criterion == "force_balance":
        # bracket only GRID-ADJACENT converged states, so no interior
        # non-converged gap can be straddled (which would let brentq lock onto
        # a fabricated net=0 in the gap -- see _V_rcl).
        for k in range(len(states) - 1):
            if state_idx[k + 1] - state_idx[k] != 1:
                continue                  # a non-converged point sits between
            a, c = states[k], states[k + 1]
            fa, fc = net(a), net(c)
            if np.isfinite(fa) and np.isfinite(fc) and fa * fc < 0:
                b_star = brentq(
                    lambda b: net(BubbleState(
                        b, sigma, theta_gas,
                        *_V_rcl(shape_fn, sigma, b, theta_gas, env))),
                    a.b, c.b, rtol=1e-5)
                break
        # validate: a genuine root must land on a CONVERGED shape whose net
        # force is actually ~0 (guards against a fabricated-zero in a gap).
        if b_star is not None:
            sh = shape_fn(sigma, b_star, theta_gas, rho_l=env.rho_l, g=env.g)
            if sh.get("converged"):
                s_star = BubbleState(b_star, sigma, theta_gas,
                                     sh["V"], sh["r_cl"], sh)
                F_b = (env.rho_l - env.rho_g) * env.g * sh["V"]
                if abs(net(s_star)) / max(F_b, 1e-30) > 1e-3:
                    b_star = None
            else:
                b_star = None
        if b_star is None:
            return dict(converged=False, criterion="force_balance",
                        reason="no_true_root", sigma_eff=sigma, sigma_0=sigma0)
    else:  # max_volume
        s_top = max(states, key=lambda s: s.V)
        fallback_state = s_top
        b_star = s_top.b
        # if the max sits just before the convergence boundary, refine that
        # boundary to squeeze V up to the true closure limit
        idx = np.where(bs == s_top.b)[0]
        j_top = int(idx[0]) if len(idx) else None
        if j_top is not None and j_top + 1 < len(bs) and not converged_mask[j_top + 1]:
            b_star = _refine_closure(shape_fn, sigma, theta_gas, env,
                                     bs[j_top], bs[j_top + 1])

    # 3. build the reported shape + force budget at b_star
    shape = shape_fn(sigma, b_star, theta_gas, rho_l=env.rho_l, g=env.g)
    if not shape.get("converged"):
        if criterion == "force_balance":
            # never silently substitute an unrelated shape for a force balance
            return dict(converged=False, criterion="force_balance",
                        reason="no_true_root", sigma_eff=sigma, sigma_0=sigma0)
        # max_volume: boundary refinement can land a hair over the edge
        shape = fallback_state.shape
        b_star = fallback_state.b
    V_d = shape["V"]
    s_star = BubbleState(b_star, sigma, theta_gas, V_d, shape["r_cl"], shape)

    force_budget = {m.name: m.force(s_star, env) for m in forces}
    net_force = sum(force_budget.values())
    F_buoy = force_budget.get("buoyancy", 0.0)
    F_pin = -force_budget.get("pinning", 0.0)   # report as positive magnitude
    F_DEP = force_budget.get("dep_bulk", 0.0)

    return dict(
        converged=True,
        criterion=criterion,
        b=b_star,
        V=V_d,
        D=(6.0 * V_d / np.pi) ** (1.0 / 3.0),
        r_cl=shape["r_cl"],
        H=shape["H"],
        shape=shape,
        sigma_eff=sigma,
        sigma_0=sigma0,
        theta_eff=theta_gas,
        theta_0=theta_gas0,
        forces=force_budget,
        F_buoy=F_buoy,
        F_pin=F_pin,
        F_DEP=F_DEP,
        net_force=net_force,
        force_imbalance=abs(net_force) / max(F_buoy, 1e-30),
    )


def _V_rcl(shape_fn, sigma, b, theta_gas, env):
    """Helper for the force_balance root find: (V, r_cl, shape) at b."""
    sh = shape_fn(sigma, b, theta_gas, rho_l=env.rho_l, g=env.g)
    if not sh.get("converged"):
        return (0.0, 0.0, sh)
    return (sh["V"], sh["r_cl"], sh)
