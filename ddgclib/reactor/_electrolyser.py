"""
Reduced-order alkaline water electrolyser model for RL training.

Integrates the Lippmann control chain (modelling document §3.2):

    E_cell → σ_elec(Lippmann) → R_d(Fritz) → V_d → f_d → θ_avg → R_eff

Physics layers
--------------
1. Electrochemistry : Nernst + Butler-Volmer overpotentials
2. Thermodynamics   : e-NRTL activity coefficients (existing code, imported)
3. Lippmann         : σ_elec = σ_butler − ½ C_dl (E − E_pzc)²  (Eq. C18)
4. Bubble dynamics  : Modified Fritz balance with σ_elec
5. Mass balance     : Faraday's law → H₂/O₂ production rates
6. Energy balance   : Lumped-parameter cell temperature ODE

The main MARL agent provides environmental conditions via the Gymnasium
environment's ``set_conditions()`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ddgclib.reactor._lippmann import (
    sigma_lippmann,
    LippmannParams,
    lippmann_detachment_scaling,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
F_CONST: float = 96485.3329     # C/mol  Faraday constant
R_GAS: float = 8.314462618      # J/(mol·K)  universal gas constant
SIGMA_WATER: float = 0.07197    # N/m  pure water, 298 K


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class ElectrolyserParams:
    """Physical parameters for the reduced-order electrolyser.

    All SI units unless otherwise noted.
    """

    # ── Cell geometry ──
    A_cell: float = 0.01          # m²   active electrode area
    n_cells: int = 10             # –    number of cells in stack
    V_electrolyte: float = 0.001  # m³   electrolyte volume per cell

    # ── Electrochemistry ──
    E_rev_0: float = 1.229        # V    reversible cell voltage at STP
    alpha_a: float = 0.5          # –    anodic transfer coefficient
    alpha_c: float = 0.5          # –    cathodic transfer coefficient
    j_0: float = 1e-3             # A/m² exchange current density
    R_ohmic: float = 0.5e-4       # Ω·m² area-specific ohmic resistance

    # ── Electrolyte (KOH, alkaline) ──
    m_molality: float = 6.0       # mol/kg   KOH molality
    kappa_e: float = 21.5         # S/m      ionic conductivity
    eps_r: float = 78.4           # –        relative permittivity

    # ── Lippmann (Pt/KOH, modelling document §5.3) ──
    C_dl: float = 0.30            # F/m²     (30 µF/cm²)
    E_pzc: float = -0.07         # V vs RHE
    delta_V_max: float = 0.50     # V        electrowetting saturation limit

    # ── Thermal ──
    C_th: float = 500.0           # J/K  thermal capacitance
    UA_loss: float = 2.0          # W/K  heat-loss coefficient

    # ── Mars environment ──
    g: float = 3.721              # m/s²  Mars surface gravity
    T_ambient: float = 210.0      # K     Mars average surface temperature
    P_reactor: float = 5.0e5      # Pa    internal reactor pressure (pressurised)

    # ── Bubble dynamics ──
    theta_gas: float = 0.5236     # rad   (30°) gas-side contact angle
    sigma_0: float = SIGMA_WATER  # N/m   baseline surface tension

    # ── Operating limits ──
    j_max: float = 10000.0        # A/m²  maximum current density
    T_max: float = 473.15         # K     maximum cell temperature (200 °C)
    E_cell_min: float = 1.4       # V     minimum useful cell voltage
    E_cell_max: float = 2.5       # V     maximum cell voltage

    @property
    def lippmann_params(self) -> LippmannParams:
        """Build a :class:`LippmannParams` from these electrolyser params."""
        return LippmannParams(
            C_dl=self.C_dl,
            E_pzc=self.E_pzc,
            delta_V_max=self.delta_V_max,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Electrolyser:
    """Reduced-order electrolyser plant model.

    A single ``step()`` call runs the full Lippmann control chain and
    returns the updated state.  Execution time is ~0.1 ms, enabling
    millions of RL training steps.

    Parameters
    ----------
    params : ElectrolyserParams, optional
        Physical and operating parameters.  Uses Mars defaults if *None*.
    """

    def __init__(self, params: ElectrolyserParams | None = None):
        self.p = params or ElectrolyserParams()
        self._butler_sigma_cache: float | None = None
        self._reset_state()

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self.T_cell: float = 353.15           # K   (80 °C nominal)
        self.P_cell: float = self.p.P_reactor
        self.bubble_coverage: float = 0.0
        self.H2_produced_total: float = 0.0   # mol cumulative
        self.O2_produced_total: float = 0.0
        self.t: float = 0.0

        # Cached outputs (updated each step)
        self._H2_rate: float = 0.0
        self._O2_rate: float = 0.0
        self._V_cell: float = 0.0
        self._eta_f: float = 0.90
        self._spec_E: float = 0.0
        self._j_actual: float = 0.0
        self._sigma_elec: float = self.p.sigma_0
        self._det_volume: float = 0.0

    def reset(self, T_cell: float = 353.15) -> np.ndarray:
        """Reset to initial conditions and return the state array."""
        self._reset_state()
        self.T_cell = T_cell
        self._butler_sigma_cache = None
        return self.get_state_array()

    def get_state_array(self) -> np.ndarray:
        """Return the state as a flat 8-element array."""
        return np.array(
            [
                self.T_cell,
                self.bubble_coverage,
                self._H2_rate,
                self._O2_rate,
                self._V_cell,
                self._eta_f,
                self._spec_E,
                self._sigma_elec,
            ],
            dtype=np.float64,
        )

    def get_state_dict(self) -> dict:
        """Return the state as a named dictionary."""
        return {
            "T_cell": self.T_cell,
            "P_cell": self.P_cell,
            "bubble_coverage": self.bubble_coverage,
            "H2_rate": self._H2_rate,
            "O2_rate": self._O2_rate,
            "V_cell": self._V_cell,
            "eta_faradaic": self._eta_f,
            "specific_energy": self._spec_E,
            "sigma_elec": self._sigma_elec,
            "j_actual": self._j_actual,
            "det_volume": self._det_volume,
            "H2_total": self.H2_produced_total,
            "O2_total": self.O2_produced_total,
            "t": self.t,
        }

    # ------------------------------------------------------------------ #
    #  Electrochemistry                                                    #
    # ------------------------------------------------------------------ #
    def cell_voltage(self, j: float) -> float:
        """Compute cell voltage from current density.

        .. math::
            V_\\text{cell} = E_\\text{rev}(T)
                           + 2\\,\\eta_\\text{act}(j)
                           + j \\cdot R_\\Omega \\cdot (1 + 0.3\\,\\theta_\\text{cov})
        """
        p = self.p
        # Nernst temperature correction
        E_rev = p.E_rev_0 - 0.000846 * (self.T_cell - 298.15)

        # Butler-Volmer activation (Tafel approx for |j| >> j₀)
        if abs(j) > 1e-10:
            eta_act = (
                R_GAS * self.T_cell / (p.alpha_a * F_CONST)
            ) * np.arcsinh(j / (2.0 * p.j_0))
        else:
            eta_act = 0.0

        # Ohmic with bubble-coverage penalty
        eta_ohm = j * p.R_ohmic * (1.0 + 0.3 * self.bubble_coverage)

        return E_rev + 2.0 * abs(eta_act) + eta_ohm

    # ------------------------------------------------------------------ #
    #  Butler / e-NRTL sigma (lazy import from existing code)              #
    # ------------------------------------------------------------------ #
    def _get_butler_sigma(self) -> float:
        """Get Butler e-NRTL surface tension, with import fallback."""
        if self._butler_sigma_cache is not None:
            return self._butler_sigma_cache

        try:
            import sys
            import os

            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from cases_mean_flow.equil_bubble.bubble_enrtl_electrostatic_toy import (
                butler_surface_tension,
                KOH,
            )

            sigma_butler, _ = butler_surface_tension(self.p.m_molality, KOH)
            self._butler_sigma_cache = sigma_butler
        except Exception:
            # Fallback to pure-water value
            self._butler_sigma_cache = self.p.sigma_0

        return self._butler_sigma_cache

    # ------------------------------------------------------------------ #
    #  Bubble dynamics  (Lippmann-modified Fritz)                          #
    # ------------------------------------------------------------------ #
    def _compute_bubble_dynamics(
        self, j: float, sigma_elec: float, dt: float
    ) -> None:
        """Update bubble coverage using Lippmann-modified Fritz scaling.

        Uses the analytical scaling relations from modelling document §2.3:

        .. math::
            V_d^\\text{Lipp} / V_d^\\text{Fritz} = (\\sigma_\\text{elec} / \\sigma_0)^{3/2}

        and a first-order relaxation for electrode coverage.
        """
        p = self.p
        scaling = lippmann_detachment_scaling(p.sigma_0, sigma_elec)

        # Baseline Fritz detachment volume at Mars gravity
        rho_H2 = 2.016e-3 * self.P_cell / (R_GAS * self.T_cell)
        delta_rho = 997.05 - rho_H2
        delta_rho = max(delta_rho, 1.0)

        R_fritz = np.sin(p.theta_gas) * np.sqrt(
            3.0 * p.sigma_0 / (2.0 * delta_rho * p.g)
        )
        V_fritz = (4.0 / 3.0) * np.pi * R_fritz ** 3

        # Lippmann-modified detachment volume
        V_d = V_fritz * scaling["ratio_V"]
        self._det_volume = V_d

        # Departure frequency and coverage update
        if V_d > 0 and j > 0:
            n_dot_H2 = self._eta_f * j * p.A_cell / (2 * F_CONST)
            Q_H2 = n_dot_H2 * 2.016e-3 / max(rho_H2, 1e-6)   # m³/s
            f_dep = Q_H2 / V_d                                 # Hz

            # Equilibrium coverage (simplified Vogt-type model)
            coverage_eq = np.clip(j / p.j_max * 0.7, 0.0, 0.9)
            # Lippmann reduces coverage via smaller bubbles
            coverage_eq *= scaling["ratio_V"]

            # First-order relaxation
            tau_bubble = 1.0 / max(f_dep, 0.01)
            self.bubble_coverage += (
                (coverage_eq - self.bubble_coverage) * min(dt / tau_bubble, 1.0)
            )
        else:
            # Coverage decays when not producing
            self.bubble_coverage *= max(1.0 - dt * 0.1, 0.0)

        self.bubble_coverage = float(np.clip(self.bubble_coverage, 0.0, 0.95))

    # ------------------------------------------------------------------ #
    #  Main integration step                                               #
    # ------------------------------------------------------------------ #
    def step(
        self,
        j: float,
        E_cell: float,
        dt: float,
        P_available: float = np.inf,
        T_ambient: float | None = None,
    ) -> dict:
        """Advance the electrolyser state by one time step.

        Parameters
        ----------
        j : float
            Current density [A/m²] — from the PID controller.
        E_cell : float
            Applied cell voltage [V] — from the RL agent (Lippmann input).
        dt : float
            Time step [s].
        P_available : float
            Available electrical power [W] (solar constraint).
        T_ambient : float or None
            Ambient temperature [K] (from the main MARL agent).

        Returns
        -------
        dict
            Updated state (see :meth:`get_state_dict`).
        """
        p = self.p
        if T_ambient is None:
            T_ambient = p.T_ambient

        # ── 1. Lippmann surface tension (Eq. C18) ──
        sigma_butler = self._get_butler_sigma()
        self._sigma_elec = sigma_lippmann(
            sigma_butler, E_cell, p.lippmann_params
        )

        # ── 2. Power limiting ──
        V_cell_calc = self.cell_voltage(j)
        P_electrical = V_cell_calc * j * p.A_cell * p.n_cells
        if P_electrical > P_available > 0:
            j = j * (P_available / P_electrical)
            V_cell_calc = self.cell_voltage(j)
            P_electrical = V_cell_calc * j * p.A_cell * p.n_cells
        j = float(np.clip(j, 0.0, p.j_max))
        self._j_actual = j
        self._V_cell = V_cell_calc

        # ── 3. Faradaic efficiency (coverage-dependent) ──
        self._eta_f = max(0.90 - 0.10 * self.bubble_coverage, 0.50)

        # ── 4. Faraday's law ──
        n_H2 = self._eta_f * j * p.A_cell * p.n_cells * dt / (2 * F_CONST)
        n_O2 = n_H2 / 2.0

        # ── 5. Bubble dynamics (Lippmann-modified Fritz) ──
        self._compute_bubble_dynamics(j, self._sigma_elec, dt)

        # ── 6. Energy balance (lumped thermal) ──
        E_tn = 1.481   # V  thermoneutral voltage
        Q_gen = max(V_cell_calc - E_tn, 0.0) * j * p.A_cell * p.n_cells
        Q_loss = p.UA_loss * (self.T_cell - T_ambient)
        dT = (Q_gen - Q_loss) * dt / p.C_th
        self.T_cell += dT
        self.T_cell = float(np.clip(self.T_cell, T_ambient, p.T_max))

        # ── 7. Bookkeeping ──
        self.H2_produced_total += n_H2
        self.O2_produced_total += n_O2
        self.t += dt
        self._H2_rate = n_H2 / dt if dt > 0 else 0.0
        self._O2_rate = n_O2 / dt if dt > 0 else 0.0
        self._spec_E = (
            (P_electrical * dt) / max(n_H2 * 2.016e-3, 1e-12)
        )

        return self.get_state_dict()
