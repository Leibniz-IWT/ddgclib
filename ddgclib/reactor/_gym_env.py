"""
Gymnasium environment for RL-based electrolysis reactor control.

Architecture (matches modelling document §3.2)
----------------------------------------------
::

    Main MARL Agent
        ↓  environmental conditions: T_ambient, P_solar, demand_H2
    RL Sub-Agent  (PPO / SAC)
        ↓  action: [E_cell_setpoint, H2_rate_setpoint]  (normalised ∈ [-1, 1])
    PID Controller  (inner loop, dt_pid = 10 s)
        ↓  manipulated variable: j(t)
    Electrolyser + Lippmann
        ↓  observations → RL Sub-Agent  (next decision)

Observation space  (Box, 11-dim, normalised to [0, 1])
------------------------------------------------------
=====  ==============================================
Index  Meaning
=====  ==============================================
  0    T_cell / T_max
  1    bubble_coverage
  2    H2_rate / H2_rate_max
  3    O2_rate / O2_rate_max
  4    V_cell / V_cell_max
  5    eta_faradaic
  6    specific_energy / spec_E_max
  7    sigma_elec / sigma_0   (Lippmann reduction ratio)
  8    P_solar / P_solar_max
  9    demand_H2 / demand_max
 10    time_of_sol / sol_duration
=====  ==============================================

Action space  (Box, 2-dim, ∈ [-1, 1])
-------------------------------------
=====  ============================================
Index  Mapping
=====  ============================================
  0    E_cell   ∈ [E_cell_min, E_cell_max]
  1    H₂ rate setpoint  ∈ [0, H2_rate_max]
=====  ============================================

Reward  (dual-objective: production + efficiency)
-------------------------------------------------
::

    r = w_prod   · min(H2_rate / demand, 1.0)
      − w_energy · (specific_energy / baseline)
      − w_temp   · max(0, T_cell − T_safe)² / 100
      − w_bubble · bubble_coverage²
      − w_smooth · |Δaction|²

Episode
-------
One Mars sol (88 775 s ≈ 24.66 hr).  RL decision every *dt_rl* seconds
(default 600 s = 10 min → 148 decisions per episode).
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ddgclib.reactor._electrolyser import Electrolyser, ElectrolyserParams
from ddgclib.reactor._pid import PIDController, PIDParams, GainScheduledPID


class ElectrolysisEnv(gym.Env):
    """Gymnasium environment for the electrolysis RL sub-agent.

    Parameters
    ----------
    config : dict, optional
        Configuration overrides.  Keys include ``dt_pid``, ``dt_rl``,
        ``demand_H2``, ``reactor`` (dict forwarded to
        :class:`ElectrolyserParams`), ``pid`` (dict forwarded to
        :class:`PIDParams`), and reward weights ``w_prod``, ``w_energy``,
        ``w_temp``, ``w_bubble``, ``w_smooth``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        # ── Timing ──
        self.sol_duration: float = 88775.0             # Mars sol [s]
        self.dt_pid: float = config.get("dt_pid", 10.0)
        self.dt_rl: float = config.get("dt_rl", 600.0)
        self.pid_steps_per_rl: int = int(self.dt_rl / self.dt_pid)

        # ── Normalisation limits ──
        self.H2_rate_max: float = config.get("H2_rate_max", 0.01)
        self.V_cell_max: float = config.get("V_cell_max", 3.0)
        self.spec_E_max: float = config.get("spec_E_max", 200e6)
        self.P_solar_max: float = config.get("P_solar_max", 5000.0)
        self.T_safe: float = config.get("T_safe", 363.15)   # 90 °C

        # ── Reward weights ──
        self.w_prod: float = config.get("w_prod", 1.0)
        self.w_energy: float = config.get("w_energy", 0.5)
        self.w_temp: float = config.get("w_temp", 10.0)
        self.w_bubble: float = config.get("w_bubble", 0.5)
        self.w_smooth: float = config.get("w_smooth", 0.1)

        # ── Demand (constant; main agent overrides via set_conditions) ──
        self.demand_H2: float = config.get("demand_H2", 0.005)

        # ── Sub-components ──
        reactor_cfg = config.get("reactor", {})
        self.reactor = Electrolyser(ElectrolyserParams(**reactor_cfg))
        self.pid = GainScheduledPID()

        # ── Spaces ──
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Episode state ──
        self.t: float = 0.0
        self.prev_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self.episode_H2: float = 0.0
        self.episode_energy: float = 0.0
        self._P_solar_override: float | None = None
        
        # ── Temporal Tracking for Obs ──
        self._prev_T = 353.15
        self._prev_H2 = 0.0
        self._prev_coverage = 0.0
        self._H2_integral_error = 0.0

    # ------------------------------------------------------------------ #
    #  Main-agent interface                                                #
    # ------------------------------------------------------------------ #
    def set_conditions(
        self,
        T_ambient: float | None = None,
        P_solar: float | None = None,
        demand_H2: float | None = None,
    ) -> None:
        """Called by the main MARL agent to update environmental conditions.

        Parameters
        ----------
        T_ambient : float, optional
            Ambient temperature [K].
        P_solar : float, optional
            Available solar power [W].  Overrides the internal cosine model.
        demand_H2 : float, optional
            Downstream H₂ demand [mol/s].
        """
        if T_ambient is not None:
            self.reactor.p.T_ambient = T_ambient
        if P_solar is not None:
            self._P_solar_override = P_solar
        if demand_H2 is not None:
            self.demand_H2 = demand_H2

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _solar_power(self, t: float) -> float:
        """Simple cosine solar power model for Mars."""
        if self._P_solar_override is not None:
            return self._P_solar_override
        phase = (t % self.sol_duration) / self.sol_duration
        return self.P_solar_max * max(
            np.cos(2 * np.pi * (phase - 0.5)), 0.0
        ) ** 1.2

    def _get_obs(self) -> np.ndarray:
        """Build 15-dim normalised observation vector."""
        s = self.reactor.get_state_dict()
        P_solar = self._solar_power(self.t)
        phase = (self.t % self.sol_duration) / self.sol_duration
        
        dT_dt = (s["T_cell"] - self._prev_T) / max(self.dt_rl, 1e-6)
        dH2_dt = (s["H2_rate"] - self._prev_H2) / max(self.dt_rl, 1e-6)
        dcov_dt = (s["bubble_coverage"] - self._prev_coverage) / max(self.dt_rl, 1e-6)
        self._H2_integral_error += (self.demand_H2 - s["H2_rate"]) * self.dt_rl

        obs = np.array(
            [
                s["T_cell"] / self.reactor.p.T_max,
                s["bubble_coverage"],
                min(s["H2_rate"] / self.H2_rate_max, 1.0),
                min(s["O2_rate"] / (self.H2_rate_max / 2), 1.0),
                min(s["V_cell"] / self.V_cell_max, 1.0),
                s["eta_faradaic"],
                min(s["specific_energy"] / self.spec_E_max, 1.0),
                s["sigma_elec"] / self.reactor.p.sigma_0 if self.reactor.p.sigma_0 > 0 else 0.0,
                P_solar / self.P_solar_max if self.P_solar_max > 0 else 0.0,
                self.demand_H2 / self.H2_rate_max if self.H2_rate_max > 0 else 0.0,
                phase,
                np.clip(dT_dt / 0.1, -1.0, 1.0),
                np.clip(dH2_dt / 0.001, -1.0, 1.0),
                np.clip(dcov_dt / 0.01, -1.0, 1.0),
                np.clip(self._H2_integral_error / 0.1, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        
        self._prev_T = s["T_cell"]
        self._prev_H2 = s["H2_rate"]
        self._prev_coverage = s["bubble_coverage"]
        
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        """Reset the environment to the start of a new Mars sol."""
        super().reset(seed=seed)
        self.reactor.reset()
        self.pid.reset(setpoint=self.demand_H2)
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.episode_H2 = 0.0
        self.episode_energy = 0.0
        self._P_solar_override = None
        
        self._prev_T = self.reactor.T_cell
        self._prev_H2 = 0.0
        self._prev_coverage = 0.0
        self._H2_integral_error = 0.0
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Execute one RL decision (multiple PID inner-loop steps).

        Parameters
        ----------
        action : np.ndarray, shape (2,)
            Normalised action from the RL agent.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        p = self.reactor.p

        # ── Decode action → physical setpoints ──
        E_cell = p.E_cell_min + (action[0] + 1) / 2 * (
            p.E_cell_max - p.E_cell_min
        )
        H2_setpoint = (action[1] + 1) / 2 * self.H2_rate_max

        self.pid.setpoint = H2_setpoint

        # ── PID inner loop ──
        interval_H2: float = 0.0
        interval_energy: float = 0.0
        for _ in range(self.pid_steps_per_rl):
            current_rate = self.reactor._H2_rate
            j = self.pid.update(current_rate, self.dt_pid)

            P_solar = self._solar_power(self.t)
            state = self.reactor.step(
                j=j,
                E_cell=E_cell,
                dt=self.dt_pid,
                P_available=P_solar,
            )
            self.t += self.dt_pid
            interval_H2 += state["H2_rate"] * self.dt_pid
            interval_energy += (
                state["specific_energy"]
                * state["H2_rate"]
                * 2.016e-3
                * self.dt_pid
            )

        self.episode_H2 += interval_H2
        self.episode_energy += interval_energy

        # ── Reward (dual-objective) ──
        s = self.reactor.get_state_dict()

        # Production tracking (higher is better)
        prod_ratio = min(
            s["H2_rate"] / max(self.demand_H2, 1e-10), 1.0
        )
        r_prod = self.w_prod * prod_ratio

        # Energy efficiency (lower specific energy is better)
        baseline_E = 55.0e6 * 3600.0    # 55 kWh/kg → J/kg
        r_energy = -self.w_energy * min(
            s["specific_energy"] / baseline_E, 2.0
        )

        # Thermal safety
        r_temp = (
            -self.w_temp
            * max(0.0, s["T_cell"] - self.T_safe) ** 2
            / 100.0
        )

        # Bubble coverage penalty
        r_bubble = -self.w_bubble * s["bubble_coverage"] ** 2

        # Action smoothness
        r_smooth = -self.w_smooth * float(
            np.sum((action - self.prev_action) ** 2)
        )

        reward = r_prod + r_energy + r_temp + r_bubble + r_smooth
        self.prev_action = action.copy()

        # ── Termination ──
        terminated = s["T_cell"] > self.reactor.p.T_max
        truncated = self.t >= self.sol_duration
        if terminated:
            reward -= 100.0

        info = {
            "H2_rate": s["H2_rate"],
            "demand": self.demand_H2,
            "T_cell": s["T_cell"],
            "V_cell": s["V_cell"],
            "E_cell": float(E_cell),
            "sigma_elec": s["sigma_elec"],
            "bubble_coverage": s["bubble_coverage"],
            "specific_energy_kWh_kg": s["specific_energy"] / 3.6e6,
            "episode_H2_mol": self.episode_H2,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
