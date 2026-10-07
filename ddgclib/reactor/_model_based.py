"""
Model-based planning for the electrolysis reactor.

Since we have an analytical, fast dynamics model (the ``Electrolyser``
class), we can use it as a "world model" for lookahead planning.  This
module implements a **random-shooting** planner that evaluates candidate
action sequences by simulating them forward and selecting the best one.

This is a model-based alternative to model-free RL (PPO / SAC).  It
requires no training but is limited by the planning horizon and the
number of candidate trajectories evaluated per step.

See Also
--------
ddgclib.reactor.benchmark : Comparison of model-free vs model-based.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddgclib.reactor._gym_env import ElectrolysisEnv


@dataclass
class PlannerConfig:
    """Configuration for the model-based planner.

    Attributes
    ----------
    n_candidates : int
        Number of random action sequences to evaluate per planning step.
    horizon : int
        Number of RL steps to simulate into the future.
    elite_fraction : float
        Fraction of top candidates to use for CEM refinement (0 = pure
        random shooting, > 0 = Cross-Entropy Method).
    n_cem_iterations : int
        Number of CEM refinement iterations (only if elite_fraction > 0).
    """

    n_candidates: int = 64
    horizon: int = 5
    elite_fraction: float = 0.2
    n_cem_iterations: int = 3


class _StateSnapshot:
    """Lightweight snapshot of the full environment + reactor state.

    This allows us to save and restore the environment for hypothetical
    rollouts without deepcopy overhead.
    """

    __slots__ = (
        "T_cell", "P_cell", "bubble_coverage",
        "H2_total", "O2_total", "t_reactor",
        "_H2_rate", "_O2_rate", "_V_cell", "_eta_f",
        "_spec_E", "_j_actual", "_sigma_elec", "_det_volume",
        "t_env", "pid_integral", "pid_prev_error", "pid_prev_derivative",
        "pid_setpoint", "prev_action", "episode_H2", "episode_energy",
        "_P_solar_override", "demand_H2",
    )

    @classmethod
    def capture(cls, env: ElectrolysisEnv) -> _StateSnapshot:
        """Capture the current environment state."""
        snap = cls()
        r = env.reactor
        snap.T_cell = r.T_cell
        snap.P_cell = r.P_cell
        snap.bubble_coverage = r.bubble_coverage
        snap.H2_total = r.H2_produced_total
        snap.O2_total = r.O2_produced_total
        snap.t_reactor = r.t
        snap._H2_rate = r._H2_rate
        snap._O2_rate = r._O2_rate
        snap._V_cell = r._V_cell
        snap._eta_f = r._eta_f
        snap._spec_E = r._spec_E
        snap._j_actual = r._j_actual
        snap._sigma_elec = r._sigma_elec
        snap._det_volume = r._det_volume
        snap.t_env = env.t
        snap.pid_integral = env.pid.integral
        snap.pid_prev_error = env.pid.prev_error
        snap.pid_prev_derivative = env.pid.prev_derivative
        snap.pid_setpoint = env.pid.setpoint
        snap.prev_action = env.prev_action.copy()
        snap.episode_H2 = env.episode_H2
        snap.episode_energy = env.episode_energy
        snap._P_solar_override = env._P_solar_override
        snap.demand_H2 = env.demand_H2
        return snap

    def restore(self, env: ElectrolysisEnv) -> None:
        """Restore a previously captured state into the environment."""
        r = env.reactor
        r.T_cell = self.T_cell
        r.P_cell = self.P_cell
        r.bubble_coverage = self.bubble_coverage
        r.H2_produced_total = self.H2_total
        r.O2_produced_total = self.O2_total
        r.t = self.t_reactor
        r._H2_rate = self._H2_rate
        r._O2_rate = self._O2_rate
        r._V_cell = self._V_cell
        r._eta_f = self._eta_f
        r._spec_E = self._spec_E
        r._j_actual = self._j_actual
        r._sigma_elec = self._sigma_elec
        r._det_volume = self._det_volume
        env.t = self.t_env
        env.pid.integral = self.pid_integral
        env.pid.prev_error = self.pid_prev_error
        env.pid.prev_derivative = self.pid_prev_derivative
        env.pid.setpoint = self.pid_setpoint
        env.prev_action = self.prev_action.copy()
        env.episode_H2 = self.episode_H2
        env.episode_energy = self.episode_energy
        env._P_solar_override = self._P_solar_override
        env.demand_H2 = self.demand_H2


class ModelBasedPlanner:
    """Lookahead planner using the analytical reactor model.

    At each RL decision step, evaluates *n_candidates* random action
    sequences by simulating *horizon* steps ahead with the ``Electrolyser``
    model, then returns the first action of the best-performing sequence.

    With ``elite_fraction > 0``, uses the Cross-Entropy Method (CEM) to
    iteratively refine the action distribution towards high-reward regions.

    Parameters
    ----------
    env : ElectrolysisEnv
        The Gymnasium environment (used as the world model).
    config : PlannerConfig, optional
        Planning configuration.

    Examples
    --------
    >>> from ddgclib.reactor._gym_env import ElectrolysisEnv
    >>> env = ElectrolysisEnv()
    >>> planner = ModelBasedPlanner(env)
    >>> obs, _ = env.reset()
    >>> action = planner.plan(obs)
    """

    def __init__(
        self,
        env: ElectrolysisEnv,
        config: PlannerConfig | None = None,
    ):
        self.env = env
        self.cfg = config or PlannerConfig()

    def plan(self, obs: np.ndarray) -> np.ndarray:
        """Select the best action via model-based lookahead.

        Parameters
        ----------
        obs : np.ndarray
            Current observation (unused by the planner, but kept for API
            compatibility with RL agents).

        Returns
        -------
        np.ndarray, shape (2,)
            The best first action found, in [-1, 1].
        """
        cfg = self.cfg
        n_elite = max(1, int(cfg.n_candidates * cfg.elite_fraction))

        # Initial action distribution: uniform [-1, 1]
        mu = np.zeros((cfg.horizon, 2), dtype=np.float64)
        sigma = np.ones((cfg.horizon, 2), dtype=np.float64)

        best_action = np.zeros(2, dtype=np.float32)
        best_reward = -np.inf

        for cem_iter in range(max(1, cfg.n_cem_iterations)):
            # Sample candidates
            candidates = np.clip(
                mu[np.newaxis] + sigma[np.newaxis] * np.random.randn(
                    cfg.n_candidates, cfg.horizon, 2
                ),
                -1.0, 1.0,
            )

            # Evaluate each candidate sequence
            rewards = np.empty(cfg.n_candidates, dtype=np.float64)
            snapshot = _StateSnapshot.capture(self.env)

            for i in range(cfg.n_candidates):
                rewards[i] = self._rollout(candidates[i])
                snapshot.restore(self.env)

            # Track global best
            idx_best = np.argmax(rewards)
            if rewards[idx_best] > best_reward:
                best_reward = rewards[idx_best]
                best_action = candidates[idx_best, 0].astype(np.float32)

            # CEM: refit distribution to elite candidates
            if cfg.elite_fraction > 0 and cem_iter < cfg.n_cem_iterations - 1:
                elite_idx = np.argsort(rewards)[-n_elite:]
                elite = candidates[elite_idx]
                mu = np.mean(elite, axis=0)
                sigma = np.std(elite, axis=0) + 1e-6  # prevent collapse

        return best_action

    def _rollout(self, action_seq: np.ndarray) -> float:
        """Simulate a trajectory and return cumulative reward."""
        total_reward = 0.0
        for action in action_seq:
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward
