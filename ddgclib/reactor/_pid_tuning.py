"""
Auto-tuning for the PID controller via relay feedback.

Implements the Åström-Hägglund relay feedback method to determine
the ultimate gain and period of the plant, then applies Ziegler-Nichols
tuning rules to compute optimal PID gains.

References
----------
[1] Åström, K.J. & Hägglund, T. "Automatic Tuning of PID Controllers"
    (ISA, 1988).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ddgclib.reactor._electrolyser import Electrolyser, ElectrolyserParams
from ddgclib.reactor._pid import PIDParams


@dataclass(frozen=True)
class TuningResult:
    """Results from a relay feedback auto-tuning experiment.

    Attributes
    ----------
    converged : bool
        Whether enough oscillation cycles were observed.
    K_u : float
        Ultimate gain [A/m² per mol/s].
    T_u : float
        Ultimate period [s].
    Kp : float
        Recommended proportional gain (Ziegler-Nichols).
    Ki : float
        Recommended integral gain.
    Kd : float
        Recommended derivative gain.
    Ti : float
        Integral time [s].
    Td : float
        Derivative time [s].
    message : str
        Human-readable status message.
    """

    converged: bool
    K_u: float = 0.0
    T_u: float = 0.0
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    Ti: float = 0.0
    Td: float = 0.0
    message: str = ""

    def to_pid_params(self, u_min: float = 0.0, u_max: float = 10000.0) -> PIDParams:
        """Convert tuning result to a PIDParams instance."""
        return PIDParams(
            Kp=self.Kp,
            Ki=self.Ki,
            Kd=self.Kd,
            tau_d=max(self.Td * 0.1, 0.1),
            u_min=u_min,
            u_max=u_max,
        )


def relay_feedback_test(
    E_cell: float = 1.95,
    relay_amplitude: float = 3000.0,
    j_centre: float = 5000.0,
    dt: float = 1.0,
    n_warmup: int = 300,
    n_steps: int = 3000,
    reactor_params: ElectrolyserParams | None = None,
) -> TuningResult:
    """Run a relay feedback test to find the plant's ultimate gain and period.

    The method applies a symmetric relay (bang-bang) controller around a
    steady-state operating point and measures the resulting limit-cycle
    oscillation to extract the ultimate gain (K_u) and ultimate period (T_u).
    Ziegler-Nichols rules are then applied.

    Parameters
    ----------
    E_cell : float
        Fixed cell voltage during the test [V].
    relay_amplitude : float
        Half-amplitude of the relay switching [A/m²].
    j_centre : float
        Centre current density [A/m²].
    dt : float
        Simulation time step [s].
    n_warmup : int
        Number of warm-up steps to reach steady state.
    n_steps : int
        Number of relay test steps (must be long enough for ≥ 3 full cycles).
    reactor_params : ElectrolyserParams, optional
        Custom reactor parameters. Uses defaults if None.

    Returns
    -------
    TuningResult
        Contains K_u, T_u, and Ziegler-Nichols PID gains.
    """
    reactor = Electrolyser(reactor_params or ElectrolyserParams())
    reactor.reset()

    # ── Warm-up to steady state at j_centre ──
    for _ in range(n_warmup):
        reactor.step(j=j_centre, E_cell=E_cell, dt=dt, P_available=np.inf)
    setpoint = reactor._H2_rate

    # ── Relay feedback loop ──
    rates = np.empty(n_steps, dtype=np.float64)
    for i in range(n_steps):
        error = setpoint - reactor._H2_rate
        j = j_centre + relay_amplitude if error > 0 else j_centre - relay_amplitude
        j = float(np.clip(j, 0, 10000))
        reactor.step(j=j, E_cell=E_cell, dt=dt, P_available=np.inf)
        rates[i] = reactor._H2_rate

    # ── Extract oscillation parameters ──
    centred = rates - setpoint
    sign_changes = np.where(np.diff(np.sign(centred)))[0]

    if len(sign_changes) < 6:
        return TuningResult(
            converged=False,
            message=f"Insufficient oscillations: {len(sign_changes)} "
                    f"sign changes (need ≥ 6). Try increasing n_steps or "
                    f"relay_amplitude.",
        )

    # Skip first 2 crossings (transient), use the rest
    crossings = sign_changes[2:]
    half_periods = np.diff(crossings).astype(np.float64) * dt
    T_u = 2.0 * np.median(half_periods)  # Ultimate period [s]

    # Oscillation amplitude (peak-to-peak / 2) from stable region
    stable_start = crossings[0]
    a = (np.max(rates[stable_start:]) - np.min(rates[stable_start:])) / 2.0

    if a < 1e-12:
        return TuningResult(
            converged=False,
            message="Oscillation amplitude effectively zero. Plant may be "
                    "too heavily damped for relay feedback.",
        )

    # ── Ultimate gain ──
    K_u = 4.0 * relay_amplitude / (np.pi * a)

    # ── Ziegler-Nichols PID tuning rules ──
    Kp = 0.6 * K_u
    Ti = 0.5 * T_u
    Td = 0.125 * T_u
    Ki = Kp / Ti if Ti > 0 else 0.0
    Kd = Kp * Td

    return TuningResult(
        converged=True,
        K_u=K_u,
        T_u=T_u,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        Ti=Ti,
        Td=Td,
        message=f"Converged. K_u={K_u:.1f}, T_u={T_u:.1f}s. "
                f"ZN: Kp={Kp:.1f}, Ki={Ki:.2f}, Kd={Kd:.1f}",
    )
