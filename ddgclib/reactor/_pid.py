"""
Discrete PID controller with anti-windup for the electrolysis inner loop.

The RL agent sets: H₂ production-rate setpoint + E_cell.
The PID tracks: H₂ rate by adjusting current density *j*.

Features
--------
* Anti-windup via back-calculation (prevents integral saturation).
* Derivative low-pass filter (suppresses measurement noise).
* Output clamping to [u_min, u_max].
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PIDParams:
    """PID tuning parameters.

    Defaults are sized for tracking an H₂ production rate of ~0.005 mol/s
    by adjusting current density *j* in [0, 10 000] A/m².
    """

    Kp: float = 5000.0        # Proportional gain  [A/m²  per  mol/s error]
    Ki: float = 500.0          # Integral gain
    Kd: float = 50.0           # Derivative gain
    tau_d: float = 0.5         # Derivative filter time constant [s]
    u_min: float = 0.0         # Minimum current density [A/m²]
    u_max: float = 10000.0     # Maximum current density [A/m²]


class PIDController:
    """Discrete PID with anti-windup and derivative filtering.

    Usage
    -----
    >>> pid = PIDController(PIDParams(Kp=5000))
    >>> pid.setpoint = 0.005          # mol/s H₂
    >>> j = pid.update(current_rate, dt=10.0)
    """

    def __init__(self, params: PIDParams | None = None):
        self.p = params or PIDParams()
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.prev_derivative: float = 0.0
        self.setpoint: float = 0.0

    def reset(self, setpoint: float = 0.0) -> None:
        """Reset internal state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.setpoint = setpoint

    def update(self, measurement: float, dt: float) -> float:
        """Compute PID output (current density *j*) for one time step.

        Parameters
        ----------
        measurement : float
            Current H₂ production rate [mol/s].
        dt : float
            Time step [s].  Must be > 0.

        Returns
        -------
        float
            Current density setpoint *j* [A/m²], clamped to
            [u_min, u_max].
        """
        error = self.setpoint - measurement

        # -- Proportional --
        P = self.p.Kp * error

        # -- Integral --
        self.integral += error * dt
        I = self.p.Ki * self.integral

        # -- Derivative with first-order low-pass filter --
        if dt > 0:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.p.tau_d + dt)
            filt_deriv = alpha * raw_deriv + (1 - alpha) * self.prev_derivative
        else:
            filt_deriv = 0.0
        D = self.p.Kd * filt_deriv

        # -- Total and clamp --
        u = P + I + D
        u_clamped = float(np.clip(u, self.p.u_min, self.p.u_max))

        # -- Anti-windup back-calculation --
        if abs(self.p.Ki) > 1e-12 and u != u_clamped:
            self.integral -= (u - u_clamped) / self.p.Ki * 0.5

        self.prev_error = error
        self.prev_derivative = filt_deriv
        return u_clamped

class GainScheduledPID(PIDController):
    """PID controller with gain scheduling based on the operating point.
    
    Dynamically switches PID parameters depending on the last output 
    current density.
    """

    def __init__(self, schedules: list[tuple[float, PIDParams]] | None = None):
        """
        Parameters
        ----------
        schedules : list of tuple
            List of (j_threshold, PIDParams) sorted in ascending order of threshold.
        """
        # Default schedules if none provided
        self.schedules = schedules or [
            (1000.0, PIDParams(Kp=8000, Ki=800, Kd=80)),   # Low j: high gain
            (5000.0, PIDParams(Kp=5000, Ki=500, Kd=50)),   # Mid j: nominal
            (8000.0, PIDParams(Kp=3000, Ki=300, Kd=30)),   # High j: reduced
        ]
        # Start with mid-range or lowest available
        default_params = self.schedules[1][1] if len(self.schedules) > 1 else self.schedules[0][1]
        super().__init__(default_params)
        self._last_output = 0.0

    def update(self, measurement: float, dt: float) -> float:
        # Select gains based on last output
        for j_thresh, params in reversed(self.schedules):
            if self._last_output >= j_thresh:
                self.p = params
                break
        else:
            self.p = self.schedules[0][1]

        self._last_output = super().update(measurement, dt)
        return self._last_output
