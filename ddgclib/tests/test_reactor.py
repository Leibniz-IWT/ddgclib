"""
Unit and integration tests for the ``ddgclib.reactor`` sub-package.

Numerical checkpoints are taken from the Lippmann modelling document
(§2.3 Table and §3.4 Mars applicability).

Run with::

    pytest ddgclib/tests/test_reactor.py -v
"""

import pytest
import numpy as np


# ====================================================================== #
#  Lippmann electrocapillarity                                             #
# ====================================================================== #
class TestLippmann:
    """Validate Lippmann model against modelling document tables."""

    def test_zero_voltage_returns_baseline(self):
        """At E_cell = E_pzc, σ_elec = σ₀ (no reduction)."""
        from ddgclib.reactor._lippmann import sigma_lippmann, LippmannParams

        p = LippmannParams(E_pzc=-0.07)
        result = sigma_lippmann(0.07197, -0.07, p)
        assert result == pytest.approx(0.07197, rel=1e-6)

    def test_reduction_at_010V(self):
        """ΔV = 0.10 V → Δσ/σ₀ ≈ −2.1 % (doc Table row 2)."""
        from ddgclib.reactor._lippmann import sigma_lippmann, LippmannParams

        p = LippmannParams(C_dl=0.30, E_pzc=-0.07)
        sigma = sigma_lippmann(0.07197, -0.07 + 0.10, p)
        reduction = (0.07197 - sigma) / 0.07197
        assert reduction == pytest.approx(0.021, abs=0.005)

    def test_reduction_at_030V(self):
        """ΔV = 0.30 V → Δσ/σ₀ ≈ −18.8 % (doc Table row 4)."""
        from ddgclib.reactor._lippmann import sigma_lippmann, LippmannParams

        p = LippmannParams(C_dl=0.30, E_pzc=-0.07)
        sigma = sigma_lippmann(0.07197, -0.07 + 0.30, p)
        reduction = (0.07197 - sigma) / 0.07197
        assert reduction == pytest.approx(0.188, abs=0.01)

    def test_reduction_at_050V(self):
        """ΔV = 0.50 V → Δσ/σ₀ ≈ −52.1 % (doc Table row 6)."""
        from ddgclib.reactor._lippmann import sigma_lippmann, LippmannParams

        p = LippmannParams(C_dl=0.30, E_pzc=-0.07)
        sigma = sigma_lippmann(0.07197, -0.07 + 0.50, p)
        reduction = (0.07197 - sigma) / 0.07197
        assert reduction == pytest.approx(0.521, abs=0.01)

    def test_clamp_at_saturation(self):
        """σ_elec must never go negative (electrowetting saturation)."""
        from ddgclib.reactor._lippmann import sigma_lippmann

        sigma = sigma_lippmann(0.07197, 5.0)
        assert sigma >= 0.0

    def test_scaling_ratios_030V(self):
        """Volume and frequency scaling at ΔV = 0.30 V (doc §2.3)."""
        from ddgclib.reactor._lippmann import lippmann_detachment_scaling

        # σ_elec = 71.97 - 0.5 * 0.30 * 0.30² = 71.97 - 13.50 = 58.47 mN/m
        res = lippmann_detachment_scaling(0.07197, 0.05847)
        # V_d ratio = (58.47/71.97)^1.5 ≈ 0.732
        assert res["ratio_V"] == pytest.approx(0.732, abs=0.02)
        # f_d change ≈ +36.6 %
        assert res["delta_f_rel"] == pytest.approx(0.366, abs=0.03)

    def test_scaling_ratios_040V(self):
        """Volume scaling at ΔV = 0.40 V (doc §2.3)."""
        from ddgclib.reactor._lippmann import lippmann_detachment_scaling

        # σ_elec = 71.97 - 0.5 * 0.30 * 0.40² = 71.97 - 24.00 = 47.97 mN/m
        res = lippmann_detachment_scaling(0.07197, 0.04797)
        # V_d ratio = (47.97/71.97)^1.5 ≈ 0.544
        assert res["ratio_V"] == pytest.approx(0.544, abs=0.02)
        # f_d change ≈ +83.8 %
        assert res["delta_f_rel"] == pytest.approx(0.838, abs=0.05)


# ====================================================================== #
#  Electrolyser plant model                                                #
# ====================================================================== #
class TestElectrolyser:
    """Tests for the reduced-order electrolyser model."""

    def test_init_defaults(self):
        from ddgclib.reactor import Electrolyser

        r = Electrolyser()
        state = r.reset()
        assert state[0] == pytest.approx(353.15, rel=1e-3)  # T_cell
        assert state[2] == 0.0  # H2_rate

    def test_faraday_law(self):
        """At known j, H₂ rate should match Faraday's law within 15 %."""
        from ddgclib.reactor import Electrolyser, ElectrolyserParams

        p = ElectrolyserParams(A_cell=0.01, n_cells=1)
        r = Electrolyser(p)
        r.reset()
        state = r.step(j=1000.0, E_cell=1.8, dt=1.0, P_available=np.inf)
        # η_f * j * A / (2F) ≈ 0.90 * 1000 * 0.01 / (2 * 96485) ≈ 4.66e-5
        expected = 0.90 * 1000.0 * 0.01 / (2 * 96485.3329)
        assert state["H2_rate"] == pytest.approx(expected, rel=0.15)

    def test_power_limiting(self):
        """Limited power should reduce production rate."""
        from ddgclib.reactor import Electrolyser

        r1 = Electrolyser()
        r1.reset()
        s_unlimited = r1.step(j=5000.0, E_cell=2.0, dt=1.0, P_available=np.inf)

        r2 = Electrolyser()
        r2.reset()
        s_limited = r2.step(j=5000.0, E_cell=2.0, dt=1.0, P_available=10.0)

        assert s_limited["H2_rate"] < s_unlimited["H2_rate"]

    def test_lippmann_reduces_coverage(self):
        """Higher E_cell (stronger Lippmann) should reduce bubble coverage."""
        from ddgclib.reactor import Electrolyser, ElectrolyserParams

        # Low voltage → minimal Lippmann effect
        r1 = Electrolyser(ElectrolyserParams())
        r1.reset()
        for _ in range(50):
            r1.step(j=3000.0, E_cell=-0.07, dt=10.0, P_available=np.inf)
        cov_low_V = r1.bubble_coverage

        # Higher voltage → stronger Lippmann → smaller bubbles → less coverage
        r2 = Electrolyser(ElectrolyserParams())
        r2.reset()
        for _ in range(50):
            r2.step(j=3000.0, E_cell=0.40, dt=10.0, P_available=np.inf)
        cov_high_V = r2.bubble_coverage

        assert cov_high_V < cov_low_V, (
            f"Lippmann should reduce coverage: {cov_high_V:.4f} >= {cov_low_V:.4f}"
        )

    def test_mars_gravity_default(self):
        """Default gravity should be Mars (3.721 m/s²)."""
        from ddgclib.reactor import ElectrolyserParams

        assert ElectrolyserParams().g == pytest.approx(3.721, rel=1e-3)

    def test_thermal_dynamics(self):
        """Cell should heat up under sustained load."""
        from ddgclib.reactor import Electrolyser

        r = Electrolyser()
        r.reset()
        T0 = r.T_cell
        for _ in range(200):
            r.step(j=5000.0, E_cell=2.0, dt=1.0, P_available=np.inf)
        assert r.T_cell > T0

    def test_cell_voltage_monotone(self):
        """V_cell should increase monotonically with j."""
        from ddgclib.reactor import Electrolyser

        r = Electrolyser()
        r.reset()
        js = [100, 500, 1000, 3000, 5000, 8000]
        Vs = [r.cell_voltage(j) for j in js]
        for i in range(len(Vs) - 1):
            assert Vs[i + 1] > Vs[i], f"V({js[i+1]}) <= V({js[i]})"


# ====================================================================== #
#  PID controller                                                          #
# ====================================================================== #
class TestPID:
    """Tests for the PID controller."""

    def test_tracks_setpoint(self):
        """PID should converge to a setpoint on a simple plant."""
        from ddgclib.reactor._pid import PIDController, PIDParams

        pid = PIDController(PIDParams(Kp=5000.0, Ki=500.0, u_max=10000))
        pid.setpoint = 0.005
        meas = 0.0
        for _ in range(500):
            u = pid.update(meas, dt=1.0)
            # Simplified first-order plant: rate ∝ current density
            meas += (u * 1e-5 - meas) * 0.5
        assert abs(meas - 0.005) / 0.005 < 0.20  # within 20 %

    def test_anti_windup(self):
        """Output should not exceed u_max even with large error."""
        from ddgclib.reactor._pid import PIDController, PIDParams

        pid = PIDController(PIDParams(Kp=100.0, Ki=50.0, u_min=0, u_max=10))
        pid.setpoint = 1000.0
        u = pid.update(0.0, dt=1.0)
        assert u <= 10.0

    def test_reset_clears_state(self):
        """After reset, integral and errors should be zero."""
        from ddgclib.reactor._pid import PIDController, PIDParams

        pid = PIDController(PIDParams(Ki=100.0))
        pid.setpoint = 1.0
        pid.update(0.0, dt=1.0)  # accumulates integral
        pid.reset(setpoint=0.0)
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0


# ====================================================================== #
#  Gymnasium environment                                                   #
# ====================================================================== #
class TestGymEnv:
    """Integration tests for the Gymnasium environment."""

    def test_reset_and_step(self):
        """Basic reset → step should not raise."""
        from ddgclib.reactor._gym_env import ElectrolysisEnv

        env = ElectrolysisEnv()
        obs, info = env.reset()
        assert obs.shape == (11,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert obs2.shape == (11,)
        assert isinstance(reward, float)
        assert "E_cell" in info
        assert "sigma_elec" in info

    def test_main_agent_interface(self):
        """Main MARL agent can override environmental conditions."""
        from ddgclib.reactor._gym_env import ElectrolysisEnv

        env = ElectrolysisEnv()
        env.reset()
        env.set_conditions(T_ambient=195.0, P_solar=3000.0, demand_H2=0.008)

        assert env.reactor.p.T_ambient == 195.0
        assert env._P_solar_override == 3000.0
        assert env.demand_H2 == 0.008

    def test_observation_bounds(self):
        """Observations should always be in [0, 1] after steps."""
        from ddgclib.reactor._gym_env import ElectrolysisEnv

        env = ElectrolysisEnv()
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            assert np.all(obs >= 0.0), f"Obs below 0: {obs}"
            assert np.all(obs <= 1.0), f"Obs above 1: {obs}"
            if term or trunc:
                break

    def test_full_episode(self):
        """Run a complete episode (fast, with large dt_rl) without crash."""
        from ddgclib.reactor._gym_env import ElectrolysisEnv

        env = ElectrolysisEnv({"dt_rl": 5000.0})
        obs, _ = env.reset()
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            steps += 1
            if term or trunc:
                break
        assert steps > 0
        assert info["episode_H2_mol"] >= 0

    def test_action_space_valid(self):
        """Action space should be 2-dim, bounded [-1, 1]."""
        from ddgclib.reactor._gym_env import ElectrolysisEnv

        env = ElectrolysisEnv()
        assert env.action_space.shape == (2,)
        assert float(env.action_space.low[0]) == -1.0
        assert float(env.action_space.high[0]) == 1.0
