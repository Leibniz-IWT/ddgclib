# Martian Electrolysis Surrogate Control & RL Architecture

This directory contains the control architecture and surrogate models for the Martian alkaline water electrolysis plant (Arcadia Planitia).

## Architecture Overview
Due to the low gravity on Mars ($3.72 \text{ m/s}^2$), buoyancy forces are insufficient for normal bubble detachment, leading to massive electrode bubble blanketing and subsequent thermal runaway.

To solve this, we implemented a dual-loop control architecture:
1. **Inner Loop (Classical):** An auto-tuned Gain-Scheduled PID controller tracks the Hydrogen ($H_2$) demand.
2. **Outer Loop (AI/Surrogate):** A Soft Actor-Critic (SAC) Reinforcement Learning agent dynamically modulates the cell voltage ($E_{cell}$). This exploits the **Lippmann electrocapillarity effect** to artificially lower surface tension, forcing early bubble detachment and preventing temperatures from exceeding the $90^\circ$C safety limit.

## File Structure & Components
* **`_electrolyser.py`**: The core plant physics. Integrates e-NRTL thermodynamics, Butler-Volmer kinetics, and the Lippmann/Fritz bubble detachment scaling equations.
* **`_gym_env.py`**: The OpenAI Gym wrapper for the RL agent. Defines the 15-dimensional observation space (including temporal derivatives) and the reward/penalty function (hard $90^\circ$C limit).
* **`_pid_tuning.py` & `_pid.py`**: Implementation of the Åström-Hägglund relay feedback method and the resulting `GainScheduledPID` controller.
* **`_model_based.py`**: A Cross-Entropy Method (CEM) planner used to validate the mathematical possibility of control (computational baseline).
* **`train_rl.py`**: The main training script for the Stable-Baselines3 agents.
* **`benchmark.py`**: A comprehensive evaluation suite pitting the PID, CEM, and SAC models against each other.

---

## Development Iterations & Failed Attempts
As part of the project methodology, several control strategies were attempted before arriving at the final SAC surrogate model:

1. **Iteration 1: Manual/Fixed PID (Failed)**
   * *Approach:* Standard PID with fixed gains pursuing max $H_2$.
   * *Result:* The plant's non-linear gain caused massive thermal violations (140+ overheating steps per Sol) due to unmanaged bubble blanketing. Addressed by building the Relay-Feedback tuner.
2. **Iteration 2: PPO RL Agent (Failed)**
   * *Approach:* Proximal Policy Optimization (PPO) using raw environment rewards.
   * *Result:* The Critic network collapsed (explained variance = 0) because the specific energy penalties were astronomically high ($\sim -10^5$). Addressed by implementing `VecNormalize` for reward scaling.
3. **Iteration 3: Model-Based CEM Planner**
   * *Approach:* Pure mathematical lookahead using the plant equations.
   * *Result:* Achieved safe control but inference took ~3 minutes per episode. Unsuitable for real-time control due to dead-time.
4. **Iteration 4: Soft Actor-Critic (Final)**
   * *Approach:* Maximum entropy off-policy SAC with `VecNormalize`.
   * *Result:* Solved the environment. Reduced thermal violations by 30%, suppressed bubble coverage to 8.9%, and executes in 1.41 seconds.

---

## How to Run

**1. Train the Agent**
```bash
python -m ddgclib.reactor.train_rl --algo SAC --total-timesteps 200000
```
*Note: Make sure to monitor training using TensorBoard:*
```bash
tensorboard --logdir=rl_output/SAC/tb_logs
```

**2. Run the Benchmark**
After training is complete, the `final_model.zip` and `vec_normalize.pkl` will be saved to `rl_output/SAC/`. Run the benchmark to evaluate it against the baselines:
```bash
python -m ddgclib.reactor.benchmark
```
