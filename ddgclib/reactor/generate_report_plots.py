import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append(r"f:\Varun - MSc Space Engineering\Masters Project\Electrolysis Simulation in Mars\Code\env_ddg")
from ddgclib.reactor._gym_env import ElectrolysisEnv

# Setup Paths
rl_dir = r"f:\Varun - MSc Space Engineering\Masters Project\Electrolysis Simulation in Mars\Code\env_ddg\rl_output"
model_path = os.path.join(rl_dir, "SAC", "final_model.zip")
vec_norm_path = os.path.join(rl_dir, "SAC", "vec_normalize.pkl")
out_img = r"C:\Users\Varun\.gemini\antigravity\brain\d288a7e4-1d0b-4b73-b7e7-8ef5c164e741\sac_episode_results.png"

print("1. Imports done")

# Setup Environment & Model
print("2. Setting up environment")
env = DummyVecEnv([lambda: ElectrolysisEnv()])
env = VecNormalize.load(vec_norm_path, env)
env.training = False
env.norm_reward = False
print("3. Loading model")
model = SAC.load(model_path, env=env)
print("4. Model loaded")

# Run Rollout
obs = env.reset()
dones = [False]

times = []
e_cells = []
h2_rates = []
t_cells = []
coverages = []

raw_env = env.envs[0]
demand = raw_env.demand_H2
t_safe = raw_env.T_safe - 273.15

while not dones[0]:
    action, _ = model.predict(obs, deterministic=True)
    
    # Get physical state before step for plotting
    s = raw_env.reactor.get_state_dict()
    
    times.append(raw_env.t / 3600)  # Convert seconds to hours
    
    # Decode E_cell action to actual Volts
    e_max, e_min = 0.40, -0.07
    a = np.clip(action[0][0], -1.0, 1.0)
    real_e_cell = e_max - (e_max - e_min) * (1.0 - a) / 2.0
    
    e_cells.append(real_e_cell)
    h2_rates.append(s["H2_rate"])
    t_cells.append(s["T_cell"] - 273.15)  # Kelvin to Celsius
    coverages.append(s["bubble_coverage"] * 100)  # Fraction to %
    
    obs, rewards, dones, infos = env.step(action)

# --- Plotting ---
plt.style.use('ggplot')
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 1. Voltage Action
axs[0].plot(times, e_cells, color='#9b59b6', lw=2)
axs[0].set_ylabel("Applied $E_{cell}$ [V]")
axs[0].set_title("SAC Agent Action: Lippmann Voltage Modulation", fontweight='bold')
axs[0].axhline(-0.07, color='gray', linestyle='--', alpha=0.5, label='E_pzc')
axs[0].legend()

# 2. H2 Production
axs[1].plot(times, h2_rates, color='#2ecc71', lw=2, label='Actual Rate')
axs[1].axhline(demand, color='#e74c3c', linestyle='--', label='Demand Setpoint')
axs[1].set_ylabel("H$_2$ Rate [mol/s]")
axs[1].set_title("Hydrogen Production (PID Tracking)", fontweight='bold')
axs[1].legend(loc='lower right')

# 3. Temperature
axs[2].plot(times, t_cells, color='#e74c3c', lw=2, label='Reactor Temp')
axs[2].axhline(t_safe, color='#2c3e50', linestyle='--', lw=2, label='Safe Limit (90°C)')
axs[2].set_ylabel("Cell Temp [°C]")
axs[2].set_title("Thermal Safety Management", fontweight='bold')
axs[2].legend(loc='lower right')

# 4. Bubble Coverage
axs[3].plot(times, coverages, color='#3498db', lw=2)
axs[3].set_ylabel("Bubble Coverage [%]")
axs[3].set_title("Electrode Bubble Blanketing", fontweight='bold')
axs[3].set_xlabel("Mars Sol Time [Hours]", fontweight='bold')

plt.tight_layout()
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print("Plot successfully saved to:", out_img)
