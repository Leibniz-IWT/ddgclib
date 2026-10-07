import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add the repo root to sys.path so we can import ddgclib
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ddgclib.reactor._gym_env import ElectrolysisEnv

def main():
    # 1. Load the newly trained model
    model_path = os.path.join(repo_root, "rl_output", "PPO", "final_model.zip")
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # 2. Setup the environment
    env = ElectrolysisEnv()
    obs, info = env.reset()

    # 3. Data logging arrays
    times = []
    h2_rates = []
    demands = []
    e_cells = []
    coverages = []
    solar_powers = []
    sigmas = []

    # 4. Run one full Mars Sol (episode)
    done = False
    while not done:
        # Use deterministic=True for evaluation
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        
        times.append(env.t / 3600.0) # Convert seconds to hours
        h2_rates.append(info['H2_rate'] * 1000) # Convert mol/s to mmol/s
        demands.append(info['demand'] * 1000) 
        e_cells.append(info['E_cell'])
        coverages.append(info['bubble_coverage'])
        solar_powers.append(env._solar_power(env.t))
        sigmas.append(info['sigma_elec'] * 1000) # mN/m
        
        done = term or trunc

    # 5. Create a beautiful multi-subplot figure for the supervisor
    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle('RL Sub-Agent Performance: 1 Mars Sol Evaluation', fontsize=16, fontweight='bold')

    # Plot 1: Environmental Condition (Solar Power)
    axs[0].plot(times, solar_powers, color='orange', linewidth=2, label='Available Solar Power')
    axs[0].fill_between(times, 0, solar_powers, color='orange', alpha=0.2)
    axs[0].set_ylabel('Power (W)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].set_title('Environmental Constraints (Main Agent Input)', loc='left', fontsize=10)

    # Plot 2: Control Objective (H2 Production vs Demand)
    axs[1].plot(times, demands, 'k--', linewidth=2, label='H$_2$ Demand')
    axs[1].plot(times, h2_rates, 'b-', linewidth=2, label='H$_2$ Produced (PID tracked)')
    axs[1].set_ylabel('Rate (mmol/s)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_title('Production Tracking', loc='left', fontsize=10)

    # Plot 3: RL Action Space (Cell Voltage)
    axs[2].plot(times, e_cells, 'g-', linewidth=2, label='E$_{cell}$ Setpoint (RL Action)')
    axs[2].set_ylabel('Voltage (V)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].set_title('Primary Electrostatic Control Action', loc='left', fontsize=10)

    # Plot 4: Lippmann Physics (Interfacial Tension)
    axs[3].plot(times, sigmas, 'm-', linewidth=2, label='$\\sigma_{elec}$ (Lippmann Effect)')
    axs[3].set_ylabel('Tension (mN/m)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    axs[3].set_title('Electrocapillarity Physics', loc='left', fontsize=10)

    # Plot 5: Physical Constraint (Bubble Coverage)
    axs[4].plot(times, coverages, 'r-', linewidth=2, label='Electrode Bubble Coverage')
    axs[4].fill_between(times, 0, coverages, color='red', alpha=0.1)
    axs[4].set_ylabel('Coverage Fraction')
    axs[4].set_xlabel('Time of Sol (Hours)')
    axs[4].legend(loc='upper right')
    axs[4].grid(True, linestyle='--', alpha=0.7)
    axs[4].set_ylim(0, 1.0)
    axs[4].set_title('Electrode Health Constraint', loc='left', fontsize=10)

    plt.tight_layout()
    
    # Save directly to your artifacts directory so you can view it
    output_path = r"C:\Users\Varun\.gemini\antigravity\brain\d288a7e4-1d0b-4b73-b7e7-8ef5c164e741\rl_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    main()
