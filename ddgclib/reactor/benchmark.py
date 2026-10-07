"""
Benchmark suite comparing Model-Free RL (SAC/PPO), Model-Based planning, 
and PID baselines for the Mars Electrolysis reactor.

Usage
-----
::

    python -m ddgclib.reactor.benchmark

"""

import os
import numpy as np
import time

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ddgclib.reactor._gym_env import ElectrolysisEnv
from ddgclib.reactor._model_based import ModelBasedPlanner, PlannerConfig


def evaluate_agent(name: str, agent_fn, env: ElectrolysisEnv, n_episodes: int = 3):
    """Run an agent for N episodes and collect metrics."""
    results = []
    
    print(f"--- Evaluating {name} ---")
    start_time = time.time()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        ep_reward = 0.0
        h2_total = 0.0
        energy_total = 0.0
        violations = 0
        coverages = []
        
        while not done:
            action = agent_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            
            coverages.append(info['bubble_coverage'])
            if info['T_cell'] > env.T_safe:
                violations += 1
                
            done = term or trunc
            
        h2_total = info['episode_H2_mol']
        energy_total = info['specific_energy_kWh_kg']
        
        results.append({
            'reward': ep_reward,
            'h2_total': h2_total,
            'spec_energy': energy_total,
            'violations': violations,
            'mean_coverage': np.mean(coverages)
        })
        
    elapsed = time.time() - start_time
    
    # Aggregate
    avg_reward = np.mean([r['reward'] for r in results])
    avg_h2 = np.mean([r['h2_total'] for r in results])
    avg_energy = np.mean([r['spec_energy'] for r in results])
    avg_viol = np.mean([r['violations'] for r in results])
    avg_cov = np.mean([r['mean_coverage'] for r in results])
    
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Total H2: {avg_h2:.2f} mol")
    print(f"Specific Energy: {avg_energy:.2f} kWh/kg")
    print(f"Thermal Violations: {avg_viol:.1f} steps")
    print(f"Mean Coverage: {avg_cov*100:.1f}%")
    print()
    return results


def main():
    env = ElectrolysisEnv()
    n_episodes = 2
    
    # 1. Baseline: Fixed actions (equivalent to pure PID tracking a constant setpoint)
    def pid_baseline_agent(obs):
        # E_cell = 0 (maps to 1.95V), H2_setpoint = 0.5 (maps to 75% max)
        return np.array([0.0, 0.5], dtype=np.float32)
        
    evaluate_agent("PID Baseline (Fixed Setpoint)", pid_baseline_agent, env, n_episodes)
    
    # 2. Model-Based Planner (Random Shooting CEM)
    planner = ModelBasedPlanner(env, PlannerConfig(n_candidates=32, horizon=4))
    
    def mb_agent(obs):
        return planner.plan(obs)
        
    # evaluate_agent("Model-Based Planner (CEM)", mb_agent, env, 1) # Only 1 episode, planning is slow
    
    # 3. Model-Free RL (if models exist)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rl_dir = os.path.join(repo_root, "rl_output")
    
    for algo in ["PPO", "SAC"]:
        model_path = os.path.join(rl_dir, algo, "final_model.zip")
        vec_norm_path = os.path.join(rl_dir, algo, "vec_normalize.pkl")
        
        if os.path.exists(model_path) and os.path.exists(vec_norm_path):
            try:
                # Load normalized env
                eval_env = DummyVecEnv([lambda: ElectrolysisEnv()])
                eval_env = VecNormalize.load(vec_norm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
                
                model_cls = PPO if algo == "PPO" else SAC
                model = model_cls.load(model_path, env=eval_env)
                
                def rl_agent(obs):
                    # We must normalize the obs first for the model
                    norm_obs = eval_env.normalize_obs(obs)
                    action, _ = model.predict(norm_obs, deterministic=True)
                    return action
                    
                evaluate_agent(f"Model-Free {algo}", rl_agent, env, n_episodes)
            except Exception as e:
                print(f"Failed to load {algo} model: {e}")
        else:
            print(f"--- Model-Free {algo} ---")
            print(f"Model not found at {model_path}. Train it first to include in benchmark.\n")


if __name__ == "__main__":
    main()
