"""
RL training script for the electrolysis reactor sub-agent.

Usage
-----
::

    # From the repository root, with the ddg conda env active:
    python -m ddgclib.reactor.train_rl --algo SAC --total-timesteps 500000

Requires
--------
``pip install stable-baselines3 gymnasium tensorboard``
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from ddgclib.reactor._gym_env import ElectrolysisEnv


def make_env(config: dict | None = None, seed: int = 0):
    """Factory that returns a zero-argument callable for *DummyVecEnv*."""

    def _init():
        env = ElectrolysisEnv(config=config)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def train(args: argparse.Namespace) -> None:
    """Run the training loop."""

    config: dict = {"dt_rl": args.dt_rl}
    out_dir = os.path.join(args.output_dir, args.algo)
    os.makedirs(out_dir, exist_ok=True)

    # Vectorised environments
    raw_env = DummyVecEnv(
        [make_env(config, seed=i) for i in range(args.n_envs)]
    )
    # Apply VecNormalize for reward and observation scaling (Crucial improvement)
    env = VecNormalize(
        raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.995
    )

    raw_eval_env = DummyVecEnv([make_env(config, seed=99)])
    eval_env = VecNormalize(
        raw_eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )
    # Sync eval_env with training env normalization stats
    eval_env.obs_rms = env.obs_rms

    # Algorithm setup
    if args.algo == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            tensorboard_log=os.path.join(out_dir, "tb_logs"),
            device="auto",
            verbose=1,
        )
    else:
        # PPO tuned hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lambda progress: args.lr * (1 - 0.9 * progress),
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.Tanh,
            ),
            tensorboard_log=os.path.join(out_dir, "tb_logs"),
            device="auto",
            verbose=1,
        )

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best_model"),
        eval_freq=max(1000, 5000 // args.n_envs),
        n_eval_episodes=3,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1000, 25000 // args.n_envs),
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix="electrolysis",
    )

    # Train
    print(f"Starting {args.algo} training for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_cb, ckpt_cb],
    )

    final_path = os.path.join(out_dir, "final_model")
    model.save(final_path)
    env.save(os.path.join(out_dir, "vec_normalize.pkl"))
    print(f"Training complete. Final model saved to: {final_path}")


# ------------------------------------------------------------------ #
#  CLI entry-point                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train the electrolysis RL sub-agent"
    )
    p.add_argument(
        "--algo",
        choices=["PPO", "SAC"],
        default="SAC",  # Defaulting to SAC now as per improvement plan
        help="RL algorithm (default: SAC)",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500 000)",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--dt-rl",
        type=float,
        default=600.0,
        help="RL decision interval in seconds (default: 600)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./rl_output",
        help="Output directory for models and logs",
    )
    train(p.parse_args())
