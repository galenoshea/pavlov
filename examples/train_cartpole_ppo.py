"""Complete CartPole PPO training example with optimizations.

Demonstrates:
- Full training to convergence (~195 avg reward)
- Observation and reward normalization
- Mixed precision training (AMP)
- Logging with operant.Logger
- Model checkpointing (optional)
- Evaluation

Usage:
    poetry run python examples/train_cartpole_ppo.py
"""

import os
from pathlib import Path

import numpy as np
import torch

# Import from operant package
try:
    from operant.envs import CartPoleVecEnv
    from operant.models import PPO
except ImportError:
    # Fallback for development
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    from operant.envs import CartPoleVecEnv
    from operant.models import PPO


def main():
    """Train PPO on CartPole until solved."""
    # Configuration - Use MANY environments for throughput
    num_envs = 131072  # Maximum that fits in GPU memory (tested: 131K works!)
    total_timesteps = 50_000_000  # Need many updates with large batch sizes
    checkpoint_interval = None  # Set to number of timesteps for periodic checkpoints, None to disable

    # Create environment
    print(f"Creating {num_envs} CartPole environments...")
    env = CartPoleVecEnv(num_envs=num_envs)

    # Create PPO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing PPO:")
    print(f"  - Device: {device}")
    print(f"  - Num envs: {num_envs}")
    print(f"  - Mixed precision (AMP): {'ON' if device == 'cuda' else 'OFF'}")
    print(f"  - Checkpointing: {'Disabled' if checkpoint_interval is None else f'Every {checkpoint_interval:,} timesteps'}\n")

    model = PPO(
        env,
        lr=2.5e-4,
        n_steps=128,
        batch_size=16384,  # Scaled up for 131K environments (16K batch = 1024 mini-batches)
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        use_amp=(device == "cuda"),
    )

    # Setup checkpoint directory if checkpointing is enabled
    checkpoint_dir = None
    if checkpoint_interval is not None:
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir.absolute()}\n")

    # Training metrics
    best_reward = -float("inf")
    solved = False

    # Training callback for logging and optional checkpointing
    def training_callback(metrics):
        nonlocal best_reward, solved

        # Print progress
        timesteps = metrics["timesteps"]
        mean_reward = metrics.get("mean_reward", 0)
        mean_length = metrics.get("mean_length", 0)
        sps = metrics.get("sps", 0)
        episodes = metrics.get("episodes", 0)

        print(
            f"Timesteps: {timesteps:>7} | "
            f"Episodes: {episodes:>5} | "
            f"Reward: {mean_reward:>7.2f} | "
            f"Length: {mean_length:>6.1f} | "
            f"SPS: {sps:>8.0f} | "
            f"PG Loss: {metrics.get('policy_loss', 0):>7.4f} | "
            f"V Loss: {metrics.get('value_loss', 0):>6.4f}"
        )

        # Checkpointing (if enabled)
        if checkpoint_dir is not None:
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                model.save(checkpoint_dir / "ppo_cartpole_best.pt")

            # Periodic checkpoint
            if checkpoint_interval and timesteps % checkpoint_interval == 0:
                model.save(checkpoint_dir / f"ppo_cartpole_{timesteps}.pt")

        # Early stopping if solved (195+ avg reward)
        if mean_reward >= 195 and not solved:
            print(f"\n{'='*80}")
            print(f"✓ SOLVED at {timesteps} timesteps with {mean_reward:.2f} avg reward!")
            print(f"{'='*80}\n")
            solved = True
            return False  # Stop training

        return True

    # Train
    print(f"Training PPO on CartPole...")
    print(f"Target: 195+ avg reward over 100 episodes\n")
    print(f"{'='*80}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=training_callback,
        log_interval=1,
    )

    # Save final model if checkpointing is enabled
    if checkpoint_dir is not None:
        final_path = checkpoint_dir / "ppo_cartpole_final.pt"
        model.save(str(final_path))
        print(f"\nFinal model saved to: {final_path}")

    # Evaluation
    print("\n" + "="*80)
    print("Evaluating trained policy...")
    print("="*80)
    evaluate_policy(model, env, n_episodes=100)


def evaluate_policy(model, env, n_episodes=100):
    """Evaluate policy for n episodes."""
    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    episodes_completed = 0
    steps = 0
    max_steps = n_episodes * 500  # Safety limit

    while episodes_completed < n_episodes and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        steps += 1

        # Check for completed episodes
        logs = env.get_logs()
        if logs["episode_count"] > 0:
            episode_rewards.append(logs["mean_reward"])
            episode_lengths.append(logs["mean_length"])
            episodes_completed += int(logs["episode_count"])
            env.clear_logs()

    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)

        print(f"\nEvaluation Results (n={len(episode_rewards)} episodes):")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Mean Length: {mean_length:.2f} ± {std_length:.2f}")
        print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
        print(f"  Max Reward:  {np.max(episode_rewards):.2f}")

        # Check if solved
        if mean_reward >= 195:
            print(f"\n  ✓ Policy is SOLVED (≥195 reward)")
        else:
            print(f"\n  ✗ Policy not yet solved (need ≥195 reward)")
    else:
        print("No episodes completed during evaluation.")


if __name__ == "__main__":
    main()
