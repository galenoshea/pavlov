#!/usr/bin/env python3
"""Demo script for the TUILogger with CartPole environment.

Run with: python examples/tui_demo.py

Keyboard controls:
  q - Quit
  p - Pause/resume display
  m - Toggle minimal/dashboard mode
"""

import time
import numpy as np
from operant.envs import CartPoleVecEnv
from operant import TUILogger


def main():
    # Create environment and TUI logger
    num_envs = 4096
    env = CartPoleVecEnv(num_envs=num_envs)

    # Start with dashboard mode to show all features
    tui = TUILogger(
        mode="dashboard",
        render_interval_ms=100,
        history_len=100,
        csv_path="training_log.csv",  # Also logs to CSV
    )

    obs, _ = env.reset(seed=42)
    start_time = time.time()
    total_steps = 0

    try:
        while not tui.should_quit():
            # Random policy for demo
            actions = np.random.randint(0, 2, size=num_envs, dtype=np.int32)
            obs, rewards, terms, truncs, _ = env.step(actions)
            total_steps += num_envs

            # Calculate metrics
            elapsed = time.time() - start_time
            sps = total_steps / elapsed if elapsed > 0 else 0

            logs = env.get_logs()

            # Update TUI (minimal overhead - atomic writes only)
            tui.update(
                steps=total_steps,
                episodes=int(logs["episode_count"]),
                mean_reward=logs["mean_reward"],
                sps=sps,
                # Optional training metrics (pass None or omit if not training)
                policy_loss=np.random.uniform(0.01, 0.1),  # Demo values
                value_loss=np.random.uniform(0.1, 1.0),
                entropy=np.random.uniform(0.5, 0.8),
            )

            # Small sleep to not overwhelm the system (in real training this isn't needed)
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        tui.close()
        print(f"\nTraining complete! Total steps: {total_steps:,}")
        print(f"Check training_log.csv for logged data.")


if __name__ == "__main__":
    main()
