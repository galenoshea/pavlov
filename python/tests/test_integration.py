"""Integration tests for Pavlov Python bindings.

Tests environment correctness, API compliance, and performance characteristics.
"""

import numpy as np
import pytest

from pavlov.envs import CartPoleVecEnv, MountainCarVecEnv, PendulumVecEnv
from pavlov.utils import Logger


# ============================================================
# CartPoleVecEnv Tests
# ============================================================


class TestCartPole:
    """Tests for CartPoleVecEnv environment."""

    def test_initialization(self):
        """Test environment can be created with various sizes."""
        for num_envs in [1, 8, 64, 4096]:
            env = CartPoleVecEnv(num_envs)
            assert env.num_envs == num_envs
            assert env.observation_size == 4

    def test_reset_shape(self):
        """Test reset returns correct observation shape."""
        env = CartPoleVecEnv(128)
        obs = env.reset(seed=42)
        assert obs.shape == (128, 4)
        assert obs.dtype == np.float32

    def test_reset_deterministic(self):
        """Test reset with same seed produces same initial states."""
        env1 = CartPoleVecEnv(64)
        env2 = CartPoleVecEnv(64)

        obs1 = env1.reset(seed=12345)
        obs2 = env2.reset(seed=12345)

        np.testing.assert_array_equal(obs1, obs2)

    def test_step_shapes(self):
        """Test step returns correct shapes for all outputs."""
        env = CartPoleVecEnv(256)
        env.reset(seed=0)

        actions = np.zeros(256, dtype=np.int32)
        obs, rewards, terminals, truncations = env.step(actions)

        assert obs.shape == (256, 4)
        assert obs.dtype == np.float32
        assert rewards.shape == (256,)
        assert rewards.dtype == np.float32
        assert terminals.shape == (256,)
        assert terminals.dtype == np.uint8
        assert truncations.shape == (256,)
        assert truncations.dtype == np.uint8

    def test_action_effects(self):
        """Test that different actions produce different results."""
        env = CartPoleVecEnv(2)
        obs = env.reset(seed=42)

        # Apply opposite actions to two environments
        actions = np.array([0, 1], dtype=np.int32)
        obs_next, _, _, _ = env.step(actions)

        # Observations should differ between the two environments
        assert not np.allclose(obs_next[0], obs_next[1])

    def test_terminal_conditions(self):
        """Test that terminal states are properly detected."""
        env = CartPoleVecEnv(32)
        env.reset(seed=0)

        # Run until we see at least one terminal state in the batch
        # With auto-reset, we track via episode logging
        max_iterations = 100
        found_terminal = False

        for _ in range(max_iterations):
            actions = np.random.randint(0, 2, 32, dtype=np.int32)
            env.step(actions)

            # Check if any episodes completed (which means terminal was hit)
            logs = env.get_logs()
            if logs["episode_count"] > 0:
                found_terminal = True
                break

        assert found_terminal, "Should complete at least one episode within 100 iterations"

    def test_truncation_conditions(self):
        """Test that episodes can complete via truncation."""
        env = CartPoleVecEnv(32)
        env.reset(seed=42)

        # Run enough steps to ensure some episodes complete
        # With auto-reset, we verify via episode logging
        for _ in range(200):
            actions = np.zeros(32, dtype=np.int32)  # Stable actions
            env.step(actions)

        # Should have completed some episodes (either terminal or truncation)
        logs = env.get_logs()
        assert logs["episode_count"] > 0, "Should complete at least one episode"
        assert logs["mean_length"] > 0, "Completed episodes should have positive length"

    def test_logging_functionality(self):
        """Test that episode logging tracks statistics correctly."""
        env = CartPoleVecEnv(32)
        env.reset(seed=0)

        # Run some steps
        for _ in range(100):
            actions = np.random.randint(0, 2, 32, dtype=np.int32)
            env.step(actions)

        logs = env.get_logs()
        assert "episode_count" in logs
        assert "total_reward" in logs
        assert "mean_reward" in logs
        assert "total_steps" in logs
        assert "mean_length" in logs

        # Should have completed some episodes
        assert logs["episode_count"] > 0

        # Clear and verify reset
        env.clear_logs()
        logs = env.get_logs()
        assert logs["episode_count"] == 0
        assert logs["total_reward"] == 0.0


# ============================================================
# MountainCarVecEnv Tests
# ============================================================


class TestMountainCar:
    """Tests for MountainCarVecEnv environment."""

    def test_initialization(self):
        """Test environment can be created with various sizes."""
        for num_envs in [1, 8, 64, 4096]:
            env = MountainCarVecEnv(num_envs)
            assert env.num_envs == num_envs
            assert env.observation_size == 2

    def test_reset_shape(self):
        """Test reset returns correct observation shape."""
        env = MountainCarVecEnv(128)
        obs = env.reset(seed=42)
        assert obs.shape == (128, 2)
        assert obs.dtype == np.float32

    def test_step_shapes(self):
        """Test step returns correct shapes for all outputs."""
        env = MountainCarVecEnv(256)
        env.reset(seed=0)

        actions = np.zeros(256, dtype=np.int32)
        obs, rewards, terminals, truncations = env.step(actions)

        assert obs.shape == (256, 2)
        assert obs.dtype == np.float32
        assert rewards.shape == (256,)
        assert rewards.dtype == np.float32
        assert terminals.shape == (256,)
        assert terminals.dtype == np.uint8
        assert truncations.shape == (256,)
        assert truncations.dtype == np.uint8

    def test_action_range(self):
        """Test that all three actions (0, 1, 2) are valid."""
        env = MountainCarVecEnv(3)
        env.reset(seed=0)

        # Test all three actions
        actions = np.array([0, 1, 2], dtype=np.int32)
        obs, rewards, _, _ = env.step(actions)

        # Should complete without error
        assert obs.shape == (3, 2)
        assert rewards.shape == (3,)

    def test_logging_functionality(self):
        """Test that episode logging works."""
        env = MountainCarVecEnv(16)
        env.reset(seed=0)

        for _ in range(100):
            actions = np.random.randint(0, 3, 16, dtype=np.int32)
            env.step(actions)

        logs = env.get_logs()
        assert "episode_count" in logs
        assert "mean_reward" in logs


# ============================================================
# PendulumVecEnv Tests
# ============================================================


class TestPendulum:
    """Tests for PendulumVecEnv environment."""

    def test_initialization(self):
        """Test environment can be created with various sizes."""
        for num_envs in [1, 8, 64, 4096]:
            env = PendulumVecEnv(num_envs)
            assert env.num_envs == num_envs
            assert env.observation_size == 3

    def test_reset_shape(self):
        """Test reset returns correct observation shape."""
        env = PendulumVecEnv(128)
        obs = env.reset(seed=42)
        assert obs.shape == (128, 3)
        assert obs.dtype == np.float32

    def test_step_shapes(self):
        """Test step returns correct shapes for all outputs."""
        env = PendulumVecEnv(256)
        env.reset(seed=0)

        actions = np.zeros(256, dtype=np.float32)
        obs, rewards, terminals, truncations = env.step(actions)

        assert obs.shape == (256, 3)
        assert obs.dtype == np.float32
        assert rewards.shape == (256,)
        assert rewards.dtype == np.float32
        assert terminals.shape == (256,)
        assert terminals.dtype == np.uint8
        assert truncations.shape == (256,)
        assert truncations.dtype == np.uint8

    def test_continuous_actions(self):
        """Test that continuous action space works."""
        env = PendulumVecEnv(4)
        env.reset(seed=0)

        # Test various torque values
        actions = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
        obs, rewards, _, _ = env.step(actions)

        assert obs.shape == (4, 3)
        assert rewards.shape == (4,)

    def test_observation_bounds(self):
        """Test that cos/sin observations are approximately in valid range."""
        env = PendulumVecEnv(64)
        env.reset(seed=42)

        for _ in range(50):
            actions = np.random.uniform(-2.0, 2.0, 64).astype(np.float32)
            obs, _, _, _ = env.step(actions)

            # cos(theta) and sin(theta) should be approximately in [-1, 1]
            # Allow small tolerance for Taylor approximation error
            assert np.all(obs[:, 0] >= -1.01) and np.all(obs[:, 0] <= 1.01)
            assert np.all(obs[:, 1] >= -1.01) and np.all(obs[:, 1] <= 1.01)

    def test_no_terminal_states(self):
        """Test that Pendulum has no terminal states (only truncations)."""
        env = PendulumVecEnv(8)
        env.reset(seed=0)

        for _ in range(200):
            actions = np.random.uniform(-2.0, 2.0, 8).astype(np.float32)
            _, _, terminals, _ = env.step(actions)

            # Pendulum should never have terminal states
            assert np.all(terminals == 0)

    def test_logging_functionality(self):
        """Test that episode logging works."""
        env = PendulumVecEnv(16)
        env.reset(seed=0)

        for _ in range(100):
            actions = np.random.uniform(-2.0, 2.0, 16).astype(np.float32)
            env.step(actions)

        logs = env.get_logs()
        assert "episode_count" in logs
        assert "mean_reward" in logs


# ============================================================
# Logger Tests
# ============================================================


class TestLogger:
    """Tests for Logger class."""

    def test_initialization(self):
        """Test Logger can be created with default and custom parameters."""
        logger1 = Logger()
        assert logger1.print_interval == 0.5
        assert logger1.newline_interval == 10

        logger2 = Logger(print_interval=1.0, newline_interval=5)
        assert logger2.print_interval == 1.0
        assert logger2.newline_interval == 5

    def test_logging_without_csv(self):
        """Test logging to console works without errors."""
        logger = Logger(print_interval=0.01)

        for i in range(10):
            logger.log(steps=100, reward=i * 10.0, length=i * 5)

        logger.close()

    def test_logging_with_csv(self, tmp_path):
        """Test logging to CSV file."""
        csv_path = tmp_path / "test_log.csv"

        logger = Logger(csv_path=str(csv_path), print_interval=0.01)

        for i in range(5):
            logger.log(steps=100, reward=float(i), length=float(i * 2))

        logger.close()

        # Verify CSV was created and has content
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "timestamp" in content
        assert "sps" in content
        assert "reward" in content

    def test_step_counting(self):
        """Test that step counting accumulates correctly."""
        logger = Logger()

        logger.log(steps=100)
        assert logger.step_count == 100

        logger.log(steps=200)
        assert logger.step_count == 300

        logger.log(steps=50)
        assert logger.step_count == 350


# ============================================================
# Cross-Environment Tests
# ============================================================


class TestCrossEnvironment:
    """Tests that apply across multiple environment types."""

    @pytest.mark.parametrize(
        "env_class,num_envs,obs_size",
        [
            (CartPoleVecEnv, 128, 4),
            (MountainCarVecEnv, 128, 2),
            (PendulumVecEnv, 128, 3),
        ],
    )
    def test_consistent_reset(self, env_class, num_envs, obs_size):
        """Test that multiple resets produce consistent behavior."""
        env = env_class(num_envs)

        obs1 = env.reset(seed=999)
        obs2 = env.reset(seed=999)

        assert obs1.shape == (num_envs, obs_size)
        assert obs2.shape == (num_envs, obs_size)
        np.testing.assert_array_equal(obs1, obs2)

    @pytest.mark.parametrize(
        "env_class,num_envs",
        [
            (CartPoleVecEnv, 1024),
            (MountainCarVecEnv, 1024),
            (PendulumVecEnv, 1024),
        ],
    )
    def test_large_batch_performance(self, env_class, num_envs):
        """Test that large batches work efficiently."""
        env = env_class(num_envs)
        env.reset(seed=0)

        # Run 100 steps with large batch
        for _ in range(100):
            if env_class == PendulumVecEnv:
                actions = np.random.uniform(-2.0, 2.0, num_envs).astype(np.float32)
            else:
                actions = np.random.randint(0, 2, num_envs, dtype=np.int32)
            env.step(actions)

        # Should complete without errors
        logs = env.get_logs()
        assert logs["episode_count"] >= 0


# ============================================================
# Performance Tests
# ============================================================


class TestPerformance:
    """Performance benchmarks and regression tests."""

    def test_cartpole_throughput(self):
        """Test CartPole achieves >50M steps/sec with 4096 envs."""
        import time

        env = CartPoleVecEnv(4096)
        env.reset(seed=0)

        # Warmup
        actions = np.random.randint(0, 2, 4096, dtype=np.int32)
        for _ in range(100):
            env.step(actions)

        # Measure
        start = time.time()
        total_steps = 0
        for _ in range(1000):
            env.step(actions)
            total_steps += 4096
        elapsed = time.time() - start

        steps_per_sec = total_steps / elapsed
        assert steps_per_sec > 50_000_000, f"Expected >50M steps/sec, got {steps_per_sec/1e6:.2f}M"


# ============================================================
# Error Handling Tests
# ============================================================


class TestErrorHandling:
    """Tests for error handling and input validation."""

    def test_invalid_num_envs_zero(self):
        """Test that num_envs=0 raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            CartPoleVecEnv(0)

    def test_invalid_num_envs_zero_mountaincar(self):
        """Test that num_envs=0 raises error for MountainCar."""
        with pytest.raises(ValueError, match="greater than 0"):
            MountainCarVecEnv(0)

    def test_invalid_num_envs_zero_pendulum(self):
        """Test that num_envs=0 raises error for Pendulum."""
        with pytest.raises(ValueError, match="greater than 0"):
            PendulumVecEnv(0)

    def test_invalid_action_shape(self):
        """Test that wrong action shape raises error."""
        env = CartPoleVecEnv(4)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Expected 4 actions"):
            env.step(np.array([0, 1], dtype=np.int32))

    def test_invalid_action_shape_too_many(self):
        """Test that too many actions raises error."""
        env = CartPoleVecEnv(4)
        env.reset(seed=42)
        with pytest.raises(ValueError, match="Expected 4 actions"):
            env.step(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))

    def test_step_determinism(self):
        """Test that stepping with same actions is deterministic."""
        env1 = CartPoleVecEnv(4)
        env2 = CartPoleVecEnv(4)

        env1.reset(seed=42)
        env2.reset(seed=42)

        actions = np.array([0, 1, 0, 1], dtype=np.int32)

        for _ in range(10):
            obs1, r1, t1, tr1 = env1.step(actions)
            obs2, r2, t2, tr2 = env2.step(actions)
            np.testing.assert_array_equal(obs1, obs2)
            np.testing.assert_array_equal(r1, r2)


class TestLoggerErrors:
    """Tests for Logger error handling."""

    def test_invalid_print_interval_zero(self):
        """Test that print_interval <= 0 raises error."""
        with pytest.raises(ValueError, match="print_interval"):
            Logger(print_interval=0)

    def test_invalid_print_interval_negative(self):
        """Test that negative print_interval raises error."""
        with pytest.raises(ValueError, match="print_interval"):
            Logger(print_interval=-1)

    def test_invalid_newline_interval_zero(self):
        """Test that newline_interval <= 0 raises error."""
        with pytest.raises(ValueError, match="newline_interval"):
            Logger(newline_interval=0)

    def test_invalid_newline_interval_negative(self):
        """Test that negative newline_interval raises error."""
        with pytest.raises(ValueError, match="newline_interval"):
            Logger(newline_interval=-1)

    def test_context_manager(self, tmp_path):
        """Test that context manager properly closes file."""
        csv_path = tmp_path / "context_test.csv"

        with Logger(csv_path=str(csv_path), print_interval=0.01) as logger:
            logger.log(steps=100, reward=1.0)

        # After context, file should be closed
        assert logger.csv_file is None
        # File should exist with content
        assert csv_path.exists()

    def test_context_manager_exception(self, tmp_path):
        """Test that context manager closes file even on exception."""
        csv_path = tmp_path / "exception_test.csv"

        with pytest.raises(RuntimeError):
            with Logger(csv_path=str(csv_path), print_interval=0.01) as logger:
                logger.log(steps=100, reward=1.0)
                raise RuntimeError("Test exception")

        # File should still be closed
        assert logger.csv_file is None


# ============================================================
# Gymnasium Space Compatibility Tests
# ============================================================


class TestGymnasiumSpaces:
    """Tests for Gymnasium-compatible space properties."""

    def test_cartpole_observation_space(self):
        """Test CartPole observation space matches Gymnasium spec."""
        env = CartPoleVecEnv(4)
        space = env.observation_space

        assert space["shape"] == (4,)
        assert space["dtype"] == "float32"
        assert len(space["low"]) == 4
        assert len(space["high"]) == 4

    def test_cartpole_action_space(self):
        """Test CartPole action space matches Gymnasium spec."""
        env = CartPoleVecEnv(4)
        space = env.action_space

        assert space["n"] == 2
        assert space["dtype"] == "int32"

    def test_mountaincar_observation_space(self):
        """Test MountainCar observation space matches Gymnasium spec."""
        env = MountainCarVecEnv(4)
        space = env.observation_space

        assert space["shape"] == (2,)
        assert space["dtype"] == "float32"
        assert len(space["low"]) == 2
        assert len(space["high"]) == 2

    def test_mountaincar_action_space(self):
        """Test MountainCar action space matches Gymnasium spec."""
        env = MountainCarVecEnv(4)
        space = env.action_space

        assert space["n"] == 3
        assert space["dtype"] == "int32"

    def test_pendulum_observation_space(self):
        """Test Pendulum observation space matches Gymnasium spec."""
        env = PendulumVecEnv(4)
        space = env.observation_space

        assert space["shape"] == (3,)
        assert space["dtype"] == "float32"
        assert len(space["low"]) == 3
        assert len(space["high"]) == 3

    def test_pendulum_action_space(self):
        """Test Pendulum action space matches Gymnasium spec (continuous)."""
        env = PendulumVecEnv(4)
        space = env.action_space

        # Pendulum uses continuous actions (Box space)
        assert space["shape"] == (1,)
        assert space["dtype"] == "float32"
        assert len(space["low"]) == 1
        assert len(space["high"]) == 1

    def test_single_observation_space_alias(self):
        """Test that single_observation_space is alias for observation_space."""
        env = CartPoleVecEnv(4)
        assert env.single_observation_space == env.observation_space

    def test_single_action_space_alias(self):
        """Test that single_action_space is alias for action_space."""
        env = CartPoleVecEnv(4)
        assert env.single_action_space == env.action_space


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
