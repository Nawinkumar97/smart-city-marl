"""Baseline controllers for evaluation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from config import ProjectConfig
from environment.traffic import TrafficEnv
from environment.energy import EnergyEnv
from environment.transport import TransportEnv


class FixedTimeTrafficController:
    """Fixed-time traffic signal controller."""

    def __init__(self, n_agents=25, switch_interval=10):
        self.n_agents = n_agents
        self.switch_interval = switch_interval

    def select_actions(self, obs, step) -> np.ndarray:
        """
        Select traffic signal actions.

        Args:
            obs: Observation array (not used)
            step: Current step number

        Returns:
            Actions array of shape (25,) with values 0 or 1
        """
        # Return 1 (switch) every switch_interval steps, else 0
        if step % self.switch_interval == 0:
            return np.ones(self.n_agents, dtype=np.int32)
        return np.zeros(self.n_agents, dtype=np.int32)


class ThresholdEnergyController:
    """Threshold-based energy controller."""

    def __init__(self, n_agents=5):
        self.n_agents = n_agents

    def select_actions(self, obs) -> np.ndarray:
        """
        Select energy control actions based on supply/demand threshold.

        Args:
            obs: Observation array of shape (5, 5)
                - col 0: supply
                - col 1: demand

        Returns:
            Actions array of shape (5, 1) with values in {-0.5, 0.0, 0.8}
        """
        actions = np.zeros((self.n_agents, 1), dtype=np.float32)

        for i in range(self.n_agents):
            supply = obs[i, 0]
            demand = obs[i, 1]

            if demand > supply:
                # Increase power output
                actions[i, 0] = 0.8
            elif supply > demand * 1.2:
                # Decrease power output
                actions[i, 0] = -0.5
            else:
                # Maintain current level
                actions[i, 0] = 0.0

        return actions


class FixedScheduleTransportController:
    """Fixed-schedule transport controller."""

    def __init__(self, n_agents=8):
        self.n_agents = n_agents

    def select_actions(self, obs) -> np.ndarray:
        """
        Select transport actions (no schedule changes).

        Args:
            obs: Observation array (not used)

        Returns:
            Actions array of shape (8,) with all zeros
        """
        return np.zeros(self.n_agents, dtype=np.int32)


def run_traffic_baseline():
    """Run FixedTimeTrafficController on TrafficEnv."""
    config = ProjectConfig()
    env = TrafficEnv(config.traffic)
    controller = FixedTimeTrafficController(n_agents=25, switch_interval=10)

    episode_rewards = []

    print("\n" + "=" * 60)
    print("Running FixedTimeTrafficController baseline")
    print("=" * 60)

    for episode in range(1, 4):
        obs, info = env.reset()
        episode_reward = 0.0
        step = 0
        done = False

        while not done:
            actions = controller.select_actions(obs, step)
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += np.sum(rewards)
            step += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}: reward = {episode_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    print(f"\nMean reward over 3 episodes: {mean_reward:.2f}")
    return mean_reward


def run_energy_baseline():
    """Run ThresholdEnergyController on EnergyEnv."""
    config = ProjectConfig()
    env = EnergyEnv(config.energy)
    controller = ThresholdEnergyController(n_agents=5)

    episode_rewards = []

    print("\n" + "=" * 60)
    print("Running ThresholdEnergyController baseline")
    print("=" * 60)

    for episode in range(1, 4):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            actions = controller.select_actions(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += np.sum(rewards)
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}: reward = {episode_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    print(f"\nMean reward over 3 episodes: {mean_reward:.2f}")
    return mean_reward


def run_transport_baseline():
    """Run FixedScheduleTransportController on TransportEnv."""
    config = ProjectConfig()
    env = TransportEnv(config.transport)
    controller = FixedScheduleTransportController(n_agents=8)

    episode_rewards = []

    print("\n" + "=" * 60)
    print("Running FixedScheduleTransportController baseline")
    print("=" * 60)

    for episode in range(1, 4):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            actions = controller.select_actions(obs)
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += np.sum(rewards)
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}: reward = {episode_reward:.2f}")

    mean_reward = np.mean(episode_rewards)
    print(f"\nMean reward over 3 episodes: {mean_reward:.2f}")
    return mean_reward


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Controller Evaluation")
    print("=" * 60)

    traffic_reward = run_traffic_baseline()
    energy_reward = run_energy_baseline()
    transport_reward = run_transport_baseline()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"FixedTimeTrafficController:       {traffic_reward:.2f}")
    print(f"ThresholdEnergyController:        {energy_reward:.2f}")
    print(f"FixedScheduleTransportController: {transport_reward:.2f}")
