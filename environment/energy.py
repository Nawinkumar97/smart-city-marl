"""Energy environment: 5-node power grid with supply, demand, and storage management."""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from config import EnergyConfig


class EnergyEnv(gym.Env):
    """
    Multi-agent energy grid environment.

    5 energy nodes, each managing supply, demand, and storage.
    Demand follows a sine wave pattern simulating day/night cycle.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnergyConfig = None, render_mode: str = None):
        super().__init__()
        self.config = config or EnergyConfig()
        self.render_mode = render_mode

        self.n_agents = self.config.n_nodes
        self.max_power_adjust = self.config.max_power_adjust
        self.storage_capacity = self.config.storage_capacity
        self.episode_steps = 200

        # Action space: continuous [-1, 1] maps to [-max_power_adjust, +max_power_adjust] MW
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space per agent: [supply, demand, storage_level, grid_frequency, neighbor_load]
        self.observation_space = spaces.Box(
            low=0.0, high=200.0, shape=(5,), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Supply for each node (MW)
        self.supply = np.random.uniform(50, 80, size=self.n_agents).astype(np.float32)

        # Storage level for each node (MWh)
        self.storage = np.random.uniform(20, 80, size=self.n_agents).astype(np.float32)

        # Base demand for each node
        self.base_demand = np.random.uniform(40, 60, size=self.n_agents).astype(np.float32)

        # Current demand (will be computed based on time)
        self.demand = self.base_demand.copy()

        # Grid frequency (nominal = 60 Hz)
        self.grid_frequency = np.full(self.n_agents, 60.0, dtype=np.float32)

        # Neighbor load for each node
        self.neighbor_load = np.zeros(self.n_agents, dtype=np.float32)

        self.current_step = 0

        obs = self._get_observations()
        info = self._get_info()

        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute one step in the environment."""
        # Actions: scale from [-1, 1] to [-max_power_adjust, +max_power_adjust]
        power_adjustment = actions.flatten() * self.max_power_adjust

        # Update supply based on actions
        self.supply = np.clip(self.supply + power_adjustment, 10.0, 150.0)

        # Update demand based on time of day (sine wave + noise)
        time_of_day = (self.current_step % 24) / 24.0 * 2 * np.pi
        demand_factor = 1.0 + 0.3 * np.sin(time_of_day - np.pi / 2)  # Peak at noon
        self.demand = self.base_demand * demand_factor + np.random.uniform(-5, 5, size=self.n_agents)
        self.demand = np.clip(self.demand, 10.0, 150.0)

        # Update storage (charge if supply > demand, discharge if demand > supply)
        net_flow = self.supply - self.demand
        self.storage = np.clip(self.storage + net_flow * 0.5, 0.0, self.storage_capacity)

        # Update grid frequency based on supply/demand balance
        imbalance = (self.demand - self.supply) / self.supply
        self.grid_frequency = 60.0 + imbalance * 2.0  # Frequency deviation
        self.grid_frequency = np.clip(self.grid_frequency, 59.0, 61.0)

        # Update neighbor load (average of other nodes' demand)
        for i in range(self.n_agents):
            neighbors = np.delete(self.demand, i)
            self.neighbor_load[i] = np.mean(neighbors)

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_steps

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        obs = np.zeros((self.n_agents, 5), dtype=np.float32)

        for i in range(self.n_agents):
            obs[i] = [
                self.supply[i],
                self.demand[i],
                self.storage[i],
                self.grid_frequency[i],
                self.neighbor_load[i],
            ]

        return obs

    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards per agent."""
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            # Primary reward: -abs(supply - demand)
            rewards[i] = -abs(self.supply[i] - self.demand[i])

            # Bonus: storage in 20-80% range
            storage_pct = self.storage[i] / self.storage_capacity
            if 0.2 <= storage_pct <= 0.8:
                rewards[i] += 0.5

            # Penalty: brownout (demand > supply + storage available)
            available_power = self.supply[i] + max(0, self.storage[i] - 20)  # Only count usable storage
            if self.demand[i] > available_power:
                rewards[i] -= 100.0

        return rewards

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "total_supply": float(np.sum(self.supply)),
            "total_demand": float(np.sum(self.demand)),
            "storage_levels": self.storage.tolist(),
            "grid_frequencies": self.grid_frequency.tolist(),
            "step": self.current_step,
        }

    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Supply: {np.sum(self.supply):.1f}, Demand: {np.sum(self.demand):.1f}")


if __name__ == "__main__":
    # Test with 3 random episodes
    config = EnergyConfig()
    env = EnergyEnv(config)

    print("Running 3 random episodes...")

    episode_rewards_list = []

    for episode in range(3):
        obs, info = env.reset(seed=episode)
        episode_rewards = []
        total_reward = 0.0

        for step in range(env.episode_steps):
            # Random actions in [-1, 1] range
            actions = np.random.uniform(-1.0, 1.0, size=(env.n_agents, 1))
            obs, rewards, terminated, truncated, info = env.step(actions)

            step_reward = np.sum(rewards)
            episode_rewards.append(step_reward)
            total_reward += step_reward

            if terminated or truncated:
                break

        mean_reward = total_reward / len(episode_rewards)
        episode_rewards_list.append(mean_reward)
        print(f"Episode {episode + 1}: mean reward = {mean_reward:.2f}, steps = {len(episode_rewards)}")

    print(f"Overall mean reward: {np.mean(episode_rewards_list):.2f}")
    print("Done!")
