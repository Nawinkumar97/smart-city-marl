"""Transport environment: 8 bus/tram lines with schedule and passenger management."""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from config import TransportConfig


class TransportEnv(gym.Env):
    """
    Multi-agent public transport environment.

    8 bus/tram lines, each managing schedule, passenger load, and capacity.
    Passenger demand varies by time of day with peak hours at 8am and 5pm.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: TransportConfig = None, render_mode: str = None):
        super().__init__()
        self.config = config or TransportConfig()
        self.render_mode = render_mode

        self.n_agents = self.config.n_lines
        self.max_capacity = self.config.max_capacity
        self.schedule_tolerance = self.config.schedule_tolerance
        self.episode_steps = 200

        # Number of stops per line
        self.n_stops = 10

        # Action space: discrete 4 actions
        # 0 = maintain schedule, 1 = speed up, 2 = slow down, 3 = skip stop
        self.action_space = spaces.Discrete(4)

        # Observation space per agent: [current_load, schedule_deviation, next_stop_queue, fuel_level, time_of_day]
        self.observation_space = spaces.Box(
            low=0.0, high=200.0, shape=(5,), dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Current passenger load for each line
        self.current_load = np.random.uniform(20, 50, size=self.n_agents).astype(np.float32)

        # Schedule deviation (minutes early = negative, late = positive)
        self.schedule_deviation = np.random.uniform(-2, 2, size=self.n_agents).astype(np.float32)

        # Next stop queue (passengers waiting)
        self.next_stop_queue = np.random.uniform(5, 15, size=self.n_agents).astype(np.float32)

        # Fuel level (0-100%)
        self.fuel_level = np.random.uniform(60, 90, size=self.n_agents).astype(np.float32)

        # Current time of day (0-24 hours, simulated)
        self.time_of_day = 6.0  # Start at 6am

        # Number of stops skipped this episode
        self.stops_skipped = np.zeros(self.n_agents, dtype=np.int32)

        # Passenger wait time accumulator
        self.passenger_wait_time = np.zeros(self.n_agents, dtype=np.float32)

        self.current_step = 0

        obs = self._get_observations()
        info = self._get_info()

        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute one step in the environment."""
        # Process actions
        for i in range(self.n_agents):
            action = actions[i]

            if action == 0:  # maintain schedule
                pass
            elif action == 1:  # speed up
                self.schedule_deviation[i] -= 1.0  # Arrive earlier
                self.fuel_level[i] -= 2.0  # Use more fuel
            elif action == 2:  # slow down
                self.schedule_deviation[i] += 1.0  # Arrive later
                self.fuel_level[i] -= 1.0
            elif action == 3:  # skip stop
                self.stops_skipped[i] += 1
                self.next_stop_queue[i] += self.next_stop_queue[i] * 0.2  # More passengers accumulate

        # Update time of day (each step = ~7.2 minutes in a 24-hour cycle for 200 steps)
        self.time_of_day = (6.0 + self.current_step * 0.12) % 24.0

        # Generate passenger demand based on time of day
        demand = self._get_passenger_demand()

        # Boarding passengers
        for i in range(self.n_agents):
            # Passengers waiting board
            boarding = min(self.next_stop_queue[i], self.max_capacity - self.current_load[i])
            self.current_load[i] += boarding

            # Passengers alight (20% at each stop)
            alighting = self.current_load[i] * 0.2
            self.current_load[i] -= alighting

            # New passengers arrive at stop
            self.next_stop_queue[i] = max(0, self.next_stop_queue[i] - boarding + demand[i])

            # Accumulate wait time
            self.passenger_wait_time[i] += self.next_stop_queue[i] * 0.12  # ~7.2 minutes per step

        # Fuel consumption
        self.fuel_level = np.clip(self.fuel_level - 0.5, 0.0, 100.0)

        # Update schedule deviation based on load
        for i in range(self.n_agents):
            if self.current_load[i] > self.max_capacity * 0.8:
                self.schedule_deviation[i] += 0.5  # Slow down due to crowding

        # Calculate rewards
        rewards = self._calculate_rewards()

        # Check termination
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_steps

        obs = self._get_observations()
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    def _get_passenger_demand(self) -> np.ndarray:
        """Get passenger demand based on time of day."""
        demand = np.ones(self.n_agents, dtype=np.float32) * 5.0  # Base demand

        # Peak hours: 8am and 5pm (±1 hour)
        peak_morning = 8.0
        peak_evening = 17.0

        # Morning peak
        if abs(self.time_of_day - peak_morning) <= 1.0:
            demand += 15.0
        # Evening peak
        elif abs(self.time_of_day - peak_evening) <= 1.0:
            demand += 15.0
        # Slight increase during day
        elif 9.0 <= self.time_of_day <= 16.0:
            demand += 5.0
        # Night - lower demand
        elif self.time_of_day < 6.0 or self.time_of_day > 21.0:
            demand -= 2.0

        # Add random noise
        demand += np.random.uniform(-2, 2, size=self.n_agents)

        return np.clip(demand, 1.0, 30.0)

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        obs = np.zeros((self.n_agents, 5), dtype=np.float32)

        for i in range(self.n_agents):
            obs[i] = [
                self.current_load[i],
                self.schedule_deviation[i],
                self.next_stop_queue[i],
                self.fuel_level[i],
                self.time_of_day,
            ]

        return obs

    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards per agent."""
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            # Primary reward: -abs(schedule_deviation)
            rewards[i] = -abs(self.schedule_deviation[i])

            # Secondary reward: -0.5 * passenger_wait_time
            rewards[i] -= 0.5 * self.passenger_wait_time[i]

            # Reset wait time accumulator
            self.passenger_wait_time[i] = 0.0

            # Penalty: over capacity
            if self.current_load[i] > self.max_capacity:
                rewards[i] -= 50.0

        return rewards

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "total_passengers": float(np.sum(self.current_load)),
            "avg_schedule_deviation": float(np.mean(np.abs(self.schedule_deviation))),
            "stops_skipped": self.stops_skipped.tolist(),
            "step": self.current_step,
        }

    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Passengers: {np.sum(self.current_load):.0f}, Time: {self.time_of_day:.1f}")


if __name__ == "__main__":
    # Test with 3 random episodes
    config = TransportConfig()
    env = TransportEnv(config)

    print("Running 3 random episodes...")

    episode_rewards_list = []

    for episode in range(3):
        obs, info = env.reset(seed=episode)
        episode_rewards = []
        total_reward = 0.0

        for step in range(env.episode_steps):
            # Random actions (0-3)
            actions = np.random.randint(0, 4, size=env.n_agents)
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
