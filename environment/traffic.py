"""Traffic environment: 5x5 grid of traffic intersections with queue-based vehicle simulation."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from config import TrafficConfig


class TrafficEnv(gym.Env):
    """
    Multi-agent traffic signal environment.

    5x5 grid of intersections (25 agents), each controlling a traffic light.
    Vehicles arrive via Poisson process and queue at intersections.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: TrafficConfig = None, render_mode: str = None):
        super().__init__()
        self.config = config or TrafficConfig()
        self.render_mode = render_mode

        self.grid_size = self.config.grid_size
        self.n_agents = self.grid_size * self.grid_size
        self.episode_steps = self.config.episode_steps
        self.vehicle_arrival_rate = self.config.vehicle_arrival_rate
        self.max_queue = self.config.max_queue
        self.phase_min_duration = self.config.phase_min_duration

        # Action space: binary (0=keep, 1=switch)
        self.action_space = spaces.Discrete(2)

        # Observation space per agent: [local_queue, n, s, e, w, current_phase, time_in_phase]
        self.observation_space = spaces.Box(
            low=0, high=max(self.max_queue, 10), shape=(7,), dtype=np.float32
        )

        # 2 phases: NS green (0) or EW green (1)
        self.current_step = 0

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Queues: [local_queue, north, south, east, west] for each intersection
        # Shape: (n_agents, 5)
        self.queues = np.zeros((self.n_agents, 5), dtype=np.int32)

        # Current phase for each intersection (0 = NS green, 1 = EW green)
        self.phases = np.zeros(self.n_agents, dtype=np.int32)

        # Time in current phase for each intersection
        self.time_in_phase = np.zeros(self.n_agents, dtype=np.int32)

        self.current_step = 0

        obs = self._get_observations()
        info = self._get_info()

        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute one step in the environment."""
        # Process actions (switch phases where action=1 and min duration met)
        for i in range(self.n_agents):
            if actions[i] == 1 and self.time_in_phase[i] >= self.phase_min_duration:
                self.phases[i] = 1 - self.phases[i]  # Toggle phase
                self.time_in_phase[i] = 0
            else:
                self.time_in_phase[i] += 1

        # Process vehicles through (simulate traffic flow)
        self._process_traffic()

        # New vehicle arrivals via Poisson process
        self._generate_arrivals()

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
        obs = np.zeros((self.n_agents, 7), dtype=np.float32)

        for i in range(self.n_agents):
            row, col = i // self.grid_size, i % self.grid_size

            # Get neighbor queue information
            north_queue = self._get_neighbor_queue(row - 1, col, 1)  # incoming from north
            south_queue = self._get_neighbor_queue(row + 1, col, 1)  # incoming from south
            east_queue = self._get_neighbor_queue(row, col + 1, 2)   # incoming from east
            west_queue = self._get_neighbor_queue(row, col - 1, 2)   # incoming from west

            obs[i] = [
                float(self.queues[i, 0]),  # local_queue
                float(north_queue),
                float(south_queue),
                float(east_queue),
                float(west_queue),
                float(self.phases[i]),
                float(self.time_in_phase[i]),
            ]

        return obs

    def _get_neighbor_queue(self, row: int, col: int, lane: int) -> int:
        """Get queue at neighbor intersection. Lane: 0=local, 1=NS incoming, 2=EW incoming."""
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            neighbor_idx = row * self.grid_size + col
            return int(self.queues[neighbor_idx, lane])
        return 0

    def _process_traffic(self):
        """Process vehicles through intersections based on green phase."""
        for i in range(self.n_agents):
            if self.phases[i] == 0:  # NS green
                # Clear from local (NS direction)
                vehicles_passed = min(self.queues[i, 0], 2)  # 2 vehicles per step max
                self.queues[i, 0] -= vehicles_passed
            else:  # EW green
                # Clear from local (EW direction) - stored in different slot
                vehicles_passed = min(self.queues[i, 0], 2)
                self.queues[i, 0] -= vehicles_passed

    def _generate_arrivals(self):
        """Generate new vehicle arrivals via Poisson process."""
        for i in range(self.n_agents):
            # Arrivals on each incoming lane
            for lane in range(5):  # local, north, south, east, west
                arrivals = np.random.poisson(self.vehicle_arrival_rate)
                self.queues[i, lane] = min(
                    self.queues[i, lane] + arrivals, self.max_queue
                )

    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards: global -sum(all queues) + local -local_queue."""
        global_queue_sum = float(np.sum(self.queues[:, 0]))
        global_reward = -global_queue_sum

        # Local reward per agent
        local_queues = -self.queues[:, 0].astype(np.float32)

        # Combined reward
        rewards = global_reward + local_queues

        return rewards

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "total_queues": int(np.sum(self.queues)),
            "phases": self.phases.tolist(),
            "step": self.current_step,
        }

    def render(self):
        """Render the environment (placeholder for now)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Total queues: {np.sum(self.queues)}")


if __name__ == "__main__":
    # Test with 3 random episodes
    config = TrafficConfig()
    env = TrafficEnv(config)

    print("Running 3 random episodes...")

    for episode in range(3):
        obs, info = env.reset(seed=episode)
        episode_rewards = []
        total_reward = 0.0

        for step in range(config.episode_steps):
            # Random actions
            actions = np.random.randint(0, 2, size=env.n_agents)
            obs, rewards, terminated, truncated, info = env.step(actions)

            step_reward = np.sum(rewards)
            episode_rewards.append(step_reward)
            total_reward += step_reward

            if terminated or truncated:
                break

        mean_reward = total_reward / len(episode_rewards)
        print(f"Episode {episode + 1}: mean reward = {mean_reward:.2f}, steps = {len(episode_rewards)}")

    print("Done!")
