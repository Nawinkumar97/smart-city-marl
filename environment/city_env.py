"""City environment: unified wrapper combining Traffic, Energy, and Transport environments."""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any

from config import ProjectConfig, TrafficConfig, EnergyConfig, TransportConfig
from environment.traffic import TrafficEnv
from environment.energy import EnergyEnv
from environment.transport import TransportEnv


class CityEnv(gym.Env):
    """
    Unified city environment wrapping Traffic, Energy, and Transport environments.

    Total agents: 25 (traffic) + 5 (energy) + 8 (transport) = 38
    """

    def __init__(self, config: ProjectConfig = None):
        super().__init__()
        self.config = config or ProjectConfig()

        # Initialize all three environments
        self.traffic_env = TrafficEnv(self.config.traffic)
        self.energy_env = EnergyEnv(self.config.energy)
        self.transport_env = TransportEnv(self.config.transport)

        # Agent counts
        self.n_traffic_agents = 25
        self.n_energy_agents = 5
        self.n_transport_agents = 8
        self.n_agents = self.n_traffic_agents + self.n_energy_agents + self.n_transport_agents

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """Reset all environments and return observations."""
        obs_traffic, info_traffic = self.traffic_env.reset(seed=seed)
        obs_energy, info_energy = self.energy_env.reset(seed=seed)
        obs_transport, info_transport = self.transport_env.reset(seed=seed)

        obs_dict = {
            "traffic": obs_traffic,
            "energy": obs_energy,
            "transport": obs_transport,
        }

        info_dict = {
            "traffic": info_traffic,
            "energy": info_energy,
            "transport": info_transport,
        }

        return obs_dict, info_dict

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """Execute one step in all environments."""
        # Step each environment
        obs_traffic, rewards_traffic, term_t, trunc_t, info_traffic = self.traffic_env.step(
            actions.get("traffic", np.array([]))
        )
        obs_energy, rewards_energy, term_e, trunc_e, info_energy = self.energy_env.step(
            actions.get("energy", np.array([]))
        )
        obs_transport, rewards_transport, term_tr, trunc_tr, info_transport = self.transport_env.step(
            actions.get("transport", np.array([]))
        )

        # Build observation dict
        obs_dict = {
            "traffic": obs_traffic,
            "energy": obs_energy,
            "transport": obs_transport,
        }

        # Build reward dict
        rewards_dict = {
            "traffic": rewards_traffic,
            "energy": rewards_energy,
            "transport": rewards_transport,
        }

        # Build info dict
        info_dict = {
            "traffic": info_traffic,
            "energy": info_energy,
            "transport": info_transport,
        }

        # Terminated if ANY env is done
        terminated = term_t or term_e or term_tr
        truncated = trunc_t or trunc_e or trunc_tr

        return obs_dict, rewards_dict, terminated, truncated, info_dict

    @property
    def total_reward(self) -> float:
        """Calculate total reward across all environments."""
        return 0.0  # Placeholder - actual calculation done in step results


if __name__ == "__main__":
    # Test with 3 random episodes
    config = ProjectConfig()
    env = CityEnv(config)

    print("Running 3 random episodes...")
    print(f"Total agents: {env.n_agents} (traffic={env.n_traffic_agents}, energy={env.n_energy_agents}, transport={env.n_transport_agents})")

    for episode in range(3):
        obs_dict, info_dict = env.reset(seed=episode)

        traffic_rewards = []
        energy_rewards = []
        transport_rewards = []

        for step in range(200):
            # Random actions for each environment
            actions = {
                "traffic": np.random.randint(0, 2, size=env.n_traffic_agents),
                "energy": np.random.uniform(-1.0, 1.0, size=(env.n_energy_agents, 1)),
                "transport": np.random.randint(0, 4, size=env.n_transport_agents),
            }

            obs_dict, rewards_dict, terminated, truncated, info_dict = env.step(actions)

            traffic_rewards.append(np.sum(rewards_dict["traffic"]))
            energy_rewards.append(np.sum(rewards_dict["energy"]))
            transport_rewards.append(np.sum(rewards_dict["transport"]))

            if terminated or truncated:
                break

        mean_traffic = np.mean(traffic_rewards)
        mean_energy = np.mean(energy_rewards)
        mean_transport = np.mean(transport_rewards)
        mean_total = mean_traffic + mean_energy + mean_transport

        print(f"Episode {episode + 1}: traffic={mean_traffic:.2f}, energy={mean_energy:.2f}, transport={mean_transport:.2f}, total={mean_total:.2f}")

    print("Done!")
