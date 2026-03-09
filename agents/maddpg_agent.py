"""MADDPG agent: Multi-Agent Deep Deterministic Policy Gradient for continuous control."""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple
import random

from config import ProjectConfig


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise for exploration in continuous action spaces."""

    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.zeros(self.size)

    def sample(self) -> np.ndarray:
        """Sample noise."""
        dx = -self.theta * self.x + self.sigma * np.random.randn(self.size)
        self.x += dx
        return self.x.astype(np.float32)


class Actor(nn.Module):
    """Actor network for a single agent."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass with tanh activation."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Centralized critic network that sees all observations and actions."""

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int):
        super().__init__()
        input_dim = n_agents * obs_dim + n_agents * action_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: All agent observations, shape (batch, n_agents * obs_dim)
            actions: All agent actions, shape (batch, n_agents * action_dim)

        Returns:
            Q-values, shape (batch, 1)
        """
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGReplayBuffer:
    """Replay buffer for MADDPG."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.dones = None

        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,       # (n_agents, obs_dim)
        actions: np.ndarray,   # (n_agents, action_dim)
        reward: float,
        next_obs: np.ndarray, # (n_agents, obs_dim)
        done: bool
    ):
        """Add a transition to the buffer."""
        if self.obs is None:
            self.obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.capacity, self.n_agents, self.action_dim), dtype=np.float32)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=np.float32)

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.FloatTensor(self.obs[indices]),
            "actions": torch.FloatTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "next_obs": torch.FloatTensor(self.next_obs[indices]),
            "dones": torch.FloatTensor(self.dones[indices]),
        }

    def __len__(self):
        return self.size


class MADDPGAgent:
    """
    MADDPG agent for multi-agent continuous control.

    Uses separate actor and critic for each agent. Critics are centralized
    and see all observations and actions during training.
    """

    def __init__(
        self,
        config: ProjectConfig,
        n_agents: int = 5,
        obs_dim: int = 5,
        action_dim: int = 1,
        device: str = None
    ):
        self.config = config
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.device = device or config.device
        self.gamma = config.training.gamma
        self.lr = config.training.learning_rate
        self.batch_size = config.training.batch_size

        # Noise
        self.noise = OrnsteinUhlenbeckNoise(action_dim, theta=0.15, sigma=0.2)

        # Actor and Critic networks (one per agent)
        self.actors = [Actor(obs_dim, action_dim).to(self.device) for _ in range(n_agents)]
        self.critics = [Critic(n_agents, obs_dim, action_dim).to(self.device) for _ in range(n_agents)]

        # Target networks
        self.target_actors = [Actor(obs_dim, action_dim).to(self.device) for _ in range(n_agents)]
        self.target_critics = [Critic(n_agents, obs_dim, action_dim).to(self.device) for _ in range(n_agents)]

        # Copy weights to targets
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics]

        # Replay buffer
        self.buffer = MADDPGReplayBuffer(
            config.training.replay_buffer_size,
            n_agents,
            obs_dim,
            action_dim
        )

        self.training_step = 0

    def reset_noise(self):
        """Reset the exploration noise at the start of each episode."""
        self.noise.reset()

    def select_actions(
        self,
        obs: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select actions for all agents.

        Args:
            obs: Observations, shape (n_agents, obs_dim)
            add_noise: Whether to add exploration noise

        Returns:
            Actions, shape (n_agents, action_dim)
        """
        actions = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)

        obs_tensor = torch.FloatTensor(obs).to(self.device)

        with torch.no_grad():
            for i in range(self.n_agents):
                action = self.actors[i](obs_tensor[i:i+1]).cpu().numpy().squeeze(0)
                actions[i] = action

        if add_noise:
            noise = self.noise.sample()
            # Reshape noise to match action_dim
            noise = noise[:self.action_dim]
            actions += noise
            actions = np.clip(actions, -1.0, 1.0)

        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all agents.

        Args:
            batch: Dictionary containing:
                - obs: (batch, n_agents, obs_dim)
                - actions: (batch, n_agents, action_dim)
                - rewards: (batch,)
                - next_obs: (batch, n_agents, obs_dim)
                - dones: (batch,)

        Returns:
            Dictionary of losses
        """
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        batch_size = obs.size(0)

        # Flatten for centralized critic
        obs_flat = obs.view(batch_size, -1)  # (batch, n_agents * obs_dim)
        actions_flat = actions.view(batch_size, -1)  # (batch, n_agents * action_dim)
        next_obs_flat = next_obs.view(batch_size, -1)

        actor_losses = []
        critic_losses = []

        # Update each agent
        for i in range(self.n_agents):
            # ===== Critic update =====
            # Get target actions from target actors
            with torch.no_grad():
                target_actions = torch.zeros_like(actions)
                for j in range(self.n_agents):
                    target_actions[:, j] = self.target_actors[j](next_obs[:, j]).detach()
                target_actions_flat = target_actions.view(batch_size, -1)

                # Target Q-values
                target_q = self.target_critics[i](next_obs_flat, target_actions_flat)  # (batch, 1)
                target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q

            # Current Q-values
            current_q = self.critics[i](obs_flat, actions_flat)  # (batch, 1)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=0.5)
            self.critic_optimizers[i].step()

            # ===== Actor update =====
            # Get current actions (with noise removed for clean gradient)
            current_actions = actions.clone()
            current_actions[:, i] = self.actors[i](obs[:, i])

            # Actor loss (maximize Q-value)
            actor_loss = -self.critics[i](obs_flat, current_actions.view(batch_size, -1)).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.training_step += 1

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
        }

    def update_targets(self, tau: float = 0.01):
        """
        Soft update target networks.

        Args:
            tau: Soft update coefficient
        """
        for i in range(self.n_agents):
            # Update target actor
            for target_param, param in zip(
                self.target_actors[i].parameters(),
                self.actors[i].parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Update target critic
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save agent state."""
        checkpoint = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "target_actors": [actor.state_dict() for actor in self.target_actors],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "training_step": self.training_step,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        for i in range(self.n_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.target_actors[i].load_state_dict(checkpoint["target_actors"][i])
            self.target_critics[i].load_state_dict(checkpoint["target_critics"][i])
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizers"][i])
            self.critic_optimizers[i].load_state_dict(checkpoint["critic_optimizers"][i])

        self.training_step = checkpoint["training_step"]


if __name__ == "__main__":
    # Test MADDPGAgent for Energy environment
    config = ProjectConfig()
    n_agents = 5
    obs_dim = 5
    action_dim = 1

    print("=" * 60)
    print("Testing MADDPGAgent for Energy Environment")
    print("=" * 60)
    print(f"n_agents: {n_agents}, obs_dim: {obs_dim}, action_dim: {action_dim}")

    agent = MADDPGAgent(config, n_agents, obs_dim, action_dim)

    # Fill buffer with random data
    print("\nFilling replay buffer with random data...")
    for i in range(100):
        obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
        actions = np.random.randn(n_agents, action_dim).astype(np.float32)
        reward = np.random.randn()
        next_obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
        done = False
        agent.buffer.push(obs, actions, reward, next_obs, done)

    print(f"Buffer size: {len(agent.buffer)}")

    # Run 5 update batches
    print("\nRunning 5 update batches...")
    actor_losses = []
    critic_losses = []

    for batch_idx in range(5):
        batch = agent.buffer.sample(32)
        loss_dict = agent.update(batch)
        agent.update_targets(tau=0.01)

        actor_losses.append(loss_dict["actor_loss"])
        critic_losses.append(loss_dict["critic_loss"])

        print(f"  Batch {batch_idx + 1}: actor_loss = {loss_dict['actor_loss']:.4f}, "
              f"critic_loss = {loss_dict['critic_loss']:.4f}")

    print(f"\nMean actor_loss: {np.mean(actor_losses):.4f}")
    print(f"Mean critic_loss: {np.mean(critic_losses):.4f}")

    # Test action selection
    print("\nTesting action selection...")
    test_obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
    actions = agent.select_actions(test_obs, add_noise=True)
    print(f"Selected actions shape: {actions.shape}")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Test without noise
    actions_no_noise = agent.select_actions(test_obs, add_noise=False)
    print(f"Actions (no noise) range: [{actions_no_noise.min():.3f}, {actions_no_noise.max():.3f}]")

    # Test save/load
    print("\nTesting save/load...")
    agent.save("test_maddpg_agent.pt")
    agent.load("test_maddpg_agent.pt")
    print("Save/load successful!")

    import os
    os.remove("test_maddpg_agent.pt")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
