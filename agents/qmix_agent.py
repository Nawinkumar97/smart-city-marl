"""QMIX agent: Value-based multi-agent reinforcement learning algorithm."""

import sys
from pathlib import Path

# Fix sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional
from collections import deque
import random

from config import ProjectConfig


class QNetwork(nn.Module):
    """Individual Q-network for each agent."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns Q-values for each action."""
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QMIXMixer(nn.Module):
    """QMIX mixing network that combines agent Q-values into total Q-value."""

    def __init__(self, n_agents: int, obs_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.state_dim = n_agents * obs_dim

        # Hypernetworks for first layer
        self.hyper_w1 = nn.Linear(self.state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(self.state_dim, embed_dim)

        # Hypernetworks for second layer
        self.hyper_w2 = nn.Linear(self.state_dim, embed_dim)
        self.hyper_b2 = nn.Linear(self.state_dim, 1)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: combines agent Q-values into total Q-value.

        Args:
            agent_qs: Agent Q-values, shape (batch, n_agents)
            states: Global states, shape (batch, n_agents * obs_dim)

        Returns:
            Total Q-value, shape (batch,)
        """
        batch_size = agent_qs.size(0)

        # Generate weights for first layer
        w1 = torch.abs(self.hyper_w1(states))  # (batch, n_agents * embed_dim)
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)  # (batch, n_agents, embed_dim)
        b1 = self.hyper_b1(states)  # (batch, embed_dim)

        # First layer: ELU activation
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1  # (batch, embed_dim)
        hidden = F.elu(hidden)

        # Generate weights for second layer
        w2 = torch.abs(self.hyper_w2(states))  # (batch, embed_dim)
        w2 = w2.view(batch_size, self.embed_dim, 1)  # (batch, embed_dim, 1)
        b2 = self.hyper_b2(states)  # (batch, 1)

        # Second layer: linear
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2  # (batch, 1)
        return q_tot.squeeze(-1)  # (batch,)


class QMIXReplayBuffer:
    """Experience replay buffer for QMIX."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = n_agents * obs_dim

        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.dones = None

        self.ptr = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,      # (n_agents, obs_dim)
        actions: np.ndarray,  # (n_agents,)
        reward: float,        # scalar
        next_obs: np.ndarray, # (n_agents, obs_dim)
        done: bool
    ):
        """Add a transition to the buffer."""
        if self.obs is None:
            self.obs = np.zeros((self.capacity, self.n_agents, self.obs_dim), dtype=np.float32)
            self.actions = np.zeros((self.capacity, self.n_agents), dtype=np.int64)
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

        # Get states from obs
        states = self.obs[indices].reshape(batch_size, self.n_agents * self.obs_dim)
        next_states = self.next_obs[indices].reshape(batch_size, self.n_agents * self.obs_dim)

        return {
            "obs": torch.FloatTensor(self.obs[indices]),
            "states": torch.FloatTensor(states),
            "actions": torch.LongTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "next_obs": torch.FloatTensor(self.next_obs[indices]),
            "next_states": torch.FloatTensor(next_states),
            "dones": torch.FloatTensor(self.dones[indices]),
        }

    def __len__(self):
        return self.size


class QMIXAgent:
    """
    QMIX agent for multi-agent cooperative tasks.

    Uses individual Q-networks for each agent and a mixing network
    to combine Q-values while maintaining monotonicity.
    """

    def __init__(
        self,
        config: ProjectConfig,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        device: str = None
    ):
        self.config = config
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.device = device or config.device
        self.gamma = config.training.gamma
        self.lr = config.training.learning_rate
        self.batch_size = config.training.batch_size

        # Q-networks
        self.q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_network = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Mixer
        self.mixer = QMIXMixer(n_agents, obs_dim, embed_dim=32).to(self.device)
        self.target_mixer = QMIXMixer(n_agents, obs_dim, embed_dim=32).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=self.lr
        )

        # Replay buffer
        self.buffer = QMIXReplayBuffer(
            config.training.replay_buffer_size,
            n_agents,
            obs_dim
        )

        self.training_step = 0

    def select_actions(
        self,
        obs: np.ndarray,
        epsilon: float = 0.0
    ) -> np.ndarray:
        """
        Select actions using epsilon-greedy policy.

        Args:
            obs: Observations, shape (n_agents, obs_dim)
            epsilon: Exploration rate

        Returns:
            Actions, shape (n_agents,)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)  # (1, n_agents, n_actions)

        q_values = q_values.squeeze(0)  # (n_agents, n_actions)

        actions = np.zeros(self.n_agents, dtype=np.int64)

        for i in range(self.n_agents):
            if random.random() < epsilon:
                actions[i] = random.randint(0, self.n_actions - 1)
            else:
                actions[i] = q_values[i].argmax().item()

        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent using a batch of experiences.

        Args:
            batch: Dictionary containing:
                - obs: (batch, n_agents, obs_dim)
                - states: (batch, n_agents * obs_dim)
                - actions: (batch, n_agents)
                - rewards: (batch,)
                - next_obs: (batch, n_agents, obs_dim)
                - next_states: (batch, n_agents * obs_dim)
                - dones: (batch,)

        Returns:
            Dictionary of losses
        """
        # Get current Q-values
        obs = batch["obs"].to(self.device)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)

        # Q-values for each agent (batch, n_agents, n_actions)
        q_vals = self.q_network(obs)  # (batch, n_agents, n_actions)

        # Select Q-values for taken actions
        q_vals_chosen = torch.gather(q_vals, dim=2, index=actions.unsqueeze(2)).squeeze(2)  # (batch, n_agents)

        # Mix to get total Q-value
        q_tot = self.mixer(q_vals_chosen, states)  # (batch,)

        # Compute target
        with torch.no_grad():
            next_obs = batch["next_obs"].to(self.device)
            next_states = batch["next_states"].to(self.device)

            # Target Q-values
            next_q_vals = self.target_q_network(next_obs)  # (batch, n_agents, n_actions)
            next_q_vals_max = next_q_vals.max(dim=2)[0]  # (batch, n_agents)

            # Target mixed Q-value
            target_q_tot = self.target_mixer(next_q_vals_max, next_states)  # (batch,)

            # Bellman target
            rewards = batch["rewards"].to(self.device)
            dones = batch["dones"].to(self.device)
            target = rewards + self.gamma * (1 - dones) * target_q_tot

        # Compute loss
        loss = F.mse_loss(q_tot, target)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            max_norm=0.5
        )
        self.optimizer.step()

        self.training_step += 1

        return {
            "q_loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "target_mean": target.mean().item(),
        }

    def update_target(self, tau: float = 0.005):
        """
        Soft update target networks.

        Args:
            tau: Soft update coefficient
        """
        # Update Q-network
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update mixer
        for target_param, param in zip(
            self.target_mixer.parameters(),
            self.mixer.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]


def check_monotonicity(agent: QMIXAgent, batch: Dict[str, torch.Tensor]) -> bool:
    """
    Check if the mixing network satisfies monotonicity: dQ_tot/dQ_i >= 0.

    Args:
        agent: QMIX agent
        batch: Batch of data

    Returns:
        True if monotonicity holds
    """
    obs = batch["obs"].to(agent.device).requires_grad_(True)
    states = batch["states"].to(agent.device)

    # Get Q-values
    q_vals = agent.q_network(obs)  # (batch, n_agents, n_actions)
    q_vals_max = q_vals.max(dim=2)[0]  # (batch, n_agents)

    # Mix
    q_tot = agent.mixer(q_vals_max, states)  # (batch,)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=q_tot,
        inputs=q_vals_max,
        grad_outputs=torch.ones_like(q_tot),
        create_graph=True,
        retain_graph=True,
    )[0]  # (batch, n_agents)

    # Check monotonicity: all gradients should be >= 0
    return (gradients >= 0).all().item()


if __name__ == "__main__":
    # Test QMIXAgent with traffic environment
    config = ProjectConfig()
    n_agents = 25  # Traffic
    obs_dim = 7
    n_actions = 2

    print("=" * 60)
    print("Testing QMIXAgent for Traffic Environment")
    print("=" * 60)
    print(f"n_agents: {n_agents}, obs_dim: {obs_dim}, n_actions: {n_actions}")

    agent = QMIXAgent(config, n_agents, obs_dim, n_actions)

    # Fill buffer with random data
    print("\nFilling replay buffer with random data...")
    for i in range(100):
        obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
        actions = np.random.randint(0, n_actions, size=n_agents)
        reward = np.random.randn()
        next_obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
        done = False
        agent.buffer.push(obs, actions, reward, next_obs, done)

    print(f"Buffer size: {len(agent.buffer)}")

    # Run 5 random update batches
    print("\nRunning 5 update batches...")
    losses = []

    for batch_idx in range(5):
        batch = agent.buffer.sample(32)

        # Check monotonicity before update
        mono_before = check_monotonicity(agent, batch)

        loss_dict = agent.update(batch)

        # Check monotonicity after update
        mono_after = check_monotonicity(agent, batch)

        losses.append(loss_dict["q_loss"])
        print(f"  Batch {batch_idx + 1}: Q_loss = {loss_dict['q_loss']:.4f}, "
              f"Q_tot_mean = {loss_dict['q_tot_mean']:.4f}, "
              f"monotonicity: before={mono_before}, after={mono_after}")

    print(f"\nMean Q_loss: {np.mean(losses):.4f}")

    # Test action selection
    print("\nTesting action selection...")
    test_obs = np.random.randn(n_agents, obs_dim).astype(np.float32)
    actions = agent.select_actions(test_obs, epsilon=0.1)
    print(f"Selected actions shape: {actions.shape}")
    print(f"Unique actions: {np.unique(actions)}")

    # Test save/load
    print("\nTesting save/load...")
    agent.save("test_qmix_agent.pt")
    agent.load("test_qmix_agent.pt")
    print("Save/load successful!")

    import os
    os.remove("test_qmix_agent.pt")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
