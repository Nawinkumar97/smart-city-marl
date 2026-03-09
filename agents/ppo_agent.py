import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional
import config


class PolicyNetwork(nn.Module):
    """MLP Policy Network for PPO agent."""

    def __init__(self, obs_dim: int = 7, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    """MLP Value Network for PPO agent."""

    def __init__(self, obs_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOAgent:
    """
    PPO Agent with shared policy across all traffic agents.
    Uses separate policy and value networks.
    """

    def __init__(self, cfg: config.ProjectConfig = None):
        self.config = cfg or config.ProjectConfig()
        self.obs_dim = 7
        self.action_dim = 2

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.value = ValueNetwork(self.obs_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.training.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value.parameters(), lr=self.config.training.learning_rate
        )

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.max_norm = 0.5

        # Storage for current episode
        self.log_probs = []
        self.values = []
        self.rewards = []

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Select action given observation.

        Args:
            obs: Observation array (obs_dim,) or (n_agents, obs_dim)
            deterministic: If True, select argmax policy; else sample

        Returns:
            action: Selected action (int)
            log_prob: Log probability of selected action (float)
        """
        # Convert to float32 tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        # If batch of observations, process all agents
        if obs_tensor.dim() == 2:
            # Shape: (n_agents, obs_dim)
            n_agents = obs_tensor.shape[0]
            actions = []
            log_probs = []

            with torch.no_grad():
                logits = self.policy(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)

                if deterministic:
                    actions = torch.argmax(probs, dim=-1).tolist()
                else:
                    actions = dist.sample().tolist()

                log_probs = dist.log_prob(
                    torch.tensor(actions).to(self.device)
                ).tolist()

            return actions, log_probs

        # Single observation
        with torch.no_grad():
            logits = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = int(torch.argmax(probs))
            else:
                action = int(dist.sample())

            log_prob = float(dist.log_prob(
                torch.tensor(action).to(self.device)
            ))

        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observations (batch_size, obs_dim)
            actions: Actions (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
        """
        logits = self.policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        values = self.value(obs).squeeze(-1)

        return log_probs, values

    def update(
        self,
        obs_batch: np.ndarray,
        actions_batch: np.ndarray,
        old_log_probs_batch: np.ndarray,
        rewards_batch: np.ndarray,
        dones_batch: np.ndarray,
    ) -> dict:
        """
        Perform PPO update on a batch.

        Args:
            obs_batch: Observations (batch_size, obs_dim)
            actions_batch: Actions (batch_size,)
            old_log_probs_batch: Old log probabilities (batch_size,)
            rewards_batch: Rewards (batch_size,)
            dones_batch: Done flags (batch_size,)

        Returns:
            Dictionary of loss values
        """
        # Convert to tensors and cast to float32
        obs = torch.as_tensor(obs_batch, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions_batch, dtype=torch.long).to(self.device)
        old_log_probs = torch.as_tensor(
            old_log_probs_batch, dtype=torch.float32
        ).to(self.device)
        rewards = torch.as_tensor(rewards_batch, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones_batch, dtype=torch.float32).to(self.device)

        # Compute returns (simplified: no GAE, just rewards)
        returns = rewards

        # Get values and log probs
        log_probs, values = self.evaluate_actions(obs, actions)

        # PPO policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * returns
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_pred = values
        value_loss = nn.functional.mse_loss(value_pred, returns)

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm)
        self.policy_optimizer.step()

        # Update value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.max_norm)
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])


if __name__ == "__main__":
    # Test the PPO agent
    print("Creating PPO agent...")
    agent = PPOAgent()

    print("Running 5 random batches through update...")

    for i in range(5):
        # Generate random batch
        batch_size = 32
        obs_batch = np.random.randint(0, 10, size=(batch_size, 7)).astype(np.float32)
        actions_batch = np.random.randint(0, 2, size=batch_size).astype(np.int64)
        old_log_probs_batch = np.random.randn(batch_size).astype(np.float32)
        rewards_batch = np.random.randn(batch_size).astype(np.float32)
        dones_batch = np.zeros(batch_size).astype(np.float32)

        # Update
        losses = agent.update(
            obs_batch, actions_batch, old_log_probs_batch, rewards_batch, dones_batch
        )

        print(f"Batch {i+1}: policy_loss = {losses['policy_loss']:.4f}, value_loss = {losses['value_loss']:.4f}")

    print("\nPPO agent test complete!")
