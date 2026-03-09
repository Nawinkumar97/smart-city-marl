import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import os
import torch
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import asdict


class ReplayBuffer:
    """Experience replay buffer for storing trajectories."""

    def __init__(self, config):
        self.max_size = config.training.replay_buffer_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays
        self.obs = None
        self.actions = None
        self.log_probs = None
        self.rewards = None
        self.dones = None

    def push(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """Add a transition to the buffer."""
        # Initialize arrays on first push
        if self.obs is None:
            self.obs = np.zeros((self.max_size, obs.shape[-1]), dtype=np.float32)
            self.actions = np.zeros(self.max_size, dtype=np.int64)
            self.log_probs = np.zeros(self.max_size, dtype=np.float32)
            self.rewards = np.zeros(self.max_size, dtype=np.float32)
            self.dones = np.zeros(self.max_size, dtype=np.float32)

        # Circular buffer
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch from the buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }

    def __len__(self):
        """Return current buffer size."""
        return self.size

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        self.obs = None
        self.actions = None
        self.log_probs = None
        self.rewards = None
        self.dones = None


class Logger:
    """Logger for training metrics."""

    def __init__(self, algo: str, env: str):
        self.algo = algo
        self.env = env
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.losses: List[Dict[str, float]] = []

        # Create logs directory
        self.log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{algo}_{env}_{timestamp}.jsonl"

    def log(self, episode: int, reward: float, losses: Dict[str, float]):
        """Log metrics for an episode."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.losses.append(losses)

    def save(self):
        """Write logs to JSONL file."""
        with open(self.log_file, "w") as f:
            for episode, reward, losses in zip(
                self.episodes, self.rewards, self.losses
            ):
                record = {
                    "episode": int(episode),
                    "reward": float(reward),
                    **{k: float(v) for k, v in losses.items()},
                }
                f.write(json.dumps(record) + "\n")

    def get_recent_mean(self, n: int = 50) -> float:
        """Get mean reward over last n episodes."""
        if len(self.rewards) == 0:
            return 0.0
        recent = self.rewards[-n:]
        return float(np.mean(recent))


def save_checkpoint(
    agent,
    episode: int,
    reward_history: List[float],
    algo: str,
    env: str,
    config,
    is_best: bool = False,
):
    """Save training checkpoint."""
    # Create checkpoints directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Regular checkpoint
    checkpoint_path = checkpoint_dir / f"{algo}_{env}_ep{episode}.pt"
    torch.save(
        {
            "episode": episode,
            "reward_history": reward_history,
            "policy_state_dict": agent.policy.state_dict(),
            "value_state_dict": agent.value.state_dict(),
            "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": agent.value_optimizer.state_dict(),
            "config": asdict(config),
        },
        checkpoint_path,
    )

    # Best model checkpoint
    if is_best:
        best_path = checkpoint_dir / f"{algo}_{env}_best.pt"
        torch.save(
            {
                "episode": episode,
                "reward_history": reward_history,
                "policy_state_dict": agent.policy.state_dict(),
                "value_state_dict": agent.value.state_dict(),
                "policy_optimizer_state_dict": agent.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": agent.value_optimizer.state_dict(),
                "config": asdict(config),
            },
            best_path,
        )


def load_checkpoint(path: str, agent):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.value.load_state_dict(checkpoint["value_state_dict"])
    agent.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
    agent.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
    return checkpoint


if __name__ == "__main__":
    import config

    print("=" * 50)
    print("Testing ReplayBuffer...")
    print("=" * 50)

    cfg = config.ProjectConfig()
    buffer = ReplayBuffer(cfg)

    # Push some data
    for i in range(100):
        obs = np.random.randn(7).astype(np.float32)
        actions = np.random.randint(0, 2)
        log_probs = np.random.randn()
        rewards = np.random.randn()
        dones = 0.0
        buffer.push(obs, actions, log_probs, rewards, dones)

    print(f"Buffer size: {len(buffer)}")

    # Sample batch
    batch = buffer.sample(32)
    print(f"Sampled batch keys: {batch.keys()}")
    print(f"  obs shape: {batch['obs'].shape}")
    print(f"  actions shape: {batch['actions'].shape}")

    print("\n" + "=" * 50)
    print("Testing Logger...")
    print("=" * 50)

    logger = Logger("ppo", "traffic")

    # Log some episodes
    for episode in range(1, 11):
        reward = np.random.randn()
        losses = {
            "policy_loss": np.random.randn(),
            "value_loss": np.random.randn(),
        }
        logger.log(episode, reward, losses)

    print(f"Recent mean (n=5): {logger.get_recent_mean(5):.4f}")
    print(f"Recent mean (n=50): {logger.get_recent_mean(50):.4f}")

    # Save logs
    logger.save()
    print(f"Logs saved to: {logger.log_file}")

    print("\n" + "=" * 50)
    print("Testing save_checkpoint and load_checkpoint...")
    print("=" * 50)

    from agents.ppo_agent import PPOAgent

    agent = PPOAgent(cfg)
    reward_history = [1.0, 2.0, 3.0, 4.0, 5.0]

    save_checkpoint(agent, 5, reward_history, "ppo", "traffic", cfg, is_best=True)
    print("Checkpoint saved")

    load_checkpoint("checkpoints/ppo_traffic_ep5.pt", agent)
    print("Checkpoint loaded")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
