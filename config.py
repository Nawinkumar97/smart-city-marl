from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class TrafficConfig:
    grid_size: int = 5
    episode_steps: int = 200
    vehicle_arrival_rate: float = 0.3
    max_queue: int = 20
    phase_min_duration: int = 5

@dataclass
class EnergyConfig:
    n_nodes: int = 5
    max_power_adjust: float = 10.0
    storage_capacity: float = 100.0
    target_storage_pct: float = 0.5

@dataclass
class TransportConfig:
    n_lines: int = 8
    max_capacity: int = 150
    schedule_tolerance: int = 3

@dataclass
class TrainingConfig:
    total_episodes: int = 2000
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    checkpoint_every: int = 100
    eval_every: int = 200
    n_eval_episodes: int = 20

@dataclass
class ProjectConfig:
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("checkpoints/")
    log_dir: Path = Path("logs/")
