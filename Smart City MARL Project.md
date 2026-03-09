# CLAUDE.md — Smart City MARL Project

## Deadline
**Sunday night** — full project must be complete, documented, and dashboard-ready.

---

## Project Overview
Multi-Agent Reinforcement Learning simulation of a smart city with three interconnected
environments: Traffic Signals, Energy Distribution, and Public Transport. Multiple agents
learn cooperative policies to minimize congestion, energy waste, and transport delays.

---

## Sprint Plan (Stick to This Order)

### Friday Night — Foundation
**Goal: One environment running, PPO baseline working, dashboard skeleton live.**
1. `config.py` — full config dataclass for all three environments
2. `environment/traffic.py` — TrafficEnv complete and tested
3. `agents/ppo_agent.py` — PPO as baseline agent (shared policy)
4. `training/train.py` — training loop, reward logging, checkpointing
5. `dashboard/app.py` — skeleton with reward curve (Tab 1 only)
6. Confirm: `python training/train.py` runs 100 episodes without crashing

### Saturday — Environments + QMIX
**Goal: All three environments done, QMIX working on traffic.**
1. `environment/energy.py` — EnergyEnv complete
2. `environment/transport.py` — TransportEnv complete
3. `environment/city_env.py` — CityEnv wrapper combining all three
4. `agents/qmix_agent.py` — QMIX implementation (primary algorithm)
5. Update `training/train.py` to support QMIX + multi-env
6. Dashboard Tab 2 — grid heatmap visualization
7. Confirm: QMIX trains on all three envs, rewards trending upward

### Sunday — MADDPG + Evaluation + Polish
**Goal: MADDPG on energy, full evaluation, complete dashboard, README.**
1. `agents/maddpg_agent.py` — MADDPG for energy environment (continuous actions)
2. `evaluation/evaluate.py` — run all three agents, collect all metrics
3. `evaluation/plot_results.py` — comparison plots (QMIX vs MADDPG vs PPO vs baseline)
4. Dashboard Tab 3 — full evaluation comparison
5. `README.md` — complete with setup, architecture, results
6. `docker/Dockerfile` — containerize (last task, 30 min)

---

## Architecture Decisions

### Environment Design

#### TrafficEnv (`environment/traffic.py`)
- Gymnasium-compatible (`gymnasium.Env`)
- Grid: **5x5 intersections** (25 traffic light agents)
- State per agent: `[local_queue, north_queue, south_queue, east_queue, west_queue, current_phase, time_in_phase]` shape (7,)
- Action per agent: `0` = keep phase, `1` = switch phase (discrete, binary)
- Vehicles modeled as **queue counts** (integers) — no individual vehicle simulation
- Each step = 5 simulation seconds; episode = 200 steps
- Vehicle arrival: Poisson process per incoming lane (lambda configurable)
- Reward: `-sum(all_queue_lengths)` globally shared + local `-local_queue` per agent

#### EnergyEnv (`environment/energy.py`)
- Grid: **5 energy nodes** (generators + storage + consumers)
- State per agent: `[supply, demand, storage_level, grid_frequency, neighbor_load]` shape (5,)
- Action per agent: **continuous** `[-1, 1]` maps to `[-max_adjust, +max_adjust]` MW
- Reward: `-abs(supply - demand)` per node; bonus for keeping storage 20-80% full
- Penalty: `-100` for brownout (demand > supply + storage) or overload

#### TransportEnv (`environment/transport.py`)
- **8 bus/tram lines** as agents
- State per agent: `[current_load, schedule_deviation, next_stop_queue, fuel_level, time_of_day]` shape (5,)
- Action per agent: `0` = maintain, `1` = speed up, `2` = slow down, `3` = skip stop (discrete, 4 actions)
- Reward: `-abs(schedule_deviation)` - `0.5 * passenger_wait_time`
- Penalty for over-capacity loading

#### CityEnv (`environment/city_env.py`)
- Wrapper combining all three environments
- Manages joint observations and rewards
- Supports running environments independently or jointly
- Provides unified `step()` and `reset()` interface

---

### Agents

#### PPO (`agents/ppo_agent.py`) — Baseline
- Shared policy across all agents of same type
- Policy network: MLP `[obs_dim -> 256 -> 128 -> action_dim]` with ReLU
- Value network: same architecture, separate weights
- Discrete actions only (traffic + transport)
- Update every `N_STEPS` with clipping `epsilon=0.2`
- **Purpose: baseline comparison only** — shows how much QMIX/MADDPG improves

#### QMIX (`agents/qmix_agent.py`) — Primary Algorithm
- **Centralized training, decentralized execution**
- Individual Q-networks per agent: MLP `[obs_dim -> 128 -> 64 -> n_actions]`
- Mixing network: all Q-values + global state -> monotonic combination -> Q_tot
- Hypernetworks generate mixing weights from global state
- Target network with soft update `tau=0.005`
- Shared replay buffer across all agents
- Use for: **Traffic** (primary) and **Transport** (primary)

#### MADDPG (`agents/maddpg_agent.py`) — Continuous Actions
- Actor per agent: MLP `[obs_dim -> 256 -> 128 -> action_dim]` + tanh output
- Critic per agent: MLP `[all_obs + all_actions -> 256 -> 128 -> 1]` (centralized)
- Target networks for both actor and critic; soft updates `tau=0.01`
- Ornstein-Uhlenbeck noise for exploration
- Use for: **Energy** environment (continuous allocation)

---

## Algorithm Selection

| Environment | Best Algorithm | Why |
|-------------|---------------|-----|
| Traffic | **QMIX** | Cooperative discrete actions, global reward decomposition |
| Energy | **MADDPG** | Continuous allocation, centralized critic handles node dependencies |
| Transport | **QMIX** | Cooperative discrete actions (speed/skip decisions) |
| Baseline (all) | PPO | Independent agents — shows benefit of cooperation |

---

## Config (`config.py`)

```python
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
```

---

## File Structure
```
smart_city_marl/
├── environment/
│   ├── traffic.py          # TrafficEnv — 5x5 grid, 25 agents, discrete
│   ├── energy.py           # EnergyEnv — 5 nodes, continuous actions
│   ├── transport.py        # TransportEnv — 8 lines, discrete
│   └── city_env.py         # CityEnv — unified wrapper
├── agents/
│   ├── base_agent.py       # Abstract base class
│   ├── ppo_agent.py        # PPO — baseline
│   ├── qmix_agent.py       # QMIX — traffic + transport
│   └── maddpg_agent.py     # MADDPG — energy
├── training/
│   ├── train.py            # Entry: --algo [ppo|qmix|maddpg] --env [traffic|energy|transport|all]
│   ├── trainer.py          # Training loop logic
│   └── utils.py            # ReplayBuffer, Logger, checkpoint helpers
├── evaluation/
│   ├── evaluate.py         # Load checkpoint, run N episodes, collect metrics
│   ├── baseline.py         # Rule-based fixed-time controllers
│   └── plot_results.py     # Comparison figures
├── dashboard/
│   └── app.py              # Streamlit — 3 tabs
├── config.py
├── requirements.txt
├── docker/Dockerfile
├── README.md
├── codex.md
└── CLAUDE.md
```

---

## Running the Project

```bash
# Install
pip install -r requirements.txt

# Train PPO on traffic (Friday baseline)
python training/train.py --algo ppo --env traffic --episodes 500

# Train QMIX on all environments (Saturday)
python training/train.py --algo qmix --env all --episodes 2000

# Train MADDPG on energy (Sunday)
python training/train.py --algo maddpg --env energy --episodes 1000

# Evaluate all
python evaluation/evaluate.py --checkpoint checkpoints/qmix_all_best.pt --env all

# Dashboard
streamlit run dashboard/app.py
```

---

## Dashboard (`dashboard/app.py`)

### Tab 1 — Training Progress
- Live episode reward curve (Plotly line chart)
- Loss curves (policy loss, value loss, Q_tot loss)
- Rolling mean +/- std band (window=50 episodes)
- Sidebar: select algorithm + environment
- Auto-refresh every 3 seconds via `st.rerun()`
- Reads from `logs/` directory (JSON lines format)

### Tab 2 — City Live View
- **Traffic**: 5x5 Plotly heatmap of queue lengths (red=congested)
- **Energy**: Bar chart supply vs demand per node + storage gauge
- **Transport**: Timeline of schedule deviation per line (green=on time, red=late)
- Slider to step through episode frames from saved rollout

### Tab 3 — Evaluation Comparison
- Side-by-side bars: QMIX vs MADDPG vs PPO vs Rule-Based Baseline
- Traffic metrics: avg queue length, avg wait time, throughput
- Energy metrics: imbalance, brownout events, storage efficiency
- Transport metrics: avg deviation, passenger wait time, capacity utilization
- % improvement over baseline highlighted in green

---

## Rule-Based Baselines (`evaluation/baseline.py`)

```python
class FixedTimeTrafficController:
    # Switches every 10 ticks regardless of queue state

class ThresholdEnergyController:
    # Static demand forecast, no adaptation

class FixedScheduleTransportController:
    # Follows timetable exactly, no speed adjustment
```

---

## Evaluation Metrics

| Environment | Metric | Description |
|-------------|--------|-------------|
| Traffic | avg_queue | Mean vehicles waiting across all intersections |
| Traffic | avg_wait_time | Avg ticks a vehicle spends waiting |
| Traffic | throughput | Vehicles cleared per episode |
| Energy | imbalance | Mean abs(supply - demand) per step |
| Energy | brownout_rate | % steps with demand unmet |
| Energy | storage_efficiency | % time storage in 20-80% range |
| Transport | avg_deviation | Mean abs schedule deviation in ticks |
| Transport | passenger_wait | Avg passenger wait time at stops |
| Transport | capacity_util | Mean load / capacity ratio |
| All | vs_baseline_pct | % improvement over rule-based controller |

---

## Logging Convention
Each training run writes to `logs/{algo}_{env}_{timestamp}.jsonl`:
```json
{"episode": 1, "reward": -245.3, "loss": 0.043, "epsilon": 0.95, "timestamp": 1234567890}
```
Dashboard reads these files live. No database needed.

---

## Checkpointing
```python
torch.save({
    'episode': episode,
    'algo': algo_name,
    'env': env_name,
    'model_state': agent.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'reward_history': reward_history,
    'config': asdict(config),
}, f"checkpoints/{algo}_{env}_ep{episode}.pt")
# Also save best: checkpoints/{algo}_{env}_best.pt
```

---

## Code Style
- Python 3.10+, type hints everywhere, docstrings on all classes/public methods
- No global state — pass config dataclass
- `pathlib.Path` not `os.path`
- `logging` module not `print` (except dashboard)
- Zero magic numbers — all values in `config.py`
- snake_case files, PascalCase classes, UPPER_SNAKE_CASE constants

---

## Hard Rules for Claude Code
- **Simplest working version first**, then add features
- **Test each file after writing it** — never write 5 files then debug
- **No stable-baselines3, rllib, or tianshou** — PyTorch from scratch only
- **No individual vehicle simulation** — queues are integer counts
- **No async/multiprocessing** in training loop
- **All config values from config.py** — zero hardcoded numbers in logic
- **Friday goal: PPO running on traffic** — do not start Saturday tasks on Friday
- **When two approaches exist**, ask before implementing

---

## Dependencies (`requirements.txt`)
```
gymnasium>=0.29
torch>=2.1
numpy>=1.26
matplotlib>=3.8
plotly>=5.18
streamlit>=1.30
pandas>=2.1
scipy>=1.11
tqdm>=4.66
```

---

## v3 Backlog (Post-Sunday, Do Not Implement)
- [ ] Scale to 50+ agents
- [ ] Agent communication with message passing / attention
- [ ] RAG-based decision explainer (LLM integration)
- [ ] 3D visualization with PyGame or NVIDIA Omniverse
- [ ] Real traffic dataset calibration (SUMO integration)
- [ ] Centralized vs decentralized training comparison study
