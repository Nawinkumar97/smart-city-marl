# 🏙️ Smart City Traffic & Resource Management with Multi-Agent RL

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Gymnasium-0.29%2B-008000?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>A full-stack Multi-Agent Reinforcement Learning system simulating cooperative control of traffic signals, energy distribution, and public transport across a smart city with 38 agents.</b>
</p>

---

## 📌 Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** framework for smart city resource optimization. Three types of agents — traffic light controllers, energy node managers, and transport line coordinators — learn cooperative policies to minimize city-wide congestion, energy imbalance, and transport delays.

Three state-of-the-art MARL algorithms are implemented from scratch in PyTorch and evaluated against rule-based baselines:

- **QMIX** — Cooperative value decomposition with monotonic mixing network (primary algorithm)
- **MADDPG** — Centralized critic with decentralized actors for continuous control
- **PPO** — Independent proximal policy optimization (baseline comparison)

The system includes a live **Streamlit dashboard** for real-time training monitoring, city-state visualization, and evaluation comparison — all containerized with Docker for full reproducibility.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Smart City Environment                   │
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   TrafficEnv    │  │  EnergyEnv   │  │ TransportEnv  │  │
│  │  5x5 grid       │  │  5 nodes     │  │  8 lines      │  │
│  │  25 agents      │  │  5 agents    │  │  8 agents     │  │
│  │  discrete       │  │  continuous  │  │  discrete     │  │
│  └────────┬────────┘  └──────┬───────┘  └──────┬────────┘  │
│           └──────────────────┴──────────────────┘           │
│                    CityEnv (38 total agents)                  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │    QMIX     │    │   MADDPG    │    │     PPO     │
   │  Traffic +  │    │   Energy    │    │  Baseline   │
   │  Transport  │    │             │    │             │
   │  Mixing     │    │ Centralized │    │ Independent │
   │  Network    │    │   Critic    │    │   Policy    │
   └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🤖 Algorithms

| Algorithm | Type | Action Space | Used For | Key Innovation |
|-----------|------|-------------|----------|----------------|
| **QMIX** | Value-based | Discrete | Traffic, Transport | Monotonic mixing network guarantees cooperative Q-value decomposition |
| **MADDPG** | Actor-Critic | Continuous | Energy | Centralized critic observes all agents during training, decentralized execution |
| **PPO** | Policy Gradient | Discrete | Baseline | Clipped surrogate objective prevents destructive policy updates |

---

## 🌆 Environments

| Environment | Agents | State Space | Action Space | Reward Signal |
|-------------|--------|------------|-------------|---------------|
| **TrafficEnv** | 25 (5x5 grid) | Queue lengths + phase + neighbors (7,) | Binary: keep/switch phase | -sum(all queue lengths) |
| **EnergyEnv** | 5 nodes | Supply, demand, storage, frequency (5,) | Continuous: power adjustment [-1,1] | -abs(supply-demand) + storage bonus |
| **TransportEnv** | 8 lines | Load, deviation, queue, fuel, time (5,) | Discrete: maintain/speed/slow/skip | -abs(schedule deviation) - wait time |

---

## 📊 Results

### Traffic Environment (20 evaluation episodes)

| Algorithm | Mean Reward | Std | vs. Baseline |
|-----------|------------|-----|-------------|
| **PPO** | -1,559 | ±1,281 | **+2.3%** ✅ |
| **QMIX** | -1,576 | ±882 | **+1.3%** ✅ |
| Fixed-Time Baseline | -1,596 | ±1,190 | — |

Both PPO and QMIX outperform the fixed-time switching baseline. The margin reflects that symmetric Poisson arrivals create a relatively easy baseline task — adaptive control shows greater advantage under asymmetric rush-hour demand patterns.

### Energy Environment (20 evaluation episodes)

| Algorithm | Mean Reward | vs. Baseline |
|-----------|------------|-------------|
| **MADDPG** | -114 | — |
| Threshold Baseline | -7 | — |

MADDPG did not converge within 1,000 training episodes. The sparse brownout penalty creates a challenging exploration landscape for continuous control. Reward shaping and extended training (5,000+ episodes) are planned as future work.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (tested on 8GB VRAM)
- 4GB RAM minimum

### Installation

```bash
# Clone the repository
git clone https://github.com/Nawinkumar97/smart-city-marl.git
cd smart-city-marl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train QMIX on all environments (recommended)
PYTHONPATH=. python training/train.py --algo qmix --env all --episodes 2000

# Train MADDPG on energy environment
PYTHONPATH=. python training/train.py --algo maddpg --env energy --episodes 1000

# Train PPO baseline on traffic
PYTHONPATH=. python training/train.py --algo ppo --env traffic --episodes 500

# Available options
# --algo   [ppo | qmix | maddpg]
# --env    [traffic | energy | transport | all]
# --episodes  <int>
```

### Evaluation

```bash
# Run full evaluation against baselines
PYTHONPATH=. python evaluation/evaluate.py

# Generate comparison plots (saved to evaluation/plots/)
PYTHONPATH=. python evaluation/plot_results.py
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

| Tab | Description |
|-----|-------------|
| **Training Progress** | Live reward curves, loss charts, rolling mean window=50 |
| **City Live View** | Traffic queue heatmap, energy supply/demand bars, transport deviation timeline |
| **Evaluation Results** | Algorithm comparison charts, % improvement metrics |

### Docker

```bash
docker build -t smart-city-marl .
docker run -p 8501:8501 smart-city-marl
```

---

## 📁 Project Structure

```
smart_city_marl/
├── environment/
│   ├── traffic.py          # TrafficEnv — 5x5 grid, 25 agents, discrete
│   ├── energy.py           # EnergyEnv — 5 nodes, continuous actions
│   ├── transport.py        # TransportEnv — 8 lines, discrete actions
│   └── city_env.py         # CityEnv — unified wrapper (38 agents)
├── agents/
│   ├── base_agent.py       # Abstract base class
│   ├── ppo_agent.py        # PPO with shared policy across agents
│   ├── qmix_agent.py       # QMIX with hypernetwork mixing network
│   └── maddpg_agent.py     # MADDPG with centralized critic per agent
├── training/
│   ├── train.py            # CLI entry point
│   ├── trainer.py          # Training loop logic
│   └── utils.py            # ReplayBuffer, Logger, checkpointing
├── evaluation/
│   ├── baseline.py         # Rule-based baseline controllers
│   ├── evaluate.py         # Full evaluation pipeline
│   └── plot_results.py     # Plotly HTML figure generation
├── dashboard/
│   └── app.py              # Streamlit dashboard (3 tabs)
├── config.py               # All hyperparameters as dataclasses
├── requirements.txt
├── docker/
│   └── Dockerfile
└── README.md
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py` with no magic numbers in logic files:

```python
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
```

---

## 📚 References

### Core Algorithms

1. Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018).
   **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.**
   *ICML 2018.* [arXiv:1803.11605](https://arxiv.org/abs/1803.11605)

2. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).
   **Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.**
   *NeurIPS 2017.* [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   **Proximal Policy Optimization Algorithms.**
   *arXiv preprint.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

4. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016).
   **Continuous Control with Deep Reinforcement Learning.**
   *ICLR 2016.* [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

### Foundational MARL Theory

5. Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... & Graepel, T. (2018).
   **Value-Decomposition Networks for Cooperative Multi-Agent Learning.**
   *AAMAS 2018.* [arXiv:1706.05296](https://arxiv.org/abs/1706.05296)

6. Foerster, J., Assael, I. A., de Freitas, N., & Whiteson, S. (2016).
   **Learning to Communicate with Deep Multi-Agent Reinforcement Learning.**
   *NeurIPS 2016.* [arXiv:1605.06676](https://arxiv.org/abs/1605.06676)

7. Oliehoek, F. A., & Amato, C. (2016).
   **A Concise Introduction to Decentralized POMDPs.**
   *Springer Briefs in Intelligent Systems.*

### Smart City & Traffic Applications

8. Chu, T., Wang, J., Codecà, L., & Li, Z. (2019).
   **Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control.**
   *IEEE Transactions on Intelligent Transportation Systems.* [arXiv:1903.04527](https://arxiv.org/abs/1903.04527)

9. Wei, H., Zheng, G., Yao, H., & Li, Z. (2018).
   **IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control.**
   *KDD 2018.*

10. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015).
    **Human-level control through deep reinforcement learning.**
    *Nature, 518(7540), 529-533.*

---

## 🔭 Future Work

- **Asymmetric demand patterns** — Rush-hour vehicle arrival spikes to demonstrate adaptive control advantage over fixed-time switching
- **Reward shaping for MADDPG** — Replace sparse brownout penalty with smooth gradient signal to improve energy agent convergence
- **Agent communication** — Attention-based message passing between neighboring agents for emergent coordination
- **Scale to 50+ agents** — Benchmark QMIX scalability with larger grid configurations
- **Real data calibration** — SUMO traffic simulator integration and real smart grid demand profiles
- **LLM-powered explainability** — RAG-based system explaining agent decisions in natural language

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome. Feel free to check the [issues page](https://github.com/Nawinkumar97/smart-city-marl/issues).

```bash
# Fork the repo, then:
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Nawinkumar Vasanthakumar**
Master's Student in Electromobility — AI & Autonomous Driving
University of Erlangen-Nuremberg (FAU), Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nawinkumar-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/nawinkumar)
[![GitHub](https://img.shields.io/badge/GitHub-Nawinkumar97-181717?style=flat&logo=github)](https://github.com/Nawinkumar97)
[![Email](https://img.shields.io/badge/Email-nawinkumar.vasanthakumar@fau.de-D14836?style=flat&logo=gmail)](mailto:nawinkumar.vasanthakumar@fau.de)

---

<p align="center">
  <i>If you find this project useful, please consider giving it a ⭐ — it helps others discover the work!</i>
</p>
