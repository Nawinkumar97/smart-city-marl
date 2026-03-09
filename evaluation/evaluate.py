import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from pathlib import Path
from config import ProjectConfig
from environment.traffic import TrafficEnv
from environment.energy import EnergyEnv
from agents.ppo_agent import PPOAgent
from agents.qmix_agent import QMIXAgent
from agents.maddpg_agent import MADDPGAgent
from evaluation.baseline import FixedTimeTrafficController
from evaluation.baseline import ThresholdEnergyController


def evaluate_ppo_traffic(config, n_episodes=20) -> dict:
    """Evaluate PPO agent on traffic environment."""
    env = TrafficEnv(config.traffic)
    agent = PPOAgent(config)
    checkpoint = torch.load("checkpoints/ppo_traffic_best.pt",
                            weights_only=False, map_location="cpu")
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.value.load_state_dict(checkpoint["value_state_dict"])
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            actions, _ = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.mean(r))
            done = terminated or truncated
        rewards.append(episode_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards))
    }


def evaluate_qmix_traffic(config, n_episodes=20) -> dict:
    """Evaluate QMIX agent on traffic environment."""
    env = TrafficEnv(config.traffic)
    agent = QMIXAgent(config, 25, 7, 2)
    checkpoint = torch.load("checkpoints/qmix_all_best.pt",
                            weights_only=False, map_location="cpu")
    agent.q_network.load_state_dict(checkpoint["traffic"])
    agent.mixer.load_state_dict(checkpoint["mixer_traffic"])
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            actions = agent.select_actions(obs, epsilon=0.0)
            obs, r, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.mean(r))
            done = terminated or truncated
        rewards.append(episode_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards))
    }


def evaluate_maddpg_energy(config, n_episodes=20) -> dict:
    """Evaluate MADDPG agent on energy environment."""
    env = EnergyEnv(config.energy)
    agent = MADDPGAgent(config, 5, 5, 1)
    agent.load("checkpoints/maddpg_energy_best.pt")
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            actions = agent.select_actions(obs, add_noise=False)
            obs, r, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.mean(r))
            done = terminated or truncated
        rewards.append(episode_reward / 200.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards))
    }


def evaluate_baseline_traffic(config, n_episodes=20) -> dict:
    """Evaluate baseline traffic controller."""
    env = TrafficEnv(config.traffic)
    controller = FixedTimeTrafficController()
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        while not done:
            actions = controller.select_actions(obs, step)
            obs, r, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.mean(r))
            done = terminated or truncated
            step += 1
        rewards.append(episode_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards))
    }


def evaluate_baseline_energy(config, n_episodes=20) -> dict:
    """Evaluate baseline energy controller."""
    env = EnergyEnv(config.energy)
    controller = ThresholdEnergyController()
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            actions = controller.select_actions(obs)
            obs, r, terminated, truncated, _ = env.step(actions)
            episode_reward += float(np.mean(r))
            done = terminated or truncated
        rewards.append(episode_reward / 200.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards))
    }


def run_full_evaluation(config):
    """Run full evaluation of all agents and baselines."""
    print("Running evaluation...")
    print("1/5 PPO Traffic...")
    ppo = evaluate_ppo_traffic(config)
    print("2/5 QMIX Traffic...")
    qmix = evaluate_qmix_traffic(config)
    print("3/5 MADDPG Energy...")
    maddpg = evaluate_maddpg_energy(config)
    print("4/5 Baseline Traffic...")
    base_traffic = evaluate_baseline_traffic(config)
    print("5/5 Baseline Energy...")
    base_energy = evaluate_baseline_energy(config)

    # Calculate improvement percentages
    if base_traffic["mean_reward"] != 0:
        ppo_imp = (ppo["mean_reward"] - base_traffic["mean_reward"]) / abs(base_traffic["mean_reward"]) * 100
        qmix_imp = (qmix["mean_reward"] - base_traffic["mean_reward"]) / abs(base_traffic["mean_reward"]) * 100
    else:
        ppo_imp = 0.0
        qmix_imp = 0.0

    if base_energy["mean_reward"] != 0:
        maddpg_imp = (maddpg["mean_reward"] - base_energy["mean_reward"]) / abs(base_energy["mean_reward"]) * 100
    else:
        maddpg_imp = 0.0

    results = {
        "traffic": {
            "ppo": {**ppo, "improvement_pct": round(ppo_imp, 2)},
            "qmix": {**qmix, "improvement_pct": round(qmix_imp, 2)},
            "baseline": base_traffic
        },
        "energy": {
            "maddpg": {**maddpg, "improvement_pct": round(maddpg_imp, 2)},
            "baseline": base_energy
        }
    }

    # Save results
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== TRAFFIC ===")
    print(f"PPO:      mean={ppo['mean_reward']:.0f}  improvement={ppo_imp:+.1f}%")
    print(f"QMIX:     mean={qmix['mean_reward']:.0f}  improvement={qmix_imp:+.1f}%")
    print(f"Baseline: mean={base_traffic['mean_reward']:.0f}")
    print("\n=== ENERGY ===")
    print(f"MADDPG:   mean={maddpg['mean_reward']:.0f}  improvement={maddpg_imp:+.1f}%")
    print(f"Baseline: mean={base_energy['mean_reward']:.0f}")
    print(f"\nResults saved to evaluation/results.json")


if __name__ == "__main__":
    config = ProjectConfig()
    if not Path("checkpoints/ppo_traffic_best.pt").exists():
        print("ERROR: Missing checkpoints. Run training first.")
        exit(1)
    run_full_evaluation(config)
