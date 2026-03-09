import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import random
import numpy as np
import torch

from tqdm import tqdm

from config import ProjectConfig
from environment.traffic import TrafficEnv
from environment.energy import EnergyEnv
from environment.transport import TransportEnv
from environment.city_env import CityEnv
from agents.ppo_agent import PPOAgent
from agents.qmix_agent import QMIXAgent
from agents.maddpg_agent import MADDPGAgent
from training.utils import ReplayBuffer, Logger, save_checkpoint, load_checkpoint


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_env(args, config):
    """Create environment based on args."""
    if args.env == "traffic":
        return TrafficEnv(config.traffic)
    elif args.env == "energy":
        return EnergyEnv(config.energy)
    elif args.env == "transport":
        return TransportEnv(config.transport)
    elif args.env == "all":
        return CityEnv(config)
    else:
        raise ValueError(f"Unknown environment: {args.env}")


def create_agent(args, config, env):
    """Create agent based on args and environment."""
    if args.algo == "ppo":
        return PPOAgent(config)
    elif args.algo == "qmix":
        # Check if using CityEnv
        if args.env == "all":
            # Create 3 separate QMIXAgents for CityEnv
            return {
                "traffic": QMIXAgent(config, 25, 7, 2),
                "energy": QMIXAgent(config, 5, 5, 2),
                "transport": QMIXAgent(config, 8, 5, 4),
            }
        else:
            # Single QMIXAgent for single environment
            if args.env == "traffic":
                return QMIXAgent(config, 25, 7, 2)
            elif args.env == "energy":
                return QMIXAgent(config, 5, 5, 2)
            elif args.env == "transport":
                return QMIXAgent(config, 8, 5, 4)
            else:
                raise ValueError(f"Unknown environment: {args.env}")
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")


def train_ppo(args, config, env, agent, buffer, logger):
    """PPO training loop."""
    n_agents = env.n_agents
    update_interval = 200
    save_interval = 100

    best_reward = float("-inf")
    total_steps = 0
    episode_rewards = []

    pbar = tqdm(total=args.episodes, desc=f"Training PPO on {args.env}")

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0

        done = False
        while not done:
            actions, log_probs = agent.select_action(obs)

            if isinstance(actions, list):
                actions = np.array(actions)
                log_probs = np.array(log_probs)

            obs_next, rewards, terminated, truncated, info = env.step(actions)

            for i in range(n_agents):
                buffer.push(
                    obs[i].astype(np.float32),
                    int(actions[i]) if isinstance(actions[i], np.int64) else actions[i],
                    float(log_probs[i]) if isinstance(log_probs[i], np.float64) else log_probs[i],
                    float(rewards[i]),
                    1.0 if terminated else 0.0,
                )

            episode_reward += np.sum(rewards)
            total_steps += 1

            if total_steps % update_interval == 0 and len(buffer) >= 32:
                batch = buffer.sample(32)
                losses = agent.update(
                    batch["obs"],
                    batch["actions"],
                    batch["log_probs"],
                    batch["rewards"],
                    batch["dones"],
                )
                logger.log(episode, episode_reward, losses)

            obs = obs_next
            done = terminated or truncated

        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-50:])
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "mean(50)": f"{mean_reward:.2f}"
            })

        if episode % save_interval == 0:
            is_best = episode_reward > best_reward
            if is_best:
                best_reward = episode_reward

            save_checkpoint(
                agent,
                episode,
                episode_rewards,
                args.algo,
                args.env,
                config,
                is_best=is_best,
            )

        pbar.update(1)

    pbar.close()
    logger.save()

    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Logs saved to: {logger.log_file}")


def train_qmix(args, config, env, agents, logger):
    """QMIX training loop."""
    save_interval = 100

    # Handle single agent or multiple agents
    is_multi_agent = isinstance(agents, dict)

    if is_multi_agent:
        traffic_agent = agents["traffic"]
        energy_agent = agents["energy"]
        transport_agent = agents["transport"]
    else:
        single_agent = agents

    best_reward = float("-inf")
    episode_rewards = []

    pbar = tqdm(total=args.episodes, desc=f"Training QMIX on {args.env}")

    for episode in range(1, args.episodes + 1):
        # Epsilon decay
        epsilon = max(0.05, 1.0 - episode / 500)

        if is_multi_agent:
            # Training with CityEnv
            obs_dict, info_dict = env.reset()

            episode_reward = 0.0
            done = False

            while not done:
                # Select actions for each agent type
                actions_dict = {
                    "traffic": traffic_agent.select_actions(obs_dict["traffic"], epsilon),
                    "energy": energy_agent.select_actions(obs_dict["energy"], epsilon),
                    "transport": transport_agent.select_actions(obs_dict["transport"], epsilon),
                }

                # Step environment
                obs_dict_next, rewards_dict, terminated, truncated, info_dict = env.step(actions_dict)

                # Calculate total reward
                total_reward = (
                    np.sum(rewards_dict["traffic"]) +
                    np.sum(rewards_dict["energy"]) +
                    np.sum(rewards_dict["transport"])
                )
                episode_reward += total_reward

                # Push to each buffer
                done = terminated or truncated

                # Push traffic transitions
                for i in range(25):
                    traffic_agent.buffer.push(
                        obs_dict["traffic"][i],
                        actions_dict["traffic"][i],
                        float(rewards_dict["traffic"][i]),
                        obs_dict_next["traffic"][i],
                        done
                    )

                # Push energy transitions
                for i in range(5):
                    energy_agent.buffer.push(
                        obs_dict["energy"][i],
                        actions_dict["energy"][i],
                        float(rewards_dict["energy"][i]),
                        obs_dict_next["energy"][i],
                        done
                    )

                # Push transport transitions
                for i in range(8):
                    transport_agent.buffer.push(
                        obs_dict["transport"][i],
                        actions_dict["transport"][i],
                        float(rewards_dict["transport"][i]),
                        obs_dict_next["transport"][i],
                        done
                    )

                # Update each agent
                batch_size = config.training.batch_size

                if len(traffic_agent.buffer) >= batch_size:
                    batch = traffic_agent.buffer.sample(batch_size)
                    losses = traffic_agent.update(batch)
                    traffic_agent.update_target()

                if len(energy_agent.buffer) >= batch_size:
                    batch = energy_agent.buffer.sample(batch_size)
                    losses = energy_agent.update(batch)
                    energy_agent.update_target()

                if len(transport_agent.buffer) >= batch_size:
                    batch = transport_agent.buffer.sample(batch_size)
                    losses = transport_agent.update(batch)
                    transport_agent.update_target()

                obs_dict = obs_dict_next

            episode_rewards.append(episode_reward)

            losses = {"q_loss": 0.0}  # default if no update happened
            logger.log(episode, float(episode_reward), losses)

        else:
            # Training with single environment
            obs, info = env.reset()

            episode_reward = 0.0
            done = False

            while not done:
                actions = single_agent.select_actions(obs, epsilon)
                obs_next, rewards, terminated, truncated, info = env.step(actions)

                # Calculate total reward
                total_reward = np.sum(rewards)
                episode_reward += total_reward

                # Push to buffer (need to handle per-agent transitions)
                for i in range(len(obs)):
                    single_agent.buffer.push(
                        obs[i],
                        actions[i],
                        float(rewards[i]),
                        obs_next[i],
                        terminated
                    )

                # Update
                batch_size = config.training.batch_size
                if len(single_agent.buffer) >= batch_size:
                    batch = single_agent.buffer.sample(batch_size)
                    losses = single_agent.update(batch)
                    single_agent.update_target()

                obs = obs_next
                done = terminated or truncated

            episode_rewards.append(episode_reward)

            losses = {"q_loss": 0.0}  # default if no update happened
            logger.log(episode, float(episode_reward), losses)

        # Print progress every 10 episodes
        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-50:])
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "mean(50)": f"{mean_reward:.2f}",
                "epsilon": f"{epsilon:.3f}"
            })

        # Save checkpoint every 100 episodes
        if episode % save_interval == 0:
            is_best = episode_reward > best_reward
            if is_best:
                best_reward = episode_reward

            if is_multi_agent:
                # Save each agent separately
                torch.save({
                    "traffic": traffic_agent.q_network.state_dict(),
                    "energy": energy_agent.q_network.state_dict(),
                    "transport": transport_agent.q_network.state_dict(),
                    "mixer_traffic": traffic_agent.mixer.state_dict(),
                    "mixer_energy": energy_agent.mixer.state_dict(),
                    "mixer_transport": transport_agent.mixer.state_dict(),
                    "episode": episode,
                    "reward_history": episode_rewards,
                }, f"checkpoints/qmix_{args.env}_ep{episode}.pt")
            else:
                single_agent.save(f"checkpoints/qmix_{args.env}_ep{episode}.pt")

        pbar.update(1)

    pbar.close()
    logger.save()

    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.2f}")


def train_maddpg(args):
    """MADDPG training loop for continuous control."""
    config = ProjectConfig()
    set_seed(config.seed)

    # Validate environment
    if args.env != "energy":
        print(f"Error: MADDPG is only supported for 'energy' environment, got '{args.env}'")
        sys.exit(1)

    # Create environment
    env = EnergyEnv(config.energy)

    # Create MADDPG agent
    agent = MADDPGAgent(config, n_agents=5, obs_dim=5, action_dim=1)

    # Create logger
    logger = Logger("maddpg", "energy")

    best_reward = float("-inf")
    episode_rewards = []

    pbar = tqdm(total=args.episodes, desc=f"Training MADDPG on energy")

    for episode in range(1, args.episodes + 1):
        # Noise annealing
        sigma = max(0.05, 0.2 - episode / 1000 * 0.15)
        agent.noise.sigma = sigma

        obs, info = env.reset()
        agent.reset_noise()
        episode_reward = 0.0
        done = False
        losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        while not done:
            actions = agent.select_actions(obs, add_noise=True)
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            total_reward = float(np.sum(rewards))
            agent.buffer.push(obs, actions, total_reward, next_obs, terminated)

            episode_reward += total_reward

            if len(agent.buffer) >= config.training.batch_size:
                batch = agent.buffer.sample(config.training.batch_size)
                losses = agent.update(batch)
                agent.update_targets()

            obs = next_obs
            done = terminated or truncated

        episode_rewards.append(episode_reward)

        logger.log(episode, float(episode_reward), losses)

        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Save best checkpoint
            agent.save("checkpoints/maddpg_energy_best.pt")

        # Print every 10 episodes
        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards[-50:])
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "mean(50)": f"{mean_reward:.2f}",
                "noise": f"{sigma:.3f}"
            })

        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            agent.save(f"checkpoints/maddpg_energy_ep{episode}.pt")

        pbar.update(1)

    pbar.close()
    logger.save()

    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.2f}")


def train(args):
    """Main training loop."""
    config = ProjectConfig()
    set_seed(config.seed)

    # Route to MADDPG training
    if args.algo == "maddpg":
        train_maddpg(args)
        return

    # Create environment
    env = create_env(args, config)

    # Create agent
    agents = create_agent(args, config, env)

    # Create buffer and logger
    if args.algo == "ppo":
        buffer = ReplayBuffer(config)
    else:
        buffer = None

    logger = Logger(args.algo, args.env)

    # Route to appropriate training function
    if args.algo == "ppo":
        train_ppo(args, config, env, agents, buffer, logger)
    elif args.algo == "qmix":
        train_qmix(args, config, env, agents, logger)


def main():
    parser = argparse.ArgumentParser(description="Train agent on city environment")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "qmix", "maddpg"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="traffic", choices=["traffic", "energy", "transport", "all"], help="Environment to use")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
