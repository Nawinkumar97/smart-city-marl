import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_reward_comparison(results: dict):
    """Create grouped bar chart comparing algorithm rewards."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Traffic", "Energy"),
        horizontal_spacing=0.15
    )

    # Traffic data
    traffic_algorithms = ["PPO", "QMIX", "Baseline"]
    traffic_means = [
        results["traffic"]["ppo"]["mean_reward"],
        results["traffic"]["qmix"]["mean_reward"],
        results["traffic"]["baseline"]["mean_reward"]
    ]
    traffic_stds = [
        results["traffic"]["ppo"]["std_reward"],
        results["traffic"]["qmix"]["std_reward"],
        results["traffic"]["baseline"]["std_reward"]
    ]

    # Energy data
    energy_algorithms = ["MADDPG", "Baseline"]
    energy_means = [
        results["energy"]["maddpg"]["mean_reward"],
        results["energy"]["baseline"]["mean_reward"]
    ]
    energy_stds = [
        results["energy"]["maddpg"]["std_reward"],
        results["energy"]["baseline"]["std_reward"]
    ]

    # Traffic bars (left subplot)
    colors_traffic = ["#00CC96", "#636EFA", "#EF553B"]
    for i, (alg, mean, std) in enumerate(zip(traffic_algorithms, traffic_means, traffic_stds)):
        fig.add_trace(
            go.Bar(
                name=alg,
                x=[alg],
                y=[mean],
                error_y=dict(type="data", array=[std]),
                marker_color=colors_traffic[i],
                showlegend=False
            ),
            row=1, col=1
        )

    # Energy bars (right subplot)
    colors_energy = ["#AB63FA", "#EF553B"]
    for i, (alg, mean, std) in enumerate(zip(energy_algorithms, energy_means, energy_stds)):
        fig.add_trace(
            go.Bar(
                name=alg,
                x=[alg],
                y=[mean],
                error_y=dict(type="data", array=[std]),
                marker_color=colors_energy[i],
                showlegend=True
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=dict(text="Algorithm Performance Comparison", font=dict(size=20)),
        template="plotly_dark",
        height=500,
        width=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        barmode="group"
    )

    fig.update_yaxes(title_text="Mean Reward", row=1, col=1)
    fig.update_yaxes(title_text="Mean Reward", row=1, col=2)

    plots_dir = Path("evaluation/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(plots_dir / "reward_comparison.html")
    return plots_dir / "reward_comparison.html"


def plot_improvement(results: dict):
    """Create horizontal bar chart showing % improvement over baseline."""
    comparisons = []

    # Traffic comparisons
    ppo_imp = results["traffic"]["ppo"]["improvement_pct"]
    qmix_imp = results["traffic"]["qmix"]["improvement_pct"]

    comparisons.append(("PPO vs Baseline Traffic", ppo_imp))
    comparisons.append(("QMIX vs Baseline Traffic", qmix_imp))

    # Energy comparison
    maddpg_imp = results["energy"]["maddpg"]["improvement_pct"]
    comparisons.append(("MADDPG vs Baseline Energy", maddpg_imp))

    labels = [c[0] for c in comparisons]
    values = [c[1] for c in comparisons]

    # Color based on positive/negative
    colors = ["#00CC96" if v >= 0 else "#EF553B" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in values],
        textposition="auto"
    ))

    fig.update_layout(
        title=dict(text="% Improvement over Rule-Based Baseline", font=dict(size=20)),
        template="plotly_dark",
        height=400,
        width=800,
        xaxis_title="Improvement (%)",
        yaxis=dict(autorange="reversed")
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    plots_dir = Path("evaluation/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(plots_dir / "improvement.html")
    return plots_dir / "improvement.html"


def plot_training_curves():
    """Read jsonl logs and plot training curves with rolling mean."""
    logs_dir = Path("logs")
    plots_dir = Path("evaluation/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    log_files = list(logs_dir.glob("*.jsonl"))
    log_files = [f for f in log_files if f.stat().st_size > 0]

    if not log_files:
        print("Warning: No log files found in logs/")
        return None

    colors = ["#00CC96", "#636EFA", "#EF553B", "#AB63FA", "#FFA15A", "#FF6692", "#19D3F3", "#FFD700"]
    color_idx = 0

    for log_file in log_files:
        # Extract algo name from filename
        filename = log_file.stem  # e.g., "ppo_traffic_20260306_230807"
        algo_name = filename.split("_")[0].upper()  # "PPO"

        # Read log data
        episodes = []
        rewards = []
        with open(log_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                episodes.append(data.get("episode", 0))
                rewards.append(data.get("reward", 0))

        if not episodes:
            continue

        # Compute rolling mean (window=50)
        rewards = np.array(rewards)
        if len(rewards) >= 50:
            rolling = np.convolve(rewards, np.ones(50)/50, mode="valid")
            # Align episodes with rolling mean
            rolling_episodes = episodes[:len(rolling)]
        else:
            rolling = rewards
            rolling_episodes = episodes

        fig.add_trace(go.Scatter(
            x=rolling_episodes,
            y=rolling,
            mode="lines",
            name=algo_name,
            line=dict(color=colors[color_idx % len(colors)], width=2)
        ))

        color_idx += 1

    fig.update_layout(
        title=dict(text="Training Curves", font=dict(size=20)),
        template="plotly_dark",
        height=500,
        width=900,
        xaxis_title="Episode",
        yaxis_title="Reward (Rolling Mean, window=50)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    fig.write_html(plots_dir / "training_curves.html")
    return plots_dir / "training_curves.html"


def main():
    """Generate all evaluation plots."""
    # Read results
    results_path = Path("evaluation/results.json")
    if not results_path.exists():
        print("Error: evaluation/results.json not found. Run evaluate.py first.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    print("Generating plots...")

    # Figure 1: Reward comparison
    path1 = plot_reward_comparison(results)
    print(f"1. {path1}")

    # Figure 2: Improvement
    path2 = plot_improvement(results)
    print(f"2. {path2}")

    # Figure 3: Training curves
    path3 = plot_training_curves()
    if path3:
        print(f"3. {path3}")
    else:
        print("3. No training curves generated (no log files)")


if __name__ == "__main__":
    main()
