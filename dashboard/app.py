import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Smart City MARL Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Smart City MARL Dashboard")


def get_log_files():
    """Get all .jsonl files from logs directory."""
    log_dir = Path(__file__).parent.parent / "logs"
    if not log_dir.exists():
        return []
    return sorted([f for f in log_dir.glob("*.jsonl")
                   if f.stat().st_size > 0])


def load_log_data(file_path: str) -> pd.DataFrame:
    """Load log data from jsonl file."""
    records = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(eval(line))
    return pd.DataFrame(records)


def generate_dummy_data():
    """Generate realistic dummy data for city visualization."""
    # Traffic: 5x5 grid of queue lengths
    traffic_queues = np.random.randint(0, 20, size=(5, 5))

    # Energy: supply and demand per node (40-80 MW)
    energy_supply = np.random.uniform(40, 80, size=5)
    energy_demand = np.random.uniform(40, 80, size=5)

    # Transport: schedule deviation per line (-10 to 10)
    transport_deviation = np.random.uniform(-10, 10, size=8)

    return {
        "traffic": traffic_queues,
        "energy_supply": energy_supply,
        "energy_demand": energy_demand,
        "transport": transport_deviation,
    }


# Sidebar
st.sidebar.header("Settings")

# Live toggle
live_mode = st.sidebar.checkbox("Live", value=False)

# Log file selection
log_files = get_log_files()

if not log_files:
    st.info("No training logs found. Start training first.")
    st.stop()

# Extract filenames for dropdown
file_options = [f.name for f in log_files]
selected_file = st.sidebar.selectbox("Select Log File", file_options)

# Get full path
selected_path = log_files[file_options.index(selected_file)]

# Load data
df = load_log_data(str(selected_path))

if df.empty:
    st.info("Selected log file is empty. Start training first.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["Tab 1: Training Progress", "Tab 2: City Live View", "Tab 3: Evaluation Results"])

# ============ TAB 1: Training Progress ============
with tab1:
    # Metrics
    col1, col2 = st.columns(2)

    best_reward = df["reward"].max() if "reward" in df.columns else 0.0
    latest_mean_50 = df["reward"].tail(50).mean() if len(df) >= 50 else df["reward"].mean()

    col1.metric("Best Reward", f"{best_reward:.2f}")
    col2.metric("Latest Mean(50)", f"{latest_mean_50:.2f}")

    # Reward chart
    st.subheader("Episode Rewards")

    fig_reward = go.Figure()

    # Raw rewards
    fig_reward.add_trace(go.Scatter(
        x=df["episode"],
        y=df["reward"],
        mode="lines",
        name="Reward",
        line=dict(color="blue", width=1),
        opacity=0.5,
    ))

    # Rolling mean
    if len(df) >= 50:
        rolling_mean = df["reward"].rolling(window=50).mean()
        fig_reward.add_trace(go.Scatter(
            x=df["episode"],
            y=rolling_mean,
            mode="lines",
            name="Rolling Mean (50)",
            line=dict(color="orange", width=2),
        ))

    fig_reward.update_layout(
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    st.plotly_chart(fig_reward, use_container_width=True)

    # Loss charts (if available)
    if "policy_loss" in df.columns and "value_loss" in df.columns:
        st.subheader("Losses")

        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=df["episode"],
            y=df["policy_loss"],
            mode="lines",
            name="Policy Loss",
            line=dict(color="red", width=1),
        ))

        fig_loss.add_trace(go.Scatter(
            x=df["episode"],
            y=df["value_loss"],
            mode="lines",
            name="Value Loss",
            line=dict(color="green", width=1),
        ))

        fig_loss.update_layout(
            xaxis_title="Episode",
            yaxis_title="Loss",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        st.plotly_chart(fig_loss, use_container_width=True)

# ============ TAB 2: City Live View ============
with tab2:
    st.header("City Live View")

    # Note about live data
    st.info("📡 Live data will connect to training in v2")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()

    # Generate or retrieve session state for dummy data
    if "city_data" not in st.session_state:
        st.session_state.city_data = generate_dummy_data()

    data = st.session_state.city_data

    # Section 1: Traffic Grid
    st.subheader("🚦 Traffic Grid")
    fig_traffic = go.Figure(data=go.Heatmap(
        z=data["traffic"],
        colorscale=[
            [0, "green"],      # 0 = green
            [1, "red"]          # 20 = red
        ],
        zmin=0,
        zmax=20,
        showscale=True,
        colorbar=dict(title="Queue Length"),
    ))

    fig_traffic.update_layout(
        title="Traffic Queue Lengths (5x5 Grid)",
        xaxis_title="Column",
        yaxis_title="Row",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_traffic, use_container_width=True)

    # Section 2: Energy Nodes
    st.subheader("⚡ Energy Nodes")

    nodes = [f"Node {i+1}" for i in range(5)]

    fig_energy = go.Figure()

    # Supply bars
    fig_energy.add_trace(go.Bar(
        x=nodes,
        y=data["energy_supply"],
        name="Supply (MW)",
        marker_color="blue",
    ))

    # Demand bars
    fig_energy.add_trace(go.Bar(
        x=nodes,
        y=data["energy_demand"],
        name="Demand (MW)",
        marker_color="orange",
    ))

    fig_energy.update_layout(
        title="Energy Supply vs Demand",
        xaxis_title="Node",
        yaxis_title="Power (MW)",
        barmode="group",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_energy, use_container_width=True)

    # Section 3: Transport Lines
    st.subheader("🚌 Transport Lines")

    lines = [f"Line {i+1}" for i in range(8)]

    # Color based on deviation
    colors = ["green" if abs(d) < 3 else "red" for d in data["transport"]]

    fig_transport = go.Figure()

    fig_transport.add_trace(go.Bar(
        y=lines,
        x=data["transport"],
        orientation="h",
        marker_color=colors,
        text=[f"{d:.1f}" for d in data["transport"]],
        textposition="outside",
    ))

    fig_transport.update_layout(
        title="Transport Schedule Deviation (minutes)",
        xaxis_title="Deviation (minutes)",
        yaxis_title="Line",
        xaxis=dict(range=[-15, 15]),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig_transport, use_container_width=True)

# ============ TAB 3: Evaluation Results ============
with tab3:
    st.header("Evaluation Results")

    results_path = Path(__file__).parent.parent / "evaluation" / "results.json"

    if not results_path.exists():
        st.warning("Run evaluate.py first to generate evaluation results.")
    else:
        # Read results
        import json
        with open(results_path, "r") as f:
            results = json.load(f)

        # Row 1: Metric cards
        st.subheader("Improvement over Baseline")
        col1, col2, col3 = st.columns(3)

        ppo_imp = results["traffic"]["ppo"]["improvement_pct"]
        qmix_imp = results["traffic"]["qmix"]["improvement_pct"]
        maddpg_imp = results["energy"]["maddpg"]["improvement_pct"]

        col1.metric("PPO vs Baseline", f"{ppo_imp:+.2f}%", delta=ppo_imp)
        col2.metric("QMIX vs Baseline", f"{qmix_imp:+.2f}%", delta=qmix_imp)
        col3.metric("MADDPG vs Baseline", f"{maddpg_imp:+.2f}%", delta=maddpg_imp)

        st.markdown("---")

        # Helper function to embed HTML files
        def embed_html(file_path: Path, height: int = 500):
            if not file_path.exists():
                st.warning(f"{file_path.name} not found. Run plot_results.py first.")
                return
            with open(file_path, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=height)

        # Row 2: Reward comparison
        st.subheader("Reward Comparison")
        embed_html(Path(__file__).parent.parent / "evaluation" / "plots" / "reward_comparison.html")

        # Row 3: Improvement chart
        st.subheader("Improvement over Baseline")
        embed_html(Path(__file__).parent.parent / "evaluation" / "plots" / "improvement.html")

        # Row 4: Training curves
        st.subheader("Training Curves")
        embed_html(Path(__file__).parent.parent / "evaluation" / "plots" / "training_curves.html")

# Live mode auto-refresh
if live_mode:
    import time
    time.sleep(5)
    st.rerun()
