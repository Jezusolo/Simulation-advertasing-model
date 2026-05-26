import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

# ── Agent ─────────────────────────────────────────────────────────────────────

class CustomerAgent:
    def __init__(self, node_id, persuadability):
        self.node_id        = node_id
        self.persuadability = persuadability  # [0.1, 1.0]
        self.interest_level = 0.0             # [0, 1]
        self.fatigue_level  = 0.0             # [0, 1]
        self.is_client      = False

    @staticmethod
    def _clamp(v):
        return max(0.0, min(1.0, v))

    def step(self, neighbors, opinion_threshold, fatigue_threshold):
        # 1. Natural decay & fatigue recovery
        self.interest_level = self._clamp(self.interest_level - 0.02)
        self.fatigue_level  = self._clamp(self.fatigue_level  - 0.03)

        # 2. Opinion Dynamics Threshold spread
        self._opinion_threshold_interaction(neighbors, opinion_threshold)

        # 3. Ad fatigue penalty
        if self.fatigue_level > fatigue_threshold:
            self.interest_level = self._clamp(self.interest_level - 0.05)

        # 4. Conversion
        if not self.is_client and self.interest_level > 0.75:
            self.is_client = True

    def _opinion_threshold_interaction(self, neighbors, opinion_threshold):
        if not neighbors:
            return
        my_threshold = 1.0 - self.persuadability
        active = [n for n in neighbors if n.interest_level > opinion_threshold]
        if not active:
            return
        fraction_active = len(active) / len(neighbors)
        if fraction_active >= my_threshold:
            mean_interest = sum(n.interest_level for n in active) / len(active)
            delta = (mean_interest - self.interest_level) * 0.3
            self.interest_level = self._clamp(self.interest_level + delta)


# ── Model ─────────────────────────────────────────────────────────────────────

class AdvertisingModel:
    def __init__(self, N, k, beta, opinion_threshold, fatigue_threshold,
                 ad_strategy, seed=None):
        rng = random.Random(seed)
        self.N                 = N
        self.opinion_threshold = opinion_threshold
        self.fatigue_threshold = fatigue_threshold
        self.ad_strategy       = ad_strategy

        self.G   = nx.watts_strogatz_graph(n=N, k=k, p=beta, seed=seed)
        self.pos = nx.circular_layout(self.G)

        self.agents    = [CustomerAgent(node, rng.uniform(0.1, 1.0))
                          for node in self.G.nodes()]
        self.agent_map = {a.node_id: a for a in self.agents}

        self.history_clients  = []
        self.history_interest = []
        self.history_fatigue  = []
        self.step_count = 0

    def _neighbors_of(self, agent):
        return [self.agent_map[n] for n in self.G.neighbors(agent.node_id)]

    def _apply_mass(self):
        sample = random.sample(self.agents, max(1, int(len(self.agents) * 0.3)))
        for a in sample:
            a.interest_level = min(1.0, a.interest_level + 0.15)
            a.fatigue_level  = min(1.0, a.fatigue_level  + 0.12)

    def _apply_targeted(self):
        for a in self.agents:
            if a.persuadability > 0.6:
                a.interest_level = min(1.0, a.interest_level + 0.20)
                a.fatigue_level  = min(1.0, a.fatigue_level  + 0.10)

    def _apply_influencer(self):
        hubs = sorted(self.G.degree, key=lambda x: x[1], reverse=True)[:3]
        for node_id, _ in hubs:
            self.agent_map[node_id].interest_level = min(
                1.0, self.agent_map[node_id].interest_level + 0.20)

    def apply_advertising(self):
        if self.ad_strategy == "mass":
            self._apply_mass()
        elif self.ad_strategy == "targeted":
            self._apply_targeted()
        elif self.ad_strategy == "influencer":
            self._apply_influencer()
        elif self.ad_strategy == "all":
            self._apply_mass()
            self._apply_targeted()
            self._apply_influencer()

    def step(self):
        self.apply_advertising()
        random.shuffle(self.agents)
        for agent in self.agents:
            agent.step(self._neighbors_of(agent),
                       self.opinion_threshold,
                       self.fatigue_threshold)
        self.step_count += 1
        self.history_clients.append(sum(1 for a in self.agents if a.is_client))
        self.history_interest.append(
            sum(a.interest_level for a in self.agents) / len(self.agents))
        self.history_fatigue.append(
            sum(a.fatigue_level for a in self.agents) / len(self.agents))

    def run(self, steps):
        for _ in range(steps):
            self.step()


# ── Visualisation helpers ─────────────────────────────────────────────────────

def node_color(agent):
    if agent.is_client:      return "#FF2222"
    elif agent.interest_level > 0.5:  return "#FF8800"
    elif agent.interest_level > 0.25: return "#FFCC00"
    return "#888888"

def node_size(agent):
    return 18 if agent.is_client else max(8, int(agent.interest_level * 16) + 6)

def build_network_figure(model):
    pos = model.pos
    edge_x, edge_y = [], []
    for u, v in model.G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.8, color="#CCCCCC"), hoverinfo="none")

    nx_list  = [a.node_id for a in model.agents]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in nx_list],
        y=[pos[n][1] for n in nx_list],
        mode="markers",
        marker=dict(size=[node_size(a) for a in model.agents],
                    color=[node_color(a) for a in model.agents],
                    line=dict(width=0.5, color="#333333")),
        text=[f"ID: {a.node_id}<br>Interest: {a.interest_level:.2f}<br>"
              f"Fatigue: {a.fatigue_level:.2f}<br>"
              f"Persuadability: {a.persuadability:.2f}<br>Client: {a.is_client}"
              for a in model.agents],
        hoverinfo="text",
    )

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Social Network", font=dict(size=15)),
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       scaleanchor="x"),
            plot_bgcolor="#FAFAFA", height=460,
        )
    )

def build_live_charts(model):
    steps = list(range(1, model.step_count + 1))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    # Client count
    axes[0].plot(steps, model.history_clients, color="#FF2222", linewidth=2)
    axes[0].set_title("Client Count"); axes[0].set_xlabel("Step")
    axes[0].set_ylim(0, len(model.agents)); axes[0].grid(True, alpha=0.3)

    # Avg interest
    axes[1].plot(steps, model.history_interest, color="#2266CC", linewidth=2)
    axes[1].set_title("Avg Interest"); axes[1].set_xlabel("Step")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    # Avg fatigue
    axes[2].plot(steps, model.history_fatigue, color="#FF8800", linewidth=2)
    axes[2].set_title("Avg Fatigue"); axes[2].set_xlabel("Step")
    axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

    # Opinion concentration histogram (current step)
    interests = [a.interest_level for a in model.agents]
    axes[3].hist(interests, bins=20, range=(0, 1), color="#9933CC",
                 edgecolor="white", linewidth=0.5)
    axes[3].set_title("Opinion Concentration (now)")
    axes[3].set_xlabel("Interest level"); axes[3].set_ylabel("Agents")
    axes[3].set_xlim(0, 1); axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Comparison helpers ────────────────────────────────────────────────────────

def run_label(params):
    return (f"{params['ad_strategy']} | β={params['beta']:.2f} | "
            f"fatigue_thr={params['fatigue_threshold']:.2f} | "
            f"op_thr={params['opinion_threshold']:.2f} | "
            f"N={params['N']} k={params['k']}")

def build_comparison_charts(runs):
    """runs: list of dicts with keys label, history_clients, history_interest,
       history_fatigue, N"""
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, run in enumerate(runs):
        c     = colors[i % len(colors)]
        label = run["label"]
        steps = list(range(1, len(run["history_clients"]) + 1))
        N     = run["N"]

        # Client count over time
        axes[0].plot(steps, run["history_clients"], color=c, label=label, linewidth=2)
        # Avg interest over time
        axes[1].plot(steps, run["history_interest"], color=c, label=label, linewidth=2)
        # Avg fatigue over time
        axes[2].plot(steps, run["history_fatigue"], color=c, label=label, linewidth=2)

    # Final state bars
    labels       = [r["label"] for r in runs]
    final_clients  = [r["history_clients"][-1]  for r in runs]
    final_interest = [r["history_interest"][-1] for r in runs]
    final_fatigue  = [r["history_fatigue"][-1]  for r in runs]
    x = np.arange(len(runs))
    bar_colors = [colors[i % len(colors)] for i in range(len(runs))]

    axes[3].bar(x, final_clients, color=bar_colors)
    axes[3].set_title("Final Client Count"); axes[3].set_ylabel("Clients")
    axes[3].set_xticks(x); axes[3].set_xticklabels(
        [f"Run {i+1}" for i in range(len(runs))], rotation=15)
    axes[3].grid(True, alpha=0.3, axis="y")

    axes[4].bar(x, final_interest, color=bar_colors)
    axes[4].set_title("Final Avg Interest"); axes[4].set_ylabel("Interest")
    axes[4].set_ylim(0, 1)
    axes[4].set_xticks(x); axes[4].set_xticklabels(
        [f"Run {i+1}" for i in range(len(runs))], rotation=15)
    axes[4].grid(True, alpha=0.3, axis="y")

    axes[5].bar(x, final_fatigue, color=bar_colors)
    axes[5].set_title("Final Avg Fatigue"); axes[5].set_ylabel("Fatigue")
    axes[5].set_ylim(0, 1)
    axes[5].set_xticks(x); axes[5].set_xticklabels(
        [f"Run {i+1}" for i in range(len(runs))], rotation=15)
    axes[5].grid(True, alpha=0.3, axis="y")

    # Shared config for time-series axes
    for ax, title, ylim in [
        (axes[0], "Client Count over Time",   None),
        (axes[1], "Avg Interest over Time",   (0, 1)),
        (axes[2], "Avg Fatigue over Time",    (0, 1)),
    ]:
        ax.set_title(title); ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    return fig


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Marketing Diffusion Model", layout="wide")
st.title("Marketing Diffusion Model")
st.caption("Watts-Strogatz small-world network · Opinion Dynamics Threshold · Agent-Based Simulation")

# ── Sidebar (shared parameters) ───────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    N                 = st.slider("Number of agents (N)",        10,  200,  50,   1)
    k                 = st.slider("Connections per agent (k)",    2,   10,   4,   1)
    beta              = st.slider("Rewiring probability (β)",    0.0,  1.0, 0.05, 0.05)
    st.divider()
    opinion_threshold = st.slider("Opinion threshold",           0.0,  1.0, 0.30, 0.05)
    fatigue_threshold = st.slider("Ad fatigue threshold",        0.0,  1.0, 0.70, 0.05)
    st.divider()
    ad_strategy = st.selectbox("Advertising strategy",
                                ["mass", "targeted", "influencer", "all"])
    st.divider()
    st.markdown("**Node colours**")
    st.markdown("🔴 Client &nbsp;&nbsp; 🟠 > 0.5 &nbsp;&nbsp; 🟡 > 0.25 &nbsp;&nbsp; ⚫ Low")

# ── Session state init ────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "comparison_runs" not in st.session_state:
    st.session_state.comparison_runs = []

def make_model():
    return AdvertisingModel(N=N, k=k, beta=beta,
                            opinion_threshold=opinion_threshold,
                            fatigue_threshold=fatigue_threshold,
                            ad_strategy=ad_strategy)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_sim, tab_cmp = st.tabs(["🔴 Live Simulation", "📊 Comparison"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    c1, c2, c3, c4, c5 = st.columns(5)
    init_btn   = c1.button("Initialise",  use_container_width=True)
    reset_btn  = c2.button("Reset",       use_container_width=True)
    step1_btn  = c3.button("Step ×1",     use_container_width=True)
    step10_btn = c4.button("Step ×10",    use_container_width=True)
    step50_btn = c5.button("Step ×50",    use_container_width=True)

    if st.session_state.model is None or init_btn or reset_btn:
        st.session_state.model = make_model()

    model = st.session_state.model
    model.ad_strategy       = ad_strategy
    model.opinion_threshold = opinion_threshold
    model.fatigue_threshold = fatigue_threshold

    if step1_btn:
        model.step()
    if step10_btn:
        for _ in range(10): model.step()
    if step50_btn:
        for _ in range(50): model.step()

    col_net, col_stats = st.columns([3, 1])
    with col_net:
        st.plotly_chart(build_network_figure(model), use_container_width=True)
    with col_stats:
        st.metric("Step",         model.step_count)
        st.metric("Clients",      sum(1 for a in model.agents if a.is_client))
        st.metric("Avg interest", f"{sum(a.interest_level for a in model.agents)/len(model.agents):.3f}")
        st.metric("Avg fatigue",  f"{sum(a.fatigue_level  for a in model.agents)/len(model.agents):.3f}")

    if model.step_count > 0:
        st.pyplot(build_live_charts(model))
    else:
        st.info("Press **Initialise** then **Step ×1 / ×10 / ×50** to run the simulation.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.subheader("Scenario Comparison")
    st.markdown(
        "Set parameters in the **sidebar**, choose how many steps to run, "
        "then click **Add run**. Repeat with different parameters to compare."
    )

    col_steps, col_add, col_clear = st.columns([2, 2, 1])
    with col_steps:
        cmp_steps = st.number_input("Steps per run", min_value=10,
                                    max_value=500, value=100, step=10)
    with col_add:
        add_btn = st.button("➕  Add run", use_container_width=True)
    with col_clear:
        clear_btn = st.button("🗑️  Clear", use_container_width=True)

    if clear_btn:
        st.session_state.comparison_runs = []

    if add_btn:
        if len(st.session_state.comparison_runs) >= 10:
            st.warning("Maximum 10 runs reached. Clear some before adding more.")
        else:
            with st.spinner(f"Running {cmp_steps} steps…"):
                m = make_model()
                m.run(cmp_steps)
                params = dict(N=N, k=k, beta=beta,
                              opinion_threshold=opinion_threshold,
                              fatigue_threshold=fatigue_threshold,
                              ad_strategy=ad_strategy)
                st.session_state.comparison_runs.append({
                    "label":            run_label(params),
                    "N":                N,
                    "history_clients":  m.history_clients,
                    "history_interest": m.history_interest,
                    "history_fatigue":  m.history_fatigue,
                    "params":           params,
                })
            st.success(f"Run {len(st.session_state.comparison_runs)} added.")

    runs = st.session_state.comparison_runs
    if runs:
        st.divider()
        st.markdown(f"**{len(runs)} run(s) recorded:**")
        for i, r in enumerate(runs):
            st.markdown(f"- **Run {i+1}:** {r['label']}")

        st.pyplot(build_comparison_charts(runs))
    else:
        st.info("No runs yet. Set parameters and click **➕ Add run**.")

# ── Run with: streamlit run advertising_model.py ─────────────────────────────
