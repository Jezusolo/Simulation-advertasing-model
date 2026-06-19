import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import random

# ── Agent ─────────────────────────────────────────────────────────────────────

class CustomerAgent:
    """
    One potential customer in the social network.

    Attributes
    ----------
    node_id        : int   — index in the graph
    persuadability : float — [0.1, 1.0]; personal adoption threshold = 1 - persuadability
    interest_level : float — [0, 1]
    fatigue_level  : float — [0, 1]
    is_client      : bool
    wom_boost      : float — cascade multiplier; 1.5 for followers of influencer nodes
    is_influencer  : bool  — True for the 3 super-hub nodes
    """
    __slots__ = ("node_id","persuadability","interest_level",
                 "fatigue_level","is_client","wom_boost","is_influencer")

    def __init__(self, node_id, persuadability, is_influencer=False):
        self.node_id        = node_id
        self.persuadability = persuadability
        self.interest_level = 0.0
        self.fatigue_level  = 0.0
        self.is_client      = False
        self.wom_boost      = 1.0
        self.is_influencer  = is_influencer

    def step(self, neighbor_interests, opinion_threshold, fatigue_threshold):
        # 1. Natural decay & fatigue recovery
        # Hubs: no decay, no fatigue — sustained by the sponsorship contract
        # Clients: no interest decay — they already bought, they stay satisfied
        # Others: normal decay and fatigue recovery
        if self.is_influencer:
            il = self.interest_level
            fl = 0.0
        elif self.is_client:
            il = self.interest_level   # clients keep high interest as brand advocates
            fl = max(0.0, self.fatigue_level - 0.02)
        else:
            il = self.interest_level - 0.02
            fl = self.fatigue_level  - 0.02

        # 2. Opinion Dynamics Threshold spread
        if neighbor_interests:
            n = len(neighbor_interests)
            active_sum, active_count = 0.0, 0
            for ni in neighbor_interests:
                if ni > opinion_threshold:
                    active_sum   += ni
                    active_count += 1
            if active_count > 0:
                fraction_active = active_count / n
                my_threshold    = 1.0 - self.persuadability
                if fraction_active >= my_threshold:
                    mean_active = active_sum / active_count
                    delta = (mean_active - il) * 0.3 * self.wom_boost
                    il += delta

        # 3. Ad fatigue penalty
        if fl > fatigue_threshold:
            il -= 0.05

        # 4. Clamp
        if il < 0.0: il = 0.0
        elif il > 1.0: il = 1.0
        if fl < 0.0: fl = 0.0
        elif fl > 1.0: fl = 1.0

        self.interest_level = il
        self.fatigue_level  = fl

        # 5. Conversion
        if not self.is_client and il > 0.75:
            self.is_client = True


# ── Model ─────────────────────────────────────────────────────────────────────

class AdvertisingModel:
    """
    Hybrid network: Watts-Strogatz small-world for regular agents +
    3 influencer super-nodes each connected to follower_fraction of the population.
    Opinion Dynamics Threshold cascade.
    """
    def __init__(self, N=50, k=4, beta=0.05,
                 opinion_threshold=0.5,
                 fatigue_threshold=0.6,
                 ad_strategy="mass",
                 follower_fraction=0.25,
                 seed=None):
        rng = random.Random(seed)
        self.N                  = N          # regular agents only
        self.opinion_threshold  = opinion_threshold
        self.fatigue_threshold  = fatigue_threshold
        self.ad_strategy        = ad_strategy
        self.follower_fraction  = follower_fraction
        self._seed              = seed

        # ── Build hybrid network ───────────────────────────────────────────
        # Regular agents: Watts-Strogatz ring
        self.G = nx.watts_strogatz_graph(n=N, k=k, p=beta, seed=seed)

        # Influencer super-nodes: nodes N, N+1
        # Two hubs each reaching 25% of population (50% total coverage)
        N_INF = 2
        self._inf_ids = list(range(N, N + N_INF))
        for inf_id in self._inf_ids:
            self.G.add_node(inf_id)
            # Each influencer connects to a random follower_fraction of regular agents
            # Use deterministic shuffle so seed works
            regular = list(range(N))
            rng.shuffle(regular)
            followers = regular[:max(1, int(N * follower_fraction))]
            for f in followers:
                self.G.add_edge(inf_id, f)

        total_nodes = N + N_INF

        # Circular layout for regular agents; influencers placed at centre
        pos_ring = nx.circular_layout(self.G.subgraph(range(N)))
        self.pos = dict(pos_ring)
        angles = [i * 2 * 3.14159 / N_INF for i in range(N_INF)]
        for i, inf_id in enumerate(self._inf_ids):
            self.pos[inf_id] = (0.40 * np.cos(angles[i]),
                                0.40 * np.sin(angles[i]))

        # ── Create agents ──────────────────────────────────────────────────
        self.agents = []
        for node in range(N):
            self.agents.append(CustomerAgent(node, rng.uniform(0.1, 1.0)))
        for inf_id in self._inf_ids:
            self.agents.append(CustomerAgent(inf_id, 1.0, is_influencer=True))

        self.agent_map = {a.node_id: a for a in self.agents}

        # Pre-compute neighbour lists (fast lookup in hot loop)
        self._neighbor_ids = {
            node: list(self.G.neighbors(node))
            for node in self.G.nodes()
        }

        # Give followers of influencers a wom_boost
        for inf_id in self._inf_ids:
            for nb_id in self._neighbor_ids[inf_id]:
                self.agent_map[nb_id].wom_boost = 1.5

        # ── History ───────────────────────────────────────────────────────
        self.history_clients  = []
        self.history_interest = []
        self.history_fatigue  = []
        self.step_count = 0

    # ── Advertising strategies ─────────────────────────────────────────────

    def _apply_mass(self):
        """
        Reaches 20% of regular agents at random each step.
        Small boost (+0.08) gives agents more steps before converting, allowing
        the social cascade time to build up between direct hits.
        Fatigued agents (fatigue > threshold) are skipped — ads do nothing for
        them, but they still gain interest from social cascade (word of mouth
        does not cause fatigue). (Keen, 2026: fatigue after ~7-10 hits)
        """
        regular = [a for a in self.agents if not a.is_influencer]
        sample  = random.sample(regular, max(1, int(len(regular) * 0.20)))
        for a in sample:
            if not a.is_client:
                if a.fatigue_level <= self.fatigue_threshold:
                    # Not fatigued: ad works normally
                    il = a.interest_level + 0.12
                    fl = a.fatigue_level  + 0.10
                    a.interest_level = il if il <= 1.0 else 1.0
                    a.fatigue_level  = fl if fl <= 1.0 else 1.0
                # Fatigued: ad is simply ignored — no effect at all

    def _apply_targeted(self):
        """
        Reaches top 10% of regular non-client agents by current interest level.
        Moderate boost (+0.12) — stronger than mass but still small enough to
        give the social cascade time to propagate between direct hits.
        Fatigued agents are skipped; they can still convert via social cascade.
        Grzyb et al. (2018): direct ads work best for already-engaged audiences.
        """
        regular_nc = [a for a in self.agents
                      if not a.is_influencer and not a.is_client]
        if not regular_nc:
            return
        cutoff  = max(1, int(len(regular_nc) * 0.10))
        targets = sorted(regular_nc,
                         key=lambda a: a.interest_level, reverse=True)[:cutoff]
        for a in targets:
            if a.fatigue_level <= self.fatigue_threshold:
                il = a.interest_level + 0.18
                fl = a.fatigue_level  + 0.10
                a.interest_level = il if il <= 1.0 else 1.0
                a.fatigue_level  = fl if fl <= 1.0 else 1.0
                # Fatigued: ad is simply ignored

    def _apply_influencer(self):
        """
        Influencer advertising:
        - Each of the 3 super-hub nodes is boosted every step (+0.20, fatigue +0.08).
        - Each hub directly seeds its followers at 35% of its current interest level.
          Since each hub reaches 33% of the population, the 3 hubs together can
          reach the entire population.
        - Followers have wom_boost = 1.5 so they respond more strongly to cascade.
        - Watts & Dodds (2007): influencer effect works through raising
          neighbourhood susceptibility so the wider cascade can propagate.
        """
        for inf_id in self._inf_ids:
            hub = self.agent_map[inf_id]
            # Boost hub with smaller value so hubs also take time to saturate
            il = hub.interest_level + 0.12
            fl = hub.fatigue_level  + 0.06
            hub.interest_level = il if il <= 1.0 else 1.0
            hub.fatigue_level  = fl if fl <= 1.0 else 1.0
            # Smaller seed (15%) means followers need cascade to reach conversion
            seed = hub.interest_level * 0.25  # raised so followers reliably reach 0.75
            for nb_id in self._neighbor_ids[inf_id]:
                nb = self.agent_map[nb_id]
                if not nb.is_client:
                    il2 = nb.interest_level + seed
                    nb.interest_level = il2 if il2 <= 1.0 else 1.0

    def apply_advertising(self):
        if   self.ad_strategy == "mass":       self._apply_mass()
        elif self.ad_strategy == "targeted":   self._apply_targeted()
        elif self.ad_strategy == "influencer": self._apply_influencer()

    def step(self):
        self.apply_advertising()
        random.shuffle(self.agents)

        # Snapshot before updates to avoid order-of-activation bias
        snapshot = {a.node_id: a.interest_level for a in self.agents}

        for agent in self.agents:
            nb_interests = [snapshot[nid]
                            for nid in self._neighbor_ids[agent.node_id]]
            agent.step(nb_interests, self.opinion_threshold, self.fatigue_threshold)

        self.step_count += 1
        # Count only regular agents for history (exclude influencer nodes)
        regular = [a for a in self.agents if not a.is_influencer]
        n_clients = sum(1 for a in regular if a.is_client)
        avg_int   = sum(a.interest_level for a in regular) / self.N
        avg_fat   = sum(a.fatigue_level  for a in regular) / self.N
        self.history_clients.append(n_clients)
        self.history_interest.append(avg_int)
        self.history_fatigue.append(avg_fat)

    def run(self, steps=500, stall_window=50, warmup=100):
        """
        Run until one of three stopping conditions:
        1. All regular agents are clients (full conversion).
        2. Client count has not changed for stall_window consecutive steps,
           AND at least warmup steps have elapsed (giving interest time to
           accumulate before the stall check activates).
        3. steps hard maximum reached.

        The warmup period prevents premature stopping for mass advertising,
        where agents build interest slowly for many steps before converting.
        After warmup, if clients stop growing the cascade has exhausted itself.
        """
        stall_count  = 0
        last_clients = -1
        for step_i in range(steps):
            self.step()
            current = self.history_clients[-1]
            if current >= self.N:
                break
            if step_i >= warmup:
                if current == last_clients:
                    stall_count += 1
                    if stall_count >= stall_window:
                        break
                else:
                    stall_count = 0
            last_clients = current


# ── Visualisation helpers ──────────────────────────────────────────────────────

def node_color(agent):
    if agent.is_influencer:               return "#AA00FF"   # purple = influencer hub
    if agent.is_client:                   return "#FF2222"
    elif agent.interest_level > 0.5:      return "#FF8800"
    elif agent.interest_level > 0.25:     return "#FFCC00"
    return "#888888"

def node_size(agent):
    if agent.is_influencer: return 20
    return 18 if agent.is_client else max(8, int(agent.interest_level * 16) + 6)

def build_network_figure(model):
    pos = model.pos
    edge_x, edge_y = [], []
    for u, v in model.G.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.5, color="#DDDDDD"), hoverinfo="none")

    agents = model.agents
    node_trace = go.Scatter(
        x=[pos[a.node_id][0] for a in agents],
        y=[pos[a.node_id][1] for a in agents],
        mode="markers",
        marker=dict(
            size=[node_size(a) for a in agents],
            color=[node_color(a) for a in agents],
            line=dict(width=0.5, color="#333333")
        ),
        text=[
            f"{'INFLUENCER HUB' if a.is_influencer else 'Agent'} {a.node_id}<br>"
            f"Interest: {a.interest_level:.2f}<br>"
            f"Fatigue: {a.fatigue_level:.2f}<br>"
            f"Persuadability: {a.persuadability:.2f}<br>"
            f"WoM boost: {a.wom_boost:.1f}<br>"
            f"Client: {a.is_client}"
            for a in agents
        ],
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

    axes[0].plot(steps, model.history_clients, color="#FF2222", linewidth=2)
    axes[0].set_title("Client Count"); axes[0].set_xlabel("Step")
    axes[0].set_ylim(0, model.N); axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, model.history_interest, color="#2266CC", linewidth=2)
    axes[1].set_title("Avg Interest"); axes[1].set_xlabel("Step")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, model.history_fatigue, color="#FF8800", linewidth=2)
    axes[2].set_title("Avg Fatigue"); axes[2].set_xlabel("Step")
    axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

    regular_interests = [a.interest_level for a in model.agents if not a.is_influencer]
    axes[3].hist(regular_interests, bins=20, range=(0,1), color="#9933CC",
                 edgecolor="white", linewidth=0.5)
    axes[3].set_title("Opinion Concentration (now)")
    axes[3].set_xlabel("Interest level"); axes[3].set_ylabel("Agents")
    axes[3].set_xlim(0, 1); axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Comparison helpers ─────────────────────────────────────────────────────────

def run_label(params):
    return (f"{params['ad_strategy']} | β={params['beta']:.2f} | "
            f"fatigue_thr={params['fatigue_threshold']:.2f} | "
            f"op_thr={params['opinion_threshold']:.2f} | "
            f"N={params['N']} k={params['k']}")

def build_comparison_charts(runs):
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for i, run in enumerate(runs):
        c     = colors[i % len(colors)]
        label = run["label"]
        steps = list(range(1, len(run["history_clients"]) + 1))
        reps  = run.get("reps", 1)

        for ax, key, std_key in [
            (axes[0], "history_clients",  "std_clients"),
            (axes[1], "history_interest", "std_interest"),
            (axes[2], "history_fatigue",  "std_fatigue"),
        ]:
            mean = run[key]
            ax.plot(steps, mean, color=c, label=label, linewidth=2)
            if reps > 1 and std_key in run:
                std = np.array(run[std_key]); mu = np.array(mean)
                ax.fill_between(steps, mu-std, mu+std, color=c, alpha=0.15, linewidth=0)

    x = np.arange(len(runs))
    bar_colors = [colors[i % len(colors)] for i in range(len(runs))]
    final_clients  = [r["history_clients"][-1]  for r in runs]
    final_interest = [r["history_interest"][-1] for r in runs]
    final_fatigue  = [r["history_fatigue"][-1]  for r in runs]

    for ax_i, vals, title, ylabel, ylim in [
        (3, final_clients,  "Final Client Count",   "Clients",  None),
        (4, final_interest, "Final Avg Interest",   "Interest", (0,1)),
        (5, final_fatigue,  "Final Avg Fatigue",    "Fatigue",  (0,1)),
    ]:
        axes[ax_i].bar(x, vals, color=bar_colors)
        axes[ax_i].set_title(title); axes[ax_i].set_ylabel(ylabel)
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels([f"Run {i+1}" for i in range(len(runs))], rotation=15)
        axes[ax_i].grid(True, alpha=0.3, axis="y")
        if ylim: axes[ax_i].set_ylim(*ylim)

    for ax, title, ylim in [
        (axes[0], "Client Count over Time",  None),
        (axes[1], "Avg Interest over Time",  (0,1)),
        (axes[2], "Avg Fatigue over Time",   (0,1)),
    ]:
        ax.set_title(title); ax.set_xlabel("Step"); ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    return fig


# ── Streamlit app ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Marketing Diffusion Model", layout="wide")
st.title("Marketing Diffusion Model")
st.caption("Hybrid small-world network · Opinion Dynamics Threshold · Agent-Based Simulation")

with st.sidebar:
    st.header("Parameters")
    N                 = st.slider("Regular agents (N)",          10, 1000,  50,  10)
    k                 = st.slider("Connections per agent (k)",    2,   10,   4,   1)
    beta              = st.slider("Rewiring probability (β)",    0.0,  1.0, 0.05, 0.05)
    st.divider()
    opinion_threshold = st.slider("Opinion threshold",           0.0,  1.0, 0.50, 0.05)
    fatigue_threshold = st.slider("Ad fatigue threshold",        0.0,  1.0, 0.50, 0.05)
    follower_fraction = st.slider("Influencer follower fraction",0.1,  0.5, 0.25, 0.01)
    st.divider()
    ad_strategy = st.selectbox("Advertising strategy",
                                ["mass", "targeted", "influencer"])
    st.divider()
    st.markdown("**Node colours**")
    st.markdown("🟣 Influencer hub · 🔴 Client · 🟠 >0.5 · 🟡 >0.25 · ⚫ Low")

if "model" not in st.session_state:
    st.session_state.model = None
if "comparison_runs" not in st.session_state:
    st.session_state.comparison_runs = []

def make_model():
    return AdvertisingModel(N=N, k=k, beta=beta,
                            opinion_threshold=opinion_threshold,
                            fatigue_threshold=fatigue_threshold,
                            ad_strategy=ad_strategy,
                            follower_fraction=follower_fraction)

tab_sim, tab_cmp = st.tabs(["🔴 Live Simulation", "📊 Comparison"])

# TAB 1
with tab_sim:
    c1,c2,c3,c4,c5 = st.columns(5)
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

    if step1_btn:  model.step()
    if step10_btn:
        for _ in range(10): model.step()
    if step50_btn:
        for _ in range(50): model.step()

    run_btn = st.button("▶  Run to convergence", use_container_width=False)
    if run_btn:
        with st.spinner("Running until convergence…"):
            model.run(stall_window=50)

    col_net, col_stats = st.columns([3, 1])
    with col_net:
        st.plotly_chart(build_network_figure(model), use_container_width=True)
    with col_stats:
        regular = [a for a in model.agents if not a.is_influencer]
        st.metric("Step",         model.step_count)
        st.metric("Clients",      sum(1 for a in regular if a.is_client))
        st.metric("Avg interest", f"{sum(a.interest_level for a in regular)/model.N:.3f}")
        st.metric("Avg fatigue",  f"{sum(a.fatigue_level  for a in regular)/model.N:.3f}")

    if model.step_count > 0:
        st.pyplot(build_live_charts(model))
    else:
        st.info("Press **Initialise** then step buttons to run the simulation.")

# TAB 2
with tab_cmp:
    st.subheader("Scenario Comparison")
    st.markdown(
        "Set parameters in the **sidebar**, choose steps and repetitions, "
        "then click **Add run**."
    )

    col_steps, col_reps, col_add, col_clear = st.columns([2, 2, 2, 1])
    with col_steps:
        cmp_steps = st.number_input("Steps per run", min_value=10,
                                    max_value=500, value=100, step=10)
    with col_reps:
        cmp_reps = st.number_input("Repetitions", min_value=1, max_value=50, value=1, step=1)
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
            with st.spinner(f"Running {cmp_reps} repetition(s) × {cmp_steps} steps…"):
                all_clients, all_interest, all_fatigue = [], [], []
                for rep in range(cmp_reps):
                    m = AdvertisingModel(N=N, k=k, beta=beta,
                                        opinion_threshold=opinion_threshold,
                                        fatigue_threshold=fatigue_threshold,
                                        ad_strategy=ad_strategy,
                                        follower_fraction=follower_fraction,
                                        seed=rep)
                    m.run(steps=cmp_steps, stall_window=50)
                    all_clients.append(m.history_clients)
                    all_interest.append(m.history_interest)
                    all_fatigue.append(m.history_fatigue)

                params = dict(N=N, k=k, beta=beta,
                              opinion_threshold=opinion_threshold,
                              fatigue_threshold=fatigue_threshold,
                              ad_strategy=ad_strategy)
                label = run_label(params)
                if cmp_reps > 1:
                    label += f" [{cmp_reps} reps]"

                st.session_state.comparison_runs.append({
                    "label":            label,
                    "N":                N,
                    "reps":             cmp_reps,
                    "history_clients":  np.mean(all_clients,  axis=0).tolist(),
                    "history_interest": np.mean(all_interest, axis=0).tolist(),
                    "history_fatigue":  np.mean(all_fatigue,  axis=0).tolist(),
                    "std_clients":      np.std(all_clients,   axis=0).tolist(),
                    "std_interest":     np.std(all_interest,  axis=0).tolist(),
                    "std_fatigue":      np.std(all_fatigue,   axis=0).tolist(),
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

# Run with: streamlit run advertising_model.py
