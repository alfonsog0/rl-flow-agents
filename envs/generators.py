import numpy as np
import networkx as nx
from dataclasses import dataclass

@dataclass
class FlowGenConfig:
    n_nodes: int = 15
    branch_p: float = 0.35
    dead_end_p: float = 0.10
    popup_p: float = 0.15
    hidden_dep_p: float = 0.20
    seed: int | None = None

class FlowGraph:
    """
    A sampled product-flow graph:
      - nodes: 0...(n-1), with start=0, goal=n-1
      - directed edges forward (acyclic) with random branching
      - failure modes annotated in node/edge attributes
    """
    def __init__(self, g: nx.DiGraph, popup_nodes: set[int], dead_ends: set[int], hidden_rules: dict):
        self.g = g
        self.start = 0
        self.goal = max(g.nodes)
        self.popup_nodes = popup_nodes
        self.dead_ends = dead_ends
        self.hidden_rules = hidden_rules  # e.g., {"flag_node": k, "affected_node": j}

def sample_flow(cfg: FlowGenConfig) -> FlowGraph:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_nodes
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    # forward edges to keep a DAG
    for i in range(n - 1):
        # must have at least one forward option sometimes
        for j in range(i + 1, n):
            if rng.random() < cfg.branch_p:
                g.add_edge(i, j)
        # ensure some connectivity
        if g.out_degree(i) == 0 and i < n - 1:
            j = rng.integers(i + 1, n)
            g.add_edge(i, j)

    # mark dead ends (skip goal)
    candidates = set(range(1, n - 1))
    dead_k = {k for k in candidates if rng.random() < cfg.dead_end_p}
    for k in dead_k:
        for _, j in list(g.out_edges(k)):
            g.remove_edge(k, j)

    # popups: subset of nodes where a popup may appear stochastically
    popup_nodes = {k for k in candidates if rng.random() < cfg.popup_p}

    # hidden dependency: toggling a flag at some node changes a later transition
    hidden_rules = {}
    if rng.random() < cfg.hidden_dep_p:
        flag_node = int(rng.integers(1, n // 2))
        affected = int(rng.integers(flag_node + 1, n - 1))
        hidden_rules = {"flag_node": flag_node, "affected_node": affected}

    return FlowGraph(g, popup_nodes, dead_k, hidden_rules)
