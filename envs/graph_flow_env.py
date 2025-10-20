import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .generators import sample_flow, FlowGenConfig, FlowGraph

class FlowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: FlowGenConfig, max_steps: int = 60, r_step=-0.01, r_success=1.0, r_failure=-0.5, seed: int | None = None, obs_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.max_steps = max_steps
        self.r_step = r_step
        self.r_success = r_success
        self.r_failure = r_failure
        self.rng = np.random.default_rng(seed)
        self._build()
        self.obs_dim = obs_dim if obs_dim is not None else self.flow.g.number_of_nodes()
        # --- guard against undersized observation space
        if self.n_nodes > self.obs_dim:
            raise ValueError(
                f"n_nodes ({self.n_nodes}) > obs_dim ({self.obs_dim}). "
                "Increase obs_dim in the config."
            )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Actions: choose among a fixed-size action menu; we map to outgoing edges by index
        self.max_actions = 6  # clickA, clickB, type, next, back, dismiss (abstract)
        self.action_space = spaces.Discrete(self.max_actions)

    def _build(self):
        self.flow: FlowGraph = sample_flow(self.cfg)
        self.n_nodes = self.flow.g.number_of_nodes()
        self._node = self.flow.start
        self._t = 0
        self._flag = False  # hidden dependency flag

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._build()
        return self._obs(), {"node_id": self._node}

    def _obs(self):
        # pad/truncate to obs_dim
        x = np.zeros(self.obs_dim, dtype=np.float32)
        idx = min(self._node, self.obs_dim - 1)  # safe-guard
        x[idx] = 1.0
        return x

    def step(self, action: int):
        self._t += 1
        reward = self.r_step
        terminated = False
        truncated = False

        # popup masking: if node has popup, sometimes force a popup
        if self._node in self.flow.popup_nodes and self.rng.random() < 0.25:
            # only "dismiss" (say action==5) avoids side-effect; others waste a step
            if action == 5:
                pass  # dismissed; stay on same node
            else:
                # waste turn or random detour to previous node if any
                reward += self.r_step
            obs = self._obs()
            if self._t >= self.max_steps:
                truncated = True
                reward += self.r_failure
            return obs, reward, terminated, truncated, {"node_id": self._node, "popup": True}

        # hidden dependency: toggling flag if at flag_node and action==0 (e.g., ClickA)
        if self.flow.hidden_rules:
            if self._node == self.flow.hidden_rules.get("flag_node") and action == 0:
                self._flag = True

        # map action index to an outgoing edge (abstract: choose k-th outgoing, else no-op)
        outs = list(self.flow.g.successors(self._node))
        if self._node in self.flow.dead_ends:
            # dead end = failure
            terminated = True
            reward += self.r_failure
            return self._obs(), reward, terminated, truncated, {"node_id": self._node, "dead_end": True}

        if len(outs) == 0:
            # stuck (also failure)
            terminated = True
            reward += self.r_failure
            return self._obs(), reward, terminated, truncated, {"node_id": self._node, "stuck": True}

        # hidden dependency effect: if flag set and affected_node is reachable, bias transition
        if self.flow.hidden_rules and self._flag:
            affected = self.flow.hidden_rules.get("affected_node")
            if affected in outs and self.rng.random() < 0.7:
                self._node = affected
            else:
                k = action % len(outs)
                self._node = outs[k]
        else:
            k = action % len(outs)
            self._node = outs[k]

        # success?
        if self._node == self.flow.goal:
            terminated = True
            reward += self.r_success

        if self._t >= self.max_steps and not terminated:
            truncated = True
            reward += self.r_failure

        return self._obs(), reward, terminated, truncated, {"node_id": self._node}
