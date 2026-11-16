# q_learning_agent.py
"""
Simple tabular Q-learning agent for DisassemblyPOMDPEnv.

Assumptions about the env:
- In the future, env.observation_space may be a Dict with:
    obs["parts"]: shape (num_parts,), 0/1 existence vector
    obs["tool"]:  shape (num_tools,), one-hot of current mounted tool
- For Phase 1, env still returns a simple parts-existence vector.
  We fall back to treating the full obs as "parts" and tool_index = 0.

Env requirements:
- env._obs_part_names: list of part names
- env._tool_names:     list of tool names (first entry is "H5")
- env._cap_action_spaces: list of MultiDiscrete spaces per capability
- env.action_space: MultiDiscrete([n_caps, max_dim0, max_dim1])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TabularQLearningAgent:
    def __init__(
        self,
        env,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        rng: np.random.Generator | None = None,
    ):
        self.env = env

        # --- State encoding: (parts_bitmask, tool_index) ---
        self.num_parts: int = len(env._obs_part_names)
        self.num_part_states: int = 2**self.num_parts
        # self.num_tools: int = len(env._tool_names)
        try:
            # Dict space: observation_space["tool"] is Box(shape=(n_tools,))
            self.num_tools = int(env.observation_space["tool"].shape[0])
        except Exception:
            # fallback to old env API
            self.num_tools = int(len(getattr(env, "_tool_names", ["H5"])))

        # --- EOL summary (V2, probability buckets) ---
        self.use_eol = bool(getattr(env, "expose_eol_belief", False))
        self.n_prob_buckets = 2

        if self.use_eol:
            # combine (stuck_bucket, missing_bucket) -> 2*2 = 4 summaries
            self.n_eol_summary = self.n_prob_buckets * self.n_prob_buckets
        else:
            self.n_eol_summary = 1

        # --- Action space: MultiDiscrete([n_caps, max_dim0, max_dim1]) ---
        n_caps, max_dim0, max_dim1 = env.action_space.nvec
        self.n_caps = int(n_caps)
        self.max_dim0 = int(max_dim0)
        self.max_dim1 = int(max_dim1)

        # Local dims per capability, used to avoid invalid actions
        self.local_dims: List[tuple[int, int]] = [
            (int(sp.nvec[0]), int(sp.nvec[1])) for sp in env._cap_action_spaces
        ]

        # --- Q-table: [parts_state, tool_idx, cap_idx, sub0, sub1] ---
        self.Q = np.zeros(
            (
                self.num_part_states,
                self.num_tools,
                self.n_eol_summary,
                self.n_caps,
                self.max_dim0,
                self.max_dim1,
            ),
            dtype=float,
        )

        # Convenience summary (for logging / saving)
        self.n_states = self.num_part_states * self.num_tools * self.n_eol_summary
        self.n_actions = self.n_caps * self.max_dim0 * self.max_dim1

        # Learning hyperparameters
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # Epsilon schedule
        self.eps = float(eps_start)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def q_table(self) -> np.ndarray:
        """Alias to make saving easier."""
        return self.Q

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------
    def _bucket_prob(self, p: float) -> int:
        return 1 if float(p) >= 0.5 else 0

    def encode_state(self, obs) -> Tuple[int, int, int]:
        """
        Encode observation into a discrete state:
            (parts_mask, tool_index, eol_summary_index)
        """
        if isinstance(obs, dict):
            parts_vec = np.asarray(obs["parts"], dtype=float)
            tool_vec = np.asarray(obs["tool"], dtype=float)
        else:
            parts_vec = np.asarray(obs, dtype=float)
            tool_vec = np.zeros(self.num_tools, dtype=float)
            tool_vec[0] = 1.0  # default tool (index 0)

        # parts existence -> bitmask
        mask = 0
        parts_vec = parts_vec[: self.num_parts]
        for i, v in enumerate(parts_vec):
            if v > 0.5:
                mask |= 1 << i

        # one-hot -> index
        tool_idx = int(np.argmax(tool_vec))
        if not (0 <= tool_idx < self.num_tools):
            tool_idx = 0

        # --- EOL compressed summary (V2) ---
        eol_idx = 0
        if not self.use_eol:
            return mask, tool_idx, 0

        if isinstance(obs, dict) and ("eol" in obs):
            e = obs["eol"]
            p_stuck_vec = None
            p_missing_vec = None

            if isinstance(e, dict):
                if "p_stuck" in e:
                    p_stuck_vec = np.asarray(e["p_stuck"], dtype=float).reshape(-1)
                if "p_missing" in e:
                    p_missing_vec = np.asarray(e["p_missing"], dtype=float).reshape(-1)

            stuck_bucket = 0
            missing_bucket = 0

            if p_stuck_vec is not None and p_stuck_vec.size > 0:
                stuck_bucket = self._bucket_prob(float(np.max(p_stuck_vec)))
            if p_missing_vec is not None and p_missing_vec.size > 0:
                missing_bucket = self._bucket_prob(float(np.max(p_missing_vec)))

            # pack into [0..3]
            eol_idx = stuck_bucket * self.n_prob_buckets + missing_bucket

        return mask, tool_idx, eol_idx

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _get_tref(self, cap, sub0: int, sub1: int):
        """Decode raw (sub0, sub1) into a TargetRef (or None)."""
        try:
            return cap.get_target_ref(
                self.env.assembly, np.array([sub0, sub1], dtype=np.int64)
            )
        except Exception:
            return None

    def _is_valid_target(self, cap, tref) -> bool:
        """
        True iff this target should be considered by the agent.
        Filters:
          - buffer_slot
          - target_missing (if cap supports is_target_present)
        """
        if tref is None:
            # Conservative: treat decode-fail as invalid.
            # If you prefer to allow "no-target" actions, return True here.
            return False

        if getattr(tref, "kind", None) == "buffer_slot":
            return False

        if hasattr(cap, "is_target_present"):
            try:
                if not cap.is_target_present(self.env.assembly, tref):
                    return False
            except Exception:
                # If present-check is flaky, be conservative and treat invalid.
                return False

        return True

    def _epsilon_greedy_action(
        self, state: Tuple[int, int, int], explore: bool
    ) -> np.ndarray:
        """
        Epsilon-greedy action selection in (cap_idx, sub0, sub1) space.
        """
        parts_idx, tool_idx, eol_idx = state

        # Exploration
        if explore and (self.rng.random() < self.eps):
            # sample valid-ish action (avoid buffer_slot)
            for _ in range(50):
                cap_idx = int(self.rng.integers(0, self.n_caps))
                dim0, dim1 = self.local_dims[cap_idx]
                sub0 = int(self.rng.integers(0, dim0))
                sub1 = int(self.rng.integers(0, dim1))

                cap = self.env.capabilities[cap_idx]
                tref = self._get_tref(cap, sub0, sub1)
                if not self._is_valid_target(cap, tref):
                    continue
                return np.array([cap_idx, sub0, sub1], dtype=int)

            # fallback
            cap_idx = int(self.rng.integers(0, self.n_caps))
            dim0, dim1 = self.local_dims[cap_idx]
            sub0 = int(self.rng.integers(0, dim0))
            sub1 = int(self.rng.integers(0, dim1))
            return np.array([cap_idx, sub0, sub1], dtype=int)

        # Exploitation
        best_q = -np.inf
        best_action = None

        for cap_idx in range(self.n_caps):
            dim0, dim1 = self.local_dims[cap_idx]
            cap = self.env.capabilities[cap_idx]
            for sub0 in range(dim0):
                for sub1 in range(dim1):
                    tref = self._get_tref(cap, sub0, sub1)
                    if not self._is_valid_target(cap, tref):
                        continue
                    q_val = self.Q[parts_idx, tool_idx, eol_idx, cap_idx, sub0, sub1]
                    if q_val > best_q:
                        best_q = q_val
                        best_action = (cap_idx, sub0, sub1)

        if best_action is None:
            # Corner case: all Q equal
            cap_idx = int(self.rng.integers(0, self.n_caps))
            dim0, dim1 = self.local_dims[cap_idx]
            sub0 = int(self.rng.integers(0, dim0))
            sub1 = int(self.rng.integers(0, dim1))
            best_action = (cap_idx, sub0, sub1)

        cap_idx, sub0, sub1 = best_action
        return np.array([cap_idx, sub0, sub1], dtype=int)

    def select_action(self, obs, explore: bool = True) -> np.ndarray:
        """Public interface: given an observation, choose an action."""
        state = self.encode_state(obs)
        return self._epsilon_greedy_action(state, explore=explore)

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------
    def _max_next_q(self, next_state: Tuple[int, int, int]) -> float:
        """
        Compute max_a' Q(next_state, a') over all legal actions.
        """
        parts_idx, tool_idx, eol_idx = next_state
        best_next_q = -np.inf

        for cap_idx in range(self.n_caps):
            dim0, dim1 = self.local_dims[cap_idx]
            cap = self.env.capabilities[cap_idx]
            for sub0 in range(dim0):
                for sub1 in range(dim1):
                    tref = self._get_tref(cap, sub0, sub1)
                    if not self._is_valid_target(cap, tref):
                        continue
                    q_val = self.Q[parts_idx, tool_idx, eol_idx, cap_idx, sub0, sub1]
                    if q_val > best_next_q:
                        best_next_q = q_val

        return 0.0 if best_next_q == -np.inf else float(best_next_q)

    def update(
        self,
        obs,
        action: np.ndarray,
        reward: float,
        next_obs,
        done: bool,
        info: dict | None = None,
    ) -> None:
        """One Q-learning update."""
        parts_idx, tool_idx, eol_idx = self.encode_state(obs)
        next_parts_idx, next_tool_idx, next_eol_idx = self.encode_state(next_obs)
        cap_idx, sub0, sub1 = map(int, action)
        if not (0 <= cap_idx < self.n_caps):
            return
        dim0, dim1 = self.local_dims[cap_idx]
        if not (0 <= sub0 < dim0 and 0 <= sub1 < dim1):
            return

        old_q = self.Q[parts_idx, tool_idx, eol_idx, cap_idx, sub0, sub1]

        # ---- NEW: bootstrap gating ----
        # cap = self.env.capabilities[cap_idx]
        # tref = self._get_tref(cap, sub0, sub1)
        # valid = self._is_valid_target(cap, tref)
        # illegal = bool(info.get("illegal_action", False)) if isinstance(info, dict) else False
        # exec_gate = (info.get("failure_stage") == "exec_gate") if isinstance(info, dict) else False
        illegal = (
            bool(info.get("illegal_action", False)) if isinstance(info, dict) else False
        )
        exec_gate = (
            (info.get("failure_stage") == "exec_gate")
            if isinstance(info, dict)
            else False
        )

        # if done or illegal or exec_gate or (not valid):
        if done or illegal or exec_gate:
            target = float(reward)
        else:
            best_next_q = self._max_next_q(
                (next_parts_idx, next_tool_idx, next_eol_idx)
            )
            target = float(reward) + self.gamma * float(best_next_q)

        self.Q[parts_idx, tool_idx, eol_idx, cap_idx, sub0, sub1] = (
            old_q + self.alpha * (target - old_q)
        )

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Exponential decay for epsilon, clipped at eps_end."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ------------------------------------------------------------------
    # High-level training loop
    # ------------------------------------------------------------------
    def train(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int | None = None,
        print_every: int = 50,
    ) -> list[float]:
        """
        Train the agent on the attached env using tabular Q-learning.

        Returns:
            List of episodic returns.
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = getattr(self.env, "max_steps", 50)

        episode_returns: list[float] = []

        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            done = False
            ep_return = 0.0

            for t in range(max_steps_per_episode):
                action = self.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, step_info = self.env.step(
                    action
                )

                done = bool(terminated or truncated)

                self.update(obs, action, reward, next_obs, done, info=step_info)

                ep_return += reward
                obs = next_obs

                if done:
                    break

            episode_returns.append(ep_return)
            self.decay_epsilon()

            if print_every is not None and (ep + 1) % print_every == 0:
                avg_ret = float(np.mean(episode_returns[-print_every:]))
                print(
                    f"[Episode {ep + 1}/{num_episodes}] "
                    f"avg_return(last {print_every}) = {avg_ret:.2f}, "
                    f"eps = {self.eps:.3f}"
                )

        return episode_returns

    def set_eval_mode(self, enabled: bool = True) -> None:
        """
        Evaluation mode means deterministic greedy action selection.
        We implement it by forcing eps=0 while preserving the training eps.
        """
        if enabled:
            if not hasattr(self, "_eps_backup"):
                self._eps_backup = float(self.eps)
            self.eps = 0.0
        else:
            if hasattr(self, "_eps_backup"):
                self.eps = float(self._eps_backup)
                delattr(self, "_eps_backup")

    def save_checkpoint(
        self, path: str | Path, *, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a training checkpoint:
          - Q-table
          - epsilon and hyperparams (so you can resume)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        np.save(p.with_suffix(".q.npy"), self.Q)

        meta = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps": float(self.eps),
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "use_eol": bool(self.use_eol),
            "n_prob_buckets": int(self.n_prob_buckets),
            "n_eol_summary": int(self.n_eol_summary),
            "shape_Q": list(self.Q.shape),
        }
        if extra:
            meta.update(extra)

        with open(p.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load checkpoint saved by save_checkpoint().
        """
        p = Path(path)
        q_path = p.with_suffix(".q.npy")
        meta_path = p.with_suffix(".meta.json")

        Q = np.load(q_path)
        if Q.shape != self.Q.shape:
            raise ValueError(
                f"Q shape mismatch: ckpt {Q.shape} vs agent {self.Q.shape}"
            )
        self.Q[...] = Q

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # restoring eps is optional; for eval you’ll set eps=0 anyway
            self.eps = float(meta.get("eps", self.eps))

    # ------------------------------------------------------------------
    # Greedy evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        num_episodes: int = 1,
        max_steps_per_episode: int | None = None,
        render: bool = False,
    ) -> None:
        """
        Run greedy episodes using the learned Q-table and print transitions.
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = getattr(self.env, "max_steps", 50)

        old_eps = self.eps
        self.eps = 0.0  # pure greedy

        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=int(self.rng.integers(0, 1_000_000)))
            done = False
            total_reward = 0.0

            print(f"\n=== Greedy rollout #{ep + 1} ===")
            for t in range(max_steps_per_episode):
                action = self.select_action(obs, explore=False)
                next_obs, reward, terminated, truncated, step_info = self.env.step(
                    action
                )
                done = bool(terminated or truncated)

                total_reward += reward

                parts = [p.name for p in self.env.assembly.parts.values()]
                tool = getattr(self.env, "current_mounted_tool", None)
                print(
                    f"t={t:02d}, action={action.tolist()}, "
                    f"reward={reward:.2f}, terminated={terminated}, truncated={truncated}, "
                    f"parts={parts}, tool={tool}"
                )

                if render:
                    try:
                        _ = self.env.render()
                    except Exception:
                        pass

                obs = next_obs

                if done:
                    break

            print(f"Total reward: {total_reward:.2f}")

        self.eps = old_eps
