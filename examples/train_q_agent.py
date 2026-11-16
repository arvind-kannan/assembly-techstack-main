# train_q_agent.py
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from build_env import build_env
from q_learning_agent import TabularQLearningAgent


def _jsonl_write(f, obj, *, flush: bool = False) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if flush:
        f.flush()


def _infer_tool_names(env) -> list[str] | None:
    """
    Best-effort tool name list for logging/config.
    Now that observer owns vocab, prefer env.observer._vocab.tool_names.
    """
    # 0) Source of truth: observer vocab
    obs = getattr(env, "observer", None)
    vocab = getattr(obs, "_vocab", None) if obs is not None else None
    names = getattr(vocab, "tool_names", None) if vocab is not None else None
    if (
        isinstance(names, (list, tuple))
        and names
        and all(isinstance(x, str) for x in names)
    ):
        return list(names)

    # 1) Back-compat: env stored names
    names = getattr(env, "_tool_names", None)
    if (
        isinstance(names, (list, tuple))
        and names
        and all(isinstance(x, str) for x in names)
    ):
        return list(names)

    names = getattr(env, "tool_names", None)
    if (
        isinstance(names, (list, tuple))
        and names
        and all(isinstance(x, str) for x in names)
    ):
        return list(names)

    # 2) Derive from capabilities (still ok, but should match observer)
    caps = getattr(env, "capabilities", None)
    if isinstance(caps, list) and caps:
        collected: list[str] = []
        seen = set()

        for cap in caps:
            # list_tools()
            list_tools_fn = getattr(cap, "list_tools", None)
            if callable(list_tools_fn):
                try:
                    tools = list_tools_fn()
                    if isinstance(tools, (list, tuple)):
                        for t in tools:
                            if isinstance(t, str) and t not in seen:
                                seen.add(t)
                                collected.append(t)
                except Exception:
                    pass

            # common attributes
            for attr in (
                "available_screwdrivers",
                "available_grippers",
                "available_milling_tools",
            ):
                tools = getattr(cap, attr, None)
                if isinstance(tools, (list, tuple)):
                    for t in tools:
                        if isinstance(t, str) and t not in seen:
                            seen.add(t)
                            collected.append(t)

        if collected:
            return collected

    # 3) Last resort: dimension only
    try:
        n_tools = int(env.observation_space["tool"].shape[0])
        return [f"tool_{i}" for i in range(n_tools)]
    except Exception:
        return None


# def _infer_tool_names(env) -> list[str] | None:
#    """Best-effort tool name list for logging/config."""
#    names = getattr(env, "_tool_names", None)
#    if isinstance(names, (list, tuple)) and len(names) > 0:
#        return list(names)
#    try:
#        n_tools = int(env.observation_space["tool"].shape[0])
#        return [f"tool_{i}" for i in range(n_tools)]
#    except Exception:
#        return None


def _moving_average(x: list[float], window: int) -> np.ndarray:
    """Simple trailing moving average; returns an array aligned to x[window-1:]."""
    arr = np.asarray(x, dtype=float)
    if window <= 1 or arr.size < window:
        return np.asarray([], dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def save_training_results(
    agent: TabularQLearningAgent,
    rewards: list[float],
    successes: list[int],
    lengths: list[int],
    env,
    run_dir: Path,
) -> None:
    """Save Q-table, metrics and plots."""
    # save_short_episodes = False

    # 1. Save config
    config = {
        "n_parts": agent.num_parts,
        "num_part_states": agent.num_part_states,
        "num_tools": agent.num_tools,
        "n_states": agent.n_states,
        "n_actions": agent.n_actions,
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "eps_start": agent.eps_start,
        "eps_end": agent.eps_end,
        "eps_decay": agent.eps_decay,
        "env_class": type(env).__name__,
        "target_part_names": getattr(env, "target_part_names", None),
        "max_steps": getattr(env, "max_steps", None),
        "part_names": getattr(env, "_obs_part_names", None),
        "tool_names": _infer_tool_names(env),
        "use_eol": getattr(agent, "use_eol", None),
        "n_prob_buckets": getattr(agent, "n_prob_buckets", None),
        "n_eol_summary": getattr(agent, "n_eol_summary", None),
        "expose_eol_belief": getattr(env, "expose_eol_belief", None),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 2. Q-table
    np.save(run_dir / "q_table.npy", agent.Q)

    # 3. Metrics
    np.save(run_dir / "rewards.npy", np.asarray(rewards, dtype=float))
    np.save(run_dir / "success.npy", np.asarray(successes, dtype=int))
    np.save(run_dir / "episode_lengths.npy", np.asarray(lengths, dtype=int))

    # 4. Plot reward curve
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="episode reward", color="blue")
    # Add moving average for readability
    ma_window = 50
    ma = _moving_average(rewards, ma_window)
    if ma.size > 0:
        plt.plot(
            np.arange(ma_window - 1, ma_window - 1 + ma.size),
            ma,
            label=f"moving avg (window={ma_window})",
            color="red",
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "reward_curve.png")
    plt.close()


def main():
    # ---- long-run logging toggles ----
    LOG_STEP_TRACE = False  # per-step trace (big!)
    LOG_EOL_EVENTS = False  # per-step, but only when eol_event exists
    LOG_EPISODE_SUMMARY = True  # per-episode, small

    STEP_TRACE_EVERY = 1  # record every N steps (only if LOG_STEP_TRACE)
    STEP_TRACE_MAX_PER_EP = 50  # cap per episode (0 = unlimited)

    EOL_EVENTS_EVERY = 1  # record every N eol-events (only if LOG_EOL_EVENTS)
    EOL_EVENTS_MAX_PER_EP = 50  # 0 = unlimited

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs/q_learning") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_f = None
    eol_f = None
    summ_f = None

    env = build_env()
    agent = TabularQLearningAgent(env)

    checkpoint_every = 50

    num_episodes = 2000

    rewards: list[float] = []
    successes: list[int] = []
    lengths: list[int] = []

    eol_summary = {
        "num_eol_events": 0,
        "outcomes": {"SUCCESS": 0, "FAIL": 0},
    }

    try:
        if LOG_STEP_TRACE:
            trace_f = open(run_dir / "episode_trace.jsonl", "w", encoding="utf-8")
        if LOG_EOL_EVENTS:
            eol_f = open(run_dir / "eol_events.jsonl", "w", encoding="utf-8")
        if LOG_EPISODE_SUMMARY:
            summ_f = open(run_dir / "episode_summaries.jsonl", "w", encoding="utf-8")

        for ep in range(num_episodes):
            episode_seed = int(agent.rng.integers(0, 1_000_000))
            obs, info = env.reset(seed=episode_seed)
            prev_parts = (
                {p.name for p in env.assembly.parts.values()}
                if LOG_STEP_TRACE
                else None
            )
            terminated = truncated = False
            total_reward = 0.0
            t = 0

            step_trace_written = 0
            eol_written = 0

            episode_summary = {
                "episode": int(ep + 1),
                "seed": episode_seed,
            }

            p_init = info.get("eol_p_success_ideal_init", None)
            if p_init is not None:
                p_init = np.asarray(p_init, dtype=float).reshape(-1)
                finite = p_init[np.isfinite(p_init)]
                episode_summary["ideal_mean_init"] = (
                    float(np.mean(finite)) if finite.size else float("nan")
                )
                episode_summary["ideal_min_init"] = (
                    float(np.min(finite)) if finite.size else float("nan")
                )

            while not (terminated or truncated):
                action = agent.select_action(obs, explore=True)

                tool_before = getattr(env, "current_mounted_tool", None)

                obs_next, base_reward, terminated, truncated, info = env.step(action)

                tool_after = getattr(env, "current_mounted_tool", None)

                sem = info.get("semantic_action", None)
                cap_ev = info.get("eol_event", None)

                if dataclasses.is_dataclass(cap_ev):
                    cap_ev = dataclasses.asdict(cap_ev)
                elif cap_ev is not None and not isinstance(cap_ev, dict):
                    cap_ev = {"repr": repr(cap_ev)}

                if LOG_STEP_TRACE:
                    after_parts = {p.name for p in env.assembly.parts.values()}
                    removed_parts = sorted(prev_parts - after_parts)
                    prev_parts = after_parts
                else:
                    removed_parts = []

                rec = {
                    "episode": int(ep + 1),
                    "t": int(t),
                    "seed": episode_seed,
                    "action": [int(x) for x in action],
                    # "capability_index": int(info.get("capability_index", -1)),
                    "success": bool(info.get("success", False)),
                    "noop": bool(info.get("noop", False)),
                    "failure_stage": info.get("failure_stage"),
                    "noop_reason": info.get("noop_reason"),
                    "exec_gate_ok": info.get("exec_gate_ok"),
                    "exec_stage": info.get("exec_stage"),
                    "exec_fail_reason": info.get("exec_fail_reason"),
                    "exec_fail_details": info.get("exec_fail_details"),
                    # "terminated": bool(terminated),
                    # "truncated": bool(truncated),
                    "tool_mounted": getattr(env, "current_mounted_tool", None),
                    "tool_before": tool_before,
                    "tool_after": tool_after,
                    "removed_parts": removed_parts,
                    # "removed_any": bool(removed_parts),
                    # semantic action (minimal, but enough)
                    "semantic": {
                        "type": type(sem).__name__ if sem is not None else None,
                        "tool_name": getattr(sem, "tool_name", None),
                        "part_name": getattr(sem, "part_name", None),
                        "screw_part_name": getattr(sem, "screw_part_name", None),
                        "base_part_name": getattr(sem, "base_part_name", None),
                    },
                    # eol event if any
                    "eol_event": cap_ev,
                    "base_reward": float(base_reward),
                }
                if LOG_STEP_TRACE and trace_f is not None:
                    if (t % STEP_TRACE_EVERY) == 0:
                        if (
                            STEP_TRACE_MAX_PER_EP <= 0
                            or step_trace_written < STEP_TRACE_MAX_PER_EP
                        ):
                            _jsonl_write(trace_f, rec)
                            step_trace_written += 1

                ev = info.get("eol_event", None)
                if ev is not None:
                    if dataclasses.is_dataclass(ev):
                        ev = dataclasses.asdict(ev)
                    elif not isinstance(ev, dict):
                        ev = {"repr": repr(ev)}

                    sem = info.get("semantic_action", None)
                    sem_small = None
                    if sem is not None:
                        sem_small = {
                            "tool_name": getattr(sem, "tool_name", None),
                            "screw_part_name": getattr(sem, "screw_part_name", None),
                            "base_part_name": getattr(sem, "base_part_name", None),
                        }

                    rec = {
                        "episode": int(ep + 1),
                        "t": int(t),
                        "action": [int(x) for x in action],
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "task_terminated": bool(terminated),
                        "noop": bool(info.get("noop", False)),
                        "semantic_action": sem_small,
                        "eol_event": ev,
                        "base_reward": float(base_reward),
                    }

                    if LOG_EOL_EVENTS and eol_f is not None:
                        if (eol_written % EOL_EVENTS_EVERY) == 0:
                            if (
                                EOL_EVENTS_MAX_PER_EP <= 0
                                or eol_written < EOL_EVENTS_MAX_PER_EP
                            ):
                                _jsonl_write(eol_f, rec)
                        eol_written += 1

                    eol_summary["num_eol_events"] += 1
                    k = "SUCCESS" if ev.get("success", False) else "FAIL"
                    eol_summary["outcomes"][k] += 1

                agent.update(
                    obs,
                    action,
                    base_reward,
                    obs_next,
                    done=bool(terminated or truncated),
                    info=info,
                )

                total_reward += base_reward
                obs = obs_next
                t += 1

            episode_summary["return"] = float(total_reward)
            episode_summary["steps"] = int(t)
            episode_summary["terminated"] = bool(terminated)
            episode_summary["truncated"] = bool(truncated)

            if LOG_EPISODE_SUMMARY and summ_f is not None:
                _jsonl_write(summ_f, dict(episode_summary))

            rewards.append(total_reward)
            successes.append(1 if info.get("success", False) else 0)
            lengths.append(t)

            agent.decay_epsilon()

            if (ep + 1) % 50 == 0:
                avg_ret = float(np.mean(rewards[-50:]))
                print(
                    f"[Episode {ep + 1}/{num_episodes}] "
                    f"avg_return(last 50) = {avg_ret:.2f}, "
                    f"eps = {agent.eps:.3f}"
                )

            print(f"Episode {ep + 1} | Reward={total_reward:.1f} | Steps={t}")

            if (ep + 1) % checkpoint_every == 0:
                # ---------- 1) Save checkpoint ----------
                ckpt_base = run_dir / f"checkpoint_ep{ep + 1:05d}"
                agent.save_checkpoint(
                    ckpt_base,
                    extra={
                        "episode": int(ep + 1),
                        "avg_return_last_50": float(
                            np.mean(rewards[-checkpoint_every:])
                        ),
                    },
                )

    finally:
        save_training_results(agent, rewards, successes, lengths, env, run_dir)

        for f in (trace_f, eol_f, summ_f):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

        with open(run_dir / "eol_summary.json", "w", encoding="utf-8") as f:
            json.dump(eol_summary, f, indent=2, ensure_ascii=False)

        print(f"\n[WARN] Training interrupted or finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
