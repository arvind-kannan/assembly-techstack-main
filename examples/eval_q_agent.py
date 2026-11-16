from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from build_env import build_env
from q_learning_agent import TabularQLearningAgent

from disassembly_pomdp.analysis.rollout_cdf_plotter import (
    build_goal_times,
    cdf_on_grid,
    plot_curve,
    run_rollouts,
    save_rollout_results_json,
    success_tmax,
)


def _load_q_table_into_agent(agent: TabularQLearningAgent, q_path: Path) -> None:
    Q = np.load(q_path)
    if Q.shape != agent.Q.shape:
        raise ValueError(
            "Q shape mismatch.\n"
            f"  file: {q_path}\n"
            f"  file_shape: {Q.shape}\n"
            f"  agent_shape: {agent.Q.shape}\n"
            "This usually means your eval env/tools/parts differ from the training run."
        )
    agent.Q[...] = Q


def main():
    ap = argparse.ArgumentParser()

    # New: run_dir-first interface
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory produced by trainer, e.g. runs/q_learning/<run_name>",
    )

    # Optional: still support checkpoint evaluation
    ap.add_argument(
        "--checkpoint_base",
        type=str,
        default=None,
        help=(
            "Optional checkpoint BASE name inside run_dir (no suffix), e.g.\n"
            "  checkpoint_ep00050\n"
            "If provided, we load <run_dir>/<checkpoint_base>.q.npy (+ meta if needed).\n"
            "If omitted, we load <run_dir>/q_table.npy (FINAL policy)."
        ),
    )

    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir. Default: <run_dir>/eval_final or eval_<ckpt>",
    )
    ap.add_argument("--n_rollouts", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--save_rollouts_json", action="store_true")
    ap.add_argument(
        "--dt_time",
        type=float,
        default=1.0,
        help="Time grid step for CDF (e.g. seconds).",
    )
    ap.add_argument(
        "--smooth_time",
        type=int,
        default=9,
        help="Moving-average window on time CDF grid (odd preferred).",
    )
    ap.add_argument(
        "--dt_steps", type=float, default=1.0, help="Steps grid step for CDF."
    )
    ap.add_argument(
        "--smooth_steps",
        type=int,
        default=1,
        help="Moving-average window on steps CDF grid.",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # Decide which Q to load
    if args.checkpoint_base is None:
        policy_tag = "final"
        q_path = run_dir / "q_table.npy"
        meta_path = None
        if not q_path.exists():
            raise FileNotFoundError(f"Missing final Q-table: {q_path}")
    else:
        policy_tag = args.checkpoint_base
        ckpt_base = (run_dir / args.checkpoint_base).resolve()
        q_path = ckpt_base.with_suffix(".q.npy")
        meta_path = ckpt_base.with_suffix(".meta.json")
        if not q_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint Q file: {q_path}\n"
                "You must pass checkpoint_base without suffix.\n"
                f"Got checkpoint_base={args.checkpoint_base}"
            )

    # Output dir
    if args.out is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = run_dir / f"eval_{policy_tag}_{timestamp}"
    else:
        out_dir = Path(args.out).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=False)
    # if args.out is None:
    #    out_dir = run_dir / (f"eval_{policy_tag}")
    # else:
    #    out_dir = Path(args.out).expanduser().resolve()
    # out_dir.mkdir(parents=True, exist_ok=True)

    # Build env + agent (shape must match)
    eval_env = build_env()
    agent = TabularQLearningAgent(eval_env)

    # Load Q
    _load_q_table_into_agent(agent, q_path)

    # Freeze policy
    agent.set_eval_mode(True)

    def select_action_eval(obs_):
        return agent.select_action(obs_, explore=False)

    # Rollouts
    eval_max_steps = getattr(eval_env, "max_steps", 50)
    if args.max_steps is not None:
        eval_max_steps = int(args.max_steps)

    results = run_rollouts(
        env=eval_env,
        select_action=select_action_eval,
        n_rollouts=int(args.n_rollouts),
        max_steps=int(eval_max_steps),
        seed=int(args.seed),
    )

    # Optional save rollouts
    if args.save_rollouts_json:
        save_rollout_results_json(results, str(out_dir / "rollouts.json"))

    # -------- steps curve on fixed grid (end at max successful steps) --------
    steps_arr, stats_s = build_goal_times(
        results, axis="steps", treat_incomplete_as_inf=True
    )
    steps_tmax = success_tmax(steps_arr, fallback=float(eval_max_steps))

    x_steps, y_steps = cdf_on_grid(
        steps_arr,
        t_max=steps_tmax,
        dt=float(args.dt_steps),
        smooth_window=int(args.smooth_steps),
    )

    plot_curve(
        x_steps,
        y_steps,
        title=f"Goal CDF (steps) [{policy_tag}]  (p_succ={stats_s['completion_rate']:.2f})",
        xlabel="Steps to goal",
        save_path=str(out_dir / "cdf_steps.png"),
    )

    # -------- time curve on fixed grid (end at max successful time) --------
    time_arr, stats_t = build_goal_times(
        results, axis="time", treat_incomplete_as_inf=True
    )
    time_tmax = success_tmax(time_arr, fallback=float(eval_max_steps) * 100.0)

    x_time, y_time = cdf_on_grid(
        time_arr,
        t_max=time_tmax,
        dt=float(args.dt_time),
        smooth_window=int(args.smooth_time),
    )

    plot_curve(
        x_time,
        y_time,
        title=f"Goal CDF (time) [{policy_tag}]  (p_succ={stats_t['completion_rate']:.2f})",
        xlabel="Time to goal (sum time_cost)",
        save_path=str(out_dir / "cdf_time.png"),
    )

    # Save stats
    payload = {
        "run_dir": str(run_dir),
        "policy_tag": policy_tag,
        "q_path": str(q_path),
        "meta_path": str(meta_path)
        if meta_path is not None and meta_path.exists()
        else None,
        "n_rollouts": int(args.n_rollouts),
        "max_steps": int(eval_max_steps),
        "seed": int(args.seed),
        "steps": stats_s,
        "time": stats_t,
    }
    with open(out_dir / "cdf_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"[EVAL] policy={policy_tag} n_rollouts={args.n_rollouts} max_steps={eval_max_steps} seed={args.seed}\n"
        f"  completion_rate={stats_s['completion_rate']:.3f}\n"
        f"  median_steps={stats_s['median_completed']:.2f}, p90_steps={stats_s['p90_completed']:.2f}\n"
        f"  median_time={stats_t['median_completed']:.2f}, p90_time={stats_t['p90_completed']:.2f}\n"
        f"Outputs: {out_dir}"
    )


if __name__ == "__main__":
    main()
