from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RolloutResult:
    completed: bool
    steps_to_goal: Optional[int]
    time_to_goal: Optional[float]
    total_steps: int
    total_time: float
    # --- optional diagnostics (rollout-level) ---
    ideal_mean_init: Optional[float] = None
    ideal_min_init: Optional[float] = None


def run_rollouts(
    env,
    select_action: Callable[[Dict[str, Any]], Any],
    n_rollouts: int,
    max_steps: int,
    seed: Optional[int] = None,
) -> List[RolloutResult]:
    """
    Run evaluation rollouts under a fixed (frozen) policy.

    Assumptions:
      - env.reset(seed=...) returns (obs, info)
      - env.step(action) returns (obs, reward, terminated, truncated, info)
      - info contains "time_cost" (float) per step (as in your env)
    """
    rng = np.random.default_rng(seed)
    results: List[RolloutResult] = []

    for i in range(n_rollouts):
        # Use different seeds per rollout for stochastic env / world sampling
        rollout_seed = None if seed is None else int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=rollout_seed)

        ideal_mean_init = None
        ideal_min_init = None

        p_init = info.get("eol_p_success_ideal_init", None)
        if p_init is not None:
            p_init = np.asarray(p_init, dtype=float).reshape(-1)
            finite = p_init[np.isfinite(p_init)]
            if finite.size > 0:
                ideal_mean_init = float(np.mean(finite))
                ideal_min_init = float(np.min(finite))

        total_time = 0.0
        steps_to_goal: Optional[int] = None
        time_to_goal: Optional[float] = None
        completed = False

        steps_executed = 0
        for _ in range(max_steps):
            action = select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_executed += 1

            step_time = float(info.get("time_cost", 0.0))
            total_time += step_time

            if terminated:
                completed = True
                steps_to_goal = steps_executed
                time_to_goal = total_time
                break

            if truncated:
                break

        results.append(
            RolloutResult(
                completed=completed,
                steps_to_goal=steps_to_goal,
                time_to_goal=time_to_goal,
                total_steps=steps_executed,
                total_time=total_time,
                ideal_mean_init=ideal_mean_init,
                ideal_min_init=ideal_min_init,
            )
        )

    return results


def cdf_on_grid(
    times: np.ndarray,
    *,
    t_max: float,
    dt: float,
    smooth_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).reshape(-1)

    if dt <= 0:
        raise ValueError("dt must be > 0")

    t_grid = np.arange(0.0, float(t_max) + 1e-12, float(dt))

    # Vectorized CDF on grid
    y = (times[:, None] <= t_grid[None, :]).mean(axis=0).astype(float)

    # Optional smoothing + enforce monotone
    if smooth_window and smooth_window > 1:
        sw = int(smooth_window)
        if sw % 2 == 0:
            sw += 1  # prefer odd window for symmetric smoothing

        kernel = np.ones(sw, dtype=float) / float(sw)
        y = np.convolve(y, kernel, mode="same")
        y = np.clip(y, 0.0, 1.0)

    y = np.maximum.accumulate(y)
    return t_grid, y


def build_goal_times(
    results: List[RolloutResult],
    axis: str = "time",
    treat_incomplete_as_inf: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if axis not in ("steps", "time"):
        raise ValueError("axis must be 'steps' or 'time'")

    vals: List[float] = []
    n = len(results)
    n_done = sum(1 for r in results if r.completed)

    for r in results:
        x = r.steps_to_goal if axis == "steps" else r.time_to_goal
        if x is None:
            vals.append(float("inf") if treat_incomplete_as_inf else float("nan"))
        else:
            vals.append(float(x))

    arr = np.asarray(vals, dtype=float)

    finite = arr[np.isfinite(arr)]
    stats = {
        "n_rollouts": float(n),
        "n_completed": float(n_done),
        "completion_rate": float(n_done / n) if n > 0 else 0.0,
        "median_completed": float(np.median(finite))
        if finite.size > 0
        else float("nan"),
        "p90_completed": float(np.quantile(finite, 0.90))
        if finite.size > 0
        else float("nan"),
        "max_completed": float(np.max(finite)) if finite.size > 0 else float("nan"),
    }
    return arr, stats


def success_tmax(times: np.ndarray, *, fallback: float = 1.0) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        return float(fallback)
    return float(np.max(finite))


def plot_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "P(T_goal ≤ t)",
    save_path: Optional[str] = None,
) -> None:
    plt.figure()
    if x.size > 0:
        plt.plot(x, y)  # continuous-looking curve
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_rollout_results_json(results: List[RolloutResult], path: str) -> None:
    payload = [
        {
            "completed": r.completed,
            "steps_to_goal": r.steps_to_goal,
            "time_to_goal": r.time_to_goal,
            "total_steps": r.total_steps,
            "total_time": r.total_time,
            "ideal_mean_init": r.ideal_mean_init,
            "ideal_min_init": r.ideal_min_init,
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
