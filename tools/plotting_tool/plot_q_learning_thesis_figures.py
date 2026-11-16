from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# KIT / wbk visual style
# -----------------------------------------------------------------------------

KIT_DEEP_BLUE = "#144466"
KIT_BLUE_GREEN = "#00838f"
KIT_YELLOW = "#EEB70D"
KIT_RED = "#B2372C"
KIT_GREY = "#D9DEE2"
DARK = "#222222"
MUTED = "#666666"
LIGHT_GRID = "#E5E8EB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.color": LIGHT_GRID,
        "grid.linewidth": 0.8,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values.copy()

    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")

    return np.convolve(padded, kernel, mode="valid")


def save_figure(
    fig: plt.Figure, out_dir: Path, stem: str, formats: tuple[str, ...]
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)

    plt.close(fig)
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_existing_key(row: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in row:
            return row[key]
    return None


# -----------------------------------------------------------------------------
# Reward and success curves
# -----------------------------------------------------------------------------


def plot_training_reward_curve(
    run_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
    reward_window: int,
    success_window: int,
) -> list[Path]:
    rewards_path = run_dir / "rewards.npy"
    success_path = run_dir / "success.npy"
    lengths_path = run_dir / "episode_lengths.npy"
    summaries_path = run_dir / "episode_summaries.jsonl"

    if not rewards_path.exists():
        raise FileNotFoundError(f"Missing rewards.npy: {rewards_path}")

    rewards = np.load(rewards_path).astype(float)
    episodes = np.arange(1, rewards.size + 1)

    reward_ma = moving_average(rewards, reward_window)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(
        episodes,
        rewards,
        color=KIT_GREY,
        linewidth=0.6,
        alpha=0.55,
        label="episode return",
    )
    ax.plot(
        episodes,
        reward_ma,
        color=KIT_DEEP_BLUE,
        linewidth=1.8,
        label=f"moving average ({reward_window} episodes)",
    )

    ax.set_title("Q-learning training return", fontweight="bold", pad=6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_xlim(1, max(1, rewards.size))
    ax.legend(frameon=False, loc="best")

    generated = save_figure(fig, out_dir, "q_learning_reward_curve", formats)

    summary: dict[str, Any] = {
        "source_rewards_npy": str(rewards_path),
        "num_episodes": int(rewards.size),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "reward_last": float(rewards[-1]),
        "reward_moving_average_window": int(reward_window),
    }

    if success_path.exists():
        success = np.load(success_path).astype(float)
        success_ma = moving_average(success, success_window)

        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.plot(
            np.arange(1, success.size + 1),
            success_ma,
            color=KIT_BLUE_GREEN,
            linewidth=1.8,
            label=f"moving success rate ({success_window} episodes)",
        )

        ax.set_title("Q-learning training success trend", fontweight="bold", pad=6)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success rate")
        ax.set_xlim(1, max(1, success.size))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, loc="best")

        generated.extend(save_figure(fig, out_dir, "q_learning_success_curve", formats))

        summary.update(
            {
                "source_success_npy": str(success_path),
                "success_mean": float(np.mean(success)),
                "success_last": float(success[-1]),
                "success_moving_average_window": int(success_window),
            }
        )

    if lengths_path.exists():
        lengths = np.load(lengths_path).astype(float)
        summary.update(
            {
                "source_episode_lengths_npy": str(lengths_path),
                "episode_length_mean": float(np.mean(lengths)),
                "episode_length_median": float(np.median(lengths)),
                "episode_length_p90": float(np.percentile(lengths, 90)),
            }
        )

    if summaries_path.exists():
        summary["source_episode_summaries_jsonl"] = str(summaries_path)
        try:
            rows = read_jsonl(summaries_path)
            summary["episode_summaries_rows"] = len(rows)
        except Exception as exc:
            summary["episode_summaries_read_error"] = str(exc)

    summary_path = out_dir / "q_learning_training_curves_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated.append(summary_path)

    return generated


# -----------------------------------------------------------------------------
# Rollout CDF curves
# -----------------------------------------------------------------------------


def extract_rollout_arrays(
    rollouts_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = json.loads(rollouts_path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            rows = data["results"]
        elif "rollouts" in data and isinstance(data["rollouts"], list):
            rows = data["rollouts"]
        else:
            raise ValueError(
                f"Could not find a rollout list in {rollouts_path}. "
                f"Top-level keys: {list(data.keys())}"
            )
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported rollouts.json structure: {type(data)}")

    success_values: list[bool] = []
    step_values: list[float] = []
    time_values: list[float] = []

    debug_missing: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        success_raw = first_existing_key(
            row,
            [
                "success",
                "completed",
                "goal_reached",
                "done_success",
                "is_success",
            ],
        )

        if success_raw is None:
            status = row.get("status")
            success = bool(status in ("success", "completed", "goal_reached"))
        else:
            success = bool(success_raw)

        steps_raw = first_existing_key(
            row,
            [
                "steps",
                "num_steps",
                "n_steps",
                "steps_to_goal",
                "episode_steps",
                "length",
            ],
        )

        time_raw = first_existing_key(
            row,
            [
                "time",
                "total_time",
                "time_to_goal",
                "episode_time",
                "completed_time",
                "sum_time_cost",
                "return_time",
            ],
        )

        if steps_raw is None:
            debug_missing.append({"missing": "steps", "keys": list(row.keys())})
            continue

        steps = float(steps_raw)

        if time_raw is None:
            # Keep time as NaN if the JSON does not store it.
            time = float("nan")
        else:
            time = float(time_raw)

        success_values.append(success)
        step_values.append(steps if success else float("inf"))
        time_values.append(time if success else float("inf"))

    success_arr = np.asarray(success_values, dtype=bool)
    steps_arr = np.asarray(step_values, dtype=float)
    time_arr = np.asarray(time_values, dtype=float)

    meta = {
        "source_rollouts_json": str(rollouts_path),
        "num_rows_raw": len(rows),
        "num_rows_used": int(success_arr.size),
        "num_success": int(np.sum(success_arr)),
        "completion_rate": float(np.mean(success_arr)) if success_arr.size else 0.0,
        "debug_missing_first_entries": debug_missing[:5],
    }

    return success_arr, steps_arr, time_arr, meta


def cdf_on_grid(
    values: np.ndarray, dt: float, fallback_tmax: float
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        t_max = float(fallback_tmax)
    else:
        t_max = float(np.max(finite))

    t_max = max(t_max, dt)
    x = np.arange(0.0, t_max + 0.5 * dt, dt)
    y = np.asarray([np.mean(values <= xx) for xx in x], dtype=float)

    return x, y


def plot_rollout_cdf_curves(
    eval_dir: Path,
    out_dir: Path,
    formats: tuple[str, ...],
    dt_steps: float,
    dt_time: float,
) -> list[Path]:
    rollouts_path = eval_dir / "rollouts.json"
    cdf_stats_path = eval_dir / "cdf_stats.json"

    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing rollouts.json: {rollouts_path}")

    success_arr, steps_arr, time_arr, meta = extract_rollout_arrays(rollouts_path)

    generated: list[Path] = []

    # Steps CDF
    x_steps, y_steps = cdf_on_grid(steps_arr, dt=dt_steps, fallback_tmax=50.0)

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(x_steps, y_steps, color=KIT_DEEP_BLUE, linewidth=2.0)
    ax.set_title("Final-policy rollout completion CDF", fontweight="bold", pad=6)
    ax.set_xlabel("Steps to completion")
    ax.set_ylabel("Cumulative completion probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(x_steps[-1], 1.0))

    generated.extend(save_figure(fig, out_dir, "q_learning_rollout_cdf_steps", formats))

    # Time CDF, only if time is available for at least one successful rollout.
    finite_time = time_arr[np.isfinite(time_arr)]
    if finite_time.size > 0 and not np.all(np.isnan(finite_time)):
        valid_time_arr = time_arr.copy()
        valid_time_arr[np.isnan(valid_time_arr)] = np.inf

        x_time, y_time = cdf_on_grid(valid_time_arr, dt=dt_time, fallback_tmax=100.0)

        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        ax.plot(x_time, y_time, color=KIT_BLUE_GREEN, linewidth=2.0)
        ax.set_title("Final-policy rollout completion CDF", fontweight="bold", pad=6)
        ax.set_xlabel("Time to completion")
        ax.set_ylabel("Cumulative completion probability")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max(x_time[-1], 1.0))

        generated.extend(
            save_figure(fig, out_dir, "q_learning_rollout_cdf_time", formats)
        )
    else:
        meta["time_cdf_skipped"] = "No valid time field found in rollouts.json."

    if cdf_stats_path.exists():
        try:
            meta["source_cdf_stats_json"] = str(cdf_stats_path)
            meta["cdf_stats"] = json.loads(cdf_stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            meta["cdf_stats_read_error"] = str(exc)

    summary_path = out_dir / "q_learning_rollout_cdf_source_summary.json"
    summary_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    generated.append(summary_path)

    return generated


# -----------------------------------------------------------------------------
# Source package helper
# -----------------------------------------------------------------------------


def copy_traceability_files(
    run_dir: Path, eval_dir: Path, source_dir: Path, script_path: Path | None
) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        run_dir / "config.json",
        run_dir / "eol_summary.json",
        run_dir / "episode_summaries.jsonl",
        run_dir / "rewards.npy",
        run_dir / "success.npy",
        run_dir / "episode_lengths.npy",
        eval_dir / "cdf_stats.json",
        eval_dir / "rollouts.json",
    ]

    copied: list[Path] = []

    for src in files_to_copy:
        if src.exists():
            dst = source_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    if script_path is not None and script_path.exists():
        dst = source_dir / script_path.name
        shutil.copy2(script_path, dst)
        copied.append(dst)

    readme = source_dir / "README_source.txt"
    readme.write_text(
        "Q-learning thesis figure source package\n\n"
        "The figures in this folder were generated from the stored Q-learning\n"
        "training arrays and final-policy rollout logs.\n\n"
        "Main inputs:\n"
        "- rewards.npy: per-episode training return\n"
        "- success.npy: per-episode success indicator\n"
        "- episode_lengths.npy: per-episode length\n"
        "- rollouts.json: final-policy rollout records\n"
        "- cdf_stats.json: rollout CDF summary statistics\n\n"
        "Generated outputs are stored as PNG, PDF, and SVG for thesis use,\n"
        "printing, and editable/vector post-processing.\n",
        encoding="utf-8",
    )
    copied.append(readme)

    return copied


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render thesis-style Q-learning reward and rollout CDF figures."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath(
            "Bau_Yuan/05_Supplementary_Materials/"
            "01_Experiment_Results/01_Experiment_6.1/q_learning_results"
        ),
        help="Q-learning result directory.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Evaluation directory containing rollouts.json and cdf_stats.json. "
        "Default: newest eval_final_* directory inside run-dir.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__)
        .resolve()
        .parents[3]
        .joinpath("Bau_Yuan/02_Figures/04_Chapter_6/01_Section_6.1"),
        help="Output root for thesis figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Output formats.",
    )
    parser.add_argument("--reward-window", type=int, default=25)
    parser.add_argument("--success-window", type=int, default=25)
    parser.add_argument("--dt-steps", type=float, default=1.0)
    parser.add_argument("--dt-time", type=float, default=1.0)
    parser.add_argument("--copy-source", action="store_true")

    return parser.parse_args()


def find_default_eval_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("eval_final_*"))
    if not candidates:
        raise FileNotFoundError(f"No eval_final_* directory found in {run_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    eval_dir = (
        args.eval_dir.expanduser().resolve()
        if args.eval_dir is not None
        else find_default_eval_dir(run_dir)
    )
    if not eval_dir.exists():
        raise FileNotFoundError(f"eval-dir not found: {eval_dir}")

    out_root = args.out.expanduser().resolve()
    formats = tuple(args.formats)

    reward_out = out_root / "q_learning_reward_curve"
    cdf_out = out_root / "q_learning_rollout_cdf"
    source_out = out_root / "q_learning_sources"

    generated: list[Path] = []

    generated.extend(
        plot_training_reward_curve(
            run_dir=run_dir,
            out_dir=reward_out,
            formats=formats,
            reward_window=int(args.reward_window),
            success_window=int(args.success_window),
        )
    )

    generated.extend(
        plot_rollout_cdf_curves(
            eval_dir=eval_dir,
            out_dir=cdf_out,
            formats=formats,
            dt_steps=float(args.dt_steps),
            dt_time=float(args.dt_time),
        )
    )

    if args.copy_source:
        script_path = Path(__file__).resolve()
        generated.extend(
            copy_traceability_files(
                run_dir=run_dir,
                eval_dir=eval_dir,
                source_dir=source_out,
                script_path=script_path,
            )
        )

    print("\nGenerated files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
