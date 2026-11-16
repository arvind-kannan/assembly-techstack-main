"""Line-balance report: text summary, JSON plan and Gantt diagram of a line plan, plus the command line.

Ported from the reporting half of Jan's line balancing (`print_plan`, `save_plan`, `plot_line_balance`,
`main`; #133, #137). The command line wires the precedence relation (`graph_precedence`), the solver
(`line_balancing`) and this report together:

    python -m disassembly_pomdp.line_balance_report --graph <export.json> --stations auto|K --goal full|target

Algorithm by Jan (`line_balance_disassembly.py`, received as a zip in 2026-10); the licence is open, ask Jan
before this code is made public outside the KIT GitLab.

Input is a sequence graph export (`graph_export.read_export`); its provenance header goes into the JSON
plan so a plan can be traced to its graph. Times are in seconds. `--time-scale` multiplies every
action_time; the factor (e.g. a sim-to-real calibration) comes from outside this repo.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from disassembly_pomdp.graph_export import read_export
from disassembly_pomdp.graph_precedence import (
    Task,
    choose_goal_state,
    extract_precedence,
)
from disassembly_pomdp.line_balancing import (
    LinePlan,
    allocate_stations,
    choose_k_salbp_e,
)

PLAN_FILE = "line_balance_plan.json"
GANTT_FILE = "line_balance_gantt.png"

# A categorical palette (light mode) whose slot order was checked for colour-vision-deficiency
# separation of neighbouring slots on the chart surface `_SURFACE` (#137). Beyond three processes not
# every pair is safe by colour alone, so each station's label also names its process. A process takes
# the next slot in line order; from the 9th process on, slots repeat with a hatch, never silently.
_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
_HATCHES = ["", "//", "\\\\", "xx", ".."]
_SURFACE, _INK, _INK_MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def scale_times(tasks: dict[str, Task], factor: float) -> dict[str, Task]:
    """The tasks with every action_time multiplied by `factor` (> 0)."""
    if not factor > 0:
        raise ValueError(f"time scale factor must be > 0, got {factor}")
    return {
        part: dataclasses.replace(t, action_time=t.action_time * factor)
        for part, t in tasks.items()
    }


def plan_report(
    plan: LinePlan, tasks: dict[str, Task], provenance: dict, **settings
) -> dict:
    """The JSON form of a plan: units in the keys, the steps of each station, the graph's provenance
    header and the settings it was made with (goal, time scale, SALBP-E sweep, ...)."""
    ct = plan.cycle_time
    return {
        "provenance": provenance,
        **settings,
        "n_stations": plan.n_stations,
        "n_stations_used": plan.n_stations_used,
        "cycle_time_s": ct,
        "cycle_time_optimal": plan.cycle_time_optimal,
        "stations_used_optimal": plan.stations_used_optimal,
        "throughput_per_h": 3600.0 / ct if ct > 0 else None,
        "total_work_s": plan.total_work,
        "utilization": plan.utilization,
        "stations": [
            {
                "station_id": s.station_id,
                "process": s.process,
                "total_time_s": s.total_time,
                "idle_time_s": ct - s.total_time,
                "utilization": s.total_time / ct if ct > 0 else 0.0,
                "steps": [
                    {
                        "target_part": p,
                        "tool": tasks[p].tool,
                        "action_time_s": tasks[p].action_time,
                    }
                    for p in s.parts
                ],
            }
            for s in plan.stations
        ],
    }


def format_summary(report: dict) -> str:
    """The plan as text: line figures, then each station with its steps."""
    tp = report["throughput_per_h"]
    lines = [
        f"=== Line balance: {report['n_stations_used']} of {report['n_stations']} stations used ===",
        f"Cycle time:           {report['cycle_time_s']:.2f} s"
        + ("" if report["cycle_time_optimal"] else " (not proven optimal)"),
        f"Throughput:           {tp:.1f} units/h"
        if tp is not None
        else "Throughput:           n/a",
        f"Total work content:   {report['total_work_s']:.2f} s",
        f"Line efficiency:      {report['utilization'] * 100:.1f}%",
    ]
    for s in report["stations"]:
        lines.append(
            f"\nStation {s['station_id']} [{s['process']}]: {s['total_time_s']:.2f} s busy, "
            f"{s['idle_time_s']:.2f} s idle ({s['utilization'] * 100:.0f}% utilized)"
        )
        lines += [
            f"    {st['target_part']:<24} tool={st['tool']!s:<16} {st['action_time_s']:.2f} s"
            for st in s["steps"]
        ]
    return "\n".join(lines)


def plot_line_balance(report: dict, path: str | Path | None = None):
    """Gantt-style diagram of a plan report: x = process time, y = station in line order, bar colour =
    process, dashed line = cycle time, utilization per station. Saved to `path` if given; returns the
    matplotlib figure. Raises ValueError for more processes than colour and hatch pairs."""
    import matplotlib.pyplot as plt  # here, so the solver and CLI import without a display
    from matplotlib.patches import Patch

    stations = report["stations"]
    processes = list(dict.fromkeys(s["process"] for s in stations))  # line order
    if len(processes) > len(_PALETTE) * len(_HATCHES):
        raise ValueError(
            f"{len(processes)} processes, but only {len(_PALETTE) * len(_HATCHES)} distinct styles"
        )
    style = {
        p: (_PALETTE[i % len(_PALETTE)], _HATCHES[i // len(_PALETTE)])
        for i, p in enumerate(processes)
    }
    ct = report["cycle_time_s"]
    n = len(stations)

    legend_rows = -(-len(processes) // 4)
    fig, ax = plt.subplots(
        figsize=(11, 0.6 * n + 1.8 + 0.3 * legend_rows), facecolor=_SURFACE
    )
    ax.set_facecolor(_SURFACE)
    ys = list(range(n))[::-1]  # first station at the top
    for y, s in zip(ys, stations):
        color, hatch = style[s["process"]]
        left = 0.0
        for step in s["steps"]:
            width = step["action_time_s"]
            ax.barh(
                y,
                width,
                left=left,
                height=0.6,
                color=color,
                hatch=hatch,
                edgecolor=_SURFACE,
                linewidth=2,
                zorder=3,
                label=s["process"],
            )
            if (
                width >= ct * 0.06
            ):  # label only segments wide enough to hold it; the JSON plan has all
                ax.text(
                    left + width / 2,
                    y,
                    step["target_part"],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=_text_on(color),
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none")
                    if hatch
                    else None,
                )
            left += width
        ax.text(
            ct * 1.02,
            y,
            f"{s['utilization'] * 100:.0f}% util",
            ha="left",
            va="center",
            fontsize=8.5,
            color=_INK_MUTED,
        )

    ax.axvline(ct, color=_INK, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(
        ct,
        n - 0.3,
        f"  cycle time = {ct:.1f} s",
        ha="left",
        va="bottom",
        fontsize=9,
        color=_INK,
        fontweight="bold",
    )
    ax.set_yticks(
        ys,
        [f"Station {s['station_id']} ({s['process']})" for s in stations],
        color=_INK,
    )
    ax.set_xlabel("Process time (s)", labelpad=8, color=_INK)
    ax.set_xlim(0, ct * 1.22 if ct > 0 else 1)
    ax.set_ylim(-0.7, n - 0.3 + 0.6)
    ax.grid(axis="x", color=_GRID, linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, colors=_INK_MUTED)

    fig.legend(
        [
            Patch(facecolor=style[p][0], hatch=style[p][1], edgecolor=_SURFACE)
            for p in processes
        ],
        processes,
        loc="lower center",
        ncol=min(len(processes), 4),
        frameon=False,
        labelcolor=_INK,
    )
    tp = report["throughput_per_h"]
    throughput = f"{tp:.1f} units/h" if tp is not None else "n/a"
    ax.set_title(
        f"cycle time {ct:.1f} s  •  throughput {throughput}  •  line efficiency "
        f"{report['utilization'] * 100:.0f}%",
        fontsize=9.5,
        color=_INK_MUTED,
        loc="left",
    )
    fig.suptitle(
        f"Line balance: {report['n_stations_used']} of {report['n_stations']} stations",
        fontsize=13,
        fontweight="bold",
        color=_INK,
    )
    height = (
        fig.get_figheight()
    )  # margins in inches, so they hold for any number of stations
    fig.subplots_adjust(
        top=1 - 0.8 / height,
        bottom=(0.75 + 0.3 * legend_rows) / height,
        left=0.2,
        right=0.92,
    )
    if path is not None:
        fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    return fig


def _text_on(fill: str) -> str:
    """White or ink, whichever has the higher contrast on `fill`."""

    def lum(hex_color):
        c = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    on_white = 1.05 / (lum(fill) + 0.05)
    on_ink = (lum(fill) + 0.05) / (lum(_INK) + 0.05)
    return "white" if on_white >= on_ink else _INK


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m disassembly_pomdp.line_balance_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--graph",
        required=True,
        type=Path,
        help="sequence graph export (.json, from export_graph)",
    )
    ap.add_argument(
        "--stations",
        "-k",
        default="auto",
        help="number of stations K, or 'auto' to choose K by SALBP-E (default: auto)",
    )
    ap.add_argument(
        "--goal",
        choices=["full", "target"],
        default="full",
        help="'full': balance the whole disassembly; 'target': up to the target parts (default: full)",
    )
    ap.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="multiply every action_time by this factor, e.g. a sim-to-real calibration (default: 1)",
    )
    ap.add_argument(
        "--time-limit-s",
        type=float,
        default=10.0,
        help="solver time limit per phase (default: 10)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="output directory (default: current directory)",
    )
    args = ap.parse_args(argv)

    if args.stations != "auto" and not (
        args.stations.isdigit() and int(args.stations) > 0
    ):
        ap.error(
            f"--stations must be a positive integer or 'auto', got {args.stations!r}"
        )
    if args.graph.suffix.lower() != ".json":
        ap.error(
            f"--graph must be a sequence graph export (.json, written by export_graph); got {args.graph}"
        )
    try:
        graph, provenance = read_export(args.graph)
    except (
        OSError,
        ValueError,
        KeyError,
    ) as e:  # missing file, not JSON, unknown schema_version, not an export
        ap.error(f"cannot read sequence graph export {args.graph}: {e!r}")

    settings = {
        "graph_file": str(args.graph),
        "goal": args.goal,
        "time_scale": args.time_scale,
    }
    try:
        tasks, precedence = extract_precedence(
            graph, choose_goal_state(graph, args.goal)
        )
        tasks = scale_times(tasks, args.time_scale)
        if args.stations == "auto":
            result = choose_k_salbp_e(tasks, precedence, time_limit_s=args.time_limit_s)
            plan = result.plan
            settings["salbp_e"] = {
                "sweep": [dataclasses.asdict(e) for e in result.sweep],
                "complete": result.complete,
            }
        else:
            plan = allocate_stations(
                tasks, precedence, int(args.stations), time_limit_s=args.time_limit_s
            )
    except (
        ValueError,
        TimeoutError,
    ) as e:  # precedence error, too few stations, infeasible, time limit
        ap.error(str(e))
    report = plan_report(plan, tasks, provenance, **settings)

    print(format_summary(report))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / PLAN_FILE).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_line_balance(report, args.out_dir / GANTT_FILE)
    print(f"\nSaved {args.out_dir / PLAN_FILE} and {args.out_dir / GANTT_FILE}")


if __name__ == "__main__":
    main(sys.argv[1:])
