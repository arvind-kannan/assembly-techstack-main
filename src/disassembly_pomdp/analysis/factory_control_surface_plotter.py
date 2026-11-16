from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
from diss_assembly_dataclasses.assembly import Assembly

from disassembly_pomdp.eol_operator import EOLContext


@dataclass(frozen=True)
class SurfaceData:
    T: np.ndarray  # (ny, nx)
    C: np.ndarray  # (ny, nx)
    Z: np.ndarray  # (ny, nx)


def quantize(Z: np.ndarray, step: float = 0.1) -> np.ndarray:
    if step <= 0:
        raise ValueError("quantization step must be > 0")
    return np.clip(np.round(Z / step) * step, 0.0, 1.0)


def build_goal_surface(
    *,
    goal_belief,  # GoalBeliefOperator
    goal_root,  # GoalNode
    # X axis definition
    t_values: np.ndarray,  # shape (nx,)
    # Y axis definition
    checkpoints: np.ndarray,  # shape (ny,)
    # Provide assemblies for each checkpoint
    assemblies_by_ckpt: Sequence[
        Sequence["Assembly"]
    ],  # len=ny, each is a list of eval assemblies
    # Provide "e" (maturity index) for each checkpoint, if you want it in ctx
    e_by_ckpt: Optional[Sequence[int]] = None,
    # Optional: customize ctx per grid point (hook for future)
    ctx_builder: Optional[Callable[[int, int, int], "EOLContext"]] = None,
    # Aggregation over multiple eval assemblies
    reduce: str = "mean",  # "mean" | "median" | "min" | "max"
    # Visualization mode
    mode: str = "smooth",  # "smooth" or "blocky"
    q_step: float = 0.1,
) -> SurfaceData:
    """
    Build a regular (checkpoint, t) grid where Z[iy, ix] = P(goal_success | ctx, assembly).

    assemblies_by_ckpt:
      - Row iy corresponds to checkpoints[iy]
      - You can pass multiple assemblies (different random seeds / initializations) and reduce them.

    Fork points:
      - If later you want Z to mean "eventual completion probability", this is where you would
        replace the per-step query with a rollout / planner-based estimator.
    """
    t_values = np.asarray(t_values, dtype=float).reshape(-1)
    checkpoints = np.asarray(checkpoints, dtype=float).reshape(-1)
    nx = t_values.shape[0]
    ny = checkpoints.shape[0]

    if len(assemblies_by_ckpt) != ny:
        raise ValueError(
            f"assemblies_by_ckpt must have len=ny ({ny}), got {len(assemblies_by_ckpt)}"
        )

    if e_by_ckpt is None:
        e_by_ckpt = list(range(ny))
    if len(e_by_ckpt) != ny:
        raise ValueError(f"e_by_ckpt must have len=ny ({ny}), got {len(e_by_ckpt)}")

    T, C = np.meshgrid(t_values, checkpoints)  # (ny, nx)

    Z = np.zeros((ny, nx), dtype=float)

    def reduce_vals(vals: np.ndarray) -> float:
        if vals.size == 0:
            return 0.0
        if reduce == "mean":
            return float(np.mean(vals))
        if reduce == "median":
            return float(np.median(vals))
        if reduce == "min":
            return float(np.min(vals))
        if reduce == "max":
            return float(np.max(vals))
        raise ValueError("reduce must be one of: mean, median, min, max")

    for iy in range(ny):
        ckpt = int(checkpoints[iy])
        e_val = int(e_by_ckpt[iy])
        eval_asms = list(assemblies_by_ckpt[iy])
        if not eval_asms:
            raise ValueError(f"assemblies_by_ckpt[{iy}] is empty")

        for ix in range(nx):
            t_val = int(round(float(t_values[ix])))

            if ctx_builder is not None:
                ctx = ctx_builder(t_val, e_val, ckpt)
            else:
                ctx = EOLContext(t=t_val, e=e_val, checkpoint=ckpt)

            # Evaluate goal probability on multiple assemblies and aggregate
            vals = []
            for asm in eval_asms:
                p = goal_belief.prob(goal_root, asm, ctx)
                vals.append(float(p))
            Z[iy, ix] = reduce_vals(np.asarray(vals, dtype=float))

    Z = np.clip(Z, 0.0, 1.0)

    if mode == "blocky":
        Z = quantize(Z, step=q_step)
    elif mode != "smooth":
        raise ValueError("mode must be 'smooth' or 'blocky'")

    return SurfaceData(T=T, C=C, Z=Z)


def make_surface_figure(
    data: SurfaceData,
    title: str = "Factory Control Surface",
    zmin: float = 0.0,
    zmax: float = 1.0,
    goal_label: str = "",
) -> go.Figure:
    """
    Create an interactive Plotly 3D surface figure with informative hover text.
    """
    T, C, Z = data.T, data.C, data.Z

    if T.shape != C.shape or T.shape != Z.shape:
        raise ValueError(f"Shape mismatch: T{T.shape}, C{C.shape}, Z{Z.shape}")

    # Custom hover: show X/Y bin coords + probability
    hover = np.empty(T.shape, dtype=object)
    for iy in range(T.shape[0]):
        for ix in range(T.shape[1]):
            hover[iy, ix] = (
                f"Goal: {goal_label}<br>"
                f"Time/Step: {T[iy, ix]:.2f}<br>"
                f"Checkpoint: {C[iy, ix]:.2f}<br>"
                f"P_success: {Z[iy, ix]:.3f}"
            )

    surf = go.Surface(
        x=T,
        y=C,
        z=Z,
        surfacecolor=Z,  # color encodes probability
        cmin=zmin,
        cmax=zmax,
        colorscale="Viridis",  # nice default
        hoverinfo="text",
        text=hover,
        showscale=True,
        colorbar=dict(title="P_success"),
    )

    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Time / Step bin",
            yaxis_title="Checkpoint index",
            zaxis_title="P(success)",
            zaxis=dict(range=[zmin, zmax]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def export_html(fig: go.Figure, out_path: str | Path) -> Path:
    """
    Export Plotly figure to a standalone HTML file.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path
