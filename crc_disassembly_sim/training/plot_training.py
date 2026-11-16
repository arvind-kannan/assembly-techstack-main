"""Plot training metrics across all stages from tensorboard event files."""

import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_tb_scalars(logdir: str, tag: str) -> list[tuple[int, float]]:
    """Read a scalar tag from tensorboard event files in logdir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, STORE_EVERYTHING_SIZE_GUIDANCE)
    except ImportError:
        print('tensorboard package not found. Install with: pip install tensorboard')
        return []

    ea = EventAccumulator(logdir, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return []
    return [(e.step, e.value) for e in ea.Tensors(tag)] if False else \
           [(e.step, e.value) for e in ea.Scalars(tag)]


STAGE_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#DD8452', '#937860']
STAGE_NAMES = ['Stage 1\n(3t 5o 15cm)', 'Stage 2\n(3t 10o 5cm)',
               'Stage 3\n(5t 15o ori)', 'Stage 4\n(10t 30o 2.5cm)',
               'Stage 5\n(2t parallel)', 'Stage 6\n(1t sync arrive)',
               'Stage 7\n(2t rigid body)']


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_base', default='.',
                        help='Directory containing checkpoints_stage1..5')
    parser.add_argument('--output', default='training_progress.png')
    parser.add_argument('--smooth_window', type=int, default=5)
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle('RoboBallet Staged Curriculum Training Progress', fontsize=14, fontweight='bold')

    metrics = [
        ('eval/num_targets_done', 'Avg targets done', axes[0]),
        ('eval/total_collision_penalty', 'Collision penalty', axes[1]),
        ('eval/final_score', 'Final score', axes[2]),
    ]

    legend_patches = []
    x_offset = 0  # shift steps so stages appear sequential on x-axis

    for stage_idx in range(len(STAGE_NAMES)):
        stage_name = f'stage{stage_idx + 1}'
        tb_dir = os.path.join(args.checkpoint_base, f'checkpoints_{stage_name}',
                              'tensorboard_log')
        if not os.path.isdir(tb_dir):
            print(f'No tensorboard log found for {stage_name} at {tb_dir}')
            continue

        color = STAGE_COLORS[stage_idx]
        stage_label = STAGE_NAMES[stage_idx]
        has_data = False

        for tag, ylabel, ax in metrics:
            data = read_tb_scalars(tb_dir, tag)
            if not data:
                # try alternative tag names
                for alt in [tag.replace('eval/', 'evaluator/'),
                             tag.replace('eval/', '')]:
                    data = read_tb_scalars(tb_dir, alt)
                    if data:
                        break
            if not data:
                continue

            has_data = True
            steps, values = zip(*sorted(data))
            steps = np.array(steps) + x_offset
            values = np.array(values)

            ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
            if len(values) >= args.smooth_window:
                s = smooth(values.tolist(), args.smooth_window)
                s_steps = steps[args.smooth_window - 1:]
                ax.plot(s_steps, s, color=color, linewidth=2.0)

            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)

            if stage_idx > 0:
                ax.axvline(x=x_offset, color=color, linestyle='--',
                           alpha=0.6, linewidth=1)

        if has_data and steps is not None:
            x_offset = int(steps[-1])

        if has_data:
            legend_patches.append(Patch(color=color, label=stage_label))

    for tag, ylabel, ax in metrics:
        ax.set_xlabel('Training step', fontsize=9)

    if legend_patches:
        axes[0].legend(handles=legend_patches, loc='upper left',
                       fontsize=9, framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(args.checkpoint_base, args.output)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
