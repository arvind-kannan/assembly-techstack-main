"""Automated staged curriculum trainer for RoboBallet 2-robot Comau setup.

Runs start-state generation and training for each stage sequentially,
advancing to the next stage once the evaluator's rolling average
num_targets_done/num_targets exceeds the configured success threshold.

Usage (inside main tmux window):
    python roboballet/launch/stage_manager.py [--start_stage=1] [--resume]

Flags:
    --start_stage   Stage to begin from (1-7, see STAGES below). Default 1.
    --resume        If set, skips start-state generation when .pb already exists
                    and skips training if checkpoint dir already has steps.
    --dry_run       Print commands without executing.

NOTE: the module-level constants below (REPLAY_MAX_SIZE, STALL_TIMEOUT_S, ...)
are read once at process start. Editing this file has NO EFFECT on a
stage_manager process that is already running — it must be killed and
restarted for a changed constant to take effect. This caused a real ~9-day
stall in 2026-09 where a `sed`-applied fix sat unused in a live process.
"""

import argparse
import dataclasses
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StageConfig:
    name: str
    num_targets: int
    num_obstacles: int
    targets_have_orientation: bool
    distance_threshold: float
    max_shaped_score: float
    collision_penalty_scaling: float
    # Advance when rolling avg (targets_done/num_targets) >= success_threshold
    # for `success_window` consecutive evaluator summaries.
    success_threshold: float
    success_window: int = 3
    is_final: bool = False
    # Cooperative task settings (Stage 5+).
    cooperative: bool = False
    robot_tip_edges: bool = False       # switch robot→robot edges to tip pose
    sync_reward_weight: float = 0.0
    simultaneous_bonus_weight: float = 0.0
    hold_reward_weight: float = 0.0
    distance_maintenance_weight: float = 0.0
    distance_maintenance_tolerance: float = 0.05
    relative_pose_maintenance_weight: float = 0.0
    relative_pose_maintenance_tolerance_pos: float = 0.05
    relative_pose_maintenance_tolerance_rot: float = 0.1
    # Separation constraint for start-state generation (0 = disabled).
    pair_sep_min: float = 0.0
    pair_sep_max: float = 0.0
    # Derive robot 1 target from robot 0 target + initial tip relative pose.
    derive_second_target: bool = False
    # Per-stage state-gen timeout override (0 = use global STATE_GEN_TIMEOUT_S).
    state_gen_timeout_s: float = 0.0


STAGES: list[StageConfig] = [
    StageConfig(
        name='stage1', num_targets=3, num_obstacles=5,
        targets_have_orientation=False,
        distance_threshold=0.15, max_shaped_score=0.3,
        collision_penalty_scaling=3.0,
        success_threshold=0.9, success_window=3,
    ),
    StageConfig(
        name='stage2', num_targets=3, num_obstacles=10,
        targets_have_orientation=False,
        distance_threshold=0.05, max_shaped_score=0.2,
        collision_penalty_scaling=5.0,
        success_threshold=0.5, success_window=3,
    ),
    StageConfig(
        name='stage3', num_targets=5, num_obstacles=15,
        targets_have_orientation=True,
        distance_threshold=0.05, max_shaped_score=0.1,
        collision_penalty_scaling=8.0,
        success_threshold=0.7, success_window=3,
    ),
    StageConfig(
        name='stage4', num_targets=10, num_obstacles=30,
        targets_have_orientation=True,
        distance_threshold=0.025, max_shaped_score=0.0,
        collision_penalty_scaling=15.0,
        success_threshold=0.7, success_window=5,
    ),
    # Stage 5: graph-structure adaptation — introduce robot_tip_edges with the
    # same task as Stage 4 (independent reaching). No cooperative rewards yet;
    # those are introduced in Stage 6. Stronger shaped score to overcome the
    # Stage 4 weight bias toward cautious hovering.
    StageConfig(
        name='stage5', num_targets=2, num_obstacles=10,
        targets_have_orientation=True,
        distance_threshold=0.05, max_shaped_score=0.15,
        collision_penalty_scaling=5.0,
        success_threshold=0.8, success_window=5,
        cooperative=True,
        robot_tip_edges=True,
        sync_reward_weight=0.0,
        simultaneous_bonus_weight=0.0,
        hold_reward_weight=0.0,
        distance_maintenance_weight=0.0,
        distance_maintenance_tolerance=0.05,
    ),
    # Stage 6: synchronized arrival at geometrically consistent targets.
    # Single target pair so the simultaneous bonus is unambiguous.
    # Derived targets (B from A via T_rel) so the goal geometry already
    # matches the grasp — Stage 7 only adds the pose maintenance reward.
    StageConfig(
        name='stage6', num_targets=1, num_obstacles=10,
        targets_have_orientation=True,
        distance_threshold=0.025, max_shaped_score=0.05,
        collision_penalty_scaling=5.0,
        success_threshold=0.8, success_window=5,
        cooperative=True,
        robot_tip_edges=True,
        sync_reward_weight=0.3,
        simultaneous_bonus_weight=0.1,
        hold_reward_weight=0.1,
        distance_maintenance_weight=0.3,
        distance_maintenance_tolerance=0.05,
        derive_second_target=True,
    ),
    # Stage 7: rigid body transport — full 6D relative TCP pose maintenance
    # with targets derived from the initial grasp geometry. Fine-tuned from
    # Stage 6 which already has the "arrive together" behavior.
    StageConfig(
        name='stage7', num_targets=2, num_obstacles=10,
        targets_have_orientation=True,
        distance_threshold=0.025, max_shaped_score=0.0,
        collision_penalty_scaling=5.0,
        success_threshold=0.8, success_window=5,
        is_final=True,
        cooperative=True,
        robot_tip_edges=True,
        sync_reward_weight=0.3,
        simultaneous_bonus_weight=0.0,
        hold_reward_weight=0.0,
        distance_maintenance_weight=0.2,
        distance_maintenance_tolerance=0.05,
        relative_pose_maintenance_weight=0.4,
        relative_pose_maintenance_tolerance_pos=0.05,
        relative_pose_maintenance_tolerance_rot=0.1,
        derive_second_target=True,
    ),
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HPARAMS_PATH = REPO_ROOT / 'roboballet' / 'launch' / 'hparams.py'
START_STATES_DIR = REPO_ROOT / 'roboballet' / 'data' / 'start_states'

# These constants are read once when this process starts. Changing a value
# here and saving the file does NOT affect a stage_manager that is already
# running — kill and restart the process for the new value to take effect.
NUM_ROBOTS = 2
REPLAY_MAX_SIZE = 50000
NUM_ACTORS = 1
STATE_GEN_TIMEOUT_S = 30    # stop state generation if no new state for this long
POLL_INTERVAL_S = 60        # how often to check tensorboard for new metrics
HEALTH_CHECK_EVERY = 5      # check tmux pane liveness every N polls
STALL_TIMEOUT_S = 3600      # restart if checkpoint step unchanged for 60 min
CHECKPOINT_WAIT_TIMEOUT_S = 10800  # max wait for checkpoint after threshold (3 h)


# ---------------------------------------------------------------------------
# hparams patching
# ---------------------------------------------------------------------------

def _patch_hparams(stage: StageConfig) -> None:
    """Rewrite hparams.py Comau branch to match stage config."""
    import re
    src = HPARAMS_PATH.read_text()
    original = src

    if stage.targets_have_orientation:
        src = src.replace('targets_have_orientation=False,',
                          'targets_have_orientation=True,')
    else:
        src = src.replace('targets_have_orientation=True,',
                          'targets_have_orientation=False,')

    src = re.sub(r'distance_threshold=[\d.]+,',
                 f'distance_threshold={stage.distance_threshold},', src)
    src = re.sub(r'max_shaped_score=[\d.]+,',
                 f'max_shaped_score={stage.max_shaped_score},', src)
    src = re.sub(r'collision_penalty_scaling=[\d.]+,',
                 f'collision_penalty_scaling={stage.collision_penalty_scaling},', src)

    # Cooperative reward weights — injected as env_config.replace() call.
    # Remove any existing cooperative block first, then re-inject if needed.
    src = re.sub(
        r'\n\s*# cooperative rewards.*?env_config\s*=\s*env_config\.replace\(.*?\)',
        '', src, flags=re.DOTALL)
    if stage.cooperative:
        cooperative_block = (
            f'\n    # cooperative rewards for {stage.name}\n'
            f'    env_config = env_config.replace(\n'
            f'        sync_reward_weight={stage.sync_reward_weight},\n'
            f'        simultaneous_bonus_weight={stage.simultaneous_bonus_weight},\n'
            f'        hold_reward_weight={stage.hold_reward_weight},\n'
            f'        distance_maintenance_weight={stage.distance_maintenance_weight},\n'
            f'        distance_maintenance_tolerance={stage.distance_maintenance_tolerance},\n'
            f'        relative_pose_maintenance_weight={stage.relative_pose_maintenance_weight},\n'
            f'        relative_pose_maintenance_tolerance_pos={stage.relative_pose_maintenance_tolerance_pos},\n'
            f'        relative_pose_maintenance_tolerance_rot={stage.relative_pose_maintenance_tolerance_rot},\n'
            f'    )'
        )
        src = src.replace('    return env_config',
                          cooperative_block + '\n    return env_config', 1)

    if src == original:
        print(f'[stage_manager] WARNING: hparams.py unchanged for {stage.name}')
    HPARAMS_PATH.write_text(src)
    print(f'[stage_manager] hparams.py patched for {stage.name}')


# ---------------------------------------------------------------------------
# Start-state generation
# ---------------------------------------------------------------------------

def _start_states_path(stage: StageConfig) -> pathlib.Path:
    return START_STATES_DIR / (
        f'{NUM_ROBOTS}r_{stage.num_targets}t_{stage.num_obstacles}o.pb')


def _generate_start_states(stage: StageConfig, dry_run: bool = False) -> None:
    pb_path = _start_states_path(stage)
    if pb_path.exists():
        if pb_path.stat().st_size > 100:
            print(f'[stage_manager] Start states already exist: {pb_path}')
            return
        print(f'[stage_manager] Start states file is empty or corrupt, regenerating.')
        pb_path.unlink()

    timeout_s = stage.state_gen_timeout_s if stage.state_gen_timeout_s > 0 else STATE_GEN_TIMEOUT_S
    cmd = [
        sys.executable, '-m',
        'roboballet.environments.generate_start_states',
        f'--num_robots={NUM_ROBOTS}',
        f'--num_targets={stage.num_targets}',
        f'--num_obstacles={stage.num_obstacles}',
        '--num_states_to_generate=1000',
        f'--timeout_s={timeout_s}',
        f'--output_dir={START_STATES_DIR}',
    ]
    if stage.pair_sep_min > 0:
        cmd += [
            f'--pair_sep_min={stage.pair_sep_min}',
            f'--pair_sep_max={stage.pair_sep_max}',
        ]
    if stage.derive_second_target:
        cmd.append('--derive_second_target=true')
    print(f'[stage_manager] Generating start states: {" ".join(cmd)}')
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    if not pb_path.exists() or pb_path.stat().st_size <= 100:
        raise RuntimeError(
            f'State generation produced an empty file for {stage.name}. '
            'The pair_sep constraint may be too tight for this robot workspace. '
            f'File: {pb_path}'
        )


# ---------------------------------------------------------------------------
# Training launch
# ---------------------------------------------------------------------------

def _checkpoint_dir(stage: StageConfig) -> pathlib.Path:
    return REPO_ROOT / f'checkpoints_{stage.name}'


def _latest_checkpoint_step(stage: StageConfig) -> Optional[int]:
    ckpt_dir = _checkpoint_dir(stage) / 'model_checkpoints'
    if not ckpt_dir.exists():
        return None
    steps = [int(d) for d in os.listdir(ckpt_dir) if d.isdigit()]
    return max(steps) if steps else None


def _kill_launchpad() -> None:
    for session in ('launchpad', 'launchpad_0'):
        subprocess.run(['tmux', 'kill-session', '-t', session],
                       capture_output=True)
    time.sleep(2)


def _launch_training(
    stage: StageConfig,
    prev_stage: Optional[StageConfig],
    dry_run: bool = False,
) -> subprocess.Popen:
    _kill_launchpad()

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'roboballet' / 'launch' / 'launch.py'),
        f'--num_actors={NUM_ACTORS}',
        f'--num_robots={NUM_ROBOTS}',
        f'--num_targets={stage.num_targets}',
        f'--num_obstacles={stage.num_obstacles}',
        f'--artifacts_prefix=./checkpoints_{stage.name}',
        f'--exp_name=RoboBallet_{stage.name.capitalize()}',
        f'--replay_max_size={REPLAY_MAX_SIZE}',
        '--terminal=tmux_session',
    ]
    if prev_stage is not None:
        cmd.append(f'--restore_checkpoint=./checkpoints_{prev_stage.name}')
    if stage.robot_tip_edges:
        cmd.append('--robot_tip_edges=true')
    # no --restore_checkpoint_step: pick latest automatically

    print(f'[stage_manager] Launching training: {" ".join(cmd)}')
    if dry_run:
        return None  # type: ignore

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    # Give Launchpad time to spawn tmux windows before we start polling.
    time.sleep(30)
    return proc


# ---------------------------------------------------------------------------
# Progress monitoring via tensorboard
# ---------------------------------------------------------------------------

def _read_tb_targets_done(stage: StageConfig) -> list[tuple[int, float]]:
    """Return (step, num_targets_done) pairs from the stage's tensorboard log."""
    tb_dir = _checkpoint_dir(stage) / 'tensorboard_log'
    if not tb_dir.exists():
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
    except ImportError:
        return []

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    for tag in ['eval/num_targets_done', 'evaluator/num_targets_done',
                'num_targets_done']:
        if tag in tags:
            return [(e.step, e.value) for e in ea.Scalars(tag)]
    return []


def _check_threshold(
    stage: StageConfig, seen_steps: set[int]
) -> tuple[bool, set[int], Optional[int]]:
    """Return (should_advance, updated_seen_steps, threshold_tb_step).

    threshold_tb_step is the latest tensorboard step when threshold was first
    crossed; None when not yet crossed.
    """
    data = _read_tb_targets_done(stage)
    new_data = [(s, v) for s, v in data if s not in seen_steps]
    if new_data:
        seen_steps = seen_steps | {s for s, _ in new_data}
        fracs = [v / stage.num_targets for _, v in new_data]
        print(f'[stage_manager] New evaluator results: '
              f'{[f"{f:.2f}" for f in fracs[-5:]]} '
              f'(threshold={stage.success_threshold})')

    # Check rolling window from all data seen so far
    all_data = sorted(data)
    all_fracs = [v / stage.num_targets for _, v in all_data]
    if len(all_fracs) < stage.success_window:
        return False, seen_steps, None

    window = all_fracs[-stage.success_window:]
    if all(f >= stage.success_threshold for f in window):
        threshold_step = all_data[-1][0]  # TB step of the last eval in the window
        print(f'[stage_manager] SUCCESS: last {stage.success_window} evals '
              f'= {[f"{f:.2f}" for f in window]} >= {stage.success_threshold} '
              f'(at TB step {threshold_step})')
        return True, seen_steps, threshold_step

    return False, seen_steps, None


def _is_launchpad_alive() -> bool:
    """Return True if the launchpad tmux session exists with no dead panes."""
    r = subprocess.run(['tmux', 'has-session', '-t', 'launchpad'],
                       capture_output=True)
    if r.returncode != 0:
        return False
    r = subprocess.run(
        ['tmux', 'list-panes', '-t', 'launchpad', '-F', '#{pane_dead}'],
        capture_output=True, text=True)
    # Any line equal to '1' means a dead pane (replay crash etc.)
    return '1' not in r.stdout.split()


def _wait_for_checkpoint(stage: StageConfig, min_step: int) -> bool:
    """Poll until a checkpoint >= min_step exists. Returns True on success."""
    deadline = time.time() + CHECKPOINT_WAIT_TIMEOUT_S
    print(f'[stage_manager] Waiting for checkpoint >= step {min_step} '
          f'(timeout {CHECKPOINT_WAIT_TIMEOUT_S//60} min)...')
    while time.time() < deadline:
        ckpt_step = _latest_checkpoint_step(stage)
        if ckpt_step is not None and ckpt_step >= min_step:
            print(f'[stage_manager] Checkpoint found at step {ckpt_step}. '
                  'Advancing.')
            return True
        remaining = int(deadline - time.time())
        print(f'[stage_manager] Latest checkpoint: {ckpt_step} '
              f'(need {min_step}, {remaining}s remaining)')
        time.sleep(POLL_INTERVAL_S)
    print(f'[stage_manager] WARNING: checkpoint wait timed out after '
          f'{CHECKPOINT_WAIT_TIMEOUT_S//60} min. Advancing anyway.')
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_stage(
    stage: StageConfig,
    prev_stage: Optional[StageConfig],
    resume: bool,
    dry_run: bool,
) -> None:
    print(f'\n{"="*60}')
    print(f'[stage_manager] Starting {stage.name.upper()}')
    print(f'  targets={stage.num_targets}, obstacles={stage.num_obstacles}, '
          f'orientation={stage.targets_have_orientation}')
    print(f'  dist_threshold={stage.distance_threshold}, '
          f'max_shaped_score={stage.max_shaped_score}, '
          f'collision_penalty={stage.collision_penalty_scaling}')
    print(f'  success: avg(targets_done/{stage.num_targets}) >= '
          f'{stage.success_threshold} for {stage.success_window} evals')
    print(f'{"="*60}')

    _patch_hparams(stage)
    _generate_start_states(stage, dry_run=dry_run)

    existing_step = _latest_checkpoint_step(stage)
    if resume and existing_step is not None:
        print(f'[stage_manager] Resuming {stage.name} from step {existing_step}')
        proc = _launch_training(stage, prev_stage=stage, dry_run=dry_run)
    else:
        proc = _launch_training(stage, prev_stage=prev_stage, dry_run=dry_run)

    if stage.is_final:
        print(f'[stage_manager] {stage.name} is the final stage. '
              'Training indefinitely. Ctrl+C to stop.')
        if dry_run:
            return
        try:
            while True:
                try:
                    proc.wait(timeout=POLL_INTERVAL_S * HEALTH_CHECK_EVERY)
                    # Proc exited — crash recovery.
                    print(f'[stage_manager] Final stage training exited '
                          f'(code {proc.returncode}). Restarting in 60s.')
                    _kill_launchpad()
                    time.sleep(60)
                    proc = _launch_training(stage, prev_stage=stage,
                                            dry_run=dry_run)
                except subprocess.TimeoutExpired:
                    pass  # still running, loop
        except KeyboardInterrupt:
            print(f'\n[stage_manager] Interrupted during {stage.name}.')
            proc.terminate()
            _kill_launchpad()
            sys.exit(0)
        return

    seen_steps: set[int] = set()
    poll_count = 0
    # Stall detection: track last checkpoint step and when it last changed.
    last_ckpt_step: Optional[int] = None
    last_ckpt_change_time = time.time()

    print(f'[stage_manager] Monitoring evaluator every {POLL_INTERVAL_S}s '
          f'(health check every {HEALTH_CHECK_EVERY * POLL_INTERVAL_S}s, '
          f'stall timeout {STALL_TIMEOUT_S // 60} min)...')
    try:
        while True:
            time.sleep(POLL_INTERVAL_S)
            poll_count += 1

            # --- Crash / health check (every HEALTH_CHECK_EVERY polls) ---
            if not dry_run and poll_count % HEALTH_CHECK_EVERY == 0:
                # Check launcher process
                if proc and proc.poll() is not None:
                    print(f'[stage_manager] Launcher process exited '
                          f'(code {proc.returncode}). Re-launching in 30s.')
                    time.sleep(30)
                    proc = _launch_training(stage, prev_stage=stage,
                                            dry_run=dry_run)
                    last_ckpt_change_time = time.time()
                    continue

                # Check tmux pane liveness (replay die = dead pane)
                if not _is_launchpad_alive():
                    print('[stage_manager] Launchpad session dead or has dead '
                          'pane (replay crash?). Re-launching in 60s.')
                    time.sleep(60)
                    if proc:
                        proc.terminate()
                    proc = _launch_training(stage, prev_stage=stage,
                                            dry_run=dry_run)
                    last_ckpt_change_time = time.time()
                    continue

                # Check for step stagnation
                current_ckpt = _latest_checkpoint_step(stage)
                if current_ckpt is not None:
                    if current_ckpt != last_ckpt_step:
                        last_ckpt_step = current_ckpt
                        last_ckpt_change_time = time.time()
                    else:
                        stall_s = time.time() - last_ckpt_change_time
                        if stall_s > STALL_TIMEOUT_S:
                            print(f'[stage_manager] Checkpoint step has not '
                                  f'advanced in {stall_s/60:.0f} min '
                                  f'(still at {current_ckpt}). '
                                  'Re-launching.')
                            if proc:
                                proc.terminate()
                            _kill_launchpad()
                            time.sleep(30)
                            proc = _launch_training(stage, prev_stage=stage,
                                                    dry_run=dry_run)
                            last_ckpt_change_time = time.time()
                            continue

            # --- Threshold check ---
            advance, seen_steps, threshold_step = _check_threshold(
                stage, seen_steps)
            if advance:
                # Wait for a checkpoint at or beyond the threshold TB step
                # before terminating, so we don't lose the weights.
                if not dry_run and threshold_step is not None:
                    _wait_for_checkpoint(stage, threshold_step)

                print(f'[stage_manager] Terminating {stage.name} training.')
                if proc and not dry_run:
                    proc.terminate()
                _kill_launchpad()
                return

    except KeyboardInterrupt:
        print(f'\n[stage_manager] Interrupted during {stage.name}.')
        if proc and not dry_run:
            proc.terminate()
        _kill_launchpad()
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--start_stage', type=int, default=1,
                        help=f'Stage to begin from (1-{len(STAGES)})')
    parser.add_argument('--resume', action='store_true',
                        help='Skip generation/launch if already in progress')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    start_idx = args.start_stage - 1
    if not 0 <= start_idx < len(STAGES):
        print(f'--start_stage must be 1-{len(STAGES)}')
        sys.exit(1)

    prev_stage: Optional[StageConfig] = STAGES[start_idx - 1] if start_idx > 0 else None

    for stage in STAGES[start_idx:]:
        run_stage(stage, prev_stage=prev_stage,
                  resume=args.resume, dry_run=args.dry_run)
        prev_stage = stage

    print('[stage_manager] All stages complete.')


if __name__ == '__main__':
    main()
