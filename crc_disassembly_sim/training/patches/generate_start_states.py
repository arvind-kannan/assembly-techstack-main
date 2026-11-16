# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to pre-generate start states for PlanningEnv."""

import multiprocessing as mp
from multiprocessing import sharedctypes
import os
import queue
from typing import Sequence

from absl import app
from absl import flags
import tqdm

from roboballet.environments import data_pb2
from roboballet.environments import planning_env
from roboballet.launch import hparams


_NUM_TARGETS = flags.DEFINE_multi_integer(
    'num_targets', [10, 40], 'Number of targets to generate.'
)

_NUM_OBSTACLES = flags.DEFINE_multi_integer(
    'num_obstacles', [30], 'Number of obstacles to generate.'
)

_NUM_ROBOTS = flags.DEFINE_multi_integer(
    'num_robots', [4, 8], 'Number of robots in the environment.'
)

_NUM_STATES_TO_GENERATE = flags.DEFINE_integer(
    'num_states_to_generate', 50000, 'Number of start states to generate.'
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/start_states',
    'Path to the output file.',
)

_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 0, 'Number of workers to use. 0 for CPU count.'
)

_TIMEOUT_S = flags.DEFINE_float(
    'timeout_s', 60.0,
    'Stop collecting after this many seconds with no new state. '
    'Writes whatever was collected so far.'
)

_PAIR_SEP_MIN = flags.DEFINE_float(
    'pair_sep_min', 0.0,
    'Minimum target-to-target (and tip-to-tip) distance in metres for '
    'cooperative pair tasks. 0 = disabled.'
)

_PAIR_SEP_MAX = flags.DEFINE_float(
    'pair_sep_max', 0.0,
    'Maximum target-to-target (and tip-to-tip) distance in metres for '
    'cooperative pair tasks. 0 = disabled.'
)

_DERIVE_SECOND_TARGET = flags.DEFINE_bool(
    'derive_second_target', False,
    'When True, derive robot 1 target from robot 0 target using the initial '
    'tip-to-tip relative pose. Ensures target separation matches grasp '
    'geometry (rigid body transport training).'
)


def _derive_second_target(env) -> bool:
  """Replace env._targets[1] with a pose derived from target 0 + initial T_rel.

  Returns True if successful.
  """
  import numpy as np  # pylint: disable=import-outside-toplevel
  from roboballet.environments import target as target_lib  # pylint: disable=import-outside-toplevel

  if len(env._targets) < 2:  # pylint: disable=protected-access
    return False

  obs = env.observation()
  base_poses = obs['robots']['base_poses']
  tip_rel = obs['robots']['tip_relative_poses']
  if len(base_poses) < 2:
    return False

  tip_A = np.matmul(base_poses[0], tip_rel[0])
  tip_B = np.matmul(base_poses[1], tip_rel[1])
  t_rel = np.linalg.inv(tip_A) @ tip_B  # B relative to A at episode start

  # Build 4x4 pose for target A.
  t_info_A = env._targets[0][0]  # pylint: disable=protected-access
  target_A_pose = np.eye(4, dtype=np.float32)
  target_A_pose[:3, 3] = t_info_A.position
  if t_info_A.rotation is not None:
    target_A_pose[:3, :3] = t_info_A.rotation

  # Derive target B.
  target_B_pose = target_A_pose @ t_rel
  new_target_B = target_lib.TargetInfo(
      position=target_B_pose[:3, 3].copy(),
      rotation=target_B_pose[:3, :3].copy() if t_info_A.rotation is not None else None,
  )
  env._targets[1] = (new_target_B, target_lib.TargetState())  # pylint: disable=protected-access
  return True


def _check_pair_separation(env, pair_sep_min, pair_sep_max, attempt, max_retries):
  """Return True if the env's target and tip separation meets the constraint."""
  import numpy as np  # pylint: disable=import-outside-toplevel
  obs = env.observation()

  # Hard constraint: target-to-target separation.
  target_poses = obs['targets']['poses']
  if len(target_poses) >= 2:
    td = np.linalg.norm(target_poses[0][:3, 3] - target_poses[1][:3, 3])
    if not (pair_sep_min <= td <= pair_sep_max):
      return False

  # Soft constraint: tip-to-tip separation (relaxed in the second half of
  # retries so impossible robot configurations don't block generation).
  if attempt < max_retries // 2:
    base_poses = obs['robots']['base_poses']
    tip_rel = obs['robots']['tip_relative_poses']
    tip0 = np.matmul(base_poses[0], tip_rel[0])
    tip1 = np.matmul(base_poses[1], tip_rel[1])
    tip_d = np.linalg.norm(tip0[:3, 3] - tip1[:3, 3])
    if not (pair_sep_min <= tip_d <= pair_sep_max):
      return False

  return True


def _generate_worker(
    configs: hparams.LaunchConfigs,
    requests_remaining: 'sharedctypes.Synchronized[int]',
    output_queue: mp.Queue,
    pair_sep_min: float = 0.0,
    pair_sep_max: float = 0.0,
    derive_second_target: bool = False,
):
  """Generate worker."""
  env = planning_env.PlanningEnv(config=configs.env_config)
  max_retries = 100 if pair_sep_min > 0 else 1

  while True:
    with requests_remaining.get_lock():
      if requests_remaining.value <= 0:
        print('Worker exiting because queue is empty.')
        break
      requests_remaining.value -= 1

    accepted = False
    for attempt in range(max_retries):
      try:
        env.reset()
        if pair_sep_min > 0 and not _check_pair_separation(
            env, pair_sep_min, pair_sep_max, attempt, max_retries):
          continue
        if derive_second_target:
          _derive_second_target(env)
        start_state = env.get_start_state()
        output_queue.put(start_state.to_proto().SerializeToString())
        accepted = True
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        output_queue.put(e)
        accepted = True
        break

    if not accepted:
      # Could not satisfy constraint — return the slot for another attempt.
      with requests_remaining.get_lock():
        requests_remaining.value += 1


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  if _NUM_WORKERS.value == 0:
    num_workers = mp.cpu_count()
  else:
    num_workers = _NUM_WORKERS.value

  print(f'Using {num_workers} workers.')

  for num_targets in _NUM_TARGETS.value:
    for num_obstacles in _NUM_OBSTACLES.value:
      for num_robots in _NUM_ROBOTS.value:
        print(f'Generating start states for {num_targets} targets, '
              f'{num_obstacles} obstacles, {num_robots} robots.')
        config = hparams.make_optimised_configs(
            num_targets=num_targets,
            num_obstacles=num_obstacles,
            num_robots=num_robots,
        )

        requests_remaining = mp.Value('i', _NUM_STATES_TO_GENERATE.value,
                                      lock=True)

        output_queue = mp.Queue(maxsize=num_workers)

        processes = []
        for _ in range(num_workers):
          process = mp.Process(
              target=_generate_worker,
              args=(config, requests_remaining, output_queue,
                    _PAIR_SEP_MIN.value, _PAIR_SEP_MAX.value,
                    _DERIVE_SECOND_TARGET.value))
          process.start()
          processes.append(process)

        out_proto = data_pb2.StartStates()

        with tqdm.tqdm(total=_NUM_STATES_TO_GENERATE.value) as pbar:
          for _ in range(_NUM_STATES_TO_GENERATE.value):
            try:
              output_item = output_queue.get(timeout=_TIMEOUT_S.value)
            except queue.Empty:
              print(f'\nNo new state in {_TIMEOUT_S.value}s — '
                    f'saving {len(out_proto.start_states)} states and stopping.')
              break
            if isinstance(output_item, Exception):
              raise output_item
            start_state_proto = data_pb2.StartState()
            start_state_proto.ParseFromString(output_item)
            out_proto.start_states.append(start_state_proto)
            pbar.update(1)

        for process in processes:
          process.terminate()
          process.join()

        output_path = os.path.join(
            _OUTPUT_DIR.value,
            f'{num_robots}r_{num_targets}'
            f't_{num_obstacles}o.pb',
        )

        with open(output_path, 'wb') as f:
          f.write(out_proto.SerializeToString())


if __name__ == '__main__':
  mp.set_start_method('forkserver')
  app.run(main)
