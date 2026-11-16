"""Interactive visual simulation for RoboBallet.

Renders episodes using RenderScene → OpenCV window.
This avoids the mujoco.viewer pointer-invalidation problem that occurs when
planning_env.reset() replaces the internal world object each episode.

Usage (from robo_ballet/ directory):

  # Random policy (no checkpoint needed):
  PYTHONPATH=/home/ignadeg/robotics_env/robo_ballet \
    python roboballet/visualise.py --policy=random

  # Trained checkpoint:
  PYTHONPATH=/home/ignadeg/robotics_env/robo_ballet \
    python roboballet/visualise.py \
    --policy=checkpoint \
    --checkpoint_dir=./model_checkpoints

Controls (OpenCV window):
  Q / Esc     Quit
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.3')

import random
import time

from absl import app
from absl import flags
from absl import logging
import cv2
import jax
from jax import numpy as jnp
import numpy as np
from orbax import checkpoint

from roboballet.agents import compute_features
from roboballet.agents import networks
from roboballet.environments import planning_env
from roboballet.environments import simulator as sim_lib
from roboballet.launch import hparams
from roboballet.train import train as train_lib

FLAGS = flags.FLAGS

flags.DEFINE_enum('policy', 'random', ['random', 'checkpoint'],
                  'Which policy to use.')
flags.DEFINE_string('checkpoint_dir', './model_checkpoints',
                    'Directory containing model checkpoints.')
flags.DEFINE_integer('num_robots', 4, 'Number of robots (4 or 8).')
flags.DEFINE_integer('num_targets', 10, 'Number of targets.')
flags.DEFINE_integer('num_obstacles', 30, 'Number of obstacles.')
flags.DEFINE_integer('num_episodes', 0,
                     'Number of episodes to run (0 = run forever).')
flags.DEFINE_float('step_delay', 0.05,
                   'Seconds between rendered steps (controls playback speed).')
flags.DEFINE_integer('render_width', 1280, 'Render width in pixels.')
flags.DEFINE_integer('render_height', 720, 'Render height in pixels.')


def _load_checkpoint(checkpoint_dir: str, configs: hparams.LaunchConfigs):
  """Load policy network params from the latest checkpoint."""
  ckptr = checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())
  steps = []
  for name in os.listdir(checkpoint_dir):
    try:
      steps.append(int(name))
    except ValueError:
      pass
  if not steps:
    raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')
  latest_step = sorted(steps)[-1]
  restore_path = os.path.abspath(os.path.join(checkpoint_dir, str(latest_step)))
  logging.info('Restoring checkpoint from step %d: %s', latest_step,
               restore_path)

  policy_net = networks.RoboBalletPolicyNet(config=configs.model_config)
  twin_critic_net = networks.RoboBalletTwinCriticNet(config=configs.model_config)
  example_tree = train_lib.make_example_checkpoint_tree(
      planning_env_config=configs.env_config,
      policy_net=policy_net,
      twin_critic_net=twin_critic_net,
      train_config=configs.train_config,
  )
  restore_args = jax.tree_util.tree_map(
      lambda _: checkpoint.ArrayRestoreArgs(), example_tree)
  restored = ckptr.restore(
      restore_path,
      item={'step': 0, 'train_state': example_tree['train_state'],
            'elapsed_time_ns': 0},
      restore_args=restore_args,
  )
  return policy_net, restored['train_state'].policy_params, latest_step


def _render_frame(env: planning_env.PlanningEnv, width: int,
                  height: int) -> np.ndarray:
  """Render current env state, return BGR uint8 image for cv2."""
  # Fixed overhead camera looking straight down at the workspace.
  camera_pos = np.array([0.0, 0.0, 3.5])
  lookat = np.array([0.0, 0.0, 0.5])
  frame_rgb = env._world.RenderScene(  # pylint: disable=protected-access
      markers=[],
      width=width,
      height=height,
      camera_pos=camera_pos,
      lookat=lookat,
      fovy=60.0,
  )
  return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def main(argv):
  del argv

  configs = hparams.make_optimised_configs(
      num_targets=FLAGS.num_targets,
      num_obstacles=FLAGS.num_obstacles,
      num_robots=FLAGS.num_robots,
      pregen=True,
  )

  use_random = FLAGS.policy == 'random'
  policy_net = None
  policy_params = None
  network_step = 0

  if not use_random:
    policy_net, policy_params, network_step = _load_checkpoint(
        FLAGS.checkpoint_dir, configs)
    logging.info('Loaded policy at step %d', network_step)

  from roboballet.agents.actor import load_start_states
  start_states = load_start_states(configs.actor_config.start_states_path)

  feature_config = configs.model_config.feature_config
  action_scale = configs.env_config.base_config.max_joint_velocity
  rng_key = jax.random.PRNGKey(42)

  cv2.namedWindow('RoboBallet', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('RoboBallet', FLAGS.render_width, FLAGS.render_height)

  episode_count = 0

  while True:
    if FLAGS.num_episodes > 0 and episode_count >= FLAGS.num_episodes:
      logging.info('Reached %d episodes.', episode_count)
      break

    # Each episode gets a fresh env so reset() works cleanly.
    # We render via RenderScene each step, so the viewer doesn't hold
    # any stale C-level pointers.
    start_state = random.choice(start_states)
    env = planning_env.PlanningEnv(configs.env_config, start_state=start_state)
    try:
      obs = env.reset().observation
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Reset failed: %s — skipping episode', e)
      continue

    obs_spec = env.observation_spec()
    episode_count += 1
    total_reward = 0.0
    step_idx = 0

    logging.info('Episode %d start  (policy=%s  checkpoint_step=%d)',
                 episode_count, FLAGS.policy, network_step)

    while not env.terminal():
      graph_features = compute_features.make_graph_features(
          obs, obs_spec, feature_config)
      rng_key, subkey = jax.random.split(rng_key)

      if use_random:
        action = jax.random.uniform(
            key=subkey,
            shape=env.action_spec().shape,
            minval=-action_scale,
            maxval=action_scale,
        )
      else:
        action = policy_net.apply(policy_params, graph_features)
        # Policy outputs (num_graphs, num_robots, dof); env expects (robots, dof)
        action = action.squeeze(0) if action.ndim == 3 else action
        noise = jax.random.normal(key=subkey, shape=action.shape) * 0.05
        action = jnp.clip(action + noise, -action_scale, action_scale)

      step_result = env.step(np.array(action))
      obs = step_result.observation
      total_reward += float(step_result.reward) if step_result.reward else 0.0
      step_idx += 1

      # Render and display.
      frame = _render_frame(env, FLAGS.render_width, FLAGS.render_height)

      # Overlay episode/step info.
      cv2.putText(frame,
                  f'Ep {episode_count}  step {step_idx}  '
                  f'reward {total_reward:.2f}  policy_step {network_step}',
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
      cv2.imshow('RoboBallet', frame)

      key = cv2.waitKey(1)
      if key in (ord('q'), ord('Q'), 27):  # Q or Esc
        logging.info('Quit requested.')
        cv2.destroyAllWindows()
        return

      if FLAGS.step_delay > 0:
        time.sleep(FLAGS.step_delay)

    logging.info('Episode %d done — steps=%d  total_reward=%.3f',
                 episode_count, step_idx, total_reward)

  cv2.destroyAllWindows()


if __name__ == '__main__':
  app.run(main)
