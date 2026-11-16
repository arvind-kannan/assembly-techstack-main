"""GNNWaypointPlanner — bridges CRC DisassemblySim with the RoboBallet GNN.

Architecture
------------
The RoboBallet GNN was trained on Panda arms in MuJoCo.  The CRC uses a Comau
NJ290 in PyBullet.  Direct joint-space transfer is not possible (different DOF,
kinematics, workspace).

Instead the planner operates in *normalised Cartesian space*:

  1. The CRC scene (EEF pose, target pose, obstacles) is mapped into the
     RoboBallet workspace coordinate frame via a configurable affine transform.

  2. A single-arm virtual RoboBallet environment is stepped with the trained
     policy.  At each step the Panda end-effector position is recorded.

  3. The resulting EEF trajectory is mapped back to CRC world coordinates and
     emitted as a list of G-code waypoint dicts that GCodeProcessor can execute.

This means the GNN does collision-aware path planning in an abstract workspace
that is geometrically equivalent to the CRC scene, while the CRC robot follows
the planned Cartesian path via its own IK (handled by GCodeProcessor).

Constraint support
------------------
SceneConstraint objects (see scene.py) are encoded as additional graph nodes in
_build_obs_from_scene().  Currently they are passed through as padding nodes so
the network sees them but was not trained to react to them.  To make the network
*respond* to constraints:
  1. Add the corresponding reward terms in planning_env.py.
  2. Fine-tune from the existing checkpoint with --restore_checkpoint.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports of RoboBallet so the CRC can be used without the roboballet
# venv active (e.g. for testing scene.py in isolation).
# ---------------------------------------------------------------------------
_roboballet_imported = False
_roboballet_error: Optional[Exception] = None

def _ensure_roboballet(roboballet_root: str):
    global _roboballet_imported, _roboballet_error
    if _roboballet_imported:
        return
    if roboballet_root not in sys.path:
        sys.path.insert(0, roboballet_root)
    try:
        import jax
        import jax.numpy as jnp
        from orbax import checkpoint
        _roboballet_imported = True
    except Exception as e:
        _roboballet_error = e
        raise RuntimeError(
            f'Failed to import RoboBallet dependencies from {roboballet_root}. '
            f'Make sure the roboballet venv is active.\n{e}'
        ) from e


# ---------------------------------------------------------------------------

from modular_planner.scene import ObstacleBox, SceneConstraint, ConstraintType


class GNNWaypointPlanner:
    """Generates collision-aware Cartesian waypoints using the RoboBallet GNN.

    Parameters
    ----------
    checkpoint_dir:
        Path to the model_checkpoints directory produced by roboballet training.
    roboballet_root:
        Path to the directory that contains the ``roboballet`` Python package
        (i.e. the directory where ``import roboballet`` would work).
    num_robots:
        Number of robots the GNN was trained with (default 4).
    num_targets:
        Number of targets the GNN was trained with (default 10).
    num_obstacles:
        Maximum obstacles the GNN was trained with (default 30).
    max_plan_steps:
        Maximum policy steps per planning call (safety cap).
    convergence_tol:
        Distance in metres at which the EEF is considered to have reached the
        target and planning stops early.
    workspace_scale:
        Metres-per-unit scale of the RoboBallet workspace relative to the CRC
        workspace.  Adjust if the CRC cell is much larger/smaller than the ~1 m
        RoboBallet table.
    waypoint_step:
        Only emit every Nth EEF position as a waypoint (reduces waypoint count).
    feed_forward_speed:
        G-code feed rate (mm/min equivalent) assigned to all waypoints.
    """

    # Default path — relative to this file's location; override in constructor.
    _DEFAULT_ROBOBALLET_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '..', 'robo_ballet'))

    def __init__(
        self,
        checkpoint_dir: str,
        roboballet_root: Optional[str] = None,
        num_robots: int = 2,
        num_targets: int = 10,
        num_obstacles: int = 30,
        max_plan_steps: int = 200,
        convergence_tol: float = 0.05,
        workspace_scale: float = 1.0,
        waypoint_step: int = 5,
        feed_forward_speed: float = 1000.0,
        scene_conditioning: bool = True,
        use_target_orientation: bool = True,
        jit_policy: bool = True,
        virtual_base_poses: Optional[list] = None,
        strict_ik: bool = True,
        env_stage: Optional[str] = None,
        scale_actions: bool = True,
        ladder_heights: Optional[tuple] = None,
        pad_obstacles_to: int = 0,
    ):
        # scene_conditioning: inject the CRC robot poses / targets / obstacles into the virtual
        #   environment as a StartState (the mechanism the training pipeline uses).  False
        #   restores the original behaviour, in which env.reset() discarded them (see
        #   _run_policy_multi) - kept only to reproduce old results.
        # jit_policy: compile the policy network (eager calls cost ~400 ms per step).
        # virtual_base_poses: base pose of each virtual robot; defaults to the measured CRC
        #   cell (robot_C at the origin, robot_D at (-1.99, -3.75, 0) yawed +90 deg) instead of
        #   the MJCF's older -3.5153 m.
        # strict_ik: a current or target pose the virtual robot cannot reach raises RuntimeError
        #   (the caller falls back to DIRECT) instead of planning from a default configuration.
        self._strict_ik = strict_ik
        # env_stage='stage4': build the virtual environment with the Stage 4 settings the checkpoint
        #   was trained with (oriented targets, 2.5 cm threshold, collision penalty 15); None keeps
        #   whatever the checked-in hparams.py holds (it was left at Stage 1: 15 cm, position-only).
        # ladder_heights: several targets stacked vertically above each target (metres above it); the
        #   rollout stops for a robot as soon as it reaches ANY rung (hand-off to G-code) and the path
        #   ends on that rung.
        # pad_obstacles_to: add tiny far-away boxes up to this many obstacles (training scenes had 30).
        # scale_actions: the training actor and the evaluator step the environment with the network
        #   output (tanh, [-1, 1]) MULTIPLIED by base_config.max_joint_velocity (rad/s); the original
        #   adapter passed the raw [-1, 1] values (2.2x too slow). False reproduces that.
        self._scale_actions = scale_actions
        self._env_stage = env_stage
        self._ladder_heights = tuple(ladder_heights) if ladder_heights else None
        self._pad_obstacles_to = int(pad_obstacles_to)
        self._slot_targets = {}
        self._scene_conditioning = scene_conditioning
        self._use_target_orientation = use_target_orientation
        self._jit_policy = jit_policy
        if virtual_base_poses is None:
            base_d = np.eye(4)
            base_d[:3, :3] = Rotation.from_euler('z', 90.0, degrees=True).as_matrix()
            base_d[:3, 3] = [-1.99, -3.75, 0.0]
            virtual_base_poses = [np.eye(4), base_d]
        self._virtual_base_poses = [np.asarray(b, dtype=np.float64) for b in virtual_base_poses]
        self.last_plan_info = {}
        self._checkpoint_dir = os.path.abspath(checkpoint_dir)
        self._roboballet_root = roboballet_root or self._DEFAULT_ROBOBALLET_ROOT
        self._num_robots = num_robots
        self._num_targets = num_targets
        self._num_obstacles = num_obstacles
        self._max_plan_steps = max_plan_steps
        self._convergence_tol = convergence_tol
        self._workspace_scale = workspace_scale
        self._waypoint_step = waypoint_step
        self._feed_forward_speed = feed_forward_speed

        # Lazy-loaded after first plan() call.
        self._policy_net = None
        self._policy_params = None
        self._configs = None
        self._env = None
        self._checkpoint_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        current_eef_pose: np.ndarray,
        target_eef_pose: np.ndarray,
        obstacles: list[ObstacleBox],
        constraints: list[SceneConstraint] | None = None,
        robot_name: str = 'robot_D',
    ) -> list[dict]:
        """Generate a collision-aware Cartesian path as G-code waypoints.

        Parameters
        ----------
        current_eef_pose:
            4×4 homogeneous matrix of the robot's current end-effector pose in
            the CRC world frame (metres).
        target_eef_pose:
            4×4 homogeneous matrix of the desired end-effector pose.
        obstacles:
            List of ObstacleBox objects extracted from the live PyBullet scene.
        constraints:
            Optional list of SceneConstraint objects.  Currently encoded as
            padding nodes; retrain to make the policy react to them.
        robot_name:
            Identifier string embedded in the waypoint metadata (for logging).

        Returns
        -------
        List of G-code command dicts compatible with GCodeProcessor:
            [{'G': 1, 'X': ..., 'Y': ..., 'Z': ...,
              'A': ..., 'B': ..., 'C': ..., 'F': ...}, ...]
        """
        self._lazy_load()

        constraints = constraints or []

        # Map CRC scene into RoboBallet workspace.
        rb_current = self._crc_to_rb(current_eef_pose)
        rb_target = self._crc_to_rb(target_eef_pose)
        rb_obstacles = [self._obstacle_crc_to_rb(o) for o in obstacles]

        # Build virtual environment with the mapped scene.
        if self._scene_conditioning:
            slot = 0 if str(robot_name).endswith('C') else 1
            eef_trajectory_rb = self._rollout_scene(
                {slot: (rb_current, rb_target)}, rb_obstacles, constraints)[slot]
        else:
            eef_trajectory_rb = self._run_policy(
                rb_current, rb_target, rb_obstacles, constraints)

        # Map trajectory back to CRC frame.
        eef_trajectory_crc = [self._rb_to_crc(p) for p in eef_trajectory_rb]

        waypoints = self._trajectory_to_gcode(eef_trajectory_crc)
        logger.info(
            'GNNWaypointPlanner: planned %d waypoints for %s '
            '(checkpoint step %d)',
            len(waypoints), robot_name, self._checkpoint_step)
        return waypoints

    def plan_multi(
        self,
        current_eef_poses: list,
        target_eef_poses: list,
        obstacles: list,
        constraints: list | None = None,
        robot_names: list | None = None,
    ) -> list:
        """Plan simultaneous collision-aware paths for N robots in one GNN pass.

        Both CRC robots (robot_C and robot_D) are mapped into the RoboBallet
        virtual environment as robots 0 and 1.  The remaining slots (up to
        num_robots, default 4) are padded with their pregenerated start poses
        and do not interfere with planning.  The policy runs once per step and
        produces actions for all robots simultaneously, so inter-robot collision
        avoidance is handled natively by the GNN.

        Parameters
        ----------
        current_eef_poses:
            List of 4×4 homogeneous matrices, one per CRC robot, in CRC world
            frame.  Length must be ≤ num_robots (default 4).
        target_eef_poses:
            List of 4×4 target poses, same length and order as
            current_eef_poses.
        obstacles:
            List of ObstacleBox objects from the live PyBullet scene.
        constraints:
            Optional SceneConstraint list (currently encoded as padding nodes).
        robot_names:
            Optional list of string labels used in log messages.

        Returns
        -------
        List of G-code waypoint lists, one per input robot:
            [[{'G':1,'X':...}, ...], [{'G':1,'X':...}, ...]]
        """
        self._lazy_load()

        n = len(current_eef_poses)
        assert len(target_eef_poses) == n, \
            'current_eef_poses and target_eef_poses must have the same length'
        assert n <= self._num_robots, \
            f'Got {n} robots but planner was built for {self._num_robots}'

        constraints = constraints or []
        robot_names = robot_names or [f'robot_{i}' for i in range(n)]

        # Map all poses into RoboBallet workspace.
        rb_currents = [self._crc_to_rb(p) for p in current_eef_poses]
        rb_targets  = [self._crc_to_rb(p) for p in target_eef_poses]
        rb_obstacles = [self._obstacle_crc_to_rb(o) for o in obstacles]

        # Run policy with all robots active simultaneously.
        if self._scene_conditioning:
            traj = self._rollout_scene(
                {i: (rb_currents[i], rb_targets[i]) for i in range(n)}, rb_obstacles, constraints)
            trajectories_rb = [traj[i] for i in range(n)]
        else:
            trajectories_rb = self._run_policy_multi(
                rb_currents, rb_targets, rb_obstacles, constraints)

        # Map trajectories back to CRC frame and convert to G-code.
        result = []
        for i, traj_rb in enumerate(trajectories_rb):
            traj_crc = [self._rb_to_crc(pose) for pose in traj_rb]
            waypoints = self._trajectory_to_gcode(traj_crc)
            logger.info(
                'GNNWaypointPlanner.plan_multi: robot %s → %d waypoints '
                '(checkpoint step %d)',
                robot_names[i], len(waypoints), self._checkpoint_step)
            result.append(waypoints)

        return result

    @staticmethod
    def extract_obstacles_from_pybullet(
        assembly,
        margin: float = 0.02,
    ) -> list[ObstacleBox]:
        """Convenience helper: extract ObstacleBoxes from a CRC AssemblyPB.

        Parameters
        ----------
        assembly:
            A ``Classes.AssemblyPB.AssemblyPB`` instance with ``part_ids`` dict.
        margin:
            Extra padding added to each AABB (metres).

        Returns
        -------
        List of ObstacleBox, one per part in the assembly.
        """
        try:
            import pybullet as p
        except ImportError:
            raise RuntimeError('pybullet must be installed to use this helper.')

        obstacles = []
        for part_name, body_id in assembly.part_ids.items():
            aabb_min, aabb_max = p.getAABB(body_id)
            aabb_min = np.array(aabb_min) - margin
            aabb_max = np.array(aabb_max) + margin
            obstacles.append(ObstacleBox.from_pybullet_aabb(aabb_min, aabb_max))
        return obstacles

    # ------------------------------------------------------------------
    # Internal: loading
    # ------------------------------------------------------------------

    def _lazy_load(self):
        """Load GNN checkpoint and build virtual env on first call."""
        if self._policy_net is not None:
            return

        _ensure_roboballet(self._roboballet_root)

        import jax
        from orbax import checkpoint as orbax_ckpt

        # Import roboballet modules.
        from roboballet.agents import networks
        from roboballet.launch import hparams
        from roboballet.train import train as train_lib
        from roboballet.environments import planning_env
        from roboballet.agents.actor import load_start_states

        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.3')

        # Build configs. Use pregen=False so we don't require a pre-generated
        # start-states file locally — the env resets randomly during inference.
        self._configs = hparams.make_optimised_configs(
            num_targets=self._num_targets,
            num_obstacles=self._num_obstacles,
            num_robots=self._num_robots,
            pregen=False,
        )
        if self._env_stage == 'stage4':
            from roboballet.environments import target as target_lib
            self._configs = self._configs.replace(env_config=self._configs.env_config.replace(
                targets_have_orientation=True,
                target_scoring_config=target_lib.TargetScoringConfig(
                    distance_threshold=0.025, max_shaped_score=0.0),
                collision_penalty_scaling=15.0))

        # Find latest checkpoint.
        steps = []
        for name in os.listdir(self._checkpoint_dir):
            try:
                steps.append(int(name))
            except ValueError:
                pass
        if not steps:
            raise FileNotFoundError(
                f'No checkpoints found in {self._checkpoint_dir}')
        latest_step = sorted(steps)[-1]
        self._checkpoint_step = latest_step
        restore_path = os.path.abspath(
            os.path.join(self._checkpoint_dir, str(latest_step)))
        logger.info('GNNWaypointPlanner: loading checkpoint step %d from %s',
                    latest_step, restore_path)

        # Restore policy params.
        policy_net = networks.RoboBalletPolicyNet(
            config=self._configs.model_config)
        twin_critic_net = networks.RoboBalletTwinCriticNet(
            config=self._configs.model_config)
        example_tree = train_lib.make_example_checkpoint_tree(
            planning_env_config=self._configs.env_config,
            policy_net=policy_net,
            twin_critic_net=twin_critic_net,
            train_config=self._configs.train_config,
        )
        ckptr = orbax_ckpt.PyTreeCheckpointer()
        restored = ckptr.restore(
            restore_path,
            item={'step': 0, 'train_state': example_tree['train_state'],
                  'elapsed_time_ns': 0},
        )
        self._policy_net = policy_net
        self._policy_params = restored['train_state'].policy_params
        self._apply = jax.jit(policy_net.apply) if self._jit_policy else policy_net.apply
        self._action_scale = float(self._configs.env_config.base_config.max_joint_velocity) if self._scale_actions else 1.0

        # Build virtual env. No pregenerated start-states file is required for
        # inference — start_state=None causes the env to reset randomly, which
        # is fine since we override robot/target positions at plan time.
        start_state = None
        start_states_path = self._configs.actor_config.start_states_path
        if start_states_path is not None:
            full_path = os.path.join(self._roboballet_root,
                                     'roboballet', 'data', start_states_path)
            if os.path.exists(full_path):
                start_states = load_start_states(start_states_path)
                start_state = start_states[0]
            else:
                logger.warning(
                    'GNNWaypointPlanner: start_states file not found (%s) '
                    '— using random env reset for inference.', full_path)
        self._env = planning_env.PlanningEnv(
            self._configs.env_config, start_state=start_state)
        self._env.reset()
        logger.info('GNNWaypointPlanner: ready (checkpoint step %d)',
                    latest_step)

    # ------------------------------------------------------------------
    # Internal: coordinate transforms
    # ------------------------------------------------------------------

    def _crc_to_rb(self, pose_4x4: np.ndarray) -> np.ndarray:
        """Map a CRC world-frame 4×4 pose into the RoboBallet workspace frame.

        The RoboBallet workspace is centred on a ~1 m table; the CRC cell is
        larger.  We apply a uniform scale then centre on the midpoint between
        the two robot bases (approximately the table centre in RoboBallet).

        Override this method if your CRC cell has a specific origin offset.
        """
        out = pose_4x4.copy().astype(np.float32)
        out[:3, 3] *= self._workspace_scale
        return out

    def _rb_to_crc(self, pose_4x4: np.ndarray) -> np.ndarray:
        """Inverse of _crc_to_rb."""
        out = pose_4x4.copy().astype(np.float32)
        if self._workspace_scale != 0:
            out[:3, 3] /= self._workspace_scale
        return out

    def _obstacle_crc_to_rb(self, obs: ObstacleBox) -> ObstacleBox:
        """Map an ObstacleBox from CRC to RoboBallet coordinates."""
        return ObstacleBox(
            center=obs.center * self._workspace_scale,
            half_extents=obs.half_extents * self._workspace_scale,
            rotation=obs.rotation,
        )

    # ------------------------------------------------------------------
    # Internal: policy rollout
    # ------------------------------------------------------------------

    def _default_tip_rot(self, slot: int):
        """Tip orientation of virtual robot `slot` in its default configuration (world frame)."""
        try:
            o = self._env.observation()['robots']
            return np.asarray((o['base_poses'][slot] @ o['tip_relative_poses'][slot])[:3, :3], dtype=np.float64)
        except Exception:  # pylint: disable=broad-except
            return None

    def _build_start_state(self, poses: dict, obstacles: list, constraints: list):
        """StartState for the CRC scene: robot placements, joint configurations that put each
        active robot's tip on its current pose (IK in the virtual world), the targets, and the
        obstacles.  The training pipeline reproduces scenes with exactly this object."""
        from roboballet.environments import planning_env, target as target_lib
        env = self._env
        world = env._world  # pylint: disable=protected-access
        robots = list(env._robots)  # pylint: disable=protected-access
        # the previous rollout's obstacles are still in the world: they would make the collision-aware
        # IK below fail for poses that are free in the new scene (reset() re-creates the obstacles
        # from the start state)
        for obj_id, _ in list(env._generated_obstacles):  # pylint: disable=protected-access
            try:
                world.RemoveObject(obj_id)
            except Exception:  # pylint: disable=broad-except
                pass
        env._generated_obstacles = []  # pylint: disable=protected-access
        placements = []
        for i, robot in enumerate(robots):
            if i < len(self._virtual_base_poses) and not np.allclose(
                    np.asarray(world.GetWorldTransform(robot), dtype=np.float64),
                    self._virtual_base_poses[i], atol=1e-4):
                world.SetWorldTransform(robot, self._virtual_base_poses[i])
            placements.append(np.asarray(world.GetWorldTransform(robot), dtype=np.float32))
        ik_cfg = {'max_attempts': 30, 'max_iterations': 60}
        norm = {s_: [None if p_ is None else np.asarray(p_, dtype=np.float64) for p_ in poses[s_]]
                for s_ in poses}

        def with_roll(slot, T):
            """The virtual tip frame and the CRC flange frame agree up to a roll about z (a multiple
            of 90 deg, found by calibration): return T rolled by the first multiple that has an IK
            solution, or None."""
            for k in range(4):
                c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
                M = T.copy()
                M[:3, :3] = T[:3, :3] @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
                if list(world.ComputeIk(robots[slot], M, num_solutions=1, solver_configs=ik_cfg)):
                    return M
            return None

        # 1) every target (every rung of its ladder) must be reachable; unreachable rungs are dropped,
        #    a robot without any reachable rung is an error (checked before any joint is moved, so the
        #    answer does not depend on which arm branch an earlier robot was placed in)
        self._slot_targets = {}
        for slot in sorted(norm):
            tgt = norm[slot][1]
            if tgt is None or slot >= len(robots):
                continue
            cands = [tgt]
            if self._ladder_heights:
                cands = []
                for h in self._ladder_heights:
                    up = tgt.copy()
                    up[2, 3] += h
                    cands.append(up)
            good = []
            for c in cands:
                rolled = with_roll(slot, c)
                if rolled is not None:
                    good.append((c, rolled))
            if not good:
                if self._strict_ik:
                    raise RuntimeError(f'GNN: target {np.round(tgt[:3, 3], 3)} (or its ladder) is not '
                                       f'reachable by the virtual robot {slot} (its reach/joint limits '
                                       f'differ from the real arm)')
                good = [(tgt, tgt)]
            self._slot_targets[slot] = good
        # 2) put every active robot's tip on its current pose
        for slot in sorted(norm):
            cur = norm[slot][0]
            if cur is None or slot >= len(robots):
                continue
            rolled = with_roll(slot, cur)
            if rolled is None:
                # the arm may stand anywhere (tool storage, box): the exact start orientation is not
                # needed, only a configuration with the tip at the right place - try the target's
                # orientation and the robot's default tip orientation at that position
                for alt in [norm[slot][1]] + [self._default_tip_rot(slot)]:
                    if alt is None:
                        continue
                    M = np.eye(4)
                    M[:3, :3] = alt[:3, :3] if alt.shape == (4, 4) else alt
                    M[:3, 3] = cur[:3, 3]
                    rolled = with_roll(slot, M)
                    if rolled is not None:
                        break
            if rolled is not None:
                world.SetJointValues(robots[slot], list(world.ComputeIk(
                    robots[slot], rolled, num_solutions=1, solver_configs=ik_cfg))[0])
            elif self._strict_ik:
                raise RuntimeError(f'GNN: start pose {np.round(cur[:3, 3], 3)} is not reachable by the '
                                   f'virtual robot {slot}')
            else:
                logger.warning('GNN start state: no IK solution for virtual robot %d at %s; '
                               'using its default configuration', slot, np.round(cur[:3, 3], 3))
        joints = np.asarray(env.joint_configurations())
        targets = []
        for slot in sorted(self._slot_targets):
            for _, tgt in self._slot_targets[slot]:
                rot = np.asarray(tgt[:3, :3], dtype=np.float32) if self._use_target_orientation else None
                targets.append(target_lib.TargetInfo(
                    position=np.asarray(tgt[:3, 3], dtype=np.float32), rotation=rot))
        boxes = [planning_env.BoxInfo(spans=np.asarray(o.spans, dtype=np.float32),
                                      centroid=np.asarray(o.center, dtype=np.float32),
                                      rotation=np.asarray(o.rotation, dtype=np.float32))
                 for o in obstacles]
        for c in constraints:
            if c.constraint_type == ConstraintType.NO_GO_ZONE:
                ext = c.half_extents if c.half_extents is not None else np.array([0.05, 0.05, 0.05])
                boxes.append(planning_env.BoxInfo(spans=np.asarray(2.0 * ext, dtype=np.float32),
                                                  centroid=np.asarray(c.pose[:3, 3], dtype=np.float32),
                                                  rotation=np.asarray(c.pose[:3, :3], dtype=np.float32)))
        far = [(sx * 3.4, sy, 2.4) for sx in (-1, 1) for sy in (-5.4, 1.4)] + \
              [(sx * 3.4, -2.0, sz) for sx in (-1, 1) for sz in (0.3, 1.2, 2.4)]
        k = 0
        while len(boxes) < self._pad_obstacles_to:
            x, y, z = far[k % len(far)]
            boxes.append(planning_env.BoxInfo(spans=np.full(3, 0.05, dtype=np.float32),
                                              centroid=np.array([x, y, z + 0.02 * (k // len(far))], dtype=np.float32),
                                              rotation=np.eye(3, dtype=np.float32)))
            k += 1
        return planning_env.StartState(generated_obstacles=boxes, targets=targets,
                                       initial_joint_configurations=joints,
                                       initial_robot_placements=np.stack(placements))

    def _rollout_scene(self, poses: dict, obstacles: list, constraints: list) -> dict:
        """Roll the policy out on the CRC scene.  poses = {robot slot: (current 4x4, target 4x4)}.
        Positions come from the policy; orientation is blended from the current to the target
        orientation (the policy's tip frame is not the CRC tool frame).  A robot's path ends as soon
        as its tip is within convergence_tol of ONE OF ITS OWN targets (one rung of the ladder when
        ladder_heights is set) with that target's pose appended; a robot that never gets there keeps
        the whole rollout and the exact target is appended (a jump).  The environment itself never
        ends an episode on contact with a target.  Sets self.last_plan_info."""
        from roboballet.agents import compute_features
        from scipy.spatial.transform import Slerp
        env = self._env
        env._start_state = self._build_start_state(poses, obstacles, constraints)  # consumed by reset()
        obs = env.reset().observation
        obs_spec = env.observation_spec()
        feature_config = self._configs.model_config.feature_config
        slots = sorted(poses)
        own = {s: self._slot_targets.get(s) or [(poses[s][1], poses[s][1])] for s in slots}
        rung_pos = {s: [np.asarray(c[:3, 3]) for c, _ in own[s]] for s in slots}
        tips_traj = {s: [] for s in slots}
        arrived = {}
        steps_run = 0
        for step in range(self._max_plan_steps):
            graph_features = compute_features.make_graph_features(obs, obs_spec, feature_config)
            action = self._apply(self._policy_params, graph_features)
            action = action.squeeze(0) if action.ndim == 3 else action
            obs = env.step(np.array(action) * self._action_scale).observation
            steps_run = step + 1
            tips = {s: self._get_eef_pos_from_obs_by_index(obs, s) for s in slots}
            record = step % self._waypoint_step == 0
            for s in slots:
                if s in arrived:
                    continue
                dists = [float(np.linalg.norm(tips[s] - rp)) for rp in rung_pos[s]]
                hit = int(np.argmin(dists))
                if record or dists[hit] < self._convergence_tol:
                    tips_traj[s].append(np.array(tips[s], dtype=np.float64))
                if dists[hit] < self._convergence_tol:
                    arrived[s] = hit
            if len(arrived) == len(slots):
                break
        info = {'steps': steps_run, 'final_dist_own': {}, 'arrived': {}, 'rung_index': {}, 'assignment_ok': True}
        for s in slots:
            end = tips_traj[s][-1]
            d_own = min(float(np.linalg.norm(end - rp)) for rp in rung_pos[s])
            info['final_dist_own'][s] = d_own
            info['arrived'][s] = s in arrived
            info['rung_index'][s] = arrived.get(s)
            others = [float(np.linalg.norm(end - rp)) for t in slots if t != s for rp in rung_pos[t]]
            if s not in arrived and others and min(others) < d_own:
                info['assignment_ok'] = False
        if not info['assignment_ok']:
            logger.warning('GNN rollout: a robot ended nearer to another robot\'s target (the policy '
                           'allocated the targets differently): %s', info['final_dist_own'])
        self.last_plan_info = info
        out = {}
        for s in slots:
            cur, tgt = poses[s]
            pts = tips_traj[s]
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([cur[:3, :3], tgt[:3, :3]]))
            path = []
            for k, pos in enumerate(pts):
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = slerp([min(1.0, (k + 1) / max(len(pts), 1))]).as_matrix()[0]
                T[:3, 3] = pos
                path.append(T)
            final = own[s][arrived[s]][0] if s in arrived else tgt
            path.append(np.asarray(final, dtype=np.float32).copy())
            out[s] = path
        return out

    def _run_policy(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        obstacles: list[ObstacleBox],
        constraints: list[SceneConstraint],
    ) -> list[np.ndarray]:
        """Roll out the policy in the virtual env, return EEF pose trajectory."""
        import jax
        import jax.numpy as jnp
        from roboballet.agents import compute_features

        env = self._env
        feature_config = self._configs.model_config.feature_config
        action_scale = self._configs.env_config.base_config.max_joint_velocity

        # Teleport the virtual Panda to match the CRC robot's current position.
        self._set_virtual_robot_pose(current_pose)
        # Place target in virtual env.
        self._set_virtual_target(target_pose)
        # Replace obstacles.
        self._set_virtual_obstacles(obstacles, constraints)

        # Reset env state without changing the scene we just built.
        obs = env.reset().observation
        obs_spec = env.observation_spec()

        trajectory: list[np.ndarray] = []
        rng_key = jax.random.PRNGKey(0)

        for step in range(self._max_plan_steps):
            graph_features = compute_features.make_graph_features(
                obs, obs_spec, feature_config)
            rng_key, subkey = jax.random.split(rng_key)

            action = self._apply(self._policy_params, graph_features)
            action = action.squeeze(0) if action.ndim == 3 else action

            step_result = env.step(np.array(action) * self._action_scale)
            obs = step_result.observation

            # Record EEF pose every waypoint_step steps.
            if step % self._waypoint_step == 0:
                eef_pos = obs['robots']['tip_poses'][0, :3, 3] \
                    if 'tip_poses' in obs.get('robots', {}) \
                    else self._get_eef_pos_from_obs(obs)
                eef_pose = np.eye(4, dtype=np.float32)
                eef_pose[:3, 3] = eef_pos
                trajectory.append(eef_pose)

            # Early stopping: EEF close enough to target.
            if len(trajectory) > 0:
                dist = np.linalg.norm(
                    trajectory[-1][:3, 3] - target_pose[:3, 3])
                if dist < self._convergence_tol:
                    logger.debug('Converged at step %d (dist=%.4f m)', step, dist)
                    break

        # Always end at the exact target pose.
        trajectory.append(target_pose.copy())
        return trajectory

    def _run_policy_multi(
        self,
        current_poses: list,
        target_poses: list,
        obstacles: list,
        constraints: list,
    ) -> list:
        """Roll out the policy with N robots active, return N EEF trajectories.

        Robots 0…N-1 are teleported to current_poses and given target_poses.
        Robots N…num_robots-1 remain at their pregenerated start positions and
        act as background agents; their trajectories are discarded.
        """
        import jax
        from roboballet.agents import compute_features

        n = len(current_poses)
        env = self._env
        feature_config = self._configs.model_config.feature_config
        action_scale = self._configs.env_config.base_config.max_joint_velocity

        # Teleport each active virtual robot to match its CRC counterpart.
        for i, pose in enumerate(current_poses):
            self._set_virtual_robot_pose_by_index(i, pose)

        # Assign targets for each active robot.
        for i, pose in enumerate(target_poses):
            self._set_virtual_target_by_index(i, pose)

        # Replace obstacles.
        self._set_virtual_obstacles(obstacles, constraints)

        obs = env.reset().observation
        obs_spec = env.observation_spec()

        # One trajectory list per active robot.
        trajectories = [[] for _ in range(n)]
        rng_key = jax.random.PRNGKey(0)

        for step in range(self._max_plan_steps):
            graph_features = compute_features.make_graph_features(
                obs, obs_spec, feature_config)
            rng_key, subkey = jax.random.split(rng_key)

            action = self._apply(self._policy_params, graph_features)
            action = action.squeeze(0) if action.ndim == 3 else action

            step_result = env.step(np.array(action) * self._action_scale)
            obs = step_result.observation

            if step % self._waypoint_step == 0:
                for i in range(n):
                    eef_pos = self._get_eef_pos_from_obs_by_index(obs, i)
                    eef_pose = np.eye(4, dtype=np.float32)
                    eef_pose[:3, 3] = eef_pos
                    trajectories[i].append(eef_pose)

            # Early stopping: all active robots within tolerance of their targets.
            if all(len(trajectories[i]) > 0 for i in range(n)):
                all_converged = all(
                    np.linalg.norm(
                        trajectories[i][-1][:3, 3] - target_poses[i][:3, 3])
                    < self._convergence_tol
                    for i in range(n)
                )
                if all_converged:
                    logger.debug('plan_multi converged at step %d', step)
                    break

        # Append exact target poses as final waypoints.
        for i in range(n):
            trajectories[i].append(target_poses[i].copy())

        return trajectories

    def _get_eef_pos_from_obs_by_index(self, obs: dict, robot_idx: int) -> np.ndarray:
        """Extract EEF position for robot robot_idx from a RoboBallet observation."""
        robots = obs.get('robots', {})
        base_poses = robots.get('base_poses', None)
        tip_rel = robots.get('tip_relative_poses', None)
        if base_poses is not None and tip_rel is not None:
            tip_world = base_poses[robot_idx] @ tip_rel[robot_idx]
            return tip_world[:3, 3].astype(np.float32)
        if base_poses is not None:
            return base_poses[robot_idx, :3, 3].astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _set_virtual_robot_pose_by_index(self, robot_idx: int, pose_4x4: np.ndarray):
        """Teleport virtual robot at robot_idx to approximately match pose_4x4."""
        try:
            world = self._env._world  # pylint: disable=protected-access
            robots_list = list(world._robots.keys()) \
                if hasattr(world, '_robots') \
                else self._env._robots
            if robot_idx >= len(robots_list):
                logger.debug('_set_virtual_robot_pose_by_index: index %d out of '
                             'range (%d robots)', robot_idx, len(robots_list))
                return
            robot = robots_list[robot_idx]
            solutions = list(world.ComputeIk(
                robot, pose_4x4,
                solver_configs={'max_attempts': 3, 'max_iterations': 20}))
            if solutions:
                world.SetJointValues(robot, solutions[0])
        except Exception as e:  # pylint: disable=broad-except
            logger.debug('_set_virtual_robot_pose_by_index(%d): IK failed (%s), '
                         'leaving default config', robot_idx, e)

    def _set_virtual_target_by_index(self, target_idx: int, target_pose: np.ndarray):
        """Set virtual target at target_idx to target_pose."""
        try:
            from roboballet.environments import target as target_lib
            env = self._env
            if target_idx < len(env._targets):  # pylint: disable=protected-access
                target_info, _ = env._targets[target_idx]  # pylint: disable=protected-access
                new_info = target_info.replace(pose=target_pose)
                env._targets[target_idx] = (  # pylint: disable=protected-access
                    new_info, target_lib.TargetState())
        except Exception as e:  # pylint: disable=broad-except
            logger.debug('_set_virtual_target_by_index(%d) failed: %s',
                         target_idx, e)

    def _get_eef_pos_from_obs(self, obs: dict) -> np.ndarray:
        """Extract first robot's EEF position from a RoboBallet observation."""
        robots = obs.get('robots', {})
        # tip_relative_poses is relative to base; base_poses is absolute.
        base_poses = robots.get('base_poses', None)
        tip_rel = robots.get('tip_relative_poses', None)
        if base_poses is not None and tip_rel is not None:
            # tip_world = base @ tip_relative
            tip_world = base_poses[0] @ tip_rel[0]
            return tip_world[:3, 3].astype(np.float32)
        # Fallback: use base position.
        if base_poses is not None:
            return base_poses[0, :3, 3].astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _set_virtual_robot_pose(self, pose_4x4: np.ndarray):
        """Teleport the virtual Panda to approximately match the given EEF pose.

        Uses the env's IK solver if available; otherwise leaves the robot at its
        default configuration (the GNN will still plan from the correct target).
        """
        try:
            world = self._env._world  # pylint: disable=protected-access
            robot = list(world._robots.keys())[0] \
                if hasattr(world, '_robots') \
                else self._env._robots[0]
            target_quat = Rotation.from_matrix(
                pose_4x4[:3, :3]).as_quat()  # scalar last
            target_quat_mj = np.array([target_quat[3],
                                       target_quat[0],
                                       target_quat[1],
                                       target_quat[2]], dtype=np.float32)
            solutions = list(world.ComputeIk(
                robot, pose_4x4, solver_configs={'max_attempts': 3,
                                                 'max_iterations': 20}))
            if solutions:
                world.SetJointValues(robot, solutions[0])
        except Exception as e:  # pylint: disable=broad-except
            logger.debug('_set_virtual_robot_pose: IK failed (%s), '
                         'leaving default config', e)

    def _set_virtual_target(self, target_pose: np.ndarray):
        """Set the first target in the virtual env to the given pose."""
        try:
            from roboballet.environments import target as target_lib
            env = self._env
            if env._targets:  # pylint: disable=protected-access
                target_info, _ = env._targets[0]  # pylint: disable=protected-access
                new_info = target_info.replace(pose=target_pose)
                env._targets[0] = (new_info, target_lib.TargetState())  # pylint: disable=protected-access
        except Exception as e:  # pylint: disable=broad-except
            logger.debug('_set_virtual_target failed: %s', e)

    def _set_virtual_obstacles(
        self,
        obstacles: list[ObstacleBox],
        constraints: list[SceneConstraint],
    ):
        """Replace generated obstacles in the virtual env with CRC obstacles.

        Constraints are injected as additional obstacles for now (no-go zones as
        solid boxes, others as zero-size markers).  When the GNN is retrained
        with constraint-aware rewards, this method should encode each constraint
        type as its intended graph node.
        """
        try:
            from roboballet.environments import data_pb2
            env = self._env
            world = env._world  # pylint: disable=protected-access

            # Remove old generated obstacles.
            for obj_id, _ in env._generated_obstacles:  # pylint: disable=protected-access
                try:
                    world.RemoveObject(obj_id)
                except Exception:
                    pass
            env._generated_obstacles = []  # pylint: disable=protected-access

            # Add CRC obstacles as boxes.
            for obs_box in obstacles:
                box_id = world.AddBox(obs_box.spans)
                world.SetWorldTransform(box_id, obs_box.pose_4x4)
                env._generated_obstacles.append((box_id, obs_box))  # pylint: disable=protected-access

            # Encode no-go constraints as solid obstacle boxes.
            for constraint in constraints:
                if constraint.constraint_type == ConstraintType.NO_GO_ZONE:
                    extents = constraint.half_extents if constraint.half_extents \
                        is not None else np.array([0.05, 0.05, 0.05])
                    box_id = world.AddBox(2.0 * extents)
                    world.SetWorldTransform(box_id, constraint.pose)
                    placeholder = ObstacleBox(
                        center=constraint.pose[:3, 3],
                        half_extents=extents,
                        rotation=constraint.pose[:3, :3])
                    env._generated_obstacles.append((box_id, placeholder))  # pylint: disable=protected-access
                    logger.debug('Encoded NO_GO_ZONE constraint as obstacle box')
                else:
                    # Future constraint types: log and skip until retrained.
                    logger.info(
                        'Constraint type %s is not yet encoded in the graph. '
                        'Add a reward term and retrain to enable reactive '
                        'behaviour.', constraint.constraint_type.name)
        except Exception as e:  # pylint: disable=broad-except
            logger.debug('_set_virtual_obstacles failed: %s', e)

    # ------------------------------------------------------------------
    # Internal: trajectory → G-code
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: trajectory → G-code
    # ------------------------------------------------------------------

    def _trajectory_to_gcode(
        self,
        trajectory: list[np.ndarray],
    ) -> list[dict]:
        """Convert a list of 4×4 EEF poses to GCodeProcessor-compatible dicts."""
        waypoints = []
        for pose in trajectory:
            pos = pose[:3, 3]
            rot = Rotation.from_matrix(pose[:3, :3])
            euler_xyz = rot.as_euler('xyz', degrees=False)  # radians
            waypoints.append({
                'G': 1,
                'X': float(pos[0]),
                'Y': float(pos[1]),
                'Z': float(pos[2]),
                'A': float(euler_xyz[0]),
                'B': float(euler_xyz[1]),
                'C': float(euler_xyz[2]),
                'F': self._feed_forward_speed,
            })
        return waypoints


class GNNPlannerRegistry:
    """Manages named GNNWaypointPlanner instances, one per task type.

    Task types
    ----------
    'independent':
        Independent multi-target reaching — used for screw approach moves and
        any single-robot or dual-robot move where each robot has its own target.
        Checkpoint: Stage 4 (step 1 730 000).

    'cooperative':
        Synchronized rigid-body transport — used when both robots must maintain
        a fixed relative pose while moving (e.g. battery cover carry).
        Checkpoint: Stage 7 (not yet trained; configure when available).

    Planners are lazy-loaded on first use so importing the registry does not
    trigger JAX initialisation or checkpoint loading.

    Parameters
    ----------
    checkpoint_dirs:
        Dict mapping task name → absolute path to the checkpoint directory.
        A task whose path is None or points to a non-existent directory is not
        available; calling plan/plan_multi with that task raises RuntimeError.
    **shared_kwargs:
        Keyword arguments forwarded to every GNNWaypointPlanner constructor
        (roboballet_root, num_robots, num_targets, num_obstacles, …).
    """

    TASK_INDEPENDENT = 'independent'
    TASK_COOPERATIVE = 'cooperative'

    def __init__(self, checkpoint_dirs: dict[str, str | None], **shared_kwargs):
        self._checkpoint_dirs = checkpoint_dirs
        self._shared_kwargs = shared_kwargs
        self._planners: dict[str, GNNWaypointPlanner] = {}

    def _get(self, task: str) -> GNNWaypointPlanner:
        if task not in self._planners:
            ckpt_dir = self._checkpoint_dirs.get(task)
            if not ckpt_dir or not os.path.isdir(ckpt_dir):
                raise RuntimeError(
                    f'GNNPlannerRegistry: no valid checkpoint configured for '
                    f'task={task!r}. '
                    f'Pass --checkpoint_dir_{task} to set it.')
            self._planners[task] = GNNWaypointPlanner(
                checkpoint_dir=ckpt_dir, **self._shared_kwargs)
            logger.info('GNNPlannerRegistry: loaded %r planner from %s',
                        task, ckpt_dir)
        return self._planners[task]

    def plan(self, *args, task: str = TASK_INDEPENDENT, **kwargs) -> list[dict]:
        """Delegate to the planner for *task* (default: 'independent')."""
        return self._get(task).plan(*args, **kwargs)

    def plan_multi(self, *args, task: str = TASK_INDEPENDENT, **kwargs) -> list:
        """Delegate to the planner for *task* (default: 'independent')."""
        return self._get(task).plan_multi(*args, **kwargs)

    def available_tasks(self) -> list[str]:
        """Return tasks that have a valid checkpoint directory configured."""
        return [t for t, d in self._checkpoint_dirs.items()
                if d and os.path.isdir(d)]

    def __repr__(self):
        avail = self.available_tasks()
        return (f'GNNPlannerRegistry(available={avail}, '
                f'loaded={list(self._planners)})')
