"""CuMotionWaypointPlanner — NVIDIA cuMotion GPU planner for DisassemblySim.

Architecture
------------
cuMotion is a GPU-accelerated motion planner that works in joint space.  Given
a start joint configuration and a goal EEF pose it returns a collision-free
joint trajectory.  This class wraps that API in the same waypoint-list interface
as GNNWaypointPlanner so DisassemblySim can switch planners via set_planner().

cuMotion always plans in robot-local frame (robot base at origin).  The
robot_base_pose parameter (4×4 world transform of the robot base) is used to
transform targets and obstacles into local frame before planning, and to
transform the output trajectory back to world frame.  Callers always work in
world/cell coordinates — the frame conversion is handled internally.

Single-robot flow (plan)
------------------------
  1. Transform target EEF pose from world to robot-local frame.
  2. Convert to cuMotion Pose3 and build world from obstacle list.
  3. Call planner → joint trajectory (N × DOF).
  4. FK each config → EEF pose (local) → world frame → G-code waypoint dict.

Multi-robot flow (plan_multi) — used by the `reactive` execution layer
------------------------------------------------------------------------
  Sequential decoupled replanning: each robot re-plans every
  replan_interval_steps simulation steps, treating the other robots' live
  AABBs as dynamic obstacles.

    for cycle in range(max_replan_cycles):
        for each robot i (not yet arrived):
            transform obstacles + target to robot-i local frame
            plan robot_i → chunk of joint configs
            update current config and world-frame EEF position
            check convergence (EEF vs target, convergence_tol)
        if all arrived: break

  GPU replanning is fast (milliseconds per call) so short intervals are fine.

  This is the *live, reactive* coordination path (`--execution reactive`, the
  default). A second, orthogonal execution layer, `--execution tpg`
  (modular_planner/tpg.py), instead rehearses each arm's plan offline once and
  runs the two arms asynchronously with rendezvous only at classified
  conflicts — it calls into this class for the per-segment `plan()` calls but
  does not go through plan_multi's decoupled-replanning loop.

Planning-only artifacts
------------------------
  `_planning_urdf()` writes a `*_plan.urdf` sibling of a robot's URDF with
  multi-turn wrist joint limits clamped to +-wrist_limit_rad — used only for
  planning (keeps cuMotion from proposing multi-turn wrist solutions), never
  for the kinematics the rest of the sim reports. `_urdf_overrides` lets a
  specific robot name load its own URDF instead of the shared default (per-arm
  URDFs), via `_select(robot_name)`. `_build_world()` caches built
  `cumotion.World` objects by obstacle geometry — during a dry rehearsal the
  same static scene is rebuilt for every plan/IK/start-diagnosis call, so
  caching it is a significant speedup, not just a micro-optimisation.

Robot descriptor files
----------------------
The Comau NJ290 is not shipped with cuMotion's built-in library.  The required
files are in robot_descriptions/:
  comau_nj290.xrdf                   — tool frame, cspace joint order
  comau_nj290_planner_config.yaml    — task-space limits, collision spheres
  comau_nj290_robotNC_sensor.urdf    — URDF used by cuMotion for FK/IK

cuMotion documentation: https://nvidia-isaac.github.io/cumotion/
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from modular_planner.scene import ObstacleBox, SceneConstraint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard — cumotion is only available in .venv (Python 3.14, cp314 wheel).
# ---------------------------------------------------------------------------
_cumotion_imported = False
_cumotion_error: Optional[Exception] = None


def _ensure_cumotion():
    global _cumotion_imported, _cumotion_error
    if _cumotion_imported:
        return
    try:
        import cumotion  # noqa: F401
        _cumotion_imported = True
    except Exception as e:
        _cumotion_error = e
        raise RuntimeError(
            'Failed to import cumotion.  Make sure .venv (Python 3.14, cp314 wheel) '
            f'is active.\n{e}'
        ) from e


# ---------------------------------------------------------------------------

class CuMotionWaypointPlanner:
    """GPU motion planner for the CRC disassembly cell using NVIDIA cuMotion.

    Parameters
    ----------
    urdf_path:
        Absolute path to the Comau NJ290 URDF.
        (robot_descriptions/comau_nj290_robotNC_sensor.urdf)
    xrdf_path:
        Absolute path to the matching XRDF file (cuMotion robot descriptor).
        Must be created for the Comau — not shipped with cuMotion by default.
    planning_config_path:
        Absolute path to the cuMotion planner config YAML.
    tool_frame:
        Name of the end-effector frame in the URDF/XRDF (default 'tool0').
    replan_interval_steps:
        In plan_multi(), how many PyBullet steps to execute before replanning.
        Smaller → more reactive, more GPU calls.  Default 10 (~33 ms at 300 Hz).
    max_replan_cycles:
        Safety cap on replanning iterations in plan_multi().
    convergence_tol:
        EEF distance (metres) at which a robot is considered to have arrived.
    feed_forward_speed:
        G-code feed rate assigned to all waypoints.
    waypoint_step:
        Only emit every Nth joint config as a waypoint (reduces waypoint count).
    """

    _ROBOT_DESCRIPTIONS_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'robot_descriptions'))

    _DEFAULT_URDF   = os.path.join(_ROBOT_DESCRIPTIONS_DIR,
                                   'comau_nj290_robotNC_sensor.urdf')
    _DEFAULT_XRDF   = os.path.join(_ROBOT_DESCRIPTIONS_DIR,
                                   'comau_nj290.xrdf')
    _DEFAULT_CONFIG = os.path.join(_ROBOT_DESCRIPTIONS_DIR,
                                   'comau_nj290_planner_config.yaml')

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        xrdf_path: Optional[str] = None,
        planning_config_path: Optional[str] = None,
        tool_frame: str = 'tcp_frame',
        replan_interval_steps: int = 10,
        max_replan_cycles: int = 200,
        convergence_tol: float = 0.02,
        feed_forward_speed: float = 1000.0,
        waypoint_step: int = 3,
        urdf_overrides: Optional[dict] = None,
        prefer_joint_goal: bool = True,
        wrist_limit_rad: Optional[float] = 3.25,
        cspace_max_iterations: int = 4000,
        coupled_limit: Optional[dict] = None,
    ):
        # Optional coupled limit between two joints (NJ290 parallelogram, axes
        # 2/3):  lo <= q[j] - k*q[i] <= hi  in RADIANS, e.g.
        # {'i': 1, 'j': 2, 'k': 1.0, 'lo': -2.8, 'hi': 0.2}.  Applied to IK
        # solutions (goals); planned paths are checked and reported.
        # one dict or a list of dicts {i, j, k, lo, hi}: lo <= q[j] - k*q[i] <= hi
        self._coupled_limit = ([coupled_limit] if isinstance(coupled_limit, dict)
                               else list(coupled_limit or []))
        # The NJ290 URDF allows q4 +-31 rad and q6 +-47 rad (multi-turn).  The
        # real cell's dress-pack cables allow +-360 deg on axes 4 and 6 and
        # +-360 deg on their sum; letting the RRT explore beyond that produced
        # 9-13 rad wrist wind-ups per move.  The PLANNING model clamps those
        # joints to +-wrist_limit_rad (3.25 rad = 186 deg: every orientation
        # stays reachable, each axis < 360 deg and the sum < 360 deg by
        # construction; a little over 180 deg so that a start configuration
        # left at exactly +-pi by the tool-fetch IK is still a valid start —
        # at 3.1 that cost ~200 "Invalid initial robot configuration" cycles).
        # None = leave the URDF limits.  Sim/robot URDFs untouched.
        self._wrist_limit = wrist_limit_rad
        # Budget for the joint-goal attempt before falling back to the pose
        # target (one unlucky joint goal once cost 320 s of exhaustive search).
        self._cspace_max_iterations = cspace_max_iterations
        # prefer_joint_goal=False restores the original behaviour: cuMotion's
        # pose-target planner chooses the goal IK itself (may pick another
        # wrist branch / extra wrist turns).
        self._prefer_joint_goal = prefer_joint_goal
        # robot_name -> URDF for arms that differ from the default model (the
        # cell's robot_C has no force/torque sensor stack: its tcp_frame is
        # 9 cm closer to the flange and yawed 90 deg relative to the sensor URDF).
        self._urdf_overrides = {k: os.path.abspath(v) for k, v in (urdf_overrides or {}).items()}
        self._models = {}          # urdf path -> (robot_description, kinematics, inspector)
        self._urdf_path = os.path.abspath(urdf_path or self._DEFAULT_URDF)
        self._xrdf_path = os.path.abspath(xrdf_path or self._DEFAULT_XRDF)
        self._planning_config_path = os.path.abspath(
            planning_config_path or self._DEFAULT_CONFIG)
        self._tool_frame = tool_frame
        self._replan_interval_steps = replan_interval_steps
        self._max_replan_cycles = max_replan_cycles
        self._convergence_tol = convergence_tol
        self._feed_forward_speed = feed_forward_speed
        self._waypoint_step = waypoint_step

        # Lazy-loaded on first plan() call.
        self._robot_description = None
        self._kinematics = None
        self._base_world = None  # cumotion.World with static scene obstacles
        self._inspector = None   # RobotWorldInspector, for the robot's own sphere model
        self.last_joint_path = None  # joint configs of the last plan(), one per waypoint

    # ------------------------------------------------------------------
    # Public API — same waypoint-list interface as GNNWaypointPlanner
    # ------------------------------------------------------------------

    def robot_spheres(self, conf: np.ndarray, robot_name: Optional[str] = None) -> List[tuple]:
        """cuMotion's world-collision spheres of this robot at joint config `conf`.

        Returns [(centre_in_base_frame, radius), ...].  This is the tightest
        possible stand-in for an arm as an obstacle to another planner: the
        very spheres cuMotion checks against the world.  A per-link AABB of an
        oblique 1.2 m link is a huge box and makes a start state 20 cm from
        the other arm look like a collision.
        """
        self._lazy_load()
        import cumotion
        if robot_name is not None:
            self._select(robot_name)
        if self._inspector is None:
            self._inspector = cumotion.create_robot_world_inspector(self._robot_description)
            self._models[self._current_urdf] = (self._robot_description, self._kinematics, self._inspector)
        q = np.asarray(conf, dtype=np.float64)
        pos = self._inspector.world_collision_sphere_positions(q)
        rad = self._inspector.world_collision_sphere_radii()
        return [(np.asarray(c, dtype=np.float64).reshape(3), float(r)) for c, r in zip(pos, rad)]

    def ik(self, target_eef_pose: np.ndarray, robot_name: str = 'robot_D',
           seed: Optional[np.ndarray] = None,
           robot_base_pose: Optional[np.ndarray] = None,
           limit_margin: float = 0.05,
           obstacles: Optional[List[ObstacleBox]] = None):
        """cuMotion inverse kinematics for one tool-frame pose (world frame if
        robot_base_pose is given), rejecting self-colliding solutions and
        solutions closer than `limit_margin` rad to a joint limit.

        Returns the joint vector or None.  Cheap compared with plan(): use it
        to test whether a goal is reachable BEFORE asking for a path.
        """
        self._lazy_load()
        self._select(robot_name)
        import cumotion
        T = target_eef_pose if robot_base_pose is None else np.linalg.inv(robot_base_pose) @ target_eef_pose
        cfg = cumotion.IkConfig()
        if seed is not None:
            cfg.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
        res = cumotion.solve_ik(self._kinematics, self._mat4_to_pose3(T), self._tool_frame, cfg)
        if not res.success:
            return None
        q = np.array(res.cspace_position, dtype=np.float64).reshape(-1).copy()
        if seed is not None:
            # Multi-turn wrist joints (q4, q6: +-31 / +-47 rad in the URDF):
            # FK is 2*pi-periodic, so bring them to the turn nearest the seed
            # instead of letting IK hand back a solution 3 turns away.
            seed = np.asarray(seed, dtype=np.float64)
            for i in range(len(q)):
                lim = self._kinematics.cspace_coord_limits(i)
                if lim.upper - lim.lower > 2.5 * np.pi:
                    q[i] = q[i] + 2.0 * np.pi * round((seed[i] - q[i]) / (2.0 * np.pi))
        if self._inspector is None:
            self._inspector = cumotion.create_robot_world_inspector(self._robot_description)
            self._models[self._current_urdf] = (self._robot_description, self._kinematics, self._inspector)
        if self._inspector.in_self_collision(q):
            return None
        for i in range(len(q)):
            lim = self._kinematics.cspace_coord_limits(i)
            if q[i] < lim.lower + limit_margin or q[i] > lim.upper - limit_margin:
                return None
        if not self.coupled_ok(q):
            return None
        if obstacles:
            # World collision of the SOLUTION (not just self-collision): a
            # wrist pose that puts link6 into the board is no use as a goal.
            T_inv = np.linalg.inv(robot_base_pose) if robot_base_pose is not None else None
            local = [o.in_frame(T_inv) for o in obstacles] if T_inv is not None else list(obstacles)
            insp = self._world_inspector(local)
            if insp.in_collision_with_obstacle(q):
                return None
        return q

    def _world_inspector(self, obstacles_local):
        """RobotWorldInspector for the current model + this obstacle set (cached
        with the world: building one costs more than the query)."""
        import cumotion
        world, world_view = self._build_world(obstacles_local, [])
        cache = getattr(self, '_inspector_cache', None)
        if cache is None:
            cache = self._inspector_cache = {}
        key = (self._current_urdf, id(world))
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = cumotion.create_robot_world_inspector(self._robot_description, world_view)
        return cache[key]

    def coupled_ok(self, q) -> bool:
        for c in self._coupled_limit:
            v = float(q[c['j']]) - c['k'] * float(q[c['i']])
            if not (c['lo'] <= v <= c['hi']):
                return False
        return True

    def coupled_violations(self, q) -> list:
        """Human-readable list of the coupled limits `q` breaks."""
        out = []
        for c in self._coupled_limit:
            v = float(q[c['j']]) - c['k'] * float(q[c['i']])
            if not (c['lo'] <= v <= c['hi']):
                out.append(f'{c.get("name", "coupled limit")} q{c["j"] + 1}{-c["k"]:+g}*q{c["i"] + 1}='
                           f'{np.degrees(v):+.0f} deg outside '
                           f'[{np.degrees(c["lo"]):+.0f}, {np.degrees(c["hi"]):+.0f}]')
        return out

    def diagnose_state(self, q: np.ndarray, robot_name: str,
                       obstacles: Optional[List[ObstacleBox]] = None,
                       robot_base_pose: Optional[np.ndarray] = None) -> str:
        """Why cuMotion would reject `q` as a start state: joint limits,
        self-collision (which frames) or world collision (how deep)."""
        self._lazy_load()
        self._select(robot_name)
        import cumotion
        q = np.asarray(q, dtype=np.float64)
        out = []
        for i in range(len(q)):
            lim = self._kinematics.cspace_coord_limits(i)
            if q[i] < lim.lower or q[i] > lim.upper:
                out.append(f'q{i + 1}={q[i]:+.3f} outside [{lim.lower:+.2f}, {lim.upper:+.2f}]')
        if self._inspector is None:
            self._inspector = cumotion.create_robot_world_inspector(self._robot_description)
            self._models[self._current_urdf] = (self._robot_description, self._kinematics, self._inspector)
        if self._inspector.in_self_collision(q):
            out.append('self-collision ' + str(self._inspector.frames_in_self_collision(q)))
        out.extend(self.coupled_violations(q))
        if obstacles:
            T_inv = np.linalg.inv(robot_base_pose) if robot_base_pose is not None else None
            local = [o.in_frame(T_inv) for o in obstacles] if T_inv is not None else list(obstacles)
            insp = self._world_inspector(local)
            if insp.in_collision_with_obstacle(q):
                out.append(f'in collision with an obstacle (min distance {insp.min_distance_to_obstacle(q):+.3f} m)')
        return '; '.join(out) if out else 'valid'

    def plan(
        self,
        current_eef_pose: np.ndarray,
        target_eef_pose: np.ndarray,
        obstacles: List[ObstacleBox],
        constraints: Optional[List[SceneConstraint]] = None,
        robot_name: str = 'robot_D',
        start_conf: Optional[np.ndarray] = None,
        robot_base_pose: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Plan a single-robot collision-free path and return G-code waypoints.

        Parameters
        ----------
        current_eef_pose:
            4×4 matrix of the current EEF pose in CRC world frame (metres).
            Used only for logging; cuMotion needs start_conf for planning.
        target_eef_pose:
            4×4 matrix of the desired EEF pose.
        obstacles:
            ObstacleBox list from the live PyBullet scene.
        constraints:
            SceneConstraint list.  NO_GO_ZONE constraints are added as cuboid
            obstacles; other types are logged and skipped until cuMotion
            supports them natively.
        robot_name:
            Label for log messages.
        start_conf:
            Starting joint configuration (DOF-length array).  If None, the
            robot_description's default config is used — pass the actual
            PyBullet joint positions for accurate planning.

        Returns
        -------
        List of G-code waypoint dicts compatible with GCodeProcessor.
        """
        self._lazy_load()
        self._select(robot_name)
        import cumotion

        constraints = constraints or []

        # cuMotion always plans in robot-local frame (robot base at origin).
        # If the robot is placed at a non-identity world pose, transform the
        # target and obstacles into local frame, then transform output back.
        if robot_base_pose is not None:
            T_inv = np.linalg.inv(robot_base_pose)
            target_local = T_inv @ target_eef_pose
            obstacles_local = [obs.in_frame(T_inv) for obs in obstacles]
        else:
            target_local = target_eef_pose
            obstacles_local = obstacles

        world, world_view = self._build_world(obstacles_local, constraints)

        goal_pose = self._mat4_to_pose3(target_local)
        if start_conf is None:
            start_conf = self._robot_description.default_cspace_configuration()

        planner_config = cumotion.create_motion_planner_config_from_file(
            self._planning_config_path,
            self._robot_description,
            self._tool_frame,
            world_view,
        )
        planner = cumotion.create_motion_planner(planner_config)
        # Prefer a JOINT goal chosen near the start (same wrist branch, no
        # wind-up) over letting the pose planner pick any IK solution: the
        # pose target once returned q4 = -17 rad / q6 = +10 rad (three wrist
        # turns) for a 1 m flight to the box.
        results = None
        # Wrist joints beyond the (clamped) planning limits: same pose one
        # turn back — the sim's continuous joints drift, the planner's do not.
        start_conf = np.array(start_conf, dtype=np.float64).reshape(-1).copy()
        for i in range(len(start_conf)):
            lim = self._kinematics.cspace_coord_limits(i)
            while start_conf[i] > lim.upper and start_conf[i] - 2 * np.pi >= lim.lower:
                start_conf[i] -= 2 * np.pi
            while start_conf[i] < lim.lower and start_conf[i] + 2 * np.pi <= lim.upper:
                start_conf[i] += 2 * np.pi
        why = self.diagnose_state(start_conf, robot_name, obstacles_local)
        if why != 'valid':
            logger.warning('[CuMotion] %s: START state rejected: %s', robot_name, why)
            print(f'[CuMotion] robot start state is invalid for {robot_name}: {why}', flush=True)
        q_goal = self.ik(target_local, robot_name=robot_name, seed=start_conf,
                         obstacles=obstacles_local) if self._prefer_joint_goal else None
        if q_goal is not None:
            # Bounded attempt at the joint goal, then the pose target.
            cfg_fast = cumotion.create_motion_planner_config_from_file(
                self._planning_config_path, self._robot_description, self._tool_frame, world_view)
            try:
                cfg_fast.set_param('max_iterations',
                                   cumotion.MotionPlannerConfig.ParamValue(int(self._cspace_max_iterations)))
            except Exception as e:
                logger.debug('[CuMotion] max_iterations not settable: %s', e)
            results = cumotion.create_motion_planner(cfg_fast).plan_to_cspace_target(
                np.asarray(start_conf, dtype=np.float64), q_goal, True)
            if not results.path_found:
                logger.info('[CuMotion] %s: cspace goal unreachable, trying pose target', robot_name)
                results = None
        if results is None:
            results = planner.plan_to_pose_target(start_conf, goal_pose, True)

        if not results.path_found:
            raise RuntimeError(
                f'[CuMotion] No path found for {robot_name} '
                f'to target {target_eef_pose[:3, 3]}')

        joint_traj = np.array(results.interpolated_path)
        logger.info('[CuMotion] %s: planned trajectory with %d configs',
                    robot_name, len(joint_traj))
        # Kept for the caller: one joint configuration per returned waypoint,
        # so the arm's collision spheres can be evaluated along the path
        # (offline pair safety classification) without re-solving IK.
        self.last_joint_path = [np.array(q, dtype=np.float64) for i, q in enumerate(joint_traj)
                                if i % self._waypoint_step == 0 or i == len(joint_traj) - 1]
        if self._coupled_limit:
            bad = sum(1 for q in joint_traj if not self.coupled_ok(q))
            if bad:
                print(f'[CuMotion] {robot_name}: {bad}/{len(joint_traj)} path configurations violate a '
                      f'coupled axis limit (parallelogram A3-A2 or wrist cable A4+A6)', flush=True)

        eef_traj = self._joint_traj_to_eef(joint_traj, robot_base_pose)
        return self._trajectory_to_gcode(eef_traj)

    def plan_multi(
        self,
        current_eef_poses: List[np.ndarray],
        target_eef_poses: List[np.ndarray],
        obstacles: List[ObstacleBox],
        constraints: Optional[List[SceneConstraint]] = None,
        robot_names: Optional[List[str]] = None,
        start_confs: Optional[List[Optional[np.ndarray]]] = None,
        robot_aabb_getters=None,
        robot_base_poses: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[List[dict]]:
        """Plan simultaneous paths for N robots using a replanning loop.

        Each robot is planned independently, treating the others' current
        bounding boxes as dynamic obstacles.  After every replan_interval_steps
        PyBullet steps, both robots' positions are updated and new trajectories
        are computed.  This makes them responsive to each other in real time.

        Parameters
        ----------
        current_eef_poses:
            List of 4×4 EEF poses, one per robot.
        target_eef_poses:
            List of 4×4 target poses, same order.
        obstacles:
            Static ObstacleBox list (assembly parts etc.).
        constraints:
            Optional SceneConstraint list.
        robot_names:
            Labels for log messages.
        start_confs:
            List of starting joint configs, one per robot.  None entries fall
            back to robot_description default.
        robot_aabb_getters:
            List of callables ``() -> (aabb_min, aabb_max)`` that return the
            live AABB of each robot in world frame.  Used to update inter-robot
            obstacle boxes every replan cycle.  If None, the initial
            current_eef_poses are used as point obstacles (less accurate).

        Returns
        -------
        List of G-code waypoint lists, one per robot.
        """
        self._lazy_load()

        n = len(current_eef_poses)
        assert len(target_eef_poses) == n
        constraints = constraints or []
        robot_names = robot_names or [f'robot_{i}' for i in range(n)]
        start_confs = start_confs or [None] * n
        robot_base_poses = robot_base_poses or [None] * n

        # Accumulated waypoints per robot across all replan cycles.
        all_waypoints: List[List[dict]] = [[] for _ in range(n)]
        self.last_joint_paths_multi = [[] for _ in range(n)]   # one config per waypoint, per robot
        current_confs = list(start_confs)
        current_poses = list(current_eef_poses)
        arrived = [False] * n

        for cycle in range(self._max_replan_cycles):
            if all(arrived):
                break

            # For each robot, build dynamic obstacle boxes from the others.
            robot_obstacles = self._build_inter_robot_obstacles(
                current_poses, robot_aabb_getters)

            for i in range(n):
                if arrived[i]:
                    continue

                self._select(robot_names[i])
                # Static obstacles + all other robots' boxes, in robot-i local frame.
                T_base_i = robot_base_poses[i]
                T_inv_i = np.linalg.inv(T_base_i) if T_base_i is not None else None

                if T_inv_i is not None:
                    static_local = [obs.in_frame(T_inv_i) for obs in obstacles]
                    dynamic_local = [obs.in_frame(T_inv_i) for obs in robot_obstacles[i]]
                    target_local_i = T_inv_i @ target_eef_poses[i]
                else:
                    static_local = obstacles
                    dynamic_local = robot_obstacles[i]
                    target_local_i = target_eef_poses[i]

                world, world_view = self._build_world(
                    static_local + dynamic_local, constraints)

                goal_pose = self._mat4_to_pose3(target_local_i)
                if current_confs[i] is None:
                    current_confs[i] = \
                        self._robot_description.default_cspace_configuration()

                import cumotion
                planner_config = cumotion.create_motion_planner_config_from_file(
                    self._planning_config_path,
                    self._robot_description,
                    self._tool_frame,
                    world_view,
                )
                planner = cumotion.create_motion_planner(planner_config)
                results = planner.plan_to_pose_target(
                    current_confs[i], goal_pose, True)

                if not results.path_found:
                    logger.warning(
                        '[CuMotion] Cycle %d: no path for %s, skipping.',
                        cycle, robot_names[i])
                    continue

                joint_traj = np.array(results.interpolated_path)
                # Only take the first replan_interval_steps configs of this
                # trajectory; we'll replan after executing them.
                chunk = joint_traj[:self._replan_interval_steps]

                eef_chunk = self._joint_traj_to_eef(chunk, T_base_i)
                waypoints = self._trajectory_to_gcode(eef_chunk)
                all_waypoints[i].extend(waypoints)
                self.last_joint_paths_multi[i].extend(
                    [np.array(q, dtype=np.float64) for k, q in enumerate(chunk)
                     if k % self._waypoint_step == 0 or k == len(chunk) - 1])

                # Update current config to end of executed chunk.
                current_confs[i] = chunk[-1]
                # current_poses tracks world-frame EEF (used for AABB fallback).
                current_poses[i] = self._conf_to_eef_mat4(chunk[-1], T_base_i)

                # current_poses[i] is now world-frame; compare against world target.
                dist = np.linalg.norm(
                    current_poses[i][:3, 3] - target_eef_poses[i][:3, 3])
                if dist < self._convergence_tol:
                    arrived[i] = True
                    logger.debug('[CuMotion] %s arrived (dist=%.4f m)',
                                 robot_names[i], dist)

        # Append exact target poses as final waypoints for robots that arrived.
        for i in range(n):
            if arrived[i]:
                all_waypoints[i].append(
                    self._pose_mat4_to_gcode(target_eef_poses[i]))

        for i, name in enumerate(robot_names):
            logger.info('[CuMotion] plan_multi: %s → %d waypoints',
                        name, len(all_waypoints[i]))
            print(f'  [{name}] arrived={arrived[i]}  cycles used={cycle+1}')

        return all_waypoints

    def plan_cooperative_lift(
        self,
        current_eef_poses: List[np.ndarray],
        target_eef_pose_A: np.ndarray,
        obstacles: List[ObstacleBox],
        constraints: Optional[List[SceneConstraint]] = None,
        T_rel: Optional[np.ndarray] = None,
        robot_names: Optional[List[str]] = None,
        start_confs: Optional[List[Optional[np.ndarray]]] = None,
        robot_aabb_getters=None,
        robot_base_poses: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[List[dict]]:
        """Plan a cooperative lift where both arms move while maintaining relative pose.

        Arm A is the leader: it plans independently to target_A.
        Arm B is the follower: at each replan cycle its virtual target is
        ``current_pose_A @ T_rel``, so it shadows arm A during the approach.
        Once arm A arrives at target_A, arm B's target becomes the fixed
        ``target_A @ T_rel`` and it converges there while arm A holds position.

        Parameters
        ----------
        current_eef_poses:
            [pose_A, pose_B] — 4×4 EEF poses in world frame at the start of
            the lift, i.e. the confirmed grasp poses.
        target_eef_pose_A:
            4×4 target EEF pose for arm A (the lift destination).
            Arm B's final target is derived automatically via T_rel.
        obstacles:
            Static ObstacleBox list.
        T_rel:
            4×4 rigid transform from arm A's TCP to arm B's TCP at grasp time:
            ``T_rel = inv(pose_A) @ pose_B``.
            If None, computed automatically from current_eef_poses.
        robot_names:
            Labels for logging (default ['robot_C', 'robot_D']).
        start_confs, robot_aabb_getters, robot_base_poses:
            Same as plan_multi.

        Returns
        -------
        [waypoints_A, waypoints_B] — G-code waypoint lists, one per robot.
        """
        self._lazy_load()

        assert len(current_eef_poses) == 2, "plan_cooperative_lift requires exactly 2 robots"
        constraints = constraints or []
        robot_names = robot_names or ['robot_C', 'robot_D']
        start_confs = list(start_confs or [None, None])
        robot_base_poses = list(robot_base_poses or [None, None])

        # Compute T_rel once from the confirmed grasp poses.
        if T_rel is None:
            pose_A0, pose_B0 = current_eef_poses
            T_rel = np.linalg.inv(pose_A0) @ pose_B0
            logger.info('[CuMotion cooperative] T_rel computed from grasp poses: '
                        'translation=%.3f m', np.linalg.norm(T_rel[:3, 3]))

        target_eef_pose_B = target_eef_pose_A @ T_rel

        all_waypoints: List[List[dict]] = [[], []]
        current_confs = list(start_confs)
        current_poses = list(current_eef_poses)
        arrived = [False, False]

        for cycle in range(self._max_replan_cycles):
            if all(arrived):
                break

            robot_obstacles = self._build_inter_robot_obstacles(
                current_poses, robot_aabb_getters)

            # ── Arm A (leader): fixed target ──────────────────────────────
            if not arrived[0]:
                waypoints_A = self._plan_one_arm(
                    robot_idx=0,
                    target=target_eef_pose_A,
                    current_conf=current_confs[0],
                    static_obstacles=obstacles,
                    dynamic_obstacles=robot_obstacles[0],
                    constraints=constraints,
                    robot_base_pose=robot_base_poses[0],
                    robot_name=robot_names[0],
                    cycle=cycle,
                )
                if waypoints_A:
                    all_waypoints[0].extend(waypoints_A['waypoints'])
                    current_confs[0] = waypoints_A['end_conf']
                    current_poses[0] = waypoints_A['end_pose']
                    if np.linalg.norm(current_poses[0][:3, 3]
                                      - target_eef_pose_A[:3, 3]) < self._convergence_tol:
                        arrived[0] = True
                        logger.debug('[CuMotion cooperative] %s arrived at target (cycle %d)',
                                     robot_names[0], cycle)

            # ── Arm B (follower): virtual target tracks arm A ─────────────
            if not arrived[1]:
                # While arm A is still moving, B shadows A's current position.
                # Once A has arrived, B converges to the fixed derived target.
                if arrived[0]:
                    virtual_target_B = target_eef_pose_B
                else:
                    virtual_target_B = current_poses[0] @ T_rel

                waypoints_B = self._plan_one_arm(
                    robot_idx=1,
                    target=virtual_target_B,
                    current_conf=current_confs[1],
                    static_obstacles=obstacles,
                    dynamic_obstacles=robot_obstacles[1],
                    constraints=constraints,
                    robot_base_pose=robot_base_poses[1],
                    robot_name=robot_names[1],
                    cycle=cycle,
                )
                if waypoints_B:
                    all_waypoints[1].extend(waypoints_B['waypoints'])
                    current_confs[1] = waypoints_B['end_conf']
                    current_poses[1] = waypoints_B['end_pose']
                    # Convergence checked against the fixed final target (not virtual).
                    if np.linalg.norm(current_poses[1][:3, 3]
                                      - target_eef_pose_B[:3, 3]) < self._convergence_tol:
                        arrived[1] = True
                        logger.debug('[CuMotion cooperative] %s arrived at derived target (cycle %d)',
                                     robot_names[1], cycle)

        # Final waypoints: exact target poses.
        if arrived[0]:
            all_waypoints[0].append(self._pose_mat4_to_gcode(target_eef_pose_A))
        if arrived[1]:
            all_waypoints[1].append(self._pose_mat4_to_gcode(target_eef_pose_B))

        for i, name in enumerate(robot_names):
            logger.info('[CuMotion cooperative] %s → %d waypoints  arrived=%s',
                        name, len(all_waypoints[i]), arrived[i])
            print(f'  [{name}] cooperative lift: arrived={arrived[i]}  cycles={cycle+1}')

        return all_waypoints

    @staticmethod
    def densify_poses(key_poses: List[np.ndarray], step_m: float = 0.02,
                      step_rad: float = np.radians(2.0)) -> List[np.ndarray]:
        """Piecewise-linear (position) / slerp (rotation) path through `key_poses`,
        resampled so that consecutive poses differ by <= step_m and <= step_rad."""
        from scipy.spatial.transform import Rotation, Slerp
        out = [np.array(key_poses[0], dtype=np.float64)]
        for a, b in zip(key_poses[:-1], key_poses[1:]):
            dist = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
            ang = float((Rotation.from_matrix(a[:3, :3]).inv() * Rotation.from_matrix(b[:3, :3])).magnitude())
            n = max(1, int(np.ceil(max(dist / step_m, ang / step_rad))))
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
            for k in range(1, n + 1):
                t = k / n
                T = np.eye(4)
                T[:3, :3] = slerp([t]).as_matrix()[0]
                T[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
                out.append(T)
        return out

    def plan_rigid_carry(
        self,
        object_poses: List[np.ndarray],
        grasp_in_object: List[np.ndarray],
        obstacles: Optional[List[ObstacleBox]] = None,
        robot_names: Optional[List[str]] = None,
        robot_base_poses: Optional[List[np.ndarray]] = None,
        start_confs: Optional[List[Optional[np.ndarray]]] = None,
        payload_half_extents: Optional[np.ndarray] = None,
        payload_obstacles: Optional[List[ObstacleBox]] = None,
        max_joint_step: float = 0.35,
        arm_clearance: float = 0.05,
        limit_margin: float = 0.05,
    ) -> dict:
        """Two arms carry ONE rigid payload along a given object path.

        The path is planned for the OBJECT (`object_poses`, dense list of 4x4
        world poses).  Each arm's flange pose is then ``O_k @ grasp_in_object[i]``,
        so the relative pose between the two grips is exact by construction and
        waypoint k of both arms belongs to the same instant (lockstep), unlike
        `plan_cooperative_lift` where the follower only shadows the leader.

        Every step is checked: IK solution inside the joint, parallelogram and
        wrist-cable limits and free of self / static-obstacle collision, no joint
        branch jump (> max_joint_step between consecutive steps), arm-arm sphere
        gap >= arm_clearance, and (optionally) the payload box against
        `payload_obstacles`.  It is a verified straight-line carry, NOT a search:
        if a step fails the result says which one and the path must be changed.

        Returns dict: ok, fail (None or reason string), joint_paths [(N,6),(N,6)],
        waypoints [[gcode...],[gcode...]] (flange poses, world), min_arm_gap_m,
        max_joint_step_rad, trel_err_mm, trel_err_deg.
        """
        self._lazy_load()
        robot_names = robot_names or ['robot_C', 'robot_D']
        bases = list(robot_base_poses or [np.eye(4), np.eye(4)])
        seeds = list(start_confs or [None, None])
        obstacles = obstacles or []
        t_rel_obj = np.linalg.inv(grasp_in_object[0]) @ grasp_in_object[1]
        paths = [[], []]
        poses = [[], []]
        min_gap, max_step, trel_pos, trel_rot = np.inf, 0.0, 0.0, 0.0
        res = {'ok': False, 'fail': None, 'joint_paths': None, 'waypoints': None}

        def payload_points(T):
            h = np.asarray(payload_half_extents, dtype=float)
            g = np.linspace(-1.0, 1.0, 5)
            pts = np.array([[x * h[0], y * h[1], z * h[2]] for x in g for y in g for z in (-1.0, 1.0)])
            return (T[:3, :3] @ pts.T).T + T[:3, 3]

        for k, O in enumerate(object_poses):
            confs = []
            for i in (0, 1):
                q = self.ik(O @ grasp_in_object[i], robot_name=robot_names[i], seed=seeds[i],
                            robot_base_pose=bases[i], limit_margin=limit_margin, obstacles=obstacles)
                if q is None:
                    res['fail'] = (f'step {k}/{len(object_poses)}: {robot_names[i]} has no in-limit, '
                                   f'collision-free IK solution')
                    return res
                if paths[i]:
                    step = float(np.max(np.abs(q - paths[i][-1])))
                    max_step = max(max_step, step)
                    if step > max_joint_step:
                        res['fail'] = (f'step {k}: {robot_names[i]} joint branch jump of {step:.2f} rad '
                                       f'(> {max_joint_step})')
                        return res
                confs.append(q)
            # arm-arm clearance with cuMotion's own collision spheres, in world frame
            sph = []
            for i in (0, 1):
                s_i = self.robot_spheres(confs[i], robot_name=robot_names[i])
                sph.append((np.array([bases[i][:3, :3] @ c + bases[i][:3, 3] for c, _ in s_i]),
                            np.array([r for _, r in s_i])))
            d = np.linalg.norm(sph[0][0][:, None, :] - sph[1][0][None, :, :], axis=2) \
                - sph[0][1][:, None] - sph[1][1][None, :]
            gap = float(d.min())
            min_gap = min(min_gap, gap)
            if gap < arm_clearance:
                res['fail'] = f'step {k}: arm-arm sphere gap {gap * 1000:.0f} mm < {arm_clearance * 1000:.0f} mm'
                return res
            if payload_half_extents is not None and payload_obstacles:
                pts = payload_points(O)
                for ob in payload_obstacles:
                    loc = (np.asarray(ob.rotation).T @ (pts - np.asarray(ob.center)).T).T
                    if np.any(np.all(np.abs(loc) < np.asarray(ob.half_extents), axis=1)):
                        res['fail'] = f'step {k}: payload collides with an obstacle'
                        return res
            for i in (0, 1):
                paths[i].append(confs[i])
                self._select(robot_names[i])
                poses[i].append(self._conf_to_eef_mat4(confs[i], bases[i]))
            rel = np.linalg.inv(poses[0][-1]) @ poses[1][-1]
            trel_pos = max(trel_pos, float(np.linalg.norm(rel[:3, 3] - t_rel_obj[:3, 3])))
            dr = rel[:3, :3] @ t_rel_obj[:3, :3].T
            trel_rot = max(trel_rot, float(np.degrees(np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1)))))
            seeds = [confs[0], confs[1]]

        res.update(ok=True, joint_paths=[np.array(p) for p in paths],
                   waypoints=[self._trajectory_to_gcode(p) for p in poses],
                   min_arm_gap_m=min_gap, max_joint_step_rad=max_step,
                   trel_err_mm=trel_pos * 1000.0, trel_err_deg=trel_rot)
        return res

    def _plan_one_arm(
        self,
        robot_idx: int,
        target: np.ndarray,
        current_conf: Optional[np.ndarray],
        static_obstacles: List[ObstacleBox],
        dynamic_obstacles: List[ObstacleBox],
        constraints: List[SceneConstraint],
        robot_base_pose: Optional[np.ndarray],
        robot_name: str,
        cycle: int,
    ) -> Optional[dict]:
        """Plan one replan chunk for one arm.  Returns dict with waypoints/end_conf/end_pose,
        or None if cuMotion finds no path."""
        import cumotion
        self._select(robot_name)

        T_inv = np.linalg.inv(robot_base_pose) if robot_base_pose is not None else None

        if T_inv is not None:
            static_local  = [obs.in_frame(T_inv) for obs in static_obstacles]
            dynamic_local  = [obs.in_frame(T_inv) for obs in dynamic_obstacles]
            target_local   = T_inv @ target
        else:
            static_local  = static_obstacles
            dynamic_local  = dynamic_obstacles
            target_local   = target

        world, world_view = self._build_world(static_local + dynamic_local, constraints)
        goal_pose = self._mat4_to_pose3(target_local)

        if current_conf is None:
            current_conf = self._robot_description.default_cspace_configuration()

        planner_config = cumotion.create_motion_planner_config_from_file(
            self._planning_config_path, self._robot_description,
            self._tool_frame, world_view)
        planner = cumotion.create_motion_planner(planner_config)
        results = planner.plan_to_pose_target(current_conf, goal_pose, True)

        if not results.path_found:
            logger.warning('[CuMotion cooperative] Cycle %d: no path for %s', cycle, robot_name)
            return None

        joint_traj = np.array(results.interpolated_path)
        chunk = joint_traj[:self._replan_interval_steps]
        eef_chunk = self._joint_traj_to_eef(chunk, robot_base_pose)
        waypoints = self._trajectory_to_gcode(eef_chunk)
        end_conf = chunk[-1]
        end_pose = self._conf_to_eef_mat4(end_conf, robot_base_pose)
        return {'waypoints': waypoints, 'end_conf': end_conf, 'end_pose': end_pose}

    # ------------------------------------------------------------------
    # Internal: lazy loading
    # ------------------------------------------------------------------

    def _lazy_load(self):
        if self._robot_description is not None:
            return
        self._load_model(self._urdf_path)

    def _planning_urdf(self, urdf_path: str) -> str:
        """URDF used for PLANNING: the given file with multi-turn wrist joint
        limits clamped to +-wrist_limit_rad (written next to it, *_plan.urdf)."""
        if self._wrist_limit is None:
            return urdf_path
        import re
        txt = open(urdf_path, encoding='utf-8').read()
        lim = self._wrist_limit

        def clamp(m):
            lo, hi = float(m.group(1)), float(m.group(2))
            if hi - lo > 2.5 * np.pi:      # a multi-turn joint
                return f'lower="{-lim}" upper="{lim}"'
            return m.group(0)
        txt2 = re.sub(r'lower="([-0-9.eE]+)" upper="([-0-9.eE]+)"', clamp, txt)
        out = urdf_path[:-5] + '_plan.urdf'
        if not os.path.exists(out) or open(out, encoding='utf-8').read() != txt2:
            open(out, 'w', encoding='utf-8').write(txt2)
        return out

    def _load_model(self, urdf_path: str):
        """Load (once) and make current the model built from `urdf_path` + the shared XRDF."""
        if urdf_path not in self._models:
            _ensure_cumotion()
            import cumotion
            plan_urdf = self._planning_urdf(urdf_path)
            # A sidecar <urdf>.xrdf (same name as the URDF) overrides the shared
            # XRDF: robot_C's flange cross bars live only in its own spheres.
            xrdf = urdf_path[:-5] + '.xrdf'
            if not os.path.exists(xrdf):
                xrdf = self._xrdf_path
            logger.info('[CuMotion] Loading robot from %s + %s', xrdf, plan_urdf)
            desc = cumotion.load_robot_from_file(xrdf, plan_urdf)
            self._models[urdf_path] = (desc, desc.kinematics(), None)
            logger.info('[CuMotion] Robot loaded. DOFs: %d, tool frame: %s',
                        desc.num_cspace_coords(), self._tool_frame)
        self._robot_description, self._kinematics, self._inspector = self._models[urdf_path]
        self._current_urdf = urdf_path

    def _select(self, robot_name: Optional[str]):
        """Make the model of `robot_name` current (its own URDF if overridden)."""
        self._load_model(self._urdf_overrides.get(robot_name, self._urdf_path))

    # ------------------------------------------------------------------
    # Internal: world building
    # ------------------------------------------------------------------

    def _build_world(
        self,
        obstacles: List[ObstacleBox],
        constraints: List[SceneConstraint],
    ):
        """Create a cumotion.World populated with obstacles.

        Returns
        -------
        (world, world_view_handle) — both needed: world keeps obstacles alive,
        world_view is what the planner config builder accepts.

        Worlds are cached by obstacle geometry: during a dry rehearsal the
        same static scene is rebuilt for every plan / IK / start diagnosis
        (4-5 times per segment, ~60 boxes each) — it is the same world.
        """
        import cumotion
        key = None
        if not constraints:
            key = tuple((tuple(np.round(o.center, 4).tolist()), tuple(np.round(o.half_extents, 4).tolist()),
                         tuple(np.round(o.rotation, 3).ravel().tolist())) for o in obstacles)
            cache = getattr(self, '_world_cache', None)
            if cache is None:
                cache = self._world_cache = {}
            if key in cache:
                return cache[key]
            if len(cache) > 64:
                cache.clear()

        world = cumotion.create_world()

        for obs in obstacles:
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            obstacle.set_attribute(
                cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                obs.spans.astype(np.float64),
            )
            world.add_obstacle(obstacle, self._mat4_to_pose3(obs.pose_4x4))

        for constraint in constraints:
            from modular_planner.scene import ConstraintType
            if constraint.constraint_type == ConstraintType.NO_GO_ZONE:
                extents = constraint.half_extents \
                    if constraint.half_extents is not None \
                    else np.array([0.05, 0.05, 0.05], dtype=np.float32)
                obs = ObstacleBox(
                    center=constraint.pose[:3, 3],
                    half_extents=extents,
                    rotation=constraint.pose[:3, :3],
                )
                obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
                obstacle.set_attribute(
                    cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                    obs.spans.astype(np.float64),
                )
                world.add_obstacle(obstacle, self._mat4_to_pose3(obs.pose_4x4))
                logger.debug('[CuMotion] NO_GO_ZONE added as cuboid obstacle')
            else:
                logger.info(
                    '[CuMotion] Constraint type %s not yet supported; skipping.',
                    constraint.constraint_type.name)

        world_view = world.add_world_view()
        if key is not None:
            self._world_cache[key] = world, world_view
        return world, world_view

    def _build_inter_robot_obstacles(
        self,
        current_poses: List[np.ndarray],
        aabb_getters,
    ) -> List[List[ObstacleBox]]:
        """For each robot i, return ObstacleBoxes for all other robots j≠i.

        A getter may return one ``(aabb_min, aabb_max)`` or a list of them
        (one per link).  Per-link boxes are what you want: PyBullet's
        ``getAABB(body)`` covers the BASE link only, and a single box around a
        whole 3 m arm swallows the workspace.
        """
        n = len(current_poses)
        robot_boxes: List[List[ObstacleBox]] = []

        for j in range(n):
            boxes: List[ObstacleBox] = []
            if aabb_getters is not None and aabb_getters[j] is not None:
                try:
                    aabbs = aabb_getters[j]()
                    if len(aabbs) == 2 and np.ndim(aabbs[0]) == 1:
                        aabbs = [aabbs]
                    for aabb_min, aabb_max in aabbs:
                        boxes.append(ObstacleBox.from_pybullet_aabb(
                            np.array(aabb_min), np.array(aabb_max)))
                except Exception as e:
                    logger.debug('[CuMotion] AABB getter for robot %d failed: %s', j, e)
            if not boxes:
                boxes = [self._pose_to_point_box(current_poses[j])]
            robot_boxes.append(boxes)

        # Result[i] = boxes for all robots except i.
        result = []
        for i in range(n):
            result.append([b for j in range(n) if j != i for b in robot_boxes[j]])
        return result

    @staticmethod
    def _pose_to_point_box(pose_4x4: np.ndarray, size: float = 0.3) -> ObstacleBox:
        """Approximate a robot as a small box at its EEF position."""
        half = np.array([size / 2] * 3, dtype=np.float32)
        return ObstacleBox(
            center=pose_4x4[:3, 3].astype(np.float32),
            half_extents=half,
            rotation=np.eye(3, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Internal: kinematics helpers
    # ------------------------------------------------------------------

    def _joint_traj_to_eef(
        self,
        joint_traj: np.ndarray,
        robot_base_pose: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """FK every config in joint_traj → list of 4×4 EEF poses in world frame."""
        poses = []
        base_frame = self._kinematics.base_frame_name()
        for i, conf in enumerate(joint_traj):
            if i % self._waypoint_step != 0 and i != len(joint_traj) - 1:
                continue
            pose3 = self._kinematics.pose(conf, self._tool_frame, base_frame)
            T_local = self._pose3_to_mat4(pose3)
            poses.append(robot_base_pose @ T_local if robot_base_pose is not None else T_local)
        return poses

    def _conf_to_eef_mat4(
        self,
        conf: np.ndarray,
        robot_base_pose: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """FK a single joint config → 4×4 EEF pose matrix in world frame."""
        base_frame = self._kinematics.base_frame_name()
        pose3 = self._kinematics.pose(conf, self._tool_frame, base_frame)
        T_local = self._pose3_to_mat4(pose3)
        return robot_base_pose @ T_local if robot_base_pose is not None else T_local

    # ------------------------------------------------------------------
    # Internal: pose conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _mat4_to_pose3(mat4: np.ndarray):
        """Convert a 4×4 numpy matrix to a cumotion Pose3."""
        from cumotion import Pose3
        return Pose3(mat4.astype(np.float64))

    @staticmethod
    def _pose3_to_mat4(pose3) -> np.ndarray:
        """Convert a cumotion Pose3 back to a 4×4 numpy matrix."""
        return np.array(pose3.matrix(), dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal: trajectory → G-code
    # ------------------------------------------------------------------

    def _trajectory_to_gcode(self, trajectory: List[np.ndarray]) -> List[dict]:
        """Convert a list of 4×4 EEF poses to GCodeProcessor-compatible dicts."""
        waypoints = []
        for pose in trajectory:
            pos = pose[:3, 3]
            euler_xyz = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=False)
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

    def _pose_mat4_to_gcode(self, pose: np.ndarray) -> dict:
        """Convert a single 4×4 pose to a G-code waypoint dict."""
        return self._trajectory_to_gcode([pose])[0]
