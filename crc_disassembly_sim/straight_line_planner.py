"""`straight_line_checked`: a motion planner for the cell simulation (techstack #149) that moves each share and
transit in a straight line and checks it for collisions in pybullet.

Every line is sampled every `step` metres; at each sample the robot's arm is placed by IK (its URDF, joint states
set directly, no physics stepping), the tools it carries move with it, and the arm and tools are tested against
the scene's other bodies with their collision shapes. A pose IK cannot reach is `out_of_reach`, any contact is
`collision`: both are a `MotionFailure`, so the operation is infeasible and nothing moves (ADR-0008 of the
techstack, never a silent fallback). The scene is put back as it was after every call.

Poses are `x, y, z` (orientation from the `orientation` option), `x, y, z, rx, ry, rz` (Euler, radians) or
`x, y, z, qx, qy, qz, qw`, in metres. An operation whose parameters name a `reference_part` gives its poses
relative to that part; the `reference_poses` option places the part in the cell frame (`x, y, z, rx, ry, rz`).
A share that starts at a supply slot keeps its first pose as a cell pose.

Options (`configure_planner`): `feed` and `transit_feed` (m/s), `step` (m), `clearance` (m, the gap that counts
as contact), `orientation`, `reference_poses`, and for robot-vs-robot conflicts `conflict_margin` (m) and
`conflict_samples` (configurations per segment).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pybullet as p

from cell_simulation import MotionFailure, MotionPlanner, ReasonCode, Segment, ShareMotion
from sim_server_protocol import InvalidRequest

NAME = "straight_line_checked"
REFERENCE_PART = "reference_part"  # operation parameter: the part its poses are relative to
SUPPLY_SLOT = "supply_slot"  # share skill parameter: the share starts at this supply slot (a cell pose)
POSITION_TOLERANCE = 1e-3  # m: IK must place the TCP this close, else the pose is out of reach
ANGLE_TOLERANCE = 1e-2  # rad
IK_ROUNDS = 5

DEFAULTS = {"feed": 0.1, "transit_feed": 0.5, "step": 0.01, "clearance": 0.0, "orientation": [math.pi, 0.0, 0.0],
            "reference_poses": {}, "conflict_margin": 0.05, "conflict_samples": 8}


@dataclass
class Arm:
    """One robot of the cell in the pybullet scene: its body, the arm link IK places, the bodies it carries now
    (tools, a camera) as `(body, (position, quaternion))` with their base pose in that link's frame, and the TCP
    the share poses command, as `(position, quaternion)` in that link's frame (None: the link itself), e.g. the
    tip of the tool it holds."""

    body: int
    tcp_link: int
    tools: Callable[[], Iterable[tuple[int, tuple]]] = lambda: ()
    tcp: Callable[[], tuple | None] = lambda: None


@dataclass
class CheckedMotion(ShareMotion):
    """A planned straight line with the arm configurations it visits, for robot-vs-robot conflicts."""

    configurations: list = field(default_factory=list)
    carried: list = field(default_factory=list)  # (tool body, position, orientation) in the arm link's frame


def register(service, arms: dict[str, Arm], client: int = 0,
             ignore: Callable[[], Iterable[int]] = lambda: ()) -> None:
    """Register `straight_line_checked` on a `CellSimulationService` for the scene of pybullet `client`; `arms`
    maps the cell's robot ids to their arms, `ignore` gives bodies that are never obstacles (e.g. the product's
    parts, which an operation touches on purpose)."""
    service.register_planner(NAME, lambda options: StraightLineChecked(arms, client, ignore, options))


class StraightLineChecked(MotionPlanner):
    name = NAME

    def __init__(self, arms: dict[str, Arm], client: int, ignore: Callable[[], Iterable[int]], options: dict):
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise InvalidRequest(f"unknown {NAME} options {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        self.options = {**DEFAULTS, **options}
        _check_options(self.options)
        self.arms, self.client, self.ignore = arms, client, ignore

    # -- the MotionPlanner interface -------------------------------------------------------------------------

    def path_start(self, operation: dict, share: dict) -> tuple:
        return self._cell_path(operation, share)[0]

    def plan_share(self, robot: dict, operation: dict, share: dict):
        return self._line(robot["id"], self._cell_path(operation, share), self.options["feed"])

    def plan_transit(self, robot: dict, start_pose, goal_pose):
        """From `start_pose`, or with None from where the arm is in the scene now (its home)."""
        arm = self.arms.get(robot["id"])
        if arm is None:
            return self._no_arm(robot["id"])
        start = self._tcp_pose(arm) if start_pose is None else tuple(start_pose)
        return self._line(robot["id"], [start, tuple(goal_pose)], self.options["transit_feed"])

    def conflicts(self, first: Segment, second: Segment) -> bool:
        """Whether some configuration of one segment comes within `conflict_margin` of some configuration of the
        other (timing-independent). A segment without configurations (a dwell) does not move, so it is not tested."""
        a, b = first.motion, second.motion
        if not isinstance(a, CheckedMotion) or not isinstance(b, CheckedMotion):
            return False
        arm_a, arm_b = self.arms.get(first.robot), self.arms.get(second.robot)
        if arm_a is None or arm_b is None:
            return False
        with self._restored():
            for qa in _spread(a.configurations, self.options["conflict_samples"]):
                self._place(arm_a, qa, a.carried)
                for qb in _spread(b.configurations, self.options["conflict_samples"]):
                    self._place(arm_b, qb, b.carried)
                    if self._contact(_bodies(arm_a, a.carried), _bodies(arm_b, b.carried), self.options["conflict_margin"]):
                        return True
        return False

    # -- straight lines --------------------------------------------------------------------------------------

    def _line(self, robot: str, poses: list[tuple], feed: float):
        arm = self.arms.get(robot)
        if arm is None:
            return self._no_arm(robot)
        with self._restored():
            carried = self._carried(arm)
            moving = _bodies(arm, carried)
            obstacles = self._obstacles()
            configurations, length = [], 0.0
            for i, pose in enumerate(poses):
                previous = poses[i - 1] if i else pose
                distance = math.dist(previous[:3], pose[:3])
                length += distance
                steps = max(1, math.ceil(distance / self.options["step"])) if i else 1
                for k in range(1 if i else 0, steps + 1):
                    position = [a + (b - a) * k / steps for a, b in zip(previous[:3], pose[:3])]
                    orientation = p.getQuaternionSlerp(previous[3:], pose[3:], k / steps)
                    if not self._reach(arm, position, orientation):
                        return MotionFailure(ReasonCode.OUT_OF_REACH,
                                             f"{robot} cannot reach {_mm(position)} (no IK solution)")
                    self._carry(arm, carried)
                    hit = self._contact(moving, obstacles, self.options["clearance"])
                    if hit:
                        return MotionFailure(ReasonCode.COLLISION, f"{robot} at {_mm(position)}: {hit}")
                    configurations.append(self._joints(arm))
        commands = [{"G": 1, "X": q[0], "Y": q[1], "Z": q[2], **_euler(q[3:]), "F": feed * 60000.0} for q in poses]
        return CheckedMotion(length / feed, commands, tuple(poses[-1]), configurations, carried)

    def _cell_path(self, operation: dict, share: dict) -> list[tuple]:
        """The share's poses in the cell frame, as `x, y, z, qx, qy, qz, qw`."""
        frame = None
        part = operation["parameters"].get(REFERENCE_PART)
        if part is not None:
            placement = self.options["reference_poses"].get(part)
            if placement is None:
                raise InvalidRequest(f"{operation['id']} is relative to {part!r}, which {NAME} has no reference pose "
                                     f"for; send it in the reference_poses option of configure_planner")
            frame = _pose(placement, self.options["orientation"])
        poses = []
        for i, raw in enumerate(share["path"]):
            pose = _pose(raw, self.options["orientation"])
            # the plan functions name the supply slot a share starts at in its skill parameters (screwing in,
            # gripping at the supply); that first pose is a cell-file pose, not relative to the part
            if frame is not None and not (i == 0 and SUPPLY_SLOT in share.get("skill_parameters", {})):
                position, orientation = p.multiplyTransforms(frame[:3], frame[3:], pose[:3], pose[3:])
                pose = (*position, *orientation)
            poses.append(pose)
        return poses

    # -- pybullet --------------------------------------------------------------------------------------------

    def _reach(self, arm: Arm, position, orientation) -> bool:
        tcp = arm.tcp()
        target = (position, orientation) if tcp is None else p.multiplyTransforms(position, orientation,
                                                                                  *p.invertTransform(*tcp))
        joints = self._movable(arm)
        limits = [p.getJointInfo(arm.body, j, physicsClientId=self.client)[8:10] for j in joints]
        lower, upper = [lo for lo, _ in limits], [hi for _, hi in limits]
        ranges = [hi - lo for lo, hi in limits]
        for _ in range(IK_ROUNDS):  # null-space IK: within the joint limits, near the current configuration
            solution = p.calculateInverseKinematics(arm.body, arm.tcp_link, *target,
                                                    lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
                                                    restPoses=list(self._joints(arm)),
                                                    maxNumIterations=200, residualThreshold=1e-6,
                                                    physicsClientId=self.client)
            self._set_joints(arm, solution)
            at = self._tcp_pose(arm)
            if math.dist(at[:3], position) < POSITION_TOLERANCE and _angle(at[3:], orientation) < ANGLE_TOLERANCE:
                return self._within_limits(arm, joints)
        return False

    def _within_limits(self, arm: Arm, joints: list[int]) -> bool:
        for joint in joints:
            info = p.getJointInfo(arm.body, joint, physicsClientId=self.client)
            lower, upper = info[8], info[9]
            value = p.getJointState(arm.body, joint, physicsClientId=self.client)[0]
            if lower < upper and not lower - 1e-6 <= value <= upper + 1e-6:
                return False
        return True

    def _contact(self, moving: list[int], others: list[int], margin: float) -> str | None:
        """The first pair of `moving` and `others` closer than `margin`, described, or None."""
        for a in moving:
            for b in others:
                points = p.getClosestPoints(a, b, margin, physicsClientId=self.client)
                if points:
                    point = min(points, key=lambda c: c[8])
                    return f"{self._name(a, point[3])} touches {self._name(b, point[4])} ({point[8] * 1000:.0f} mm)"
        return None

    def _obstacles(self) -> list[int]:
        skip = {b for arm in self.arms.values() for b in (arm.body, *(tool for tool, _ in arm.tools()))}
        skip |= set(self.ignore())
        return [b for b in (p.getBodyUniqueId(i, physicsClientId=self.client)
                            for i in range(p.getNumBodies(physicsClientId=self.client))) if b not in skip]

    def _carried(self, arm: Arm) -> list:
        return [(tool, tuple(position), tuple(orientation)) for tool, (position, orientation) in arm.tools()]

    def _carry(self, arm: Arm, carried: list) -> None:
        tcp = self._link_pose(arm)
        for tool, position, orientation in carried:
            p.resetBasePositionAndOrientation(tool, *p.multiplyTransforms(tcp[:3], tcp[3:], position, orientation),
                                              physicsClientId=self.client)

    def _place(self, arm: Arm, configuration, carried: list) -> None:
        self._set_joints(arm, configuration)
        self._carry(arm, carried)

    def _set_joints(self, arm: Arm, configuration) -> None:
        for joint, value in zip(self._movable(arm), configuration):
            p.resetJointState(arm.body, joint, value, physicsClientId=self.client)

    def _movable(self, arm: Arm) -> list[int]:
        return [j for j in range(p.getNumJoints(arm.body, physicsClientId=self.client))
                if p.getJointInfo(arm.body, j, physicsClientId=self.client)[2] != p.JOINT_FIXED]

    def _joints(self, arm: Arm) -> tuple:
        return tuple(p.getJointState(arm.body, j, physicsClientId=self.client)[0] for j in self._movable(arm))

    def _link_pose(self, arm: Arm) -> tuple:
        state = p.getLinkState(arm.body, arm.tcp_link, computeForwardKinematics=True, physicsClientId=self.client)
        return (*state[4], *state[5])

    def _tcp_pose(self, arm: Arm) -> tuple:
        link, tcp = self._link_pose(arm), arm.tcp()
        if tcp is None:
            return link
        position, orientation = p.multiplyTransforms(link[:3], link[3:], *tcp)
        return (*position, *orientation)

    def _name(self, body: int, link: int) -> str:
        name = p.getBodyInfo(body, physicsClientId=self.client)[1].decode()
        if link >= 0:
            name += "/" + p.getJointInfo(body, link, physicsClientId=self.client)[12].decode()
        return name

    @contextmanager
    def _restored(self):
        """Puts the arms' joints and every carried body's pose back as they were when the block ends."""
        joints = {name: self._joints(arm) for name, arm in self.arms.items()}
        tools = {tool: p.getBasePositionAndOrientation(tool, physicsClientId=self.client)
                 for arm in self.arms.values() for tool, _ in arm.tools()}
        try:
            yield
        finally:
            for name, arm in self.arms.items():
                self._set_joints(arm, joints[name])
            for tool, (position, orientation) in tools.items():
                p.resetBasePositionAndOrientation(tool, position, orientation, physicsClientId=self.client)

    def _no_arm(self, robot: str) -> MotionFailure:
        return MotionFailure(ReasonCode.UNSUPPORTED_PROCESS, f"{robot} has no arm in the simulation scene")


def _pose(raw, orientation) -> tuple:
    values = [float(v) for v in raw]
    if len(values) == 3:
        return (*values, *p.getQuaternionFromEuler(orientation))
    if len(values) == 6:
        return (*values[:3], *p.getQuaternionFromEuler(values[3:]))
    if len(values) == 7:
        return tuple(values)
    raise InvalidRequest(f"a pose has 3, 6 or 7 values, got {len(values)}")


def _euler(quaternion) -> dict:
    a, b, c = p.getEulerFromQuaternion(quaternion)
    return {"A": a, "B": b, "C": c}


def _angle(q1, q2) -> float:
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, dot))


def _bodies(arm: Arm, carried: list) -> list[int]:
    return [arm.body, *(tool for tool, _, _ in carried)]


def _spread(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)] if count > 1 else items[:1]


def _check_options(options: dict) -> None:
    """Refuse bad options at `configure_planner`, not later in a `simulate`."""
    def number(value, minimum, strict):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)             and (value > minimum if strict else value >= minimum)

    def numbers(value, length):
        return isinstance(value, (list, tuple)) and len(value) == length and all(number(v, -math.inf, True) for v in value)

    problems = [f"{key} must be a positive number" for key in ("feed", "transit_feed", "step")
                if not number(options[key], 0.0, True)]
    problems += [f"{key} must be a number of at least 0" for key in ("clearance", "conflict_margin")
                 if not number(options[key], 0.0, False)]
    samples = options["conflict_samples"]
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        problems.append("conflict_samples must be a positive integer")
    if not numbers(options["orientation"], 3):
        problems.append("orientation must be rx, ry, rz")
    poses = options["reference_poses"]
    if not isinstance(poses, dict) or not all(isinstance(k, str) and numbers(v, 6) for k, v in poses.items()):
        problems.append("reference_poses maps part names to x, y, z, rx, ry, rz")
    if problems:
        raise InvalidRequest(f"{NAME} options: " + "; ".join(problems))


def _mm(position) -> str:
    return "(" + ", ".join(f"{v * 1000:.0f}" for v in position) + ") mm"
