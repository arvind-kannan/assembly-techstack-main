"""Offline (planner-only, no PyBullet) check of CuMotionWaypointPlanner.plan_cooperative_lift
(leader/follower) versus plan_rigid_carry (one object path, both arms derived exactly).

Two Comau NJ290 arms grasp opposite edges of a payload above the clamping desk, lift it 25 cm
and shift it 50 cm.  Reports whether both arms arrive and how far the relative pose T_rel
drifts between replan cycles (leader/follower scheme: the follower only shadows the leader).
Run:  python training/tests/test_cooperative_lift_offline.py     (needs .venv with cumotion)
"""
import math
import os
import sys

import numpy as np

SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, SIM)
from modular_planner import CuMotionWaypointPlanner  # noqa: E402
from modular_planner.scene import ObstacleBox  # noqa: E402

RD = os.path.join(SIM, 'robot_descriptions') + os.sep
LIMITS = [{'name': 'parallelogram', 'i': 1, 'j': 2, 'k': 1.0, 'lo': math.radians(-220), 'hi': 0.0},
          {'name': 'wrist cables', 'i': 3, 'j': 5, 'k': -1.0, 'lo': -math.radians(250), 'hi': math.radians(250)}]
OBJ = np.array([0.21, -2.73, 1.45])      # payload centre: 45 cm above the desk top
HALF = 0.55                              # flange 10 cm outside the payload edge (0.45): ghost gripper reaches the edge
LIFT, SHIFT = np.array([0, 0, 0.25]), np.array([-0.45, 0, 0])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def frame(pos, zdir, roll):
    z = np.asarray(zdir, float) / np.linalg.norm(zdir)
    x = np.cross([0, 0, 1.0], z)
    x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-6 else x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z]) @ rz(roll)
    T[:3, 3] = pos
    return T


def main():
    pl = CuMotionWaypointPlanner(
        urdf_overrides={'robot_C': RD + 'comau_nj290_robotNC.urdf', 'robot_D': RD + 'comau_nj290_robotNC_sensor.urdf'},
        coupled_limit=LIMITS)
    base_c = np.eye(4)
    base_d = np.eye(4)
    base_d[:3, :3], base_d[:3, 3] = rz(math.pi / 2), [-1.99, -3.75, 0]
    table = ObstacleBox(center=[0.21, -2.73, 0.5], half_extents=[0.6, 0.6, 0.5], rotation=np.eye(3))

    best = None
    cands = []
    for el in range(-90, 91, 15):
        for az in range(0, 360, 15):
            if abs(el) == 90 and az > 0:
                continue
            e, a = math.radians(el), math.radians(az)
            z = [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
            if -z[1] >= 0.5:                      # flange z points from arm A's side toward the payload
                cands.append(z)
    cands.sort(key=lambda z: (-z[1], -z[2]))     # most horizontal, then most downward first
    for z in cands:
        for roll in np.radians(np.arange(0, 360, 45)):
            ga = frame(OBJ + [0, HALF, 0], z, roll)
            gb = frame(OBJ - [0, HALF, 0], [-z[0], -z[1], z[2]], roll)
            ta, tb = ga.copy(), gb.copy()
            ta[:3, 3] += LIFT + SHIFT
            tb[:3, 3] += LIFT + SHIFT
            qs = [pl.ik(T, robot_name=n, robot_base_pose=b, limit_margin=0.05, obstacles=[table])
                  for T, n, b in [(ga, 'robot_C', base_c), (ta, 'robot_C', base_c),
                                  (gb, 'robot_D', base_d), (tb, 'robot_D', base_d)]]
            if all(q is not None for q in qs):
                best = (z, roll, ga, gb, ta, qs)
                break
        if best:
            break
    if not best:
        print('NO collision-free, in-limit grasp/target pair found for both arms')
        return 1
    z, roll, ga, gb, ta, qs = best
    print(f'grasp orientation: flange z-axis {np.round(z, 2)}, roll {math.degrees(roll):.0f} deg; '
          f'lift {LIFT[2]} m, shift {SHIFT[0]} m')

    rec = []
    orig = pl._plan_one_arm

    def wrapped(**kw):
        r = orig(**kw)
        rec.append((kw['robot_idx'], kw['cycle'], None if r is None else r['end_pose']))
        return r

    pl._plan_one_arm = wrapped
    t_rel = np.linalg.inv(ga) @ gb
    wps = pl.plan_cooperative_lift([ga, gb], ta, [table], robot_names=['robot_C', 'robot_D'],
                                   start_confs=[qs[0], qs[2]], robot_base_poses=[base_c, base_d])
    print('waypoints per arm:', [len(w) for w in wps], '| planner calls:', len(rec),
          '| failed:', sum(r[2] is None for r in rec))
    last = {}
    for idx, cyc, pose in rec:
        if pose is None:
            continue
        last[idx] = pose
        if len(last) == 2:
            tr = np.linalg.inv(last[0]) @ last[1]
            dr = tr[:3, :3] @ t_rel[:3, :3].T
            ang = math.degrees(math.acos(max(-1, min(1, (np.trace(dr) - 1) / 2))))
            print(f'  after arm {idx} of cycle {cyc}: T_rel error {np.linalg.norm(tr[:3, 3] - t_rel[:3, 3]) * 1000:6.1f} mm, {ang:5.1f} deg')

    # ---- rigid carry: one object path (lift, shift, set down), both arms derived from it
    o0 = np.eye(4)
    o0[:3, 3] = OBJ
    keys = [o0]
    for delta in (LIFT, SHIFT, -LIFT):
        nxt = keys[-1].copy()
        nxt[:3, 3] += delta
        keys.append(nxt)
    obj_path = CuMotionWaypointPlanner.densify_poses(keys, step_m=0.02)
    g_in_obj = [np.linalg.inv(o0) @ ga, np.linalg.inv(o0) @ gb]
    res = pl.plan_rigid_carry(obj_path, g_in_obj, obstacles=[table], robot_names=['robot_C', 'robot_D'],
                              robot_base_poses=[base_c, base_d], start_confs=[qs[0], qs[2]],
                              payload_half_extents=[0.20, 0.45, 0.01], payload_obstacles=[table])
    print('rigid carry:', 'OK' if res['ok'] else 'FAILED - ' + str(res['fail']))
    if res['ok']:
        print(f"  steps {len(res['waypoints'][0])} (both arms in lockstep), T_rel error max "
              f"{res['trel_err_mm']:.2f} mm / {res['trel_err_deg']:.2f} deg, min arm-arm gap "
              f"{res['min_arm_gap_m'] * 1000:.0f} mm, max joint step {res['max_joint_step_rad']:.3f} rad")
    return 0


if __name__ == '__main__':
    sys.exit(main())
