"""Offline checks of the task (no GPU, no Isaac Sim): the scorer on the start, the goal and named layouts, and
the scripted demo's reach and length.

    pytest open-locker-combination-padlock    (in an environment with yam_arena installed)
"""

import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.pop("yam_parts", None)
import yam_parts
sys.modules.pop("yam_joints", None)
import yam_joints
sys.modules.pop("yam_mesh", None)
import yam_mesh
from dataclasses import replace

TASK = load(Path(__file__).with_name("open_locker_combination_padlock.py"))
T = TASK.module
OBJ = {o.name: o for o in T.OBJECTS}
RUN_LIMIT_STEPS = 4000


def _rot(yaw_deg, tilt_deg, axis):
    y, t = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    c, s = math.cos(t), math.sin(t)
    rt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rz @ rt


def _half(o, r):
    if hasattr(o, "size"):
        return np.abs(r) @ (np.asarray(o.size) / 2)
    if hasattr(o, "height"):
        a = r[:, 2]
        return o.height / 2 * np.abs(a) + o.radius * np.sqrt(np.clip(1 - a * a, 0, 1))
    return np.full(3, o.radius)


def place(name, pos, yaw_deg=0.0, tilt_deg=0.0, tilt_axis="x", vel=(0.0, 0.0, 0.0), quat=None, joints=None):
    """The ObjectState of `name` with its centre at `pos`, turned and tipped; its bounding box follows the turn."""
    r = (_rot(yaw_deg, tilt_deg, tilt_axis) if quat is None else
         ik.quat_xyzw_to_matrix(np.asarray(quat, float) / np.linalg.norm(quat)))
    if isinstance(OBJ[name], yam_joints.Mechanism):
        q = {link.name: float((joints or {}).get(link.name, getattr(link.joint, "initial", 0.0))) *
             (math.pi / 180 if isinstance(link.joint, yam_joints.Hinge) else 1.0)
             for link in OBJ[name].links}
        state = OBJ[name].offline_state(pos, math.radians(yaw_deg), q, R=r)
        return replace(state, lin_vel=np.asarray(vel, float))
    if isinstance(OBJ[name], yam_parts.Part):
        origin = np.asarray(pos, float)
        low, high = OBJ[name].aabb(origin, r)
        return yam_parts.PartState((low + high) / 2, np.asarray(ik.matrix_to_quat_xyzw(r), float),
                                   np.asarray(vel, float), np.zeros(3), low, high, origin)
    pos, half = np.asarray(pos, float), _half(OBJ[name], r)
    q = ik.matrix_to_quat_xyzw(r)
    return ObjectState(pos, np.asarray(q, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def layout(poses):
    """Every object's state: the loose ones as given, the fixed ones where the task file puts them."""
    out = {o.name: place(o.name, o.pos, math.degrees(o.yaw)) for o in T.OBJECTS if o.mass is None}
    out.update({name: place(name, **p) for name, p in poses.items()})
    return out


START_POSES = {
    'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 0.0, 'wheel2': -21.50969293753206, 'wheel1': -43.01938587506412, 'shackle': 0.0, 'bolt': 0.0}},
}
GOAL_POSES = {
    'padlock': {'pos': (0.2395, -0.06, 0.8805), 'joints': {'dial': 441.0, 'wheel2': 627.4, 'wheel1': 905.2, 'shackle': 0.012, 'bolt': 0.0}},
}
START, GOAL = layout(START_POSES), layout(GOAL_POSES)

GOAL_MODULE = {}


def test_the_task_loads_with_its_loose_objects():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {'padlock'}
    assert TASK.docs and TASK.docs.strip()


def test_the_start_scores_nothing():
    assert T.score(START) == 0.0
    assert not T.success(START)


def test_the_goal_scores_full_and_succeeds(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    assert T.score(GOAL) == 1.0
    assert T.success(GOAL)


def test_success_waits_for_the_objects_to_settle(monkeypatch):
    module_values = {}
    for name, expr in GOAL_MODULE.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    moving = layout({n: {**p, "vel": (0.05, 0.0, 0.0)} for n, p in GOAL_POSES.items()})
    assert not T.success(moving)

# (name, the layout it starts from, what's moved, the score it should get, module overrides)
CASES = [
    ('untouched start', START_POSES, {
    }, 0, {}),
    ('first number set (right past two turns to 24)', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 864.0, 'wheel2': 884.6, 'wheel1': 905.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0.25, {}),  # Wheel 1's gate sits under the fence; wheel 2 and the cam do not.
    ('first and second numbers set (left one turn past 24 to 8)', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 288.0, 'wheel2': 627.4, 'wheel1': 905.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0.5, {}),
    ('all three numbers set, not pulled', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 441.0, 'wheel2': 627.4, 'wheel1': 905.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0.75, {}),
    ('second number two marks off, third set', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 441.0, 'wheel2': 645.4, 'wheel1': 905.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0.25, {}),  # Wheel 2's gate is 18 degrees from the fence, beyond the one-mark clearance.
    ('each gate just inside one mark', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 449.0, 'wheel2': 619.4, 'wheel1': 913.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0.75, {}),
    ('opened but the dial still pinched', GOAL_POSES, {
    }, 0.75, {'TEST_GRIPPERS': "{'left': (np.array([0.10, 0.25, TOP + 0.20]), 1.0), 'right': (np.array(TCP_DIAL) - [0, 0, SHACKLE_TRAVEL], 0.35)}"}),
    ('opened, released, but the lock still swinging', GOAL_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8805), 'vel': (0.0, 0.03, 0.0), 'joints': {'dial': 441.0, 'wheel2': 627.4, 'wheel1': 905.2, 'shackle': 0.012, 'bolt': 0.0}},
    }, 0.75, {}),
    ('dial spun right only (first gate set by the right turns, the rest scrambled)', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 1000.0000000000001, 'wheel2': 1020.6, 'wheel1': 1041.2, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0, {}),
    ('opened, then the lock knocked off the hasp', GOAL_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8325), 'joints': {'dial': 441.0, 'wheel2': 627.4, 'wheel1': 905.2, 'shackle': 0.012, 'bolt': 0.0}},
    }, 0, {'TEST_STICKY': "('lock off the hasp',)"}),
    ('lock off the hasp earlier, put back open', GOAL_POSES, {
    }, 0, {'TEST_STICKY': "('lock off the hasp',)"}),  # Violate-then-restore; a lock that fell stays at zero.
    ('shackle open with the wheel pack scrambled (forced)', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8805), 'joints': {'dial': 0.0, 'wheel2': -21.50969293753206, 'wheel1': -43.01938587506412, 'shackle': 0.012, 'bolt': 0.0}},
    }, 0, {}),  # The bolt cannot move unless all three gates are under the fence; an open shackle without them means the lock was forced.
    ('dialled straight right to 24 without the full turns', START_POSES, {
        'padlock': {'pos': (0.2395, -0.06, 0.8925), 'joints': {'dial': 144.0, 'wheel2': -21.50969293753206, 'wheel1': -43.01938587506412, 'shackle': 0.0, 'bolt': 0.0}},
    }, 0, {}),  # From the reset (pack cleared to the left) the cam has not yet picked up wheel 2, so wheel 1 is not set.
]


@pytest.mark.parametrize(("name", "base", "moved", "expected", "module"), CASES, ids=[c[0] for c in CASES])
def test_named_layouts(name, base, moved, expected, module, monkeypatch):
    module_values = {}
    for name, expr in module.items():
        module_values[name] = eval(expr, {**vars(T), **module_values, "math": math, "__builtins__": {}})
    for name, value in module_values.items():
        monkeypatch.setattr(T, name, value)
    objects = layout({**base, **moved})
    assert T.score(objects) == pytest.approx(expected), name
    assert bool(T.success(objects)) == (expected == 1.0), name


def test_the_demo_is_reachable_within_the_joint_limits():
    plan = TASK.demo()
    assert set(plan) <= {"left", "right"}
    for side, waypoints in plan.items():
        arm = ArmPlan(waypoints, station.BASES[side])
        steps = round(arm.duration * 20) + 5
        actions = np.array([arm.action7((k + 1) / 20)[0] for k in range(steps)])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-6) and np.all(actions[:, :6] <= ik.UPPER + 1e-6), side


def test_the_demo_fits_in_the_run_limit():
    longest = max(ArmPlan(w, station.BASES[s]).duration for s, w in TASK.demo().items())
    assert (longest + TASK.success_hold_s) * 20 < RUN_LIMIT_STEPS * 0.6
