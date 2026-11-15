"""Put a smartphone into its hybrid case the right way round and snap all four corners under the case's lip.

An iPhone 15-class phone (147.6 x 71.6 x 7.80 mm, 171 g) lies screen up on a felt mat, its top end pointing away from
the robot. Beside it a clear hybrid case (polycarbonate back, TPU bumper; 151.4 x 75.4 mm footprint, 31 g) lies open,
back down, its long axis left-right and its camera cutout at the left (+y) end on the far side. The phone has to be
turned a quarter turn so its camera block drops into the cutout, set on the case's lip, and each end pressed down
corner by corner until the corners slip under the lip. The bumper's lip is represented at the two ends by four
sprung corner latches (end-wall sections that slide outward against a spring and snap back over the glass); the
camera block props the phone up if it is not in the cutout. Zero object jitter; 20 Hz.
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from yam_parts import box
from yam_joints import Link, Slide
from yam_mesh import Mesh, MeshMechanism, MeshPart
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"

# Release-and-stillness block, verbatim from hang_holder_on_keyhole_studs.py (see release-stillness.md).
# The task module must define: import math; import numpy as np; from yam_arena.constants import FINGER_OPEN;
# and SCENE ("station" or "tabletop"). Paste after the imports.

# Release and stillness
# Released means every gripper TCP is at least RELEASE_CLEAR from the object's box.
# Still means below the task's linear and angular speed limits for STILL_S seconds.
# Evaluate every sticky condition and timer at the top of score() on every call,
# before any early return, so interrupted conditions cannot retain elapsed time.
RELEASE_CLEAR = 0.030
STILL_S = 1.0
TEST_GRIPPERS = None


def _run_context():
    context = {"env": None, "t": None, "step": 0, "episode": 0}
    try:
        from yam_arena import inspect_embodiment as _ie
    except ImportError:
        return context
    cls = _ie.YamArenaEmbodiment
    if hasattr(cls, "_rc_run"):
        return cls._rc_run
    original_step, original_reset = cls._task_step, cls.reset

    def task_step(self, *args, **kwargs):
        context["env"], context["t"] = self._env, self._t
        context["step"] += 1
        return original_step(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        context["episode"] += 1
        context["t"] = 0.0
        context["env"] = self._env
        return result

    task_step._rc_wrapped = reset._rc_wrapped = True
    cls._rc_run = context
    cls._task_step, cls.reset = task_step, reset
    return context


_RUN = _run_context()
_MEMORY_EPISODE = None
_TIMERS = {}
_EVER = {}


def grippers():
    if TEST_GRIPPERS is not None:
        return TEST_GRIPPERS
    env = _RUN["env"]
    if env is None:
        return None
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, LEFT_BASE_POS, RIGHT_BASE_POS

    bases = station.BASES if SCENE == "station" else {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
    out = {}
    for arm in ("left", "right"):
        art = env.unwrapped.scene[arm + "_robot"]
        arm_ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        finger_ids = art.find_joints(FINGER_JOINTS, preserve_order=True)[0]
        joints = art.data.joint_pos
        q = getattr(joints, "torch", joints)[0]
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        q = np.asarray(q)
        tcp = ik.tcp_pose(q[arm_ids], bases[arm])[0]
        opening = float(np.clip(np.mean(q[finger_ids]) / FINGER_OPEN, 0.0, 1.0))
        out[arm] = (tcp, opening)
    return out


def released(state, clear=RELEASE_CLEAR):
    hands = grippers()
    if hands is None:
        return True  # Offline arm poses are unknown, so release is assumed.
    low, high = np.asarray(state.low), np.asarray(state.high)
    return all(np.linalg.norm(np.maximum(np.maximum(low - tcp, tcp - high), 0.0)) >= clear
               for tcp, _ in hands.values())


def _episode_memory():
    global _MEMORY_EPISODE
    if _MEMORY_EPISODE != _RUN["episode"]:
        _TIMERS.clear()
        _EVER.clear()
        _MEMORY_EPISODE = _RUN["episode"]


def held_for(key, cond):
    """Elapsed time over consecutive observed control steps; repeat calls are harmless."""
    _episode_memory()
    t, step = _RUN["t"], _RUN["step"]
    if t is None:
        return math.inf if cond else 0.0
    previous_step, start = _TIMERS.get(key, (None, None))
    if not cond:
        _TIMERS[key] = (step, None)
        return 0.0
    if start is None or step not in (previous_step, previous_step + 1) or t < start:
        start = t
    _TIMERS[key] = (step, start)
    return float(t - start)


def ever(key, cond):
    """Remember a satisfied condition until the next episode; offline each call stands alone."""
    if _RUN["t"] is None: return bool(cond)
    _episode_memory()
    _EVER[key] = bool(_EVER.get(key, False) or cond)
    return _EVER[key]



# ---------------------------------------------------------------------------------------------------------------
# Layout (metres; world x forward, y left, z up; arm bases at (0, +-0.305))
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

MAT_T = 3.0 * MM
MAT_C, MAT_SIZE = (0.180, 0.0), (0.260, 0.500)
MAT_TOP = TOP + MAT_T

# Phone (phone frame: x toward its top end, y to the screen's left, z out of the screen; origin at the body centre)
PH_L, PH_W, PH_T = 147.6 * MM, 71.6 * MM, 7.80 * MM
PH_MASS = 0.171
PH_BEVEL = 1.5 * MM                    # rounded aluminium edge, modelled as a 45-degree bevel top and bottom
CAM = (57.8 * MM, -19.8 * MM)          # camera block centre on the back: 16 mm from the top and right edges
CAM_SIZE, CAM_H = 28.0 * MM, 1.0 * MM  # square camera block, 1.0 mm proud of the back glass (stays inside the 1.2 mm back)
PHONE_START = (0.170, -0.170)          # lying screen up, top end pointing +x (away from the robot)

# Case (case frame: x along its length toward the camera end, y to the left looking into it, z up; origin at the
# centre of its underside). Placed with yaw +90 deg: case +x points world +y.
CASE_C = (0.170, 0.0)
CASE_YAW = math.pi / 2
CASE_L, CASE_W = 151.4 * MM, 75.4 * MM
BACK_T = 1.2 * MM                      # polycarbonate back
IN_HALF_L, IN_HALF_W = 74.0 * MM, 36.0 * MM   # inner pocket half-sizes (0.2 mm play at each end and side)
WALL_TOP = 10.3 * MM                   # bumper top, 1.3 mm above a seated screen
LIP_UNDER = 9.3 * MM                   # underside of the end lips (seated screen top 9.0 mm)
LIP_TIP = 72.8 * MM                    # lip edge: overlaps the screen's end edge by 1.0 mm
SEATED_TOP = BACK_T + PH_T             # 9.0 mm: screen height above the case underside when seated
CUT_X, CUT_Y = (41.8 * MM, 73.8 * MM), (-35.8 * MM, -3.8 * MM)   # camera cutout in the back
LATCH_TRAVEL = 2.5 * MM
LATCH_K = 1500.0                       # N/m
LATCH_PRELOAD = 1.3 * MM               # spring rest beyond the closed stop: 1.95 N closed, 3.75 N fully open
LATCH_Y = (0.5 * MM, 35.9 * MM)        # each corner section runs from the corner to the middle of the end
CASE_MASS, LATCH_MASS = 0.023, 0.002   # 31 g in all

CLEAR_TPU = (0.10, 0.18, 0.42)   # navy blue bumper
PC_CLEAR = (0.16, 0.26, 0.52)    # navy blue tinted back


def _convex(name, rings, color=None):
    """Closed convex mesh through stacked 4-point rings (metres), bottom ring first."""
    pts = np.array([p for ring in rings for p in ring], float)
    n, m = 4, len(rings)
    faces = []
    for r in range(m - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = r * n + i, r * n + j, (r + 1) * n + j, (r + 1) * n + i
            faces += [(a, b, c), (a, c, d)]
    faces += [(0, 2, 1), (0, 3, 2)]
    t = (m - 1) * n
    faces += [(t, t + 1, t + 2), (t, t + 2, t + 3)]
    faces = np.array(faces, int)
    tri = pts[faces]
    centre = pts.mean(0)
    if np.einsum('ij,ij->i', tri[:, 0] - centre, np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() < 0:
        faces = faces[:, ::-1]
    lo, hi = pts.min(0), pts.max(0)
    return Mesh(name, pts, faces, collision="convex", color=color, proxy=(box((lo + hi) / 2, hi - lo),))


def _rect(hx, hy, z):
    return [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]


def _phone_hull():
    hx, hy, hz, b = PH_L / 2, PH_W / 2, PH_T / 2, PH_BEVEL
    return _convex("phone_body", [_rect(hx - b, hy - b, -hz), _rect(hx, hy, -hz + b), _rect(hx, hy, hz - b),
                                  _rect(hx - b, hy - b, hz)], color=(0.45, 0.62, 0.82))


def _lip(name, sign, y0, y1):
    """Lip section at the +x (sign=+1) or -x (sign=-1) end: 1.0 mm thick, its edge 1.0 mm over the screen, a
    45-degree lead-in on its inner top edge."""
    xz = [(72.8, 9.3), (75.7, 9.3), (75.7, 10.3), (73.6, 10.3), (72.8, 9.5)]
    rings = []
    for y in (y0, y1):
        rings.append([(sign * x * MM, y, z * MM) for x, z in xz])
    pts = np.array([p for r in rings for p in r], float)
    n = len(xz)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    for k in range(1, n - 1):
        faces += [(0, k + 1, k), (n, n + k, n + k + 1)]
    faces = np.array(faces, int)
    tri = pts[faces]
    if np.einsum('ij,ij->i', tri[:, 0] - pts.mean(0), np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() < 0:
        faces = faces[:, ::-1]
    lo, hi = pts.min(0), pts.max(0)
    return Mesh(name, pts, faces, collision="convex", color=CLEAR_TPU, proxy=(box((lo + hi) / 2, hi - lo),))


LATCHES = ("top_near", "top_far", "bottom_near", "bottom_far")   # case +x end = phone top (camera) end
_LATCH_GEOM = {  # name: (end sign, y0, y1); case +y faces the robot (world -x) after the 90-degree yaw
    "top_near": (+1, LATCH_Y[0], LATCH_Y[1]),
    "top_far": (+1, -LATCH_Y[1], -LATCH_Y[0]),
    "bottom_near": (-1, LATCH_Y[0], LATCH_Y[1]),
    "bottom_far": (-1, -LATCH_Y[1], -LATCH_Y[0]),
}


def _latch(name):
    s, y0, y1 = _LATCH_GEOM[name]
    wall_x = s * (IN_HALF_L + 0.85 * MM)
    zc = (BACK_T + LIP_UNDER) / 2
    anchor = (wall_x, (y0 + y1) / 2, zc)
    lo, hi = (0.0, LATCH_TRAVEL) if s > 0 else (-LATCH_TRAVEL, 0.0)
    return Link(name, [box(anchor, (1.7 * MM, y1 - y0, LIP_UNDER - BACK_T))],
                Slide("x", anchor, lo, hi, damping=5.0, spring=LATCH_K, rest=-s * LATCH_PRELOAD, armature=0.01),
                mass=LATCH_MASS, color=CLEAR_TPU, friction=(0.25, 0.2))


def _back():
    hx, hy = CASE_L / 2, CASE_W / 2
    z = BACK_T / 2
    c0, c1 = CUT_X
    d0, d1 = CUT_Y

    def piece(x0, x1, y0, y1):
        return box(((x0 + x1) / 2, (y0 + y1) / 2, z), (x1 - x0, y1 - y0, BACK_T))
    return [piece(-hx, c0, -hy, hy), piece(c0, hx, d1, hy), piece(c0, hx, -hy, d0), piece(c1, hx, d0, d1)]


def _walls():
    hx, hy = CASE_L / 2, CASE_W / 2
    t = hy - IN_HALF_W
    return [box((0, s * (IN_HALF_W + t / 2), WALL_TOP / 2), (2 * hx, t, WALL_TOP), color=CLEAR_TPU) for s in (1, -1)]


CASE = MeshMechanism("case", (CASE_C[0], CASE_C[1], MAT_TOP), _back() + _walls(), [_latch(n) for n in LATCHES],
                     mass=CASE_MASS, color=PC_CLEAR, yaw=CASE_YAW, friction=(0.8, 0.7), contact_offset=0.001,
                     meshes={n: (_lip(n + "_lip", *_LATCH_GEOM[n]),) for n in LATCHES})

PHONE = MeshPart("phone", (PHONE_START[0], PHONE_START[1], MAT_TOP + PH_T / 2), [
    box((CAM[0], CAM[1], -PH_T / 2 - CAM_H / 2), (CAM_SIZE, CAM_SIZE, CAM_H), color=(0.15, 0.16, 0.18)),
    box((0, 0, PH_T / 2 + 0.05 * MM), (PH_L - 4 * MM, PH_W - 4 * MM, 0.1 * MM), color=(0.03, 0.03, 0.04)),  # dark screen
    box((PH_L / 2 - 8 * MM, 0, PH_T / 2 + 0.15 * MM), (3.5 * MM, 10 * MM, 0.1 * MM), color=(0.85, 0.55, 0.10)),  # earpiece mark
], mass=PH_MASS, color=(0.45, 0.62, 0.82), friction=(0.5, 0.4), contact_offset=0.001, grip=3,
                  meshes=(_phone_hull(),))

MAT = MeshPart("mat", (MAT_C[0], MAT_C[1], TOP), [box((0, 0, MAT_T / 2), (MAT_SIZE[0], MAT_SIZE[1], MAT_T))],
               color=(0.58, 0.58, 0.56), friction=(0.8, 0.7))

OBJECTS = [MAT, CASE, PHONE]
PRESSED_PARTS = ("case", "phone")
SEATED_PHONE_Z = MAT_TOP + BACK_T + PH_T / 2

# ---------------------------------------------------------------------------------------------------------------
# Scoring
CORNERS = {  # latch name: the phone corner (phone frame, screen top) it holds when the phone is the right way round
    "top_near": (PH_L / 2, PH_W / 2), "top_far": (PH_L / 2, -PH_W / 2),
    "bottom_near": (-PH_L / 2, PH_W / 2), "bottom_far": (-PH_L / 2, -PH_W / 2),
}
UNDER_LIP = LIP_UNDER          # screen corner at or below the lip's underside (at least 1.0 mm below the lip top)
MIN_OVERLAP = 0.3 * MM         # lip edge at least this far over the screen's end edge
SIDE_TOL = 1.0 * MM            # corner inside the pocket sideways
UPRIGHT = 10.0                 # degrees: screen up
ALIGNED = 10.0                 # degrees: phone long axis on the case's, camera end at the cutout end
ON_CASE_XY = 10.0 * MM         # partial: phone centre this close to the case centre ...
ON_CASE_Z = 20.0 * MM          # ... and its screen this low over the case underside (resting on the lip: 18.1)
CASE_FLAT = 3.0                # degrees
CASE_MOVE = 0.050              # m from its start
DROP_SPEED = 0.80              # m/s downward: a fall of about 33 mm
STILL_MOVE = 0.5 * MM
STILL_TURN = 1.0
TEST_PAST_HAZARD = False


def _case_frame(case):
    return np.asarray(case.origin, float), case.rotation()


def _to_case(case, p_world):
    o, R = _case_frame(case)
    return R.T @ (np.asarray(p_world, float) - o)


def right_way_round(phone, case):
    _, Rc = _case_frame(case)
    Rp = phone.rotation()
    up = float(np.dot(Rp[:, 2], Rc[:, 2]))
    along = float(np.dot(Rp[:, 0], Rc[:, 0]))
    return up >= math.cos(math.radians(UPRIGHT)) and along >= math.cos(math.radians(ALIGNED))


def corner_seated(phone, case, name):
    cx, cy = CORNERS[name]
    p = _to_case(case, np.asarray(phone.origin, float) + phone.rotation() @ np.array([cx, cy, PH_T / 2]))
    q = abs(float(case.joints[name]))
    end_edge = abs(float(p[0]))
    overlap = end_edge - (LIP_TIP + q)
    return bool(SEATED_TOP - 1.0 * MM <= p[2] <= UNDER_LIP + 1e-9 and overlap >= MIN_OVERLAP
                and end_edge <= IN_HALF_L + SIDE_TOL and np.sign(p[0]) == np.sign(cx)
                and abs(float(p[1])) <= IN_HALF_W + SIDE_TOL)


def seated_count(phone, case):
    if not right_way_round(phone, case):
        return 0
    return sum(corner_seated(phone, case, n) for n in LATCHES)


def on_case(phone, case):
    p = _to_case(case, phone.origin)
    return bool(math.hypot(float(p[0]), float(p[1])) <= ON_CASE_XY and p[2] + PH_T / 2 <= ON_CASE_Z)


def case_flat_home(case):
    tilt = math.degrees(math.acos(float(np.clip(case.rotation()[2, 2], -1, 1))))
    o = np.asarray(case.origin, float)
    return bool(tilt <= CASE_FLAT and math.hypot(o[0] - CASE_C[0], o[1] - CASE_C[1]) <= CASE_MOVE
                and abs(o[2] - MAT_TOP) <= 2.0 * MM)


def off_mat(s):
    o = np.asarray(s.origin, float)
    return bool(abs(o[0] - MAT_C[0]) > MAT_SIZE[0] / 2 or abs(o[1] - MAT_C[1]) > MAT_SIZE[1] / 2 or o[2] < MAT_TOP - 0.010)


def screen_down(phone):
    return float(phone.rotation()[2, 2]) < -0.5


def dropped(phone):
    return float(np.asarray(phone.lin_vel, float)[2]) < -DROP_SPEED


_HIST = {"episode": None, "parts": {}}


def part_still(s, key):
    """Stillness from the pose over the last STILL_S seconds (contact-chatter-proof); velocity offline."""
    if _RUN["t"] is None:
        return s.resting()
    if _HIST["episode"] != _RUN["episode"]:
        _HIST.update(episode=_RUN["episode"], parts={})
    hist = _HIST["parts"].setdefault(key, [])
    t = float(_RUN["t"])
    if not hist or hist[-1][0] != t:
        hist.append((t, np.asarray(s.origin, float).copy(), s.rotation()[:, 2].copy()))
    while hist and hist[0][0] < t - STILL_S - 1e-6:
        hist.pop(0)
    if hist[0][0] > t - STILL_S + 0.06:
        return False
    p, z = hist[-1][1], hist[-1][2]
    return all(np.linalg.norm(q - p) <= STILL_MOVE and
               math.degrees(math.acos(float(np.clip(np.dot(zq, z), -1, 1)))) <= STILL_TURN for _, q, zq in hist)


def score(objects):
    phone, case = objects["phone"], objects["case"]
    bad = ever("phone or case off the mat", off_mat(phone) or off_mat(case))
    bad = ever("phone screen down", screen_down(phone)) or bad
    bad = ever("phone dropped", dropped(phone)) or bad
    bad = bad or bool(TEST_PAST_HAZARD)
    settled = released(phone) and part_still(phone, "phone") and part_still(case, "case")
    n = seated_count(phone, case)
    resting = settled and right_way_round(phone, case) and on_case(phone, case)
    if bad or not settled:
        return 0.0
    if n == 4:
        return 1.0 if case_flat_home(case) else 0.75
    if n == 3:
        return 0.75
    if n == 2:
        return 0.5
    if n == 1:
        return 0.25
    return 0.1 if resting else 0.0


def success(objects):
    return score(objects) >= 1.0


# ---------------------------------------------------------------------------------------------------------------
# Demo: authored layout and elapsed time only
PAD = 0.013                       # pad tips below the TCP
FOFF = 0.047 + 0.00175            # open gripper: finger centre from the TCP along the closing axis
GRIP = -0.030                     # pads aimed 60 mm apart on the 71.6 mm phone: a firm squeeze
YAW_X = math.pi / 2               # fingers close along world x
PRESS_TCP = TOP + 0.0135          # press target: pad tips 0.5 mm above the table top (about 21 mm past first contact)
CORNER_IN = 9.0 * MM              # press point: 9 mm in from the phone's end ...
SIDE_IN = 6.0 * MM                # ... and 6 mm in from its side, clear of the lip


def _w(dt, x, y, z, f, label, yaw=0.0):
    return Waypoint(dt, (float(x), float(y), float(z)), float(f), label, float(yaw))


def _corner_press(xp, yp, side, yaw, label):
    """One-pad press: the pad nearer the phone lands on the corner, the TCP sits FOFF further out along x."""
    x = xp + side * FOFF
    contact = MAT_TOP + WALL_TOP + PH_T + PAD
    return [
        _w(1.5, x, yp, contact + 0.030, FINGER_OPEN, f"over the {label} corner", yaw),
        _w(1.0, x, yp, contact + 0.002, FINGER_OPEN, f"pad on the {label} corner", yaw),
        _w(1.5, x, yp, PRESS_TCP, FINGER_OPEN, f"press the {label} corner down", yaw),
        _w(1.0, x, yp, PRESS_TCP, FINGER_OPEN, f"hold the {label} corner", yaw),
        _w(1.0, x, yp, contact + 0.030, FINGER_OPEN, f"lift off the {label} corner", yaw),
    ]


def demo():
    cx, cy = CASE_C
    px, py = PHONE_START
    grip_z = MAT_TOP + 1.5 * MM + PAD
    place_z = MAT_TOP + WALL_TOP + 2.5 * MM + 1.5 * MM + PAD
    x_far, x_near = cx + PH_W / 2 - SIDE_IN, cx - PH_W / 2 + SIDE_IN
    y_end = PH_L / 2 - CORNER_IN
    right = [
        _w(1.5, px, py, TOP + 0.080, FINGER_OPEN, "over the phone"),
        _w(1.5, px, py, grip_z, FINGER_OPEN, "pads either side of the phone, tips 1.5 mm off the mat"),
        _w(1.0, px, py, grip_z, GRIP, "squeeze the phone's sides"),
        _w(1.5, px, py, TOP + 0.080, GRIP, "lift"),
        _w(2.5, cx, cy, TOP + 0.080, GRIP, "turn a quarter turn, top end to the cutout end", YAW_X),
        _w(1.5, cx, cy, place_z, GRIP, "lower onto the case's lip", YAW_X),
        _w(0.8, cx, cy, place_z, FINGER_OPEN, "let go on the lip", YAW_X),
        _w(1.0, cx, cy, TOP + 0.080, FINGER_OPEN, "rise", YAW_X),
        _w(1.5, 0.130, -0.200, TOP + 0.110, FINGER_OPEN, "park clear", YAW_X),
    ]
    t_place = sum(w.duration for w in right)
    left = [_w(1.5, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "park clear", -YAW_X),
            _w(t_place - 1.5, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "wait for the phone to be set on the case",
               -YAW_X)]
    left += _corner_press(x_far, cy + y_end, +1, -YAW_X, "top-end far")
    left += _corner_press(x_near, cy + y_end, -1, -YAW_X, "top-end near")
    left += [_w(1.5, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "park clear", -YAW_X)]
    t_left = sum(w.duration for w in left)
    right += [_w(t_left - sum(w.duration for w in right), 0.130, -0.200, TOP + 0.110, FINGER_OPEN,
                 "wait for the top end to be seated", YAW_X)]
    right += _corner_press(x_far, cy - y_end, +1, YAW_X, "bottom-end far")
    right += _corner_press(x_near, cy - y_end, -1, YAW_X, "bottom-end near")
    right += [_w(1.5, 0.130, -0.200, TOP + 0.110, FINGER_OPEN, "park clear", YAW_X)]
    t_right = sum(w.duration for w in right)
    # Second pass over the top end: a corner left riding on the lip when the bottom end snapped is pressed home.
    left += [_w(t_right - t_left, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "wait for the bottom end", -YAW_X)]
    left += _corner_press(x_far, cy + y_end, +1, -YAW_X, "top-end far (again)")
    left += _corner_press(x_near, cy + y_end, -1, -YAW_X, "top-end near (again)")
    left += [_w(1.5, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "park clear", -YAW_X),
             _w(3.0, 0.130, 0.200, TOP + 0.110, FINGER_OPEN, "hold clear", -YAW_X)]
    right += [_w(sum(w.duration for w in left) - t_right, 0.130, -0.200, TOP + 0.110, FINGER_OPEN, "hold clear",
                 YAW_X)]
    return {"left": left, "right": right}
