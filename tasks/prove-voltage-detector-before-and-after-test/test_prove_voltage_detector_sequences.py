"""Offline checks of the scorer over time (no GPU): a proof stops counting once the detector is switched off or
leaves both hands. Each test plays control steps at 20 Hz through score(), setting the buttons' travel, the
detector's pose and the grippers' centre points the way the simulator would.

    pytest test_prove_voltage_detector_sequences.py    (in an environment with yam_arena installed)
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from yam_arena.task_file import load

TASK = load(Path(__file__).with_name("prove_voltage_detector_before_and_after_test.py"))
T = TASK.module
DT = 0.05
_EPISODES = itertools.count(10_000)

HOME = (T.X0, T.Y0, T.TOP + T.AXIS_Z)                                   # lying on its posts, as at the start
PLATE = (T.X0, T.PLATE_Y - T.TIP[1], T.TOP + T.AXIS_Z)                  # top end on the tester's end plate
BAR = (0.29, T.BAR_Y + T.BAR_R + 0.001 - T.TIP[1], T.TOP + T.AXIS_Z + 0.005)   # top end 1 mm from the bar
AWAY = (T.X0, T.Y0 + T.AWAY, T.TOP + T.AXIS_Z)                          # laid down 10 cm from the bar
PARKED = np.array([0.10, 0.30, T.TOP + 0.35])                          # a gripper well clear of everything
OVER_TEST = np.array([T.X0, T.TEST_Y, T.TOP + 0.19])


class Trial:
    """One episode of the live scorer, stepped by hand."""

    def __init__(self, monkeypatch, flat=False):
        monkeypatch.setattr(T, "TEST_FLAT", flat)
        monkeypatch.setattr(T, "TEST_RECORD", None)
        monkeypatch.setattr(T, "LOG_EVENTS", False)
        self.mp = monkeypatch
        for key in ("t", "step", "episode"):
            monkeypatch.setitem(T._RUN, key, T._RUN[key])          # restored after the test
        T._RUN.update(t=0.0, step=0, episode=next(_EPISODES))
        monkeypatch.setattr(T, "_LIVE", T._new_live())
        self.score = None

    def hold(self, seconds, at=HOME, on=False, test=False, held=True, left=None):
        """Keep this state for `seconds`: detector at `at`, ON / TEST pressed or not, the left hand on the stick
        (held), parked clear of everything (not held) or at `left`."""
        det = T.DETECTOR.offline_state(at, 0.0, {"stick": 0.0, "on_button": -0.0015 if on else 0.0})
        tst = T.TESTER.offline_state(T.TESTER.pos, 0.0, {"test_button": -0.002 if test else 0.0})
        if left is None:
            left = np.array([at[0], at[1] + T.BAL, at[2]]) if held else PARKED
        self.mp.setattr(T, "TEST_GRIPPERS", {"left": (left, 0.0), "right": (OVER_TEST, 0.0)})
        objects = {"detector": det, "tester": tst}
        for _ in range(max(1, round(seconds / DT))):
            T._RUN["step"] += 1
            T._RUN["t"] = round(T._RUN["t"] + DT, 6)
            self.score = T.score(objects)
        return self

    def press_on(self, at=HOME, held=True):
        return self.hold(0.3, at, on=True, held=held).hold(0.3, at, held=held)

    def switch_on_and_prove(self):
        """ON, the full 3 s self-test, then 1.5 s on the plate with TEST held (the lights flash)."""
        return self.press_on().hold(3.2).hold(1.5, PLATE, test=True).hold(0.5, PLATE)

    @property
    def rec(self):
        return T._LIVE["rec"]


def test_the_poses_used_here_mean_what_they_say():
    assert T.bar_gap(T.DETECTOR.offline_state(BAR, 0.0, {})) <= T.TOUCH_GAP
    assert T.on_plate(T.DETECTOR.offline_state(PLATE, 0.0, {}))
    assert T.bar_gap(T.DETECTOR.offline_state(AWAY, 0.0, {})) >= T.CLEAR_BAR


def test_prove_then_touch_earns_credit(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    assert trial.score == pytest.approx(0.2)
    trial.hold(1.2, BAR)
    assert trial.score == pytest.approx(0.6)


def test_off_then_on_then_an_immediate_touch_is_a_permanent_zero(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    assert trial.score == pytest.approx(0.2)
    trial.press_on()                      # off: the proof is void
    trial.press_on()                      # on again: a new self-test starts
    assert trial.rec["proved"] is False
    trial.hold(1.2, BAR)                  # touched during the self-test
    assert trial.rec["bad_touch"] and trial.score == 0.0
    trial.hold(3.0).hold(1.5, PLATE, test=True).hold(0.5, PLATE).hold(1.2, BAR)
    assert trial.score == 0.0             # proving it afterwards does not undo that


def test_off_then_on_then_touch_after_the_self_test_but_before_a_new_proof_is_a_permanent_zero(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    trial.press_on().press_on().hold(3.2)  # self-test over, but not proved again
    trial.hold(1.2, BAR)
    assert trial.rec["bad_touch"] and trial.score == 0.0


def test_off_then_on_then_self_test_then_prove_then_touch_earns_credit(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    trial.press_on().press_on()
    trial.hold(3.2)                                          # the new self-test finishes
    trial.hold(1.5, PLATE, test=True).hold(0.5, PLATE)       # proved again on the tester
    assert trial.rec["proved"] and not trial.rec["bad_touch"]
    trial.hold(1.2, BAR)
    assert trial.score == pytest.approx(0.6)
    trial.hold(0.5, PLATE).hold(1.5, PLATE, test=True).hold(0.5, PLATE)
    assert trial.score == pytest.approx(0.8)


def test_proof_then_off_then_touch_while_off_is_a_permanent_zero(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    trial.press_on()                      # off
    trial.hold(1.2, BAR)
    assert trial.rec["bad_touch"] and trial.score == 0.0


def test_switching_off_takes_back_the_touch_credit_that_rested_on_the_proof(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove().hold(1.2, BAR)
    assert trial.score == pytest.approx(0.6)
    trial.hold(0.5, PLATE).press_on(PLATE)   # off before the second proof
    assert trial.score == pytest.approx(0.1)
    trial.press_on(PLATE).hold(3.2, PLATE).hold(2.5, PLATE, test=True)
    assert trial.score == pytest.approx(0.2)   # a proof now does not count as the proof after that touch


def test_proof_then_laid_down_then_picked_up_and_touched_is_a_permanent_zero(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    trial.hold(1.5, HOME, held=False)     # laid back on its posts and let go
    assert trial.rec["proved"] is False and trial.score == pytest.approx(0.1)
    trial.hold(0.5, HOME).hold(1.2, BAR)  # picked up again and put on the bar without a new proof
    assert trial.rec["bad_touch"] and trial.score == 0.0


def test_proof_then_dropped_on_the_table_voids_it(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    fallen = (T.X0, T.Y0, T.TOP + T.BODY_T / 2)   # slipped out of the hand onto the table
    hand = np.array([T.X0, T.Y0 + T.BAL, T.TOP + T.AXIS_Z])   # the open hand still where it held the stick
    trial.hold(1.0, fallen, left=hand)
    assert trial.rec["proved"] is False
    trial.hold(0.5, HOME).hold(1.2, BAR)
    assert trial.score == 0.0


def test_a_new_proof_after_picking_it_up_again_counts(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove()
    trial.hold(1.5, HOME, held=False).hold(0.5, HOME)              # let go, picked up again
    trial.hold(1.5, PLATE, test=True).hold(0.5, PLATE).hold(1.2, BAR)
    assert trial.score == pytest.approx(0.6)


def test_a_finished_prove_touch_prove_stands_after_switching_off_and_laying_down(monkeypatch):
    trial = Trial(monkeypatch).switch_on_and_prove().hold(1.2, BAR)
    trial.hold(0.5, PLATE).hold(1.5, PLATE, test=True).hold(0.5, PLATE)
    assert trial.score == pytest.approx(0.8)
    trial.press_on(PLATE)                 # switched off after the second proof, as a lineworker does
    trial.hold(0.5, AWAY).hold(2.5, AWAY, held=False)
    assert trial.score == 1.0 and T.success({"detector": T.DETECTOR.offline_state(AWAY, 0.0, {}),
                                             "tester": T.TESTER.offline_state(T.TESTER.pos, 0.0, {})})


def test_a_flat_tester_never_trusts_the_detector_even_after_a_restart(monkeypatch):
    trial = Trial(monkeypatch, flat=True).switch_on_and_prove()
    assert trial.rec["proved"] is False
    trial.press_on().press_on().hold(3.2).hold(1.5, PLATE, test=True).hold(1.2, BAR)
    assert trial.score == 0.0
