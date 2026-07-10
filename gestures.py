#!/usr/bin/env python3
"""
Ishara — gesture primitives (Phase 3c-8).

Extracted verbatim from robot_phase4.py. These are the debounced hand-gesture
reads the mode state machine is built on: hand_state (is a hand recently seen),
current_prediction (ensemble vote on the current buffer), clear_buffer, and
read_raw_gesture (the two-phase left/right/both classifier with a 0.4 s settle).

Reads shared / buf_lock from state, HAND_RECENT / N_FRAMES from config, predict
from model. robot_phase4.py now does
`from gestures import hand_state, current_prediction, clear_buffer, wait_hands_down, read_raw_gesture`
in place of the GESTURE PRIMITIVES block. Timing constants and logic are unchanged.
"""

import time

from config import HAND_RECENT, N_FRAMES
from state import shared, buf_lock
from model import predict


def hand_state():
    now = time.time()
    return (now - shared["l_last"]) < HAND_RECENT, (now - shared["r_last"]) < HAND_RECENT


def current_prediction():
    with buf_lock:
        if len(shared["buf"]) < N_FRAMES: return None, 0.0
        b = list(shared["buf"])
    return predict(b)


def clear_buffer():
    with buf_lock: shared["buf"].clear()


def wait_hands_down(timeout=4.0):
    t0 = time.time()
    while shared["run"]:
        l, r = hand_state()
        if not l and not r: return
        if time.time() - t0 > timeout: return
        time.sleep(0.05)


def read_raw_gesture(expect_mode, timeout=None):
    t_start = time.time()
    while shared["run"] and shared["mode"] == expect_mode:
        l, r = hand_state()
        if l or r: break
        if timeout is not None and time.time() - t_start > timeout: return None
        time.sleep(0.05)
    if not shared["run"] or shared["mode"] != expect_mode: return None
    sb = sl = sr = False; last_up = time.time()
    while shared["run"] and shared["mode"] == expect_mode:
        l, r = hand_state(); now = time.time()
        if l and r: sb = True; last_up = now
        elif l: sl = True; last_up = now
        elif r: sr = True; last_up = now
        elif now - last_up > 0.4: break
        time.sleep(0.05)
    if sb: return "both"
    if sl and not sr: return "left"
    if sr and not sl: return "right"
    if sr and sl: return "both"
    return None
