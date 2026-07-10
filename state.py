#!/usr/bin/env python3
"""
Ishara — shared runtime state (Phase 3c-6).

The single `shared` dict and `buf_lock` that every subsystem reads and writes,
extracted verbatim from robot_phase4.py. Both are mutable objects that are only
ever mutated (shared["mode"] = ...), never reassigned, so every module does
`from state import shared, buf_lock` and they all reference the same objects —
writes made anywhere are visible everywhere, exactly as when it was one file.

This module depends only on config, so nothing imports back into it: there is no
circular dependency. It is the piece that lets capture / head-tracking / gestures /
modes each import the shared state directly in the next increments.
"""

import threading
from collections import deque

from config import N_FRAMES

shared = {"buf": deque(maxlen=N_FRAMES), "run": True, "frame": None,
          "l_last": 0.0, "r_last": 0.0, "mode": "idle", "collecting": False,
          "person_cx": None, "shoulder_w": None, "left_raised": False,
          "nose_x": 0.5, "nose_y": 0.5, "pose_found": False,
          "last_word": "", "hint": "", "drive_last": 0.0, "driving": False}
buf_lock = threading.Lock()
