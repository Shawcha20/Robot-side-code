#!/usr/bin/env python3
"""
Ishara — camera capture + landmark drawing (Phase 3c-11).

Extracted verbatim from robot_phase4.py's SHARED STATE + CAPTURE section (the
shared dict itself moved to state.py in 3c-6; this is everything that was left).
Owns the camera loop: reads a frame, runs detection, updates the pose/hand fields
in `shared` that head-tracking and following read, draws the skeleton overlay for
the phone video stream, and writes the annotated frame back to shared["frame"].

Public name: capture_thread. HAND_CONN / POSE_CONN are module-level but only used
internally for drawing; nothing else in the codebase references them.

Depends on cv2 (external), time (stdlib), state (shared, buf_lock),
perception (detect_one, to_feat), config (L_SHO, R_SHO) — all "lower" modules
with no dependency on capture, so there is no circular import.
"""

import time
import cv2

from state import shared, buf_lock
from perception import detect_one, to_feat
from config import L_SHO, R_SHO

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]

def capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); shared["run"] = False; return
    print("camera opened OK")
    while shared["run"]:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c, m, lo, ro, pr, hr = detect_one(rgb)
        now = time.time()
        if lo: shared["l_last"] = now
        if ro: shared["r_last"] = now
        feat = to_feat(c, m)
        with buf_lock: shared["buf"].append(feat)

        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            nose = lm[0]
            shared["nose_x"], shared["nose_y"] = float(nose.x), float(nose.y)
            shared["pose_found"] = True
            shared["person_cx"] = float((c[L_SHO][0] + c[R_SHO][0]) / 2.0)
            shared["shoulder_w"] = float(abs(c[L_SHO][0] - c[R_SHO][0]))
            # left arm raised = left wrist (15) above left shoulder (11): reliable exit cue at distance
            shared["left_raised"] = bool(lm[15].y < lm[11].y - 0.03)
        else:
            shared["pose_found"] = False; shared["person_cx"] = None
            shared["shoulder_w"] = None; shared["left_raised"] = False

        if pr.pose_landmarks:
            pts = [(int(p.x*w), int(p.y*h)) for p in pr.pose_landmarks[0]]
            for a, b in POSE_CONN: cv2.line(frame, pts[a], pts[b], (255,150,0), 2)
            nx, ny = pts[0]; cv2.rectangle(frame, (nx-55, ny-55), (nx+55, ny+55), (0,255,255), 2)
        if hr.hand_landmarks:
            for hand in hr.hand_landmarks:
                hp = [(int(p.x*w), int(p.y*h)) for p in hand]
                for a, b in HAND_CONN: cv2.line(frame, hp[a], hp[b], (0,255,0), 2)
                for p in hp: cv2.circle(frame, p, 3, (0,0,255), -1)

        cv2.putText(frame, "MODE: " + shared["mode"], (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if shared["hint"]:
            cv2.putText(frame, shared["hint"], (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 2)
        shared["frame"] = frame
    cap.release()
