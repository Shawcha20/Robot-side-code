#!/usr/bin/env python3
"""
Ishara — perception: MediaPipe landmarks + feature extraction (Phase 3c-5).

Extracted verbatim from robot_phase4.py's MEDIAPIPE + FEATURES section. The pose
and hand landmarkers are created at import time from the .task assets (same as
before). Public names used by the app: detect_one, to_feat. _np, fingers_up and
norm_frame are internal helpers.

robot_phase4.py now does `from perception import detect_one, to_feat` in place of
this block. The 75-landmark layout (33 pose + 21 + 21 hands), the shoulder-based
normalization, and the 300-dim feature vector are unchanged.
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from config import ASSETS, N_LM, POSE_N, HAND_N, L_SHO, R_SHO

pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS, "pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS, "hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_hands=2))


def _np(lms): return np.array([[p.x, p.y, p.z] for p in lms], np.float32)


def fingers_up(hand):
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    f = sum(1 for tip, pip in zip(tips, pips) if hand[tip].y < hand[pip].y)
    if abs(hand[4].x - hand[2].x) > 0.05: f += 1
    return f


def detect_one(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr = pose_lm.detect(img); hr = hand_lm.detect(img)
    c = np.zeros((N_LM, 3), np.float32); m = np.zeros(N_LM, np.float32)
    lo = ro = False
    if pr.pose_landmarks:
        c[:POSE_N] = _np(pr.pose_landmarks[0]); m[:POSE_N] = 1
    if hr.hand_landmarks:
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            a = _np(lms); label = hd[0].category_name
            is_open = fingers_up(lms) >= 5
            if label == "Left":
                c[POSE_N:POSE_N + HAND_N] = a; m[POSE_N:POSE_N + HAND_N] = 1
                if is_open: lo = True
            else:
                c[POSE_N + HAND_N:] = a; m[POSE_N + HAND_N:] = 1
                if is_open: ro = True
    return c, m, lo, ro, pr, hr


def norm_frame(c, m):
    if m[L_SHO] > 0 and m[R_SHO] > 0:
        o = (c[L_SHO] + c[R_SHO]) / 2
        s = np.linalg.norm(c[L_SHO, :2] - c[R_SHO, :2]) + 1e-6
    else:
        pr = c[m > 0]
        if len(pr) == 0: return c
        o = pr.mean(0); s = pr[:, :2].std() + 1e-6
    out = c.copy(); out[m > 0] = (c[m > 0] - o) / s
    return out


def to_feat(c, m):
    return np.concatenate([norm_frame(c, m).reshape(-1), m]).astype("float32")
