#!/usr/bin/env python3
"""
Ishara — central configuration (Phase 3c-1).

Every constant from robot_phase4.py's CONFIG section, moved here verbatim so all
modules share one source of truth. Values are unchanged. robot_phase4.py now does
`from config import *` in place of the old inline block, so every existing
reference (CONF_TH, PAN_PIN, STM_PORT, clamp, ...) resolves exactly as before.
"""

import os

# ----- paths / device -----
BASE = "/home/hudai/Desktop/thesis"
CKPT = os.path.join(BASE, "checkpoints")
ASSETS = os.path.join(BASE, "assets")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
AUDIO_DIR = os.path.join(BASE, "audio")
DEVICE = "cpu"

# ----- STM32 link / phone -----
STM_PORT, STM_BAUD = "/dev/ttyACM0", 115200
CMD_FWD, CMD_BACK, CMD_LEFT, CMD_RIGHT, CMD_STOP = "F", "B", "L", "R", "S"
TELE_PREFIX = "D"
PHONE_PORT = 8000

# ----- landmarks / features -----
POSE_N, HAND_N = 33, 21
L_SHO, R_SHO = 11, 12
N_LM = POSE_N + 2 * HAND_N
FEATURE_DIM = N_LM * 3 + N_LM
N_FRAMES = 64

# ----- recognition -----
CONF_TH = 0.55
VOTE_NEED = 4
RECOG_WINDOW = 12.0
NAV_COOLDOWN = 3.0
HAND_RECENT = 0.5
DOUBLE_WINDOW = 2.0
LEFT_DOUBLE_WINDOW = 1.0

# ----- following -----
CENTER_DEAD = 0.12
STOP_DIST = 40
FOLLOW_DIST = 70
AUTO_STOP_AFTER = 0.6
# following uses the CAMERA first (shoulder width = distance); sonar is only a safety stop
USE_SONAR = True         # set False if your sonar misreads and wrongly blocks following
FOLLOW_NEAR_W = 0.40     # shoulders wider than this in frame => too close => stop
FOLLOW_FAR_W = 0.22      # shoulders narrower than this => person is far => move forward
STEER_DIR = 1            # set to -1 if the robot turns the WRONG way (mirrored camera)

# ----- head-tracking servos on GPIO22 (pan) / GPIO23 (tilt) -----
HT_DEAD = 0.05; HT_KP = 11.0; HT_KD = 2.0; HT_SMOOTH = 0.6
HT_MAX_STEP = 2.5; HT_MIN_MOVE = 0.6; HT_SETTLE = 0.04; HT_CENTER = 135.0
PAN_PIN, TILT_PIN = 22, 23
PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = 30, 240, 60, 210
PAN_DIR, TILT_DIR = -1, 1

# optional: fix specific words if auto-transliteration looks off  {class_id: "banglish"}
ROMAN_OVERRIDE = {}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))
