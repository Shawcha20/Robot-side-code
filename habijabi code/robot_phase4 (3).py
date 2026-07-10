#!/usr/bin/env python3
"""
PHASE 4 - Ishara robot  (Raspberry Pi 5 side, self-contained)

Bengali Sign Language word recognition + real-time translation on a rolling robot.
ONE shared camera + MediaPipe Holistic pass drives both head-tracking and the
recognition feature buffer. A phone browser (http://<pi-ip>:8000) gives manual
drive, arm control, and mode switching. Recognised words are spoken aloud and
shown (transliterated) on the robot's OLED.

STM32 serial protocol (must match stm32_phase4.ino):
    Pi -> STM32:
      F/B/L/R/S                motors fwd/back/spin-left/spin-right/stop
      G/H/J/K                  diagonals fwd-left/fwd-right/back-left/back-right
      ALU/ALD/ARU/ARD          arm nudge (10 deg)
      AL<deg>/AR<deg>          arm absolute angle 0-90 (e.g. AL045)
      #<text>                  OLED top status ('|' = line break)
      $<text>                  OLED bottom words (scrolls)
    STM32 -> Pi:
      D<distance_cm>           sonar, every 100 ms

Flash the matching stm32_phase4.ino first.

------------------------------------------------------------------------------
NOTE: This is a complete, to-spec reconstruction. Confirm these before running:
  * CAM_INDEX           (camera device number)
  * HT_PAN_PIN/TILT_PIN (head-servo BCM pins; head_tracking_final.py used 18/13)
  * MODEL_CKPTS         (checkpoint filename(s) under BASE/checkpoints)
  * AUDIO naming        (one <id>.wav per class id in BASE/audio)
------------------------------------------------------------------------------
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["PYTHONUTF8"] = "1"

import re
import time
import json
import threading
import subprocess
from collections import deque, Counter

import numpy as np

# Hardware / heavy libs are imported lazily-tolerant so the file still parses and
# gives a clear message if something is missing on a fresh Pi.
try:
    import cv2
except Exception as e:                       # pragma: no cover
    cv2 = None
    print("[warn] OpenCV not available:", e)

try:
    import torch
    import torch.nn as nn
except Exception as e:                        # pragma: no cover
    torch = None
    nn = object
    print("[warn] PyTorch not available:", e)

try:
    import mediapipe as mp
except Exception as e:                         # pragma: no cover
    mp = None
    print("[warn] MediaPipe not available:", e)

try:
    import serial as pyserial
except Exception as e:                          # pragma: no cover
    pyserial = None
    print("[warn] pyserial not available:", e)

try:
    import pigpio
except Exception:                                # pragma: no cover
    pigpio = None

import http.server
import socketserver
import urllib.parse

if torch is not None:
    torch.set_num_threads(4)

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
BASE       = "/home/hudai/Desktop/thesis"
CKPT       = os.path.join(BASE, "checkpoints")
ASSETS     = os.path.join(BASE, "assets")
WORD_XLSX  = os.path.join(BASE, "Word Label.xlsx")
AUDIO_DIR  = os.path.join(BASE, "audio")
DEVICE     = "cpu"

# one or more checkpoints; if several exist they are ensembled (mean softmax)
MODEL_CKPTS = ["word_model_full.pt"]

STM_PORT, STM_BAUD = "/dev/ttyACM0", 115200
PHONE_PORT = 8000
CAM_INDEX  = 0

POSE_N, HAND_N = 33, 21
N_LM        = POSE_N + 2 * HAND_N          # 75
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM * 3 + N_LM              # 300
N_FRAMES    = 64
N_CLASSES_DEFAULT = 102

# -- model width (production 128 config) --
D_MODEL, N_HEADS, N_LAYERS, DROPOUT = 128, 4, 4, 0.3

# -- recognition --
CONF_TH       = 0.55
VOTE_NEED     = 4
RECOG_WINDOW  = 12.0
NAV_COOLDOWN  = 3.0
HAND_RECENT   = 0.5
DOUBLE_WINDOW = 2.0
LEFT_DOUBLE_WINDOW = 1.0
PREDICT_EVERY = 0.25      # seconds between model evaluations while capturing

# -- following --
USE_SONAR      = True
STOP_DIST      = 35       # cm sonar safety stop
FOLLOW_DIST    = 70       # cm beyond this -> forward
CENTER_DEAD    = 0.10     # |offset| below this = centred
FOLLOW_NEAR_W  = 0.40     # shoulder-width frac: close -> stop
FOLLOW_BACK_W  = 0.55     # shoulder-width frac: very close -> back up
FOLLOW_FAR_W   = 0.22     # shoulder-width frac: far -> forward
FOLLOW_SPIN_TH = 0.28     # offset above this -> spin; below -> arc
STEER_DIR      = 1        # set -1 if it steers the wrong way
FOLLOW_LOST_T  = 2.5      # s before search-rotation when person lost

# -- arm auto-height in following mode (both arms symmetric) --
ARM_FOLLOW_ENABLE   = True
ARM_FOLLOW_TARGET_Y = 0.40
ARM_FOLLOW_GAIN     = 55
ARM_FOLLOW_RATE     = 0.35

# -- manual --
AUTO_STOP_AFTER = 0.6     # stop if no drive cmd in this many seconds
KEEPALIVE_MS    = 0.20    # resend active drive command this often (< STM32 700ms)
ARM_STEP_DEG    = 10

# -- head tracking (BCM pins; head_tracking_final.py used pan=18 tilt=13) --
HEAD_TRACK_ENABLE = True
HT_PAN_PIN, HT_TILT_PIN = 18, 13
HT_DEAD, HT_KP, HT_KD, HT_SMOOTH = 0.05, 11.0, 2.0, 0.6
HT_MAX_STEP   = 6.0       # deg per update cap
HT_MIN_MOVE   = 0.4       # deg below which we don't bother moving
HT_SETTLE     = 1.0       # s of no face -> recentre then detach
HT_PAN_MIN, HT_PAN_MAX   = 30, 150
HT_TILT_MIN, HT_TILT_MAX = 50, 140
HT_PAN_CENTER, HT_TILT_CENTER = 90, 95

# ════════════════════════════════════════════════════════════
#  BENGALI -> BANGLISH TRANSLITERATION (OLED can't draw Bengali)
# ════════════════════════════════════════════════════════════
_BN_VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
_BN_VOWEL_SIGNS = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}
_BN_CONS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y",
    "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": "",
}
_BN_DIGITS = {"০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
              "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"}
_HOSHONTO = "্"

def to_banglish(text):
    """Approximate Bengali->Latin transliteration for the OLED.
    Consonants carry an inherent 'o' unless followed by a vowel sign or hoshonto."""
    if not text:
        return ""
    out = []
    chars = list(text)
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if c in _BN_CONS:
            out.append(_BN_CONS[c])
            nxt = chars[i + 1] if i + 1 < n else ""
            if nxt == _HOSHONTO:
                i += 2                      # conjunct: drop inherent vowel
                continue
            if nxt in _BN_VOWEL_SIGNS:
                out.append(_BN_VOWEL_SIGNS[nxt])
                i += 2
                continue
            out.append("o")                 # inherent vowel
            i += 1
            continue
        if c in _BN_VOWELS:
            out.append(_BN_VOWELS[c]); i += 1; continue
        if c in _BN_VOWEL_SIGNS:
            out.append(_BN_VOWEL_SIGNS[c]); i += 1; continue
        if c in _BN_DIGITS:
            out.append(_BN_DIGITS[c]); i += 1; continue
        if c == " ":
            out.append(" "); i += 1; continue
        if c in (_HOSHONTO,):
            i += 1; continue
        # any ascii / punctuation passes through
        if ord(c) < 128:
            out.append(c)
        i += 1
    return "".join(out)

# ════════════════════════════════════════════════════════════
#  MODEL  (SignTransformer, encoder + temporal transformer)
# ════════════════════════════════════════════════════════════
if torch is not None:
    class SpatialEncoder(nn.Module):
        def __init__(self, in_dim, d, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p),
                nn.Linear(d, d),      nn.LayerNorm(d), nn.GELU())
        def forward(self, x): return self.net(x)

    class AttnPool(nn.Module):
        def __init__(self, d):
            super().__init__(); self.q = nn.Linear(d, 1)
        def forward(self, x):
            w = torch.softmax(self.q(x).squeeze(-1), 1).unsqueeze(-1)
            return (x * w).sum(1)

    class SignTransformer(nn.Module):
        def __init__(self, in_dim, n_classes, d, heads, layers, ff=4, p=0.3):
            super().__init__()
            self.encoder = SpatialEncoder(in_dim, d, p)
            self.pos = nn.Parameter(torch.zeros(1, 512, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, d * ff, p,
                                               batch_first=True, activation="gelu")
            self.tf = nn.TransformerEncoder(layer, layers)
            self.pool = AttnPool(d)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n_classes))
        def forward(self, x):
            t = self.encoder(x)
            t = t + self.pos[:, :t.shape[1]]
            return self.head(self.pool(self.tf(t)))


def load_labels(n_classes_fallback=N_CLASSES_DEFAULT):
    """Return {id: bengali_word}. Header-agnostic: finds the Bengali column."""
    labels = {i: f"id{i}" for i in range(n_classes_fallback)}
    try:
        import pandas as pd
        df = pd.read_excel(WORD_XLSX, header=None)
        bn_re = re.compile(r"[\u0980-\u09FF]")
        # find the column with the most Bengali text
        best_col, best_hits = None, -1
        for col in df.columns:
            hits = df[col].astype(str).apply(lambda s: bool(bn_re.search(s))).sum()
            if hits > best_hits:
                best_hits, best_col = hits, col
        if best_col is not None and best_hits > 0:
            words = [str(w).strip() for w in df[best_col].tolist()
                     if bn_re.search(str(w))]
            labels = {i: w for i, w in enumerate(words)}
    except Exception as e:
        print("[warn] could not read label file, using id<n>:", e)
    return labels


def load_models():
    """Load one or more checkpoints into eval-mode models for ensembling."""
    models = []
    if torch is None:
        return models
    for name in MODEL_CKPTS:
        path = os.path.join(CKPT, name)
        if not os.path.exists(path):
            print("[warn] checkpoint missing:", path); continue
        m = SignTransformer(FEATURE_DIM, N_CLASSES_DEFAULT, D_MODEL,
                            N_HEADS, N_LAYERS, p=DROPOUT).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state, strict=False)
        m.eval()
        models.append(m)
        print("[ok] loaded model:", name)
    return models

# ════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (matches the training pipeline exactly)
# ════════════════════════════════════════════════════════════
def holistic_to_landmarks(res):
    """MediaPipe Holistic result -> (coords(75,3), mask(75)).
    Order: 33 pose, 21 left hand, 21 right hand."""
    coords = np.zeros((N_LM, 3), dtype="float32")
    mask = np.zeros((N_LM,), dtype="float32")
    if res is None:
        return coords, mask
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark[:POSE_N]):
            coords[i] = (lm.x, lm.y, lm.z)
            mask[i] = 1.0 if getattr(lm, "visibility", 1.0) > 0.3 else 1.0
    if res.left_hand_landmarks:
        base = POSE_N
        for i, lm in enumerate(res.left_hand_landmarks.landmark[:HAND_N]):
            coords[base + i] = (lm.x, lm.y, lm.z); mask[base + i] = 1.0
    if res.right_hand_landmarks:
        base = POSE_N + HAND_N
        for i, lm in enumerate(res.right_hand_landmarks.landmark[:HAND_N]):
            coords[base + i] = (lm.x, lm.y, lm.z); mask[base + i] = 1.0
    return coords, mask


def normalize_feature(coords, mask):
    """Shoulder-centred, shoulder-scaled -> 300-dim (225 coords + 75 mask)."""
    c = coords.copy()
    ls, rs = c[L_SHO], c[R_SHO]
    center = (ls + rs) / 2.0
    scale = float(np.linalg.norm(ls[:2] - rs[:2]))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    c = (c - center) / scale
    c[mask == 0] = 0.0
    return np.concatenate([c.reshape(-1), mask]).astype("float32")

# ════════════════════════════════════════════════════════════
#  OPEN-HAND DETECTION  (for the navigation gestures)
# ════════════════════════════════════════════════════════════
def hand_is_open(hand_landmarks):
    """Heuristic: an open palm has the four fingertips far from the wrist
    relative to the knuckles. Returns True/False."""
    if hand_landmarks is None:
        return False
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y])
    tips = [8, 12, 16, 20]      # index, middle, ring, pinky tips
    pips = [6, 10, 14, 18]      # corresponding mid-joints
    extended = 0
    for t, p in zip(tips, pips):
        tip = np.array([lm[t].x, lm[t].y])
        pip = np.array([lm[p].x, lm[p].y])
        if np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist):
            extended += 1
    return extended >= 3

# ════════════════════════════════════════════════════════════
#  SERIAL LINK TO STM32  (write-locked + keepalive + telemetry)
# ════════════════════════════════════════════════════════════
class SerialLink:
    def __init__(self, port=STM_PORT, baud=STM_BAUD):
        self.lock = threading.Lock()
        self.ser = None
        self.distance = -1.0
        self.current_motor = "S"
        self.last_input = 0.0
        self._run = True
        try:
            if pyserial is not None:
                self.ser = pyserial.Serial(port, baud, timeout=0.1)
                time.sleep(2.0)               # let the board reset
                print("[ok] serial open:", port)
        except Exception as e:
            print("[warn] serial open failed:", e)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _send(self, line):
        with self.lock:
            if self.ser is None:
                return
            try:
                self.ser.write((line + "\n").encode())
            except Exception as e:
                print("[warn] serial write:", e)

    def _reader(self):
        buf = ""
        while self._run:
            if self.ser is None:
                time.sleep(0.2); continue
            try:
                data = self.ser.read(64).decode(errors="ignore")
            except Exception:
                time.sleep(0.1); continue
            if not data:
                continue
            buf += data
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line.startswith("D"):
                    try:
                        self.distance = float(line[1:])
                    except ValueError:
                        pass

    def _keepalive(self):
        """Resend the active drive command faster than the STM32 failsafe,
        and auto-stop if the user hasn't pressed anything recently."""
        while self._run:
            time.sleep(KEEPALIVE_MS)
            if self.current_motor == "S":
                continue
            if time.time() - self.last_input > AUTO_STOP_AFTER:
                self.stop()
            else:
                self._send(self.current_motor)

    # --- public API ---
    def drive(self, c):
        self.current_motor = c
        self.last_input = time.time()
        self._send(c)

    def stop(self):
        self.current_motor = "S"
        self._send("S")

    def arm_nudge(self, side, up):
        self._send(f"A{side}{'U' if up else 'D'}")

    def arm_set(self, side, deg):
        deg = max(0, min(90, int(deg)))
        self._send(f"A{side}{deg:03d}")

    def oled_top(self, text):
        self._send("#" + text[:80])

    def oled_bottom(self, text):
        self._send("$" + text[:80])

    def close(self):
        self._run = False
        self.stop()

# ════════════════════════════════════════════════════════════
#  HEAD TRACKER  (Pi GPIO servos via pigpio, PD control)
# ════════════════════════════════════════════════════════════
class HeadTracker:
    def __init__(self):
        self.pi = None
        self.pan = HT_PAN_CENTER
        self.tilt = HT_TILT_CENTER
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.last_seen = 0.0
        self.attached = False
        if HEAD_TRACK_ENABLE and pigpio is not None:
            try:
                self.pi = pigpio.pi()
                if not self.pi.connected:
                    self.pi = None
                    print("[warn] pigpio daemon not running; head tracking off")
            except Exception as e:
                self.pi = None
                print("[warn] pigpio init failed:", e)
        if self.pi is not None:
            self._write(self.pan, self.tilt)

    def _us(self, ang):
        return int(500 + (ang / 180.0) * 2000)

    def _write(self, pan, tilt):
        if self.pi is None:
            return
        self.pi.set_servo_pulsewidth(HT_PAN_PIN, self._us(pan))
        self.pi.set_servo_pulsewidth(HT_TILT_PIN, self._us(tilt))
        self.attached = True

    def _detach(self):
        if self.pi is None or not self.attached:
            return
        self.pi.set_servo_pulsewidth(HT_PAN_PIN, 0)
        self.pi.set_servo_pulsewidth(HT_TILT_PIN, 0)
        self.attached = False

    def update(self, face_x, face_y):
        """face_x, face_y in [0,1] frame coords, or None when no face."""
        now = time.time()
        if face_x is None:
            if now - self.last_seen > HT_SETTLE:
                # recentre once, then detach to kill wobble
                if abs(self.pan - HT_PAN_CENTER) > 1 or abs(self.tilt - HT_TILT_CENTER) > 1:
                    self.pan, self.tilt = HT_PAN_CENTER, HT_TILT_CENTER
                    self._write(self.pan, self.tilt)
                    time.sleep(0.05)
                self._detach()
            return
        self.last_seen = now
        ex = face_x - 0.5
        ey = face_y - 0.5
        if abs(ex) < HT_DEAD:
            ex = 0.0
        if abs(ey) < HT_DEAD:
            ey = 0.0
        # PD on each axis
        dpan  = HT_KP * ex + HT_KD * (ex - self.prev_ex)
        dtilt = HT_KP * ey + HT_KD * (ey - self.prev_ey)
        self.prev_ex, self.prev_ey = ex, ey
        dpan  = max(-HT_MAX_STEP, min(HT_MAX_STEP, dpan))
        dtilt = max(-HT_MAX_STEP, min(HT_MAX_STEP, dtilt))
        if abs(dpan) < HT_MIN_MOVE and abs(dtilt) < HT_MIN_MOVE:
            return
        # camera offset to the right means pan needs to decrease (mirror)
        self.pan  = max(HT_PAN_MIN,  min(HT_PAN_MAX,  self.pan  - dpan))
        self.tilt = max(HT_TILT_MIN, min(HT_TILT_MAX, self.tilt + dtilt))
        self._write(self.pan, self.tilt)

    def close(self):
        self._detach()
        if self.pi is not None:
            self.pi.stop()

# ════════════════════════════════════════════════════════════
#  RECOGNIZER  (rolling feature buffer + ensemble prediction)
# ════════════════════════════════════════════════════════════
class Recognizer:
    def __init__(self, models, labels):
        self.models = models
        self.labels = labels
        self.buf = deque(maxlen=N_FRAMES)
        self.last_predict = 0.0

    def push(self, feature):
        self.buf.append(feature)

    def ready(self):
        return len(self.buf) >= N_FRAMES // 2

    def predict(self):
        """Return (class_id, confidence) or (None, 0.0)."""
        if not self.models or not self.ready():
            return None, 0.0
        now = time.time()
        if now - self.last_predict < PREDICT_EVERY:
            return None, 0.0
        self.last_predict = now
        frames = list(self.buf)
        # pad/truncate to N_FRAMES
        if len(frames) < N_FRAMES:
            frames = frames + [frames[-1]] * (N_FRAMES - len(frames))
        else:
            frames = frames[-N_FRAMES:]
        x = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(DEVICE)  # (1,64,300)
        with torch.no_grad():
            probs = None
            for m in self.models:
                p = torch.softmax(m(x), 1)
                probs = p if probs is None else probs + p
            probs = (probs / len(self.models)).cpu().numpy()[0]
        cid = int(probs.argmax())
        return cid, float(probs[cid])

# ════════════════════════════════════════════════════════════
#  NAVIGATION STATE MACHINE  (open-hand sentence builder)
#    right open hand  = start capturing
#    left  open hand  = delete last word
#    both  open hands = end; finalise on hands-down
# ════════════════════════════════════════════════════════════
class NavStateMachine:
    def __init__(self, serial, recognizer, labels):
        self.s = serial
        self.rec = recognizer
        self.labels = labels
        self.state = "IDLE"          # IDLE / CAPTURING / ENDING
        self.sentence = []
        self.votes = Counter()
        self.capture_start = 0.0
        self.last_commit = 0.0
        self.last_left = 0.0
        self.both_since = 0.0

    def _say(self, cid):
        play_word(cid)
        word = self.labels.get(cid, f"id{cid}")
        self.sentence.append(word)
        self._render()

    def _render(self):
        joined = " ".join(self.sentence) if self.sentence else ""
        self.s.oled_bottom(to_banglish(joined))

    def reset_capture(self):
        self.votes.clear()
        self.capture_start = time.time()

    def update(self, right_open, left_open):
        now = time.time()
        both = right_open and left_open

        if self.state == "IDLE":
            self.s.oled_top("READY|right hand = start")
            if both:
                return
            if right_open:
                self.state = "CAPTURING"
                self.reset_capture()
                self.s.oled_top("CAPTURING|left=del both=end")
            return

        if self.state == "CAPTURING":
            if both:
                self.both_since = self.both_since or now
                if now - self.both_since > 0.4:     # hold both -> end
                    self.state = "ENDING"
                return
            else:
                self.both_since = 0.0

            if left_open and now - self.last_left > LEFT_DOUBLE_WINDOW:
                self.last_left = now
                if self.sentence:
                    self.sentence.pop()
                    self._render()
                    self.s.oled_top("DELETED|last word")
                return

            # accumulate model votes
            if now - self.last_commit > NAV_COOLDOWN:
                cid, conf = self.rec.predict()
                if cid is not None and conf >= CONF_TH:
                    self.votes[cid] += 1
                    top, cnt = self.votes.most_common(1)[0]
                    if cnt >= VOTE_NEED:
                        self._say(top)
                        self.last_commit = now
                        self.reset_capture()
            # window timeout
            if now - self.capture_start > RECOG_WINDOW:
                self.reset_capture()
            return

        if self.state == "ENDING":
            # finalise when both hands go down
            if not right_open and not left_open:
                final = " ".join(self.sentence) if self.sentence else "(empty)"
                self.s.oled_top("DONE|" + to_banglish(final)[:60])
                self.state = "IDLE"
                self.sentence = []
            return

# ════════════════════════════════════════════════════════════
#  FOLLOW CONTROLLER  (proportional, arc/spin, sonar, arm height)
# ════════════════════════════════════════════════════════════
class FollowController:
    def __init__(self, serial):
        self.s = serial
        self.last_seen = 0.0
        self.arm_angle = 45
        self.last_arm = 0.0

    def update(self, offset, width_frac, face_y, distance):
        """offset: signed horizontal offset of person centre in [-0.5,0.5]
           width_frac: shoulder width / frame width (distance proxy)
           face_y: vertical face pos [0,1] (for arm auto-height)
           distance: sonar cm (or -1)"""
        now = time.time()

        if offset is None:
            # person lost -> brief search rotation, then stop
            if now - self.last_seen > FOLLOW_LOST_T:
                self.s.drive("R")       # slow search spin
            else:
                self.s.stop()
            return
        self.last_seen = now

        # sonar safety stop overrides everything
        if USE_SONAR and 0 < distance < STOP_DIST:
            self.s.stop()
            self._arm_track(face_y, now)
            return

        off = offset * STEER_DIR
        # turning first
        if abs(off) > FOLLOW_SPIN_TH:
            self.s.drive("R" if off > 0 else "L")
        elif abs(off) > CENTER_DEAD:
            # gentle arc forward while correcting
            self.s.drive("H" if off > 0 else "G")
        else:
            # centred: decide forward/stop/back by apparent size
            if width_frac > FOLLOW_BACK_W:
                self.s.drive("B")
            elif width_frac > FOLLOW_NEAR_W:
                self.s.stop()
            elif width_frac < FOLLOW_FAR_W:
                self.s.drive("F")
            else:
                self.s.stop()

        self._arm_track(face_y, now)

    def _arm_track(self, face_y, now):
        if not ARM_FOLLOW_ENABLE or face_y is None:
            return
        if now - self.last_arm < ARM_FOLLOW_RATE:
            return
        self.last_arm = now
        err = face_y - ARM_FOLLOW_TARGET_Y       # +ve: face too low -> raise
        target = int(self.arm_angle + ARM_FOLLOW_GAIN * err)
        target = max(0, min(90, target))
        if target != self.arm_angle:
            self.arm_angle = target
            self.s.arm_set("L", target)
            self.s.arm_set("R", target)

# ════════════════════════════════════════════════════════════
#  AUDIO  (offline pre-generated wav per class id)
# ════════════════════════════════════════════════════════════
def play_word(cid):
    path = os.path.join(AUDIO_DIR, f"{cid}.wav")
    if not os.path.exists(path):
        print("[warn] audio missing:", path); return
    def _play():
        for player in ("paplay", "aplay"):
            try:
                subprocess.run([player, path],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=8)
                return
            except FileNotFoundError:
                continue
            except Exception as e:
                print("[warn] audio play:", e); return
    threading.Thread(target=_play, daemon=True).start()

# ════════════════════════════════════════════════════════════
#  SHARED APP STATE  (for the HTTP server)
# ════════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.frame_jpeg = b""
        self.frame_lock = threading.Lock()
        self.mode = "idle"           # idle / manual / follow
        self.serial = None
        self.distance = -1.0
        self.sentence = ""
        self.running = True

APP = AppState()

# ════════════════════════════════════════════════════════════
#  PHONE CONTROL PAGE  (glassmorphism / monospace console)
# ════════════════════════════════════════════════════════════
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Ishara Control</title>
<style>
 :root{--bg:#0a1628;--card:rgba(255,255,255,.08);--line:rgba(255,255,255,.18);
       --teal:#14b8a6;--mint:#5eead4;--txt:#e2ecf5;--mut:#7da0bd;}
 *{box-sizing:border-box;font-family:'JetBrains Mono',ui-monospace,monospace;}
 body{margin:0;background:linear-gradient(160deg,#0a1628,#0f2942);color:var(--txt);
      -webkit-tap-highlight-color:transparent;user-select:none;}
 .wrap{max-width:460px;margin:0 auto;padding:14px;}
 h1{font-size:18px;letter-spacing:3px;color:var(--mint);margin:6px 0 12px;}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
       backdrop-filter:blur(8px);padding:12px;margin-bottom:12px;}
 img#cam{width:100%;border-radius:10px;display:block;background:#000;}
 .row{display:flex;gap:8px;margin-bottom:8px;}
 .row>*{flex:1;}
 button{background:rgba(20,184,166,.16);border:1px solid var(--teal);color:var(--txt);
        border-radius:10px;padding:14px 0;font-size:15px;font-weight:600;}
 button:active{background:var(--teal);color:#04211d;}
 .pad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
 .pad button{padding:18px 0;font-size:18px;}
 .stop{background:rgba(220,38,38,.25);border-color:#dc2626;}
 .mode{font-size:13px;}
 .mode.active{background:var(--teal);color:#04211d;}
 .lbl{color:var(--mut);font-size:12px;letter-spacing:1px;margin:2px 0 6px;}
 #sent{min-height:20px;color:var(--mint);font-size:14px;word-break:break-word;}
 #dist{color:var(--mut);font-size:12px;}
</style></head><body><div class="wrap">
 <h1>ISHARA &middot; CONTROL</h1>
 <div class="card"><img id="cam" src="/stream"></div>
 <div class="card">
   <div class="lbl">MODE</div>
   <div class="row">
     <button class="mode" id="m_idle"   onclick="setMode('idle')">IDLE</button>
     <button class="mode" id="m_manual" onclick="setMode('manual')">MANUAL</button>
     <button class="mode" id="m_follow" onclick="setMode('follow')">FOLLOW</button>
   </div>
   <div class="lbl">RECOGNISED</div><div id="sent">&mdash;</div>
   <div id="dist">distance: &mdash;</div>
 </div>
 <div class="card">
   <div class="lbl">DRIVE (hold)</div>
   <div class="pad">
     <button onmousedown="hold('G')" onmouseup="rel()" ontouchstart="hold('G')" ontouchend="rel()">&#8598;</button>
     <button onmousedown="hold('F')" onmouseup="rel()" ontouchstart="hold('F')" ontouchend="rel()">&#8593;</button>
     <button onmousedown="hold('H')" onmouseup="rel()" ontouchstart="hold('H')" ontouchend="rel()">&#8599;</button>
     <button onmousedown="hold('L')" onmouseup="rel()" ontouchstart="hold('L')" ontouchend="rel()">&#8634;</button>
     <button class="stop" onclick="cmd('S')">STOP</button>
     <button onmousedown="hold('R')" onmouseup="rel()" ontouchstart="hold('R')" ontouchend="rel()">&#8635;</button>
     <button onmousedown="hold('J')" onmouseup="rel()" ontouchstart="hold('J')" ontouchend="rel()">&#8601;</button>
     <button onmousedown="hold('B')" onmouseup="rel()" ontouchstart="hold('B')" ontouchend="rel()">&#8595;</button>
     <button onmousedown="hold('K')" onmouseup="rel()" ontouchstart="hold('K')" ontouchend="rel()">&#8600;</button>
   </div>
 </div>
 <div class="card">
   <div class="lbl">ARM</div>
   <div class="row">
     <button onclick="arm('L','U')">L &#9650;</button>
     <button onclick="arm('L','D')">L &#9660;</button>
     <button onclick="arm('R','U')">R &#9650;</button>
     <button onclick="arm('R','D')">R &#9660;</button>
   </div>
 </div>
</div>
<script>
 let kp=null;
 function cmd(c){fetch('/cmd?c='+c);}
 function hold(c){cmd(c); clearInterval(kp); kp=setInterval(()=>cmd(c),200);}
 function rel(){clearInterval(kp); kp=null; cmd('S');}
 function arm(s,d){fetch('/arm?s='+s+'&a='+d);}
 function setMode(m){fetch('/mode?m='+m).then(()=>{
   ['idle','manual','follow'].forEach(x=>
     document.getElementById('m_'+x).classList.toggle('active',x===m));});}
 setInterval(()=>fetch('/state').then(r=>r.json()).then(j=>{
   document.getElementById('sent').textContent=j.sentence||'\\u2014';
   document.getElementById('dist').textContent='distance: '+
     (j.distance>0?j.distance.toFixed(1)+' cm':'\\u2014');
 }),700);
 setMode('idle');
</script></body></html>"""

# ════════════════════════════════════════════════════════════
#  HTTP SERVER
# ════════════════════════════════════════════════════════════
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):       # quiet
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        path = u.path

        if path == "/" or path == "/index.html":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while APP.running:
                    with APP.frame_lock:
                        jpg = APP.frame_jpeg
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if path == "/cmd":
            c = q.get("c", ["S"])[0]
            if APP.serial:
                if c == "S":
                    APP.serial.stop()
                elif c in "FBLRGHJK":
                    if APP.mode == "manual":
                        APP.serial.drive(c)
            self._send(200, "text/plain", b"ok")
            return

        if path == "/arm":
            s = q.get("s", ["L"])[0]
            a = q.get("a", [""])[0]
            deg = q.get("deg", [""])[0]
            if APP.serial:
                if deg:
                    APP.serial.arm_set(s, int(deg))
                elif a in ("U", "D"):
                    APP.serial.arm_nudge(s, a == "U")
            self._send(200, "text/plain", b"ok")
            return

        if path == "/mode":
            m = q.get("m", ["idle"])[0]
            if m in ("idle", "manual", "follow"):
                APP.mode = m
                if APP.serial:
                    APP.serial.stop()
            self._send(200, "text/plain", b"ok")
            return

        if path == "/state":
            body = json.dumps({"mode": APP.mode,
                               "distance": APP.distance,
                               "sentence": APP.sentence}).encode()
            self._send(200, "application/json", body)
            return

        self._send(404, "text/plain", b"not found")


class ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_http():
    srv = ThreadedHTTP(("0.0.0.0", PHONE_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[ok] phone control at http://0.0.0.0:{PHONE_PORT}")
    return srv

# ════════════════════════════════════════════════════════════
#  MAIN PERCEPTION LOOP
# ════════════════════════════════════════════════════════════
def person_metrics(res, w, h):
    """From a holistic result, derive (offset, width_frac, face_x, face_y)
    or (None, None, None, None) if no usable person."""
    if res is None or not res.pose_landmarks:
        return None, None, None, None
    lm = res.pose_landmarks.landmark
    ls, rs = lm[L_SHO], lm[R_SHO]
    cx = (ls.x + rs.x) / 2.0
    width_frac = abs(ls.x - rs.x)
    offset = cx - 0.5
    nose = lm[0]
    return offset, width_frac, nose.x, nose.y


def main():
    if cv2 is None or mp is None:
        print("[fatal] OpenCV and MediaPipe are required to run."); return

    labels = load_labels()
    models = load_models()
    serial = SerialLink()
    APP.serial = serial
    head = HeadTracker()
    rec = Recognizer(models, labels)
    nav = NavStateMachine(serial, rec, labels)
    follow = FollowController(serial)
    srv = start_http()

    serial.oled_top("ISHARA|booting...")

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    serial.oled_top("READY|right hand = start")
    print("[ok] running. Ctrl-C to quit.")

    try:
        while APP.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05); continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            # --- shared feature extraction ---
            coords, mask = holistic_to_landmarks(res)
            feat = normalize_feature(coords, mask)
            rec.push(feat)

            # --- person metrics for head-tracking + follow ---
            offset, width_frac, face_x, face_y = person_metrics(res, w, h)
            head.update(face_x, face_y)

            # --- mode behaviour ---
            APP.distance = serial.distance
            if APP.mode == "follow":
                follow.update(offset, width_frac, face_y, serial.distance)
            elif APP.mode == "idle":
                r_open = hand_is_open(res.right_hand_landmarks)
                l_open = hand_is_open(res.left_hand_landmarks)
                nav.update(r_open, l_open)
            # 'manual' mode: drive comes only from the phone, nothing to compute

            APP.sentence = to_banglish(" ".join(nav.sentence)) if nav.sentence else ""

            # --- annotate + publish frame for the phone ---
            if res.pose_landmarks:
                draw.draw_landmarks(frame, res.pose_landmarks,
                                    mp.solutions.holistic.POSE_CONNECTIONS,
                                    landmark_drawing_spec=styles
                                    .get_default_pose_landmarks_style())
            for hand_lm in (res.left_hand_landmarks, res.right_hand_landmarks):
                if hand_lm:
                    draw.draw_landmarks(frame, hand_lm,
                                        mp.solutions.holistic.HAND_CONNECTIONS)
            cv2.putText(frame, f"mode:{APP.mode}  d:{serial.distance:.0f}cm",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (94, 234, 212), 2)
            ok2, jpg = cv2.imencode(".jpg", frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok2:
                with APP.frame_lock:
                    APP.frame_jpeg = jpg.tobytes()

    except KeyboardInterrupt:
        print("\n[exit] stopping...")
    finally:
        APP.running = False
        serial.close()
        head.close()
        try:
            cap.release()
        except Exception:
            pass
        try:
            srv.shutdown()
        except Exception:
            pass
        print("[done]")


if __name__ == "__main__":
    main()
