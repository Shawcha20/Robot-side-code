#!/usr/bin/env python3
"""
PHASE 4 - Ishara robot (Raspberry Pi 5)
Bengali Sign Language recognition + real-time translation on a rolling robot.

CONTROL
  Hand gestures (idle):    RIGHT x1 = recognition   RIGHT x2 = following
  In recognition:          RIGHT = capture sign | LEFT x1 = delete | LEFT x2 = idle | BOTH = speak
  In following:            LEFT  = back to idle
  Phone  (http://10.42.0.1:8000):
      - mode buttons (idle / recognition / following / manual)
      - TWO joysticks: left = forward/back, right = left/right; mixing = diagonals
      - press & hold to drive, release = stop
      - arm controls, SIGN / DELETE / SPEAK recognition buttons, live video
  Keyboard (click OpenCV window): i/r/f/m  q=quit  W/A/S/D/G/H/J/K  X/space=stop

STM32 serial (newline-framed):
  Pi -> STM32: F/B/L/R/S  G/H/J/K  ALU/ALD/ARU/ARD  AL045/AR090  #top  $bottom
  STM32 -> Pi: D<distance_cm>

Flash stm32_phase4.ino first.

HEAD-TRACKING JITTER NOTE (Pi 5):
  pigpio hardware PWM is NOT available on the Pi 5, so gpiozero falls back to
  software PWM, which buzzes when a servo holds a live signal. The fix used here:
    - detach() immediately after every move (servo holds quietly by its gearing)
    - MIN_MOVE / DEAD_ZONE so tiny corrections don't pulse the servo
    - on exit (and on no-face), detach so the servos go SILENT when the code stops
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["PYTHONUTF8"] = "1"

import re, time, threading, subprocess, json, atexit
import http.server, socketserver, urllib.parse
from collections import deque, Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
import serial as pyserial

torch.set_num_threads(4)

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
BASE      = "/home/hudai/Desktop/thesis"
CKPT      = os.path.join(BASE, "checkpoints")
ASSETS    = os.path.join(BASE, "assets")
AUDIO_DIR = os.path.join(BASE, "audio")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# the 5-seed ensemble (exact filenames from the working backend loader)
ENSEMBLE_FILES = ["word_model_full.pt", "word_model_128_s2.pt", "word_model_128_s3.pt",
                  "word_model_128_s4.pt", "word_model_128_s5.pt"]

STM_PORT, STM_BAUD = "/dev/ttyACM0", 115200
PHONE_PORT = 8000

POSE_TASK = os.path.join(ASSETS, "pose_landmarker_lite.task")
HAND_TASK = os.path.join(ASSETS, "hand_landmarker.task")

POSE_N, HAND_N = 33, 21
N_LM        = POSE_N + 2 * HAND_N      # 75
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM * 3 + N_LM          # 300
N_FRAMES    = 64

CONF_TH      = 0.55
VOTE_NEED    = 4
RECOG_WINDOW = 10.0
NAV_COOLDOWN = 1.5
HAND_RECENT  = 0.5
DOUBLE_WINDOW      = 2.0
LEFT_DOUBLE_WINDOW = 1.0

# following
USE_SONAR      = True
STOP_DIST      = 40
FOLLOW_DIST    = 70
CENTER_DEAD    = 0.12
FOLLOW_NEAR_W  = 0.40
FOLLOW_BACK_W  = 0.55
FOLLOW_FAR_W   = 0.22
FOLLOW_SPIN_TH = 0.28
STEER_DIR      = 1
FOLLOW_LOST_T  = 2.5

SPEED          = 150
AUTO_STOP_AFTER = 0.6
ARM_STEP_DEG   = 10

# ── HEAD TRACKING (gpiozero AngularServo, BCM 18=pan 13=tilt) ──
HEAD_TRACK_ENABLE = True
HT_PAN_PIN, HT_TILT_PIN =22,23
HT_KP, HT_KD  = 11.0, 2.0
HT_DEAD_ZONE  = 0.06
HT_SMOOTH     = 0.7
HT_MAX_STEP   = 5.0
HT_MIN_MOVE   = 1.5        # ignore tiny corrections -> kills most idle twitch
HT_DETACH_AFTER = 0.15     # detach quickly after each move -> mostly silent
HT_CENTER     = 135.0
HT_PAN_MIN, HT_PAN_MAX   =  30, 240
HT_TILT_MIN, HT_TILT_MAX =  60, 210
HT_PAN_DIR,  HT_TILT_DIR = -1,   1

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ════════════════════════════════════════════════════════════
#  LABELS
# ════════════════════════════════════════════════════════════
ID2NAME = {}
def load_labels():
    global ID2NAME
    try:
        import openpyxl
        ws = openpyxl.load_workbook(WORD_XLSX).active
        ID2NAME = {int(r[0]): str(r[1]).strip()
                   for r in ws.iter_rows(min_row=2, values_only=True)
                   if r[0] is not None and r[1] is not None}
    except Exception as e:
        print("(names not loaded)", e)
load_labels()
def name(cid): return ID2NAME.get(int(cid), f"class_{cid}")

# ════════════════════════════════════════════════════════════
#  BANGLISH TRANSLITERATION
# ════════════════════════════════════════════════════════════
_BN_V  = {"অ":"o","আ":"a","ই":"i","ঈ":"i","উ":"u","ঊ":"u","ঋ":"ri",
           "এ":"e","ঐ":"oi","ও":"o","ঔ":"ou"}
_BN_VS = {"া":"a","ি":"i","ী":"i","ু":"u","ূ":"u","ৃ":"ri",
           "ে":"e","ৈ":"oi","ো":"o","ৌ":"ou"}
_BN_C  = {"ক":"k","খ":"kh","গ":"g","ঘ":"gh","ঙ":"ng","চ":"ch","ছ":"chh",
           "জ":"j","ঝ":"jh","ঞ":"n","ট":"t","ঠ":"th","ড":"d","ঢ":"dh","ণ":"n",
           "ত":"t","থ":"th","দ":"d","ধ":"dh","ন":"n","প":"p","ফ":"ph","ব":"b",
           "ভ":"bh","ম":"m","য":"j","র":"r","ল":"l","শ":"sh","ষ":"sh","স":"s",
           "হ":"h","ড়":"r","ঢ়":"rh","য়":"y","ৎ":"t","ং":"ng","ঃ":"h","ঁ":""}
_BN_D  = {"০":"0","১":"1","২":"2","৩":"3","৪":"4","৫":"5","৬":"6","৭":"7","৮":"8","৯":"9"}
_HOSH  = "্"

def to_banglish(text):
    if not text: return ""
    out, i, chars = [], 0, list(text); n = len(chars)
    while i < n:
        c = chars[i]
        if c in _BN_C:
            out.append(_BN_C[c])
            nxt = chars[i+1] if i+1 < n else ""
            if nxt == _HOSH:  i += 2; continue
            if nxt in _BN_VS: out.append(_BN_VS[nxt]); i += 2; continue
            out.append("o");  i += 1; continue
        if c in _BN_V:  out.append(_BN_V[c]);  i += 1; continue
        if c in _BN_VS: out.append(_BN_VS[c]); i += 1; continue
        if c in _BN_D:  out.append(_BN_D[c]);  i += 1; continue
        if c == " ":    out.append(" ");        i += 1; continue
        if c == _HOSH:                          i += 1; continue
        if ord(c) < 128: out.append(c)
        i += 1
    return "".join(out)

sentence = []
def sent_ascii(): return to_banglish(" ".join(name(s) for s in sentence))
def show(ids):   return " ".join(name(s) for s in ids)

# ════════════════════════════════════════════════════════════
#  MODEL + 5-SEED ENSEMBLE LOADER
# ════════════════════════════════════════════════════════════
class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, d=128, p=0.3):
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
    def __init__(self, in_dim, n_classes, d=128, heads=4, layers=4, ff=4, p=0.3):
        super().__init__()
        self.encoder = SpatialEncoder(in_dim, d, p)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        layer = nn.TransformerEncoderLayer(d, heads, d*ff, p,
                                           batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n_classes))
    def forward(self, x):
        t = self.encoder(x); t = t + self.pos[:, :t.shape[1]]
        return self.head(self.pool(self.tf(t)))

def load_ensemble():
    """Load all 5 checkpoints; class count inferred from the head weight."""
    first = torch.load(os.path.join(CKPT, ENSEMBLE_FILES[0]),
                       map_location=DEVICE, weights_only=True)
    n_classes = first["head.2.weight"].shape[0]
    models = []
    for fn in ENSEMBLE_FILES:
        m = SignTransformer(FEATURE_DIM, n_classes, d=128, heads=4, layers=4).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(CKPT, fn),
                                     map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)
    return models, n_classes

print("loading ensemble...")
try:
    MODELS, N_CLASSES = load_ensemble()
    print(f"loaded {len(MODELS)} models | {N_CLASSES} classes | device {DEVICE}")
except Exception as e:
    MODELS, N_CLASSES = [], 102
    print("[warn] ensemble load failed:", e, "- recognition disabled")

@torch.no_grad()
def predict(buf):
    if not MODELS: return None, 0.0
    x = torch.from_numpy(np.stack(buf)[None]).float().to(DEVICE)
    p = sum(torch.softmax(m(x), 1) for m in MODELS) / len(MODELS)
    p = p[0].cpu().numpy()
    return int(p.argmax()), float(p.max())

# ════════════════════════════════════════════════════════════
#  MEDIAPIPE
# ════════════════════════════════════════════════════════════
_pose_lm = _hand_lm = None
def init_mediapipe():
    global _pose_lm, _hand_lm
    _pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=POSE_TASK),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
    _hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=HAND_TASK),
            running_mode=vision.RunningMode.IMAGE, num_hands=2))
    print("MediaPipe ready")

def _np(lms): return np.array([[p.x, p.y, p.z] for p in lms], dtype="float32")

def count_fingers(lms):
    tips, mids = [8,12,16,20], [6,10,14,18]
    wr = (lms[0].x, lms[0].y)
    ext = sum(1 for t, m in zip(tips, mids)
              if np.hypot(lms[t].x-wr[0], lms[t].y-wr[1]) >
                 np.hypot(lms[m].x-wr[0], lms[m].y-wr[1]))
    if abs(lms[4].x - lms[0].x) > abs(lms[3].x - lms[0].x): ext += 1
    return ext

def detect_one(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr  = _pose_lm.detect(img)
    hr  = _hand_lm.detect(img)
    c   = np.zeros((N_LM, 3), np.float32)
    m   = np.zeros(N_LM, np.float32)
    lo = ro = False
    if pr.pose_landmarks:
        c[:POSE_N] = _np(pr.pose_landmarks[0]); m[:POSE_N] = 1.0
    if hr.hand_landmarks:
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            a = _np(lms)
            if hd[0].category_name == "Left":
                c[POSE_N:POSE_N+HAND_N] = a; m[POSE_N:POSE_N+HAND_N] = 1.0
                if count_fingers(lms) >= 5: lo = True
            else:
                c[POSE_N+HAND_N:] = a; m[POSE_N+HAND_N:] = 1.0
                if count_fingers(lms) >= 5: ro = True
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

# ════════════════════════════════════════════════════════════
#  SERIAL  (write-locked + background telemetry reader)
# ════════════════════════════════════════════════════════════
_ser = None
_ser_lock = threading.Lock()
_distance = -1.0

def serial_init():
    global _ser
    try:
        _ser = pyserial.Serial(STM_PORT, STM_BAUD, timeout=0.1)
        time.sleep(2.0)
        print("serial", STM_PORT, "OK")
    except Exception as e:
        print("[warn] serial:", e, "- motors/OLED off, recognition still runs")

def _send(line):
    with _ser_lock:
        if _ser is None: return
        try: _ser.write((line + "\n").encode())
        except Exception as e: print("[warn] serial write:", e)

def serial_reader():
    global _distance
    buf = ""
    while shared["run"]:
        if _ser is None: time.sleep(0.2); continue
        try: data = _ser.read(64).decode(errors="ignore")
        except Exception: time.sleep(0.1); continue
        buf += data
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line.startswith("D"):
                try: _distance = float(line[1:])
                except ValueError: pass

class RobotMotors:
    def _m(self, c): _send(c)
    def forward(self):   self._m("F")
    def backward(self):  self._m("B")
    def left(self):      self._m("L")
    def right(self):     self._m("R")
    def stop(self):      self._m("S")
    def fwdleft(self):   self._m("G")
    def fwdright(self):  self._m("H")
    def backleft(self):  self._m("J")
    def backright(self): self._m("K")
    def arm_nudge(self, side, up): _send(f"A{side}{'U' if up else 'D'}")
    def arm_set(self, side, deg):  _send(f"A{side}{clamp(int(deg),0,90):03d}")
    def oled(self, top="", bottom=None):
        if top:                _send("#" + top[:80])
        if bottom is not None: _send("$" + bottom[:80])
    def distance(self): return _distance

robot = RobotMotors()

# ════════════════════════════════════════════════════════════
#  AUDIO (offline wav; paplay -> aplay fallback; keep-alive)
# ════════════════════════════════════════════════════════════
_aud = {"awake": 0.0}
def _play(path):
    if not os.path.exists(path): print("[warn] no audio:", path); return
    _aud["awake"] = time.time()
    for player in ("paplay", "aplay"):
        try:
            subprocess.run([player, path], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10); return
        except FileNotFoundError: continue
        except subprocess.TimeoutExpired: return

def _wake():
    if time.time() - _aud["awake"] > 8.0:
        sil = os.path.join(AUDIO_DIR, "_silence.wav")
        if os.path.exists(sil): _play(sil)
        else: time.sleep(0.05)

def speak_word(cid):
    path = os.path.join(AUDIO_DIR, str(cid) + ".wav")
    threading.Thread(target=lambda: (_wake(), _play(path)), daemon=True).start()

def speak_sentence(ids):
    def _r():
        _wake()
        for cid in ids: _play(os.path.join(AUDIO_DIR, str(cid)+".wav")); time.sleep(0.15)
    threading.Thread(target=_r, daemon=True).start()

# ════════════════════════════════════════════════════════════
#  SHARED STATE
# ════════════════════════════════════════════════════════════
shared = {
    "buf": deque(maxlen=N_FRAMES), "run": True, "frame": None,
    "l_last": 0.0, "r_last": 0.0,
    "mode": "idle", "collecting": False,
    "person_cx": None, "shoulder_w": None,
    "nose_x": 0.5, "nose_y": 0.5, "pose_found": False,
    "last_word": "", "hint": "",
    "drive_last": 0.0, "driving": False, "drive_cmd": "S",
    "arm_L": 45.0, "arm_R": 45.0,
    "follow_lost_t": 0.0,
}
buf_lock = threading.Lock()

def set_mode(m):
    shared["mode"] = m
    robot.stop(); shared["driving"] = False; shared["drive_cmd"] = "S"
    if   m == "idle":        robot.oled(top="IDLE|RIGHT x1=recog  x2=follow")
    elif m == "recognition": robot.oled(top="RECOG|RIGHT=sign LEFT=del|LEFTx2=idle BOTH=say",
                                        bottom=sent_ascii())
    elif m == "following":   robot.oled(top="FOLLOWING|LEFT hand=stop")
    elif m == "manual":      robot.oled(top="MANUAL|phone joysticks active")

# ════════════════════════════════════════════════════════════
#  CAPTURE THREAD  (single camera + single MediaPipe pass)
# ════════════════════════════════════════════════════════════
_HCON = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
_PCON = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]

def capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); shared["run"] = False; return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
        with buf_lock: shared["buf"].append(to_feat(c, m))

        if pr.pose_landmarks:
            shared["pose_found"] = True
            nose = pr.pose_landmarks[0][0]
            shared["nose_x"], shared["nose_y"] = float(nose.x), float(nose.y)
            if m[L_SHO] > 0 and m[R_SHO] > 0:
                shared["person_cx"]  = float((c[L_SHO][0] + c[R_SHO][0]) / 2.0)
                shared["shoulder_w"] = float(abs(c[L_SHO][0] - c[R_SHO][0]))
            else:
                shared["person_cx"] = None; shared["shoulder_w"] = 0.0
        else:
            shared["pose_found"] = False
            shared["person_cx"] = None; shared["shoulder_w"] = 0.0

        if pr.pose_landmarks:
            pts = [(int(p.x*w), int(p.y*h)) for p in pr.pose_landmarks[0]]
            for a, b in _PCON: cv2.line(frame, pts[a], pts[b], (255,150,0), 2)
            nx, ny = pts[0]
            cv2.rectangle(frame, (nx-55, ny-55), (nx+55, ny+55), (0,255,255), 2)
        if hr.hand_landmarks:
            for hand in hr.hand_landmarks:
                hp = [(int(p.x*w), int(p.y*h)) for p in hand]
                for a, b in _HCON: cv2.line(frame, hp[a], hp[b], (0,255,0), 2)
                for p in hp: cv2.circle(frame, p, 3, (0,0,255), -1)
        if lo or ro:
            cv2.putText(frame, "HANDS DETECTED", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "MODE: " + shared["mode"], (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, shared["last_word"], (10, h-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,255), 2)
        shared["frame"] = frame
    cap.release()

# ════════════════════════════════════════════════════════════
#  HEAD TRACKING  (gpiozero, software-PWM jitter fix: detach + MIN_MOVE)
#  pigpio is NOT supported on the Pi 5, so we use detach() and tuning.
#  IMPORTANT: servos are detached on no-face and on program exit so they
#  go SILENT when the code stops (no buzzing when idle).
# ════════════════════════════════════════════════════════════
_pan_srv = _tilt_srv = None

def _ht_detach():
    for s in (_pan_srv, _tilt_srv):
        if s is not None:
            try: s.detach()
            except Exception: pass

def head_track_thread():
    global _pan_srv, _tilt_srv
    if not HEAD_TRACK_ENABLE:
        return
    try:
        from gpiozero import AngularServo
        _pan_srv  = AngularServo(HT_PAN_PIN,  min_angle=0, max_angle=270,
                                 min_pulse_width=0.0005, max_pulse_width=0.0025)
        _tilt_srv = AngularServo(HT_TILT_PIN, min_angle=0, max_angle=270,
                                 min_pulse_width=0.0005, max_pulse_width=0.0025)
    except Exception as e:
        print("[warn] head tracking disabled:", e); return

    atexit.register(_ht_detach)           # detach on exit -> silent when code stops
    pan_a = tilt_a = HT_CENTER
    prev_ex = prev_ey = 0.0
    last_seen = 0.0

    def _write(pa, ta):
        try:
            _pan_srv.angle  = clamp(pa,  HT_PAN_MIN,  HT_PAN_MAX)
            _tilt_srv.angle = clamp(ta, HT_TILT_MIN, HT_TILT_MAX)
            time.sleep(HT_DETACH_AFTER)
            _ht_detach()                  # cut the live signal -> holds quietly
        except Exception: pass

    print("head tracking running (KP=11 KD=2, detach jitter-fix)")
    while shared["run"]:
        # freeze head while capturing a sign so the camera doesn't drift
        if shared["collecting"]:
            _ht_detach(); time.sleep(0.05); continue

        if not shared["pose_found"]:
            if time.time() - last_seen > 1.0:
                if abs(pan_a - HT_CENTER) > 1 or abs(tilt_a - HT_CENTER) > 1:
                    pan_a, tilt_a = HT_CENTER, HT_CENTER
                    _write(pan_a, tilt_a)        # recenter once...
                else:
                    _ht_detach()                  # ...then stay silent
            time.sleep(0.05); continue

        last_seen = time.time()
        ex = shared["nose_x"] - 0.5
        ey = shared["nose_y"] - 0.5
        if abs(ex) < HT_DEAD_ZONE: ex = 0.0
        if abs(ey) < HT_DEAD_ZONE: ey = 0.0
        ex = HT_SMOOTH * ex + (1 - HT_SMOOTH) * prev_ex
        ey = HT_SMOOTH * ey + (1 - HT_SMOOTH) * prev_ey
        d_pan  = clamp(HT_KP * ex + HT_KD * (ex - prev_ex), -HT_MAX_STEP, HT_MAX_STEP)
        d_tilt = clamp(HT_KP * ey + HT_KD * (ey - prev_ey), -HT_MAX_STEP, HT_MAX_STEP)
        prev_ex, prev_ey = ex, ey
        if abs(d_pan) < HT_MIN_MOVE and abs(d_tilt) < HT_MIN_MOVE:
            time.sleep(0.05); continue            # ignore tiny corrections
        pan_a  = clamp(pan_a  + HT_PAN_DIR  * d_pan,  HT_PAN_MIN,  HT_PAN_MAX)
        tilt_a = clamp(tilt_a + HT_TILT_DIR * d_tilt, HT_TILT_MIN, HT_TILT_MAX)
        _write(pan_a, tilt_a)
        time.sleep(0.05)
    _ht_detach()

# ════════════════════════════════════════════════════════════
#  RECOGNITION HELPERS
# ════════════════════════════════════════════════════════════
def hand_state():
    now = time.time()
    return (now - shared["l_last"]) < HAND_RECENT, \
           (now - shared["r_last"]) < HAND_RECENT

def current_prediction():
    with buf_lock:
        if len(shared["buf"]) < N_FRAMES: return None, 0.0
        b = list(shared["buf"])
    return predict(b)

def wait_hands_down():
    while shared["run"]:
        l, r = hand_state()
        if not l and not r: return
        time.sleep(0.05)

def collect_one_sign():
    """Do NOT clear the buffer (clearing forces a 9s refill and recognition never
    completes). Use the continuously-running buffer as-is."""
    shared["collecting"] = True
    votes = deque(maxlen=9)
    t0, last_print, result = time.time(), 0.0, None
    print("  [sign now]")
    while time.time() - t0 < RECOG_WINDOW and shared["run"]:
        pid, conf = current_prediction()
        l, r = hand_state()
        now = time.time()
        if now - last_print > 0.5:
            if pid is None: print("    buffer filling...")
            else: print(f"    pred {name(pid):12s}  conf {conf:.2f}")
            last_print = now
        if pid is not None and conf >= CONF_TH and not (l or r):
            votes.append(pid)
            if len(votes) >= 5:
                v, ct = Counter(votes).most_common(1)[0]
                if ct >= VOTE_NEED:
                    print(f"  [LOCKED] {name(v)}  (conf {conf:.2f})")
                    result = v; break
        time.sleep(0.05)
    if result is None: print("  no stable sign")
    shared["collecting"] = False
    return result

def read_raw_gesture(mode, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout and shared["run"] and shared["mode"] == mode:
        l, r = hand_state()
        if l and r:     return "both"
        if r and not l: return "right"
        if l and not r: return "left"
        time.sleep(0.05)
    return None

# ════════════════════════════════════════════════════════════
#  PHONE-DRIVEN RECOGNITION ACTIONS  (SIGN / DELETE / SPEAK buttons)
# ════════════════════════════════════════════════════════════
def phone_sign():
    if shared["collecting"]: return
    robot.oled(top="RECOGNIZING...|hold the sign")
    sid = collect_one_sign()
    if sid is not None:
        sentence.append(sid); shared["last_word"] = name(sid)
        print("[WORD]", name(sid), "|", show(sentence)); speak_word(sid)
        robot.oled(top="GOT: " + to_banglish(name(sid)) + "|SIGN=next SPEAK=done",
                   bottom=sent_ascii())
    else:
        robot.oled(top="no sign|try SIGN again", bottom=sent_ascii())

def phone_delete():
    if sentence:
        rem = sentence.pop()
        shared["last_word"] = name(sentence[-1]) if sentence else ""
        print("[DELETED]", name(rem), "|", show(sentence)); speak_word(rem)
    robot.oled(top="DELETED|SIGN=add SPEAK=say", bottom=sent_ascii())

def phone_speak():
    print("[SPEAK]", show(sentence))
    if sentence: speak_sentence(list(sentence)); sentence.clear()
    shared["last_word"] = ""
    robot.oled(top="DONE|SIGN=add more", bottom="")

# ════════════════════════════════════════════════════════════
#  STATE MACHINE
# ════════════════════════════════════════════════════════════
def idle_step():
    shared["hint"] = "RIGHT x1=recognition  RIGHT x2=following"
    g = read_raw_gesture("idle")
    if g != "right": return
    robot.oled(top="1st RIGHT|waiting for 2nd...")
    second = False; t0 = time.time()
    while time.time() - t0 < DOUBLE_WINDOW and shared["run"] and shared["mode"] == "idle":
        g2 = read_raw_gesture("idle", timeout=DOUBLE_WINDOW - (time.time() - t0))
        if g2 is None: break
        if g2 == "right": second = True; break
    if shared["mode"] != "idle": return
    robot.oled(top=("2nd RIGHT -> FOLLOWING" if second else "1st RIGHT -> RECOGNITION"))
    time.sleep(0.5)
    set_mode("following" if second else "recognition")
    wait_hands_down()

def recognition_step():
    shared["hint"] = "RIGHT=sign  LEFT=delete  LEFTx2=idle  BOTH=speak"
    robot.oled(top="RECOG|RIGHT=sign LEFT=del|LEFTx2=idle BOTH=say", bottom=sent_ascii())
    g = read_raw_gesture("recognition")
    if g is None: return

    if g == "both":
        phone_speak()
        time.sleep(NAV_COOLDOWN); wait_hands_down()

    elif g == "right":
        print("[START recognition]")
        robot.oled(top="RECOGNIZING...|hold the sign still")
        time.sleep(NAV_COOLDOWN); wait_hands_down()
        sid = collect_one_sign()
        if sid is not None:
            sentence.append(sid); shared["last_word"] = name(sid)
            print("[WORD]", name(sid), "|", show(sentence)); speak_word(sid)
            robot.oled(top="GOT: " + to_banglish(name(sid)) + "|RIGHT=next BOTH=say",
                       bottom=sent_ascii())
        else:
            robot.oled(top="no sign|try RIGHT again", bottom=sent_ascii())
        wait_hands_down()

    elif g == "left":
        second = False; t0 = time.time()
        while time.time() - t0 < LEFT_DOUBLE_WINDOW and shared["run"] \
              and shared["mode"] == "recognition":
            g2 = read_raw_gesture("recognition",
                                  timeout=LEFT_DOUBLE_WINDOW - (time.time() - t0))
            if g2 is None: break
            if g2 == "left": second = True; break
        if second:
            print("[EXIT recognition -> idle]"); set_mode("idle"); wait_hands_down()
        else:
            phone_delete()
            time.sleep(NAV_COOLDOWN); wait_hands_down()

_fdiag_t = 0.0
def following_step():
    global _fdiag_t
    shared["hint"] = "following... LEFT hand = back to idle"
    l, r = hand_state()
    if l:
        print("[following] left hand -> idle")
        robot.stop(); set_mode("idle"); wait_hands_down(); return

    cx = shared["person_cx"]; sw = shared["shoulder_w"]
    dist = robot.distance(); now = time.time(); decision = "stop"

    if cx is None:
        if shared["follow_lost_t"] == 0.0: shared["follow_lost_t"] = now
        if now - shared["follow_lost_t"] > FOLLOW_LOST_T:
            robot.right(); decision = "search"
        else:
            robot.stop()
    else:
        shared["follow_lost_t"] = 0.0
        if USE_SONAR and 0 < dist < STOP_DIST:
            robot.stop(); decision = "sonar stop"
        else:
            off = (cx - 0.5) * STEER_DIR
            if abs(off) > FOLLOW_SPIN_TH:
                robot.right() if off > 0 else robot.left(); decision = "spin"
            elif abs(off) > CENTER_DEAD:
                robot.fwdright() if off > 0 else robot.fwdleft(); decision = "arc"
            else:
                if   sw is not None and sw > FOLLOW_BACK_W: robot.backward(); decision = "back"
                elif sw is not None and sw > FOLLOW_NEAR_W: robot.stop();     decision = "near"
                elif sw is not None and sw < FOLLOW_FAR_W:  robot.forward();  decision = "forward"
                else:                                       robot.stop();     decision = "centred"

    if now - _fdiag_t > 0.5:
        _fdiag_t = now
        cxs = "None" if cx is None else f"{cx:.2f}"
        sws = "None" if sw is None else f"{sw:.2f}"
        print(f"[follow] cx={cxs} sw={sws} dist={dist:.0f} -> {decision}")
        robot.oled(top=f"FOLLOW:{decision[:12]}|cx={cxs} d={dist:.0f}cm")
    time.sleep(0.04)

# ════════════════════════════════════════════════════════════
#  MANUAL DRIVE  (dual joystick mix + press-and-hold watchdog)
# ════════════════════════════════════════════════════════════
def manual_watchdog():
    while shared["run"]:
        if (shared["mode"] == "manual" and shared["driving"] and
                time.time() - shared["drive_last"] > AUTO_STOP_AFTER):
            robot.stop(); shared["driving"] = False; shared["drive_cmd"] = "S"
        time.sleep(0.05)

def _emit_drive(cmd):
    if cmd == shared["drive_cmd"] and cmd != "S":
        return  # already sending this; keepalive comes from repeated /drive calls
    shared["drive_cmd"] = cmd
    if   cmd == "F": robot.forward()
    elif cmd == "B": robot.backward()
    elif cmd == "L": robot.left()
    elif cmd == "R": robot.right()
    elif cmd == "G": robot.fwdleft()
    elif cmd == "H": robot.fwdright()
    elif cmd == "J": robot.backleft()
    elif cmd == "K": robot.backright()
    elif cmd == "S": robot.stop()

def mix_joysticks(fb, lr):
    """fb,lr in {-1,0,1}. Mix into one motor command (diagonals when both set)."""
    if fb == 0 and lr == 0:  return "S"
    if fb > 0 and lr == 0:   return "F"
    if fb < 0 and lr == 0:   return "B"
    if fb == 0 and lr < 0:   return "L"
    if fb == 0 and lr > 0:   return "R"
    if fb > 0 and lr < 0:    return "G"   # forward-left
    if fb > 0 and lr > 0:    return "H"   # forward-right
    if fb < 0 and lr < 0:    return "J"   # back-left
    if fb < 0 and lr > 0:    return "K"   # back-right
    return "S"

def do_drive_raw(c):
    """Single-letter drive (keyboard / legacy)."""
    if shared["mode"] != "manual": return
    shared["drive_last"] = time.time()
    shared["driving"] = (c != "S")
    _emit_drive(c)

def do_joystick(fb, lr):
    """Dual-joystick drive from the phone; fb/lr in {-1,0,1}."""
    if shared["mode"] != "manual": return
    shared["drive_last"] = time.time()
    cmd = mix_joysticks(fb, lr)
    shared["driving"] = (cmd != "S")
    _emit_drive(cmd)

def do_arm(cmd):
    if len(cmd) != 3 or cmd[0] != "A" or cmd[1] not in ("L","R") or cmd[2] not in ("U","D"):
        return
    side, up = cmd[1], (cmd[2] == "U")
    key = "arm_L" if side == "L" else "arm_R"
    shared[key] = clamp(shared[key] + (ARM_STEP_DEG if up else -ARM_STEP_DEG), 0, 90)
    robot.arm_nudge(side, up)

# ════════════════════════════════════════════════════════════
#  INTERACTION LOOP
# ════════════════════════════════════════════════════════════
def interaction_loop():
    while shared["run"]:
        m = shared["mode"]
        if   m == "idle":        idle_step()
        elif m == "recognition": recognition_step()
        elif m == "following":   following_step()
        else: time.sleep(0.05)

# ════════════════════════════════════════════════════════════
#  PHONE PAGE  (two joysticks + recognition + arm + video)
# ════════════════════════════════════════════════════════════
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Ishara</title>
<style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;
  -webkit-tap-highlight-color:transparent;touch-action:none}
body{margin:0;background:linear-gradient(160deg,#0a1628,#0f2942);
     color:#e2ecf5;font-family:'JetBrains Mono',ui-monospace,monospace;text-align:center}
h2{margin:10px 0 4px;font-size:16px;letter-spacing:3px;color:#5eead4}
.card{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.16);
      border-radius:14px;backdrop-filter:blur(8px);padding:10px;
      margin:0 8px 10px;max-width:460px;display:inline-block;width:calc(100% - 16px)}
.row{display:flex;gap:8px;margin-bottom:8px}.row>*{flex:1}
button{background:rgba(20,184,166,.18);border:1px solid #14b8a6;color:#e2ecf5;
       border-radius:10px;padding:13px 0;font-size:13px;font-weight:600;
       font-family:inherit}
button:active{background:#14b8a6;color:#04211d}
.on{background:#14b8a6!important;color:#04211d!important}
.stop{background:rgba(220,38,38,.25);border-color:#dc2626}
label{color:#7da0bd;font-size:11px;letter-spacing:1px;display:block;text-align:left;margin-bottom:4px}
#word{font-size:15px;color:#5eead4;min-height:20px}
#sent{font-size:12px;color:#7da0bd;word-break:break-word;min-height:14px}
#dv{font-size:12px;color:#7da0bd;margin-top:4px}
img#vid{width:100%;border-radius:10px;display:block;background:#000;margin-bottom:6px}
.sticks{display:flex;gap:12px;justify-content:center}
.stick{position:relative;width:46%;max-width:190px;aspect-ratio:1;border-radius:50%;
       background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.2);
       touch-action:none}
.knob{position:absolute;width:34%;height:34%;border-radius:50%;left:33%;top:33%;
      background:#14b8a6;box-shadow:0 0 12px rgba(20,184,166,.6)}
.sticklbl{font-size:10px;color:#7da0bd;margin-top:4px}
</style></head><body>
<h2>ISHARA · CONTROL</h2>
<div class="card">
  <img id="vid" src="/video" onerror="this.style.display='none'">
  <label>LAST WORD</label><div id="word">—</div>
  <label style="margin-top:6px">SENTENCE</label><div id="sent">—</div>
  <div id="dv">distance: —</div>
</div>
<div class="card">
  <label>MODE</label>
  <div class="row">
    <button id="b_idle"   onclick="setMode('idle')">IDLE</button>
    <button id="b_recog"  onclick="setMode('recognition')">RECOG</button>
    <button id="b_follow" onclick="setMode('following')">FOLLOW</button>
    <button id="b_manual" onclick="setMode('manual')">MANUAL</button>
  </div>
</div>
<div class="card" id="dsec" style="display:none">
  <label>DRIVE — hold both sticks; mixing = diagonals</label>
  <div class="sticks">
    <div><div class="stick" id="stk_fb"><div class="knob" id="kn_fb"></div></div>
         <div class="sticklbl">FWD / BACK</div></div>
    <div><div class="stick" id="stk_lr"><div class="knob" id="kn_lr"></div></div>
         <div class="sticklbl">LEFT / RIGHT</div></div>
  </div>
  <div class="row" style="margin-top:8px"><button class="stop" onclick="stopAll()">STOP</button></div>
</div>
<div class="card">
  <label>ARM</label>
  <div class="row">
    <button onclick="arm('ALU')">L ▲</button><button onclick="arm('ALD')">L ▼</button>
    <button onclick="arm('ARU')">R ▲</button><button onclick="arm('ARD')">R ▼</button>
  </div>
</div>
<div class="card">
  <label>RECOGNITION</label>
  <div class="row">
    <button onclick="nav('sign')">SIGN</button>
    <button onclick="nav('delete')">DELETE</button>
    <button onclick="nav('speak')">SPEAK</button>
  </div>
</div>
<script>
let fb=0, lr=0, timer=null;
function arm(c){fetch('/arm?c='+c)}
function nav(c){fetch('/nav?c='+c)}
function send(){fetch('/joy?fb='+fb+'&lr='+lr)}
function startLoop(){ if(!timer) timer=setInterval(send,150); }   // press&hold keepalive
function stopLoop(){ if(timer){clearInterval(timer);timer=null;} fb=0;lr=0; fetch('/joy?fb=0&lr=0'); }
function stopAll(){ stopLoop(); }
function setMode(m){
  fetch('/mode?m='+m).then(()=>{
    ['idle','recog','follow','manual'].forEach(x=>{
      const e=document.getElementById('b_'+x); if(e)e.classList.remove('on');});
    const b=document.getElementById('b_'+m.replace('recognition','recog'));
    if(b)b.classList.add('on');
    document.getElementById('dsec').style.display=(m==='manual'?'block':'none');
    if(m!=='manual') stopLoop();
  });
}
// generic 1-axis stick: returns -1/0/+1 by drag direction, snaps back on release
function bindStick(stickId, knobId, axis){
  const stick=document.getElementById(stickId), knob=document.getElementById(knobId);
  function setKnob(dx,dy){ knob.style.left=(33+dx*22)+'%'; knob.style.top=(33+dy*22)+'%'; }
  function handle(e){
    e.preventDefault();
    const t=(e.touches&&e.touches[0])||e;
    const r=stick.getBoundingClientRect();
    const cx=r.left+r.width/2, cy=r.top+r.height/2;
    const nx=Math.max(-1,Math.min(1,(t.clientX-cx)/(r.width/2)));
    const ny=Math.max(-1,Math.min(1,(t.clientY-cy)/(r.height/2)));
    if(axis==='fb'){ fb = ny<-0.3?1:(ny>0.3?-1:0); setKnob(0,ny); }   // up=forward
    else           { lr = nx<-0.3?-1:(nx>0.3?1:0);  setKnob(nx,0); }   // right=right
    startLoop(); send();
  }
  function release(e){ e.preventDefault();
    if(axis==='fb'){fb=0;} else {lr=0;}
    setKnob(0,0);
    if(fb===0&&lr===0) stopLoop(); else send();
  }
  stick.addEventListener('touchstart',handle); stick.addEventListener('touchmove',handle);
  stick.addEventListener('touchend',release);  stick.addEventListener('touchcancel',release);
  stick.addEventListener('mousedown',e=>{stick._d=true;handle(e);});
  window.addEventListener('mousemove',e=>{if(stick._d)handle(e);});
  window.addEventListener('mouseup',e=>{if(stick._d){stick._d=false;release(e);}});
}
bindStick('stk_fb','kn_fb','fb');
bindStick('stk_lr','kn_lr','lr');
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(s=>{
    document.getElementById('word').textContent=s.word||'—';
    document.getElementById('sent').textContent=s.sentence||'—';
    document.getElementById('dv').textContent='distance: '+(s.dist>0?s.dist.toFixed(1)+' cm':'—');
  }).catch(()=>{});
},500);
setMode('idle');
</script></body></html>"""

# ════════════════════════════════════════════════════════════
#  HTTP SERVER
# ════════════════════════════════════════════════════════════
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _resp(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def do_GET(self):
        try:
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            p = u.path

            if p in ("/", "/index.html"):
                self._resp(200, "text/html; charset=utf-8", PAGE); return

            if p == "/status":
                self._resp(200, "application/json", json.dumps({
                    "mode": shared["mode"], "word": shared["last_word"],
                    "sentence": sent_ascii(), "dist": round(robot.distance(), 1),
                    "arm_L": round(shared["arm_L"]), "arm_R": round(shared["arm_R"])})); return

            if p == "/mode":
                m = q.get("m", [""])[0]
                if m in ("idle","recognition","following","manual"): set_mode(m)
                self._resp(200, "application/json", json.dumps({"mode": shared["mode"]})); return

            if p == "/joy":   # dual-joystick: fb,lr in {-1,0,1}
                try: fb = int(q.get("fb", ["0"])[0]); lr = int(q.get("lr", ["0"])[0])
                except ValueError: fb = lr = 0
                do_joystick(fb, lr)
                self._resp(200, "application/json", b'{"ok":true}'); return

            if p == "/drive":  # single-letter (legacy / keyboard)
                do_drive_raw(q.get("c", ["S"])[0])
                self._resp(200, "application/json", b'{"ok":true}'); return

            if p == "/arm":
                do_arm(q.get("c", [""])[0])
                self._resp(200, "application/json", json.dumps(
                    {"arm_L": round(shared["arm_L"]), "arm_R": round(shared["arm_R"])})); return

            if p == "/nav":
                c = q.get("c", [""])[0]
                if   c == "sign":   threading.Thread(target=phone_sign,   daemon=True).start()
                elif c == "delete": phone_delete()
                elif c == "speak":  phone_speak()
                self._resp(200, "application/json", b'{"ok":true}'); return

            if p == "/video":
                self.send_response(200)
                self.send_header("Age", "0"); self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while shared["run"]:
                        f = shared["frame"]
                        if f is not None:
                            ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if ok:
                                d = jpg.tobytes()
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                                 + f"Content-Length: {len(d)}\r\n\r\n".encode()
                                                 + d + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError): pass
                return

            self._resp(404, "text/plain", "not found")
        except Exception as e:
            print("[warn] handler:", e)

class ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True

def start_server():
    srv = ThreadedHTTP(("0.0.0.0", PHONE_PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"phone control: http://10.42.0.1:{PHONE_PORT}")
    return srv

# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main():
    init_mediapipe()
    serial_init()
    threading.Thread(target=serial_reader,  daemon=True).start()
    threading.Thread(target=capture_thread, daemon=True).start()

    print("priming buffer (~9 s at 7 fps) ...")
    t0 = time.time()
    while True:
        with buf_lock: n = len(shared["buf"])
        if n >= N_FRAMES: print("buffer primed"); break
        if time.time() - t0 > 30: print("WARN: buffer slow to fill"); break
        time.sleep(0.3)

    threading.Thread(target=head_track_thread, daemon=True).start()
    threading.Thread(target=interaction_loop,  daemon=True).start()
    threading.Thread(target=manual_watchdog,   daemon=True).start()
    srv = start_server()
    set_mode("idle")
    robot.oled(top="ISHARA READY|RIGHT x1=recog x2=follow", bottom="")

    print("\nGESTURES:")
    print("  idle:        RIGHT x1 = recognition  |  RIGHT x2 = following")
    print("  recognition: RIGHT = sign | LEFT x1 = delete | LEFT x2 = idle | BOTH = speak")
    print("  following:   LEFT = idle")
    print("PHONE: two joysticks (FB + LR, mix = diagonals, hold to run), SIGN/DELETE/SPEAK")
    print("KEYS (OpenCV window): i/r/f/m  q=quit  W/A/S/D/G/H/J/K  X/space=stop\n")

    try:
        while shared["run"]:
            f = shared["frame"]
            if f is not None: cv2.imshow("Ishara", f)
            key = cv2.waitKey(30) & 0xFF
            if   key == ord('q'): shared["run"] = False
            elif key == ord('i'): set_mode("idle")
            elif key == ord('r'): set_mode("recognition")
            elif key == ord('f'): set_mode("following")
            elif key == ord('m'): set_mode("manual")
            elif key == ord('w'): do_drive_raw("F")
            elif key == ord('s'): do_drive_raw("B")
            elif key == ord('a'): do_drive_raw("L")
            elif key == ord('d'): do_drive_raw("R")
            elif key == ord('g'): do_drive_raw("G")
            elif key == ord('h'): do_drive_raw("H")
            elif key == ord('j'): do_drive_raw("J")
            elif key == ord('k'): do_drive_raw("K")
            elif key in (ord('x'), ord(' ')): do_drive_raw("S")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[exit]")
    finally:
        shared["run"] = False
        robot.stop()
        _ht_detach()          # make sure head servos go silent on exit
        time.sleep(0.5)
        cv2.destroyAllWindows()
        try: srv.shutdown()
        except Exception: pass
        print("[done]")

if __name__ == "__main__":
    main()
