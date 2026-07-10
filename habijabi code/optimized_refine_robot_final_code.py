#!/usr/bin/env python3
"""
PHASE 4 - integrated robot (self-contained) + OLED status + PHONE control.

Control options (both work at once):
  * Hand gestures (idle: RIGHT x1 -> recognition, RIGHT x2 -> following;
    recognition: RIGHT=sign LEFT=delete LEFTx2=idle BOTH=speak; following: LEFT=idle)
  * Phone / browser: open  http://<pi-ip>:8000  -> mode buttons + a drive pad
  * Keyboard fallback (click the OpenCV window): i/r/f/m, q=quit; manual W/A/S/D

STM32 link (newline-framed): Pi sends "F/B/L/R/S" (motors), "#<top>" / "$<bottom>"
(OLED); STM32 streams "D<distance>". Flash the matching stm32_phase4.ino.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["PYTHONUTF8"] = "1"

import time, threading, subprocess, json
import http.server, socketserver, urllib.parse
import numpy as np, cv2, torch, torch.nn as nn
from collections import deque, Counter

torch.set_num_threads(4)

# ============================================================
#  CONFIG
# ============================================================
BASE = "/home/hudai/Desktop/thesis"
CKPT = os.path.join(BASE, "checkpoints")
ASSETS = os.path.join(BASE, "assets")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
AUDIO_DIR = os.path.join(BASE, "audio")
DEVICE = "cpu"

STM_PORT, STM_BAUD = "/dev/ttyACM0", 115200
CMD_FWD, CMD_BACK, CMD_LEFT, CMD_RIGHT, CMD_STOP = "F", "B", "L", "R", "S"
TELE_PREFIX = "D"
PHONE_PORT = 8000

POSE_N, HAND_N = 33, 21
N_LM = POSE_N + 2 * HAND_N
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM * 3 + N_LM
N_FRAMES = 64

CONF_TH = 0.55
VOTE_NEED = 4
RECOG_WINDOW = 12.0
NAV_COOLDOWN = 3.0
HAND_RECENT = 0.5
DOUBLE_WINDOW = 2.0
LEFT_DOUBLE_WINDOW = 1.0

CENTER_DEAD = 0.12
STOP_DIST = 40
FOLLOW_DIST = 70
AUTO_STOP_AFTER = 0.6
# following uses the CAMERA first (shoulder width = distance); sonar is only a safety stop
USE_SONAR = True         # set False if your sonar misreads and wrongly blocks following
FOLLOW_NEAR_W = 0.40     # shoulders wider than this in frame => too close => stop
FOLLOW_FAR_W = 0.22      # shoulders narrower than this => person is far => move forward
STEER_DIR = 1            # set to -1 if the robot turns the WRONG way (mirrored camera)

# head-tracking servos on GPIO22 (pan) / GPIO23 (tilt)
HT_DEAD = 0.05; HT_KP = 11.0; HT_KD = 2.0; HT_SMOOTH = 0.6
HT_MAX_STEP = 2.5; HT_MIN_MOVE = 0.6; HT_SETTLE = 0.04; HT_CENTER = 135.0
PAN_PIN, TILT_PIN = 22, 23
PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = 30, 240, 60, 210
PAN_DIR, TILT_DIR = -1, 1

# optional: fix specific words if auto-transliteration looks off  {class_id: "banglish"}
ROMAN_OVERRIDE = {}

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ============================================================
#  BENGALI -> BANGLISH (Latin) transliteration for the OLED
# ============================================================
_BN_VOWEL = {'অ':'o','আ':'a','ই':'i','ঈ':'i','উ':'u','ঊ':'u','ঋ':'ri','এ':'e','ঐ':'oi','ও':'o','ঔ':'ou'}
_BN_CONS = {'ক':'k','খ':'kh','গ':'g','ঘ':'gh','ঙ':'ng','চ':'ch','ছ':'chh','জ':'j','ঝ':'jh','ঞ':'n',
            'ট':'t','ঠ':'th','ড':'d','ঢ':'dh','ণ':'n','ত':'t','থ':'th','দ':'d','ধ':'dh','ন':'n',
            'প':'p','ফ':'ph','ব':'b','ভ':'bh','ম':'m','য':'j','র':'r','ল':'l','শ':'sh','ষ':'sh',
            'স':'s','হ':'h','ড়':'r','ঢ়':'rh','য়':'y','ৎ':'t'}
_BN_MATRA = {'া':'a','ি':'i','ী':'i','ু':'u','ূ':'u','ৃ':'ri','ে':'e','ৈ':'oi','ো':'o','ৌ':'ou'}
_BN_OTHER = {'ং':'ng','ঃ':'h','ঁ':'n','়':''}
_BN_DIGIT = {'০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9'}
_HASANTA = '্'

def bn_to_latin(s):
    out = []; ch = list(s); i = 0; n = len(ch)
    while i < n:
        c = ch[i]
        if c in _BN_CONS:
            out.append(_BN_CONS[c])
            nxt = ch[i + 1] if i + 1 < n else ''
            if nxt == _HASANTA:           # conjunct: drop inherent vowel, join next
                i += 2; continue
            if nxt in _BN_MATRA:
                out.append(_BN_MATRA[nxt]); i += 2; continue
            last = (i + 1 >= n) or (ch[i + 1] == ' ')
            if not last: out.append('o')  # inherent vowel
            i += 1; continue
        if c in _BN_VOWEL: out.append(_BN_VOWEL[c]); i += 1; continue
        if c in _BN_MATRA: out.append(_BN_MATRA[c]); i += 1; continue
        if c in _BN_OTHER: out.append(_BN_OTHER[c]); i += 1; continue
        if c in _BN_DIGIT: out.append(_BN_DIGIT[c]); i += 1; continue
        if c == ' ': out.append(' '); i += 1; continue
        if ord(c) < 128: out.append(c)
        i += 1
    return ''.join(out)

# ============================================================
#  STM32 SERIAL LINK (built in)
# ============================================================
class Robot:
    def __init__(self):
        self.ser = None; self.ok = False; self._dist = -1.0; self._buf = ""; self._last = ""
        self._wlock = threading.Lock()
    def connect(self):
        try:
            import serial
            self.ser = serial.Serial(STM_PORT, STM_BAUD, timeout=0.1)
            time.sleep(2.0)
            self.ok = True; print("STM32 serial connected on", STM_PORT)
            threading.Thread(target=self._reader, daemon=True).start()
        except Exception as e:
            print("STM32 NOT connected:", e, "- motors/OLED disabled, recognition still runs")
            self.ok = False
    def _reader(self):
        while shared["run"] and self.ok:
            try: data = self.ser.read(64).decode(errors="ignore")
            except Exception: break
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line.startswith(TELE_PREFIX):
                    try: self._dist = float(line[1:])
                    except ValueError: pass
    def _raw(self, s):
        if not self.ok: return
        with self._wlock:
            try: self.ser.write(s.encode())
            except Exception: pass
    def _cmd(self, ch):
        if ch == self._last: return
        self._last = ch; self._raw(ch + "\n")
    def forward(self):  self._cmd(CMD_FWD)
    def backward(self): self._cmd(CMD_BACK)
    def left(self):     self._cmd(CMD_LEFT)
    def right(self):    self._cmd(CMD_RIGHT)
    def stop(self):     self._cmd(CMD_STOP)
    def distance(self): return self._dist
    def oled(self, top=None, bottom=None):
        if top is not None:    self._raw("#" + top + "\n")
        if bottom is not None: self._raw("$" + bottom + "\n")

robot = Robot()

# ============================================================
#  MODEL
# ============================================================
class SpatialEncoder(nn.Module):
    def __init__(self, d_in, d=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p),
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
    def forward(self, x): return self.net(x)

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.q = nn.Linear(d, 1)
    def forward(self, x):
        w = torch.softmax(self.q(x).squeeze(-1), 1).unsqueeze(-1)
        return (x * w).sum(1)

class SignTransformer(nn.Module):
    def __init__(self, d_in, n, d=128, heads=4, layers=4, ff=4, p=0.3):
        super().__init__()
        self.encoder = SpatialEncoder(d_in, d, p)
        self.pos = nn.Parameter(torch.zeros(1, 512, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * ff, p,
                                           batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers)
        self.pool = AttnPool(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n))
    def forward(self, x):
        t = self.encoder(x); t = t + self.pos[:, :t.shape[1]]
        return self.head(self.pool(self.tf(t)))

import openpyxl
ws = openpyxl.load_workbook(WORD_XLSX).active
ID2NAME = {int(r[0]): str(r[1]).strip()
           for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}
def name(i): return ID2NAME.get(int(i), "class_" + str(i))

files = ["word_model_full.pt", "word_model_128_s2.pt", "word_model_128_s3.pt",
         "word_model_128_s4.pt", "word_model_128_s5.pt"]
first = torch.load(os.path.join(CKPT, files[0]), map_location=DEVICE, weights_only=True)
N_CLASSES = first["head.2.weight"].shape[0]
MODELS = []
for fn in files:
    m = SignTransformer(FEATURE_DIM, N_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT, fn), map_location=DEVICE, weights_only=True))
    m.eval(); MODELS.append(m)
print("loaded", len(MODELS), "models |", N_CLASSES, "classes")

# ============================================================
#  MEDIAPIPE + FEATURES
# ============================================================
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
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

@torch.no_grad()
def predict(buf):
    x = torch.from_numpy(np.stack(buf)[None]).float()
    p = sum(torch.softmax(m(x), 1) for m in MODELS) / len(MODELS)
    p = p[0].numpy()
    return int(p.argmax()), float(p.max())

# ============================================================
#  AUDIO
# ============================================================
SIL = {"awake": 0.0}
def _wake():
    if time.time() - SIL["awake"] > 8.0:
        sil = os.path.join(AUDIO_DIR, "_silence.wav")
        if os.path.exists(sil): subprocess.run(["paplay", sil], check=False)
        else: time.sleep(0.1)
    SIL["awake"] = time.time()
def _play(path):
    if os.path.exists(path):
        subprocess.run(["paplay", path], check=False); SIL["awake"] = time.time()
def speak_word(cid):
    threading.Thread(target=lambda: (_wake(), _play(os.path.join(AUDIO_DIR, str(cid) + ".wav"))),
                     daemon=True).start()
def speak_sentence(ids):
    def _r():
        _wake()
        for cid in ids: _play(os.path.join(AUDIO_DIR, str(cid) + ".wav")); time.sleep(0.15)
    threading.Thread(target=_r, daemon=True).start()

# ============================================================
#  SHARED STATE + CAPTURE
# ============================================================
shared = {"buf": deque(maxlen=N_FRAMES), "run": True, "frame": None,
          "l_last": 0.0, "r_last": 0.0, "mode": "idle", "collecting": False,
          "person_cx": None, "shoulder_w": None, "left_raised": False,
          "nose_x": 0.5, "nose_y": 0.5, "pose_found": False,
          "last_word": "", "hint": "", "drive_last": 0.0, "driving": False}
buf_lock = threading.Lock()

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

# ============================================================
#  HEAD-TRACKING (recognition only, frozen during capture)
# ============================================================
HEAD_OK = False; pan = tilt = None
try:
    from gpiozero import AngularServo
    pan = AngularServo(PAN_PIN, min_angle=0, max_angle=270, min_pulse_width=0.0005, max_pulse_width=0.0025)
    tilt = AngularServo(TILT_PIN, min_angle=0, max_angle=270, min_pulse_width=0.0005, max_pulse_width=0.0025)
    HEAD_OK = True; print("head-tracking servos ready on GPIO", PAN_PIN, "/", TILT_PIN)
except Exception as e:
    print("head servos NOT available:", e)

def _park():
    if HEAD_OK:
        try: pan.angle = HT_CENTER; tilt.angle = HT_CENTER; time.sleep(0.4); pan.detach(); tilt.detach()
        except Exception: pass

def head_track_thread():
    if not HEAD_OK: return
    _park()
    pa = ta = HT_CENTER; sx = sy = 0.5; pdx = pdy = 0.0; was = False
    while shared["run"]:
        active = (shared["mode"] == "recognition" and not shared["collecting"] and shared["pose_found"])
        if not active:
            if was: _park(); was = False
            time.sleep(0.06); continue
        was = True
        sx = HT_SMOOTH*sx + (1-HT_SMOOTH)*shared["nose_x"]
        sy = HT_SMOOTH*sy + (1-HT_SMOOTH)*shared["nose_y"]
        dx, dy = sx-0.5, sy-0.5; ddx, ddy = dx-pdx, dy-pdy; pdx, pdy = dx, dy
        moved = False
        if abs(dx) > HT_DEAD:
            mv = clamp(HT_KP*dx + HT_KD*ddx, -HT_MAX_STEP, HT_MAX_STEP)
            if abs(mv) >= HT_MIN_MOVE:
                pa = clamp(pa + PAN_DIR*mv, PAN_MIN, PAN_MAX)
                try: pan.angle = pa
                except Exception: pass
                moved = True
        if abs(dy) > HT_DEAD:
            mv = clamp(HT_KP*dy + HT_KD*ddy, -HT_MAX_STEP, HT_MAX_STEP)
            if abs(mv) >= HT_MIN_MOVE:
                ta = clamp(ta + TILT_DIR*mv, TILT_MIN, TILT_MAX)
                try: tilt.angle = ta
                except Exception: pass
                moved = True
        if moved:
            time.sleep(HT_SETTLE)
            try: pan.detach(); tilt.detach()
            except Exception: pass
        time.sleep(0.03)

# ============================================================
#  GESTURE PRIMITIVES
# ============================================================
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

# ============================================================
#  MODE STATE MACHINE
# ============================================================
sentence = []
def roman(cid):
    if int(cid) in ROMAN_OVERRIDE: return ROMAN_OVERRIDE[int(cid)]
    return bn_to_latin(name(cid))
def sent_ascii(): return " ".join(roman(s) for s in list(sentence))
def show(ids): return " ".join(name(s) for s in ids)

def set_mode(new):
    if new == shared["mode"]: return
    robot.stop(); shared["driving"] = False; shared["mode"] = new; print("MODE ->", new)
    if new == "idle":
        robot.oled(top="IDLE|RIGHT x1 = recognize|RIGHT x2 = follow", bottom=sent_ascii())
    elif new == "recognition":
        robot.oled(top="RECOGNITION|RIGHT=add sign|LEFT=delete|LEFTx2=exit  BOTH=say", bottom=sent_ascii())
    elif new == "following":
        robot.oled(top="FOLLOWING|tracking person|LEFT = back to idle", bottom="")
    elif new == "manual":
        robot.oled(top="MANUAL|drive from phone|or W/A/S/D here", bottom="")

def collect_one_sign():
    shared["collecting"] = True
    clear_buffer(); votes = deque(maxlen=9); t0 = time.time(); lastp = 0; result = None
    print("  [sign now]")
    while time.time() - t0 < RECOG_WINDOW and shared["run"] and shared["mode"] == "recognition":
        pid, conf = current_prediction(); l, r = hand_state(); now = time.time()
        if now - lastp > 0.5:
            print("    pred", (name(pid) if pid is not None else "..."), round(conf, 2)); lastp = now
        if pid is not None and conf >= CONF_TH and not (l or r):
            votes.append(pid)
            if len(votes) >= 5:
                v, ct = Counter(votes).most_common(1)[0]
                if ct >= VOTE_NEED: print("  [LOCKED]", name(v)); result = v; break
        time.sleep(0.05)
    shared["collecting"] = False
    if result is None: print("  no stable sign")
    return result

def idle_step():
    shared["hint"] = "RIGHT x1 = recognition, x2 = following"
    g = read_raw_gesture("idle")
    if g != "right": return
    print("[idle] 1st right hand - waiting for a 2nd...")
    robot.oled(top="1st RIGHT hand seen|waiting for 2nd...")
    second = False; t0 = time.time()
    while time.time() - t0 < DOUBLE_WINDOW and shared["run"] and shared["mode"] == "idle":
        g2 = read_raw_gesture("idle", timeout=DOUBLE_WINDOW - (time.time() - t0))
        if g2 is None: break
        if g2 == "right": second = True; break
    if shared["mode"] == "idle":
        robot.oled(top=("2nd RIGHT hand|-> FOLLOWING" if second else "no 2nd hand|-> RECOGNITION"))
        time.sleep(0.5)
        set_mode("following" if second else "recognition"); wait_hands_down()

def recognition_step():
    shared["hint"] = "RIGHT=sign  LEFT=delete  LEFTx2=idle  BOTH=speak"
    g = read_raw_gesture("recognition")
    if g is None: return
    if g == "both":
        print("[SPEAK]", show(sentence))
        robot.oled(top="SPEAKING sentence...")
        if sentence: speak_sentence(list(sentence)); sentence.clear()
        shared["last_word"] = ""
        robot.oled(top="RECOGNITION|RIGHT=add sign|LEFT=delete|LEFTx2=exit  BOTH=say", bottom="")
        time.sleep(NAV_COOLDOWN); wait_hands_down()
    elif g == "right":
        print("[START]")
        robot.oled(top="RECOGNIZING...|hold the sign still")
        time.sleep(NAV_COOLDOWN); wait_hands_down()
        sid = collect_one_sign()
        if shared["mode"] != "recognition":
            return
        if sid is not None:
            sentence.append(sid); shared["last_word"] = name(sid)
            print("[WORD]", name(sid), "|", show(sentence)); speak_word(sid)
            robot.oled(top="GOT: " + roman(sid) + "|RIGHT=next  BOTH=say", bottom=sent_ascii())
        else:
            robot.oled(top="no stable sign|try RIGHT again", bottom=sent_ascii())
        wait_hands_down()
    elif g == "left":
        second = False; t0 = time.time()
        while time.time() - t0 < LEFT_DOUBLE_WINDOW and shared["run"] and shared["mode"] == "recognition":
            g2 = read_raw_gesture("recognition", timeout=LEFT_DOUBLE_WINDOW - (time.time() - t0))
            if g2 is None: break
            if g2 == "left": second = True; break
        if second:
            print("[recognition] LEFT x2 -> idle"); set_mode("idle"); wait_hands_down()
        else:
            if sentence:
                rem = sentence.pop()
                shared["last_word"] = name(sentence[-1]) if sentence else ""
                print("[DELETED]", name(rem), "|", show(sentence)); speak_word(rem)
            robot.oled(top="DELETED last word|RIGHT=add  BOTH=say", bottom=sent_ascii())
            time.sleep(NAV_COOLDOWN); wait_hands_down()

_follow_exit_n = 0
_follow_diag_t = 0.0
def following_step():
    global _follow_exit_n, _follow_diag_t
    shared["hint"] = "following... raise LEFT hand = back to idle"

    # ---- exit: left open palm OR left arm raised, held 2 frames (robust at distance) ----
    l, r = hand_state()
    if l or shared["left_raised"]:
        _follow_exit_n += 1
        if _follow_exit_n >= 2:
            _follow_exit_n = 0
            print("[following] LEFT -> idle"); robot.stop(); set_mode("idle"); wait_hands_down(); return
    else:
        _follow_exit_n = 0

    cx = shared["person_cx"]; sw = shared["shoulder_w"]; dist = robot.distance()
    decision = "stop"
    if cx is None:
        robot.stop(); decision = "no person"
    else:
        off = cx - 0.5
        sonar_close = USE_SONAR and (8.0 < dist < STOP_DIST)   # trust only a plausible reading
        cam_close = (sw is not None and sw > FOLLOW_NEAR_W)
        if sonar_close or cam_close:
            robot.stop(); decision = "close-stop"
        elif abs(off) > CENTER_DEAD:
            go_left = (off < 0) if STEER_DIR > 0 else (off > 0)
            if go_left: robot.left();  decision = "turn LEFT"
            else:       robot.right(); decision = "turn RIGHT"
        elif (sw is not None and sw < FOLLOW_FAR_W) or (dist > FOLLOW_DIST):
            robot.forward(); decision = "forward"
        else:
            robot.stop(); decision = "good-dist stop"

    now = time.time()
    if now - _follow_diag_t > 0.5:
        _follow_diag_t = now
        cxs = "None" if cx is None else ("%.2f" % cx)
        sws = "None" if sw is None else ("%.2f" % sw)
        print("[follow] cx=%s sw=%s dist=%.0f -> %s" % (cxs, sws, dist, decision))
        robot.oled(top="FOLLOW: %s|cx=%s d=%.0f|raise LEFT = idle" % (decision, cxs, dist))
    time.sleep(0.05)

def interaction_loop():
    while shared["run"]:
        mode = shared["mode"]
        if mode == "idle": idle_step()
        elif mode == "recognition": recognition_step()
        elif mode == "following": following_step()
        else: time.sleep(0.05)

# manual driving safety: stop if no drive command arrives in time
def manual_watchdog():
    while shared["run"]:
        if shared["mode"] == "manual" and shared["driving"] and \
           time.time() - shared["drive_last"] > AUTO_STOP_AFTER:
            robot.stop(); shared["driving"] = False
        time.sleep(0.05)

def do_drive(c):
    if shared["mode"] != "manual": return
    now = time.time()
    if c == "F": robot.forward();  shared["drive_last"] = now; shared["driving"] = True
    elif c == "B": robot.backward(); shared["drive_last"] = now; shared["driving"] = True
    elif c == "L": robot.left();     shared["drive_last"] = now; shared["driving"] = True
    elif c == "R": robot.right();    shared["drive_last"] = now; shared["driving"] = True
    elif c == "S": robot.stop();     shared["driving"] = False

# ============================================================
#  PHONE / BROWSER CONTROL SERVER
# ============================================================
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Robot</title><style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
body{margin:0;background:#0d0f12;color:#e8e8e8;font-family:-apple-system,Segoe UI,Roboto,sans-serif;text-align:center}
h2{margin:14px 0 4px}
.bar{font-size:14px;color:#9aa;margin-bottom:10px}
.bar b{color:#6cf}
.modes{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;padding:0 10px 14px}
.modes button{flex:1 1 40%;padding:14px;font-size:16px;border:0;border-radius:12px;background:#1c2128;color:#e8e8e8}
.modes button.on{background:#2d6cdf;color:#fff}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;max-width:300px;margin:0 auto;padding:10px}
.pad button{padding:22px 0;font-size:24px;border:0;border-radius:14px;background:#1c2128;color:#e8e8e8}
.pad .stop{background:#7a1f1f;color:#fff}
.pad .blank{visibility:hidden}
.sentence{margin:14px 10px;padding:10px;background:#11151a;border-radius:10px;min-height:22px;color:#6cf;word-break:break-all}
.note{font-size:12px;color:#778;margin:8px}
</style></head><body>
<h2>Ishara</h2>
<img id="cam" src="/video" style="width:100%;max-width:460px;border-radius:10px;background:#000;display:block;margin:4px auto"/>
<div class="bar">mode: <b id="mode">-</b> &nbsp; dist: <b id="dist">-</b> cm</div>
<div class="modes">
  <button id="m_idle" onclick="mode('idle')">Idle</button>
  <button id="m_recognition" onclick="mode('recognition')">Recognition</button>
  <button id="m_following" onclick="mode('following')">Following</button>
  <button id="m_manual" onclick="mode('manual')">Manual</button>
</div>
<div class="pad">
  <span class="blank"></span><button id="bF">&#9650;</button><span class="blank"></span>
  <button id="bL">&#9664;</button><button class="stop" id="bS">STOP</button><button id="bR">&#9654;</button>
  <span class="blank"></span><button id="bB">&#9660;</button><span class="blank"></span>
</div>
<div class="sentence" id="sent">-</div>
<div class="note">Tap <b>Manual</b> first, then hold the arrows to drive.</div>
<script>
function mode(m){fetch('/mode?m='+m).catch(()=>{});}
function send(c){fetch('/drive?c='+c).catch(()=>{});}
function hold(id,c){var b=document.getElementById(id);
 var t=null;
 var dn=function(e){e.preventDefault();send(c);t=setInterval(function(){send(c);},150);};
 var up=function(e){e.preventDefault();if(t){clearInterval(t);t=null;}send('S');};
 b.addEventListener('pointerdown',dn);b.addEventListener('pointerup',up);
 b.addEventListener('pointerleave',up);b.addEventListener('pointercancel',up);}
hold('bF','F');hold('bB','B');hold('bL','L');hold('bR','R');
document.getElementById('bS').addEventListener('click',function(){send('S');});
setInterval(async function(){try{var r=await fetch('/status');var s=await r.json();
 document.getElementById('mode').textContent=s.mode;
 document.getElementById('dist').textContent=s.dist;
 document.getElementById('sent').textContent=s.sentence||'-';
 ['idle','recognition','following','manual'].forEach(function(m){
   document.getElementById('m_'+m).className=(s.mode===m?'on':'');});
}catch(e){}},500);
</script></body></html>"""

class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)
    def do_GET(self):
        try:
            u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
            if u.path == "/":
                self._send(200, "text/html", PAGE)
            elif u.path == "/status":
                self._send(200, "application/json", json.dumps({
                    "mode": shared["mode"], "word": shared["last_word"],
                    "sentence": sent_ascii(), "dist": round(robot.distance(), 1)}))
            elif u.path == "/mode":
                m = q.get("m", [""])[0]
                if m in ("idle", "recognition", "following", "manual"): set_mode(m)
                self._send(200, "application/json", json.dumps({"mode": shared["mode"]}))
            elif u.path == "/drive":
                do_drive(q.get("c", [""])[0])
                self._send(200, "application/json", json.dumps({"mode": shared["mode"]}))
            elif u.path == "/video":
                # live MJPEG stream of the annotated camera frame (writes to the HTTP
                # socket, never to the serial port, so no serial lock is needed)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Connection", "close")
                self.end_headers()
                while shared["run"]:
                    f = shared["frame"]
                    if f is None:
                        time.sleep(0.05); continue
                    ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                    if not ok:
                        time.sleep(0.05); continue
                    data = jpg.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(data)).encode())
                    self.wfile.write(data); self.wfile.write(b"\r\n")
                    time.sleep(0.08)   # ~12 fps cap; lower quality/fps if the Pi struggles
            else:
                self._send(404, "text/plain", "not found")
        except Exception:
            pass
    def log_message(self, *a): pass

class Srv(socketserver.ThreadingTCPServer):
    allow_reuse_address = True; daemon_threads = True

def start_server():
    try:
        Srv(("0.0.0.0", PHONE_PORT), Handler).serve_forever()
    except Exception as e:
        print("phone server not started:", e)

# ============================================================
#  START
# ============================================================
robot.connect()
threading.Thread(target=capture_thread, daemon=True).start()

print("priming buffer (~9s)...")
t0 = time.time()
while shared["run"]:
    with buf_lock: n = len(shared["buf"])
    if n >= N_FRAMES: print("buffer primed"); break
    if time.time() - t0 > 30: print("WARN: camera not filling buffer"); break
    time.sleep(0.3)

threading.Thread(target=head_track_thread, daemon=True).start()
threading.Thread(target=interaction_loop, daemon=True).start()
threading.Thread(target=manual_watchdog, daemon=True).start()
threading.Thread(target=start_server, daemon=True).start()
robot.oled(top="IDLE|RIGHT x1 = recognize|RIGHT x2 = follow", bottom="")

print("\nphone/browser control: http://<this-pi-ip>:%d" % PHONE_PORT)
print("gestures: idle RIGHT x1=recog RIGHT x2=follow | recog RIGHT=sign LEFT=del LEFTx2=idle BOTH=say")
print("keyboard (click window): i/r/f/m, q=quit; manual W/A/S/D\n")

# ============================================================
#  MAIN LOOP (display + keyboard)
# ============================================================
gui = True
try:
    while shared["run"]:
        key = 255
        if gui:
            try:
                f = shared["frame"]
                if f is not None: cv2.imshow("Robot", f)
                key = cv2.waitKey(30) & 0xFF
            except cv2.error:
                gui = False
                print("no display - running HEADLESS (control from phone at :%d, Ctrl+C to quit)" % PHONE_PORT)
        else:
            time.sleep(0.03)
        if key == ord('q'): shared["run"] = False
        elif key == ord('i'): set_mode("idle")
        elif key == ord('r'): set_mode("recognition")
        elif key == ord('f'): set_mode("following")
        elif key == ord('m'): set_mode("manual")
        if shared["mode"] == "manual":
            if key == ord('w'): do_drive("F")
            elif key == ord('s'): do_drive("B")
            elif key == ord('a'): do_drive("L")
            elif key == ord('d'): do_drive("R")
            elif key == ord('x') or key == ord(' '): do_drive("S")
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    shared["run"] = False; robot.stop()
    robot.oled(top="STOPPED", bottom=""); _park(); time.sleep(0.6)
    try: cv2.destroyAllWindows()
    except cv2.error: pass
    print("robot stopped")
