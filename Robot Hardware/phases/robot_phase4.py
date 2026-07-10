#!/usr/bin/env python3
"""
PHASE 4 - integrated robot, SELF-CONTAINED (one file, nothing imported).

Talks to the STM32 directly over USB serial with the protocol we built and
tested in Phase 3:  Pi sends 'F'/'B'/'L'/'R'/'S';  STM32 streams 'D<distance>'.
(If your STM32 firmware uses different letters, change the CMD_* constants below.)

MODE SWITCHING IS BY HAND GESTURE (roadmap step 9):
  IDLE:        RIGHT hand x1 -> recognition ;  RIGHT hand x2 -> following
               (manual is entered from the phone app in Phase 5)
  RECOGNITION: RIGHT = capture a sign | LEFT = delete | LEFT x2 = back to idle
               BOTH  = speak the sentence
               head-tracking runs here only, frozen during a capture
  FOLLOWING:   drives toward the person, sonar stop ; LEFT hand = back to idle
  MANUAL:      phone app (Phase 5); locally W/A/S/D, X/space = stop

Keyboard fallback (click the window first): i/r/f/m switch modes, q quits.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"     # so the OpenCV window gets keys under Wayland
os.environ["PYTHONUTF8"] = "1"

import time, threading, subprocess
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

# ---- STM32 serial link (the protocol we tested in Phase 3) ----
STM_PORT, STM_BAUD = "/dev/ttyACM0", 115200
CMD_FWD, CMD_BACK, CMD_LEFT, CMD_RIGHT, CMD_STOP = "F", "B", "L", "R", "S"
TELE_PREFIX = "D"            # STM32 sends "D<distance_cm>" lines

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
DOUBLE_WINDOW = 2.0          # window for a 2nd RIGHT hand (idle -> following)
LEFT_DOUBLE_WINDOW = 1.0     # window for a 2nd LEFT hand (recognition -> idle)

CENTER_DEAD = 0.12
STOP_DIST = 40
FOLLOW_DIST = 70
AUTO_STOP_AFTER = 0.4

HT_DEAD = 0.05; HT_KP = 11.0; HT_KD = 2.0; HT_SMOOTH = 0.6
HT_MAX_STEP = 2.5; HT_MIN_MOVE = 0.6; HT_SETTLE = 0.04; HT_CENTER = 135.0
PAN_PIN, TILT_PIN = 22, 23
PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = 30, 240, 60, 210
PAN_DIR, TILT_DIR = -1, 1

def clamp(v, lo, hi): return max(lo, min(hi, v))

# ============================================================
#  STM32 SERIAL LINK (built in -- no external file)
# ============================================================
class Robot:
    def __init__(self):
        self.ser = None; self.ok = False; self._dist = -1.0; self._buf = ""; self._last = ""
    def connect(self):
        try:
            import serial
            self.ser = serial.Serial(STM_PORT, STM_BAUD, timeout=0.1)
            time.sleep(2.0)                      # let the STM32 reset/settle
            self.ok = True
            print("STM32 serial connected on", STM_PORT)
            threading.Thread(target=self._reader, daemon=True).start()
        except Exception as e:
            print("STM32 NOT connected:", e, "- motors disabled, recognition still runs")
            self.ok = False
    def _reader(self):
        while shared["run"] and self.ok:
            try:
                data = self.ser.read(64).decode(errors="ignore")
            except Exception:
                break
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line.startswith(TELE_PREFIX):
                    try: self._dist = float(line[1:])
                    except ValueError: pass
    def _send(self, ch):
        if not self.ok: return
        if ch == self._last: return              # don't spam identical commands
        try: self.ser.write(ch.encode()); self._last = ch
        except Exception: pass
    def forward(self):  self._send(CMD_FWD)
    def backward(self): self._send(CMD_BACK)
    def left(self):     self._send(CMD_LEFT)
    def right(self):    self._send(CMD_RIGHT)
    def stop(self):     self._send(CMD_STOP)
    def distance(self): return self._dist

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
          "person_cx": None, "nose_x": 0.5, "nose_y": 0.5, "pose_found": False,
          "last_word": "", "hint": ""}
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
        with buf_lock: shared["buf"].append(to_feat(c, m))

        if pr.pose_landmarks:
            nose = pr.pose_landmarks[0][0]
            shared["nose_x"], shared["nose_y"] = float(nose.x), float(nose.y)
            shared["pose_found"] = True
            if m[L_SHO] > 0 and m[R_SHO] > 0:
                shared["person_cx"] = float((c[L_SHO][0] + c[R_SHO][0]) / 2.0)
            else:
                shared["person_cx"] = None
        else:
            shared["pose_found"] = False; shared["person_cx"] = None

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
        if shared["last_word"]:
            cv2.putText(frame, "word: " + shared["last_word"], (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
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
    HEAD_OK = True; print("head-tracking servos ready")
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

def wait_hands_down():
    while shared["run"]:
        l, r = hand_state()
        if not l and not r: return
        time.sleep(0.05)

def read_raw_gesture(expect_mode, timeout=None):
    """Wait for a hand up, see which (left/right/both), act on hands-down.
    Returns 'right'|'left'|'both'|None. Bails if mode changes or (optional) timeout."""
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
def show(ids): return " ".join(name(s) for s in ids)

def set_mode(new):
    if new == shared["mode"]: return
    robot.stop(); shared["mode"] = new; print("MODE ->", new)

def collect_one_sign():
    shared["collecting"] = True
    clear_buffer(); votes = deque(maxlen=9); t0 = time.time(); lastp = 0; result = None
    print("  [sign now]")
    while time.time() - t0 < RECOG_WINDOW and shared["run"]:
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
    shared["hint"] = "RIGHT hand x1 = recognition, x2 = following"
    g = read_raw_gesture("idle")
    if g != "right": return
    print("[idle] 1st right hand - waiting for a 2nd...")
    second = False; t0 = time.time()
    while time.time() - t0 < DOUBLE_WINDOW and shared["run"] and shared["mode"] == "idle":
        g2 = read_raw_gesture("idle", timeout=DOUBLE_WINDOW - (time.time() - t0))
        if g2 is None: break
        if g2 == "right": second = True; break
    if shared["mode"] == "idle":
        set_mode("following" if second else "recognition"); wait_hands_down()

def recognition_step():
    shared["hint"] = "RIGHT=sign  LEFT=delete  LEFTx2=idle  BOTH=speak"
    g = read_raw_gesture("recognition")
    if g is None: return
    if g == "end":
        print("[END]", show(sentence))
        if sentence: speak_sentence(list(sentence)); sentence.clear()
        shared["last_word"] = ""
        time.sleep(NAV_COOLDOWN); wait_hands_down()
    elif g == "start":
        print("[START]")
        time.sleep(NAV_COOLDOWN); wait_hands_down()
        sid = collect_one_sign()
        if sid is not None:
            sentence.append(sid); shared["last_word"] = name(sid)
            print("[WORD]", name(sid), "|", show(sentence)); speak_word(sid)
        wait_hands_down()
    elif g == "left":
        # single LEFT = delete ; LEFT x2 (a 2nd left within the window) = back to idle
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
            time.sleep(NAV_COOLDOWN); wait_hands_down()

def following_step():
    shared["hint"] = "following... LEFT hand = back to idle"
    l, r = hand_state()
    if l:
        print("[following] left hand -> idle"); robot.stop(); set_mode("idle"); wait_hands_down(); return
    cx = shared["person_cx"]; dist = robot.distance()
    if cx is None: robot.stop()
    elif 0 < dist < STOP_DIST: robot.stop()
    else:
        off = cx - 0.5
        if abs(off) > CENTER_DEAD: robot.left() if off < 0 else robot.right()
        elif dist < 0 or dist > FOLLOW_DIST: robot.forward()
        else: robot.stop()
    time.sleep(0.05)

def interaction_loop():
    while shared["run"]:
        mode = shared["mode"]
        if mode == "idle": idle_step()
        elif mode == "recognition": recognition_step()
        elif mode == "following": following_step()
        else: time.sleep(0.05)

# ============================================================
#  START
# ============================================================
robot.connect()
threading.Thread(target=capture_thread, daemon=True).start()

print("priming buffer (~9s)...")
t0 = time.time()
while True:
    with buf_lock: n = len(shared["buf"])
    if n >= N_FRAMES: print("buffer primed"); break
    if time.time() - t0 > 30: print("WARN: camera not filling buffer"); break
    time.sleep(0.3)

threading.Thread(target=head_track_thread, daemon=True).start()
threading.Thread(target=interaction_loop, daemon=True).start()

print("\nGESTURES: idle -> RIGHT x1 = recognition, RIGHT x2 = following")
print("recognition: RIGHT=sign LEFT=delete LEFTx2=idle BOTH=speak | following: LEFT=idle")
print("keyboard fallback (click window): i/r/f/m, q=quit; manual W/A/S/D\n")

# ============================================================
#  MAIN LOOP (display + keyboard fallback + manual driving)
# ============================================================
last_manual = 0.0; manual_moving = False
try:
    while shared["run"]:
        f = shared["frame"]
        if f is not None: cv2.imshow("Robot", f)
        key = cv2.waitKey(30) & 0xFF
        now = time.time()
        if key == ord('q'): shared["run"] = False
        elif key == ord('i'): set_mode("idle")
        elif key == ord('r'): set_mode("recognition")
        elif key == ord('f'): set_mode("following")
        elif key == ord('m'): set_mode("manual")
        if shared["mode"] == "manual":
            if key == ord('w'): robot.forward(); last_manual = now; manual_moving = True
            elif key == ord('s'): robot.backward(); last_manual = now; manual_moving = True
            elif key == ord('a'): robot.left(); last_manual = now; manual_moving = True
            elif key == ord('d'): robot.right(); last_manual = now; manual_moving = True
            elif key == ord('x') or key == ord(' '): robot.stop(); manual_moving = False
            if manual_moving and now - last_manual > AUTO_STOP_AFTER:
                robot.stop(); manual_moving = False
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    shared["run"] = False; robot.stop(); _park(); time.sleep(0.6)
    cv2.destroyAllWindows(); print("robot stopped")
