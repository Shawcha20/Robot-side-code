import os, time, numpy as np, cv2, torch, torch.nn as nn, threading, subprocess
from collections import deque, Counter

os.environ["PYTHONUTF8"] = "1"
torch.set_num_threads(4)

BASE = "/home/hudai/Desktop/thesis"
CKPT = os.path.join(BASE, "checkpoints")
ASSETS = os.path.join(BASE, "assets")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
AUDIO_DIR = os.path.join(BASE, "audio")
DEVICE = "cpu"

POSE_N, HAND_N = 33, 21
N_LM = POSE_N + 2 * HAND_N
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM * 3 + N_LM
N_FRAMES = 64

# --- tunable behavior ---
CONF_TH = 0.55
VOTE_NEED = 4
RECOG_WINDOW = 12.0       # seconds to perform one sign
NAV_COOLDOWN = 3.0        # pause after a nav gesture (lower your hands)
HAND_RECENT = 0.5         # a hand counts as open if seen open within this many seconds

# ==================== MODEL ====================
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

# ==================== LABELS ====================
import openpyxl
ws = openpyxl.load_workbook(WORD_XLSX).active
ID2NAME = {int(r[0]): str(r[1]).strip()
           for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}
def name(i): return ID2NAME.get(int(i), "class_" + str(i))

# ==================== LOAD ENSEMBLE ====================
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

# ==================== MEDIAPIPE ====================
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
    if abs(hand[4].x - hand[2].x) > 0.05:
        f += 1
    return f

def detect_one(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr = pose_lm.detect(img); hr = hand_lm.detect(img)
    c = np.zeros((N_LM, 3), np.float32); m = np.zeros(N_LM, np.float32)
    left_open = False; right_open = False
    if pr.pose_landmarks:
        c[:POSE_N] = _np(pr.pose_landmarks[0]); m[:POSE_N] = 1
    if hr.hand_landmarks:
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            a = _np(lms)
            label = hd[0].category_name
            is_open = fingers_up(lms) >= 5
            if label == "Left":
                c[POSE_N:POSE_N + HAND_N] = a; m[POSE_N:POSE_N + HAND_N] = 1
                if is_open: left_open = True
            else:
                c[POSE_N + HAND_N:] = a; m[POSE_N + HAND_N:] = 1
                if is_open: right_open = True
    return c, m, left_open, right_open, pr, hr

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

# ==================== AUDIO (paplay -> Bluetooth) ====================
SILENCE_DONE = {"awake": 0.0}

def _wake_speaker():
    """If the speaker has been idle, play a brief silence to wake it."""
    now = time.time()
    if now - SILENCE_DONE["awake"] > 8.0:   # idle for >8s -> needs waking
        sil = os.path.join(AUDIO_DIR, "_silence.wav")
        if os.path.exists(sil):
            subprocess.run(["paplay", sil], check=False)
        else:
            time.sleep(0.1)
    SILENCE_DONE["awake"] = time.time()

def _play(path):
    if os.path.exists(path):
        subprocess.run(["paplay", path], check=False)
        SILENCE_DONE["awake"] = time.time()

def speak_word(class_id):
    def _run():
        _wake_speaker()
        _play(os.path.join(AUDIO_DIR, str(class_id) + ".wav"))
    threading.Thread(target=_run, daemon=True).start()

def speak_sentence(ids):
    def _run():
        _wake_speaker()
        for cid in ids:
            _play(os.path.join(AUDIO_DIR, str(cid) + ".wav"))
            time.sleep(0.15)   # small gap between words
    threading.Thread(target=_run, daemon=True).start()

# ==================== CONTINUOUS CAPTURE ====================
shared = {"buf": deque(maxlen=N_FRAMES), "run": True, "frame": None,
          "l_last": 0.0, "r_last": 0.0}
buf_lock = threading.Lock()

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]

def capture_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: capture thread could not open camera")
        shared["run"] = False; return
    print("camera opened OK (in capture thread)")
    while shared["run"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c, m, l_open, r_open, pr, hr = detect_one(rgb)

        now = time.time()
        if l_open:
            shared["l_last"] = now
        if r_open:
            shared["r_last"] = now

        feat = to_feat(c, m)
        with buf_lock:
            shared["buf"].append(feat)

        # draw skeletons
        if pr.pose_landmarks:
            pts = [(int(p.x*w), int(p.y*h)) for p in pr.pose_landmarks[0]]
            for a, b in POSE_CONN:
                cv2.line(frame, pts[a], pts[b], (255,150,0), 2)
            nx, ny = pts[0]
            cv2.rectangle(frame, (nx-55, ny-55), (nx+55, ny+55), (0,255,255), 2)
        if hr.hand_landmarks:
            for hand in hr.hand_landmarks:
                hp = [(int(p.x*w), int(p.y*h)) for p in hand]
                for a, b in HAND_CONN:
                    cv2.line(frame, hp[a], hp[b], (0,255,0), 2)
                for p in hp:
                    cv2.circle(frame, p, 3, (0,0,255), -1)

        # recent-open state for the on-screen label
        l_disp = (now - shared["l_last"]) < HAND_RECENT
        r_disp = (now - shared["r_last"]) < HAND_RECENT
        tag = ""
        if l_disp and r_disp: tag = "BOTH HANDS (stop)"
        elif r_disp: tag = "RIGHT HAND (start)"
        elif l_disp: tag = "LEFT HAND (delete)"
        if tag:
            cv2.putText(frame, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        shared["frame"] = frame
    cap.release()

threading.Thread(target=capture_thread, daemon=True).start()

print("priming buffer (about 9s)...")
t_prime = time.time()
while True:
    with buf_lock:
        filled = len(shared["buf"])
    if filled >= N_FRAMES:
        print("buffer primed (", filled, "frames in", round(time.time() - t_prime, 1), "s)"); break
    if time.time() - t_prime > 30:
        print("WARNING: buffer not filling, camera issue?"); break
    time.sleep(0.3)

def current_prediction():
    with buf_lock:
        if len(shared["buf"]) < N_FRAMES:
            return None, 0.0
        b = list(shared["buf"])
    return predict(b)

def hand_state():
    now = time.time()
    left = (now - shared["l_last"]) < HAND_RECENT
    right = (now - shared["r_last"]) < HAND_RECENT
    return left, right

def clear_buffer():
    with buf_lock:
        shared["buf"].clear()

# ==================== RECOGNIZE one sign ====================
def collect_one_sign():
    clear_buffer()
    votes = deque(maxlen=9)
    t0 = time.time()
    last_print = 0
    print("  [sign now]")
    while time.time() - t0 < RECOG_WINDOW and shared["run"]:
        pid, conf = current_prediction()
        l, r = hand_state()
        now = time.time()
        if now - last_print > 0.5:
            if pid is None:
                print("    buffer filling...")
            else:
                print("    pred", name(pid), "conf", round(conf, 2))
            last_print = now
        # skip frames where an open palm shows (that's a control pose)
        if pid is not None and conf >= CONF_TH and not (l or r):
            votes.append(pid)
            if len(votes) >= 5:
                v, ct = Counter(votes).most_common(1)[0]
                if ct >= VOTE_NEED:
                    print("  [LOCKED]", name(v), "(conf", round(conf, 2), ")")
                    return v
        time.sleep(0.05)
    print("  no stable sign")
    return None

def wait_hands_down():
    while shared["run"]:
        l, r = hand_state()
        if not l and not r:
            return
        time.sleep(0.05)

# ==================== MAIN STATE MACHINE ====================
sentence = []
def show(ids):
    return " ".join(name(s) for s in ids)

def read_gesture():
    """Wait for hands to come up, track the strongest gesture shown,
    then act when hands go down. Returns 'start', 'delete', 'end', or None."""
    # wait until some hand is up
    while shared["run"]:
        l, r = hand_state()
        if l or r:
            break
        time.sleep(0.05)
    if not shared["run"]:
        return None

    # hands are up  track the strongest gesture until they go down
    saw_both = False
    saw_left = False
    saw_right = False
    last_up = time.time()
    while shared["run"]:
        l, r = hand_state()
        now = time.time()
        if l and r:
            saw_both = True; last_up = now
        elif l:
            saw_left = True; last_up = now
        elif r:
            saw_right = True; last_up = now
        else:
            # no hand seen right now; if down long enough, finish
            if now - last_up > 0.4:
                break
        time.sleep(0.05)

    # decide: both takes priority, then which single hand
    if saw_both:
        return "end"
    elif saw_left and not saw_right:
        return "delete"
    elif saw_right and not saw_left:
        return "start"
    elif saw_right and saw_left:
        # both hands seen but never simultaneously -> treat as end
        return "end"
    return None

def interaction_loop():
    print("\nRIGHT hand = START (then sign) | LEFT hand = DELETE | BOTH hands = END")
    print("hold the gesture, then LOWER your hands to confirm it\n")
    while shared["run"]:
        g = read_gesture()
        if not shared["run"] or g is None:
            continue

        if g == "end":
            print("[END]", show(sentence))
            if sentence:
                speak_sentence(list(sentence))   # pass a COPY
                sentence.clear()
            time.sleep(NAV_COOLDOWN)
            wait_hands_down()

        elif g == "start":
            print("[START]")
            time.sleep(NAV_COOLDOWN)
            wait_hands_down()
            sid = collect_one_sign()
            if sid is not None:
                sentence.append(sid)
                print("[WORD ADDED]", name(sid), "| sentence:", show(sentence))
                speak_word(sid)
            else:
                print("[no sign detected]")
            wait_hands_down()

        elif g == "delete":
            if sentence:
                removed = sentence.pop()
                print("[DELETED]", name(removed), "| sentence:", show(sentence))
                speak_word(removed)
            else:
                print("[nothing to delete]")
            time.sleep(NAV_COOLDOWN)
            wait_hands_down()

threading.Thread(target=interaction_loop, daemon=True).start()

try:
    while shared["run"]:
        f = shared["frame"]
        if f is not None:
            cv2.imshow("Robot Camera View", f)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    shared["run"] = False
    time.sleep(0.7)
    cv2.destroyAllWindows()
    print("robot stopped (camera released)")
