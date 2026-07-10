import os, time, numpy as np, cv2, torch, torch.nn as nn, threading, subprocess
from collections import deque, Counter
from gtts import gTTS

os.environ["PYTHONUTF8"] = "1"
BASE = "/home/hudai/Desktop/thesis"
CKPT = os.path.join(BASE, "checkpoints")
ASSETS = os.path.join(BASE, "assets")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")
DEVICE = "cpu"

POSE_N, HAND_N = 33, 21
N_LM = POSE_N + 2 * HAND_N
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM * 3 + N_LM
N_FRAMES = 64

CONTROL_SIGN_ID = 78
CONF_TH = 0.55
RECOG_WINDOW = 10.0
COUNT_WINDOW = 2.0
MOTION_TH = 0.015

# MODEL
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

# LABELS
import openpyxl
ws = openpyxl.load_workbook(WORD_XLSX).active
ID2NAME = {int(r[0]): str(r[1]).strip()
           for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None}
def name(i): return ID2NAME.get(int(i), "class_" + str(i))

# LOAD ENSEMBLE
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

# MEDIAPIPE
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
def detect_one(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr = pose_lm.detect(img); hr = hand_lm.detect(img)
    c = np.zeros((N_LM, 3), np.float32); m = np.zeros(N_LM, np.float32)
    if pr.pose_landmarks:
        c[:POSE_N] = _np(pr.pose_landmarks[0]); m[:POSE_N] = 1
    if hr.hand_landmarks:
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            a = _np(lms)
            if hd[0].category_name == "Left":
                c[POSE_N:POSE_N + HAND_N] = a; m[POSE_N:POSE_N + HAND_N] = 1
            else:
                c[POSE_N + HAND_N:] = a; m[POSE_N + HAND_N:] = 1
    return c, m

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

# SPEECH (live gtts, needs internet)
def speak(text):
    def _run():
        try:
            path = "/tmp/say.mp3"
            gTTS(str(text), lang="bn").save(path)
            subprocess.run(["mpg123", "-q", path], check=False)
        except Exception as e:
            print("[tts error]", e)
    threading.Thread(target=_run, daemon=True).start()

# CONTINUOUS CAPTURE
shared = {"buf": deque(maxlen=N_FRAMES), "motion": 0.0, "prev": None, "run": True}
buf_lock = threading.Lock()

def get_motion(c, m, prev):
    if m[16] > 0:
        wrist = c[16, :2]
        motion = 0.0 if prev is None else float(np.linalg.norm(wrist - prev))
        return motion, wrist
    return 0.0, prev

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c, m = detect_one(rgb)
        feat = to_feat(c, m)
        mo, shared["prev"] = get_motion(c, m, shared["prev"])
        with buf_lock:
            shared["buf"].append(feat)
            shared["motion"] = mo
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
        print("WARNING: buffer only", filled, "of", N_FRAMES, "after 30s, camera issue?"); break
    time.sleep(0.3)

def current_prediction():
    with buf_lock:
        if len(shared["buf"]) < N_FRAMES:
            return None, 0.0, 0.0
        b = list(shared["buf"]); mo = shared["motion"]
    pid, conf = predict(b)
    return pid, conf, mo

def clear_buffer():
    with buf_lock:
        shared["buf"].clear()

# INTERACTION LOGIC
def collect_one_sign():
    votes = deque(maxlen=9)
    t0 = time.time()
    print("  [sign now]")
    while time.time() - t0 < RECOG_WINDOW:
        pid, conf, mo = current_prediction()
        if pid is not None and conf >= CONF_TH and pid != CONTROL_SIGN_ID and mo > MOTION_TH:
            votes.append(pid)
            if len(votes) >= 5:
                v, ct = Counter(votes).most_common(1)[0]
                if ct >= 4:
                    print("  [LOCKED]", name(v), "(conf", round(conf, 2), ")")
                    return v
        time.sleep(0.05)
    print("  no stable sign")
    return None

def wait_for_control_count():
    count = 0
    t_last_seen = 0
    t0 = time.time()
    seen_now = False
    while True:
        pid, conf, mo = current_prediction()
        now = time.time()
        is_pesha = (pid is not None and conf >= CONF_TH and pid == CONTROL_SIGN_ID)
        if is_pesha and not seen_now:
            count += 1
            seen_now = True
            t_last_seen = now
            t0 = now
            print("pesha detected (", count, ")")
            clear_buffer()
            time.sleep(0.3)
        elif is_pesha:
            t_last_seen = now
        else:
            if seen_now and now - t_last_seen > 0.6:
                seen_now = False
        if count > 0 and not seen_now and now - t_last_seen > COUNT_WINDOW:
            break
        if count == 0:
            t0 = now
        time.sleep(0.05)
    clear_buffer()
    return count

# MAIN
sentence = []
print("")
print("pesha 1x  = START then perform one sign")
print("pesha 2x  = DELETE last word")
print("pesha 3x  = END and speak full sentence")
print("(lower hands briefly between each pesha repeat)")
print("")

def show(ids):
    return " ".join(name(s) for s in ids)

try:
    while True:
        n = wait_for_control_count()

        if n == 1:
            print("[STARTED]")
            sid = collect_one_sign()
            if sid is not None:
                sentence.append(sid)
                print("[WORD ADDED]", name(sid), "| sentence:", show(sentence))
                speak(name(sid))
            else:
                print("[no sign detected]")

        elif n == 2:
            if sentence:
                removed = sentence.pop()
                print("[DELETED]", name(removed), "| sentence:", show(sentence))
            else:
                print("[nothing to delete]")

        elif n >= 3:
            print("[SENTENCE ENDED]", show(sentence))
            if sentence:
                speak(" ".join(name(s) for s in sentence))
            print("stopped, pesha 1x to start a new sentence")
            print("")

except KeyboardInterrupt:
    pass
finally:
    shared["run"] = False
    time.sleep(0.7)
    print("robot stopped (camera released)")
