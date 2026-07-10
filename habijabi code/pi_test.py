import os, time, numpy as np, cv2, torch, torch.nn as nn
from collections import deque, Counter

BASE = "/home/hudai/Desktop/thesis"
CKPT = os.path.join(BASE, "checkpoints")
ASSETS = os.path.join(BASE, "assets")
WORD_XLSX = os.path.join(BASE, "Word Label.xlsx")

DEVICE = "cpu"   # Pi runs on CPU
POSE_N, HAND_N = 33, 21; N_LM = POSE_N + 2*HAND_N
L_SHO, R_SHO = 11, 12
FEATURE_DIM = N_LM*3 + N_LM
N_FRAMES = 64

class SpatialEncoder(nn.Module):
    def __init__(s, d_in, d=128, p=0.3):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d_in,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(p),
                            nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
    def forward(s,x): return s.net(x)
class AttnPool(nn.Module):
    def __init__(s,d): super().__init__(); s.q=nn.Linear(d,1)
    def forward(s,x):
        w=torch.softmax(s.q(x).squeeze(-1),1).unsqueeze(-1); return (x*w).sum(1)
class SignTransformer(nn.Module):
    def __init__(s,d_in,n,d=128,heads=4,layers=4,ff=4,p=0.3):
        super().__init__()
        s.encoder=SpatialEncoder(d_in,d,p)
        s.pos=nn.Parameter(torch.zeros(1,512,d))
        L=nn.TransformerEncoderLayer(d,heads,d*ff,p,batch_first=True,activation="gelu")
        s.tf=nn.TransformerEncoder(L,layers); s.pool=AttnPool(d)
        s.head=nn.Sequential(nn.LayerNorm(d),nn.Dropout(p),nn.Linear(d,n))
    def forward(s,x):
        t=s.encoder(x); t=t+s.pos[:,:t.shape[1]]
        return s.head(s.pool(s.tf(t)))

ID2NAME={}
try:
    import openpyxl
    ws=openpyxl.load_workbook(WORD_XLSX).active
    ID2NAME={int(r[0]):str(r[1]).strip() for r in ws.iter_rows(min_row=2,values_only=True) if r[1] is not None}
except Exception as e: print("(names not loaded)",e)
def name(i): return ID2NAME.get(int(i),f"class_{i}")

files=["word_model_full.pt","word_model_128_s2.pt","word_model_128_s3.pt","word_model_128_s4.pt","word_model_128_s5.pt"]
first=torch.load(os.path.join(CKPT,files[0]),map_location=DEVICE,weights_only=True)
N_CLASSES=first["head.2.weight"].shape[0]
MODELS=[]
for fn in files:
    m=SignTransformer(FEATURE_DIM,N_CLASSES).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT,fn),map_location=DEVICE,weights_only=True)); m.eval()
    MODELS.append(m)
print(f"loaded {len(MODELS)} models | {N_CLASSES} classes")

import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
pose_lm=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS,"pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.IMAGE,num_poses=1))
hand_lm=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS,"hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE,num_hands=2))
def _np(lms): return np.array([[p.x,p.y,p.z] for p in lms],np.float32)
def detect_one(rgb):
    img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    pr,hr=pose_lm.detect(img),hand_lm.detect(img)
    c=np.zeros((N_LM,3),np.float32); m=np.zeros(N_LM,np.float32)
    if pr.pose_landmarks: c[:POSE_N]=_np(pr.pose_landmarks[0]); m[:POSE_N]=1
    if hr.hand_landmarks:
        for lms,hd in zip(hr.hand_landmarks,hr.handedness):
            a=_np(lms)
            if hd[0].category_name=="Left": c[POSE_N:POSE_N+HAND_N]=a; m[POSE_N:POSE_N+HAND_N]=1
            else: c[POSE_N+HAND_N:]=a; m[POSE_N+HAND_N:]=1
    return c,m
def norm_frame(c,m):
    if m[L_SHO]>0 and m[R_SHO]>0:
        o=(c[L_SHO]+c[R_SHO])/2; s=np.linalg.norm(c[L_SHO,:2]-c[R_SHO,:2])+1e-6
    else:
        pr=c[m>0]
        if len(pr)==0: return c
        o=pr.mean(0); s=pr[:,:2].std()+1e-6
    out=c.copy(); out[m>0]=(c[m>0]-o)/s; return out

def to_feat(c,m): return np.concatenate([norm_frame(c,m).reshape(-1),m]).astype("float32")

@torch.no_grad()
def predict(buf):
    x=torch.from_numpy(np.stack(buf)[None]).float()
    p=sum(torch.softmax(m(x),1) for m in MODELS)/len(MODELS)
    p=p[0].numpy(); return int(p.argmax()),float(p.max())
cap=cv2.VideoCapture(0)
buf=deque(maxlen=N_FRAMES); recent=deque(maxlen=7)
last=None; since=999; prev=None
fps_t=time.time(); fps_n=0
print("running - sign with hands up. ctrl+C to stop")
try:
    while True:
        ok,frame=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        c,m=detect_one(rgb); buf.append(to_feat(c,m))
        wrist=c[16,:2]
        motion=0.0 if prev is None else float(np.linalg.norm(wrist-prev)); prev=wrist; since+=1
        fps_n+=1
        if time.time()-fps_t>=2.0:
            print(f"[fps: {fps_n/(time.time()-fps_t):.1f}]"); fps_t=time.time(); fps_n=0
        if len(buf)==N_FRAMES and motion>0.015:
            pid,conf=predict(buf)
            if conf>=0.5:
                recent.append(pid)
                v,ct=Counter(recent).most_common(1)[0]
                if ct>=4 and (v!=last or since>=15):
                    last=v; since=0; print(f"  WORD: {name(v)} ({conf:.2f})")
except KeyboardInterrupt:
    pass
cap.release()
print("stopped")









