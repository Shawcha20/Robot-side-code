#!/usr/bin/env python3
"""
Ishara — sign-recognition web app (reuses the REAL robot pipeline).

Imports your actual model.py + perception.py + config.py, so recognition is
byte-identical to what the robot does. No reconstruction, no guessing.

  browser: sends camera frames (JPEG) to the server
  server : perception.detect_one -> to_feat -> 64-frame buffer
           -> model.predict (your 5-seed ensemble) -> model.name(id)
           -> returns the word + draws landmarks back onto the frame

UI: live camera feed with landmarks overlaid, a Start button, and an output box
showing the recognized Bangla word.

RUN (on the Pi, or any machine with the model/assets present)
    pip install flask
    # (torch, mediapipe, numpy, openpyxl already needed by model.py/perception.py)
    python sign_web.py
    open http://<host>:8000   (PHONE_PORT from config)

NOTE
    This imports model.py at startup, which loads the 5 checkpoints and prints
    "loaded 5 models | N classes" -- same as the robot. If that line doesn't
    appear, the checkpoints/paths in config.py are wrong for this machine.
"""

import io
import time
import base64
import threading
from collections import deque

import numpy as np
from flask import Flask, request, jsonify, Response

# ---- import YOUR real pipeline (guarantees identical behavior) ----
from config import N_FRAMES, CONF_TH, PHONE_PORT, L_SHO, R_SHO, POSE_N, HAND_N
from perception import detect_one, to_feat
from model import predict, name

# Pillow for JPEG decode/encode + landmark drawing (no cv2 dependency needed)
from PIL import Image, ImageDraw

app = Flask(__name__)

# ---- recognition state (single active session; this is a 1-user tool) ----
_buf = deque(maxlen=N_FRAMES)
_lock = threading.Lock()
_collecting = False
_result = {"word": "—", "conf": 0.0, "id": -1, "frames": 0}


# hand-skeleton bone pairs (MediaPipe hand topology) for nicer overlay
_HAND_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),
               (18,19),(19,20),(0,17)]
# a few pose bones (shoulders/arms) worth drawing
_POSE_BONES = [(11,13),(13,15),(12,14),(14,16),(11,12)]


def _draw_overlay(img, c, m):
    """Draw pose + hand landmarks (normalized coords in c) onto a PIL image."""
    W, H = img.size
    d = ImageDraw.Draw(img)

    def pt(i):
        return (c[i][0] * W, c[i][1] * H)

    # pose bones
    for a, b in _POSE_BONES:
        if m[a] > 0 and m[b] > 0:
            d.line([pt(a), pt(b)], fill=(80, 200, 120), width=3)
    # hand bones (left: POSE_N.., right: POSE_N+HAND_N..)
    for base in (POSE_N, POSE_N + HAND_N):
        if m[base] > 0:
            for a, b in _HAND_BONES:
                ia, ib = base + a, base + b
                if m[ia] > 0 and m[ib] > 0:
                    d.line([pt(ia), pt(ib)], fill=(90, 160, 255), width=2)
    # landmark dots
    for i in range(len(c)):
        if m[i] > 0:
            x, y = pt(i)
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 220, 90))
    return img


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/start", methods=["POST"])
def start():
    global _collecting, _result
    with _lock:
        _buf.clear()
        _collecting = True
        _result = {"word": "…", "conf": 0.0, "id": -1, "frames": 0}
    return jsonify({"ok": True})


@app.route("/api/frame", methods=["POST"])
def frame():
    """Receive one camera frame (JPEG), run detection, overlay landmarks, and
    (while collecting) feed the buffer + predict when full. Returns the
    annotated frame + current result."""
    global _collecting, _result
    data = request.get_data()
    if not data:
        return jsonify({"error": "no frame"}), 400
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "bad image: %s" % e}), 400

    rgb = np.asarray(img, dtype=np.uint8)
    c, m, lo, ro, pr, hr = detect_one(rgb)

    hand_present = bool(m[POSE_N] > 0 or m[POSE_N + HAND_N] > 0)

    done = False
    with _lock:
        if _collecting and hand_present:
            _buf.append(to_feat(c, m))
            _result["frames"] = len(_buf)
            if len(_buf) >= N_FRAMES:
                b = list(_buf)
                _collecting = False
                done = True
        frames_now = _result["frames"]

    if done:
        pid, conf = predict(b)
        with _lock:
            _result = {"word": name(pid) if conf >= CONF_TH else "(low conf: %s)" % name(pid),
                       "conf": conf, "id": pid, "frames": N_FRAMES}

    # draw overlay for the live view
    _draw_overlay(img, c, m)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=70)
    annotated = base64.b64encode(out.getvalue()).decode("ascii")

    with _lock:
        res = dict(_result)
        collecting = _collecting
    return jsonify({"image": annotated, "result": res,
                    "collecting": collecting, "frames": frames_now,
                    "need": N_FRAMES, "hand": hand_present})


@app.route("/api/status")
def status():
    return jsonify({"n_frames": N_FRAMES, "conf_th": CONF_TH})


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ishara — Sign Recognition</title>
<style>
  :root{--bg:#0e1116;--card:#171b22;--fg:#e8ecf2;--mut:#aab3c0;--acc:#2d6cdf;--dan:#c0392b}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  header{padding:14px 18px;background:#0b0d10;border-bottom:1px solid #222}
  header b{font-size:18px}#status{color:var(--mut);font-size:13px;margin-left:10px}
  main{max-width:720px;margin:0 auto;padding:18px;display:grid;gap:16px}
  .card{background:var(--card);border:1px solid #232833;border-radius:14px;padding:16px}
  #view{width:100%;border-radius:10px;background:#000;display:block}
  .big{font-size:16px;padding:14px 18px;border:0;border-radius:12px;color:#fff;cursor:pointer;width:100%}
  .go{background:var(--acc)}.stop{background:var(--dan)}
  #word{font-size:42px;font-weight:700;text-align:center;padding:20px;min-height:46px}
  #conf{color:var(--mut);text-align:center;font-size:14px}
  #bar{height:8px;background:#0d1015;border-radius:5px;overflow:hidden;margin-top:10px}
  #barfill{height:100%;width:0;background:var(--acc);transition:width .08s}
  .note{color:var(--mut);font-size:13px}
</style></head><body>
<header><b>Ishara — Sign Recognition</b><span id="status"></span></header>
<main>
  <div class="card">
    <img id="view" alt="camera">
    <div id="bar"><div id="barfill"></div></div>
    <p class="note" id="hint">Click Start, then perform ONE sign with your hand in view.</p>
  </div>
  <div class="card"><button id="btn" class="big go">Start Recognizing</button></div>
  <div class="card">
    <div id="word">—</div>
    <div id="conf"></div>
  </div>
  <video id="cam" autoplay playsinline muted style="display:none"></video>
  <canvas id="cv" style="display:none"></canvas>
</main>
<script>
const btn=document.getElementById('btn'), view=document.getElementById('view');
const wordEl=document.getElementById('word'), confEl=document.getElementById('conf');
const barfill=document.getElementById('barfill'), statusEl=document.getElementById('status');
const cam=document.getElementById('cam'), cv=document.getElementById('cv');
let need=64, sending=false, collecting=false;

async function init(){
  const s=await fetch('/api/status').then(r=>r.json()).catch(()=>null);
  if(s){ need=s.n_frames; statusEl.textContent='server ready · '+need+' frames/sign'; }
  const stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  cam.srcObject=stream; await cam.play();
  cv.width=640; cv.height=480;
  loop();
}

// send frames continuously; server does detection + overlay + (when collecting) prediction
async function loop(){
  if(!sending && cam.readyState>=2){
    sending=true;
    const ctx=cv.getContext('2d');
    ctx.drawImage(cam,0,0,cv.width,cv.height);
    cv.toBlob(async (blob)=>{
      try{
        const r=await fetch('/api/frame',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:blob}).then(r=>r.json());
        if(r.image) view.src='data:image/jpeg;base64,'+r.image;
        collecting=r.collecting;
        barfill.style.width=(100*(r.frames||0)/(r.need||need))+'%';
        if(r.result){
          if(r.result.word && r.result.word!=='…'){ wordEl.textContent=r.result.word;
            confEl.textContent = r.result.id>=0 ? ('confidence '+(r.result.conf*100).toFixed(1)+'%  (id '+r.result.id+')') : ''; }
          if(!collecting && r.result.frames>=need){
            btn.textContent='Start Recognizing'; btn.className='big go';
          }
        }
      }catch(e){}
      sending=false;
    },'image/jpeg',0.7);
  }
  requestAnimationFrame(loop);
}

btn.onclick=async ()=>{
  wordEl.textContent='…'; confEl.textContent=''; barfill.style.width='0';
  btn.textContent='Recording — perform the sign'; btn.className='big stop';
  await fetch('/api/start',{method:'POST'});
};

init().catch(e=>{statusEl.textContent='init error: '+e;});
</script></body></html>"""


if __name__ == "__main__":
    print("Ishara sign-recognition web app")
    print("  (model + perception imported from your project files)")
    print("  open: http://localhost:%d" % PHONE_PORT)
    app.run(host="0.0.0.0", port=PHONE_PORT, threaded=True)