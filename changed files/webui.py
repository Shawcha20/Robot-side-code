#!/usr/bin/env python3
"""
Ishara — phone / browser control server (Phase 3d-1).

3d-1 is a PURE INTERNAL RESTRUCTURE of the Phase 3c-12 server. Behaviour is
identical: same routes, same responses byte-for-byte, same dashboard. The only
changes are organizational, to make the file grow additively in later 3d steps:

  * the dashboard page is assembled from _CSS / _BODY / _JS string constants
    (concatenated once at import into PAGE) instead of one monolithic literal;
  * _snapshot() reads shared + robot once and builds the /status payload, so
    every future endpoint shares one state-read;
  * do_GET dispatches through the _ROUTES table instead of an if/elif chain;
    /video stays inline because it owns the raw multipart response stream.

No route added or removed, no HTML/CSS/JS character changed, no new shared key.
Public name: start_server. Everything else is internal.

Depends on config (PHONE_PORT), state (shared), robot_io (robot), modes (set_mode,
do_drive, do_recognize_once, do_delete_word, do_speak_sentence, sent_ascii).
"""

import time
import threading
import json
import http.server
import socketserver
import urllib.parse
import cv2

from config import PHONE_PORT
from state import shared
from robot_io import robot
from modes import set_mode, do_drive, do_recognize_once, do_delete_word, do_speak_sentence, sent_ascii

# ============================================================
#  DASHBOARD PAGE  (assembled from _CSS + _BODY + _JS at import)
# ============================================================
_CSS = """
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
body{margin:0;background:#0d0f12;color:#e8e8e8;font-family:-apple-system,Segoe UI,Roboto,sans-serif;text-align:center}
h2{margin:14px 0 4px}
.bar{font-size:14px;color:#9aa;margin-bottom:10px}
.bar b{color:#6cf}
.modes{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;padding:0 10px 14px}
.modes button{flex:1 1 40%;padding:14px;font-size:16px;border:0;border-radius:12px;background:#1c2128;color:#e8e8e8}
.modes button.on{background:#2d6cdf;color:#fff}
.sticks{display:flex;justify-content:space-between;gap:18px;max-width:430px;margin:0 auto;padding:10px}
.stick{flex:1;display:flex;flex-direction:column;gap:12px}
.stick .lbl{font-size:12px;color:#889;letter-spacing:1px}
.stick button{padding:30px 0;font-size:30px;border:0;border-radius:16px;background:#1c2128;color:#e8e8e8}
.stick button:active{background:#2d6cdf;color:#fff}
.stoprow{max-width:430px;margin:0 auto;padding:0 10px 6px}
.stoprow button{width:100%;padding:16px;font-size:18px;border:0;border-radius:14px;background:#7a1f1f;color:#fff}
.recog{display:flex;gap:10px;max-width:430px;margin:0 auto;padding:10px}
.recog button{flex:1;padding:20px 0;font-size:16px;border:0;border-radius:14px;background:#15321f;color:#cfe}
.recog button:active{background:#1f8a4c;color:#fff}
.sentence{margin:14px 10px;padding:10px;background:#11151a;border-radius:10px;min-height:22px;color:#6cf;word-break:break-all}
.note{font-size:12px;color:#778;margin:8px}
.hide{display:none}
"""

_BODY = """
<h2>Ishara</h2>
<img id="cam" src="/video" style="width:100%;max-width:460px;border-radius:10px;background:#000;display:block;margin:4px auto"/>
<div class="bar">mode: <b id="mode">-</b> &nbsp; dist: <b id="dist">-</b> cm</div>
<div class="modes">
  <button id="m_idle" onclick="mode('idle')">Idle</button>
  <button id="m_recognition" onclick="mode('recognition')">Recognition</button>
  <button id="m_following" onclick="mode('following')">Following</button>
  <button id="m_manual" onclick="mode('manual')">Manual</button>
</div>

<!-- MANUAL: two button-joysticks. LEFT side = forward/backward,
     RIGHT side = left/right. Hold to drive. Holding one from each side
     at the same time mixes into a diagonal (G/H/J/K). -->
<div id="drive" class="hide">
  <div class="sticks">
    <div class="stick">
      <div class="lbl">FORWARD / BACK</div>
      <button id="bF">&#9650;</button>
      <button id="bB">&#9660;</button>
    </div>
    <div class="stick">
      <div class="lbl">LEFT / RIGHT</div>
      <button id="bL">&#9664;</button>
      <button id="bR">&#9654;</button>
    </div>
  </div>
  <div class="stoprow"><button id="bS">STOP</button></div>
  <div class="note">Hold a button to move; release to stop. Hold one from each side for a diagonal.</div>
</div>

<!-- RECOGNITION: phone buttons mirroring the hand gestures -->
<div id="recogctl" class="recog hide">
  <button id="rStart">START</button>
  <button id="rDelete">DELETE</button>
  <button id="rSpeak">SPEAK</button>
</div>

<div class="sentence" id="sent">-</div>
"""

_JS = """
function mode(m){fetch('/mode?m='+m).catch(()=>{});}
function send(c){fetch('/drive?c='+c).catch(()=>{});}
function nav(a){fetch('/nav?a='+a).catch(()=>{});}

// Press-and-hold driving with two button-joysticks.
// fb holds 'F'/'B'/''; lr holds 'L'/'R'/''. They are MIXED into one motor
// command, so holding one from each side gives a diagonal. A 150ms keepalive
// re-sends the command while held; releasing all sends STOP.
var fb="", lr="", drvTimer=null;
function mixedCmd(){
  if(fb==="F"&&lr==="")  return "F";
  if(fb==="B"&&lr==="")  return "B";
  if(fb===""&&lr==="L")  return "L";
  if(fb===""&&lr==="R")  return "R";
  if(fb==="F"&&lr==="L") return "G";   // forward-left
  if(fb==="F"&&lr==="R") return "H";   // forward-right
  if(fb==="B"&&lr==="L") return "J";   // back-left
  if(fb==="B"&&lr==="R") return "K";   // back-right
  return "S";
}
function sendMixed(){ send(mixedCmd()); }
function startPump(){ if(!drvTimer) drvTimer=setInterval(sendMixed,150); }
function stopPumpIfIdle(){
  if(!fb&&!lr){ if(drvTimer){clearInterval(drvTimer);drvTimer=null;} send('S'); }
}
function bindHold(id,axis,val){
  var b=document.getElementById(id);
  var dn=function(e){e.preventDefault(); if(axis==="fb")fb=val; else lr=val; startPump(); sendMixed();};
  var up=function(e){e.preventDefault(); if(axis==="fb")fb=""; else lr=""; sendMixed(); stopPumpIfIdle();};
  b.addEventListener('pointerdown',dn); b.addEventListener('pointerup',up);
  b.addEventListener('pointerleave',up); b.addEventListener('pointercancel',up);
}
bindHold('bF','fb','F'); bindHold('bB','fb','B');
bindHold('bL','lr','L'); bindHold('bR','lr','R');
document.getElementById('bS').addEventListener('click',function(){
  fb="";lr=""; if(drvTimer){clearInterval(drvTimer);drvTimer=null;} send('S');});

document.getElementById('rStart').addEventListener('click',function(){nav('start');});
document.getElementById('rDelete').addEventListener('click',function(){nav('delete');});
document.getElementById('rSpeak').addEventListener('click',function(){nav('speak');});

setInterval(async function(){try{var r=await fetch('/status');var s=await r.json();
 document.getElementById('mode').textContent=s.mode;
 document.getElementById('dist').textContent=s.dist;
 document.getElementById('sent').textContent=s.sentence||'-';
 ['idle','recognition','following','manual'].forEach(function(m){
   document.getElementById('m_'+m).className=(s.mode===m?'on':'');});
 document.getElementById('drive').className=(s.mode==='manual'?'':'hide');
 document.getElementById('recogctl').className=(s.mode==='recognition'?'recog':'recog hide');
}catch(e){}},500);
"""

def _render_page():
    # Assembled to reproduce the exact original PAGE byte-for-byte:
    # <head> ... <style>{_CSS}</style></head><body>{_BODY}<script>{_JS}</script></body></html>
    return ('<!doctype html><html><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">\n'
            '<title>Robot</title><style>'
            + _CSS +
            '</style></head><body>'
            + _BODY +
            '<script>'
            + _JS +
            '</script></body></html>')

PAGE = _render_page()

# ============================================================
#  STATE SNAPSHOT  (single source for status payloads)
# ============================================================
def _snapshot():
    return {"mode": shared["mode"], "word": shared["last_word"],
            "sentence": sent_ascii(), "dist": round(robot.distance(), 1)}

# ============================================================
#  ROUTE HANDLERS  (each returns (code, content_type, body_str))
# ============================================================
def _h_root(q):
    return 200, "text/html", PAGE

def _h_status(q):
    return 200, "application/json", json.dumps(_snapshot())

def _h_mode(q):
    m = q.get("m", [""])[0]
    if m in ("idle", "recognition", "following", "manual"): set_mode(m)
    return 200, "application/json", json.dumps({"mode": shared["mode"]})

def _h_drive(q):
    do_drive(q.get("c", [""])[0])
    return 200, "application/json", json.dumps({"mode": shared["mode"]})

def _h_nav(q):
    # phone recognition buttons -> same actions as the hand gestures
    a = q.get("a", [""])[0]
    if shared["mode"] == "recognition":
        if a == "start":
            threading.Thread(target=do_recognize_once, daemon=True).start()
        elif a == "delete":
            do_delete_word()
        elif a == "speak":
            do_speak_sentence()
    return 200, "application/json", json.dumps({"ok": True})

_ROUTES = {
    "/": _h_root,
    "/status": _h_status,
    "/mode": _h_mode,
    "/drive": _h_drive,
    "/nav": _h_nav,
}

# ============================================================
#  HTTP HANDLER / SERVER
# ============================================================
class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)
    def do_GET(self):
        try:
            u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
            handler = _ROUTES.get(u.path)
            if handler is not None:
                code, ctype, body = handler(q)
                self._send(code, ctype, body)
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
