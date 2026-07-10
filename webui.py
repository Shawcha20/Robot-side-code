#!/usr/bin/env python3
"""
Ishara — phone / browser control server (Phase 3d-2).

3d-2 introduces the CLIENT-SIDE SHELL (API / Poller / Router) and moves the
existing dashboard into a "Control" tab, WITHOUT changing any control's
behaviour and WITHOUT adding/removing server routes. The manual-drive logic
(mixedCmd / bindHold / 150ms keepalive) and the recognition buttons are carried
over verbatim into ControlTab; the status-polling DOM updates are the same
operations, now driven by Poller.subscribe instead of a bare setInterval.

Tabs present in the shell (only Control has content in 3d-2; the rest are empty
sections proving the shell hosts them — they are filled in later 3d increments):
Control, Status, Camera, Arms, Balance, Tuning, Telemetry, System.

Server side is unchanged from 3d-1: same _snapshot(), same _ROUTES, same /video,
same responses. Public name: start_server.

Depends on config (PHONE_PORT), state (shared), robot_io (robot), modes (set_mode,
do_drive, do_recognize_once, do_delete_word, do_speak_sentence, sent_ascii).
"""

import time
import threading
import json
import os
import shutil
import socket
import http.server
import socketserver
import urllib.parse
import cv2

from config import PHONE_PORT, SOFTWARE_VERSION, AUDIO_DIR
from state import shared
from robot_io import robot
from modes import (set_mode, do_drive, do_recognize_once, do_delete_word, do_speak_sentence,
                    sent_ascii, status_extra)

# ============================================================
#  DASHBOARD PAGE  (assembled from _CSS + _BODY + _JS at import)
# ============================================================
_CSS = """
/* --- color tokens: one source of truth for the palette --- */
:root{
  --bg-0:#0d0f12; --bg-1:#0b0d10; --bg-2:#11151a; --bg-3:#1c2128;
  --fg-0:#e8e8e8; --fg-mut:#9aa; --fg-dim:#889; --fg-faint:#778; --fg-faintest:#667;
  --accent:#2d6cdf; --accent-hi:#6cf;
  --ok:#15321f; --ok-hi:#1f8a4c; --ok-fg:#cfe;
  --danger:#7a1f1f; --on-fg:#fff;
}
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
body{margin:0;background:var(--bg-0);color:var(--fg-0);font-family:-apple-system,Segoe UI,Roboto,sans-serif;text-align:center}
h2{margin:14px 0 4px}
.bar{font-size:14px;color:var(--fg-mut);margin-bottom:10px}
.bar b{color:var(--accent-hi)}
/* --- control status grid + last word --- */
.statusgrid{display:flex;gap:8px;max-width:460px;margin:8px auto;padding:0 10px}
.scell{flex:1;background:var(--bg-2);border-radius:10px;padding:8px 6px;font-size:13px;color:var(--fg-mut)}
.scell .sk{display:block;font-size:11px;color:var(--fg-faintest);letter-spacing:1px;margin-bottom:2px}
.scell b{color:var(--accent-hi);font-size:15px}
.lastword{font-size:13px;color:var(--fg-dim);margin:6px 10px}
.lastword b{color:var(--accent-hi)}
/* --- tab shell --- */
.tabbar{display:flex;overflow-x:auto;gap:6px;padding:8px 10px;background:var(--bg-1);position:sticky;top:0;z-index:5;-webkit-overflow-scrolling:touch;scrollbar-width:none}
.tabbar::-webkit-scrollbar{display:none}
.tabbar button{flex:0 0 auto;padding:9px 14px;font-size:13px;border:0;border-radius:9px;background:var(--bg-3);color:var(--fg-mut);white-space:nowrap;transition:background 120ms ease,color 120ms ease}
.tabbar button:hover{color:var(--fg-0)}
.tabbar button.active{background:var(--accent);color:var(--on-fg)}
.tab{display:none;animation:fadein 140ms ease-out}
.tab.active{display:block}
@keyframes fadein{from{opacity:0}to{opacity:1}}
/* --- existing control styles --- */
.modes{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;padding:0 10px 14px}
.modes button{flex:1 1 40%;padding:14px;font-size:16px;border:0;border-radius:12px;background:var(--bg-3);color:var(--fg-0);transition:background 120ms ease}
.modes button:hover{background:#252a34}
.modes button.on{background:var(--accent);color:var(--on-fg)}
.sticks{display:flex;justify-content:space-between;gap:18px;max-width:430px;margin:0 auto;padding:10px}
.stick{flex:1;display:flex;flex-direction:column;gap:12px}
.stick .lbl{font-size:12px;color:var(--fg-dim);letter-spacing:1px}
.stick button{padding:30px 0;font-size:30px;border:0;border-radius:16px;background:var(--bg-3);color:var(--fg-0);touch-action:none}
.stick button:active{background:var(--accent);color:var(--on-fg)}
.stoprow{max-width:430px;margin:0 auto;padding:0 10px 6px}
.stoprow button{width:100%;padding:16px;font-size:18px;border:0;border-radius:14px;background:var(--danger);color:var(--on-fg);touch-action:none}
.recog{display:flex;gap:10px;max-width:430px;margin:0 auto;padding:10px}
.recog button{flex:1;padding:20px 0;font-size:16px;border:0;border-radius:14px;background:var(--ok);color:var(--ok-fg)}
.recog button:active{background:var(--ok-hi);color:var(--on-fg)}
.sentence{margin:14px 10px;padding:10px;background:var(--bg-2);border-radius:10px;min-height:22px;color:var(--accent-hi);word-break:break-all}
.note{font-size:12px;color:var(--fg-faint);margin:8px}
.placeholder{color:var(--fg-faintest);font-size:14px;padding:40px 20px}
/* --- arms tab --- */
.armwrap{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;max-width:460px;margin:10px auto;padding:0 10px}
.armcard{flex:1 1 180px;background:var(--bg-2);border-radius:12px;padding:14px}
.armtitle{font-size:13px;color:var(--fg-dim);letter-spacing:1px;margin-bottom:10px}
.armsteps{display:flex;gap:10px;margin-bottom:14px}
.armsteps button{flex:1;padding:18px 0;font-size:16px;border:0;border-radius:12px;background:var(--bg-3);color:var(--fg-0)}
.armsteps button:active{background:var(--accent);color:var(--on-fg)}
.armslide input[type=range]{width:100%}
.armval{font-size:13px;color:var(--fg-mut);margin-top:6px}
.armval b{color:var(--accent-hi)}
/* --- status / camera tabs --- */
.statuslist{max-width:460px;margin:8px auto;padding:0 10px}
.srow{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;
      background:var(--bg-2);border-radius:9px;margin-bottom:6px;font-size:14px}
.srow span{color:var(--fg-dim)}
.srow b{color:var(--accent-hi);text-align:right}
/* --- section titles + tuning cards + telemetry charts --- */
.sectiontitle{max-width:460px;margin:14px auto 4px;padding:0 12px;font-size:12px;color:var(--fg-faintest);
              letter-spacing:1px;text-align:left}
.tunecard{max-width:460px;margin:8px auto;padding:10px 12px;background:var(--bg-2);border-radius:12px}
.tunetitle{font-size:13px;color:var(--fg-mut);letter-spacing:1px;margin-bottom:8px;text-align:left}
.tunerow{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:6px 0;font-size:13px}
.tunerow span{color:var(--fg-dim);flex:1;text-align:left}
.tunerow input{width:120px}
.tunerow input:disabled{opacity:0.5}
.tunebtns{display:flex;gap:8px}
.tunebtns button{flex:1;padding:12px 0;font-size:13px;border:0;border-radius:10px;background:var(--bg-3);color:var(--fg-faintest)}
.chartgrid{display:flex;flex-direction:column;gap:8px;max-width:460px;margin:8px auto;padding:0 10px}
.chartcard{background:var(--bg-2);border-radius:10px;padding:8px 10px}
.charttitle{font-size:12px;color:var(--fg-dim);text-align:left;margin-bottom:4px}
.chartcard canvas{width:100%;height:70px;display:block;background:var(--bg-1);border-radius:6px}
.chartval{font-size:12px;color:var(--accent-hi);text-align:right;margin-top:4px}
.hide{display:none}
"""

# Tab bar + one <section class="tab"> per tab. Only #tab-control has content in
# 3d-2; the rest are empty placeholders that later increments fill in.
_BODY = """
<h2>Ishara</h2>
<div class="tabbar" id="tabbar">
  <button data-tab="control" class="active">Control</button>
  <button data-tab="status">Status</button>
  <button data-tab="camera">Camera</button>
  <button data-tab="arms">Arms</button>
  <button data-tab="balance">Balance</button>
  <button data-tab="tuning">Tuning</button>
  <button data-tab="telemetry">Telemetry</button>
  <button data-tab="system">System</button>
</div>

<section id="tab-control" class="tab active">
  <img id="cam" src="/video" style="width:100%;max-width:460px;border-radius:10px;background:#000;display:block;margin:4px auto"/>

  <div class="statusgrid">
    <div class="scell"><span class="sk">mode</span><b id="mode">-</b></div>
    <div class="scell"><span class="sk">distance</span><b id="dist">-</b> cm</div>
    <div class="scell"><span class="sk">STM32</span><b id="conn">-</b></div>
  </div>

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
  <div class="lastword">last word: <b id="lastw">-</b></div>

  <!-- ARMS (embedded on Control; the full Arms tab remains as well) -->
  <div class="armwrap">
    <div class="armcard">
      <div class="armtitle">LEFT ARM</div>
      <div class="armsteps">
        <button id="cLU">Up</button>
        <button id="cLD">Down</button>
      </div>
      <div class="armslide">
        <input id="cLslider" type="range" min="0" max="90" value="45"/>
        <div class="armval">angle: <b id="cLval">45</b>&deg;</div>
      </div>
    </div>
    <div class="armcard">
      <div class="armtitle">RIGHT ARM</div>
      <div class="armsteps">
        <button id="cRU">Up</button>
        <button id="cRD">Down</button>
      </div>
      <div class="armslide">
        <input id="cRslider" type="range" min="0" max="90" value="45"/>
        <div class="armval">angle: <b id="cRval">45</b>&deg;</div>
      </div>
    </div>
  </div>
</section>

<section id="tab-status" class="tab">
  <div class="statuslist">
    <div class="srow"><span>Robot Mode</span><b id="st_mode">-</b></div>
    <div class="srow"><span>Current Distance</span><b id="st_dist">-</b></div>
    <div class="srow"><span>Current Sentence</span><b id="st_sentence">-</b></div>
    <div class="srow"><span>Last Recognized Word</span><b id="st_word">-</b></div>
    <div class="srow"><span>Current Prediction</span><b id="st_pred">-</b></div>
    <div class="srow"><span>Prediction Confidence</span><b id="st_conf">-</b></div>
    <div class="srow"><span>Recognition State</span><b id="st_collecting">-</b></div>
    <div class="srow"><span>Following State</span><b id="st_follow">-</b></div>
    <div class="srow"><span>Person Detected</span><b id="st_person">-</b></div>
    <div class="srow"><span>Left Hand Detected</span><b id="st_lhand">-</b></div>
    <div class="srow"><span>Right Hand Detected</span><b id="st_rhand">-</b></div>
    <div class="srow"><span>Head Tracking</span><b id="st_headtrack">-</b></div>
    <div class="srow"><span>STM32 Connection</span><b id="st_stm32">-</b></div>
    <div class="srow"><span>Camera Connection</span><b id="st_cam">-</b></div>
  </div>
</section>
<section id="tab-camera" class="tab">
  <img id="cam2" src="/video" style="width:100%;max-width:600px;border-radius:10px;background:#000;display:block;margin:4px auto"/>
  <div class="statuslist" style="max-width:460px">
    <div class="srow"><span>FPS</span><b id="cam_fps">-</b></div>
    <div class="srow"><span>Resolution</span><b id="cam_res">-</b></div>
    <div class="srow"><span>Pose Detection</span><b id="cam_pose">-</b></div>
    <div class="srow"><span>Hand Detection</span><b id="cam_hand">-</b></div>
    <div class="srow"><span>Person Detection</span><b id="cam_person">-</b></div>
    <div class="srow"><span>Current Mode</span><b id="cam_mode">-</b></div>
  </div>
</section>
<section id="tab-arms" class="tab">
  <div class="armwrap">
    <div class="armcard">
      <div class="armtitle">LEFT ARM</div>
      <div class="armsteps">
        <button id="aLU">Up</button>
        <button id="aLD">Down</button>
      </div>
      <div class="armslide">
        <input id="aLslider" type="range" min="0" max="90" value="45"/>
        <div class="armval">angle: <b id="aLval">45</b>&deg;</div>
      </div>
    </div>
    <div class="armcard">
      <div class="armtitle">RIGHT ARM</div>
      <div class="armsteps">
        <button id="aRU">Up</button>
        <button id="aRD">Down</button>
      </div>
      <div class="armslide">
        <input id="aRslider" type="range" min="0" max="90" value="45"/>
        <div class="armval">angle: <b id="aRval">45</b>&deg;</div>
      </div>
    </div>
  </div>
  <div class="note">Step buttons nudge the arm (ALU/ALD/ARU/ARD). Sliders set an absolute angle 0-90&deg; (ALxxx/ARxxx). Manual positioning only.</div>
</section>
<section id="tab-balance" class="tab">
  <div class="note" style="margin:10px">Sonar, STM32 link, and commanded arm angles are LIVE. IMU/encoder rows show
    "Waiting for firmware..." because the current STM32 firmware does not yet transmit that data
    (Phase 4). This tab is built to the final shape so Phase 4 only has to feed real values in.</div>
  <div class="statuslist">
    <div class="srow"><span>Sonar Distance</span><b id="bal_sonar">-</b></div>
    <div class="srow"><span>STM32 Connection</span><b id="bal_stm32">-</b></div>
    <div class="srow"><span>Telemetry Rate</span><b id="bal_telrate">-</b></div>
    <div class="srow"><span>Left Arm (commanded)</span><b id="bal_arml">-</b></div>
    <div class="srow"><span>Right Arm (commanded)</span><b id="bal_armr">-</b></div>
  </div>
  <div class="sectiontitle">IMU — Raw</div>
  <div class="statuslist">
    <div class="srow"><span>Accelerometer X/Y/Z</span><b id="bal_accel">Waiting for firmware...</b></div>
    <div class="srow"><span>Gyroscope X/Y/Z</span><b id="bal_gyro">Waiting for firmware...</b></div>
  </div>
  <div class="sectiontitle">IMU — Filtered</div>
  <div class="statuslist">
    <div class="srow"><span>Pitch</span><b id="bal_pitch">Waiting for firmware...</b></div>
    <div class="srow"><span>Roll</span><b id="bal_roll">Waiting for firmware...</b></div>
    <div class="srow"><span>Yaw</span><b id="bal_yaw">Waiting for firmware...</b></div>
  </div>
  <div class="sectiontitle">Encoders</div>
  <div class="statuslist">
    <div class="srow"><span>Left / Right</span><b id="bal_enc">Waiting for firmware...</b></div>
  </div>
</section>
<section id="tab-tuning" class="tab">
  <div class="note" style="margin:10px">Live controls below are active and talk to the STM32 now. Enable balancing,
    then adjust gains — changes apply instantly (the STM32 resets its integral on each change). Use the direct
    USB serial monitor as a backup kill (BD) during first tests. Cards further down remain placeholders for
    features not yet in firmware.</div>

  <div class="tunecard"><div class="tunetitle">Balance — Enable</div>
    <div class="tunerow">
      <button id="bal_en" style="flex:1;padding:14px;font-size:16px;border:0;border-radius:12px;background:var(--accent);color:var(--on-fg)">Enable (BE)</button>
      <button id="bal_dis" style="flex:1;padding:14px;font-size:16px;border:0;border-radius:12px;background:var(--danger);color:var(--on-fg);margin-left:8px">Disable / STOP (BD)</button>
    </div>
    <div class="tunerow"><span>State</span><span id="bal_state">disabled</span></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Balance — Recalibrate</div>
    <div class="note" style="margin:0 0 8px">Hold the robot steady at the exact posture you want as the new zero, THEN tap this.
      Disables balancing and stops motors first, then takes ~2s to measure the held posture.</div>
    <button id="bal_cal" style="width:100%;padding:14px;font-size:16px;border:0;border-radius:12px;background:var(--warn, #b8860b);color:var(--on-fg)">Recalibrate here (CAL)</button>
    <div class="tunerow"><span>Last result</span><span id="cal_state">-</span></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Balance — Live PID Gains</div>
    <div class="tunerow"><span>Kp</span>
      <input id="kp_in" type="number" step="1" value="140" style="width:90px"/>
      <button class="kbtn" data-g="kp">Set</button></div>
    <div class="tunerow"><span>Ki</span>
      <input id="ki_in" type="number" step="0.05" value="0" style="width:90px"/>
      <button class="kbtn" data-g="ki">Set</button></div>
    <div class="tunerow"><span>Kd</span>
      <input id="kd_in" type="number" step="0.1" value="1" style="width:90px"/>
      <button class="kbtn" data-g="kd">Set</button></div>
    <div class="note" style="margin-top:6px">Last sent from dashboard: <span id="kgains_sent">none yet</span>.
      The STM32 holds the authoritative values.</div>
  </div>

  <div class="note" style="margin:16px 10px 4px">— Below: placeholders for firmware features not yet implemented (not active) —</div>

  <div class="tunecard"><div class="tunetitle">Balance Controller (LQR)</div>
    <div class="tunerow"><span>K_pitch</span><input type="range" min="0" max="100" value="50" disabled/></div>
    <div class="tunerow"><span>K_pitch_rate</span><input type="range" min="0" max="100" value="50" disabled/></div>
    <div class="tunerow"><span>K_position</span><input type="range" min="0" max="100" value="50" disabled/></div>
    <div class="tunerow"><span>K_velocity</span><input type="range" min="0" max="100" value="50" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">EKF / Kalman Filter</div>
    <div class="tunerow"><span>Process noise Q</span><input type="number" value="0.01" disabled/></div>
    <div class="tunerow"><span>Measurement noise R</span><input type="number" value="0.10" disabled/></div>
    <div class="tunerow"><span>Initial covariance P0</span><input type="number" value="1.00" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">PID (legacy / non-balance loops)</div>
    <div class="tunerow"><span>Turn Kp</span><input type="number" value="1.0" disabled/></div>
    <div class="tunerow"><span>Turn Ki</span><input type="number" value="0.0" disabled/></div>
    <div class="tunerow"><span>Turn Kd</span><input type="number" value="0.0" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Servo (body height / balance assist)</div>
    <div class="tunerow"><span>Min height</span><input type="number" value="0" disabled/></div>
    <div class="tunerow"><span>Max height</span><input type="number" value="90" disabled/></div>
    <div class="tunerow"><span>Servo slew rate</span><input type="number" value="60" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Motion Manager</div>
    <div class="tunerow"><span>Max linear velocity</span><input type="number" value="0.5" disabled/></div>
    <div class="tunerow"><span>Max angular velocity</span><input type="number" value="1.0" disabled/></div>
    <div class="tunerow"><span>Accel limit</span><input type="number" value="2.0" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Safety Limits</div>
    <div class="tunerow"><span>Max tilt before E-stop</span><input type="number" value="30" disabled/></div>
    <div class="tunerow"><span>Heartbeat timeout (ms)</span><input type="number" value="500" disabled/></div>
    <div class="tunerow"><span>Current limit (A)</span><input type="number" value="5.0" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Motor Limits</div>
    <div class="tunerow"><span>Max PWM</span><input type="number" value="255" disabled/></div>
    <div class="tunerow"><span>Min PWM (deadband)</span><input type="number" value="30" disabled/></div>
  </div>

  <div class="tunecard"><div class="tunetitle">Parameters</div>
    <div class="tunebtns">
      <button disabled>Save to Flash</button>
      <button disabled>Load from Flash</button>
      <button disabled>Restore Defaults</button>
    </div>
  </div>
</section>
<section id="tab-telemetry" class="tab">
  <div class="note" style="margin:10px">Sonar is live now. Every other chart uses the identical component and is
    reserved for Phase 4 controller output -- no redesign needed when that data arrives.</div>
  <div class="chartgrid" id="chartgrid">
    <div class="chartcard"><div class="charttitle">Sonar (cm)</div><canvas id="ch_sonar" width="300" height="70"></canvas><div class="chartval" id="chv_sonar">-</div></div>
    <div class="chartcard"><div class="charttitle">Pitch (deg)</div><canvas id="ch_pitch" width="300" height="70"></canvas><div class="chartval" id="chv_pitch">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Roll (deg)</div><canvas id="ch_roll" width="300" height="70"></canvas><div class="chartval" id="chv_roll">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Pitch Rate (deg/s)</div><canvas id="ch_pitch_rate" width="300" height="70"></canvas><div class="chartval" id="chv_pitch_rate">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Velocity (m/s)</div><canvas id="ch_velocity" width="300" height="70"></canvas><div class="chartval" id="chv_velocity">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Position (m)</div><canvas id="ch_position" width="300" height="70"></canvas><div class="chartval" id="chv_position">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Motor PWM</div><canvas id="ch_motor_pwm" width="300" height="70"></canvas><div class="chartval" id="chv_motor_pwm">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Servo Angles (deg)</div><canvas id="ch_servo_angles" width="300" height="70"></canvas><div class="chartval" id="chv_servo_angles">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Body Height</div><canvas id="ch_body_height" width="300" height="70"></canvas><div class="chartval" id="chv_body_height">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Loop Frequency (Hz)</div><canvas id="ch_loop_freq_hz" width="300" height="70"></canvas><div class="chartval" id="chv_loop_freq_hz">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Loop Exec Time (ms)</div><canvas id="ch_loop_exec_ms" width="300" height="70"></canvas><div class="chartval" id="chv_loop_exec_ms">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Battery Voltage (V)</div><canvas id="ch_battery_v" width="300" height="70"></canvas><div class="chartval" id="chv_battery_v">Waiting for firmware...</div></div>
    <div class="chartcard"><div class="charttitle">Controller Output</div><canvas id="ch_controller_output" width="300" height="70"></canvas><div class="chartval" id="chv_controller_output">Waiting for firmware...</div></div>
  </div>
</section>
<section id="tab-system" class="tab">
  <div class="sectiontitle">Raspberry Pi</div>
  <div class="statuslist">
    <div class="srow"><span>CPU Usage</span><b id="sys_cpu">-</b></div>
    <div class="srow"><span>RAM Usage</span><b id="sys_ram">-</b></div>
    <div class="srow"><span>Disk Usage</span><b id="sys_disk">-</b></div>
    <div class="srow"><span>CPU Temperature</span><b id="sys_temp">-</b></div>
    <div class="srow"><span>Uptime</span><b id="sys_uptime">-</b></div>
    <div class="srow"><span>IP Address</span><b id="sys_ip">-</b></div>
  </div>
  <div class="sectiontitle">Hardware</div>
  <div class="statuslist">
    <div class="srow"><span>STM32 Connected</span><b id="sys_stm32">-</b></div>
    <div class="srow"><span>Camera Connected</span><b id="sys_cam">-</b></div>
    <div class="srow"><span>Audio Files Present</span><b id="sys_audio">-</b></div>
    <div class="srow"><span>Software Version</span><b id="sys_version">-</b></div>
  </div>
</section>
"""

# Client shell: API (fetch wrapper) + Poller (single /status loop, subscribers) +
# Router (hash tabs). ControlTab holds the EXISTING drive + recognition + status
# logic, carried over unchanged; it only subscribes to Poller instead of running
# its own setInterval. Behaviour on the Control tab is identical to 3d-1.
_JS = """
// ---- API: single fetch wrapper (all requests route through here) ----
var API = {
  get: function(path){ return fetch(path).catch(function(){ return null; }); }
};

// ---- Poller: one /status loop, distributes to subscribers ----
var Poller = (function(){
  var subs = [], timer = null, last = null;
  async function tick(){
    try{ var r = await fetch('/api/v1/status'); if(r){ last = await r.json();
      for(var i=0;i<subs.length;i++){ try{ subs[i](last); }catch(e){} } } }catch(e){}
  }
  return {
    subscribe: function(fn){ subs.push(fn); if(last) try{ fn(last); }catch(e){} },
    start: function(ms){ if(!timer){ timer = setInterval(tick, ms||500); tick(); } }
  };
})();

// ---- Router: hash-based tab show/hide (pure client, no server calls) ----
// Router: hash-based tab show/hide, PLUS onShow/onHide lifecycle hooks (3d-6) so
// tabs with their own live data (Balance/Telemetry/System) can start polling only
// while visible and stop when the user navigates away. The original tab-toggle
// behavior (classList/hash) is unchanged; hooks are purely additive.
var Router = (function(){
  var showHandlers = {}, hideHandlers = {}, activeTab = null;
  function show(name){
    var tabs = document.querySelectorAll('.tab');
    for(var i=0;i<tabs.length;i++){ tabs[i].classList.toggle('active', tabs[i].id === 'tab'+'-'+name); }
    var btns = document.querySelectorAll('#tabbar button');
    for(var j=0;j<btns.length;j++){ btns[j].classList.toggle('active', btns[j].getAttribute('data-tab') === name); }
    if(location.hash !== '#'+name) location.hash = name;
    if(activeTab && activeTab !== name && hideHandlers[activeTab]) hideHandlers[activeTab]();
    if(activeTab !== name && showHandlers[name]) showHandlers[name]();
    activeTab = name;
  }
  function current(){ return (location.hash || '#control').substring(1); }
  return {
    init: function(){
      var btns = document.querySelectorAll('#tabbar button');
      for(var i=0;i<btns.length;i++){
        btns[i].addEventListener('click', function(){ show(this.getAttribute('data-tab')); });
      }
      window.addEventListener('hashchange', function(){ show(current()); });
      show(current());
    },
    onShow: function(name, fn){ showHandlers[name] = fn; },
    onHide: function(name, fn){ hideHandlers[name] = fn; }
  };
})();

// TabPoller: like Poller, but only runs while its owning tab is visible. Used by
// Balance/Telemetry/System so their (slightly heavier: OS stats, disk reads) data
// isn't fetched in the background when the user is looking at a different tab.
function makeTabPoller(url, ms, onData){
  var timer = null;
  async function tick(){
    try{ var r = await fetch(url); if(r){ onData(await r.json()); } }catch(e){}
  }
  return {
    start: function(){ if(!timer){ timer = setInterval(tick, ms); tick(); } },
    stop: function(){ if(timer){ clearInterval(timer); timer = null; } }
  };
}

// ---- mode / nav helpers (unchanged wire behaviour) ----
function mode(m){ API.get('/mode?m='+m); }
function nav(a){ API.get('/nav?a='+a); }

// ---- ControlTab: EXISTING manual-drive + recognition + status logic ----
var ControlTab = (function(){
  function send(c){ API.get('/drive?c='+c); }
  // Press-and-hold driving with two button-joysticks. fb holds 'F'/'B'/'';
  // lr holds 'L'/'R'/''. MIXED into one motor command so holding one from each
  // side gives a diagonal. A 150ms keepalive re-sends while held; release -> STOP.
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
  function startPump(){ if(!drvTimer) drvTimer=setInterval(sendMixed,80); }   // 80ms keepalive (game-controller feel)
  // Stop the keepalive and send STOP exactly once when no direction is held.
  // (Per 3d-4 spec: release sends STOP once. If a direction is still held, we
  // resend the still-active mixed command instead.)
  function stopPumpIfIdle(){
    if(!fb&&!lr){ if(drvTimer){clearInterval(drvTimer);drvTimer=null;} send('S'); }
    else { sendMixed(); }
  }
  // Hold-to-drive across mouse + touch + pointer. Pointer events are the primary
  // path (cover mouse/touch/pen on modern browsers); we also bind touch + mouse
  // explicitly for older iOS/Android/desktop. A per-button "held" flag makes a
  // press send exactly once even when several event families fire together, and
  // touchstart/touchend preventDefault() to suppress iOS synthetic mouse events.
  //
  // ROOT CAUSE FIX: pointerleave fires whenever the pointer's COORDINATES leave
  // the element's box -- not when the finger lifts. A held finger always drifts
  // a few pixels, so on a real touchscreen pointerleave fires mid-hold and was
  // being treated as a release. setPointerCapture() locks all further events for
  // this pointer to this element regardless of where it physically moves, so
  // pointerleave no longer fires from drift -- only on a genuine release.
  function bindHold(id,axis,val){
    var b=document.getElementById(id); var held=false;
    function press(e){
      if(e&&e.preventDefault)e.preventDefault();
      if(held)return; held=true;
      if(e && e.pointerId!=null && b.setPointerCapture){
        try{ b.setPointerCapture(e.pointerId); }catch(err){}
      }
      if(axis==="fb")fb=val; else lr=val; startPump(); sendMixed();
    }
    function release(e){ if(e&&e.preventDefault)e.preventDefault(); if(!held)return; held=false;
      if(axis==="fb")fb=""; else lr=""; stopPumpIfIdle(); }
    b.addEventListener('pointerdown',press);  b.addEventListener('pointerup',release);
    b.addEventListener('pointerleave',release); b.addEventListener('pointercancel',release);
    b.addEventListener('touchstart',press,{passive:false});  b.addEventListener('touchend',release,{passive:false});
    b.addEventListener('touchcancel',release);
    b.addEventListener('mousedown',press);    b.addEventListener('mouseup',release);
    b.addEventListener('mouseleave',release);
    b.addEventListener('contextmenu', function(e){ e.preventDefault(); });
  }
  function onStatus(s){
    document.getElementById('mode').textContent=s.mode;
    document.getElementById('dist').textContent=s.dist;
    document.getElementById('sent').textContent=s.sentence||'-';
    var lw=document.getElementById('lastw'); if(lw) lw.textContent=s.word||'-';
    var cn=document.getElementById('conn'); if(cn) cn.textContent=(s.stm32?'connected':'offline');
    ['idle','recognition','following','manual'].forEach(function(m){
      document.getElementById('m_'+m).className=(s.mode===m?'on':'');});
    document.getElementById('drive').className=(s.mode==='manual'?'':'hide');
    document.getElementById('recogctl').className=(s.mode==='recognition'?'recog':'recog hide');
  }
  return {
    init: function(){
      bindHold('bF','fb','F'); bindHold('bB','fb','B');
      bindHold('bL','lr','L'); bindHold('bR','lr','R');
      document.getElementById('bS').addEventListener('click',function(){
        fb="";lr=""; if(drvTimer){clearInterval(drvTimer);drvTimer=null;} send('S');});
      document.getElementById('rStart').addEventListener('click',function(){nav('start');});
      document.getElementById('rDelete').addEventListener('click',function(){nav('delete');});
      document.getElementById('rSpeak').addEventListener('click',function(){nav('speak');});
      Poller.subscribe(onStatus);
    }
  };
})();

// ---- ArmsTab: step buttons (ALU/ALD/ARU/ARD) + absolute-angle sliders (ALxxx/ARxxx) ----
// All requests go to /api/v1/arms -> robot.arm_*() -> SerialLink -> STM32.
// Slider sends are throttled to ~80ms while dragging (plus a final send on release)
// so a drag can't flood the serial line; step buttons send immediately per press.
// bindStep/bindSlider are exposed so the Control tab's embedded arm block reuses
// the identical logic instead of duplicating it.
var ArmsTab = (function(){
  function step(side, dir){ API.get('/api/v1/arms?side='+side+'&dir='+dir); }
  function angle(side, deg){ API.get('/api/v1/arms?side='+side+'&angle='+deg); }
  function bindStep(btnId, side, dir){
    var b=document.getElementById(btnId); if(!b) return;
    b.addEventListener('click', function(){ step(side, dir); });
  }
  function bindSlider(sliderId, valId, side){
    var s = document.getElementById(sliderId), v = document.getElementById(valId);
    if(!s) return;
    var last = 0, pending = null;
    function flush(){ pending = null; angle(side, s.value); }
    s.addEventListener('input', function(){
      if(v) v.textContent = s.value;
      var now = Date.now();
      if(now - last >= 80){ last = now; angle(side, s.value); }
      else if(!pending){ pending = setTimeout(function(){ last = Date.now(); flush(); }, 80); }
    });
    s.addEventListener('change', function(){ if(pending){ clearTimeout(pending); pending=null; } angle(side, s.value); });
  }
  function wire(ids){
    // ids: {lu,ld,ru,rd, lslider,lval, rslider,rval}
    bindStep(ids.lu,'left','up');   bindStep(ids.ld,'left','down');
    bindStep(ids.ru,'right','up');  bindStep(ids.rd,'right','down');
    bindSlider(ids.lslider, ids.lval, 'left');
    bindSlider(ids.rslider, ids.rval, 'right');
  }
  return {
    init: function(){
      // dedicated Arms tab controls
      wire({lu:'aLU',ld:'aLD',ru:'aRU',rd:'aRD',lslider:'aLslider',lval:'aLval',rslider:'aRslider',rval:'aRval'});
    },
    wire: wire
  };
})();

// ---- StatusTab: read-only display of existing runtime info (Phase 3d-5) ----
var StatusTab = (function(){
  function yn(b){ return b ? 'yes' : 'no'; }
  function onStatus(s){
    var g = function(id){ return document.getElementById(id); };
    g('st_mode').textContent = s.mode;
    g('st_dist').textContent = s.dist + ' cm';
    g('st_sentence').textContent = s.sentence || '-';
    g('st_word').textContent = s.word || '-';
    g('st_pred').textContent = s.cur_pred_name || '-';
    g('st_conf').textContent = (s.cur_conf!=null) ? Math.round(s.cur_conf*100)+'%' : '-';
    g('st_collecting').textContent = s.collecting ? 'capturing sign...' : 'idle';
    g('st_follow').textContent = s.follow_decision || '-';
    g('st_person').textContent = yn(s.pose_found);
    g('st_lhand').textContent = yn(s.left_hand);
    g('st_rhand').textContent = yn(s.right_hand);
    g('st_headtrack').textContent = s.head_tracking_active ? 'active' : 'inactive';
    g('st_stm32').textContent = s.stm32 ? 'connected' : 'offline';
    g('st_cam').textContent = (s.camera && s.camera.connected) ? 'streaming' : 'no signal';
  }
  return { init: function(){ Poller.subscribe(onStatus); } };
})();

// ---- CameraTab: enlarged feed + existing detection info (Phase 3d-5) ----
var CameraTab = (function(){
  function onStatus(s){
    var g = function(id){ return document.getElementById(id); };
    var cam = s.camera || {};
    g('cam_fps').textContent = cam.fps != null ? cam.fps : '-';
    g('cam_res').textContent = cam.resolution || '-';
    g('cam_pose').textContent = s.pose_found ? 'detected' : 'not detected';
    g('cam_hand').textContent = (s.left_hand || s.right_hand) ? 'detected' : 'not detected';
    g('cam_person').textContent = s.pose_found ? 'detected' : 'not detected';
    g('cam_mode').textContent = s.mode;
  }
  return { init: function(){ Poller.subscribe(onStatus); } };
})();

// ---- Sparkline: minimal canvas line chart, used by TelemetryTab (3d-6) ----
// Rolling buffer of up to maxPoints values (nulls allowed = gap in the line).
// If fewer than 2 real points exist, shows "no data" instead of a misleading flat line.
function makeSparkline(canvasId, maxPoints){
  maxPoints = maxPoints || 40;
  var canvas = document.getElementById(canvasId);
  var ctx = canvas ? canvas.getContext('2d') : null;
  var buf = [];
  function draw(){
    if(!ctx) return;
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    var real = buf.filter(function(v){ return v != null; });
    if(real.length < 2){ ctx.fillText('no data', 4, h/2); return; }
    var min = Math.min.apply(null, real), max = Math.max.apply(null, real);
    if(min === max){ min -= 1; max += 1; }
    ctx.beginPath();
    var started = false;
    for(var i=0;i<buf.length;i++){
      var v = buf[i];
      if(v == null){ started = false; continue; }
      var x = (i/(maxPoints-1)) * w;
      var y = h - ((v-min)/(max-min)) * h;
      if(!started){ ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
    }
    ctx.stroke();
  }
  return {
    push: function(v){ buf.push(v); if(buf.length > maxPoints) buf.shift(); draw(); }
  };
}

// ---- BalanceTab: real sonar/STM32/arm fields; honest "waiting" for IMU/encoders (3d-6) ----
var BalanceTab = (function(){
  function onData(s){
    var g = function(id){ return document.getElementById(id); };
    g('bal_sonar').textContent = s.sonar_cm + ' cm';
    g('bal_stm32').textContent = s.stm32_connected ? 'connected' : 'offline';
    g('bal_telrate').textContent = s.telemetry_rate_hz + ' Hz';
    g('bal_arml').textContent = (s.arm_left_deg != null) ? s.arm_left_deg + '\u00b0' : 'unknown';
    g('bal_armr').textContent = (s.arm_right_deg != null) ? s.arm_right_deg + '\u00b0' : 'unknown';
    if(s.imu && s.imu.available){
      g('bal_accel').textContent = s.imu.accel; g('bal_gyro').textContent = s.imu.gyro;
      g('bal_pitch').textContent = s.imu.pitch; g('bal_roll').textContent = s.imu.roll;
      g('bal_yaw').textContent = s.imu.yaw;
    }
    if(s.encoders && s.encoders.available){
      g('bal_enc').textContent = s.encoders.left + ' / ' + s.encoders.right;
    }
  }
  var poller = null;
  return {
    init: function(){
      poller = makeTabPoller('/api/v1/balance', 500, onData);
      Router.onShow('balance', function(){ poller.start(); });
      Router.onHide('balance', function(){ poller.stop(); });
    }
  };
})();

// ---- TelemetryTab: one chart component, all 13 series, sonar real / rest reserved (3d-6) ----
var TelemetryTab = (function(){
  var charts = {};
  var fields = ['sonar','pitch','roll','pitch_rate','velocity','position','motor_pwm',
                'servo_angles','body_height','loop_freq_hz','loop_exec_ms','battery_v','controller_output'];
  function onData(s){
    fields.forEach(function(f){
      var key = (f === 'sonar') ? 'sonar_cm' : f;
      var val = s[key];
      if(!charts[f]) charts[f] = makeSparkline('ch_'+f);
      charts[f].push(val == null ? null : val);
      var vEl = document.getElementById('chv_'+f);
      if(vEl) vEl.textContent = (val == null) ? 'Waiting for firmware...' : val;
    });
  }
  var poller = null;
  return {
    init: function(){
      poller = makeTabPoller('/api/v1/telemetry', 500, onData);
      Router.onShow('telemetry', function(){ poller.start(); });
      Router.onHide('telemetry', function(){ poller.stop(); });
    }
  };
})();

// ---- SystemTab: Pi/OS + hardware status (3d-6). Slower cadence: disk/CPU reads cost more. ----
var SystemTab = (function(){
  function onData(s){
    var g = function(id){ return document.getElementById(id); };
    g('sys_cpu').textContent = (s.cpu_pct != null) ? s.cpu_pct + '%' : '-';
    g('sys_ram').textContent = (s.ram_pct != null) ? s.ram_pct + '%' : '-';
    g('sys_disk').textContent = (s.disk_pct != null) ? s.disk_pct + '%' : '-';
    g('sys_temp').textContent = (s.cpu_temp_c != null) ? s.cpu_temp_c + '\u00b0C' : '-';
    g('sys_uptime').textContent = s.uptime || '-';
    g('sys_ip').textContent = s.ip || '-';
    g('sys_stm32').textContent = s.stm32_connected ? 'connected' : 'offline';
    g('sys_cam').textContent = s.camera_connected ? 'connected' : 'offline';
    g('sys_audio').textContent = s.audio_present ? 'found' : 'missing';
    g('sys_version').textContent = s.version || '-';
  }
  var poller = null;
  return {
    init: function(){
      poller = makeTabPoller('/api/v1/system', 2000, onData);
      Router.onShow('system', function(){ poller.start(); });
      Router.onHide('system', function(){ poller.stop(); });
    }
  };
})();

// ---- TuningTab: live balance enable/disable/calibrate + PID gain setting (Phase 4) ----
// Buttons/inputs call /api/v1/balance_cmd -> robot -> STM32 (BE/BD/CAL/KP/KI/KD).
// Purely fire-and-forget over the same link; the STM32 is the source of truth.
var TuningTab = (function(){
  function setState(txt){ var e=document.getElementById('bal_state'); if(e) e.textContent=txt; }
  function setCal(txt){ var e=document.getElementById('cal_state'); if(e) e.textContent=txt; }
  return {
    init: function(){
      var en=document.getElementById('bal_en'), dis=document.getElementById('bal_dis');
      if(en)  en.addEventListener('click', function(){
        API.get('/api/v1/balance_cmd?action=enable'); setState('ENABLED'); });
      if(dis) dis.addEventListener('click', function(){
        API.get('/api/v1/balance_cmd?action=disable'); setState('disabled'); });
      var cal=document.getElementById('bal_cal');
      if(cal) cal.addEventListener('click', function(){
        if(!confirm('Hold the robot steady at the position you want as the new zero. Ready?')) return;
        setCal('calibrating... (~2s, keep it still)');
        API.get('/api/v1/balance_cmd?action=calibrate').then(function(){
          setCal('sent -- check OLED / [EST] err for result'); setState('disabled (CAL disables balancing)');
        });
      });
      var btns=document.querySelectorAll('.kbtn');
      for(var i=0;i<btns.length;i++){
        (function(b){
          b.addEventListener('click', function(){
            var g=b.getAttribute('data-g');
            var inp=document.getElementById(g+'_in');
            var v=inp?inp.value:null;
            if(v===null||v==='') return;
            API.get('/api/v1/balance_cmd?gain='+g+'&value='+encodeURIComponent(v));
            var sent=document.getElementById('kgains_sent');
            if(sent) sent.textContent =
              'Kp='+document.getElementById('kp_in').value+
              ' Ki='+document.getElementById('ki_in').value+
              ' Kd='+document.getElementById('kd_in').value;
          });
        })(btns[i]);
      }
    }
  };
})();

// ---- boot ----
// NOTE: Router.init() fires an initial show() for the current tab, which dispatches
// onShow/onHide -- so every tab that registers those hooks must do so BEFORE
// Router.init() runs, or a page load landing directly on e.g. #balance would
// silently never start its poller. Router.init() is therefore last.
ControlTab.init();
ArmsTab.init();
// embedded arm controls on the Control tab reuse the identical Arms logic
ArmsTab.wire({lu:'cLU',ld:'cLD',ru:'cRU',rd:'cRD',lslider:'cLslider',lval:'cLval',rslider:'cRslider',rval:'cRval'});
StatusTab.init();
CameraTab.init();
BalanceTab.init();
TuningTab.init();
TelemetryTab.init();
SystemTab.init();
Router.init();
Poller.start(500);
"""

def _render_page():
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
            "sentence": sent_ascii(), "dist": round(robot.distance(), 1),
            "stm32": bool(robot.ok)}

# ----- webui-local video instrumentation (Phase 3d-5) -----
# Counts frames THIS server actually wrote to the /video stream and derives an
# fps once per second. This instruments webui's own streaming loop only -- it
# does not read from or write to capture.py / state.py in any way.
_video_stats = {"frames": 0, "window_start": None, "fps": 0.0}

def _video_frame_sent():
    now = time.time()
    if _video_stats["window_start"] is None:
        _video_stats["window_start"] = now
    _video_stats["frames"] += 1
    elapsed = now - _video_stats["window_start"]
    if elapsed >= 1.0:
        _video_stats["fps"] = round(_video_stats["frames"] / elapsed, 1)
        _video_stats["frames"] = 0
        _video_stats["window_start"] = now

def _camera_snapshot():
    f = shared["frame"]
    if f is None:
        return {"connected": False, "resolution": None, "fps": 0.0}
    h, w = f.shape[0], f.shape[1]
    return {"connected": True, "resolution": "%dx%d" % (w, h), "fps": _video_stats["fps"]}

# ----- Balance tab data (Phase 3d-6) -----
# Honest split: fields that genuinely exist on the Pi today (sonar, STM32 link,
# commanded arm angles, telemetry rate) are real. The legacy STM32 firmware does
# NOT transmit IMU data -- there is no accelerometer/gyro/pitch/roll/encoder
# telemetry to show yet, so those fields report available:false rather than a
# fabricated number. Phase 4's TLM_STATE will fill these in without a UI change.
def _balance_snapshot():
    la, ra = robot.arm_angles()
    return {
        "sonar_cm": round(robot.distance(), 1),
        "stm32_connected": bool(robot.ok),
        "telemetry_rate_hz": robot.telemetry_rate(),
        "arm_left_deg": la, "arm_right_deg": ra,
        "imu": {"available": False, "accel": None, "gyro": None,
                "pitch": None, "roll": None, "yaw": None},
        "encoders": {"available": False, "left": None, "right": None},
    }

# ----- Telemetry tab data (Phase 3d-6) -----
# Same honesty rule: sonar is real and charted now; every other series is
# structured and reserved but reports null until Phase 4 provides real data,
# so the SAME chart component just starts receiving values later.
def _telemetry_snapshot():
    return {
        "t": time.time(),
        "sonar_cm": round(robot.distance(), 1),
        "pitch": None, "roll": None, "pitch_rate": None,
        "velocity": None, "position": None,
        "motor_pwm": None, "servo_angles": None, "body_height": None,
        "loop_freq_hz": None, "loop_exec_ms": None,
        "battery_v": None, "controller_output": None,
    }

# ----- System tab data (Phase 3d-6) -----
# Pure Pi/OS-level introspection: nothing here touches recognition, following,
# head tracking, or the STM32 link. Every read is wrapped so a failure (e.g. no
# thermal zone on a non-Pi dev machine) yields None instead of crashing.
def _cpu_load_pct():
    try:
        load1 = os.getloadavg()[0]
        n = os.cpu_count() or 1
        return round(min(100.0, load1 / n * 100.0), 1)
    except Exception:
        return None

def _ram_pct():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0])  # kB
        total = info.get("MemTotal"); avail = info.get("MemAvailable")
        if not total: return None
        used = total - (avail if avail is not None else 0)
        return round(used / total * 100.0, 1)
    except Exception:
        return None

def _disk_pct():
    try:
        du = shutil.disk_usage("/")
        return round(du.used / du.total * 100.0, 1)
    except Exception:
        return None

def _cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None

def _uptime_str():
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
        h, rem = divmod(int(secs), 3600); m, _ = divmod(rem, 60)
        return "%dh %dm" % (h, m)
    except Exception:
        return None

def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        return None

def _audio_present():
    try:
        if not os.path.isdir(AUDIO_DIR): return False
        return any(fn.endswith(".wav") for fn in os.listdir(AUDIO_DIR))
    except Exception:
        return False

def _system_snapshot():
    return {
        "cpu_pct": _cpu_load_pct(), "ram_pct": _ram_pct(), "disk_pct": _disk_pct(),
        "cpu_temp_c": _cpu_temp_c(), "uptime": _uptime_str(), "ip": _local_ip(),
        "stm32_connected": bool(robot.ok),
        "camera_connected": _camera_snapshot()["connected"],
        "audio_present": _audio_present(),
        "version": SOFTWARE_VERSION,
    }

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

# ----- /api/v1/arms : manual arm positioning (Phase 3d-3) -----
# Path: Arms tab -> here -> robot.arm_*() -> SerialLink -> STM32. webui never
# touches SerialLink directly. Step nudges: ?side=left|right & dir=up|down.
# Absolute angle: ?side=left|right & angle=0..90 (robot clamps + formats ALxxx/ARxxx).
def _h_arms(q):
    side = q.get("side", [""])[0]
    if side not in ("left", "right"):
        return 400, "application/json", json.dumps({"error": "side must be left or right"})
    angle = q.get("angle", [None])[0]
    if angle is not None:
        try:
            deg = int(angle)
        except (TypeError, ValueError):
            return 400, "application/json", json.dumps({"error": "angle must be an integer"})
        applied = robot.arm_left_angle(deg) if side == "left" else robot.arm_right_angle(deg)
        return 200, "application/json", json.dumps({"ok": True, "side": side, "angle": applied})
    d = q.get("dir", [""])[0]
    if d not in ("up", "down"):
        return 400, "application/json", json.dumps({"error": "dir must be up or down"})
    if side == "left":
        robot.arm_left_up() if d == "up" else robot.arm_left_down()
    else:
        robot.arm_right_up() if d == "up" else robot.arm_right_down()
    return 200, "application/json", json.dumps({"ok": True, "side": side, "dir": d})

# ----- /api/v1/status : richer, versioned status for the Status/Camera tabs -----
# Union of _snapshot() (legacy /status fields, unchanged), status_extra() (modes.py
# read-only aggregator), and _camera_snapshot() (webui-local video instrumentation).
# The legacy /status route is untouched and still returns exactly its original payload.
def _h_api_status(q):
    payload = {}
    payload.update(_snapshot())
    payload.update(status_extra())
    payload["camera"] = _camera_snapshot()
    return 200, "application/json", json.dumps(payload)

def _h_balance(q):
    return 200, "application/json", json.dumps(_balance_snapshot())

# ----- /api/v1/balance_cmd : enable/disable + live gain tuning (Phase 4) -----
# Path: Balance/Tuning tab -> here -> robot.balance_*/set_k* -> SerialLink ->
# STM32 (BE/BD/KP/KI/KD). webui never touches SerialLink directly.
#   ?action=enable | disable
#   ?gain=kp|ki|kd & value=<float>
def _h_balance_cmd(q):
    action = q.get("action", [None])[0]
    if action == "enable":
        robot.balance_enable()
        return 200, "application/json", json.dumps({"ok": True, "action": "enable"})
    if action == "disable":
        robot.balance_disable()
        return 200, "application/json", json.dumps({"ok": True, "action": "disable"})
    if action == "calibrate":
        robot.calibrate_now()
        return 200, "application/json", json.dumps({"ok": True, "action": "calibrate"})

    gain = q.get("gain", [None])[0]
    if gain in ("kp", "ki", "kd"):
        raw = q.get("value", [None])[0]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return 400, "application/json", json.dumps({"error": "value must be a number"})
        applied = {"kp": robot.set_kp, "ki": robot.set_ki, "kd": robot.set_kd}[gain](val)
        return 200, "application/json", json.dumps({"ok": True, "gain": gain, "value": applied})

    return 400, "application/json", json.dumps({"error": "action=enable|disable|calibrate or gain=kp|ki|kd&value=<n>"})

def _h_telemetry(q):
    return 200, "application/json", json.dumps(_telemetry_snapshot())

def _h_system(q):
    return 200, "application/json", json.dumps(_system_snapshot())

_ROUTES = {
    "/": _h_root,
    "/status": _h_status,
    "/mode": _h_mode,
    "/drive": _h_drive,
    "/nav": _h_nav,
    "/api/v1/arms": _h_arms,
    "/api/v1/status": _h_api_status,
    "/api/v1/balance": _h_balance,
    "/api/v1/balance_cmd": _h_balance_cmd,
    "/api/v1/telemetry": _h_telemetry,
    "/api/v1/system": _h_system,
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
                    _video_frame_sent()
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
