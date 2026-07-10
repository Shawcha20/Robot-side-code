#!/usr/bin/env python3
"""
PHASE 4 - integrated robot (self-contained) + OLED status + PHONE control.

Control options (both work at once):
  * Hand gestures (idle: RIGHT x1 -> recognition, RIGHT x2 -> following;
    recognition: RIGHT=sign LEFT=delete LEFTx2=idle BOTH=speak; following: LEFT=idle)
  * Phone / browser: open  http://<pi-ip>:8000  -> mode buttons + button-joysticks
    (manual) + START/DELETE/SPEAK buttons (recognition)
  * Keyboard fallback (click the OpenCV window): i/r/f/m, q=quit; manual W/A/S/D

STM32 link (newline-framed): Pi sends "F/B/L/R/S" + "G/H/J/K" (diagonals),
"#<top>" / "$<bottom>" (OLED); STM32 streams "D<distance>". Flash stm32_phase4.ino.

Phase 3c complete: this file is now the thin entrypoint. All logic lives in:
config, text_bengali, audio, model, perception, state, headtracking, gestures,
robot_io, modes, capture, webui (protocol/link underlie robot_io). This file only
wires those pieces together -- signal handlers, startup sequence, and the
display/keyboard main loop -- exactly as robot_phase4.py did before extraction.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["PYTHONUTF8"] = "1"

import time
import threading
import signal
import cv2

from config import N_FRAMES, PHONE_PORT
from state import shared, buf_lock
from robot_io import robot
from capture import capture_thread
from headtracking import head_track_thread, release_servos, _park
from modes import interaction_loop, manual_watchdog, set_mode, do_drive
from webui import start_server

# ============================================================
#  CLEAN SHUTDOWN (servos must not keep running after exit)
# ============================================================
def shutdown(*_a):
    shared["run"] = False
    try: robot.stop()
    except Exception: pass
    try: robot.oled(top="STOPPED", bottom="")
    except Exception: pass
    release_servos()

# fire on Ctrl-C and on `kill` / systemd stop so the servos are always freed,
# even if the program is closed without reaching the finally block below
signal.signal(signal.SIGINT,  lambda *a: shutdown())
signal.signal(signal.SIGTERM, lambda *a: shutdown())

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
    robot.oled(top="STOPPED", bottom=""); _park(); release_servos(); time.sleep(0.6)
    try: cv2.destroyAllWindows()
    except cv2.error: pass
    print("robot stopped")
