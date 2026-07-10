#!/usr/bin/env python3
"""
Ishara — head-tracking servos (Phase 3c-7).

Extracted verbatim from robot_phase4.py. Owns the pan/tilt gpiozero servos on
GPIO22/23, the PD tracking loop (active only in recognition, frozen during
capture), and the servo-release cleanup. Public names: head_track_thread,
release_servos, _park.

release_servos() is registered on atexit here (fires on import, exactly as
before). robot_phase4.py also calls release_servos()/_park() from its shutdown()
and finally block, and its SIGINT/SIGTERM handlers route through shutdown(), so
the servos are still freed on Ctrl-C, kill, and systemd stop. Reads shared from
state; all constants come from config.
"""

import time, atexit

from config import (PAN_PIN, TILT_PIN, HT_CENTER, HT_SMOOTH, HT_DEAD, HT_KP, HT_KD,
                    HT_MAX_STEP, HT_MIN_MOVE, HT_SETTLE, PAN_DIR, PAN_MIN, PAN_MAX,
                    TILT_DIR, TILT_MIN, TILT_MAX, clamp)
from state import shared

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


# --- guarantee the servos are released on ANY exit so they don't keep buzzing ---
# detach() only stops sending NEW pulses; on lgpio the pin can be left holding a
# signal when the process dies, so the servo keeps twitching with no program running.
# close() releases the GPIO line entirely. We register this on atexit AND on the
# SIGINT/SIGTERM handlers so it fires on Ctrl-C, `kill`, and systemd stop too.
_servos_released = False
def release_servos():
    global _servos_released
    if _servos_released: return
    _servos_released = True
    if HEAD_OK:
        for s in (pan, tilt):
            try: s.detach()
            except Exception: pass
            try: s.close()
            except Exception: pass
    print("head servos released (GPIO freed)")
atexit.register(release_servos)


def head_track_thread():
    if not HEAD_OK:
        print("[HT DEBUG] HEAD_OK is False -- servo init failed at import time, "
              "tracking will NEVER run. Check the 'head servos NOT available' message printed at startup.")
        return
    _park()
    pa = ta = HT_CENTER; sx = sy = 0.5; pdx = pdy = 0.0; was = False
    last_debug = 0.0   # TEMPORARY: remove once the root cause is found
    while shared["run"]:
        active = (shared["mode"] == "recognition" and not shared["collecting"] and shared["pose_found"])
        now = time.time()
        if now - last_debug > 1.0:   # TEMPORARY: once/sec diagnostic, remove later
            last_debug = now
            print("[HT DEBUG] mode=%s collecting=%s pose_found=%s active=%s nose=(%.2f,%.2f) pa=%.1f ta=%.1f" % (
                shared["mode"], shared["collecting"], shared["pose_found"], active,
                shared["nose_x"], shared["nose_y"], pa, ta))
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
                except Exception as e: print("[HT DEBUG] pan.angle write failed:", e)
                moved = True
        if abs(dy) > HT_DEAD:
            mv = clamp(HT_KP*dy + HT_KD*ddy, -HT_MAX_STEP, HT_MAX_STEP)
            if abs(mv) >= HT_MIN_MOVE:
                ta = clamp(ta + TILT_DIR*mv, TILT_MIN, TILT_MAX)
                try: tilt.angle = ta
                except Exception as e: print("[HT DEBUG] tilt.angle write failed:", e)
                moved = True
        if moved:
            time.sleep(HT_SETTLE)
            try: pan.detach(); tilt.detach()
            except Exception: pass
        time.sleep(0.03)
