#!/usr/bin/env python3
"""
Ishara — mode state machine (Phase 3c-10).

Extracted verbatim from robot_phase4.py's MODE STATE MACHINE section: the
sentence buffer and its rendering (roman/sent_ascii/show), set_mode (with its
per-mode OLED text), the sign-capture voting loop (collect_one_sign), the three
recognition actions shared by hand gestures and phone buttons (do_recognize_once/
do_delete_word/do_speak_sentence), the three per-mode step functions (idle_step/
recognition_step/following_step) and the loop that dispatches them
(interaction_loop), the manual-drive safety watchdog, and do_drive.

Public names used elsewhere in robot_phase4.py: set_mode, interaction_loop,
manual_watchdog, do_drive, do_recognize_once, do_delete_word, do_speak_sentence,
sent_ascii (the phone /status and /nav handlers call these directly).

Depends on config (constants), state (shared), robot_io (robot), gestures,
model (name), text_bengali (bn_to_latin), audio — all "lower" modules with no
dependency on modes, so there is no circular import.
"""

import time
import threading
from collections import deque, Counter

from config import (ROMAN_OVERRIDE, RECOG_WINDOW, CONF_TH, VOTE_NEED, DOUBLE_WINDOW,
                    NAV_COOLDOWN, LEFT_DOUBLE_WINDOW, USE_SONAR, STOP_DIST, CENTER_DEAD,
                    STEER_DIR, FOLLOW_FAR_W, FOLLOW_DIST, FOLLOW_NEAR_W, AUTO_STOP_AFTER)
from state import shared
from robot_io import robot
from gestures import clear_buffer, current_prediction, hand_state, read_raw_gesture, wait_hands_down
from model import name
from text_bengali import bn_to_latin
from audio import speak_word, speak_sentence

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

# ---- shared recognition actions (used by BOTH hand gestures AND phone buttons) ----
_recog_lock = threading.Lock()

def do_recognize_once():
    """RIGHT hand / phone START: capture one sign, add it + speak it."""
    if not _recog_lock.acquire(blocking=False):
        return  # a capture is already running; ignore re-trigger
    try:
        if shared["mode"] != "recognition": return
        print("[START]")
        robot.oled(top="RECOGNIZING...|hold the sign still")
        sid = collect_one_sign()
        if shared["mode"] != "recognition":
            return
        if sid is not None:
            sentence.append(sid); shared["last_word"] = name(sid)
            print("[WORD]", name(sid), "|", show(sentence)); speak_word(sid)
            robot.oled(top="GOT: " + roman(sid) + "|RIGHT=next  BOTH=say", bottom=sent_ascii())
        else:
            robot.oled(top="no stable sign|try RIGHT again", bottom=sent_ascii())
    finally:
        _recog_lock.release()

def do_delete_word():
    """LEFT hand / phone DELETE: remove the last word."""
    if sentence:
        rem = sentence.pop()
        shared["last_word"] = name(sentence[-1]) if sentence else ""
        print("[DELETED]", name(rem), "|", show(sentence)); speak_word(rem)
    robot.oled(top="DELETED last word|RIGHT=add  BOTH=say", bottom=sent_ascii())

def do_speak_sentence():
    """BOTH hands / phone SPEAK: say the whole sentence, then clear it."""
    print("[SPEAK]", show(sentence))
    robot.oled(top="SPEAKING sentence...")
    if sentence: speak_sentence(list(sentence)); sentence.clear()
    shared["last_word"] = ""
    robot.oled(top="RECOGNITION|RIGHT=add sign|LEFT=delete|LEFTx2=exit  BOTH=say", bottom="")

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
        do_speak_sentence()
        time.sleep(NAV_COOLDOWN); wait_hands_down()
    elif g == "right":
        time.sleep(NAV_COOLDOWN); wait_hands_down()
        do_recognize_once()
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
            do_delete_word()
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
    # accepts F/B/L/R/S plus the four diagonals G/H/J/K (from joystick mixing)
    if shared["mode"] != "manual": return
    now = time.time()
    if   c == "F": robot.forward();   shared["drive_last"] = now; shared["driving"] = True
    elif c == "B": robot.backward();  shared["drive_last"] = now; shared["driving"] = True
    elif c == "L": robot.left();      shared["drive_last"] = now; shared["driving"] = True
    elif c == "R": robot.right();     shared["drive_last"] = now; shared["driving"] = True
    elif c == "G": robot.fwdleft();   shared["drive_last"] = now; shared["driving"] = True
    elif c == "H": robot.fwdright();  shared["drive_last"] = now; shared["driving"] = True
    elif c == "J": robot.backleft();  shared["drive_last"] = now; shared["driving"] = True
    elif c == "K": robot.backright(); shared["drive_last"] = now; shared["driving"] = True
    elif c == "S": robot.stop();      shared["driving"] = False
