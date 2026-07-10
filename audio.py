#!/usr/bin/env python3
"""
Ishara — audio / voice output (Phase 3c-3).

Extracted verbatim from robot_phase4.py. Public names: speak_word, speak_sentence.
robot_phase4.py now does `from audio import speak_word, speak_sentence` in place of
the old inline block. The _wake() keepalive (plays a short silence so the first
real word is not clipped by the audio device waking up) and the paplay calls are
unchanged, so playback behavior and timing are identical.
"""

import os, time, subprocess, threading

from config import AUDIO_DIR

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
