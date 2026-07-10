#!/usr/bin/env python3
"""
Ishara — STM32 link / Robot facade (Phase 3c-9).

Extracted verbatim from robot_phase4.py (the Phase 3b version, backed by
SerialLink). Owns the connection to the STM32: movement commands, OLED text,
and distance telemetry. Public name: robot (the module-level singleton).

Deliberately does NOT depend on state.py — it only needs config and link. This
lets modes.py do `from robot_io import robot` without any circular import,
which is why this extraction has to happen before the mode state machine.
"""

import time

from config import (STM_PORT, STM_BAUD, CMD_FWD, CMD_BACK, CMD_LEFT, CMD_RIGHT,
                    CMD_STOP, TELE_PREFIX)
from link import SerialLink


def _fmt_gain(v):
    """Format a gain for the KP/KI/KD serial command: whole numbers as plain
    integers (KP140, not KP140.0), fractional values with up to 3 decimals
    trimmed of trailing zeros (KD1.5). The STM32 parses either with toFloat()."""
    f = float(v)
    if f == int(f):
        return str(int(f))
    return ("%.3f" % f).rstrip("0").rstrip(".")


class Robot:
    def __init__(self):
        self.link = SerialLink(STM_PORT, STM_BAUD, on_legacy=self._on_legacy)
        self.ok = False
        self._dist = -1.0
        # commanded arm angles: only known after an ABSOLUTE set (ALxxx/ARxxx).
        # A relative nudge (ALU/ALD/ARU/ARD) changes the servo by a firmware-defined
        # step the Pi doesn't know, so nudges mark the angle "unknown" (None) rather
        # than guessing -- honest display beats a plausible-looking wrong number.
        self._arm_l_deg = None
        self._arm_r_deg = None
        # last balance gains COMMANDED from the Pi (None until set from here).
        # The STM32 holds the authoritative values; these are display-only.
        self._kp = None
        self._ki = None
        self._kd = None
        # telemetry-rate meter: counts legacy lines received (mostly "D<dist>")
        # to report how often the STM32 link is actually talking, independent of
        # any specific sensor. Purely observational; does not affect the link.
        self._tele_count = 0
        self._tele_window_start = None
        self._tele_rate = 0.0

    def _on_legacy(self, line):
        # STM32 still streams the legacy "D<distance>" telemetry line
        now = time.time()
        if self._tele_window_start is None:
            self._tele_window_start = now
        self._tele_count += 1
        elapsed = now - self._tele_window_start
        if elapsed >= 1.0:
            self._tele_rate = round(self._tele_count / elapsed, 1)
            self._tele_count = 0
            self._tele_window_start = now
        if line.startswith(TELE_PREFIX):
            try: self._dist = float(line[1:])
            except ValueError: pass

    def connect(self):
        try:
            self.link.open()
            time.sleep(2.0)                    # let the STM32 finish resetting
            self.ok = True
            print("STM32 serial connected on", STM_PORT)
        except Exception as e:
            print("STM32 NOT connected:", e, "- motors/OLED disabled, recognition still runs")
            self.ok = False

    # movement: identical single-letter commands, identical "only send if changed" guard
    # Drive commands are NOT deduped: following_step() and the manual-drive
    # keepalive both call these repeatedly with the SAME direction while a
    # decision/hold is sustained (~every 50ms and ~every 80ms respectively).
    # send_legacy_dedup would silently drop every one of those repeats after
    # the first, which -- if the STM32 needs a refreshed command to keep the
    # motor engaged rather than latching state from a single byte -- would
    # make both following and hold-to-drive stall after one brief pulse. The
    # cost of NOT deduping is negligible (~20 msgs/sec of 1-2 bytes each,
    # against 115200 baud's ~14400 bytes/sec) regardless of which behavior
    # the firmware actually has, so this is a strict safety improvement.
    def _cmd(self, ch):  self.link.send_legacy(ch)
    def forward(self):   self._cmd(CMD_FWD)
    def backward(self):  self._cmd(CMD_BACK)
    def left(self):      self._cmd(CMD_LEFT)
    def right(self):     self._cmd(CMD_RIGHT)
    def stop(self):      self._cmd(CMD_STOP)
    def fwdleft(self):   self._cmd("G")
    def fwdright(self):  self._cmd("H")
    def backleft(self):  self._cmd("J")
    def backright(self): self._cmd("K")
    def distance(self):  return self._dist

    def oled(self, top=None, bottom=None):
        if top is not None:    self.link.send_legacy("#" + top)      # not deduped, like before
        if bottom is not None: self.link.send_legacy("$" + bottom)

    # ----- arm control (Phase 3d-3) -----
    # Manual positioning of the two arm servos via the existing STM32 commands.
    # Step nudges (ALU/ALD/ARU/ARD) are sent with send_legacy (NOT deduped): each
    # press must nudge the servo, so two identical presses must both go out.
    # Absolute angle (ALxxx/ARxxx) is a zero-padded 3-digit degree in 0..90, e.g.
    # AL045 / AR090, matching the confirmed STM32 format. All arm commands go
    # through self.link.send_legacy -- webui never bypasses Robot to reach SerialLink.
    def arm_left_up(self):    self._arm_l_deg = None; self.link.send_legacy("ALU")
    def arm_left_down(self):  self._arm_l_deg = None; self.link.send_legacy("ALD")
    def arm_right_up(self):   self._arm_r_deg = None; self.link.send_legacy("ARU")
    def arm_right_down(self): self._arm_r_deg = None; self.link.send_legacy("ARD")

    def arm_left_angle(self, deg):
        d = max(135, min(270, int(deg)))
        self.link.send_legacy("AL%03d" % d)
        self._arm_l_deg = d
        return d

    def arm_right_angle(self, deg):
        d = max(135, min(270, int(deg)))
        self.link.send_legacy("AR%03d" % d)
        self._arm_r_deg = d
        return d

    def arm_angles(self):
        """Last COMMANDED absolute angles, or None if unknown (after a nudge,
        or before any absolute set)."""
        return self._arm_l_deg, self._arm_r_deg

    # ----- balance control (Phase 4) -----
    # Enable/disable the STM32 balance controller and live-tune its gains, via
    # the same legacy serial link. These mirror the STM32 commands BE/BD and
    # KP/KI/KD exactly. Not deduped: BE/BD must always go out, and re-sending
    # the same gain is harmless (STM32 just re-applies + resets its integral).
    def balance_enable(self):  self.link.send_legacy("BE")
    def balance_disable(self): self.link.send_legacy("BD")

    def set_kp(self, v):
        self.link.send_legacy("KP%s" % _fmt_gain(v)); self._kp = float(v); return self._kp
    def set_ki(self, v):
        self.link.send_legacy("KI%s" % _fmt_gain(v)); self._ki = float(v); return self._ki
    def set_kd(self, v):
        self.link.send_legacy("KD%s" % _fmt_gain(v)); self._kd = float(v); return self._kd

    def gains(self):
        """Last gains COMMANDED from the Pi, or None if never set from here.
        (The STM32 is the source of truth; these are what the dashboard sent.)"""
        return self._kp, self._ki, self._kd

    def telemetry_rate(self):
        """Lines/sec arriving from the STM32 over the legacy link (observational)."""
        return self._tele_rate

robot = Robot()
