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


class Robot:
    def __init__(self):
        self.link = SerialLink(STM_PORT, STM_BAUD, on_legacy=self._on_legacy)
        self.ok = False
        self._dist = -1.0

    def _on_legacy(self, line):
        # STM32 still streams the legacy "D<distance>" telemetry line
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
    def _cmd(self, ch):  self.link.send_legacy_dedup(ch)
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

robot = Robot()
