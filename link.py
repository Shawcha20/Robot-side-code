"""
Ishara serial transport — Raspberry Pi side.

SerialLink owns the USB port and a background reader, and speaks BOTH protocols:

  * send_legacy / send_legacy_dedup  -> the current newline commands (F, G, AL045,
    "#..." , "$...") exactly as stm32_phase4.ino already expects. This is the shim
    Phase 3b routes robot_phase4.py's existing serial writes through, so the robot
    behaves identically while the framed channel is introduced beside it.
  * send_frame / send_reliable       -> the new framed protocol (parameters,
    heartbeat, e-stop, mode, telemetry subscription), with ACK/retry for reliable
    commands per the spec.

Incoming bytes are demultiplexed by protocol.Demux: legacy lines (e.g. "D62")
go to the on_legacy callback (preserving today's distance telemetry), framed
messages go to per-type callbacks.

pyserial is imported lazily so this module can be imported for host-side tests
even where pyserial is not installed; open() requires it on the Pi.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional, Tuple

import protocol as pr

try:
    import serial  # pyserial
except ImportError:  # allow import on hosts without pyserial (tests, dev)
    serial = None


class SerialLink:
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200,
                 on_legacy: Optional[Callable[[str], None]] = None) -> None:
        self.port = port
        self.baud = baud
        self._ser = None
        self._demux = pr.Demux()
        self._running = False
        self._reader: Optional[threading.Thread] = None
        self._hb: Optional[threading.Thread] = None

        self._wlock = threading.Lock()          # serialises all port writes
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._last_legacy: Optional[str] = None  # for send_legacy_dedup

        self._callbacks: Dict[int, Callable] = {}
        self._on_legacy = on_legacy

        self._pending: Dict[Tuple[int, int], Tuple[threading.Event, dict]] = {}
        self._pending_lock = threading.Lock()

        self.tx_frames = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial not installed (pip install pyserial)")
        self._ser = serial.Serial(self.port, self.baud, timeout=0.02)
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._running = False
        if self._reader:
            self._reader.join(timeout=0.5)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # callback registration
    # ------------------------------------------------------------------ #
    def on(self, mtype: int, fn: Callable) -> None:
        """Register a handler fn(frame) for a framed message type."""
        self._callbacks[mtype] = fn

    def on_legacy(self, fn: Callable[[str], None]) -> None:
        self._on_legacy = fn

    # ------------------------------------------------------------------ #
    # low-level write
    # ------------------------------------------------------------------ #
    def _raw_write(self, data: bytes) -> None:
        with self._wlock:
            if self._ser is not None:
                self._ser.write(data)

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq = (self._seq + 1) & 0xFF
            return self._seq

    # ------------------------------------------------------------------ #
    # legacy commands (preserve current behaviour)
    # ------------------------------------------------------------------ #
    def send_legacy(self, cmd: str) -> None:
        """Write one newline-terminated legacy command, e.g. 'F', 'G', 'AL045'."""
        self._raw_write((cmd + "\n").encode("ascii"))

    def send_legacy_dedup(self, cmd: str) -> None:
        """Only send if the command changed — mirrors robot_phase4.py's _cmd guard."""
        if cmd == self._last_legacy:
            return
        self._last_legacy = cmd
        self.send_legacy(cmd)

    def reset_legacy_dedup(self) -> None:
        self._last_legacy = None

    # ------------------------------------------------------------------ #
    # framed commands
    # ------------------------------------------------------------------ #
    def send_frame(self, mtype: int, payload: bytes = b"") -> int:
        seq = self._next_seq()
        self._raw_write(pr.encode(mtype, seq, payload))
        self.tx_frames += 1
        return seq

    def send_reliable(self, mtype: int, payload: bytes = b"",
                      timeout: float = 0.05, retries: int = 3) -> Tuple[bool, Optional[int]]:
        """
        Send an acknowledged command and block until ACK/NAK or give up.
        Returns (True, status) on ACK, (False, nak_reason) on NAK,
        (False, None) on timeout after all retries. Idempotent by spec, so
        retrying is always safe.
        """
        for _ in range(retries + 1):
            seq = self._next_seq()
            ev = threading.Event()
            box: dict = {"status": None, "nak": None}
            with self._pending_lock:
                self._pending[(mtype, seq)] = (ev, box)
            self._raw_write(pr.encode(mtype, seq, payload))
            self.tx_frames += 1
            got = ev.wait(timeout)
            with self._pending_lock:
                self._pending.pop((mtype, seq), None)
            if got:
                if box["nak"] is not None:
                    return (False, box["nak"])
                return (True, box["status"])
        return (False, None)

    # convenience wrappers ------------------------------------------------
    def setpoint(self, v_lin_mm_s: float, v_ang_deg_s: float) -> int:
        return self.send_frame(pr.CMD_SETPOINT, pr.setpoint_payload(v_lin_mm_s, v_ang_deg_s))

    def heartbeat(self) -> int:
        return self.send_frame(pr.CMD_HEARTBEAT)

    def estop(self):
        return self.send_reliable(pr.CMD_ESTOP)

    def set_mode(self, mode: int):
        return self.send_reliable(pr.CMD_MODE, pr.mode_payload(mode))

    def body(self, action: int, target_mm: int, duration_ms: int):
        return self.send_reliable(pr.CMD_BODY, pr.body_payload(action, target_mm, duration_ms))

    def hello(self, request_caps: int):
        return self.send_reliable(pr.HELLO, pr.hello_payload(request_caps))

    def param_set(self, param_id: int, value: float):
        return self.send_reliable(pr.PARAM_SET, pr.param_set_payload(param_id, value))

    def param_get(self, param_id: int):
        return self.send_reliable(pr.PARAM_GET, pr.param_id_payload(param_id))

    def param_get_all(self):
        return self.send_reliable(pr.PARAM_GET_ALL)

    def save_flash(self):
        return self.send_reliable(pr.PARAM_SAVE_FLASH)

    def load_flash(self):
        return self.send_reliable(pr.PARAM_LOAD_FLASH)

    def load_defaults(self):
        return self.send_reliable(pr.PARAM_LOAD_DEFAULTS)

    # ------------------------------------------------------------------ #
    # heartbeat thread (governs motion, not balance — see spec section 10)
    # ------------------------------------------------------------------ #
    def start_heartbeat(self, hz: float = 5.0) -> None:
        period = 1.0 / hz

        def loop():
            while self._running:
                try:
                    self.heartbeat()
                except Exception:
                    pass
                time.sleep(period)

        self._hb = threading.Thread(target=loop, daemon=True)
        self._hb.start()

    # ------------------------------------------------------------------ #
    # reader
    # ------------------------------------------------------------------ #
    def _read_loop(self) -> None:
        while self._running:
            try:
                data = self._ser.read(256) if self._ser else b""
            except Exception:
                time.sleep(0.05)
                continue
            if not data:
                continue
            for ev in self._demux.feed(data):
                if isinstance(ev, pr.Legacy):
                    if self._on_legacy:
                        try:
                            self._on_legacy(ev.line)
                        except Exception:
                            pass
                else:
                    self._dispatch(ev)

    def _dispatch(self, fr: "pr.Frame") -> None:
        if fr.mtype == pr.ACK:
            at, aseq, status = pr.parse_ack(fr.payload)
            self._resolve(at, aseq, status=status)
        elif fr.mtype == pr.NAK:
            t, s, reason = pr.parse_nak(fr.payload)
            self._resolve(t, s, nak=reason)
        cb = self._callbacks.get(fr.mtype)
        if cb:
            try:
                cb(fr)
            except Exception:
                pass

    def _resolve(self, mtype: int, seq: int, status=None, nak=None) -> None:
        with self._pending_lock:
            item = self._pending.get((mtype, seq))
        if item:
            ev, box = item
            box["status"] = status
            box["nak"] = nak
            ev.set()

    # diagnostics
    @property
    def rx_crc_errors(self) -> int:
        return self._demux.crc_errors

    @property
    def rx_resyncs(self) -> int:
        return self._demux.resyncs
