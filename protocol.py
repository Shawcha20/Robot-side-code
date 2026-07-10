"""
Ishara framed USB protocol — Raspberry Pi side (single source of truth).

This mirrors "Ishara Phase 2: USB Communication Protocol Specification" v1.1.
It is pure Python with no serial dependency, so it can be unit-tested on any host.

Two things live here:
  1. Constants + CRC + frame encode/decode (matches the spec byte-for-byte).
  2. A Demux that turns a raw incoming byte stream into decoded messages,
     handling BOTH the legacy newline protocol (e.g. b"D62\n") and the new
     framed protocol (0x7E ...), exactly as the STM32 RX parser does in reverse.

Nothing here replaces existing behaviour: the legacy path is preserved so the
current robot keeps working while the framed channel is introduced alongside it.
"""

from __future__ import annotations

import struct
from typing import Callable, List, NamedTuple, Optional, Tuple

# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #
PROTO_VERSION = 1          # bump only on a breaking frame-format change
PARAM_DB_VERSION_EXPECTED = 3   # what this Pi build expects from the STM32

# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
SOF = 0x7E                 # start-of-frame; legacy lines never begin with this
NEWLINE = 0x0A             # legacy line terminator
MAX_PAYLOAD = 253          # LEN is one byte; LEN = 2 + payload, so payload <= 253
MAX_LEGACY_LINE = 128      # legacy safety cap before a resync

# --------------------------------------------------------------------------- #
# Message types (TYPE byte) — see spec section 6
# --------------------------------------------------------------------------- #
# Pi -> STM32 : control
CMD_SETPOINT        = 0x01   # v_lin i16, v_ang i16, flags u8   (motion only)
CMD_MODE            = 0x02   # mode u8
CMD_ESTOP           = 0x03   # (none)
CMD_BODY            = 0x04   # action u8, target_height u16, duration_ms u16
CMD_HEARTBEAT       = 0x05   # (none)
# Pi -> STM32 : parameters
PARAM_GET           = 0x20   # id u16
PARAM_SET           = 0x21   # id u16, value f32
PARAM_GET_ALL       = 0x22   # (none)
PARAM_SAVE_FLASH    = 0x23   # (none)
PARAM_LOAD_FLASH    = 0x24   # (none)
PARAM_LOAD_DEFAULTS = 0x25   # (none)
PARAM_GET_META      = 0x26   # id u16
PARAM_PROFILE_SELECT = 0x27  # slot u8
PARAM_PROFILE_SAVE  = 0x28   # slot u8
# STM32 -> Pi : telemetry / status
TLM_STATE           = 0x40   # 44-byte struct
TLM_DIAG            = 0x41   # 26-byte struct
TLM_EVENT           = 0x42   # code u8, data i16
PARAM_VALUE         = 0x43   # id u16, value f32
PARAM_META          = 0x44   # 20-byte struct
TLM_HELLO           = 0x45   # proto u8, fw_major u8, fw_minor u8, db_ver u16, caps u16, count u16
TLM_LOG             = 0x46   # ascii string
# ack / nak (either direction)
ACK                 = 0x60   # acked_type u8, acked_seq u8, status u8
NAK                 = 0x61   # type u8, seq u8, reason u8
# session
HELLO               = 0x70   # proto u8, request_caps u16
SET_PROTOCOL        = 0x71   # mode u8 (0 dual, 1 framed-only)

TYPE_NAME = {v: k for k, v in globals().items()
             if isinstance(v, int) and k.isupper() and k.startswith(
                 ("CMD_", "PARAM_", "TLM_", "ACK", "NAK", "HELLO", "SET_"))}

# Types that must be acknowledged and retried by the sender (spec section 7).
RELIABLE_TYPES = frozenset({
    CMD_MODE, CMD_ESTOP, CMD_BODY,
    PARAM_GET, PARAM_SET, PARAM_GET_ALL, PARAM_SAVE_FLASH, PARAM_LOAD_FLASH,
    PARAM_LOAD_DEFAULTS, PARAM_GET_META, PARAM_PROFILE_SELECT, PARAM_PROFILE_SAVE,
    HELLO, SET_PROTOCOL,
})

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Mode:
    IDLE, READY, BALANCE, BODY = 0, 1, 2, 3

class BodyAction:
    STAND_UP, SIT_DOWN, GOTO_HEIGHT = 0, 1, 2

class State:
    BOOT, INIT, CALIB, READY, IDLE, BALANCE, MOVE, BODY, STAND, SIT, ESTOP, FAULT, RECOVER = range(13)

STATE_NAME = {getattr(State, n): n for n in vars(State) if n.isupper()}

class AckStatus:
    OK = 0x00
    OK_CLAMPED = 0x10

class NakReason:
    UNKNOWN_TYPE, OUT_OF_RANGE, BAD_PARAM_ID, BAD_STATE = 1, 2, 3, 4
    BUSY, NOT_SUPPORTED, BAD_LENGTH, FLASH_FAIL = 6, 7, 8, 9

class Event:
    (FALL_DETECTED, IMU_LOST, IMU_RECOVERED, ENC_STALL, MOTOR_SAT, SERVO_ERR,
     USB_TIMEOUT, OVERPITCH, OVERSPEED, STATE_CHANGE, CALIB_STEP, FLASH_SAVED,
     FLASH_LOADED, DEFAULTS_LOADED, BALANCE_ARMED, BALANCE_DISARMED,
     BODY_FROZEN) = range(1, 18)

EVENT_NAME = {getattr(Event, n): n for n in vars(Event) if n.isupper()}

class Cap:
    FRAMED_TELEMETRY = 1 << 0
    PARAM_FLASH      = 1 << 1
    BALANCE_AVAILABLE = 1 << 2
    BODY_HEIGHT_VARIABLE = 1 << 3
    LOGGING          = 1 << 4
    PROFILES         = 1 << 5

class SafetyFlag:
    FALL = 1 << 0
    IMU_LOST = 1 << 1
    ENC_STALL = 1 << 2
    MOTOR_SAT = 1 << 3
    SERVO_ERR = 1 << 4
    USB_TIMEOUT = 1 << 5
    OVERPITCH = 1 << 6
    OVERSPEED = 1 << 7
    LOW_BATT = 1 << 8
    FLASH_ERR = 1 << 9

class Profile:
    INDOOR, OUTDOOR, COMPETITION, SMOOTH, LEARNING, CUSTOM = range(6)

class SetProtocol:
    DUAL, FRAMED_ONLY = 0, 1

# Wire scale factors (engineering value -> integer on the wire)
V_ANG_SCALE     = 100.0   # deg/s  -> centi-deg/s
PITCH_SCALE     = 100.0   # deg    -> centi-deg
PITCH_RATE_SCALE = 10.0   # deg/s  -> deci-deg/s
SERVO_SCALE     = 100.0   # deg    -> centi-deg

# Reserved parameter-id ranges (exact ids assigned in Phases 5-6)
PARAM_RANGES = {
    "kalman":   (0x0000, 0x00FF),
    "lqr":      (0x0100, 0x01FF),
    "pos_ctrl": (0x0200, 0x02FF),
    "turn":     (0x0300, 0x03FF),
    "servo":    (0x0400, 0x04FF),
    "motion":   (0x0500, 0x05FF),
    "safety":   (0x0600, 0x06FF),
    "system":   (0x0700, 0x07FF),
}


# --------------------------------------------------------------------------- #
# CRC-16/CCITT-FALSE  (poly 0x1021, init 0xFFFF); check value "123456789"=0x29B1
# --------------------------------------------------------------------------- #
def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def _i16(v: float) -> int:
    """Round to int and clamp into signed 16-bit range."""
    return max(-32768, min(32767, int(round(v))))


def _u16(v: float) -> int:
    return max(0, min(65535, int(round(v))))


# --------------------------------------------------------------------------- #
# Frame encode / decode
# --------------------------------------------------------------------------- #
def encode(mtype: int, seq: int, payload: bytes = b"") -> bytes:
    """Build one framed message: SOF | LEN | TYPE | SEQ | PAYLOAD | CRC16(LE)."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too long: {len(payload)} > {MAX_PAYLOAD}")
    ln = 2 + len(payload)                       # LEN counts TYPE + SEQ + PAYLOAD
    body = bytes((ln, mtype & 0xFF, seq & 0xFF)) + payload
    return bytes((SOF,)) + body + struct.pack("<H", crc16(body))


class Frame(NamedTuple):
    mtype: int
    seq: int
    payload: bytes


class Legacy(NamedTuple):
    line: str


class Demux:
    """
    Feed raw bytes; get back a list of Frame and Legacy events in order.

    Robustness rules (match the spec RX state machine):
      - A byte 0x7E at a message boundary starts a length-prefixed framed message.
      - Anything else is a legacy ASCII line accumulated until '\\n'.
      - A framed message with a bad CRC is dropped and we resync on the next 0x7E.
      - Framed payloads are read by length, so 0x0A / 0x7E inside them are fine.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> List[object]:
        self.buf.extend(data)
        out: List[object] = []
        while self.buf:
            b0 = self.buf[0]
            if b0 == SOF:
                if len(self.buf) < 2:
                    break                                   # need LEN
                ln = self.buf[1]
                if ln < 2:
                    del self.buf[0]                         # impossible LEN -> resync
                    self.resyncs += 1
                    continue
                total = ln + 4                              # SOF+LEN+ln+CRC
                if len(self.buf) < total:
                    break                                   # wait for the rest
                body = bytes(self.buf[1:2 + ln])            # LEN..payload
                rx_crc = struct.unpack_from("<H", self.buf, 2 + ln)[0]
                if crc16(body) == rx_crc:
                    out.append(Frame(self.buf[2], self.buf[3], bytes(self.buf[4:2 + ln])))
                    del self.buf[:total]
                else:
                    del self.buf[0]                         # drop SOF, resync on next
                    self.crc_errors += 1
                    self.resyncs += 1
            else:
                nl = self.buf.find(NEWLINE)
                sof = self.buf.find(SOF)            # first SOF is at index >= 1 here
                if nl != -1 and (sof == -1 or nl < sof):
                    line = bytes(self.buf[:nl]).decode("ascii", "replace").strip()
                    del self.buf[:nl + 1]
                    if line:
                        out.append(Legacy(line))
                elif sof != -1:
                    # a frame begins before any newline: the bytes before it are
                    # garbage/noise (legacy lines never contain 0x7E on this link),
                    # so discard them and resync on the frame.
                    del self.buf[:sof]
                    self.resyncs += 1
                elif len(self.buf) > MAX_LEGACY_LINE:
                    del self.buf[0]                 # runaway with no delimiter -> resync
                    self.resyncs += 1
                else:
                    break                           # partial legacy line, wait for more
        return out


# --------------------------------------------------------------------------- #
# Typed encoders (Pi -> STM32)
# --------------------------------------------------------------------------- #
def setpoint_payload(v_lin_mm_s: float, v_ang_deg_s: float) -> bytes:
    """Motion-only setpoint. Angular velocity given in deg/s, sent as centi-deg/s."""
    return struct.pack("<hhB", _i16(v_lin_mm_s), _i16(v_ang_deg_s * V_ANG_SCALE), 0)


def mode_payload(mode: int) -> bytes:
    return struct.pack("<B", mode & 0xFF)


def body_payload(action: int, target_height_mm: int, duration_ms: int) -> bytes:
    return struct.pack("<BHH", action & 0xFF, _u16(target_height_mm), _u16(duration_ms))


def param_set_payload(param_id: int, value: float) -> bytes:
    return struct.pack("<Hf", param_id & 0xFFFF, float(value))


def param_id_payload(param_id: int) -> bytes:
    return struct.pack("<H", param_id & 0xFFFF)


def slot_payload(slot: int) -> bytes:
    return struct.pack("<B", slot & 0xFF)


def hello_payload(request_caps: int) -> bytes:
    return struct.pack("<BH", PROTO_VERSION, request_caps & 0xFFFF)


def set_protocol_payload(mode: int) -> bytes:
    return struct.pack("<B", mode & 0xFF)


# --------------------------------------------------------------------------- #
# Typed decoders (STM32 -> Pi)
# --------------------------------------------------------------------------- #
_TLM_STATE_FMT = "<BBHhhihiihhHHHhHHhhH"   # 44 bytes


def parse_tlm_state(p: bytes) -> dict:
    (state, profile, flags, pitch, pitch_rate, position, velocity, encL, encR,
     pwmL, pwmR, height, loop_hz, sonar, des_pitch, vbat,
     des_height, servoL, servoR, exec_us) = struct.unpack(_TLM_STATE_FMT, p)
    return {
        "state": state, "state_name": STATE_NAME.get(state, str(state)),
        "profile": profile, "safety_flags": flags,
        "pitch_deg": pitch / PITCH_SCALE,
        "pitch_rate_dps": pitch_rate / PITCH_RATE_SCALE,
        "position_mm": position, "velocity_mm_s": velocity,
        "enc_left": encL, "enc_right": encR,
        "pwm_left": pwmL / 1000.0, "pwm_right": pwmR / 1000.0,
        "body_height_mm": height, "loop_hz": loop_hz,
        "sonar_cm": (None if sonar == 0xFFFF else sonar),
        "desired_pitch_deg": des_pitch / PITCH_SCALE,
        "vbat_v": (None if vbat == 0 else vbat / 1000.0),
        "desired_height_mm": des_height,
        "servo_left_deg": servoL / SERVO_SCALE,
        "servo_right_deg": servoR / SERVO_SCALE,
        "ctrl_exec_us": exec_us,
    }


_TLM_DIAG_FMT = "<HHHHHHIIHHH"   # 26 bytes


def parse_tlm_diag(p: bytes) -> dict:
    (f1000, f500, f200, f100, f50, f20,
     rx, tx, crc_err, resync, dropped) = struct.unpack(_TLM_DIAG_FMT, p)
    return {
        "freq_hz": {1000: f1000, 500: f500, 200: f200, 100: f100, 50: f50, 20: f20},
        "usb_rx_frames": rx, "usb_tx_frames": tx,
        "usb_crc_errors": crc_err, "usb_resyncs": resync, "usb_rx_dropped": dropped,
    }


def parse_tlm_hello(p: bytes) -> dict:
    proto, fw_major, fw_minor, db_ver, caps, count = struct.unpack("<BBBHHH", p)
    return {
        "proto": proto, "fw": (fw_major, fw_minor), "fw_str": f"{fw_major}.{fw_minor}",
        "param_db_ver": db_ver, "caps": caps, "param_count": count,
        "proto_ok": proto == PROTO_VERSION,
        "param_db_ok": db_ver == PARAM_DB_VERSION_EXPECTED,
    }


def parse_param_value(p: bytes) -> Tuple[int, float]:
    return struct.unpack("<Hf", p)


def parse_param_meta(p: bytes) -> dict:
    pid, ptype, flags, pmin, pmax, pdef, pcur = struct.unpack("<HBBffff", p)
    return {
        "param_id": pid, "type": ptype,
        "editable": bool(flags & 0x01), "persists": bool(flags & 0x02),
        "min": pmin, "max": pmax, "default": pdef, "current": pcur,
    }


def parse_tlm_event(p: bytes) -> Tuple[int, int]:
    code, data = struct.unpack("<Bh", p)
    return code, data


def parse_ack(p: bytes) -> Tuple[int, int, int]:
    """returns (acked_type, acked_seq, status)"""
    return struct.unpack("<BBB", p)


def parse_nak(p: bytes) -> Tuple[int, int, int]:
    """returns (type, seq, reason)"""
    return struct.unpack("<BBB", p)


def parse_tlm_log(p: bytes) -> str:
    return p.decode("ascii", "replace").rstrip("\x00")
