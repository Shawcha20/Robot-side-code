"""
Host-side self-test for protocol.py. No robot or serial port needed.
Run:  python3 test_comms.py
"""
import struct
import protocol as p


def H(hexstr: str) -> bytes:
    return bytes.fromhex(hexstr.replace(" ", ""))


passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


print("CRC-16/CCITT-FALSE")
check("check value 0x29B1", p.crc16(b"123456789") == 0x29B1)

print("\nEncoder output matches the spec's verified frames (section 12)")
check("CMD_SETPOINT v=300 w=0 seq7",
      p.encode(p.CMD_SETPOINT, 7, p.setpoint_payload(300, 0)) == H("7E 07 01 07 2C 01 00 00 00 2D 73"))
check("CMD_MODE BALANCE seq8",
      p.encode(p.CMD_MODE, 8, p.mode_payload(p.Mode.BALANCE)) == H("7E 03 02 08 02 97 D8"))
check("CMD_ESTOP seq9",
      p.encode(p.CMD_ESTOP, 9) == H("7E 02 03 09 86 66"))
check("CMD_HEARTBEAT seq10",
      p.encode(p.CMD_HEARTBEAT, 10) == H("7E 02 05 0A 43 FC"))
check("PARAM_SET id0x10 val42.5 seq12",
      p.encode(p.PARAM_SET, 12, p.param_set_payload(0x0010, 42.5)) == H("7E 08 21 0C 10 00 00 00 2A 42 72 C4"))
check("PARAM_SAVE_FLASH seq13",
      p.encode(p.PARAM_SAVE_FLASH, 13) == H("7E 02 23 0D E4 20"))
check("TLM_HELLO(+db_ver) seq1",
      p.encode(p.TLM_HELLO, 1, struct.pack("<BBBHHH", 1, 0, 4, 3, 0x0007, 42))
      == H("7E 0B 45 01 01 00 04 03 00 07 00 2A 00 55 64"))

# The full 44-byte TLM_STATE from the spec, decoded back to engineering values.
TLM = H("7E 2E 40 21 05 00 00 00 85 FF 2D 00 F0 05 00 00 FA 00 C0 28 00 00 "
        "93 28 00 00 40 01 CA FE B4 00 F2 01 3E 00 B0 FF 40 2E B9 00 94 11 94 11 34 03 21 C3")

print("\nDecoder round-trips (TLM_STATE from the spec)")
frames = p.Demux().feed(TLM)
check("one frame decoded", len(frames) == 1 and isinstance(frames[0], p.Frame))
st = p.parse_tlm_state(frames[0].payload)
check("state = BALANCE", st["state_name"] == "BALANCE")
check("pitch = -1.23 deg", abs(st["pitch_deg"] - (-1.23)) < 1e-9)
check("pitch_rate = 4.5 dps", abs(st["pitch_rate_dps"] - 4.5) < 1e-9)
check("position = 1520 mm", st["position_mm"] == 1520)
check("velocity = 250 mm/s", st["velocity_mm_s"] == 250)
check("pwm_left = +0.32", abs(st["pwm_left"] - 0.320) < 1e-9)
check("pwm_right = -0.31", abs(st["pwm_right"] - (-0.310)) < 1e-9)
check("body_height = 180 mm", st["body_height_mm"] == 180)
check("desired_height = 185 mm", st["desired_height_mm"] == 185)
check("servo_left = 45.00 deg", abs(st["servo_left_deg"] - 45.0) < 1e-9)
check("servo_right = 45.00 deg", abs(st["servo_right_deg"] - 45.0) < 1e-9)
check("ctrl_exec = 820 us", st["ctrl_exec_us"] == 820)
check("loop = 498 Hz", st["loop_hz"] == 498)
check("sonar = 62 cm", st["sonar_cm"] == 62)
check("vbat = 11.84 V", abs(st["vbat_v"] - 11.840) < 1e-9)

print("\nTLM_DIAG + TLM_HELLO decode")
DIAG = H("7E 1C 41 05 E8 03 F4 01 C8 00 64 00 32 00 14 00 14 53 02 00 52 BC 00 00 03 00 01 00 00 00 A7 26")
d = p.parse_tlm_diag(p.Demux().feed(DIAG)[0].payload)
check("diag freq 1000/500 Hz", d["freq_hz"][1000] == 1000 and d["freq_hz"][500] == 500)
check("diag usb rx=152340 tx=48210", d["usb_rx_frames"] == 152340 and d["usb_tx_frames"] == 48210)
check("diag crc_err=3", d["usb_crc_errors"] == 3)
h = p.parse_tlm_hello(H("01 00 04 03 00 07 00 2A 00"))
check("hello proto ok", h["proto_ok"] and h["fw_str"] == "0.4")
check("hello db_ver ok", h["param_db_ok"] and h["param_count"] == 42)

print("\nACK / NAK / PARAM_VALUE / EVENT decode")
at, as_, stt = p.parse_ack(p.Demux().feed(H("7E 05 60 C9 21 0C 00 85 E0"))[0].payload)
check("ack -> (PARAM_SET, 12, OK)", at == p.PARAM_SET and as_ == 12 and stt == p.AckStatus.OK)
nt, ns, nr = p.parse_nak(p.Demux().feed(H("7E 05 61 CA 21 0C 02 4A F1"))[0].payload)
check("nak reason OUT_OF_RANGE", nr == p.NakReason.OUT_OF_RANGE)
pv = p.parse_param_value(p.Demux().feed(p.encode(p.PARAM_VALUE, 5, p.param_set_payload(0x0101, 3.5)))[0].payload)
check("param_value round-trip", pv[0] == 0x0101 and abs(pv[1] - 3.5) < 1e-6)
ev = p.parse_tlm_event(p.Demux().feed(p.encode(p.TLM_EVENT, 6, struct.pack("<Bh", p.Event.BODY_FROZEN, 0)))[0].payload)
check("event BODY_FROZEN", ev[0] == p.Event.BODY_FROZEN)

print("\nDemux: mixed legacy + framed stream, fed in awkward chunks, with garbage")
dm = p.Demux()
# legacy "D62", a setpoint frame, legacy "D63", one stray garbage byte, then TLM_HELLO
stream = (b"D62\n"
          + p.encode(p.CMD_SETPOINT, 1, p.setpoint_payload(120, -15))
          + b"D63\n"
          + b"\x11"                                    # garbage (not SOF, no newline yet)
          + p.encode(p.TLM_HELLO, 2, struct.pack("<BBBHHH", 1, 0, 4, 3, 7, 42)))
events = []
# feed 3 bytes at a time to prove the parser survives fragmentation
for i in range(0, len(stream), 3):
    events += dm.feed(stream[i:i + 3])
legacy = [e.line for e in events if isinstance(e, p.Legacy)]
framez = [e for e in events if isinstance(e, p.Frame)]
check("exactly the two legacy lines D62/D63 (garbage dropped)", legacy == ["D62", "D63"])
check("setpoint frame present", any(f.mtype == p.CMD_SETPOINT for f in framez))
check("hello frame present after garbage", any(f.mtype == p.TLM_HELLO for f in framez))

print("\nDemux: corrupted CRC is dropped and the next frame still decodes")
good = p.encode(p.CMD_MODE, 3, p.mode_payload(p.Mode.BALANCE))
bad = bytearray(p.encode(p.CMD_MODE, 4, p.mode_payload(p.Mode.READY)))
bad[-1] ^= 0xFF                                         # wreck the CRC
dm2 = p.Demux()
ev2 = dm2.feed(bytes(bad) + good)
check("bad frame dropped, good frame kept", len(ev2) == 1 and ev2[0].mtype == p.CMD_MODE and ev2[0].seq == 3)
check("crc error counted", dm2.crc_errors >= 1)

print(f"\nALL {passed} CHECKS PASSED")
