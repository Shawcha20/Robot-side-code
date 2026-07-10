#!/usr/bin/env python3
"""
PHASE 3 - Pi <-> STM32 serial bridge.

Pairs with phase2_robot_main.ino. The Pi sends the same line protocol you used
in the Serial Monitor and reads the 'D' telemetry stream back.

  Protocol sent TO the STM32:
    M <l> <r>     motors -255..255      (0 0 = stop)
    A <u1> <u2>   arm servos 1100..1900
    O <text>      OLED big line (e.g. a recognised sign)
    Q             ask for one telemetry line
    T <0|1>       telemetry stream off/on
    X             stop everything
  Telemetry FROM the STM32:
    D <sonar_cm> <enc1> <enc2> <pitch> <roll>   (streamed ~10 Hz)

SAFETY: the STM32 stops the motors if no 'M' arrives for 700 ms. So drive()
sets a target and a background thread RE-SENDS it ~5x/sec to keep wheels moving.
If this program crashes or the cable drops, the motors stop on their own. Good.

SETUP on the Pi (conda env 'bdsl'):
    pip install pyserial
    sudo usermod -a -G dialout $USER     # then log out / back in (USB access w/o sudo)
    ls /dev/ttyACM*                      # confirm the port (usually /dev/ttyACM0)

USE from later phases:
    from robot_link import RobotLink
    bot = RobotLink('/dev/ttyACM0').connect()
    bot.show("LAL")           # show a recognised word on the OLED
    bot.drive(1/20, 120)       # forward
    print(bot.telemetry['sonar'])
    bot.stop(); bot.close()

RUN the interactive tester:
    python robot_link.py                 # or: python robot_link.py /dev/ttyACM1
"""

import sys
import time
import threading

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("pyserial not installed.  Run:  pip install pyserial")
    sys.exit(1)


class RobotLink:
    def __init__(self, port="/dev/ttyACM1", baud=115200, keepalive_hz=5, verbose=False):
        self.port = port
        self.baud = baud
        self.verbose = verbose
        self._ser = None
        self._running = False
        self._wlock = threading.Lock()      # serialise writes
        self._tlock = threading.Lock()      # protect telemetry
        self._telemetry = {"sonar": None, "enc1": 0, "enc2": 0,
                           "pitch": 0.0, "roll": 0.0, "ts": 0.0}
        self._motor = (0, 0)                 # current motor target (resent by keepalive)
        self._ka_period = 1.0 / keepalive_hz

    # ---- connection ----
    def connect(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        time.sleep(2.0)                      # let the STM32 settle (in case USB-connect resets it)
        self._ser.reset_input_buffer()
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._keepalive_loop, daemon=True).start()
        return self

    def close(self):
        self._running = False
        try:
            self._send("X")                  # stop motors + centre servos on the way out
        except Exception:
            pass
        time.sleep(0.1)
        if self._ser:
            self._ser.close()
            self._ser = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---- low-level write ----
    def _send(self, line):
        with self._wlock:
            if self._ser:
                self._ser.write((line + "\n").encode("ascii", "ignore"))

    # ---- high-level API ----
    def drive(self, left, right):
        left = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        self._motor = (left, right)
        self._send(f"M {left} {right}")

    def stop(self):
        self._motor = (0, 0)
        self._send("M 0 0")

    def estop(self):
        self._motor = (0, 0)
        self._send("X")

    def arms(self, us1, us2):
        us1 = max(1100, min(1900, int(us1)))
        us2 = max(1100, min(1900, int(us2)))
        self._send(f"A {us1} {us2}")

    def show(self, text):
        text = str(text).replace("\n", " ")[:20]
        self._send(f"O {text}")

    def stream(self, on=True):
        self._send(f"T {1 if on else 0}")

    @property
    def telemetry(self):
        with self._tlock:
            return dict(self._telemetry)

    # ---- background threads ----
    def _keepalive_loop(self):
        while self._running:
            l, r = self._motor
            if l != 0 or r != 0:             # resend only while moving (defeats the 700ms failsafe)
                self._send(f"M {l} {r}")
            time.sleep(self._ka_period)

    def _read_loop(self):
        while self._running:
            try:
                raw = self._ser.readline()
            except Exception:
                time.sleep(0.05)
                continue
            if not raw:
                continue
            line = raw.decode("ascii", "ignore").strip()
            if not line:
                continue
            if line.startswith("D "):
                p = line.split()
                if len(p) == 6:
                    try:
                        with self._tlock:
                            self._telemetry = {
                                "sonar": int(p[1]),
                                "enc1": int(p[2]),
                                "enc2": int(p[3]),
                                "pitch": float(p[4]),
                                "roll": float(p[5]),
                                "ts": time.time(),
                            }
                    except ValueError:
                        pass
            elif self.verbose:
                print(f"[stm32] {line}")


def _interactive(port):
    print(f"connecting to {port} ...")
    try:
        bot = RobotLink(port, verbose=True).connect()
    except Exception as e:
        print(f"could not open {port}: {e}")
        ports = [p.device for p in list_ports.comports()]
        print("available ports:", ports if ports else "(none found)")
        return

    print("connected. WHEELS OFF THE GROUND.")
    print("type the protocol directly:  M 120 120 | A 1700 1300 | O HELLO | X")
    print("extras:  telem (latest reading) | stop | quit")
    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue
            low = cmd.lower()
            if low in ("quit", "exit", "q"):
                break
            elif low == "telem":
                print(bot.telemetry)
            elif low == "stop":
                bot.stop()
            elif cmd[0] in "Mm":
                try:
                    _, l, r = cmd.split()
                    bot.drive(int(l), int(r))      # goes through drive() so keepalive holds it
                except ValueError:
                    print("usage: M <left> <right>")
            else:
                bot._send(cmd)                     # forward A / O / Q / T / X raw
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        bot.close()
        print("\nstopped and disconnected.")


if __name__ == "__main__":
    _interactive(sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1")
