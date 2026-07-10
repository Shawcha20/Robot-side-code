import serial
import threading
import time

ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)

time.sleep(2)

print("Connected to STM32")
print("Type text and press Enter\n")


def reader():
    while True:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            print("\nReceived:", data.decode(errors='ignore'))


threading.Thread(target=reader, daemon=True).start()

try:
    while True:
        msg = input("Send: ")

        ser.write(msg.encode())

except KeyboardInterrupt:
    ser.close()
    print("\nSerial Closed")
