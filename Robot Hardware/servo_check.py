from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0025)

print("centering")
servo.angle = 90
sleep(1.5)

try:
    while True:
        print("-> 45 degrees")
        servo.angle = 45; sleep(1.5)
        print("-> 90 (center)")
        servo.angle = 90; sleep(1.5)
        print("-> 135 degrees")
        servo.angle = 135; sleep(1.5)
except KeyboardInterrupt:
    servo.angle = 90
    sleep(0.5)
    print("\ncentered, stopped")