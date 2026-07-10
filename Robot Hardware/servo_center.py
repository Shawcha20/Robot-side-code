from gpiozero import AngularServo
from time import sleep

pan = AngularServo(18, min_angle=0, max_angle=270,
                   min_pulse_width=0.0005, max_pulse_width=0.0025)
tilt = AngularServo(13, min_angle=0, max_angle=270,
                    min_pulse_width=0.0005, max_pulse_width=0.0025)

print("centering both servos to 135 (middle of 270)")
pan.angle = 135
tilt.angle = 135
sleep(1.0)            # give them time to reach center

pan.detach()          # stop sending signal -> no more wobble
tilt.detach()
print("centered and detached - no wobble now. Ctrl+C to exit")

try:
    while True:
        sleep(1)
except KeyboardInterrupt:
    print("\nexit")
