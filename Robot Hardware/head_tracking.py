import time, cv2
from gpiozero import AngularServo

import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ASSETS = "/home/hudai/Desktop/thesis/assets"

pan = AngularServo(22, min_angle=0, max_angle=270,
                   min_pulse_width=0.0005, max_pulse_width=0.0025)
tilt = AngularServo(23, min_angle=0, max_angle=270,
                    min_pulse_width=0.0005, max_pulse_width=0.0025)

CENTER = 135.0
pan_angle = CENTER
tilt_angle = CENTER

# --- anti-jitter tuning (Option B) ---
DEAD_ZONE = 0.10        # bigger -> moves less often near center
KP = 10.0               # gentler proportional
KD = 6.0                # damping
SMOOTH = 0.7            # more smoothing -> steadier input
MAX_STEP = 5.0
MIN_MOVE = 1.5          # ignore small twitchy corrections
PAN_MIN, PAN_MAX = 20, 240
TILT_MIN, TILT_MAX = 60, 210
PAN_DIR = -1            # flip to +1 if pan turns the wrong way
TILT_DIR = 1            # flip to -1 if tilt goes the wrong way
DETACH_AFTER = 0.15     # cut signal quickly after moving -> less holding jitter

pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=ASSETS + "/pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
print("camera open:", cap.isOpened(), "- press q to quit")

pan.angle = pan_angle
tilt.angle = tilt_angle
time.sleep(0.5)
pan.detach()
tilt.detach()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

sx = 0.5; sy = 0.5
prev_dx = 0.0; prev_dy = 0.0
last_pan_move = 0; last_tilt_move = 0
pan_attached = False; tilt_attached = False

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = pose_lm.detect(img)
        now = time.time()

        if res.pose_landmarks:
            nose = res.pose_landmarks[0][0]
            sx = SMOOTH * sx + (1 - SMOOTH) * nose.x
            sy = SMOOTH * sy + (1 - SMOOTH) * nose.y
            dx = sx - 0.5
            dy = sy - 0.5
            ddx = dx - prev_dx
            ddy = dy - prev_dy
            prev_dx = dx
            prev_dy = dy

            if abs(dx) > DEAD_ZONE:
                move = clamp(KP * dx + KD * ddx, -MAX_STEP, MAX_STEP)
                if abs(move) >= MIN_MOVE:
                    pan_angle = clamp(pan_angle + PAN_DIR * move, PAN_MIN, PAN_MAX)
                    pan.angle = pan_angle
                    pan_attached = True; last_pan_move = now

            if abs(dy) > DEAD_ZONE:
                move = clamp(KP * dy + KD * ddy, -MAX_STEP, MAX_STEP)
                if abs(move) >= MIN_MOVE:
                    tilt_angle = clamp(tilt_angle + TILT_DIR * move, TILT_MIN, TILT_MAX)
                    tilt.angle = tilt_angle
                    tilt_attached = True; last_tilt_move = now

            nx, ny = int(nose.x * w), int(nose.y * h)
            cv2.circle(frame, (nx, ny), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"pan {pan_angle:.0f} tilt {tilt_angle:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if pan_attached and now - last_pan_move > DETACH_AFTER:
            pan.detach(); pan_attached = False
        if tilt_attached and now - last_tilt_move > DETACH_AFTER:
            tilt.detach(); tilt_attached = False

        cv2.line(frame, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (0, 255, 0), 1)
        cv2.line(frame, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (0, 255, 0), 1)
        cv2.imshow("Head Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
cap.release()
cv2.destroyAllWindows()
pan.detach()
tilt.detach()
print("stopped")
