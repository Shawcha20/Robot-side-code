import os, time, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

BASE = "/home/hudai/Desktop/thesis"
ASSETS = os.path.join(BASE, "assets")
POSE_N, HAND_N = 33, 21

pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS, "pose_landmarker_lite.task")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(ASSETS, "hand_landmarker.task")),
    running_mode=vision.RunningMode.IMAGE, num_hands=2))

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(0,11),(0,12)]

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
print("camera open:", cap.isOpened())
print("Press 'q' in the window to quit.")

t_fps = time.time(); n_fps = 0; fps = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pr = pose_lm.detect(img)
        hr = hand_lm.detect(img)

        # draw pose skeleton (blue)
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            pts = [(int(p.x*w), int(p.y*h)) for p in lm]
            for a, b in POSE_CONN:
                cv2.line(frame, pts[a], pts[b], (255, 150, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 3, (255, 200, 0), -1)
            # face box around nose (landmark 0)
            nx, ny = pts[0]
            cv2.rectangle(frame, (nx-60, ny-60), (nx+60, ny+60), (0, 255, 255), 2)
            cv2.putText(frame, "FACE", (nx-55, ny-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # draw hands (green)
        if hr.hand_landmarks:
            for hand in hr.hand_landmarks:
                hp = [(int(p.x*w), int(p.y*h)) for p in hand]
                for a, b in HAND_CONN:
                    cv2.line(frame, hp[a], hp[b], (0, 255, 0), 2)
                for p in hp:
                    cv2.circle(frame, p, 3, (0, 0, 255), -1)

        # fps counter
        n_fps += 1
        if time.time() - t_fps >= 1.0:
            fps = n_fps / (time.time() - t_fps); t_fps = time.time(); n_fps = 0
        cv2.putText(frame, f"fps: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Robot Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
cap.release()
cv2.destroyAllWindows()
print("viewer closed")
