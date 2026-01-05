import cv2
import time
import pyautogui
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Konfigurasi
# =========================
SWIPE_THRESHOLD = 50
COOLDOWN_MS = 800
MIN_CONFIDENCE = 0.5

# =========================
# State
# =========================
prev_wrist_x = {"left": None, "right": None}
on_cooldown = False
last_trigger_time = 0

# =========================
# MediaPipe Tasks Init
# =========================
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=MIN_CONFIDENCE,
    min_hand_presence_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def start_cooldown():
    global on_cooldown, last_trigger_time
    on_cooldown = True
    last_trigger_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if on_cooldown:
        if (time.time() - last_trigger_time) * 1000 >= COOLDOWN_MS:
            on_cooldown = False

    result = landmarker.detect(mp_image)

    if result.hand_landmarks and not on_cooldown:
        for landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness
        ):
            label = handedness[0].category_name.lower()
            wrist = landmarks[0]

            wrist_x = int(wrist.x * w)

            if prev_wrist_x[label] is not None:
                delta_x = wrist_x - prev_wrist_x[label]

                if label == "right" and delta_x > SWIPE_THRESHOLD:
                    pyautogui.press("right")
                    start_cooldown()
                    print(label)

                if label == "left" and delta_x < -SWIPE_THRESHOLD:
                    pyautogui.press("left")
                    start_cooldown()
                    print(label)

            prev_wrist_x[label] = wrist_x

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
