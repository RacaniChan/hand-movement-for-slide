import cv2
import time
import pyautogui
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
MODEL_PATH = "yolov8n.pt"   # model default (COCO)
CONFIDENCE = 0.4
MOVE_THRESHOLD = 55        # pixel per gesture
COOLDOWN = 0.6              # detik

# ======================
# LOAD YOLO
# ======================
model = YOLO(MODEL_PATH)
# sess = ort.InferenceSession("models/faceNet.onnx", providers=["CPUExecutionProvider"])

# ======================
# WEBCAM
# ======================
cap = cv2.VideoCapture(0)

prev_center_x = None
last_action_time = 0

print("YOLO Gesture Control Active")
print("Geser tangan ke kanan → Next slide")
print("Geser tangan ke kiri → Previous slide")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # YOLO inference
    results = model(frame, conf=CONFIDENCE, verbose=False)

    if results and len(results[0].boxes) > 0:
        # Ambil box TERBESAR (asumsi tangan paling dekat kamera)
        boxes = results[0].boxes
        best_box = None
        best_area = 0

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO: person = 0
            # Kita pakai tangan yang terlihat (bagian tubuh)
            if conf < CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = best_box
            center_x = (x1 + x2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, (y1 + y2) // 2), 5, (0, 0, 255), -1)

            if prev_center_x is not None:
                diff = center_x - prev_center_x
                now = time.time()

                if abs(diff) > MOVE_THRESHOLD and (now - last_action_time) > COOLDOWN:
                    if diff > 0:
                        pyautogui.press("right")
                        print("➡ Next Slide")
                    else:
                        pyautogui.press("left")
                        print("⬅ Previous Slide")

                    last_action_time = now

            prev_center_x = center_x

    cv2.imshow("YOLO Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()