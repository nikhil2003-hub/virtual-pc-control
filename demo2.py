import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Settings
REQUIRE_FACE = False  # Set True to enable gesture only when face is visible

# Initialize
cv2.setUseOptimized(True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_w, screen_h = pyautogui.size()

# MediaPipe modules
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Cooldowns
cooldown = 1
last_time = {k: 0 for k in ['click', 'scroll', 'volume', 'alt_tab', 'close']}
prev_cursor = [0, 0]
move_thresh = 20

def fingers_up(hand):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)

    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips_ids[i]].y < hand.landmark[tips_ids[i] - 2].y else 0)

    return fingers

def get_distance(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_results = face_detector.process(rgb)
    face_present = False
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
            face_present = True
        cv2.putText(frame, 'Face Detected', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 200), 2)

    if REQUIRE_FACE and not face_present:
        cv2.putText(frame, 'No Face - Gestures Disabled', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.imshow("Gesture + Face Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Hand Tracking
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark
            index_tip = lm[8]
            thumb_tip = lm[4]

            # Cursor Movement
            screen_x, screen_y = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            if abs(prev_cursor[0] - screen_x) > move_thresh or abs(prev_cursor[1] - screen_y) > move_thresh:
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                prev_cursor = [screen_x, screen_y]

            # Visual pointer
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            fingers = fingers_up(handLms)

            # Click
            dist = get_distance(index_tip, thumb_tip, w, h)
            if dist < 30 and time.time() - last_time['click'] > cooldown:
                pyautogui.click()
                last_time['click'] = time.time()
                cv2.putText(frame, 'Click', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Scroll
            if fingers == [0, 1, 0, 0, 0] and time.time() - last_time['scroll'] > 0.4:
                pyautogui.scroll(30)
                last_time['scroll'] = time.time()
                cv2.putText(frame, 'Scroll Up', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif fingers == [0, 0, 0, 0, 0] and index_tip.y > lm[6].y + 0.05 and time.time() - last_time['scroll'] > 0.4:
                pyautogui.scroll(-30)
                last_time['scroll'] = time.time()
                cv2.putText(frame, 'Scroll Down', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Volume
            if fingers == [1, 0, 0, 0, 0]:
                if thumb_tip.y < lm[3].y and time.time() - last_time['volume'] > 0.5:
                    pyautogui.press('volumeup')
                    last_time['volume'] = time.time()
                    cv2.putText(frame, 'Volume Up', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif thumb_tip.y > lm[3].y and time.time() - last_time['volume'] > 0.5:
                    pyautogui.press('volumedown')
                    last_time['volume'] = time.time()
                    cv2.putText(frame, 'Volume Down', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # App switch
            if fingers == [1, 1, 1, 1, 1] and time.time() - last_time['alt_tab'] > 2:
                pyautogui.hotkey('alt', 'tab')
                last_time['alt_tab'] = time.time()
                cv2.putText(frame, 'App Switch', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

            # Close app
            if fingers == [0, 0, 0, 0, 1] and time.time() - last_time['close'] > 2:
                pyautogui.hotkey('alt', 'f4')
                last_time['close'] = time.time()
                cv2.putText(frame, 'Close App', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture + Face Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
