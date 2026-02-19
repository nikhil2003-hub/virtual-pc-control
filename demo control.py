import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Cooldown timers
last_click_time = 0
last_scroll_time = 0
last_volume_time = 0
last_alt_tab_time = 0
last_close_time = 0
cooldown = 1  # seconds

def fingers_up(hand):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: check x for horizontal
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)

    # Other fingers: check y for vertical
    for i in range(1, 5):
        fingers.append(1 if hand.landmark[tips_ids[i]].y < hand.landmark[tips_ids[i] - 2].y else 0)

    return fingers

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger coordinates
            lm = hand_landmarks.landmark
            x, y = int(lm[8].x * w), int(lm[8].y * h)  # Index tip
            screen_x, screen_y = int(lm[8].x * screen_w), int(lm[8].y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Cursor visual
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            fingers = fingers_up(hand_landmarks)

            # Click gesture - Index and Thumb close
            thumb_tip = lm[4]
            index_tip = lm[8]
            distance = math.hypot((thumb_tip.x - index_tip.x) * w, (thumb_tip.y - index_tip.y) * h)

            if distance < 30 and time.time() - last_click_time > cooldown:
                pyautogui.click()
                last_click_time = time.time()
                cv2.putText(frame, 'Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Scroll - Only index up
            if fingers == [0, 1, 0, 0, 0] and time.time() - last_scroll_time > 0.5:
                pyautogui.scroll(30)
                last_scroll_time = time.time()
                cv2.putText(frame, 'Scroll Up', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Scroll - Only index down
            if fingers == [0, 0, 0, 0, 0] and index_tip.y > lm[6].y + 0.05 and time.time() - last_scroll_time > 0.5:
                pyautogui.scroll(-30)
                last_scroll_time = time.time()
                cv2.putText(frame, 'Scroll Down', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Volume Up - Thumb up
            if fingers == [1, 0, 0, 0, 0] and thumb_tip.y < lm[3].y and time.time() - last_volume_time > 0.5:
                pyautogui.press("volumeup")
                last_volume_time = time.time()
                cv2.putText(frame, 'Volume Up', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            # Volume Down - Thumb down
            if fingers == [1, 0, 0, 0, 0] and thumb_tip.y > lm[3].y and time.time() - last_volume_time > 0.5:
                pyautogui.press("volumedown")
                last_volume_time = time.time()
                cv2.putText(frame, 'Volume Down', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            # App switch - All fingers up
            if fingers == [1, 1, 1, 1, 1] and time.time() - last_alt_tab_time > 2:
                pyautogui.hotkey("alt", "tab")
                last_alt_tab_time = time.time()
                cv2.putText(frame, 'App Switch (Alt+Tab)', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)

            # Close app - Only pinky up
            if fingers == [0, 0, 0, 0, 1] and time.time() - last_close_time > 2:
                pyautogui.hotkey("alt", "f4")
                last_close_time = time.time()
                cv2.putText(frame, 'Close App (Alt+F4)', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
