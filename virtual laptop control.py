import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe and webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()
last_click_time = 0
click_cooldown = 1  # seconds
last_alt_tab_time = 0
last_close_time = 0

def fingers_up(hand):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    for i in tips_ids:
        # Check if finger is up
        if i == 4:
            # Thumb (x comparison for horizontal)
            fingers.append(1 if hand.landmark[i].x < hand.landmark[i - 1].x else 0)
        else:
            # Other fingers (y comparison for vertical)
            fingers.append(1 if hand.landmark[i].y < hand.landmark[i - 2].y else 0)
    return fingers

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of fingertips
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Draw cursor circle
            cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)

            # Check click gesture (index tip near thumb tip)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            distance = math.hypot(thumb_x - x, thumb_y - y)
            if distance < 40:
                if time.time() - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, 'Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Scroll gesture (middle finger high or low)
            if middle_tip.y < 0.3:
                pyautogui.scroll(40)
                cv2.putText(frame, 'Scroll Up', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif middle_tip.y > 0.7:
                pyautogui.scroll(-40)
                cv2.putText(frame, 'Scroll Down', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Volume control using index finger up/down
            if index_tip.y < hand_landmarks.landmark[6].y:
                pyautogui.press("volumeup")
                cv2.putText(frame, 'Volume Up', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif index_tip.y > hand_landmarks.landmark[6].y:
                pyautogui.press("volumedown")
                cv2.putText(frame, 'Volume Down', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # App Switcher (All fingers up)
            fingers = fingers_up(hand_landmarks)
            if fingers == [1, 1, 1, 1, 1]:
                if time.time() - last_alt_tab_time > 2:
                    pyautogui.hotkey('alt', 'tab')
                    last_alt_tab_time = time.time()
                    cv2.putText(frame, 'App Switch (Alt+Tab)', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

            # Close app (only pinky finger up)
            if fingers == [0, 0, 0, 0, 1]:
                if time.time() - last_close_time > 2:
                    pyautogui.hotkey('alt', 'f4')
                    last_close_time = time.time()
                    cv2.putText(frame, 'Close App (Alt+F4)', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
