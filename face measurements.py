import cv2
import mediapipe as mp
import math

def distance(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            lm = face.landmark

            # Measure distance between eyes (landmarks 33 and 263)
            eye_dist = distance(lm[33], lm[263], w, h)

            # Measure face width (landmarks 234 to 454)
            face_width = distance(lm[234], lm[454], w, h)

            # Measure face height (landmarks 10 to 152)
            face_height = distance(lm[10], lm[152], w, h)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS,
                                   landmark_drawing_spec=None,
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1))

            # Display measurements
            cv2.putText(frame, f'Eye Distance: {int(eye_dist)} px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Face Width: {int(face_width)} px', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Face Height: {int(face_height)} px', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Measurement", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
