import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time
import pyttsx3

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Text-to-Speech Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)

last_spoken = None
last_speak_time = 0
SPEAK_DELAY = 1.2  # seconds

def speak(text):
    global last_spoken, last_speak_time

    current_time = time.time()

    # prevent spam + repeat voice
    if text != last_spoken and (current_time - last_speak_time) > SPEAK_DELAY:
        engine.say(text)
        engine.runAndWait()

        last_spoken = text
        last_speak_time = current_time


# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

buffer = deque(maxlen=5)

prev_time = time.time()

# -----------------------------
# Main Loop
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------
            # Bounding Box
            # -----------------------------
            h, w, _ = frame.shape

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            xmin = int(min(xs) * w)
            xmax = int(max(xs) * w)
            ymin = int(min(ys) * h)
            ymax = int(max(ys) * h)

            cv2.rectangle(
                frame,
                (xmin - 20, ymin - 20),
                (xmax + 20, ymax + 20),
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Feature Extraction
            # -----------------------------
            wrist = hand.landmark[0]

            features = []
            for lm in hand.landmark:
                features.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            features = np.array(features).reshape(1, -1)

            # -----------------------------
            # Prediction
            # -----------------------------
            pred = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = np.max(probs)
            else:
                confidence = 1.0

            buffer.append(pred)
            final_pred = max(set(buffer), key=buffer.count)

            # 🔊 VOICE OUTPUT
            speak(final_pred)

            # -----------------------------
            # Display Prediction
            # -----------------------------
            cv2.putText(
                frame,
                f"Prediction : {final_pred}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # -----------------------------
            # Confidence Bar
            # -----------------------------
            bar_x, bar_y = 20, 60
            bar_width, bar_height = 250, 20

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height),
                          (255, 255, 255), 2)

            filled_width = int(confidence * bar_width)

            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + filled_width, bar_y + bar_height),
                          (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"{confidence*100:.1f}%",
                (bar_x + bar_width + 10, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # -----------------------------
    # FPS Counter
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()