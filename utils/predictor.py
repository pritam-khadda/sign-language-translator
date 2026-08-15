import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time
import pyttsx3

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")


# -----------------------------
# MEDIAPIPE
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
# BUFFER
# -----------------------------
buffer = deque(maxlen=7)


# -----------------------------
# WORD BUILDING
# -----------------------------
current_word = ""
last_pred = ""
stable_counter = 0


# -----------------------------
# RESET FLAG
# -----------------------------
reset_word = False


# -----------------------------
# FPS
# -----------------------------
prev_time = time.time()


# -----------------------------
# SESSION STATS
# -----------------------------
session_start_time = time.time()
total_predictions_count = 0
current_confidence = 0.0


def get_session_duration():
    return int(time.time() - session_start_time)


# -----------------------------
# VOICE SETUP
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)

last_spoken = ""
last_speak_time = 0
SPEAK_DELAY = 1.2

# Voice toggle
voice_enabled = True


# -----------------------------
# VOICE TOGGLE
# -----------------------------
def toggle_voice():
    global voice_enabled

    voice_enabled = not voice_enabled

    return voice_enabled


# -----------------------------
# SPEAK FUNCTION
# -----------------------------
def speak(text):
    global last_spoken, last_speak_time, voice_enabled

    if not voice_enabled:
        return

    current_time = time.time()

    if text != last_spoken and (
        current_time - last_speak_time
    ) > SPEAK_DELAY:

        engine.say(text)
        engine.runAndWait()

        last_spoken = text
        last_speak_time = current_time


# -----------------------------
# RESET FUNCTION
# -----------------------------
def reset():

    global current_word
    global last_pred
    global stable_counter
    global reset_word
    global current_confidence

    current_word = ""
    last_pred = ""
    stable_counter = 0
    current_confidence = 0.0

    reset_word = True


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def process_frame(frame):

    global prev_time
    global current_word
    global last_pred
    global stable_counter
    global reset_word
    global current_confidence
    global total_predictions_count


    # -----------------------------
    # RESET
    # -----------------------------
    if reset_word:

        current_word = ""
        last_pred = ""
        stable_counter = 0
        current_confidence = 0.0

        buffer.clear()

        reset_word = False


    # -----------------------------
    # IMAGE PROCESSING
    # -----------------------------
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    final_pred = ""


    # -----------------------------
    # HAND DETECTED
    # -----------------------------
    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            # -----------------------------
            # BOUNDING BOX
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
            # FEATURES
            # -----------------------------
            wrist = hand.landmark[0]

            features = []

            for lm in hand.landmark:

                features.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            features = np.array(
                features
            ).reshape(1, -1)


            # -----------------------------
            # MODEL PREDICTION
            # -----------------------------
            pred = model.predict(features)[0]


            # -----------------------------
            # CONFIDENCE
            # -----------------------------
            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(features)[0]

                current_confidence = float(
                    np.max(probs)
                )

            else:

                current_confidence = 1.0


            # -----------------------------
            # BUFFER
            # -----------------------------
            buffer.append(pred)

            final_pred = max(
                set(buffer),
                key=buffer.count
            )


            # -----------------------------
            # STABILITY
            # -----------------------------
            if final_pred == last_pred:

                stable_counter += 1

            else:

                stable_counter = 0
                last_pred = final_pred


            # -----------------------------
            # ADD LETTER
            # -----------------------------
            if stable_counter == 10:

                current_word += final_pred

                # Count stable predictions
                total_predictions_count += 1

                stable_counter = 0

                # Voice
                speak(final_pred)


            # -----------------------------
            # PREDICTION TEXT
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
            # WORD TEXT
            # -----------------------------
            cv2.putText(
                frame,
                f"Word : {current_word}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )


            # -----------------------------
            # CONFIDENCE BAR
            # -----------------------------
            bar_x = 20
            bar_y = 60

            bar_width = 250
            bar_height = 20


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),
                (255, 255, 255),
                2
            )


            filled = int(
                current_confidence * bar_width
            )


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + filled,
                    bar_y + bar_height
                ),
                (0, 255, 0),
                -1
            )


            cv2.putText(
                frame,
                f"{current_confidence * 100:.1f}%",
                (
                    bar_x + bar_width + 10,
                    bar_y + 17
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    else:

        # No hand detected
        current_confidence = 0.0


    # -----------------------------
    # FPS
    # -----------------------------
    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

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


    return frame