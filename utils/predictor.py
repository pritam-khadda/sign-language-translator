import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time
import pyttsx3
import threading


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model.pkl")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# PREDICTION BUFFER
# ============================================================

buffer = deque(maxlen=7)


# ============================================================
# WORD / PREDICTION STATE
# ============================================================

current_word = ""

# Current prediction shown on screen
last_pred = ""

# Candidate gesture which is becoming stable
candidate_pred = ""

# Number of consecutive stable frames
stable_counter = 0

# Prevent same held gesture from being added repeatedly
gesture_locked = False

# Count frames where no hand is detected
no_hand_counter = 0

# Frames required before allowing another gesture
NO_HAND_REQUIRED = 8

# Frames required for a gesture to become stable
STABLE_REQUIRED = 8


# ============================================================
# RESET
# ============================================================

reset_word = False


# ============================================================
# FPS
# ============================================================

prev_time = time.time()


# ============================================================
# SESSION STATS
# ============================================================

session_start_time = time.time()

total_predictions_count = 0

current_confidence = 0.0


# Last accepted gesture
last_added_prediction = ""

# Changes every time a new gesture is accepted
translation_id = 0


def get_session_duration():
    return int(time.time() - session_start_time)


# ============================================================
# VOICE
# ============================================================

voice_enabled = True

last_spoken = ""
last_speak_time = 0

SPEAK_DELAY = 1.2


try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
except Exception as e:
    print("⚠️ Voice engine error:", e)
    engine = None


def toggle_voice():

    global voice_enabled

    voice_enabled = not voice_enabled

    print(
        "🔊 Voice:",
        "Enabled" if voice_enabled else "Disabled"
    )

    return voice_enabled


def speak(text):

    global last_spoken
    global last_speak_time
    global voice_enabled

    if not voice_enabled:
        return

    if not text:
        return

    if engine is None:
        return

    current_time = time.time()

    # Don't speak same prediction repeatedly
    if text == last_spoken:
        return

    # Small delay between speech outputs
    if current_time - last_speak_time < SPEAK_DELAY:
        return

    try:

        engine.say(str(text))
        engine.runAndWait()

        last_spoken = text
        last_speak_time = current_time

    except Exception as e:

        print("⚠️ Voice error:", e)


# ============================================================
# RESET FUNCTION
# ============================================================

def reset():

    global current_word
    global last_pred
    global candidate_pred
    global stable_counter
    global gesture_locked
    global no_hand_counter
    global current_confidence
    global reset_word
    global last_added_prediction

    current_word = ""

    last_pred = ""

    candidate_pred = ""

    stable_counter = 0

    gesture_locked = False

    no_hand_counter = 0

    current_confidence = 0.0

    last_added_prediction = ""

    buffer.clear()

    reset_word = True

    print("🧹 Word reset")


# ============================================================
# ADD ACCEPTED GESTURE
# ============================================================

def accept_gesture(prediction):

    global current_word
    global total_predictions_count
    global last_added_prediction
    global translation_id

    if not prediction:
        return

    prediction_text = str(prediction).strip()

    if not prediction_text:
        return


    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if prediction_text.lower() == "space":

        if current_word and not current_word.endswith(" "):
            current_word += " "

        last_added_prediction = "Space"

        translation_id += 1

        total_predictions_count += 1

        return


    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    if prediction_text.lower() == "delete":

        if current_word:
            current_word = current_word[:-1]

        last_added_prediction = "Delete"

        translation_id += 1

        total_predictions_count += 1

        return


    # --------------------------------------------------------
    # NORMAL GESTURE
    # --------------------------------------------------------

    current_word += prediction_text

    last_added_prediction = prediction_text

    translation_id += 1

    total_predictions_count += 1

    print(
        "✅ Accepted:",
        prediction_text,
        "| Word:",
        current_word
    )

    # Speak only accepted gesture
    speak(prediction_text)


# ============================================================
# MAIN FRAME PROCESSING
# ============================================================

def process_frame(frame):

    global prev_time

    global current_word
    global last_pred
    global candidate_pred
    global stable_counter

    global gesture_locked
    global no_hand_counter

    global reset_word
    global current_confidence


    # ========================================================
    # RESET
    # ========================================================

    if reset_word:

        current_word = ""

        last_pred = ""

        candidate_pred = ""

        stable_counter = 0

        gesture_locked = False

        no_hand_counter = 0

        current_confidence = 0.0

        buffer.clear()

        reset_word = False


    # ========================================================
    # FLIP CAMERA
    # ========================================================

    frame = cv2.flip(frame, 1)


    # ========================================================
    # MEDIAPIPE PROCESSING
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.multi_hand_landmarks:

        # Hand is present again
        no_hand_counter = 0


        for hand in result.multi_hand_landmarks:

            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            h, w, _ = frame.shape

            xs = [
                lm.x
                for lm in hand.landmark
            ]

            ys = [
                lm.y
                for lm in hand.landmark
            ]

            xmin = int(min(xs) * w)
            xmax = int(max(xs) * w)

            ymin = int(min(ys) * h)
            ymax = int(max(ys) * h)


            cv2.rectangle(
                frame,
                (
                    max(0, xmin - 20),
                    max(0, ymin - 20)
                ),
                (
                    min(w, xmax + 20),
                    min(h, ymax + 20)
                ),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # FEATURES
            # ------------------------------------------------

            wrist = hand.landmark[0]

            features = []

            for lm in hand.landmark:

                features.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])


            features = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            try:

                pred = model.predict(
                    features
                )[0]

            except Exception as e:

                print(
                    "⚠️ Prediction error:",
                    e
                )

                current_confidence = 0.0

                continue


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                try:

                    probs = model.predict_proba(
                        features
                    )[0]

                    current_confidence = float(
                        np.max(probs)
                    )

                except Exception:

                    current_confidence = 0.0

            else:

                current_confidence = 1.0


            # ------------------------------------------------
            # BUFFER
            # ------------------------------------------------

            buffer.append(pred)


            # Majority vote
            final_pred = max(
                set(buffer),
                key=buffer.count
            )


            # Current prediction shown on UI
            last_pred = str(final_pred)


            # ------------------------------------------------
            # STABILITY
            # ------------------------------------------------

            if final_pred == candidate_pred:

                stable_counter += 1

            else:

                candidate_pred = final_pred

                stable_counter = 1


            # ------------------------------------------------
            # ACCEPT GESTURE
            # ------------------------------------------------

            if (
                stable_counter >= STABLE_REQUIRED
                and not gesture_locked
            ):

                accept_gesture(
                    final_pred
                )

                # Lock this gesture
                # It cannot be added again until
                # hand is removed.
                gesture_locked = True

                stable_counter = 0


            # ------------------------------------------------
            # PREDICTION TEXT
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction : {final_pred}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # WORD TEXT
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Word : {current_word}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )


            # ------------------------------------------------
            # CONFIDENCE BAR
            # ------------------------------------------------

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
                current_confidence *
                bar_width
            )


            filled = max(
                0,
                min(
                    filled,
                    bar_width
                )
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


            # Only one hand
            break


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        current_confidence = 0.0

        last_pred = ""

        candidate_pred = ""

        stable_counter = 0

        no_hand_counter += 1


        # ----------------------------------------------------
        # UNLOCK AFTER HAND REMOVED
        # ----------------------------------------------------

        if no_hand_counter >= NO_HAND_REQUIRED:

            gesture_locked = False

            buffer.clear()


        # ----------------------------------------------------
        # SCREEN TEXT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Prediction : -",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (150, 150, 150),
            2
        )


        cv2.putText(
            frame,
            f"Word : {current_word}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )


    # ========================================================
    # FPS
    # ========================================================

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


    # ========================================================
    # RETURN FRAME
    # ========================================================

    return frame