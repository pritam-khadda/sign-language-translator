import cv2
import mediapipe as mp
import csv

# ---------------- SETTINGS ----------------
label = "Hate/FUCK"  

MAX_SAMPLES = 500
SAVE_EVERY_N_FRAMES = 5
# ------------------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    exit()

print("Camera Opened Successfully")

count = 0
frame_count = 0

with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame_count += 1

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # -------- Feature Extraction --------
                features = []

                # Wrist Normalization
                wrist = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    x = lm.x - wrist.x
                    y = lm.y - wrist.y
                    z = lm.z - wrist.z
                    features.extend([x, y, z])

                # Save only every 5th frame
                if frame_count % SAVE_EVERY_N_FRAMES == 0:

                    features.append(label)
                    writer.writerow(features)

                    count += 1

                    if count >= MAX_SAMPLES:
                        print(f"\n{MAX_SAMPLES} Samples Collected Successfully!")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

        # -------- Display --------

        cv2.putText(
            frame,
            f"Gesture : {label}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples : {count}/{MAX_SAMPLES}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()