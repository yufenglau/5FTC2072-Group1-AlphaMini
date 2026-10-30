"""STEP 3 - Live ASL detection from your webcam.

Run:  python detect_asl.py

Loads the trained model and shows, in real time, which letter you're signing.
A short rolling vote smooths out the prediction so it doesn't flicker.

Press  q  to quit.
"""
import os
from collections import Counter, deque

import cv2
import joblib

from common import draw_landmarks, extract_features, get_hand, make_hands

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "asl_model.joblib")

CONFIDENCE_THRESHOLD = 0.55  # below this we show "?" instead of guessing
SMOOTHING_WINDOW = 8         # frames to vote over for a stable letter


def main():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run  python train_model.py  first.")
        return

    clf = joblib.load(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open the webcam. Is another app using it?")
        return
    hands = make_hands(max_hands=1)
    recent = deque(maxlen=SMOOTHING_WINDOW)

    print("Detecting. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror -> selfie view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmarks, handedness = get_hand(rgb, hands)

        letter, confidence = "?", 0.0
        if hand_landmarks is not None:
            draw_landmarks(frame, hand_landmarks)
            feats = extract_features(hand_landmarks, handedness).reshape(1, -1)
            probs = clf.predict_proba(feats)[0]
            best = probs.argmax()
            confidence = float(probs[best])
            guess = clf.classes_[best] if confidence >= CONFIDENCE_THRESHOLD else "?"
            recent.append(guess)
            # Show the most common guess over the last few frames (stable).
            letter = Counter(recent).most_common(1)[0][0]
        else:
            recent.clear()

        # Big letter readout.
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Letter: {letter}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.putText(frame, f"{confidence*100:4.0f}%", (frame.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(frame, "q = quit", (frame.shape[1] - 150, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("ASL detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
