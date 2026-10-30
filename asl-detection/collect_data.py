"""STEP 1 - Collect training data from your webcam.

How to use:
  1. Run:  python collect_data.py
  2. Form an ASL letter with one hand in view of the camera.
  3. TAP that letter's key on your keyboard (e.g. press 'a' while signing A).
     It records a short burst of samples for that letter.
  4. Repeat each letter a few times, moving your hand a bit (closer/farther,
     slightly different angle) so the model learns to be flexible.
  5. Press  q  when done.

All samples are appended to data/landmarks.csv. You can run this again later to
add more data - it keeps what you already collected.
"""
import csv
import os

import cv2

from common import LABELS, draw_landmarks, extract_features, get_hand, make_hands

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "landmarks.csv")
SAMPLES_PER_BURST = 50  # how many frames one key-tap records


def count_existing():
    """Count how many samples we already have for each letter."""
    counts = {label: 0 for label in LABELS}
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


def draw_hud(frame, counts, recording_label, remaining):
    """Draw on-screen instructions and progress."""
    h = frame.shape[0]
    cv2.putText(frame, "Tap a-z to record a letter   |   q = quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    total = sum(counts.values())
    cv2.putText(frame, f"Total samples: {total}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    missing = [l for l in LABELS if counts[l] == 0]
    msg = "Still need: " + (" ".join(missing) if missing else "all letters have data!")
    cv2.putText(frame, msg, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    if recording_label is not None and remaining > 0:
        cv2.putText(frame, f"REC {recording_label}  ({remaining} left)",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    counts = count_existing()

    new_file = not os.path.exists(DATA_PATH)
    csv_file = open(DATA_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open the webcam. Is another app using it?")
        return
    hands = make_hands(max_hands=1)

    recording_label = None
    remaining = 0

    print("Collecting. Tap a letter key to record a burst. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror -> feels like a selfie
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmarks, handedness = get_hand(rgb, hands)

        if hand_landmarks is not None:
            draw_landmarks(frame, hand_landmarks)
            if recording_label is not None and remaining > 0:
                feats = extract_features(hand_landmarks, handedness)
                writer.writerow([recording_label] + feats.tolist())
                counts[recording_label] += 1
                remaining -= 1
                if remaining == 0:
                    recording_label = None

        draw_hud(frame, counts, recording_label, remaining)
        cv2.imshow("Collect ASL data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("a") <= key <= ord("z"):
            # Only start a new burst if we're not already mid-burst.
            if remaining == 0:
                recording_label = chr(key).upper()
                remaining = SAMPLES_PER_BURST

    csv_file.close()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    final = count_existing()
    print("\nDone. Samples per letter:")
    for label in LABELS:
        print(f"  {label}: {final[label]}")
    print(f"\nSaved to {DATA_PATH}")
    print("Next: run  python train_model.py")


if __name__ == "__main__":
    main()
