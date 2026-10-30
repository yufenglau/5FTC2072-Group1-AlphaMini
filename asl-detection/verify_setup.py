"""Quick health check - run this if anything seems broken.

Run:  python verify_setup.py

Checks that the libraries import, the shared pipeline loads, and the webcam
opens and delivers a frame that MediaPipe can process.
"""
import sys


def main():
    print(f"Python: {sys.version.split()[0]}")

    import cv2
    import mediapipe as mp
    import numpy as np
    import sklearn
    print(f"OpenCV: {cv2.__version__} | MediaPipe: {mp.__version__} | "
          f"NumPy: {np.__version__} | scikit-learn: {sklearn.__version__}")

    # Shared pipeline imports (this is what collect/detect rely on).
    from common import LABELS, get_hand, make_hands
    print(f"common.py OK - {len(LABELS)} letters: {''.join(LABELS)}")

    # Camera check.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA: could not open webcam (index 0). "
              "Close other apps using the camera, or try changing "
              "cv2.VideoCapture(0) to (1).")
        return
    ok, frame = cap.read()
    if not ok:
        print("CAMERA: opened but failed to read a frame.")
        cap.release()
        return
    h, w = frame.shape[:2]
    print(f"CAMERA: OK - got a {w}x{h} frame")

    # Run MediaPipe once to confirm it processes frames.
    hands = make_hands(max_hands=1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_landmarks, handedness = get_hand(rgb, hands)
    if hand_landmarks is None:
        print("MEDIAPIPE: OK - ran fine (no hand in view, which is expected now)")
    else:
        print(f"MEDIAPIPE: OK - detected a {handedness} hand already!")

    hands.close()
    cap.release()
    print("\nAll good. You're ready: run  python collect_data.py")


if __name__ == "__main__":
    main()
