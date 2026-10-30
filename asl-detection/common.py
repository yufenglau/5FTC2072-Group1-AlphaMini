"""Shared helpers for the ASL landmark pipeline.

Everything that turns a camera frame into a feature vector lives here, so that
data collection (collect_data.py) and live detection (detect_asl.py) use the
EXACT same processing. Keeping them identical is what makes the trained model
work at detection time - the model only ever sees numbers in the same format
it was trained on.

This uses MediaPipe's current "Tasks" HandLandmarker API. It needs the model
file `hand_landmarker.task` next to this script (already downloaded).
"""
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ASSET_PATH = os.path.join(HERE, "hand_landmarker.task")

# The 26 letters of the ASL fingerspelling alphabet.
# NOTE: in real ASL, J and Z are drawn with motion. Here we capture their
# representative static handshape. See the README for how to extend to motion.
LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# The 21 hand landmarks and how they connect, for drawing the skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


class HandPipeline:
    """Wraps the MediaPipe HandLandmarker and tracks video timestamps for it."""

    def __init__(self, max_hands=1, detection_conf=0.5, tracking_conf=0.5):
        if not os.path.exists(MODEL_ASSET_PATH):
            raise FileNotFoundError(
                f"Missing model file: {MODEL_ASSET_PATH}\n"
                "Download it from:\n"
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_ASSET_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0  # VIDEO mode needs an always-increasing timestamp

    def detect(self, frame_rgb):
        """Run detection on an RGB frame.

        Returns (landmark_list, handedness_label) for the first hand found,
        or (None, None) if no hand is detected. landmark_list is a list of 21
        landmarks, each with .x, .y, .z.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._timestamp_ms += 33  # ~30 fps; exact value doesn't matter, only that it grows
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        landmark_list = result.hand_landmarks[0]
        handedness_label = result.handedness[0][0].category_name  # "Left" / "Right"
        return landmark_list, handedness_label

    def close(self):
        self._detector.close()


def make_hands(max_hands=1, detection_conf=0.5, tracking_conf=0.5):
    """Create the hand-detection pipeline (kept as a function for simple use)."""
    return HandPipeline(max_hands, detection_conf, tracking_conf)


def get_hand(frame_rgb, hands):
    """Run the pipeline on an RGB frame. Returns (landmark_list, handedness_label)."""
    return hands.detect(frame_rgb)


# Fingertip landmark indices (thumb, index, middle, ring, pinky tips).
FINGERTIPS = [4, 8, 12, 16, 20]
# The "knuckle" (MCP) landmark for each finger - used to measure how curled
# a finger is (tip-to-MCP distance shrinks a lot when a finger is bent/closed).
MCPS = [2, 5, 9, 13, 17]


def _extra_distance_angle_features(coords):
    """Supplementary features computed from the already-normalized (21, 3)
    coords: pairwise fingertip distances, fingertip-to-wrist distances, and
    curl (tip-to-MCP distance per finger). These are the features that most
    directly distinguish visually similar letters (e.g. M/N/S, U/V/R) which
    raw normalized coordinates alone don't always separate cleanly for a
    tree-based classifier.
    """
    feats = []

    # Adjacent fingertip-to-fingertip distances (captures spread between
    # fingers - e.g. distinguishes V/U/R which differ mainly in finger gap).
    for a, b in zip(FINGERTIPS, FINGERTIPS[1:]):
        feats.append(float(np.linalg.norm(coords[a] - coords[b])))

    # Fingertip-to-wrist distances (already implicitly in the 63 raw coords,
    # but as an explicit scalar this helps tree splits target it directly).
    for tip in FINGERTIPS:
        feats.append(float(np.linalg.norm(coords[tip])))

    # Per-finger curl: tip-to-MCP distance. A curled/closed finger has a much
    # smaller distance than an extended one - this is what separates letters
    # like M/N/T/S that differ mainly in which fingers are tucked vs raised.
    for tip, mcp in zip(FINGERTIPS, MCPS):
        feats.append(float(np.linalg.norm(coords[tip] - coords[mcp])))

    return np.array(feats, dtype=np.float32)  # shape (14,): 4 + 5 + 5


def extract_features(landmark_list, handedness_label):
    """Turn 21 MediaPipe landmarks into a 77-length feature vector.

    The vector is made invariant to *where* the hand is and *how big* it
    appears, so the model focuses purely on the hand SHAPE:

      1. If it's a left hand, mirror x so left and right hands look the same.
         (One set of training data then covers both hands.)
      2. Move the wrist to the origin            -> position no longer matters.
      3. Scale by the wrist->middle-knuckle gap  -> distance from camera no
                                                    longer matters.

    On top of the 63 raw normalized coordinates, this appends 14 explicit
    distance features (fingertip spread, fingertip-to-wrist, per-finger curl)
    that make finger position/curl an explicit number instead of something
    the classifier has to infer from raw coordinates - this is what usually
    fixes confusion between visually similar letters.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32
    )  # shape (21, 3): 21 landmarks, each with x, y, z

    # 1. Canonicalize handedness.
    if handedness_label == "Left":
        coords[:, 0] = 1.0 - coords[:, 0]

    # 2. Translate so the wrist (landmark 0) is the origin.
    coords -= coords[0]

    # 3. Scale by the wrist -> middle-finger knuckle distance (landmark 9).
    ref = np.linalg.norm(coords[9])
    if ref < 1e-6:
        ref = 1.0
    coords /= ref

    extra = _extra_distance_angle_features(coords)
    return np.concatenate([coords.flatten(), extra])  # shape (77,)


def draw_landmarks(frame_bgr, landmark_list):
    """Draw the hand skeleton onto a BGR frame, for the preview window."""
    h, w = frame_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[a], points[b], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame_bgr, point, 4, (0, 128, 255), -1)
