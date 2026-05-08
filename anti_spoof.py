"""Eye-blink based liveness check.

Holds up a printed photo of a student? They never blink — so attendance
is never marked. Works with both backends since the Eye Aspect Ratio (EAR)
is computed from dlib's 68-point landmark predictor on the original frame.

Reference: Soukupova & Cech, 2016, "Real-Time Eye Blink Detection using
Facial Landmarks". The classic EAR formula is:

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

where p1..p6 are the six landmarks around one eye (corner, top, top, corner,
bottom, bottom). EAR is high (~0.3) when the eye is open, drops sharply
(<0.2) for the few frames during a blink, then returns. Detecting any such
drop is enough — we don't need to count blinks precisely.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import dlib
import numpy as np
from face_lib import _MODELS_DIR


# Use the 68-point predictor (already shipped with the project's dlib models).
_PREDICTOR = dlib.shape_predictor(str(_MODELS_DIR / "shape_predictor_68_face_landmarks.dat"))

# 68-point indices for the two eyes (matching iBUG 300-W convention).
_LEFT_EYE  = (36, 37, 38, 39, 40, 41)
_RIGHT_EYE = (42, 43, 44, 45, 46, 47)

# EAR thresholds. Below `_CLOSED` = eyes shut (blink in progress); above
# `_OPEN` = eyes open. Tuned from the cited paper and dlib's defaults.
_EAR_CLOSED = 0.20
_EAR_OPEN   = 0.25
_HISTORY_LEN = 60   # ~2 seconds at 30 FPS


def _ear_for_eye(landmarks, indices) -> float:
    pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices], dtype=np.float64)
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-6:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def _ear_for_box(frame_bgr: np.ndarray, box_top_right_bottom_left) -> float:
    top, right, bottom, left = box_top_right_bottom_left
    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = _PREDICTOR(frame_bgr, rect)
    left_ear  = _ear_for_eye(landmarks, _LEFT_EYE)
    right_ear = _ear_for_eye(landmarks, _RIGHT_EYE)
    return (left_ear + right_ear) / 2.0


class BlinkDetector:
    """Tracks a 'has blinked at least once' bit per recognized name."""

    def __init__(self) -> None:
        self._history: Dict[str, Deque[float]] = {}
        self._blinked: set = set()

    def reset(self) -> None:
        self._history.clear()
        self._blinked.clear()

    def has_blinked(self, name: str) -> bool:
        return name in self._blinked

    def update(
        self,
        frame_bgr: np.ndarray,
        recognized: list,        # list of RecognitionResult
    ) -> Dict[str, float]:
        """Compute EAR for each recognized face, updating per-name state.

        Returns the EAR observed this frame keyed by name (for UI display).
        """
        ears: Dict[str, float] = {}
        for r in recognized:
            if r.name == "Unknown":
                continue
            try:
                ear = _ear_for_box(frame_bgr, r.box)
            except Exception:
                continue
            ears[r.name] = ear

            history = self._history.setdefault(r.name, deque(maxlen=_HISTORY_LEN))
            history.append(ear)

            if r.name not in self._blinked and len(history) >= 4:
                # Saw a clear closed-eye sample at any point in the recent window?
                if min(history) < _EAR_CLOSED and max(history) > _EAR_OPEN:
                    self._blinked.add(r.name)
        return ears
