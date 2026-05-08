"""
face_lib.py
-----------
Minimal dlib-based face detection and 128-D embedding wrapper.

ATTRIBUTION
~~~~~~~~~~~
The detection / encoding pipeline implemented here is adapted from the
`face_recognition` library by Adam Geitgey (MIT License):
    https://github.com/ageitgey/face_recognition

The pre-trained dlib models loaded below are bundled in this project under
./models/face_recognition_models/, taken from:
    https://github.com/ageitgey/face_recognition_models
also released by Adam Geitgey under the MIT License (see
./models/LICENSE_face_recognition_models).

Only the small slice of the original API used by this project is reproduced
here, so the project can run without the upstream `face_recognition` and
`face_recognition_models` pip packages. dlib itself is still required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import dlib
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Model file locations (from the cloned face_recognition_models repo)
# ---------------------------------------------------------------------------

_MODELS_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "face_recognition_models"
    / "models"
)

_PREDICTOR_68 = _MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
_PREDICTOR_5  = _MODELS_DIR / "shape_predictor_5_face_landmarks.dat"
_RESNET       = _MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"
_CNN_DETECTOR = _MODELS_DIR / "mmod_human_face_detector.dat"

_missing = [p for p in (_PREDICTOR_68, _PREDICTOR_5, _RESNET, _CNN_DETECTOR) if not p.exists()]
if _missing:
    raise FileNotFoundError(
        "Missing dlib model files. Expected to find:\n  "
        + "\n  ".join(str(p) for p in _missing)
        + "\nDownload them from https://github.com/ageitgey/face_recognition_models "
          "and place them under ./models/face_recognition_models/models/."
    )


# ---------------------------------------------------------------------------
# dlib model handles (constructed once at import time)
# ---------------------------------------------------------------------------

_face_detector_hog = dlib.get_frontal_face_detector()
_face_detector_cnn = dlib.cnn_face_detection_model_v1(str(_CNN_DETECTOR))
_pose_predictor_68 = dlib.shape_predictor(str(_PREDICTOR_68))
_pose_predictor_5  = dlib.shape_predictor(str(_PREDICTOR_5))
_face_encoder      = dlib.face_recognition_model_v1(str(_RESNET))


# ---------------------------------------------------------------------------
# CUDA introspection
# dlib transparently uses CUDA for the CNN detector and the ResNet encoder
# IFF it was compiled with CUDA support. The pip wheel on Windows is CPU-only;
# CUDA requires building dlib from source with -DDLIB_USE_CUDA=1.
# ---------------------------------------------------------------------------

DLIB_HAS_CUDA: bool = bool(getattr(dlib, "DLIB_USE_CUDA", False))
try:
    NUM_CUDA_DEVICES: int = dlib.cuda.get_num_devices() if DLIB_HAS_CUDA else 0
except Exception:
    NUM_CUDA_DEVICES = 0


def cuda_status() -> str:
    """Human-readable description of the CUDA acceleration state."""
    if DLIB_HAS_CUDA and NUM_CUDA_DEVICES > 0:
        return f"CUDA enabled ({NUM_CUDA_DEVICES} device{'s' if NUM_CUDA_DEVICES != 1 else ''})"
    if DLIB_HAS_CUDA:
        return "CUDA built but no devices detected"
    return "CPU only (dlib was not built with CUDA)"


def recommended_detection_model() -> str:
    """Return 'cnn' if CUDA can accelerate it, else 'hog' for CPU speed."""
    return "cnn" if DLIB_HAS_CUDA and NUM_CUDA_DEVICES > 0 else "hog"


# ---------------------------------------------------------------------------
# Conversion helpers (between dlib.rectangle and the (top, right, bottom, left)
# tuple format used throughout this project, matching face_recognition's API)
# ---------------------------------------------------------------------------

def _rect_to_tuple(rect: dlib.rectangle) -> Tuple[int, int, int, int]:
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _tuple_to_rect(box: Tuple[int, int, int, int]) -> dlib.rectangle:
    top, right, bottom, left = box
    return dlib.rectangle(left, top, right, bottom)


def _clip_box(box: Tuple[int, int, int, int], shape) -> Tuple[int, int, int, int]:
    return (
        max(box[0], 0),
        min(box[1], shape[1]),
        min(box[2], shape[0]),
        max(box[3], 0),
    )


# ---------------------------------------------------------------------------
# Public API (subset used by the rest of the project)
# ---------------------------------------------------------------------------

def load_image_file(path, mode: str = "RGB") -> np.ndarray:
    """Load an image from disk into an RGB numpy array."""
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return np.array(img)


def face_locations(
    image: np.ndarray,
    number_of_times_to_upsample: int = 1,
    model: str = "hog",
) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return them as (top, right, bottom, left) tuples."""
    if model == "cnn":
        rects = _face_detector_cnn(image, number_of_times_to_upsample)
        return [_clip_box(_rect_to_tuple(r.rect), image.shape) for r in rects]

    rects = _face_detector_hog(image, number_of_times_to_upsample)
    return [_clip_box(_rect_to_tuple(r), image.shape) for r in rects]


def face_encodings(
    image: np.ndarray,
    known_face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
    num_jitters: int = 1,
    landmark_model: str = "small",
) -> List[np.ndarray]:
    """Compute 128-D face embeddings for the supplied (or detected) faces."""
    if known_face_locations is None:
        rects = list(_face_detector_hog(image, 1))
    else:
        rects = [_tuple_to_rect(b) for b in known_face_locations]

    predictor = _pose_predictor_68 if landmark_model == "large" else _pose_predictor_5
    landmarks = [predictor(image, r) for r in rects]
    return [
        np.array(_face_encoder.compute_face_descriptor(image, lm, num_jitters))
        for lm in landmarks
    ]


def match(
    query: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    tolerance: float,
) -> Tuple[str, float]:
    """Return (name, confidence_in_[0,1]). name == 'Unknown' if no match.

    dlib uses Euclidean distance on 128-D embeddings; library default
    tolerance is 0.6 and we use 0.5 by default for stricter matching.
    """
    if not known_encodings:
        return "Unknown", 0.0
    known = np.asarray(known_encodings)
    distances = np.linalg.norm(known - query, axis=1)
    best = int(np.argmin(distances))
    best_dist = float(distances[best])

    if best_dist <= tolerance:
        confidence = max(0.0, 1.0 - best_dist / tolerance)
        return known_names[best], confidence
    return "Unknown", 0.0
