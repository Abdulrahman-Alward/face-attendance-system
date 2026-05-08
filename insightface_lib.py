"""
insightface_lib.py
------------------
ArcFace backend via the `insightface` library.

Detection + recognition: `insightface.app.FaceAnalysis`, which packages
SCRFD detection (with native confidence scores) and ArcFace 512-D
embeddings into a single forward pass. Uses CUDA via onnxruntime-gpu
when available, falls back to CPU otherwise.

Install:
    pip install insightface onnxruntime-gpu     # CUDA
    pip install insightface onnxruntime          # CPU only

On first use it downloads the `buffalo_l` model pack (~280 MB) into
`%USERPROFILE%\\.insightface\\models\\`. Subsequent runs are instant.

ArcFace embeddings are L2-normalized 512-D vectors; cosine distance is
the natural metric.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import onnxruntime as _ort
from insightface.app import FaceAnalysis


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

_AVAILABLE_PROVIDERS = _ort.get_available_providers()
HAS_CUDA: bool = "CUDAExecutionProvider" in _AVAILABLE_PROVIDERS
NUM_CUDA_DEVICES: int = 1 if HAS_CUDA else 0   # FaceAnalysis uses ctx_id=0


_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if HAS_CUDA \
    else ["CPUExecutionProvider"]

# `buffalo_l` is the standard ArcFace+SCRFD pack. Detection threshold is
# tuned conservatively; users can tighten it via `configure()`.
_MIN_DETECTION_CONFIDENCE = 0.5

_app = FaceAnalysis(name="buffalo_l", providers=_PROVIDERS)
_app.prepare(ctx_id=0 if HAS_CUDA else -1, det_size=(640, 640))


# ---------------------------------------------------------------------------
# Same-frame cache: SCRFD does detection + ArcFace does encoding in a single
# pass. Calling face_locations() then face_encodings() on the same frame
# would otherwise re-run the whole pipeline. The cache shortcuts the second
# call when the caller hands us the same image object.
# ---------------------------------------------------------------------------

_FRAME_CACHE = {"image_id": None, "boxes": [], "embeddings": []}


def _iou(a, b) -> float:
    a_top, a_right, a_bottom, a_left = a
    b_top, b_right, b_bottom, b_left = b
    inter_top    = max(a_top,    b_top)
    inter_left   = max(a_left,   b_left)
    inter_bottom = min(a_bottom, b_bottom)
    inter_right  = min(a_right,  b_right)
    if inter_bottom <= inter_top or inter_right <= inter_left:
        return 0.0
    inter = (inter_bottom - inter_top) * (inter_right - inter_left)
    a_area = max(0, (a_bottom - a_top)) * max(0, (a_right - a_left))
    b_area = max(0, (b_bottom - b_top)) * max(0, (b_right - b_left))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def cuda_status() -> str:
    if HAS_CUDA:
        return "CUDA enabled via insightface · onnxruntime-gpu"
    return "insightface on CPU (install onnxruntime-gpu for CUDA)"


def recommended_detection_model() -> str:
    return "cnn"


def configure(*, min_face_size: Optional[int] = None,
              min_detection_confidence: Optional[float] = None) -> None:
    """Adjust detector sensitivity. min_face_size is a hint only; SCRFD doesn't
    expose a hard cutoff, so we filter by detection score instead."""
    global _MIN_DETECTION_CONFIDENCE
    if min_detection_confidence is not None:
        _MIN_DETECTION_CONFIDENCE = float(min_detection_confidence)
    # min_face_size: insightface scales to det_size internally; smaller faces
    # come through with lower scores. To be more permissive for distant faces,
    # nudge the score threshold down a bit.
    if min_face_size is not None and min_face_size <= 16:
        _MIN_DETECTION_CONFIDENCE = min(_MIN_DETECTION_CONFIDENCE, 0.4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image_file(path, mode: str = "RGB") -> np.ndarray:
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return np.array(img)


def _detect_and_encode(image_rgb: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """Run SCRFD + ArcFace once, returning (boxes, embeddings) in the same order."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = _app.get(bgr)
    h, w = image_rgb.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    embs:  List[np.ndarray] = []
    for f in faces:
        if float(f.det_score) < _MIN_DETECTION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = f.bbox
        top    = max(0, int(round(y1)))
        right  = min(w, int(round(x2)))
        bottom = min(h, int(round(y2)))
        left   = max(0, int(round(x1)))
        if bottom <= top or right <= left:
            continue
        boxes.append((top, right, bottom, left))
        embs.append(np.asarray(f.embedding, dtype=np.float32))
    return boxes, embs


def face_locations(
    image: np.ndarray,
    number_of_times_to_upsample: int = 1,    # API parity, unused
    model: str = "cnn",                       # API parity, unused
) -> List[Tuple[int, int, int, int]]:
    boxes, embs = _detect_and_encode(image)
    _FRAME_CACHE["image_id"]   = id(image)
    _FRAME_CACHE["boxes"]      = boxes
    _FRAME_CACHE["embeddings"] = embs
    return boxes


def face_encodings(
    image: np.ndarray,
    known_face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
    num_jitters: int = 1,             # API parity, unused
    landmark_model: str = "small",    # API parity, unused
) -> List[np.ndarray]:
    # Same-frame cache hit — match the requested boxes against detected ones.
    if _FRAME_CACHE["image_id"] == id(image) and known_face_locations is not None:
        out: List[np.ndarray] = []
        for box in known_face_locations:
            best_iou = 0.3
            best_idx = -1
            for i, cached_box in enumerate(_FRAME_CACHE["boxes"]):
                iou = _iou(box, cached_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                out.append(_FRAME_CACHE["embeddings"][best_idx])
        return out

    # Cold path (e.g. `build_database` calling on a freshly loaded image
    # without the locations cache having been seeded).
    _, embs = _detect_and_encode(image)
    return embs


def match(
    query: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    tolerance: float,
) -> Tuple[str, float]:
    """Cosine-distance match. ArcFace embeddings come pre-normalized but we
    re-normalize defensively; cost is negligible."""
    if not known_encodings:
        return "Unknown", 0.0
    known = np.asarray(known_encodings, dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-8)
    k = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-8)
    distances = 1.0 - (k @ q)
    best = int(np.argmin(distances))
    best_dist = float(distances[best])

    if best_dist <= tolerance:
        confidence = max(0.0, 1.0 - best_dist / tolerance)
        return known_names[best], confidence
    return "Unknown", 0.0
