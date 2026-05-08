"""
torch_face_lib.py
-----------------
PyTorch backend for face detection and 512-D embeddings, parallel to face_lib.py.

Detection : MTCNN (Multi-task Cascaded CNN, three-stage detector with native
            per-detection confidence scores).
Embeddings: InceptionResnetV1 pretrained on VGGFace2.

Both are provided by `facenet-pytorch` (https://github.com/timesler/facenet-pytorch),
MIT licensed. PyTorch will use CUDA automatically when torch.cuda.is_available().

Install (with CUDA 12.x):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install facenet-pytorch
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA: bool = DEVICE.type == "cuda"
NUM_CUDA_DEVICES: int = torch.cuda.device_count() if HAS_CUDA else 0


# Detection threshold — MTCNN returns probabilities; we only keep boxes above this.
MIN_DETECTION_CONFIDENCE = 0.9

# Default smallest face the detector will report, in pixels of the input frame.
# Smaller values pick up faces from farther away at the cost of speed and
# more false positives. Reconfigure at runtime with `configure(min_face_size=...)`.
_DEFAULT_MIN_FACE_SIZE = 20

# MTCNN is rebuilt lazily whenever min_face_size changes.
_mtcnn_state = {"min_face_size": None, "instance": None}


def _get_mtcnn() -> MTCNN:
    if (_mtcnn_state["instance"] is None
            or _mtcnn_state["min_face_size"] != _DEFAULT_MIN_FACE_SIZE):
        _mtcnn_state["instance"] = MTCNN(
            keep_all=True,
            device=DEVICE,
            select_largest=False,
            post_process=True,
            image_size=160,
            min_face_size=_DEFAULT_MIN_FACE_SIZE,
        )
        _mtcnn_state["min_face_size"] = _DEFAULT_MIN_FACE_SIZE
    return _mtcnn_state["instance"]


def configure(*, min_face_size: Optional[int] = None,
              min_detection_confidence: Optional[float] = None) -> None:
    """Tune detector sensitivity at runtime. Triggers MTCNN rebuild if needed."""
    global _DEFAULT_MIN_FACE_SIZE, MIN_DETECTION_CONFIDENCE
    if min_face_size is not None and min_face_size != _DEFAULT_MIN_FACE_SIZE:
        _DEFAULT_MIN_FACE_SIZE = int(min_face_size)
    if min_detection_confidence is not None:
        MIN_DETECTION_CONFIDENCE = float(min_detection_confidence)


_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# ---------------------------------------------------------------------------
# Status helpers (so the Streamlit sidebar can report what's running)
# ---------------------------------------------------------------------------

def cuda_status() -> str:
    if HAS_CUDA:
        name = torch.cuda.get_device_name(0) if NUM_CUDA_DEVICES else "?"
        return f"CUDA enabled via PyTorch · {name}"
    return "PyTorch CPU (CUDA not available)"


def recommended_detection_model() -> str:
    # MTCNN is always CNN-based; we keep the same string so the existing
    # detection-model selectbox works without conditional UI changes.
    return "cnn"


# ---------------------------------------------------------------------------
# Public API (matches face_lib.py so the rest of the project stays unchanged)
# ---------------------------------------------------------------------------

def load_image_file(path, mode: str = "RGB") -> np.ndarray:
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return np.array(img)


def face_locations(
    image: np.ndarray,
    number_of_times_to_upsample: int = 1,    # accepted for API parity, unused
    model: str = "cnn",                       # accepted for API parity, unused
) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return (top, right, bottom, left) tuples."""
    pil = Image.fromarray(image)
    boxes, probs = _get_mtcnn().detect(pil)
    if boxes is None:
        return []

    h, w = image.shape[:2]
    out: List[Tuple[int, int, int, int]] = []
    for box, prob in zip(boxes, probs):
        if prob is None or prob < MIN_DETECTION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = box
        top    = max(0, int(round(y1)))
        right  = min(w, int(round(x2)))
        bottom = min(h, int(round(y2)))
        left   = max(0, int(round(x1)))
        if bottom <= top or right <= left:
            continue
        out.append((top, right, bottom, left))
    return out


def _crop_to_tensor(image: np.ndarray, box: Tuple[int, int, int, int]) -> torch.Tensor:
    """Crop a face from the image and prepare it for InceptionResnetV1."""
    top, right, bottom, left = box
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        # Return a black 160x160 tensor as a placeholder; caller filters this.
        return torch.zeros(3, 160, 160)
    crop_pil = Image.fromarray(crop).resize((160, 160), Image.BILINEAR)
    arr = np.asarray(crop_pil, dtype=np.float32)
    arr = (arr - 127.5) / 128.0     # facenet-pytorch normalization
    return torch.from_numpy(arr).permute(2, 0, 1)


def face_encodings(
    image: np.ndarray,
    known_face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
    num_jitters: int = 1,             # accepted for API parity, unused
    landmark_model: str = "small",    # accepted for API parity, unused
) -> List[np.ndarray]:
    """Compute 512-D embeddings for the supplied (or detected) faces."""
    if known_face_locations is None:
        known_face_locations = face_locations(image)

    tensors = [_crop_to_tensor(image, b) for b in known_face_locations]
    if not tensors:
        return []

    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        embeddings = _resnet(batch).cpu().numpy()
    return [emb.astype(np.float32) for emb in embeddings]


# ---------------------------------------------------------------------------
# Match function — backends differ in distance metric, so each owns this.
# Cosine distance works well for InceptionResnetV1; threshold ~0.4 is strict,
# 0.6 is loose. Default 0.5 mirrors the dlib backend's tolerance feel.
# ---------------------------------------------------------------------------

def match(
    query: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    tolerance: float,
) -> Tuple[str, float]:
    """Return (name, confidence_in_[0,1]). name == 'Unknown' if no match."""
    if not known_encodings:
        return "Unknown", 0.0
    known = np.asarray(known_encodings, dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-8)
    k = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-8)
    similarities = k @ q
    distances = 1.0 - similarities
    best = int(np.argmin(distances))
    best_dist = float(distances[best])

    if best_dist <= tolerance:
        confidence = max(0.0, 1.0 - best_dist / tolerance)
        return known_names[best], confidence
    return "Unknown", 0.0
