"""Core face-recognition and presence-tracking logic for the attendance system.

Backends (selected at runtime via `Config.backend`):
    dlib    — HOG / dlib-CNN detection + 128-D ResNet embeddings
    torch   — MTCNN + 512-D InceptionResnetV1 (FaceNet), CUDA-accelerated
    arcface — SCRFD + 512-D ArcFace via insightface, CUDA-accelerated

Provides:
    Config              - centralized settings (with JSON persistence)
    FaceEncoder         - builds and persists face embeddings
    TrackingRecognizer  - per-frame recognition with IoU tracking and
                          a same-frame cache that skips re-encoding stable faces
    PresenceTracker     - logs attendance, ENTER and LEAVE events,
                          gates confirmation on multi-frame agreement
                          and (optionally) liveness checks
    annotate_frame      - draws boxes and labels onto a BGR frame
"""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

import face_lib  # local dlib wrapper, see face_lib.py

# Optional PyTorch / facenet-pytorch backend. Imported lazily so the project
# still works on machines without torch installed.
try:
    import torch_face_lib
    TORCH_BACKEND_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:           # noqa: BLE001 — torch import has many failure modes
    torch_face_lib = None
    TORCH_BACKEND_AVAILABLE = False
    _TORCH_IMPORT_ERROR = str(_exc)

# Optional ArcFace / insightface backend.
try:
    import insightface_lib
    INSIGHTFACE_BACKEND_AVAILABLE = True
    _INSIGHTFACE_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    insightface_lib = None
    INSIGHTFACE_BACKEND_AVAILABLE = False
    _INSIGHTFACE_IMPORT_ERROR = str(_exc)


def get_backend(name: str):
    """Return the backend module for ``name`` ('dlib', 'torch', or 'arcface')."""
    if name == "torch":
        if not TORCH_BACKEND_AVAILABLE:
            raise RuntimeError(
                "PyTorch backend is not available. "
                f"Install with `pip install facenet-pytorch torch torchvision` "
                f"(import error: {_TORCH_IMPORT_ERROR})"
            )
        return torch_face_lib
    if name == "arcface":
        if not INSIGHTFACE_BACKEND_AVAILABLE:
            raise RuntimeError(
                "ArcFace backend is not available. "
                f"Install with `pip install insightface onnxruntime-gpu` "
                f"(import error: {_INSIGHTFACE_IMPORT_ERROR})"
            )
        return insightface_lib
    return face_lib


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    dataset_dir: Optional[Path] = None
    encodings_file: Optional[Path] = None
    attendance_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    # Backend: "dlib" (HOG / CNN via dlib, CPU unless dlib was built with CUDA)
    # or "torch" (MTCNN + InceptionResnetV1, runs on CUDA via PyTorch when
    # available). The two produce different-dimension embeddings, so each
    # backend stores its database in a separate file.
    backend: str = "dlib"

    # Detection — auto-set by apply_backend_defaults():
    #   dlib  → "hog"   (CPU friendly)
    #   torch → "cnn"   (MTCNN, CUDA-accelerated via PyTorch)
    detection_model: str = "hog"
    upsample_times: int = 1

    # Smallest face the torch / MTCNN detector will report, in pixels.
    # Lower = picks up far-away faces, slower, more false positives.
    min_face_size: int = 20

    # A face has to be recognized this many consecutive frames before it counts
    # toward attendance. Reduces false positives from a single bad frame.
    # Re-entries after a LEAVE only need 1 frame (they've already been confirmed).
    consecutive_recognition_threshold: int = 5

    # Class start time as "HH:MM" (24-h). When set, attendance gets a Status
    # column of "ON_TIME" / "LATE" depending on the grace period below.
    class_start_time: Optional[str] = None
    late_grace_minutes: int = 5

    # When True, training images are encoded with horizontal-flip and
    # brightness-shifted variants too (4 embeddings per image instead of 1).
    # Improves matching robustness to lighting and angle at zero query-time
    # cost beyond an extra distance computation.
    augment_during_enrollment: bool = True

    # When True, attendance is only marked once a recognized person also blinks
    # (defense against held-up photos). Adds CPU cost per detected face.
    require_blink_to_mark: bool = False

    # Recognition tolerance — interpretation differs per backend:
    #   dlib  : Euclidean distance on 128-D embeddings (~0.5 default).
    #   torch : cosine distance on 512-D embeddings   (~0.5 default).
    recognition_tolerance: float = 0.5
    num_jitters: int = 1

    # Real-time pipeline. frame_resize_factor is auto-set per backend by
    # apply_backend_defaults(); torch on CUDA processes full-res frames so
    # faces stay detectable from distance, while dlib HOG needs downscaling
    # to keep up on CPU.
    frame_resize_factor: float = 1.0
    target_fps: float = 30.0            # display loop is paced to this rate
    camera_index: int = 0

    # IoU-based face tracking with a centroid-distance fallback. Detections
    # that overlap a previous recognition (IoU) — or, failing that, whose
    # centroid moved less than half a face-width — reuse that cached
    # identity without re-encoding. Identities are still re-confirmed every
    # `reidentify_every_seconds` to handle people swapping seats / drift.
    iou_match_threshold: float = 0.4
    reidentify_every_seconds: float = 20.0

    # A person not seen for this many minutes is considered to have LEFT.
    leave_timeout_minutes: float = 1.0

    @property
    def leave_timeout_seconds(self) -> float:
        return self.leave_timeout_minutes * 60.0

    def __post_init__(self) -> None:
        data_root = self.project_root / "data"
        if self.dataset_dir is None:
            self.dataset_dir = data_root / "dataset"
        if self.attendance_dir is None:
            self.attendance_dir = data_root / "attendance_reports"
        if self.logs_dir is None:
            self.logs_dir = data_root / "logs"

        for d in (self.dataset_dir, self.attendance_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def encodings_path(self) -> Path:
        """Backend-specific encodings file (embeddings differ in dimensionality)."""
        if self.encodings_file is not None:
            return self.encodings_file
        return self.project_root / "data" / f"encodings_{self.backend}.pkl"

    # Fields that survive across launches via data/settings.json. Excludes
    # paths (derived) and per-session values like session_label.
    _PERSISTED_FIELDS = (
        "backend",
        "detection_model", "upsample_times", "min_face_size",
        "recognition_tolerance", "num_jitters",
        "frame_resize_factor", "target_fps", "camera_index",
        "iou_match_threshold", "reidentify_every_seconds",
        "leave_timeout_minutes",
        "consecutive_recognition_threshold",
        "class_start_time", "late_grace_minutes",
        "augment_during_enrollment",
        "require_blink_to_mark",
    )

    def save_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f: getattr(self, f) for f in self._PERSISTED_FIELDS}
        path.write_text(json.dumps(data, indent=2))

    def load_from(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        for k, v in data.items():
            if k in self._PERSISTED_FIELDS:
                setattr(self, k, v)

    def apply_backend_defaults(self) -> None:
        """Reset detector / resolution settings to sensible per-backend values.

        Called when the backend changes via the UI. dlib gets HOG + downscale
        so it stays fast on CPU; torch and arcface get full-resolution CNN
        detection so faces from across the room are still detectable.
        """
        if self.backend == "torch":
            self.detection_model = "cnn"
            self.frame_resize_factor = 1.0
            self.min_face_size = 20
        elif self.backend == "arcface":
            self.detection_model = "cnn"
            self.frame_resize_factor = 1.0
            self.min_face_size = 20
        else:
            self.detection_model = "hog"
            self.frame_resize_factor = 0.5
            self.upsample_times = 1


# ---------------------------------------------------------------------------
# Recognition data class
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    name: str
    box: Tuple[int, int, int, int]   # (top, right, bottom, left) on the original frame
    confidence: float                # in [0, 1]; higher is better


# ---------------------------------------------------------------------------
# Face encoder: builds and persists the embedding database
# ---------------------------------------------------------------------------

class FaceEncoder:
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend = get_backend(config.backend)
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []

    @property
    def people(self) -> List[str]:
        return sorted(set(self.names))

    def inspect_dataset(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        if not self.config.dataset_dir.exists():
            return summary
        for person_dir in sorted(self.config.dataset_dir.iterdir()):
            if person_dir.is_dir():
                images = [f for f in person_dir.iterdir() if f.suffix.lower() in self.VALID_EXT]
                summary[person_dir.name] = len(images)
        return summary

    def build_database(
        self,
        force_rebuild: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, int]:
        """Encode every image in the dataset. Returns {encoded, skipped, people}."""
        stats = {"encoded": 0, "skipped": 0, "people": 0}

        if self.config.encodings_path.exists() and not force_rebuild:
            self.load()
            stats["encoded"] = len(self.encodings)
            stats["people"] = len(set(self.names))
            return stats

        self.encodings.clear()
        self.names.clear()

        people_dirs = [d for d in self.config.dataset_dir.iterdir() if d.is_dir()]
        if not people_dirs:
            return stats

        for i, person_dir in enumerate(people_dirs):
            if progress_callback:
                progress_callback(i, len(people_dirs), person_dir.name)

            person_name = person_dir.name
            images = [f for f in person_dir.iterdir() if f.suffix.lower() in self.VALID_EXT]

            for img_path in images:
                try:
                    image = self.backend.load_image_file(str(img_path))
                    variants = self._enrollment_variants(image)

                    encoded_at_least_once = False
                    for variant in variants:
                        boxes = self.backend.face_locations(
                            variant, model=self.config.detection_model
                        )
                        if not boxes:
                            continue
                        if len(boxes) > 1:
                            boxes = [max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))]

                        encs = self.backend.face_encodings(
                            variant, boxes, num_jitters=self.config.num_jitters
                        )
                        for enc in encs:
                            self.encodings.append(enc)
                            self.names.append(person_name)
                            encoded_at_least_once = True

                    if not encoded_at_least_once:
                        stats["skipped"] += 1
                except Exception:
                    stats["skipped"] += 1

        if progress_callback:
            progress_callback(len(people_dirs), len(people_dirs), "")

        self.save()
        stats["encoded"] = len(self.encodings)
        stats["people"] = len(set(self.names))
        return stats

    def _enrollment_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Return [original] or [original + augmentations] depending on config."""
        if not self.config.augment_during_enrollment:
            return [image]
        flipped = cv2.flip(image, 1)
        bright  = cv2.convertScaleAbs(image, alpha=1.20, beta=10)
        dim     = cv2.convertScaleAbs(image, alpha=0.80, beta=-10)
        return [image, flipped, bright, dim]

    def save(self) -> None:
        with open(self.config.encodings_path, "wb") as f:
            pickle.dump({"encodings": self.encodings, "names": self.names}, f)

    def delete_person(self, name: str) -> bool:
        """Remove a person's images and embeddings. Returns True if anything was deleted."""
        import shutil
        removed = False

        person_dir = self.config.dataset_dir / name
        if person_dir.exists() and person_dir.is_dir():
            shutil.rmtree(person_dir)
            removed = True

        kept_pairs = [(e, n) for e, n in zip(self.encodings, self.names) if n != name]
        if len(kept_pairs) != len(self.names):
            self.encodings = [e for e, _ in kept_pairs]
            self.names = [n for _, n in kept_pairs]
            self.save()
            removed = True

        return removed

    def load(self) -> None:
        if not self.config.encodings_path.exists():
            self.encodings, self.names = [], []
            return
        with open(self.config.encodings_path, "rb") as f:
            data = pickle.load(f)
        self.encodings = data["encodings"]
        self.names = data["names"]


# ---------------------------------------------------------------------------
# Tracking recognizer: re-uses identities by IoU instead of re-encoding
# ---------------------------------------------------------------------------

@dataclass
class _TrackedFace:
    box: Tuple[int, int, int, int]   # (top, right, bottom, left) on the original frame
    name: str
    confidence: float
    last_encoded_at: float           # epoch seconds


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (top, right, bottom, left) boxes."""
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


def _centroid_distance_normalized(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    """Distance between box centroids, divided by the average box width.

    Used as a fallback when IoU drops too low to match (e.g. a student
    turns their head): if the centroid moved less than ~half a face
    width, it's almost certainly still the same person.
    """
    a_top, a_right, a_bottom, a_left = a
    b_top, b_right, b_bottom, b_left = b
    ax = (a_left + a_right) * 0.5
    ay = (a_top  + a_bottom) * 0.5
    bx = (b_left + b_right) * 0.5
    by = (b_top  + b_bottom) * 0.5
    dx, dy = ax - bx, ay - by
    avg_w = (((a_right - a_left) + (b_right - b_left)) * 0.5) or 1.0
    return (dx * dx + dy * dy) ** 0.5 / avg_w


class TrackingRecognizer:
    """Faster recognizer for live video.

    On every frame we detect faces (cheap with HOG). For each detection we
    look for a previously-tracked face with high IoU; if found AND the
    identity is recent, we reuse the cached identity without re-encoding.
    Only genuinely new faces — or tracked faces older than the re-identify
    window — go through the slow encode + match step.
    """

    def __init__(self, encoder: "FaceEncoder", config: Config) -> None:
        self.encoder = encoder
        self.config = config
        self.backend = get_backend(config.backend)
        self._tracks: List[_TrackedFace] = []

    def reset(self) -> None:
        self._tracks = []

    def process(self, frame_bgr: np.ndarray) -> List[RecognitionResult]:
        if not self.encoder.encodings:
            return []

        cfg = self.config
        scale = cfg.frame_resize_factor
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # --- Detect (fast) ---
        boxes_small = self.backend.face_locations(
            rgb_small,
            number_of_times_to_upsample=cfg.upsample_times,
            model=cfg.detection_model,
        )
        inv = 1.0 / scale
        boxes_full: List[Tuple[int, int, int, int]] = [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for (t, r, b, l) in boxes_small
        ]

        now = time.time()
        results: List[RecognitionResult] = []
        new_tracks: List[_TrackedFace] = []
        needs_encoding: List[int] = []   # indices into boxes_full / boxes_small

        # --- Match each detection against existing tracks ---
        # Primary criterion: IoU >= iou_match_threshold (handles stationary faces).
        # Fallback:          centroid distance < half a face-width (handles head
        #                    motion, students turning, etc., where IoU drops sharply
        #                    even though it's obviously the same person).
        used_track_ids = set()
        for i, box_full in enumerate(boxes_full):
            best_iou, best_iou_idx = 0.0, -1
            for j, t in enumerate(self._tracks):
                if j in used_track_ids:
                    continue
                iou = _iou(box_full, t.box)
                if iou > best_iou:
                    best_iou, best_iou_idx = iou, j

            matched_idx = -1
            if best_iou_idx >= 0 and best_iou >= cfg.iou_match_threshold:
                matched_idx = best_iou_idx
            else:
                # IoU failed; try centroid distance.
                best_dist, best_dist_idx = float("inf"), -1
                for j, t in enumerate(self._tracks):
                    if j in used_track_ids:
                        continue
                    dist = _centroid_distance_normalized(box_full, t.box)
                    if dist < best_dist:
                        best_dist, best_dist_idx = dist, j
                if best_dist_idx >= 0 and best_dist < 0.5:
                    matched_idx = best_dist_idx

            reused = False
            if matched_idx >= 0:
                track = self._tracks[matched_idx]
                if (now - track.last_encoded_at) < cfg.reidentify_every_seconds:
                    used_track_ids.add(matched_idx)
                    new_tracks.append(_TrackedFace(
                        box=box_full,
                        name=track.name,
                        confidence=track.confidence,
                        last_encoded_at=track.last_encoded_at,
                    ))
                    results.append(RecognitionResult(
                        name=track.name, box=box_full, confidence=track.confidence,
                    ))
                    reused = True

            if not reused:
                needs_encoding.append(i)
                # placeholder; real values filled in after encoding
                new_tracks.append(_TrackedFace(
                    box=box_full, name="Unknown", confidence=0.0, last_encoded_at=now,
                ))
                results.append(RecognitionResult(
                    name="Unknown", box=box_full, confidence=0.0,
                ))

        # --- Encode (slow) only for unmatched / stale detections ---
        if needs_encoding:
            sub_boxes = [boxes_small[i] for i in needs_encoding]
            encs = self.backend.face_encodings(rgb_small, sub_boxes)

            for k, enc in enumerate(encs):
                idx = needs_encoding[k]
                name, confidence = self.backend.match(
                    enc,
                    self.encoder.encodings,
                    self.encoder.names,
                    cfg.recognition_tolerance,
                )

                new_tracks[idx].name = name
                new_tracks[idx].confidence = confidence
                new_tracks[idx].last_encoded_at = now
                results[idx] = RecognitionResult(
                    name=name, box=new_tracks[idx].box, confidence=confidence,
                )

        self._tracks = new_tracks
        return results


# ---------------------------------------------------------------------------
# Presence tracker: attendance + ENTER / LEAVE event log
# ---------------------------------------------------------------------------

class PresenceTracker:
    """Records attendance and emits ENTER / LEAVE events as people come and go."""

    def __init__(self, config: Config, session_label: Optional[str] = None) -> None:
        self.config = config
        self.session_started = datetime.now()
        date_str = self.session_started.strftime("%Y-%m-%d")
        suffix = f"_{session_label}" if session_label else ""

        self.attendance_csv = config.attendance_dir / f"attendance_{date_str}{suffix}.csv"
        self.events_csv = config.logs_dir / f"events_{date_str}{suffix}.csv"

        self.attendance_marked: set = set()
        self.last_seen: Dict[str, float] = {}
        self.currently_present: set = set()
        self.events: List[dict] = []   # in-memory log for the UI

        # Names that have passed the consecutive-frames check at least once.
        self.confirmed: set = set()
        # Per-name running count of consecutive frames in which they've been seen.
        self.consecutive_seen: Dict[str, int] = {}
        # Per-name punctuality status from the most recent attendance mark, so
        # the corresponding ENTER event can carry the same label.
        self._latest_status_for: Dict[str, str] = {}

        self._init_files()
        self._load_existing_attendance()

    # ---- file setup ------------------------------------------------------

    def _init_files(self) -> None:
        if not self.attendance_csv.exists():
            with open(self.attendance_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Name", "Date", "Time", "Status", "Confidence"])
        if not self.events_csv.exists():
            with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Name", "Event", "Date", "Time", "Status", "Confidence"])

    def _classify_punctuality(self, now: datetime) -> str:
        """Return 'ON_TIME', 'LATE', or '' (when no class_start_time is set)."""
        start = self.config.class_start_time
        if not start:
            return ""
        try:
            hh, mm = (int(p) for p in start.split(":"))
        except Exception:
            return ""
        from datetime import timedelta
        cutoff = now.replace(hour=hh, minute=mm, second=0, microsecond=0) \
            + timedelta(minutes=int(self.config.late_grace_minutes))
        return "ON_TIME" if now <= cutoff else "LATE"

    def _load_existing_attendance(self) -> None:
        try:
            with open(self.attendance_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.attendance_marked.add(row["Name"])
                    # Anyone already in the attendance file has been confirmed
                    # before, so re-entries shouldn't require re-confirmation.
                    self.confirmed.add(row["Name"])
        except Exception:
            pass

    # ---- per-frame update -----------------------------------------------

    def update(
        self,
        recognized: List[Tuple[str, float]],
        liveness_check: Optional[Callable[[str], bool]] = None,
    ) -> List[dict]:
        """Process one frame's recognition results.

        A name has to appear in `consecutive_recognition_threshold` consecutive
        frames before it counts toward attendance — guards against single-frame
        false positives. Re-entries after a LEAVE skip the wait because the
        person is already in `confirmed`.

        Args:
            recognized: list of (name, confidence) for every face in the frame.
            liveness_check: optional callable name -> bool. When provided AND
                `config.require_blink_to_mark` is True, confirmation also
                requires this to return True (e.g. blink detected).

        Returns:
            New event records emitted on this frame.
        """
        now_ts = time.time()
        new_events: List[dict] = []
        threshold = max(1, int(self.config.consecutive_recognition_threshold))
        liveness_required = (
            bool(self.config.require_blink_to_mark) and liveness_check is not None
        )

        seen_this_frame = set()
        for name, conf in recognized:
            if name == "Unknown":
                continue
            seen_this_frame.add(name)
            self.last_seen[name] = now_ts

            if name in self.confirmed:
                # Already confirmed this session — fast path. Just track presence.
                if name not in self.currently_present:
                    self.currently_present.add(name)
                    new_events.append(self._log_event(name, "ENTER", conf))
                continue

            # Unconfirmed: count consecutive sightings.
            count = self.consecutive_seen.get(name, 0) + 1
            self.consecutive_seen[name] = count
            if count >= threshold:
                if liveness_required and not liveness_check(name):
                    # Recognized enough times but hasn't blinked yet. Hold off.
                    continue
                self.confirmed.add(name)
                if name not in self.attendance_marked:
                    self._mark_attendance(name, conf)
                    self.attendance_marked.add(name)
                if name not in self.currently_present:
                    self.currently_present.add(name)
                    new_events.append(self._log_event(name, "ENTER", conf))

        # Reset the counter for unconfirmed names that were not seen this frame.
        for name in list(self.consecutive_seen.keys()):
            if name in seen_this_frame:
                continue
            if name not in self.confirmed:
                self.consecutive_seen[name] = 0

        # LEAVE events: anyone present but not seen for `leave_timeout_seconds`.
        for name in list(self.currently_present):
            if name in seen_this_frame:
                continue
            if now_ts - self.last_seen.get(name, 0.0) > self.config.leave_timeout_seconds:
                self.currently_present.discard(name)
                new_events.append(self._log_event(name, "LEAVE", 0.0))

        return new_events

    def force_leave_all(self) -> List[dict]:
        """Emit a LEAVE event for everyone still present (call at session end)."""
        new_events: List[dict] = []
        for name in list(self.currently_present):
            self.currently_present.discard(name)
            new_events.append(self._log_event(name, "LEAVE", 0.0))
        return new_events

    # ---- writers ---------------------------------------------------------

    def _mark_attendance(self, name: str, confidence: float) -> str:
        now = datetime.now()
        status = self._classify_punctuality(now)
        with open(self.attendance_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                status,
                f"{confidence:.3f}",
            ])
        # Cache the punctuality status so the matching ENTER event can include it
        # without re-computing or drifting across the second boundary.
        self._latest_status_for[name] = status
        return status

    def _log_event(self, name: str, event: str, confidence: float) -> dict:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        status = self._latest_status_for.get(name, "") if event == "ENTER" else ""

        with open(self.events_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, event, date_str, time_str, status, f"{confidence:.3f}"])

        record = {
            "name": name,
            "event": event,
            "date": date_str,
            "time": time_str,
            "status": status,
            "timestamp": now,
            "confidence": confidence,
        }
        self.events.append(record)
        return record


# ---------------------------------------------------------------------------
# Drawing helper (used by the Streamlit UI)
# ---------------------------------------------------------------------------

def annotate_frame(frame_bgr: np.ndarray, results: List[RecognitionResult]) -> np.ndarray:
    """Draw boxes and labels for every recognition result onto a copy of the frame."""
    out = frame_bgr.copy()
    for res in results:
        top, right, bottom, left = res.box
        if res.name == "Unknown":
            color = (0, 0, 220)
            label = "Unknown"
        else:
            color = (60, 200, 90)
            label = f"{res.name}  {res.confidence * 100:.0f}%"

        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (left, bottom - th - 12), (left + tw + 12, bottom), color, cv2.FILLED)
        cv2.putText(
            out, label, (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
