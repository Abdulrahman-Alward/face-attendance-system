"""Smoke test for the face-attendance project.

Verifies that:
  1. dlib and the bundled face_lib import cleanly.
  2. All required model files are present.
  3. The detection pipeline runs end-to-end on a synthetic image.
  4. Config, FaceEncoder and PresenceTracker construct correctly and
     write/read their CSV files.

Run with:
    python smoke_test.py

A non-zero exit code means at least one check failed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


FAILURES: list = []


def check(label: str, fn) -> None:
    print(f"[ ] {label} ... ", end="", flush=True)
    try:
        fn()
    except Exception as exc:
        FAILURES.append((label, exc))
        print(f"FAIL\n    {type(exc).__name__}: {exc}")
    else:
        print("ok")


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def _imports():
    import dlib                    # noqa: F401
    import cv2                     # noqa: F401
    import face_lib                # noqa: F401
    from face_attendance import (  # noqa: F401
        Config, FaceEncoder, TrackingRecognizer, PresenceTracker, annotate_frame,
    )
    from anti_spoof import BlinkDetector  # noqa: F401
    from camera_worker import CameraWorker  # noqa: F401


check("Imports load (dlib, cv2, face_lib, face_attendance, anti_spoof, camera_worker)", _imports)


# ---------------------------------------------------------------------------
# 2. Model files
# ---------------------------------------------------------------------------

def _models_present():
    base = Path(__file__).resolve().parent / "models" / "face_recognition_models" / "models"
    needed = [
        "shape_predictor_68_face_landmarks.dat",
        "shape_predictor_5_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat",
        "mmod_human_face_detector.dat",
    ]
    missing = [n for n in needed if not (base / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing models in {base}: {missing}")


check("All four dlib model files are bundled", _models_present)


# ---------------------------------------------------------------------------
# 3. Detection runs on a synthetic frame
# ---------------------------------------------------------------------------

def _detection_runs():
    import face_lib
    rng = np.random.default_rng(0)
    img = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    locs = face_lib.face_locations(img, model="hog")
    # We don't expect any detections in noise — we just want the call to
    # succeed and return a list.
    assert isinstance(locs, list), f"expected list, got {type(locs)}"


check("face_lib.face_locations() runs end-to-end", _detection_runs)


# ---------------------------------------------------------------------------
# 4. Encoder + recognizer construct without error
# ---------------------------------------------------------------------------

def _encoder_constructs():
    from face_attendance import Config, FaceEncoder, TrackingRecognizer
    cfg = Config()
    enc = FaceEncoder(cfg)
    enc.load()
    rec = TrackingRecognizer(enc, cfg)
    assert hasattr(rec, "process")
    assert hasattr(rec, "reset")


check("Config / FaceEncoder / TrackingRecognizer construct", _encoder_constructs)


# ---------------------------------------------------------------------------
# 5. PresenceTracker emits ENTER / LEAVE events
# ---------------------------------------------------------------------------

def _tracker_logs_events():
    import time as _time
    from face_attendance import Config, PresenceTracker

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = Config(
            project_root=root,
            dataset_dir=root / "dataset",
            encodings_file=root / "encodings.pkl",
            attendance_dir=root / "attendance",
            logs_dir=root / "logs",
            leave_timeout_minutes=0.001,            # ~0.06 s, so the test runs fast
            consecutive_recognition_threshold=3,   # confirm in 3 frames
        )
        tracker = PresenceTracker(cfg, session_label="smoke")

        # First two frames are below the confirmation threshold — no events yet.
        for _ in range(cfg.consecutive_recognition_threshold - 1):
            assert tracker.update([("Alice", 0.9), ("Bob", 0.8)]) == []

        # Frame N reaches the threshold — both should now ENTER.
        events_in = tracker.update([("Alice", 0.9), ("Bob", 0.8)])
        assert {e["event"] for e in events_in} == {"ENTER"}, events_in
        assert {e["name"] for e in events_in} == {"Alice", "Bob"}
        assert tracker.confirmed == {"Alice", "Bob"}

        _time.sleep(0.2)
        events_out = tracker.update([])    # nobody seen this frame
        assert {e["event"] for e in events_out} == {"LEAVE"}, events_out
        assert tracker.currently_present == set()

        # Re-entry skips the multi-frame check (already confirmed once).
        events_back = tracker.update([("Alice", 0.9)])
        assert events_back and events_back[0]["event"] == "ENTER", events_back

        # CSV files should now exist with header + content
        assert tracker.attendance_csv.exists()
        assert tracker.events_csv.exists()
        with open(tracker.events_csv) as f:
            lines = f.read().splitlines()
        # header + 2 ENTER (initial) + 2 LEAVE + 1 ENTER (re-entry) = 6
        assert len(lines) == 6, f"expected 6 lines in events log, got {len(lines)}: {lines}"


check("PresenceTracker logs ENTER and LEAVE events to CSV", _tracker_logs_events)


# ---------------------------------------------------------------------------
# 6. Optional backends (torch / arcface) either work or stay gracefully off
# ---------------------------------------------------------------------------

def _optional_backends_clean():
    from face_attendance import (
        TORCH_BACKEND_AVAILABLE, INSIGHTFACE_BACKEND_AVAILABLE, get_backend,
    )
    if TORCH_BACKEND_AVAILABLE:
        b = get_backend("torch")
        assert callable(b.face_locations) and callable(b.match)
    if INSIGHTFACE_BACKEND_AVAILABLE:
        b = get_backend("arcface")
        assert callable(b.face_locations) and callable(b.match)
    # When unavailable, get_backend should raise a clear RuntimeError.
    if not INSIGHTFACE_BACKEND_AVAILABLE:
        try:
            get_backend("arcface")
        except RuntimeError:
            return
        raise AssertionError("Expected RuntimeError when arcface backend is unavailable.")


check("Optional torch / arcface backends are usable or cleanly unavailable", _optional_backends_clean)


# ---------------------------------------------------------------------------
# 7. Config persistence — save / load roundtrip
# ---------------------------------------------------------------------------

def _config_persists():
    from face_attendance import Config
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = Config(project_root=root)
        cfg.target_fps = 27.0
        cfg.recognition_tolerance = 0.42
        cfg.class_start_time = "08:30"
        path = root / "settings.json"
        cfg.save_to(path)

        cfg2 = Config(project_root=root)
        cfg2.load_from(path)
        assert cfg2.target_fps == 27.0
        assert abs(cfg2.recognition_tolerance - 0.42) < 1e-9
        assert cfg2.class_start_time == "08:30"


check("Config persists to JSON and loads back", _config_persists)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)} check{'s' if len(FAILURES) > 1 else ''})")
    for label, exc in FAILURES:
        print(f"  - {label}: {exc}")
    sys.exit(1)
else:
    print("All checks passed. You're ready to run: streamlit run app.py")
