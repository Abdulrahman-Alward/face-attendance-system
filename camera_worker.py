"""
camera_worker.py
----------------
Background worker that owns the webcam and runs detection / recognition /
presence-tracking off the Streamlit main thread.

Why: in a single-threaded loop, slow recognition (CNN at full resolution,
ArcFace on a wide classroom shot) blocks the display, so the UI starts
stuttering or dropping below the target FPS. Splitting capture+recognize
into a dedicated thread lets the Streamlit loop simply pull the latest
annotated state and render at a smooth `target_fps`.

Lifecycle:
    worker = CameraWorker(config, recognizer, presence_tracker, blink_detector)
    worker.start()
    ...
    frame, results, present_count, fps, new_events = worker.snapshot()
    ...
    worker.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraWorker:
    def __init__(
        self,
        config,
        recognizer,
        presence_tracker,
        blink_detector=None,
    ) -> None:
        self.config = config
        self.recognizer = recognizer
        self.presence_tracker = presence_tracker
        self.blink_detector = blink_detector

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Shared state (latest captured + processed frame).
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_results: list = []
        self._latest_fps: float = 0.0
        self._pending_events: list = []
        self._error: Optional[str] = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- main thread API -------------------------------------------------

    def snapshot(self) -> Tuple[Optional[np.ndarray], list, float, list, Optional[str]]:
        """Atomically pull the latest frame + results + accumulated events."""
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            results = list(self._latest_results)
            fps = self._latest_fps
            events = self._pending_events
            self._pending_events = []   # caller takes ownership
            error = self._error
        return frame, results, fps, events, error

    # ---- worker loop -----------------------------------------------------

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            with self._lock:
                self._error = f"Could not open camera index {self.config.camera_index}."
            return

        fps_t0 = time.time()
        fps_frames = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                fps_frames += 1

                results = self.recognizer.process(frame)

                if (self.blink_detector is not None
                        and self.config.require_blink_to_mark):
                    self.blink_detector.update(frame, results)
                    liveness_check = self.blink_detector.has_blinked
                else:
                    liveness_check = None

                new_events = self.presence_tracker.update(
                    [(r.name, r.confidence) for r in results],
                    liveness_check=liveness_check,
                )

                if time.time() - fps_t0 >= 1.0:
                    fps = fps_frames / (time.time() - fps_t0)
                    fps_t0 = time.time()
                    fps_frames = 0
                else:
                    fps = None

                with self._lock:
                    self._latest_frame = frame
                    self._latest_results = results
                    if fps is not None:
                        self._latest_fps = fps
                    if new_events:
                        self._pending_events.extend(new_events)
        except Exception as exc:    # noqa: BLE001 — surface any worker error to the UI
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            cap.release()
