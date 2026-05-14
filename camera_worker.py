"""
camera_worker.py
----------------
Background worker that owns the webcam and the recognition pipeline.

Two independent threads:

  • Capture thread     — pulls frames from cv2.VideoCapture as fast as the
                          camera allows and stores the most recent one.
                          Essentially camera-native rate (~30 FPS).

  • Recognition thread — grabs the most recent captured frame whenever it
                          is free, runs detection + encoding + presence
                          tracking, and stores the result.

The UI's display loop reads the most recent capture (always fresh) and
the most recent recognition results (which may lag a few hundred ms
behind). That way a slow recognition pass never freezes the live feed.

Lifecycle:
    worker = CameraWorker(config, recognizer, presence_tracker, blink_detector)
    worker.start()
    frame, results, capture_fps, recognition_fps, new_events, err = worker.snapshot()
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

        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._recognize_thread: Optional[threading.Thread] = None

        # Capture state (written by capture thread, read by everyone).
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0          # monotonic; lets the UI skip dup re-renders
        self._capture_fps: float = 0.0

        # Recognition state (written by recognition thread, read by UI).
        self._results_lock = threading.Lock()
        self._latest_results: list = []
        self._recognition_fps: float = 0.0
        self._pending_events: list = []
        self._error: Optional[str] = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._recognize_thread = threading.Thread(target=self._recognize_loop, daemon=True)
        self._capture_thread.start()
        self._recognize_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        for t in (self._capture_thread, self._recognize_thread):
            if t is not None:
                t.join(timeout=timeout)
        self._capture_thread = None
        self._recognize_thread = None

    @property
    def is_running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()

    # ---- main thread API -------------------------------------------------

    def snapshot(
        self,
    ) -> Tuple[Optional[np.ndarray], int, list, float, float, list, Optional[str]]:
        """Atomically read latest frame + latest results + accumulated events.

        Returns:
            (frame, frame_id, results, capture_fps, recognition_fps, events, error)
        """
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            frame_id = self._frame_id
            capture_fps = self._capture_fps
        with self._results_lock:
            results = list(self._latest_results)
            recognition_fps = self._recognition_fps
            events = self._pending_events
            self._pending_events = []     # caller takes ownership
            error = self._error
        return frame, frame_id, results, capture_fps, recognition_fps, events, error

    # ---- capture thread --------------------------------------------------

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            with self._results_lock:
                self._error = f"Could not open camera index {self.config.camera_index}."
            return

        # Keep the camera buffer as small as possible so frames don't stale.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fps_t0 = time.time()
        fps_frames = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                fps_frames += 1
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_id += 1
                    if time.time() - fps_t0 >= 1.0:
                        self._capture_fps = fps_frames / (time.time() - fps_t0)
                        fps_t0 = time.time()
                        fps_frames = 0
        finally:
            cap.release()

    # ---- recognition thread ----------------------------------------------

    def _recognize_loop(self) -> None:
        fps_t0 = time.time()
        fps_frames = 0
        last_processed_id = -1
        try:
            while not self._stop_event.is_set():
                with self._frame_lock:
                    if self._latest_frame is None or self._frame_id == last_processed_id:
                        frame, frame_id = None, last_processed_id
                    else:
                        frame, frame_id = self._latest_frame.copy(), self._frame_id

                if frame is None:
                    time.sleep(0.005)
                    continue

                last_processed_id = frame_id

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

                fps_frames += 1
                if time.time() - fps_t0 >= 1.0:
                    recognition_fps = fps_frames / (time.time() - fps_t0)
                    fps_t0 = time.time()
                    fps_frames = 0
                else:
                    recognition_fps = None

                with self._results_lock:
                    self._latest_results = results
                    if recognition_fps is not None:
                        self._recognition_fps = recognition_fps
                    if new_events:
                        self._pending_events.extend(new_events)
        except Exception as exc:    # noqa: BLE001
            with self._results_lock:
                self._error = f"{type(exc).__name__}: {exc}"
