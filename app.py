"""Streamlit GUI for the Face-Based Attendance System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import pandas as pd
import streamlit as st

from face_attendance import (
    Config,
    FaceEncoder,
    PresenceTracker,
    TrackingRecognizer,
    TORCH_BACKEND_AVAILABLE,
    INSIGHTFACE_BACKEND_AVAILABLE,
    annotate_frame,
    get_backend,
)
from anti_spoof import BlinkDetector
from camera_worker import CameraWorker


# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --accent: #6c8cff;
    --accent-2: #8b5cf6;
    --bg-card: rgba(255,255,255,0.04);
    --bg-card-hover: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.08);
}

.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(108,140,255,0.18), transparent 60%),
        radial-gradient(1000px 500px at 110% 10%, rgba(139,92,246,0.18), transparent 55%),
        linear-gradient(135deg, #0b0f1a 0%, #0f1422 60%, #111827 100%);
}

/* Header band */
.header-band {
    background: linear-gradient(90deg, rgba(108,140,255,0.18), rgba(139,92,246,0.18));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}
.header-band h1 {
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
}
.header-band p {
    margin: 0.25rem 0 0 0;
    color: #c7d0e0;
    font-size: 0.95rem;
}

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    transition: background 120ms ease, transform 120ms ease;
}
.metric-card:hover { background: var(--bg-card-hover); }
.metric-card .label {
    color: #a1abc1;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-card .value {
    color: #ffffff;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 0.25rem;
    letter-spacing: -0.02em;
}
.metric-card .sub {
    color: #8b94a8;
    font-size: 0.8rem;
    margin-top: 0.15rem;
}

/* Status pills */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    border: 1px solid var(--border);
}
.status-pill.live { background: rgba(34,197,94,0.15); color: #4ade80; }
.status-pill.idle { background: rgba(148,163,184,0.15); color: #cbd5e1; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.dot.pulse { animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.35 } }

/* Event chips */
.event-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    margin-bottom: 0.4rem;
}
.event-row .who { color: #e6ecf8; font-weight: 600; }
.event-row .when { color: #8b94a8; font-size: 0.82rem; }
.event-row .tag {
    font-size: 0.72rem; font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
}
.tag.enter { background: rgba(34,197,94,0.2); color: #4ade80; }
.tag.leave { background: rgba(248,113,113,0.2); color: #f87171; }
.tag.late  { background: rgba(251,191,36,0.18); color: #fbbf24; }

.status-pill.count {
    background: rgba(108,140,255,0.18);
    color: #a4b5ff;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.45rem 1.0rem;
    color: #c7d0e0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(108,140,255,0.25), rgba(139,92,246,0.25)) !important;
    color: #ffffff !important;
    border-color: rgba(108,140,255,0.4) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: #e6ecf8;
    font-weight: 600;
    transition: all 120ms ease;
}
.stButton > button:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent);
    color: #ffffff;
}
.stButton > button:disabled { opacity: 0.45; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a 0%, #0d1322 100%);
    border-right: 1px solid var(--border);
}

/* Empty placeholder */
.empty-hint {
    border: 1.5px dashed var(--border);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: #8b94a8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "config" not in st.session_state:
        cfg = Config(project_root=Path(__file__).resolve().parent)

        settings_path = cfg.project_root / "data" / "settings.json"
        if settings_path.exists():
            cfg.load_from(settings_path)
        else:
            # First launch: prefer torch backend when CUDA is available.
            if TORCH_BACKEND_AVAILABLE:
                try:
                    tb = get_backend("torch")
                    if getattr(tb, "HAS_CUDA", False):
                        cfg.backend = "torch"
                except Exception:
                    pass
            cfg.apply_backend_defaults()

        st.session_state.config = cfg
        st.session_state.settings_path = settings_path

    if "encoder" not in st.session_state:
        encoder = FaceEncoder(st.session_state.config)
        encoder.load()
        st.session_state.encoder = encoder

    if "recognizer" not in st.session_state:
        st.session_state.recognizer = TrackingRecognizer(
            st.session_state.encoder, st.session_state.config
        )

    st.session_state.setdefault("running", False)
    st.session_state.setdefault("tracker", None)
    st.session_state.setdefault("session_label", "")
    st.session_state.setdefault("recent_events", [])
    if "blink_detector" not in st.session_state:
        st.session_state.blink_detector = BlinkDetector()
    st.session_state.setdefault("camera_worker", None)

_init_state()

CFG: Config = st.session_state.config
ENCODER: FaceEncoder = st.session_state.encoder
RECOGNIZER: TrackingRecognizer = st.session_state.recognizer


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-band">
        <h1>🎓 Face Attendance</h1>
        <p>Live face recognition with automatic attendance and ENTER / LEAVE logging.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    # --- Backend status pill (top of sidebar) ---
    active_backend = get_backend(CFG.backend)
    has_cuda = bool(getattr(active_backend, "HAS_CUDA", False)) or bool(
        getattr(active_backend, "DLIB_HAS_CUDA", False)
    )
    pill_cls = "live" if has_cuda else "idle"
    pill_dot = "dot pulse" if has_cuda else "dot"
    st.markdown(
        f'<div class="status-pill {pill_cls}" style="margin-bottom:0.7rem;">'
        f'<span class="{pill_dot}"></span>{active_backend.cuda_status()}</div>',
        unsafe_allow_html=True,
    )

    # --- Backend selector ---
    backend_options = ["dlib"]
    if TORCH_BACKEND_AVAILABLE:
        backend_options.append("torch")
    if INSIGHTFACE_BACKEND_AVAILABLE:
        backend_options.append("arcface")
    backend_labels = {
        "dlib":    "dlib · HOG + ResNet",
        "torch":   "torch · MTCNN + FaceNet (CUDA)",
        "arcface": "arcface · SCRFD + ArcFace (CUDA)",
    }
    new_backend = st.selectbox(
        "Backend",
        options=backend_options,
        format_func=lambda b: backend_labels[b],
        index=backend_options.index(CFG.backend) if CFG.backend in backend_options else 0,
    )
    missing_msgs = []
    if not TORCH_BACKEND_AVAILABLE:
        missing_msgs.append("`pip install facenet-pytorch torch torchvision` for the torch backend.")
    if not INSIGHTFACE_BACKEND_AVAILABLE:
        missing_msgs.append("`pip install insightface onnxruntime-gpu` for the arcface backend.")
    if missing_msgs:
        st.caption(" · ".join(missing_msgs))

    if new_backend != CFG.backend:
        CFG.backend = new_backend
        CFG.apply_backend_defaults()
        new_encoder = FaceEncoder(CFG)
        new_encoder.load()
        st.session_state.encoder = new_encoder
        st.session_state.recognizer = TrackingRecognizer(new_encoder, CFG)
        st.rerun()

    # --- Camera & basic recognition ---
    st.markdown("#### 📷 Camera")
    CFG.camera_index = st.number_input(
        "Camera index", min_value=0, max_value=8, value=int(CFG.camera_index), step=1,
        label_visibility="collapsed",
    )

    st.markdown("#### 🎯 Recognition")
    CFG.recognition_tolerance = st.slider(
        "Tolerance",
        min_value=0.30, max_value=0.80,
        value=float(CFG.recognition_tolerance), step=0.01,
        help="Lower = stricter match. dlib uses Euclidean distance, "
             "torch uses cosine distance — both feel similar at 0.5.",
    )
    CFG.leave_timeout_minutes = st.slider(
        "Leave timeout (min)",
        min_value=0.25, max_value=30.0,
        value=float(CFG.leave_timeout_minutes), step=0.25,
    )

    # --- Punctuality ---
    st.markdown("#### 🕒 Class start")
    use_start = st.checkbox(
        "Track on-time / late",
        value=CFG.class_start_time is not None,
        help="When on, attendance entries get a Status column based on the "
             "class start time and grace period below.",
    )
    if use_start:
        from datetime import time as _dtime
        try:
            hh, mm = (int(p) for p in (CFG.class_start_time or "09:00").split(":"))
            current_start = _dtime(hh, mm)
        except Exception:
            current_start = _dtime(9, 0)
        picked = st.time_input("Start time", value=current_start)
        CFG.class_start_time = f"{picked.hour:02d}:{picked.minute:02d}"
        CFG.late_grace_minutes = st.slider(
            "Grace period (min)", min_value=0, max_value=30,
            value=int(CFG.late_grace_minutes), step=1,
            help="A student arriving within this many minutes after the start "
                 "time is still ON_TIME; later is LATE.",
        )
    else:
        CFG.class_start_time = None

    # --- Distance / sensitivity (backend-specific) ---
    st.markdown("#### 📏 Distance")
    if CFG.backend == "torch":
        CFG.min_face_size = st.slider(
            "Min face size (px)",
            min_value=8, max_value=80,
            value=int(CFG.min_face_size), step=2,
            help="Smallest face MTCNN reports. Lower = picks up faces "
                 "from farther away. Try 12 for a full classroom shot.",
        )
        active_backend.configure(min_face_size=CFG.min_face_size)
    else:
        CFG.upsample_times = st.slider(
            "Detection upsample",
            min_value=0, max_value=3,
            value=int(CFG.upsample_times),
            help="HOG upsamples the frame this many times before detecting. "
                 "Higher = picks up smaller / farther faces, slower.",
        )

    # --- Performance / advanced ---
    with st.expander("⚙️ Advanced", expanded=False):
        CFG.target_fps = st.slider(
            "Target FPS", min_value=10, max_value=60,
            value=int(CFG.target_fps), step=1,
        )
        CFG.frame_resize_factor = st.slider(
            "Frame downscale", min_value=0.15, max_value=1.0,
            value=float(CFG.frame_resize_factor), step=0.05,
            help="Downscaling speeds up detection at the cost of distance. "
                 "torch on CUDA can comfortably stay at 1.0; dlib HOG often "
                 "needs 0.5 to keep up.",
        )

        st.markdown("**Confirmation**")
        CFG.consecutive_recognition_threshold = st.slider(
            "Consecutive frames to confirm", min_value=1, max_value=20,
            value=int(CFG.consecutive_recognition_threshold), step=1,
            help="A face has to be recognized this many frames in a row before "
                 "it counts toward attendance. Higher = fewer false positives.",
        )

        st.markdown("**Enrollment**")
        CFG.augment_during_enrollment = st.checkbox(
            "Augment during enrollment",
            value=bool(CFG.augment_during_enrollment),
            help="Each training image is also encoded with horizontal-flip "
                 "and brightness ±20% variants. 4× embeddings per image, "
                 "much better robustness to lighting / angle.",
        )

        st.markdown("**Anti-spoofing**")
        CFG.require_blink_to_mark = st.checkbox(
            "Require blink to confirm",
            value=bool(CFG.require_blink_to_mark),
            help="Recognized faces must blink (eye-aspect-ratio drop) before "
                 "attendance is marked. Defends against held-up photos.",
        )

        st.markdown("**Tracking** _(skips re-encoding stable faces)_")
        CFG.iou_match_threshold = st.slider(
            "IoU match threshold", min_value=0.10, max_value=0.80,
            value=float(CFG.iou_match_threshold), step=0.05,
        )
        CFG.reidentify_every_seconds = st.slider(
            "Re-identify every (s)", min_value=1.0, max_value=120.0,
            value=float(CFG.reidentify_every_seconds), step=1.0,
            help="In between encodings, the recognizer just detects + tracks "
                 "faces by IoU and centroid distance. Default is 20 s.",
        )

        # Detector model is auto-selected per backend, but expose it here for
        # power users who built dlib with CUDA and want to switch to dlib-CNN.
        if CFG.backend == "dlib":
            CFG.detection_model = st.selectbox(
                "dlib detector", options=["hog", "cnn"],
                index=0 if CFG.detection_model == "hog" else 1,
                help="HOG = fast on CPU. CNN = accurate; needs dlib built "
                     "with CUDA to be usable in real time.",
            )

    st.markdown("---")

    # --- Encodings + session ---
    summary = ENCODER.inspect_dataset()
    st.markdown(
        f'<div class="metric-card" style="padding:0.7rem 0.9rem;">'
        f'<div class="label">Database</div>'
        f'<div class="sub" style="margin-top:0.3rem;">'
        f'{len(summary)} person(s) · {sum(summary.values())} image(s) · '
        f'{len(ENCODER.encodings)} encoding(s)</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Rebuild encodings", use_container_width=True):
        progress = st.progress(0.0, text="Building encodings...")
        def _cb(i, total, name):
            progress.progress(min(1.0, i / max(total, 1)), text=f"Encoding: {name}")
        stats = ENCODER.build_database(force_rebuild=True, progress_callback=_cb)
        progress.empty()
        st.success(f"Encoded {stats['encoded']} face(s) for {stats['people']} person(s).")

    st.session_state.session_label = st.text_input(
        "Session label", value=st.session_state.session_label,
        placeholder="e.g. lecture-1",
        help="Optional tag appended to today's CSV filenames.",
    )

# Persist any changes made this rerun (cheap — JSON, ~1 KB).
try:
    CFG.save_to(st.session_state.settings_path)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

live_tab, attendance_tab, events_tab, dataset_tab, reports_tab = st.tabs(
    ["📹 Live", "📋 Attendance", "🚪 Event Logs", "👥 Dataset", "📈 Reports"]
)


# ===========================================================================
# LIVE TAB
# ===========================================================================

def _format_event_html(events: List[dict], limit: int = 12) -> str:
    if not events:
        return '<div class="empty-hint">No events yet.<br/>ENTER and LEAVE will appear here as students are recognized.</div>'

    html_parts = []
    for evt in reversed(events[-limit:]):
        tag_cls = "enter" if evt["event"] == "ENTER" else "leave"
        html_parts.append(
            f'<div class="event-row">'
            f'  <div><span class="who">{evt["name"]}</span><br/>'
            f'  <span class="when">{evt["date"]} · {evt["time"]}</span></div>'
            f'  <span class="tag {tag_cls}">{evt["event"]}</span>'
            f'</div>'
        )
    return "".join(html_parts)


def _metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}</div>'
        f'  <div class="sub">{sub}</div>'
        f'</div>'
    )


with live_tab:
    if not ENCODER.encodings:
        st.warning(
            "No face encodings loaded. Add people in the **Dataset** tab "
            "and click **Rebuild encodings** in the sidebar."
        )

    btn_col1, btn_col2, status_col = st.columns([1, 1, 5])
    with btn_col1:
        start_clicked = st.button(
            "▶ Start", use_container_width=True,
            disabled=st.session_state.running or not ENCODER.encodings,
        )
    with btn_col2:
        stop_clicked = st.button(
            "⏹ Stop", use_container_width=True,
            disabled=not st.session_state.running,
        )

    status_placeholder = status_col.empty()


    def _render_status_row(present: int, roster: int, running: bool) -> None:
        if running:
            pill = '<div class="status-pill live"><span class="dot pulse"></span>LIVE</div>'
            tail = f"{CFG.backend} backend · target {int(CFG.target_fps)} FPS"
        else:
            pill = '<div class="status-pill idle"><span class="dot"></span>Idle</div>'
            tail = f"{CFG.backend} backend · ready"
        roster_str = str(roster) if roster else "—"
        count_pill = (
            f'<div class="status-pill count" style="margin-left:0.5rem;">'
            f'{present}/{roster_str} present</div>'
        )
        status_placeholder.markdown(
            '<div style="display:flex;align-items:center;height:100%;">'
            f'{pill}{count_pill}'
            f'<span style="margin-left:0.7rem;color:#8b94a8;font-size:0.85rem;">{tail}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    initial_present = (
        len(st.session_state.tracker.currently_present)
        if st.session_state.tracker is not None else 0
    )
    _render_status_row(initial_present, len(ENCODER.people), st.session_state.running)

    if start_clicked:
        st.session_state.tracker = PresenceTracker(
            CFG, session_label=st.session_state.session_label or None
        )
        st.session_state.recent_events = []
        RECOGNIZER.reset()
        st.session_state.blink_detector.reset()

        worker = CameraWorker(
            config=CFG,
            recognizer=RECOGNIZER,
            presence_tracker=st.session_state.tracker,
            blink_detector=st.session_state.blink_detector,
        )
        worker.start()
        st.session_state.camera_worker = worker
        st.session_state.running = True
        st.rerun()

    if stop_clicked:
        st.session_state.running = False
        worker = st.session_state.camera_worker
        if worker is not None:
            worker.stop()
            st.session_state.camera_worker = None
        if st.session_state.tracker is not None:
            leave_events = st.session_state.tracker.force_leave_all()
            st.session_state.recent_events.extend(leave_events)
        st.rerun()

    video_col, side_col = st.columns([3, 1])
    with video_col:
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
    with side_col:
        st.markdown("#### 🟢 Currently present")
        present_placeholder = st.empty()
        st.markdown("#### 🚪 Recent events")
        events_placeholder = st.empty()

    # Initial render of side panels (idle state)
    if not st.session_state.running:
        if st.session_state.tracker is None:
            video_placeholder.markdown(
                '<div class="empty-hint">Press <strong>Start</strong> to begin a real-time session.<br/>'
                'The webcam feed will appear here.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Show last frame info if a session just ended
            video_placeholder.markdown(
                '<div class="empty-hint">Session stopped. Press <strong>Start</strong> to begin again.</div>',
                unsafe_allow_html=True,
            )

        tracker = st.session_state.tracker
        present_names = sorted(tracker.currently_present) if tracker else []
        present_placeholder.markdown(
            ("".join(f'<div class="event-row"><span class="who">{n}</span></div>'
                    for n in present_names)
             if present_names
             else '<div class="empty-hint">Nobody yet.</div>'),
            unsafe_allow_html=True,
        )
        events_placeholder.markdown(
            _format_event_html(st.session_state.recent_events), unsafe_allow_html=True
        )

    # ---- Live render loop (capture + recognition run in CameraWorker thread) ----
    if st.session_state.running:
        tracker: PresenceTracker = st.session_state.tracker
        worker: CameraWorker = st.session_state.camera_worker
        blink_detector = st.session_state.blink_detector

        if worker is None or not worker.is_running:
            st.session_state.running = False
            st.error("Camera worker is not running. Press Start again.")
            st.stop()

        frame_budget = 1.0 / max(float(CFG.target_fps), 1.0)
        next_frame_deadline = time.time()
        last_frame_id = -1
        last_panel_update = 0.0
        last_present_signature: tuple = ()
        last_events_signature: tuple = ()

        try:
            # Streamlit interrupts this loop with a RerunException when the
            # user clicks Stop (the rerun fires at the next placeholder call).
            while st.session_state.running:
                (
                    frame, frame_id, last_results,
                    capture_fps, recognition_fps, new_events, error,
                ) = worker.snapshot()

                if error:
                    st.session_state.running = False
                    worker.stop()
                    st.session_state.camera_worker = None
                    st.error(error)
                    st.stop()

                if new_events:
                    st.session_state.recent_events.extend(new_events)

                if frame is None:
                    video_placeholder.markdown(
                        '<div class="empty-hint">Warming up the camera…</div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.03)
                    continue

                # Only re-render the video when a NEW frame is available.
                # Skips Streamlit overhead on duplicate snapshots.
                if frame_id != last_frame_id:
                    last_frame_id = frame_id

                    display_results = last_results
                    if CFG.require_blink_to_mark:
                        display_results = []
                        for r in last_results:
                            if (r.name != "Unknown"
                                    and r.name not in tracker.confirmed
                                    and not blink_detector.has_blinked(r.name)):
                                r = type(r)(
                                    name=f"{r.name} · blink to confirm",
                                    box=r.box,
                                    confidence=r.confidence,
                                )
                            display_results.append(r)

                    annotated = annotate_frame(frame, display_results)
                    video_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                # Throttle the side / metric panels to ~5 Hz — they only need
                # to refresh when something visible to the user changes, and
                # markdown re-renders are surprisingly expensive in Streamlit.
                now_t = time.time()
                if now_t - last_panel_update >= 0.2 or new_events:
                    last_panel_update = now_t

                    metrics_placeholder.markdown(
                        f'<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; margin-top:0.6rem;">'
                        f'  {_metric_card("Present now", str(len(tracker.currently_present)), "in classroom")}'
                        f'  {_metric_card("Marked today", str(len(tracker.attendance_marked)), "unique attendees")}'
                        f'  {_metric_card("Live / Recog FPS", f"{capture_fps:.0f} / {recognition_fps:.1f}", "camera · worker")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    present_names = tuple(sorted(tracker.currently_present))
                    if present_names != last_present_signature:
                        last_present_signature = present_names
                        present_placeholder.markdown(
                            ("".join(
                                f'<div class="event-row"><span class="who">{n}</span>'
                                f'<span class="tag enter">IN</span></div>'
                                for n in present_names
                            ) if present_names
                             else '<div class="empty-hint">Nobody yet.</div>'),
                            unsafe_allow_html=True,
                        )

                    events_signature = tuple(
                        (e["name"], e["event"], e["time"])
                        for e in st.session_state.recent_events[-12:]
                    )
                    if events_signature != last_events_signature:
                        last_events_signature = events_signature
                        events_placeholder.markdown(
                            _format_event_html(st.session_state.recent_events),
                            unsafe_allow_html=True,
                        )

                    _render_status_row(
                        len(tracker.currently_present), len(ENCODER.people), True,
                    )

                # Pace the display loop to target_fps. Recognition runs
                # independently in the worker thread, so a slow recognition
                # pass no longer blocks the UI.
                next_frame_deadline += frame_budget
                sleep_for = next_frame_deadline - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_frame_deadline = time.time()
        finally:
            pass   # worker is owned by session_state; stop is handled by the Stop button


# ===========================================================================
# ATTENDANCE TAB
# ===========================================================================

with attendance_tab:
    st.markdown("### 📋 Attendance reports")
    files = sorted(CFG.attendance_dir.glob("attendance_*.csv"), reverse=True)
    if not files:
        st.info("No attendance reports yet. Run a live session to generate one.")
    else:
        choice = st.selectbox(
            "Pick a report",
            options=files,
            format_func=lambda p: p.name,
        )
        df = pd.read_csv(choice)
        c1, c2, c3 = st.columns(3)
        c1.markdown(_metric_card("Records", str(len(df)), choice.name), unsafe_allow_html=True)
        c2.markdown(
            _metric_card("Unique people", str(df["Name"].nunique()) if not df.empty else "0", ""),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _metric_card(
                "Avg confidence",
                f"{df['Confidence'].astype(float).mean():.2f}" if not df.empty else "—",
                "0–1 score",
            ),
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download CSV",
            data=choice.read_bytes(),
            file_name=choice.name,
            mime="text/csv",
        )


# ===========================================================================
# EVENT LOGS TAB
# ===========================================================================

with events_tab:
    st.markdown("### 🚪 ENTER / LEAVE event logs")
    files = sorted(CFG.logs_dir.glob("events_*.csv"), reverse=True)
    if not files:
        st.info("No event logs yet. Events are written when students enter or leave during a live session.")
    else:
        choice = st.selectbox(
            "Pick a log",
            options=files,
            format_func=lambda p: p.name,
            key="events_select",
        )
        df = pd.read_csv(choice)
        if df.empty:
            st.info("This log is empty.")
        else:
            enters = (df["Event"] == "ENTER").sum()
            leaves = (df["Event"] == "LEAVE").sum()
            c1, c2, c3 = st.columns(3)
            c1.markdown(_metric_card("Total events", str(len(df)), ""), unsafe_allow_html=True)
            c2.markdown(_metric_card("Entries", str(enters), "ENTER events"), unsafe_allow_html=True)
            c3.markdown(_metric_card("Departures", str(leaves), "LEAVE events"), unsafe_allow_html=True)
            st.markdown("&nbsp;", unsafe_allow_html=True)

            people = ["All"] + sorted(df["Name"].unique().tolist())
            who = st.selectbox("Filter by person", options=people)
            view = df if who == "All" else df[df["Name"] == who]
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                data=choice.read_bytes(),
                file_name=choice.name,
                mime="text/csv",
            )


# ===========================================================================
# DATASET TAB
# ===========================================================================

with dataset_tab:
    st.markdown("### 👥 Registered people")
    summary = ENCODER.inspect_dataset()
    if not summary:
        st.info("Dataset is empty. Add a person below.")
    else:
        st.session_state.setdefault("pending_delete", None)
        for name, count in summary.items():
            row_cols = st.columns([4, 1, 1, 1])
            row_cols[0].markdown(
                f'<div class="event-row" style="margin-bottom:0.3rem;">'
                f'<span class="who">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            status = "OK" if count >= 3 else "Low"
            row_cols[1].markdown(
                f'<div style="padding-top:0.6rem;color:#a1abc1;">{count} img</div>',
                unsafe_allow_html=True,
            )
            row_cols[2].markdown(
                f'<div style="padding-top:0.6rem;color:'
                f'{"#4ade80" if status == "OK" else "#fbbf24"};">{status}</div>',
                unsafe_allow_html=True,
            )
            if row_cols[3].button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                st.session_state.pending_delete = name

        if st.session_state.pending_delete:
            target = st.session_state.pending_delete
            st.warning(
                f"⚠️ Permanently delete **{target}** and all their training photos? "
                "This cannot be undone."
            )
            cd1, cd2, _ = st.columns([1, 1, 4])
            if cd1.button("Yes, delete", key="confirm_delete", type="primary"):
                ENCODER.delete_person(target)
                st.session_state.pending_delete = None
                st.success(f"Deleted {target}.")
                st.rerun()
            if cd2.button("Cancel", key="cancel_delete"):
                st.session_state.pending_delete = None
                st.rerun()

    st.markdown("---")
    st.markdown("### ➕ Add / extend a person")

    name_col, _ = st.columns([2, 3])
    with name_col:
        new_name = st.text_input(
            "Folder name (use underscores)", placeholder="e.g. Mohammed_Alhuwaivi"
        )
    uploaded = st.file_uploader(
        "Upload images (JPG / PNG, several recommended)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    save_col, rebuild_col = st.columns(2)
    with save_col:
        if st.button("💾 Save images", use_container_width=True):
            if not new_name.strip():
                st.error("Enter a folder name first.")
            elif not uploaded:
                st.error("Upload at least one image.")
            else:
                target_dir = CFG.dataset_dir / new_name.strip().replace(" ", "_")
                target_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for i, f in enumerate(uploaded):
                    suffix = Path(f.name).suffix.lower() or ".jpg"
                    out_path = target_dir / f"{target_dir.name}_{stamp}_{i:02d}{suffix}"
                    out_path.write_bytes(f.read())
                st.success(f"Saved {len(uploaded)} image(s) to `{target_dir.name}`.")
    with rebuild_col:
        if st.button("🧠 Rebuild encodings", use_container_width=True):
            progress = st.progress(0.0, text="Building encodings...")
            def _cb(i, total, name):
                progress.progress(min(1.0, i / max(total, 1)), text=f"Encoding: {name}")
            stats = ENCODER.build_database(force_rebuild=True, progress_callback=_cb)
            progress.empty()
            st.success(f"Encoded {stats['encoded']} face(s) for {stats['people']} person(s).")


# ===========================================================================
# REPORTS TAB
# ===========================================================================

def _load_all_attendance(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("attendance_*.csv"))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df["Source"] = fp.name
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


with reports_tab:
    st.markdown("### 📈 Attendance analytics")
    df_all = _load_all_attendance(CFG.attendance_dir)
    if df_all.empty:
        st.info("No data yet. Run a session first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            _metric_card("Total records", str(len(df_all)), "across all sessions"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _metric_card("People recognized", str(df_all["Name"].nunique()), ""),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _metric_card("Sessions", str(df_all["Source"].nunique()), "CSV files"),
            unsafe_allow_html=True,
        )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("#### Sessions attended per person")
        counts = df_all["Name"].value_counts().rename_axis("Name").reset_index(name="Sessions")
        st.bar_chart(counts.set_index("Name"))

        st.markdown("#### Sessions over time")
        df_all["Date"] = pd.to_datetime(df_all["Date"])
        per_day = df_all.groupby("Date").size().rename("Records")
        st.line_chart(per_day)

        # Punctuality breakdown — only meaningful when Status is populated.
        if "Status" in df_all.columns:
            df_status = df_all[df_all["Status"].isin(["ON_TIME", "LATE"])]
            if not df_status.empty:
                st.markdown("#### Punctuality")
                pc1, pc2 = st.columns([1, 1])
                with pc1:
                    counts = df_status["Status"].value_counts()
                    st.bar_chart(counts)
                with pc2:
                    latecomers = (
                        df_status[df_status["Status"] == "LATE"]
                        .groupby("Name").size()
                        .sort_values(ascending=False)
                        .reset_index(name="Late count")
                    )
                    st.markdown("**Top latecomers**")
                    if latecomers.empty:
                        st.caption("Nobody late yet.")
                    else:
                        st.dataframe(latecomers, use_container_width=True, hide_index=True)
