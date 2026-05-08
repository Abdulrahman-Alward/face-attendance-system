# Face-Based Attendance System — Project Report Draft

**Course:** AI385 — Project Implementation

**Team:**
- Abdulrahman Mohammed Alward — 4310028
- Mohammed Abdullah Alhuwaivi — 4310615
- Abdulazeez Talaat Mugharbel — 4411851

> This is a working draft. Sections are written so you can lift them
> into the final report with light editing — adjust phrasing to match
> your professor's preferred style and add screenshots where indicated.

---

## 1. Abstract

We built a **real-time face-based classroom attendance system**. A teacher
points a webcam at the room, presses **Start**, and the system continuously
detects the faces in view, identifies each one against a registered
roster, and writes attendance records and entry/exit events to disk. The
software is delivered as a single desktop application with a modern web
GUI (Streamlit), and supports three different face-recognition back-ends
that the user can switch between at runtime depending on the available
hardware: a CPU-friendly path based on **dlib**, a GPU-accelerated path
using **PyTorch + FaceNet**, and a state-of-the-art path using **ArcFace**
via the `insightface` library. The system can detect students sitting
across a classroom, tolerates briefly missed frames through a tracking
mechanism, defends against the common "hold up a printed photo" spoofing
attempt with an eye-blink liveness check, and produces clean CSV
attendance reports plus a separate event log of every entry and exit.

---

## 2. Problem Statement and Motivation

Classroom attendance is traditionally taken by reading names off a list
or by passing a sign-in sheet around. Both methods are slow, easy to
forge (signing for an absent classmate), and disrupt the start of every
session. Universities have experimented with RFID cards and QR-based
sign-ins, but cards can be lent, codes can be screenshotted, and any
hardware extra to the laptop already on the desk is one more thing to
break or forget.

The cleanest input that uniquely identifies a student is *the student
themselves*. A camera and a face-recognition pipeline turn the existing
classroom into the attendance system. There is nothing for the student
to carry, nothing for the teacher to pass around, no extra hardware, and
no way to "sign in" for a friend.

Our goal is therefore an attendance system that:

1. Identifies registered students automatically from a live camera feed.
2. Logs each student's entry and exit times during a session.
3. Refuses to be fooled by trivial spoofs (a phone showing a face).
4. Runs on a regular laptop, with optional acceleration on machines that
   happen to have an NVIDIA GPU.
5. Presents the result through an interface a non-technical teacher can
   use without reading documentation.

---

## 3. System Overview

The system is a single-machine desktop application written in Python.
The user interacts through a web-style GUI rendered in the browser by
Streamlit, and the same machine's camera and storage are used end to
end. There is no cloud component; all data stays on the local computer.

A high-level view of the pipeline:

```
Webcam ──▶ Camera Worker ──▶ Face Detector ──▶ Tracker (IoU)
                                  │                │
                                  ▼                ▼
                          Face Encoder       Skip if known
                                  │
                                  ▼
                          Identity Match ──▶ Presence Tracker
                                                   │
                                            ┌──────┴───────┐
                                            ▼              ▼
                                       attendance       events
                                       (CSV / db)       (CSV / db)
                                            │              │
                                            └──────┬───────┘
                                                   ▼
                                             Streamlit UI
                                             (live view +
                                              attendance + reports)
```

The same physical pipeline runs at every selected backend; only the
detection and embedding modules change.

---

## 4. Functional Features

The system supports the following user-visible capabilities:

### 4.1 Live recognition view

A real-time annotated video feed of the camera, with green boxes around
recognized students (showing name and confidence) and red boxes around
unrecognized faces. Frames per second and the current present-count are
shown next to a live indicator. *(Insert screenshot of Live tab.)*

### 4.2 Automatic attendance logging

Once a student has been continuously recognized for a configurable
number of frames (the *confirmation window*, default five), they are
recorded as present in a daily CSV file together with timestamp,
recognition confidence, and an on-time / late status.

### 4.3 Entry / exit event log

Independently of the attendance file, every "person enters frame" and
"person leaves frame" event is appended to an event log. A student can
re-enter and re-leave many times in a single session — every transition
is captured. Leaving is detected by absence: if a recognized student
hasn't been seen for the configured leave-timeout (default one minute),
a **LEAVE** event is emitted.

### 4.4 Late-arrival classification

The teacher can declare the class start time and a grace period.
Attendance taken before the cutoff is tagged **ON_TIME**; attendance
after, **LATE**. The Reports tab summarizes the breakdown and shows the
top latecomers across all sessions.

### 4.5 Anti-spoofing through blink detection

When enabled, a recognized face must visibly blink (an
eye-aspect-ratio drop and recovery) before attendance is marked.
A printed photo or a static phone display will never blink, and is
therefore never confirmed.

### 4.6 Roster management from the GUI

Students are enrolled by uploading 3–10 photos through the **Dataset**
tab; the system computes their face embeddings and stores them in a
local file. A 🗑 button next to each name allows clean removal of a
student and all their training photos with a confirmation prompt.

### 4.7 Reports and analytics

A **Reports** tab summarizes attendance across all sessions: total
records per student, sessions over time, on-time vs. late breakdown,
and a leader-board of latecomers. CSV downloads are available in every
tab so the teacher can hand the file to administration.

### 4.8 Persistent settings

The teacher's preferences (camera index, recognition tolerance, leave
timeout, target FPS, class start time, anti-spoofing, etc.) are saved
to a JSON file and survive restarts.

### 4.9 Multi-backend operation with GPU detection

The application recognizes which back-ends are installed on the host
machine and lets the teacher pick from a sidebar dropdown:

- **dlib** — works on every system, CPU-friendly.
- **torch** — appears when PyTorch is installed; uses CUDA automatically.
- **arcface** — appears when `insightface` is installed; the highest
  recognition accuracy, also CUDA-accelerated.

A status pill in the sidebar shows the active hardware (e.g., *"CUDA
enabled via PyTorch · NVIDIA GeForce RTX 4060 Laptop GPU"*).

---

## 5. Technical Approach

### 5.1 The recognition problem

Face recognition has two distinct stages. First, the **detector** finds
the rectangles around faces in an image. Second, the **encoder** maps
each face crop to a fixed-length numerical vector — an *embedding* —
designed so that two photographs of the same person produce vectors
close to one another in some distance metric, while photographs of
different people produce far-apart vectors. Identification then reduces
to a nearest-neighbour search against a library of pre-computed
embeddings of registered students.

### 5.2 The three backends

We provide three implementations of this pipeline because the trade-offs
between speed, accuracy and hardware requirements are not the same for
every user.

| Backend | Detector | Embedding | Distance | Hardware |
|---|---|---|---|---|
| `dlib` | HOG (default) or dlib-CNN | 128-D ResNet | Euclidean | CPU |
| `torch` | MTCNN | 512-D InceptionResnetV1 | Cosine | CPU or CUDA |
| `arcface` | SCRFD | 512-D ArcFace | Cosine | CPU or CUDA |

`dlib`'s HOG detector is the fastest on CPU but misses small,
non-frontal, or partially occluded faces. `MTCNN` (used by the torch
backend) is multi-stage CNN-based and considerably more robust at the
cost of compute, but with CUDA available it stays comfortably under
30 ms per frame on our test hardware. `SCRFD + ArcFace` is the current
state-of-the-art for in-the-wild face recognition and is the most
forgiving of unusual angles and lighting, but is the largest dependency
to install. The application picks the most capable available backend on
first launch automatically.

### 5.3 Real-time pipeline optimizations

A naïve implementation that detects, encodes, and matches every face on
every frame is too slow for a classroom of 15 to 20 people on a CPU —
empirically, around four frames per second. We apply two optimizations
to bring this up to interactive rates without sacrificing accuracy:

1. **Tracking by intersection-over-union.** Detection (cheap) runs every
   frame. For each detection, we compare the bounding box against the
   bounding boxes from the previous frame. If the overlap is high
   (default 40%), we *reuse* the cached identity instead of re-running
   the slow encoder. Only genuinely new faces, or faces older than a
   re-identify window (default 3 s), pay the encoding cost.

2. **Decoupled capture and display threads.** The webcam capture,
   detection, encoding, and presence-tracking all run in a background
   worker thread. The Streamlit display loop reads the latest annotated
   frame from a shared snapshot and renders at a smooth target FPS,
   independent of how long any single recognition pass takes. A slow
   encoding stage no longer drops the perceived UI frame-rate.

Together, these mean the pipeline behaves like a "free" recognition for
a static classroom: the first few seconds confirm everyone, and from
then on detection runs at near-camera-native rate with very little
extra work.

### 5.4 Confirmation, presence and leave detection

The system distinguishes three states for each known student:

- **Detected** — face seen in the most recent frame.
- **Currently present** — has been confirmed and is in the room.
- **Marked attendance** — has been confirmed at least once this session
  (written to the daily CSV).

A student moves from *detected* to *currently present* (and *marked
attendance*) only after being recognized for *N* consecutive frames.
This *multi-frame confirmation* defends against single-frame false
positives where one bad detection briefly looks like a different student.
A student returning after a previous LEAVE skips the confirmation
window, since they have already been confirmed in this session.

A student moves from *currently present* back to *not present* when
they have not been seen for the configured **leave timeout** (in
minutes). Each such transition emits a LEAVE event. Re-entries emit a
new ENTER event.

### 5.5 Embedding augmentation during enrollment

A single training photograph captures a single pose under a single
lighting. To make recognition robust to natural variation, the enroller
produces three additional embeddings per training image — a horizontal
flip, a brightened version (+20%), and a darkened version (−20%) —
quadrupling the database without requiring more photos from the
student. Distance computation at query time is unchanged in cost
because the matching loop already runs on the full database in vector
form.

### 5.6 Anti-spoofing

The blink detector uses dlib's 68-point facial landmark predictor to
compute the **eye-aspect-ratio (EAR)** every frame for every recognized
face. EAR is the average vertical-to-horizontal eye-opening ratio; it
is roughly 0.30 when the eye is open and drops below 0.20 for a few
frames during a blink. The system marks a person as *liveness-passed*
once it has observed both a low (closed) and a subsequent high (open)
EAR sample within a short rolling window (~2 seconds). Until that
condition is met, the bounding box is labeled "blink to confirm" and no
attendance is written. A printed photo or a static phone display has no
EAR motion and so is never confirmed.

### 5.7 Persistent storage

Three pieces of data survive between runs:

- **Embeddings database** (`data/encodings_<backend>.pkl`): the
  pre-computed face embeddings of every registered student. Each
  backend has its own file because the embeddings are not compatible
  across models.
- **Attendance reports** (`data/attendance_reports/attendance_<date>.csv`):
  one row per student, per session, with timestamp and on-time / late
  status.
- **Event logs** (`data/logs/events_<date>.csv`): one row per ENTER or
  LEAVE transition.

User preferences are persisted to `data/settings.json`.

---

## 6. Architecture

The codebase is small and intentionally modular; each module has a single
responsibility.

| File | Role |
|---|---|
| `app.py` | Streamlit GUI — sidebar, tabs, live view, dataset editor, reports. The only file the end user runs. |
| `face_attendance.py` | Core logic. Defines `Config`, `FaceEncoder`, `TrackingRecognizer`, `PresenceTracker`, `annotate_frame`. Picks a backend at runtime via `get_backend()`. |
| `face_lib.py` | dlib backend (HOG detection + 128-D ResNet embeddings). |
| `torch_face_lib.py` | PyTorch backend (MTCNN detection + 512-D InceptionResnetV1 embeddings). |
| `insightface_lib.py` | ArcFace backend (SCRFD detection + 512-D ArcFace embeddings). |
| `anti_spoof.py` | Eye-aspect-ratio computation and blink-detection state machine. |
| `camera_worker.py` | Background-thread wrapper around the webcam and recognition pipeline. |
| `smoke_test.py` | Seven-check verification suite covering imports, model files, recognition pipeline, presence tracking, optional backends, and config persistence. |

All three backends expose the same five-function interface
(`load_image_file`, `face_locations`, `face_encodings`, `match`,
`cuda_status`), which is what makes them interchangeable from the
core logic's point of view.

The directory layout:

```
Project/
├── app.py
├── face_attendance.py
├── face_lib.py             # dlib backend
├── torch_face_lib.py       # PyTorch / FaceNet backend
├── insightface_lib.py      # ArcFace backend
├── anti_spoof.py
├── camera_worker.py
├── smoke_test.py
├── requirements.txt
├── run.bat
├── models/
│   ├── LICENSE_face_recognition_models  (MIT, attribution)
│   └── face_recognition_models/
│       └── models/  (the four dlib .dat model files)
└── data/                   (gitignored runtime artifacts)
    ├── dataset/
    ├── attendance_reports/
    ├── logs/
    ├── encodings_dlib.pkl
    ├── encodings_torch.pkl
    └── settings.json
```

---

## 7. User Interface

The GUI was built in Streamlit because it lets us write the entire
front-end in Python (no separate web stack) and produces a clean, dark
themed, web-style interface that works either as a local desktop app or
through any browser pointed at `localhost`. Custom CSS gives it a
modern look-and-feel with translucent cards, gradient header, status
pills, and an icon-categorized sidebar.

The interface has five tabs:

1. **Live** — real-time annotated camera feed, present-now panel,
   recent events, present-count badge, FPS readout.
2. **Attendance** — daily attendance reports with a CSV viewer and
   download button.
3. **Event Logs** — every ENTER and LEAVE transition, filterable by
   student.
4. **Dataset** — list of registered students with delete buttons; image
   upload to enroll new students; rebuild-encodings button.
5. **Reports** — cross-session analytics: sessions per person,
   sessions over time, on-time/late breakdown, top latecomers.

The sidebar contains, in order: a CUDA / hardware status pill, the
backend selector, basic settings (camera, tolerance, leave timeout,
class start time), distance-detection parameters (min face size /
upsample), and an *Advanced* expander with confirmation, tracking,
enrollment-augmentation, and anti-spoofing toggles. A *Database*
summary card at the bottom shows how many students are registered, and
a *Rebuild encodings* button forces a recomputation if the user has
changed the underlying photo set.

*(Insert screenshots of the five tabs and the sidebar.)*

---

## 8. Implementation Notes

### 8.1 Language and libraries

The project is written entirely in Python 3.12. The principal external
libraries are:

- **OpenCV** — webcam capture, frame manipulation, drawing.
- **dlib** — frontal-face detection (HOG and CNN) and 128-D ResNet
  encoder.
- **PyTorch** + **facenet-pytorch** — MTCNN detector and
  InceptionResnetV1 encoder for the torch backend.
- **insightface** + **ONNX Runtime** — SCRFD + ArcFace for the arcface
  backend.
- **Streamlit** — the GUI framework.
- **pandas** + **NumPy** — data manipulation and embedding arithmetic.
- **Pillow** — image loading from disk.

### 8.2 Reusing pre-trained models

We do not train any deep network ourselves; this is a software
engineering project on top of well-established research. Specifically:

- The dlib facial landmark predictor and 128-D ResNet face encoder are
  taken from Adam Geitgey's `face_recognition_models` repository (MIT
  License), included in `models/face_recognition_models/`.
- InceptionResnetV1 weights are downloaded by `facenet-pytorch` from
  the original VGGFace2 release (~89 MB).
- ArcFace weights are downloaded by `insightface` from its standard
  `buffalo_l` model pack (~280 MB).

This is appropriate scope for a project-implementation course: the
problem we are solving is *building the system around face recognition*,
not *advancing face recognition itself*.

### 8.3 Privacy

All processing is on-device. No frames, embeddings, or attendance data
ever leave the host machine. The training images are stored as files on
disk; the user can delete them at any time through the GUI.

---

## 9. Testing and Validation

We provide a smoke-test script (`smoke_test.py`) with seven independent
checks:

1. All required modules import cleanly.
2. The four dlib model files are present at the expected location.
3. The dlib face-locations call runs end-to-end.
4. The configuration, encoder, and tracking recognizer construct.
5. The presence tracker emits ENTER/LEAVE events to CSV correctly,
   including multi-frame confirmation behaviour.
6. The optional torch/arcface backends are either usable or fail
   gracefully when absent.
7. Configuration persistence (save → load) round-trips correctly.

Manual testing was performed against a live webcam with a small
registered roster, exercising the live view, attendance logging, leave
detection, anti-spoofing (using a printed photo and a phone), backend
switching, and re-enrollment after deletion. *(Insert any concrete
performance numbers you measured: FPS at different backends, distance
at which faces are still detected, etc.)*

---

## 10. Results

*(Fill in with the numbers you collect during your demo. Suggested
metrics:)*

- **Recognition accuracy** on the registered roster, measured as the
  fraction of frames where a recognized student is correctly identified.
- **Frames per second** in the live loop, broken down by backend.
- **Maximum reliable distance** at which a face is still detected,
  measured for each backend.
- **Anti-spoofing effectiveness** — does the system correctly refuse to
  mark attendance when shown a printed photo or phone display?
- **Time to enroll** a new student (number of clicks, seconds end-to-end).

---

## 11. Limitations

- The system identifies only faces that are already enrolled. An
  unknown face is correctly labeled "Unknown" but contributes no
  information.
- Detection quality degrades sharply for faces smaller than ~12 pixels
  wide. With a standard 720p webcam in a typical classroom this means
  practical range ends at roughly 5–6 metres; beyond that, the face is
  simply too small for any of our detectors.
- The blink-based liveness check defends against the most common
  spoofing attempt (a static photograph) but is not a strong
  anti-spoofing system. A short looping video of a real face would
  pass.
- All processing is single-machine; we do not support multiple cameras
  in different rooms.
- Lighting matters. Strong backlighting (a window behind the student)
  or near-darkness will reduce recognition accuracy; we mitigate this
  with brightness-augmented embeddings during enrollment but do not
  eliminate the problem.

---

## 12. Possible Extensions

We deliberately scoped this project to "what one team can ship in a
semester," but several directions would be natural next steps:

- **A roster view** showing every enrolled student as a card with
  their photo, current status, and last-seen time — turning the data
  the system already collects into a teacher-facing dashboard.
- **Manual attendance override** so the teacher can mark a student
  present even when the recognition fails.
- **A SQLite back-end** instead of one CSV per day, for proper queries
  ("every session Ahmed was late this month") and concurrent access.
- **A REST API** exposing the attendance data so the school's other
  systems can read it without parsing files.
- **Stronger liveness detection** using a small CNN trained specifically
  on real-vs-spoof face data, beyond the blink heuristic.
- **Multi-camera support** for very wide classrooms.
- **A Roster import / bulk enrollment** tool, taking a folder of
  pre-named photos.

---

## 13. Conclusion

The project demonstrates that an effective real-time classroom
attendance system can be assembled from existing computer-vision
components, packaged behind a usable interface, and run on the kind of
laptop a teacher already owns. By offering three interchangeable
back-ends, the system stays usable on a CPU-only machine while taking
full advantage of CUDA when present. The modular architecture, with a
small per-backend interface and a single shared core, made it
straightforward to add features (multi-frame confirmation, anti-spoofing,
threading, late-classification, persistent settings) without rewriting
the recognition pipeline. The result is an honest, locally-deployed,
privacy-respecting alternative to traditional sign-in sheets.

---

## 14. References

The reproducible model weights and reference implementations we built on
are credited in the source files (`face_lib.py`, `torch_face_lib.py`,
`insightface_lib.py`). For the report bibliography:

- Geitgey, A. *face_recognition* — open-source library and pre-trained
  dlib models. https://github.com/ageitgey/face_recognition
- King, D. E. (2009). *Dlib-ml: A Machine Learning Toolkit*. Journal of
  Machine Learning Research.
- Schroff, F., Kalenichenko, D., Philbin, J. (2015). *FaceNet: A
  Unified Embedding for Face Recognition and Clustering*. CVPR.
- Zhang, K. et al. (2016). *Joint Face Detection and Alignment Using
  Multi-task Cascaded Convolutional Networks (MTCNN)*. IEEE Signal
  Processing Letters.
- Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). *ArcFace: Additive
  Angular Margin Loss for Deep Face Recognition*. CVPR.
- Soukupová, T., Čech, J. (2016). *Real-Time Eye Blink Detection Using
  Facial Landmarks*. CVWW.
- Streamlit. https://streamlit.io
