# Code Guide — How the Project Works, File by File

This document walks through every Python file in the project **except
`app.py`** (which is the Streamlit GUI; the *what-the-user-sees* code).
It is written for someone who didn't build the project — a teammate
catching up, a reviewer, or anyone who needs to explain the system to
other people.

For each file you'll see:

- **Why it exists** — the role it plays in the system.
- **What it does** — the behaviour it implements, in plain language.
- **How it works** — the internal mechanics, with enough detail to
  explain to someone else.
- **Public API** — the classes and functions other files use.
- **Design notes** — the non-obvious decisions and the reasons behind
  them.

The files form three layers:

```
                ┌─────────────────────────────────┐
       UI       │ app.py  (Streamlit GUI)         │   ← not covered here
                └────────────────┬────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────┐
       Core     │ face_attendance.py              │
                │  ┌───────────────┐              │
                │  │ Config        │              │
                │  │ FaceEncoder   │              │
                │  │ TrackingRecog │              │
                │  │ PresenceTrack │              │
                │  └───────────────┘              │
                │ camera_worker.py                │
                │ anti_spoof.py                   │
                └────────────────┬────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────┐
       Backends │ face_lib.py        (dlib)       │
                │ torch_face_lib.py  (PyTorch)    │
                │ insightface_lib.py (ArcFace)    │
                └─────────────────────────────────┘

                ┌─────────────────────────────────┐
       Tests    │ smoke_test.py                   │
                └─────────────────────────────────┘
```

The UI calls into the **Core**, which calls into one of the three
**Backends** (chosen at runtime). All three backends expose the
same five functions, so the core code never branches based on which
backend is active — it just calls the chosen module.

---

## 1. `face_attendance.py` — the core of the system

### Why it exists

This is the central module. It holds the configuration the user can
tweak, the database of registered face embeddings, the recognition
loop, and the bookkeeping that turns recognitions into attendance.
Everything the UI does ultimately runs through this file.

### What's inside

The file defines four classes and one helper function:

| Symbol | What it represents |
|---|---|
| `Config` | All the settings: backend choice, tolerances, paths, FPS targets, anti-spoofing toggles, etc. Also handles save/load to a JSON file. |
| `FaceEncoder` | The face database. Knows how to scan the dataset folder, turn photos into numerical embeddings, save them to disk, and reload them. |
| `TrackingRecognizer` | The fast-path recognizer used in the live loop. Runs on every frame; uses tracking so it doesn't waste effort re-encoding faces that haven't moved. |
| `PresenceTracker` | The bookkeeping layer. Decides who counts as "currently present", writes attendance, emits ENTER and LEAVE events. |
| `annotate_frame()` | A drawing helper — given a frame and a list of recognition results, returns the same frame with green/red bounding boxes and labels drawn on it. |

There's also a small `get_backend(name)` function that returns the
right backend module (`face_lib`, `torch_face_lib`, or
`insightface_lib`) based on a string. That's how the system swaps
backends at runtime.

### How it works — concept by concept

#### 1.1 The pipeline in one sentence

> Capture a frame → detect faces → for each face, look it up in the
> database → if a tracked match exists, reuse the identity; otherwise
> encode the face and find the nearest match — then update the
> presence tracker so attendance and events get written.

#### 1.2 `Config` — the source of truth for every setting

A simple Python *dataclass*. Reading a `Config` instance tells you
everything about the current run: which backend is active, the camera
index, the recognition tolerance, the leave-timeout, the class start
time, and so on. The UI's sidebar widgets read from and write to this
object directly.

Two non-obvious features:

- **`apply_backend_defaults()`** — when the user switches backend, the
  *sensible* values for several settings (frame resize factor,
  detection model, min face size) change too. This method resets them
  to per-backend defaults so the user doesn't have to retune manually.
- **`save_to(path)` / `load_from(path)`** — settings survive across
  app restarts by writing to `data/settings.json`. Only a curated
  allow-list of fields is persisted (`_PERSISTED_FIELDS`), so things
  like the per-session label or computed paths don't get baked in.

#### 1.3 `FaceEncoder` — building and querying the roster

When a student is enrolled, the user drops 3–10 photos in
`data/dataset/<name>/`. `FaceEncoder` is what turns those photos into
the system's mental model of that student.

The process for one photo:

1. **Load** the image as an RGB numpy array.
2. **Detect** all faces in the photo.
3. If multiple faces are detected, keep only the **largest** — we
   assume that's the subject of the photo.
4. **Encode** the face into a numerical vector (the *embedding*) using
   whichever backend is active. dlib produces a 128-D vector; the
   torch and arcface backends produce 512-D vectors.
5. Append the vector and the student's name to two parallel lists
   (`self.encodings` and `self.names`).

After all photos are processed, the encoder **saves** the two lists to
a `.pkl` file (`data/encodings_<backend>.pkl`). Each backend has its
own file because the embeddings aren't compatible between backends
(different dimensions, different distance metrics).

**Why the per-photo embeddings are augmented.** If
`Config.augment_during_enrollment` is true (the default), each photo
produces *four* embeddings instead of one: the original, a horizontal
flip, a brighter version, and a darker version. This makes the
database more robust to lighting and angle variation at zero cost
during recognition (we still do the same single-distance computation;
there are just more vectors to compare against).

**`delete_person(name)`** removes a student cleanly: it deletes their
photos folder under `data/dataset/`, filters their embeddings out of
the two lists, and saves. The roster row in the UI calls this when
the trash button is clicked.

#### 1.4 `TrackingRecognizer` — the fast-path recognizer

This is the recognizer the live video loop uses. Its goal is to be
fast enough to run on every captured frame even though the *full*
recognition pipeline (detection + encoding + matching) is too slow
when done from scratch every frame.

The trick is *don't re-encode faces that haven't moved*.

The algorithm on each call to `process(frame)`:

1. **Detect** every face in the frame. This is cheap relative to
   encoding.
2. For each detected box, try to **match** it against the previous
   frame's recognized faces:
   - **Primary criterion: IoU.** If the new box overlaps an old box
     by at least `iou_match_threshold`, that's clearly the same
     person.
   - **Fallback: centroid distance.** If IoU is too low (the person
     turned their head — boxes don't overlap much even though they're
     in the same general spot), check whether the centroid of the new
     box is within half a face-width of an old box. If yes, still the
     same person.
3. If a matched track is **fresh enough** (encoded less than
   `reidentify_every_seconds` ago, default 20 s), **reuse** the
   cached identity. **No encoding runs.**
4. Otherwise, the face is new (or stale) — run the slow encoder
   only on **this one face**, find the closest match in the database,
   and assign that identity.

A small private dataclass `_TrackedFace` (`box`, `name`, `confidence`,
`last_encoded_at`) is what the recognizer keeps in memory between
calls. The set of tracked faces is replaced on every frame so anyone
who left the frame is naturally dropped.

**The shape of the load this produces** is the key result: in a
classroom of 20 students seated normally, after the first ~5 seconds
the entire room is on cached identities, and the recognizer only does
detection on every frame plus a quick re-encode for one face every
second on average (20 students ÷ 20-second re-identify window). The
encoder — the slow part — is essentially idle.

The `reset()` method clears the tracks; the app calls it when a new
session starts so the previous session's bounding boxes don't
contaminate the new one.

#### 1.5 `PresenceTracker` — turning recognitions into attendance

`TrackingRecognizer` tells us *what's in this frame right now*. The
presence tracker decides *who actually counts as having attended*.
The two concepts are distinct:

- A student might appear in one frame because of a single bad
  recognition. We don't want that to count as attendance.
- A student might leave the frame for 10 seconds while reaching for
  something off-camera. We don't want that to log a LEAVE event.

These rules turn raw recognitions into clean attendance:

**Rule 1 — multi-frame confirmation.** A name has to appear in
`consecutive_recognition_threshold` consecutive frames (default 5)
before it is *confirmed* this session. Until that threshold is
crossed, nothing is written. After confirmation, the student is
added to `currently_present` and the attendance CSV is updated. Names
that have been confirmed once skip this check on re-entry — they're
already known.

**Rule 2 — leave by absence.** A student who has been confirmed is in
`currently_present` until the system has not seen them for
`leave_timeout_minutes` (default 1 minute). At that point a LEAVE
event is emitted. If they come back, another ENTER event is emitted.

**Rule 3 — late vs on-time.** If `class_start_time` is set, the
attendance row gets a `Status` column: `ON_TIME` if the student was
confirmed before the cutoff (start time + grace minutes), `LATE`
otherwise.

**Rule 4 — anti-spoofing (optional).** When `require_blink_to_mark`
is on, the tracker won't mark a student confirmed until a *liveness
check* function (passed in from the live loop) returns True. The
check is implemented in `anti_spoof.py` and is based on detecting an
eye-blink — a printed photograph never blinks.

Every transition is written to **two** files:

- `data/attendance_reports/attendance_<date>.csv` — one row per
  unique student per session, with timestamp and on-time/late status.
- `data/logs/events_<date>.csv` — one row per ENTER and LEAVE.

There's also an in-memory `events` list the UI reads for the
"Recent events" panel.

`force_leave_all()` is called when the user clicks Stop in the UI; it
emits a clean LEAVE event for everyone still in `currently_present`
so the records show the session ending properly.

#### 1.6 `annotate_frame()` — drawing the boxes

A small helper. Given a BGR frame and a list of recognition results,
it returns a copy of the frame with bounding boxes and labels drawn
on it: green box + name + confidence percentage if recognized, red
box + "Unknown" otherwise. This is what the live video tab actually
displays.

### Public API summary

```python
from face_attendance import (
    Config,
    FaceEncoder,
    TrackingRecognizer,
    PresenceTracker,
    annotate_frame,
    get_backend,
    TORCH_BACKEND_AVAILABLE,
    INSIGHTFACE_BACKEND_AVAILABLE,
)
```

### Design notes

- **The three backends are interchangeable.** Every backend exposes
  the same five functions (`load_image_file`, `face_locations`,
  `face_encodings`, `match`, `cuda_status`). The core code calls them
  via `self.backend.face_locations(...)` etc., and never branches on
  which backend is active. Adding a fourth backend would mean
  creating a fourth module that follows the same interface — no
  changes to `face_attendance.py` beyond an import.
- **The per-backend `.pkl` files** mean switching backends never
  destroys the previous backend's database. The user can flip
  between dlib and arcface freely; both are remembered.
- **Most state is per-session, not global.** A new `PresenceTracker`
  is created when the user clicks Start. This is what guarantees that
  multi-frame confirmation and the "already confirmed" optimisation
  reset cleanly between class periods.

---

## 2. The Three Backends

The system supports three recognition libraries. Each one is exposed
through a Python module with the exact same five-function interface,
so the core can call any of them through a `get_backend(...)`
indirection.

| Module | Detector | Embedding | Distance | Hardware |
|---|---|---|---|---|
| `face_lib.py` | HOG / dlib-CNN | 128-D ResNet | Euclidean | CPU |
| `torch_face_lib.py` | MTCNN | 512-D FaceNet | Cosine | CPU or CUDA |
| `insightface_lib.py` | SCRFD | 512-D ArcFace | Cosine | CPU or CUDA |

The five-function interface every backend implements:

```python
load_image_file(path)        # read image from disk as an RGB numpy array
face_locations(image, ...)   # detect faces, return (top, right, bottom, left)
face_encodings(image, ...)   # encode each face, return list of vectors
match(query, db_vecs, names) # nearest-neighbour, return (name, confidence)
cuda_status()                # human-readable description of GPU state
```

This is what makes the three modules interchangeable. The core
doesn't care which one is loaded; it just calls these five functions.

---

### 2a. `face_lib.py` — the dlib backend (CPU-friendly default)

#### Why it exists

dlib is the easiest backend to get working — it ships with prebuilt
binaries for every platform and runs reasonably on a plain CPU. We
treat it as the default so the project works on any machine, even
one without an NVIDIA GPU.

#### What it does

It wraps dlib's prebuilt face detector and a 128-dimensional face
embedding model. Both ship from Adam Geitgey's public
`face_recognition_models` repository (the same models the popular
`face_recognition` Python library uses). The actual `.dat` files live
under `models/face_recognition_models/models/` and are loaded once
at import time.

#### How it works

At module import:

1. The four `.dat` files are checked to exist (helpful error if not).
2. dlib's five model objects are instantiated and kept as
   module-level singletons:
   - HOG face detector (the fast, default detector).
   - CNN face detector (slower but more robust; only used when the
     user selects "cnn" in the sidebar).
   - 5-point landmark predictor (used during encoding to align the
     face).
   - 68-point landmark predictor (used by `anti_spoof.py` for blink
     detection).
   - 128-D face encoder (ResNet-style network producing the embedding).

When `face_locations(image)` is called: the chosen detector runs and
returns a list of rectangles, converted from dlib's
`(left, top, right, bottom)` convention to the project's
`(top, right, bottom, left)` convention.

When `face_encodings(image, boxes)` is called: each box is fed to the
landmark predictor to get the face's pose, and the aligned face is
fed to the encoder to produce a 128-D vector.

`match(query, db, names, tol)` computes the Euclidean distance from
the query embedding to every embedding in the database, picks the
nearest, and returns its name if the distance is below the tolerance.
The reported "confidence" is `1 - distance/tolerance` clipped to
[0, 1] — a rough heuristic, not a calibrated probability.

#### Design notes

- **The 68-point predictor is loaded even though encoding uses the
  5-point one.** Why? Because the *anti-spoofing* module uses
  the 68-point landmarks to compute eye-aspect-ratio for blink
  detection. Loading it once at backend import is cheaper than
  loading it on first blink check.
- **CUDA in dlib is detected but not automatic.** The
  `cuda_status()` function reports whether dlib was built with CUDA
  support. Most pip-installed dlib wheels on Windows are CPU-only;
  building with CUDA requires compiling dlib from source. We
  don't force this — the backend reports honestly what's available.

---

### 2b. `torch_face_lib.py` — the PyTorch / FaceNet backend (CUDA-accelerated)

#### Why it exists

When the host machine has an NVIDIA GPU, dlib's HOG detector still
struggles with faces that aren't perfectly frontal. PyTorch + the
`facenet-pytorch` library give us *MTCNN* (a strong cascade-CNN
detector) and *InceptionResnetV1* (a deep face embedding network
trained on the VGGFace2 dataset) — both of which run on CUDA
automatically when available. Installation is also far easier than
building a CUDA-enabled dlib from source.

#### What it does

It exposes the same five-function interface as `face_lib.py`, but
the heavy lifting is done by PyTorch on the GPU when one is
available.

#### How it works

At module import:

1. Check `torch.cuda.is_available()`; pick `cuda:0` if so, otherwise
   `cpu`.
2. Instantiate MTCNN with detection-time keypoint estimation enabled.
   MTCNN is reused across calls so we don't pay the load cost twice.
3. Instantiate the InceptionResnetV1 face encoder pretrained on
   VGGFace2. Weights download on first run (~89 MB, cached forever).

When `face_locations(image)` is called: MTCNN's `detect(...)` returns
a list of bounding boxes and a per-detection confidence score. Boxes
below a configurable threshold are filtered out, the rest are
converted to the project's `(top, right, bottom, left)` format.

When `face_encodings(image, boxes)` is called: each box is cropped
out of the original image, resized to 160×160 (what InceptionResnetV1
expects), normalised to the network's pixel range, and forwarded
through the network on the GPU. The result is a 512-D embedding.

`match()` uses **cosine distance** (not Euclidean) because FaceNet
embeddings live on a 512-D unit sphere — cosine is the natural metric
for that geometry. The confidence value is computed the same way as
the dlib backend: `1 - distance/tolerance`, clipped to [0, 1].

A `configure(...)` function lets the core change MTCNN's
`min_face_size` at runtime — when the user moves the *Min face size*
slider in the sidebar, we rebuild MTCNN with the new value. (MTCNN
doesn't accept a dynamic minimum; we have to instantiate a new one.)

#### Design notes

- **The first run is slow** because `facenet-pytorch` downloads the
  pretrained weights. After that it's instantaneous.
- **MTCNN actually produces five facial keypoints**, but we discard
  them — the project uses only the bounding box. The blink-detection
  module needs 68-point landmarks, which MTCNN doesn't provide, so
  it falls back to dlib's predictor for that.

---

### 2c. `insightface_lib.py` — the ArcFace backend (state-of-the-art)

#### Why it exists

ArcFace is currently the state-of-the-art face recognition model for
in-the-wild conditions. It's noticeably more accurate than FaceNet on
non-frontal angles, partial occlusion, and difficult lighting. The
`insightface` library packages it together with SCRFD (a strong
single-shot detector) into one pipeline, running on the GPU through
ONNX Runtime.

#### What it does

Same five-function interface again — `load_image_file`,
`face_locations`, `face_encodings`, `match`, `cuda_status`. The
trick here is that *detection and encoding happen in a single
forward pass* through the model, which would normally make the
project's "call face_locations then face_encodings separately"
pattern wasteful.

#### How it works

At module import:

1. Detect whether `onnxruntime-gpu` is installed and CUDA is
   available; if so, use the `CUDAExecutionProvider`, otherwise fall
   back to `CPUExecutionProvider`.
2. Instantiate `FaceAnalysis(name="buffalo_l")` — that's
   `insightface`'s shorthand for the SCRFD + ArcFace model pack.
   Weights download on first run (~280 MB, cached forever).
3. Call `.prepare(...)` to fix the detection input size to 640×640.

The key trick — **same-frame caching.** When the core calls
`face_locations(frame)`, this backend internally runs the full SCRFD
+ ArcFace pipeline (because that's what insightface gives us) and
stores *both* the bounding boxes and their embeddings in a module-
level cache keyed by `id(frame)`.

When the core then calls `face_encodings(frame, boxes)` on the
*same* frame object (the common case in the live loop), this backend
recognises the cache and returns the pre-computed embeddings —
matching the requested boxes to the cached ones by IoU. No second
pass through the network.

If the cache key doesn't match (e.g. during initial enrollment when
each photo is a different image), the backend falls back to running
the full pipeline again. That's the only correct option in that
case.

`match()` uses cosine distance over L2-normalised 512-D embeddings,
same as the FaceNet backend (the geometry is the same; ArcFace just
produces better embeddings).

#### Design notes

- **insightface ships its own ONNX inference stack**, separate from
  PyTorch. This means we can install the arcface backend without
  installing PyTorch — though both can coexist if you want all three
  backends available simultaneously.
- **The same-frame cache is keyed by `id(frame)`** (the object's
  Python id), not the frame's contents. This is a deliberate
  shortcut: we know the caller (`TrackingRecognizer.process`) holds
  on to the same numpy array between its `face_locations` and
  `face_encodings` calls, so the id match is reliable.

---

## 3. `camera_worker.py` — the threading layer

### Why it exists

The most user-visible problem with a recognition system in Python is
that the slow step — encoding faces — also stalls the camera. If
recognition takes 100 ms per frame, the camera can only produce 10
frames per second, and the live feed looks choppy.

The fix is to **run the camera and the recognizer on separate
threads**, with the camera always feeding the latest frame and
recognition catching up whenever it's ready. The display can then
read both independently and stay smooth at the camera's native rate.

`camera_worker.py` is the module that owns this threading model.

### What it does

It exposes a single class, `CameraWorker`. The lifecycle:

```python
worker = CameraWorker(config, recognizer, presence_tracker, blink_detector)
worker.start()                # spawns two background threads

# any time:
(frame, frame_id, results,
 capture_fps, recognition_fps,
 new_events, error) = worker.snapshot()

worker.stop()                 # signals both threads to exit, joins them
```

### How it works

Two threads run inside the worker:

**Capture thread** (`_capture_loop`):
- Opens `cv2.VideoCapture(camera_index)`.
- Sets `CAP_PROP_BUFFERSIZE = 1` so OpenCV doesn't queue stale frames
  if we briefly fall behind.
- In a tight loop: read the next frame, store it under a lock as
  `self._latest_frame`, increment a monotonic `_frame_id` counter,
  update `_capture_fps`.
- This loop is essentially "as fast as the camera allows", typically
  the camera's native frame rate (e.g. 30 FPS).

**Recognition thread** (`_recognize_loop`):
- In a loop: peek at `self._latest_frame` (under the lock, then copy
  out). If it's the same frame id we processed last time, sleep a few
  ms and try again — no point recomputing.
- When a new frame is available: call
  `recognizer.process(frame)`, then optionally update the blink
  detector, then call `presence_tracker.update(...)`.
- Store the new results, the recognition FPS, and any newly emitted
  events under a separate lock.
- This loop runs **as fast as recognition allows** — could be 25 FPS
  on a fast GPU or 5 FPS on a slow CPU.

The two threads use **separate locks** (`_frame_lock` for capture
state, `_results_lock` for recognition state) so they never block
each other.

**`snapshot()`** is the only thing the UI calls. It takes both locks
briefly, reads the most recent frame and the most recent results,
and returns them in a tuple. The UI uses the frame for display and
the results for drawing bounding boxes. The two might be one or
two frames out of sync — but at typical camera rates that's
~30–60 ms, which is imperceptible to the human eye and the only
visible artifact is that bounding boxes very slightly lag the actual
face when a person is moving fast.

### Design notes

- **The monotonic `_frame_id`** lets the UI's display loop skip a
  re-render when the capture thread hasn't produced a new frame yet.
  Streamlit's `placeholder.image(...)` is not free, so skipping
  duplicates saves real work.
- **All worker state is accessed under a lock**, but the locks are
  held for microseconds at a time. There's no risk of either thread
  blocking the other for long.
- **The worker captures `config` by reference**, not by value. If the
  user changes a setting while the worker is running, the change
  takes effect on the next loop iteration.
- **Errors get surfaced through `snapshot()`**, not via raised
  exceptions on threads. If `cap.read()` starts failing, or
  `recognizer.process()` throws, the worker sets `self._error` to a
  human-readable string and the UI shows it.

---

## 4. `anti_spoof.py` — the blink-based liveness check

### Why it exists

A face-recognition attendance system has an obvious weakness: a
classmate could hold up a printed photo, or a phone screen showing
the absent student's face, and the system would happily mark them
present. This is the simplest spoof attempt, and it's worth
defending against.

The cheapest reliable defence is **eye-blink detection**. Real faces
blink involuntarily every few seconds; printed photos and most static
displays never do.

### What it does

It exposes a `BlinkDetector` class. The UI hands the detector each
new frame and each recognized face, and the detector tracks per-name
state. Once a given name has been seen with their eyes both clearly
open and clearly closed within a short window, the detector
considers that person to have "blinked" — and from then on,
`blink_detector.has_blinked(name)` returns True for the rest of the
session.

The presence tracker uses this: if `Config.require_blink_to_mark` is
on, an otherwise-confirmed student doesn't get marked attendance
until `has_blinked(name)` returns True. Until then, the UI shows a
yellow-ish "blink to confirm" label next to their bounding box.

### How it works

The classical metric is the **Eye Aspect Ratio (EAR)** of Soukupová
& Čech (2016). For a given eye, six landmarks are placed around it:

```
       p2 ────── p3
     ┌─┘          └─┐
   p1  │          │  p4
     └─┐          ┌─┘
       p6 ────── p5
```

The ratio is

> EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)

That is, the average vertical eye-opening divided by the horizontal
eye width. When the eye is open this number sits around 0.30. During
a blink it drops sharply, often below 0.20 for a few frames, then
springs back up.

For each frame and each recognized face, the detector:

1. Asks dlib's 68-point landmark predictor for the landmarks of that
   face (the indices for the two eyes are fixed by the iBUG
   convention — `36..41` for left eye, `42..47` for right eye).
2. Computes EAR for each eye and averages them.
3. Appends the value to a per-name rolling history (length ~60
   samples ≈ 2 seconds at 30 FPS).
4. **A blink is registered** when the rolling history contains *both*
   a sample below the closed-threshold (0.20) and a sample above the
   open-threshold (0.25). Order doesn't matter; the existence of
   both within the window is enough — once observed, the name is
   added to `self._blinked` and stays there for the rest of the
   session.

The detector also exposes a `reset()` method, which the live loop
calls when a new session starts.

### Design notes

- **It works for all three backends.** The blink detector doesn't
  call the recognition backend — it calls dlib's 68-point predictor
  directly. So whether the user is on dlib, torch, or arcface for
  recognition, the liveness check is the same.
- **One blink is enough, forever.** We don't keep demanding blinks
  every few minutes; that would be hostile UX. Once a face has been
  confirmed real, it stays trusted for that session.
- **A short looping video of a real face would still pass.** This is
  a known limitation. Blink detection is a sensible deterrent
  against the *casual* spoof attempt (a phone photo), not a serious
  defence. A real-world deployment would want a small CNN trained
  on real-vs-spoof face data — but that's out of scope for this
  project.

---

## 5. `smoke_test.py` — the verification script

### Why it exists

After all the dependency-juggling that's gone into this project
(dlib, CUDA, three optional backends, model file downloads), the
single most useful question to be able to answer is "is the
installation actually working?". `smoke_test.py` is the answer.

It runs in a few seconds, exercises the most-likely-to-break paths,
and produces a clear seven-line green or red report. If all seven
pass, the user can confidently `streamlit run app.py`.

### What it does

Seven independent checks, run in sequence:

| # | Check | What it verifies |
|---|---|---|
| 1 | Imports load | dlib, OpenCV, `face_lib`, the core `face_attendance` module, `anti_spoof`, and `camera_worker` all import without error. |
| 2 | Model files bundled | The four required dlib `.dat` files exist under `models/face_recognition_models/models/`. |
| 3 | Detection runs | `face_lib.face_locations()` runs on a synthetic random image without throwing. |
| 4 | Config / encoder / recognizer construct | The core data classes instantiate without raising. |
| 5 | Presence tracking works | A throwaway `PresenceTracker` is created in a temp directory, fed a few synthetic frames, and verified to emit the expected ENTER, LEAVE, and re-entry events in the correct order, writing them to CSV. |
| 6 | Optional backends sane | If `torch` or `arcface` are installed, they expose a callable five-function API; if they aren't installed, `get_backend("arcface")` raises a clear `RuntimeError` rather than silently producing nonsense. |
| 7 | Config save/load roundtrips | A few sample settings are saved to a JSON file, loaded back into a fresh `Config`, and verified to be identical. |

### How it works

The structure is intentionally tiny. There's a global `FAILURES`
list, a `check(label, fn)` helper that runs each test inside a
try/except and records the outcome, and one function per check.

Tests use Python's `tempfile.TemporaryDirectory()` to avoid leaving
files behind when they create their own datasets, configs, or
presence trackers.

The exit code is 0 if everything passed, 1 otherwise — so the smoke
test can be wired into CI or a `make test` target if the project
grows past one developer.

### Design notes

- **It doesn't use a real webcam.** Running on CI is a goal, so the
  tests construct synthetic images and don't depend on hardware.
- **It doesn't test the UI.** Streamlit testing is a whole other
  problem; the smoke test covers everything *below* the UI layer.
- **It's the place to add new checks** when something starts breaking
  in production. The five-line per-check pattern is meant to invite
  copy-paste.

---

## How a typical frame flows through the system

Putting it all together — a single frame in the live loop:

```
1. Capture thread (camera_worker.py) reads a frame from the webcam.
   ─ stores it as the new "latest frame"

2. Recognition thread (camera_worker.py) sees a new frame is ready.
   ─ copies it out under a lock

3. Recognition thread calls TrackingRecognizer.process(frame)
   in face_attendance.py.

4. TrackingRecognizer calls the active backend's face_locations(frame).
   ─ dlib OR torch OR arcface, transparently.

5. For each detected box, TrackingRecognizer checks its tracking
   memory (IoU + centroid distance). If the box matches a cached
   identity and the cache is fresh, the identity is REUSED — go to
   step 8.

6. Otherwise: TrackingRecognizer calls the backend's
   face_encodings() and match() to identify the unknown face.

7. The new identity is stored in the tracking memory.

8. The recognition thread also (optionally) calls BlinkDetector.update()
   in anti_spoof.py, which updates the per-name "has blinked" bit.

9. The recognition thread calls PresenceTracker.update() with the list
   of (name, confidence) pairs.
   ─ multi-frame confirmation rule decides whether to mark anyone
     newly confirmed
   ─ leave-by-absence rule decides whether anyone leaves
   ─ CSVs are appended on either transition

10. The display loop in app.py calls worker.snapshot() and gets back:
    the frame, the recognition results, the FPS counters, and any
    new events.

11. annotate_frame() draws the bounding boxes; Streamlit shows the
    image; the side panels update.
```

The two threads in step 1–9 mean step 10–11 runs at camera-native
rate even when steps 6–9 are slow. That's the single most important
architectural choice in the project, and the reason the live feed
stays smooth on a CPU-only laptop.

---

## Suggested reading order

If you're explaining this codebase to someone else, the most useful
order is:

1. **`face_attendance.py` §1.5 (`PresenceTracker`)** — start here.
   It's the high-level "what does the system actually do with the
   data" answer and most of the user-visible behaviour is governed
   by these few rules.

2. **`face_attendance.py` §1.4 (`TrackingRecognizer`)** — the
   performance trick (skip re-encoding stable faces). Once they
   understand this, the whole "real-time" claim makes sense.

3. **One backend** (probably `face_lib.py`, the dlib one) — to give
   them a concrete sense of what *detection* and *encoding* actually
   are. Don't go through all three.

4. **`camera_worker.py`** — explain why threading was necessary.
   They'll already have the context to see why a single-threaded
   version would stutter.

5. **`anti_spoof.py`** — the cool demo trick. Save it for last; it
   leaves a strong impression.

You can skip `smoke_test.py` in a verbal walkthrough — it's a
maintenance script, not a feature.
