# Face Attendance System

Real-time, multi-backend face-recognition attendance system with a
Streamlit GUI. Detects, identifies, and logs students entering and
leaving a classroom from a live webcam, writes daily CSV attendance
reports plus a separate ENTER/LEAVE event log, and supports anti-spoof
liveness checks.

> Built for AI385 — Project Implementation. See [REPORT.md](REPORT.md)
> for the full project write-up.

## Features

- Real-time face detection, identification, and attendance logging.
- Three interchangeable recognition backends:
  - **dlib** — HOG + 128-D ResNet, CPU-friendly.
  - **torch** — MTCNN + 512-D FaceNet (`facenet-pytorch`), CUDA-accelerated.
  - **arcface** — SCRFD + 512-D ArcFace (`insightface`), CUDA-accelerated.
- ENTER / LEAVE event log with configurable leave-timeout.
- Multi-frame confirmation guards against single-frame false positives.
- Late vs on-time classification with a configurable class start time
  and grace period.
- Anti-spoofing via eye-blink (EAR) liveness check.
- IoU-based face tracking that skips re-encoding stable faces.
- Threaded capture/recognition pipeline keeps the UI smooth even when
  recognition is slow.
- Roster management, embedding augmentation, persistent settings, and
  Streamlit-based reports.

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Download the dlib model files

The `.dat` model files (~127 MB) aren't bundled with this repo.
Clone Adam Geitgey's `face_recognition_models` and copy them in:

```bash
git clone https://github.com/ageitgey/face_recognition_models /tmp/face_recognition_models
mv /tmp/face_recognition_models/face_recognition_models/models/*.dat models/face_recognition_models/models/
```

On Windows (PowerShell):

```powershell
git clone https://github.com/ageitgey/face_recognition_models $env:TEMP\face_recognition_models
Move-Item $env:TEMP\face_recognition_models\face_recognition_models\models\*.dat models\face_recognition_models\models\
```

You should now have four `.dat` files under
`models/face_recognition_models/models/`.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install PyTorch with CUDA wheels first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For the optional ArcFace backend:

```bash
pip install insightface onnxruntime-gpu     # or onnxruntime for CPU
```

### 4. Verify the install

```bash
python smoke_test.py
```

You should see seven green check-marks.

## Running

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Quick start

1. **Dataset** tab → upload 3–10 photos of each student → click
   *Save images*, then *Rebuild encodings*.
2. **Live** tab → press **▶ Start**. The annotated camera feed appears
   with green boxes around recognized students.
3. **Attendance** and **Event Logs** tabs hold the daily CSVs.
4. **Reports** tab summarizes attendance across sessions, including
   on-time vs late breakdown.

## Project layout

```
.
├── app.py                  # Streamlit GUI entry point
├── face_attendance.py      # Core: Config, FaceEncoder, TrackingRecognizer, PresenceTracker
├── face_lib.py             # dlib backend
├── torch_face_lib.py       # torch backend (MTCNN + FaceNet)
├── insightface_lib.py      # arcface backend (SCRFD + ArcFace)
├── anti_spoof.py           # blink-based liveness check
├── camera_worker.py        # background thread for capture + recognition
├── smoke_test.py           # 7-check verification suite
├── requirements.txt
├── run.bat
├── REPORT.md               # full project write-up
├── models/                 # dlib model files (download separately, see Setup)
└── data/                   # runtime artifacts (gitignored)
```

## Credits

- Pre-trained dlib models from [face_recognition_models](https://github.com/ageitgey/face_recognition_models)
  by Adam Geitgey (MIT).
- FaceNet weights from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MIT).
- ArcFace weights from [insightface](https://github.com/deepinsight/insightface) (MIT).

## License

This project is for educational use as part of AI385. Pre-trained
models retain their respective licenses (linked above).
