# K1 Kickboxing Scoring Model — Architecture Plan

## 1. Goal

Given a K-1 kickboxing video, detect each strike and classify its outcome:

| Class | Meaning |
|-------|---------|
| `hit` | Strike lands clean on the opponent (scores) |
| `blocked` | Strike lands on guard / arms / shin (no score) |
| `miss` | Strike thrown, no contact (evaded / short) |
| `none` | No strike (background / movement) |

Later extension: strike type (punch, kick, knee) and the attacker (red/blue corner), then aggregate into a round score.

## 2. Pipeline Overview

```
Video (web) ──► Frame extraction (OpenCV) ──► Pose estimation (2 fighters)
                                                    │
                                          Tracking + normalization
                                                    │
                                   Keypoint sequences (T x 2 x K x C)
                                                    │
                       Window labeling (manual/semi-auto)  ──► Dataset
                                                    │
                                     PyTorch temporal classifier
                                                    │
                           Strike events (hit / blocked / miss) ──► Score
```

> Note: OpenCV by itself does **not** produce skeletons. We use OpenCV for video I/O, preprocessing, drawing and visualization, and a pose model for the keypoints (see 3.2).

## 3. Stages

### 3.1 Data acquisition
- Source: public K-1 / kickboxing fight videos (YouTube etc.) via `yt-dlp`.
- Prefer: fixed wide camera, full-body view, 720p+, 25–60 fps.
- Store raw videos in `data/raw/`, with a `sources.csv` (url, title, fight id, license note).
- **Legal check**: web videos are copyrighted. Keep the dataset for private/research use, store only derived data (keypoints + labels) if we ever publish, never redistribute the videos.

### 3.2 Pose extraction
- OpenCV: decode frames (`cv2.VideoCapture`), resize, draw overlays.
- Pose model options (pick one after a quick benchmark on ~5 clips):
  1. **MediaPipe Pose**: easy, but single person only → poor for two fighters in clinch.
  2. **YOLOv8/11-pose (Ultralytics)** — multi-person, fast, built-in tracking. **Recommended start.**
  3. **RTMPose / MMPose** — most accurate, heavier setup.
- Output per frame: 2 fighters × 17 COCO keypoints × (x, y, confidence).
- Tracker (ByteTrack via Ultralytics) keeps fighter ID consistent; referee gets filtered out (assume the 2 largest/most central tracks, or assign by corner color).
- Save as `data/keypoints/<video_id>.npz`.
- Debug video with skeleton overlay in `data/debug/` to eyeball quality.

### 3.3 Preprocessing / features
- Normalize per fighter: center on hip midpoint, scale by torso length (camera-distance invariant).
- Smooth with a Savitzky–Golay or One-Euro filter; interpolate low-confidence joints.
- Features per frame:
  - Normalized coordinates of both fighters.
  - Velocities / accelerations of wrists, ankles, knees.
  - **Interaction features**: distance from attacker's wrist/ankle to opponent's head/torso/guard keypoints, relative position of fighters, limb extension angles.
- Fixed windows: e.g. 32–48 frames (~1–1.5 s) around a strike, with stride for sliding-window inference.

### 3.4 Labeling
Two-step to keep it cheap:
1. **Candidate strike detection (automatic)**: peaks in wrist/ankle velocity toward the opponent → proposes timestamps.
2. **Human label** of each candidate: `hit / blocked / miss / none` (+ optional strike type, attacker corner).
- Tool: small labeling UI (OpenCV window or Label Studio / simple Streamlit app) that plays the 1.5 s clip with skeleton overlay and takes a hotkey (1–4). Output: `data/labels/labels.csv` (video_id, start_frame, end_frame, attacker, class, strike_type).
- Target size for a first usable model: ~2–3k labeled windows, ≥ 20 different fights. Expect class imbalance (`miss`/`none` dominate) → handle with weighted loss / sampling.
- Splitting: **by fight/video**, never by random window (avoids leakage).
- Label quality: double-label ~10% to measure agreement. Hit vs blocked is subjective, so write clear rules in `docs/labeling_guide.md`.

### 3.5 Model (PyTorch)
Input: `(B, T, F)` keypoint feature sequence. Start simple, compare:

| Model | Idea |
|-------|------|
| **Baseline**: MLP / XGBoost on handcrafted window stats | sanity-check the data |
| **A. 1D-CNN + BiGRU** | 4 layers: Conv1D → Conv1D → BiGRU → Linear head |
| **B. Temporal Transformer** | small encoder (2–4 layers) with positional encoding |
| **C. ST-GCN** | graph conv over skeleton joints (both fighters in one graph) — best structure-aware option |

Your "at least 4 layers" idea fits model A/B. Don't fix the depth upfront; tune it.

Heads: 4-class softmax (`hit/blocked/miss/none`). Optional multi-task heads: strike type, attacker.
Loss: cross-entropy with class weights (or focal loss). Augmentation: horizontal flip (swap left/right joints), small jitter/noise, random scaling, temporal speed ±15%, random joint dropout (simulates pose errors).

### 3.6 Evaluation
- Metrics: per-class precision/recall/F1, macro-F1 (not accuracy), confusion matrix (watch `hit` vs `blocked`).
- Event-level: strike detection recall/precision with ±N-frame tolerance.
- Hold-out fights as the test set.
- Known limit: 2D skeletons can't tell actual contact depth from a single camera. If `hit` vs `blocked` plateaus, add a visual branch (cropped RGB patch around the impact point / opponent's reaction) as a second input stream.

### 3.7 Inference & scoring
- `infer.py video.mp4` → pose → sliding window → classifier → NMS over events → JSON of events + annotated output video.
- Simple scoring rule layer (configurable): e.g. clean hit = 1 pt, kick = 2 pt, then per-round totals.

## 4. Project Structure

```
K1_SCRORE/
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml / requirements.txt
├── configs/            # yaml: pose model, window size, training hyperparams
├── data/
│   ├── raw/            # videos (gitignored)
│   ├── keypoints/      # .npz per video
│   ├── labels/         # labels.csv
│   └── debug/          # overlay videos
├── docs/labeling_guide.md
├── src/k1score/
│   ├── download.py     # yt-dlp wrapper
│   ├── pose.py         # pose + tracking -> keypoints
│   ├── features.py     # normalize, smooth, interaction features
│   ├── candidates.py   # auto strike proposals
│   ├── label_tool.py   # labeling UI
│   ├── dataset.py      # torch Dataset / splits / augmentation
│   ├── models/         # cnn_gru.py, transformer.py, stgcn.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── notebooks/          # exploration
└── tests/
```

## 5. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| 1 | Env + download 5–10 videos | videos in `data/raw/` |
| 2 | Pose extraction + overlay | skeleton overlay looks right on 5 clips; fighter IDs stable |
| 3 | Features + candidate detector | candidate recall ≥ ~90% on a hand-checked clip |
| 4 | Labeling tool + first ~500 labels | labels.csv exists, guide written |
| 5 | Baseline + model A training | beats baseline on val fights |
| 6 | Scale labels (2–3k), try B / C | macro-F1 tracked per experiment |
| 7 | Inference pipeline + scoring | end-to-end video → annotated output + score |
| 8 | (If needed) RGB branch for hit/blocked | improvement on confusion matrix |

## 6. Main Risks

- **Pose quality** in clinches, occlusion, motion blur, camera cuts/replays → filter by confidence, detect shot changes (histogram diff) and break sequences there.
- **Camera angle variety** across web videos → normalization + flip augmentation; prefer main wide-angle footage.
- **Label subjectivity** (hit vs blocked) and class imbalance → labeling guide, agreement check, weighted loss.
- **Small dataset** → keep model small, heavy augmentation, validate by fight.
- **Copyright** of source videos (see 3.1).

## 7. Stack

Python 3.11+, PyTorch, OpenCV, Ultralytics (YOLO-pose + ByteTrack), yt-dlp, NumPy/Pandas, scikit-learn, (optional) PyTorch Lightning, Weights & Biases or TensorBoard.
