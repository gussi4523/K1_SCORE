# Tool 01 — `ingest`: YouTube video → dataset-ready data

## Purpose
Turn a YouTube URL into a **versioned, model-agnostic, resumable** set of files (video, metadata, shots, pose tracks) that every later tool (labeling, training, inference) can read without re-processing.

Principle: **store facts, not features.** We store raw keypoints + metadata. Normalization, velocities, windows, and augmentation are computed later in the model code, so we can change the model without re-ingesting.

## Repo layout (monorepo: one shared core, many tools)

```
K1_SCRORE/
├── ARCHITECTURE.md
├── docs/tools/01_ingest.md
├── pyproject.toml              # one package, `k1score`, CLI entry point `k1`
├── configs/
│   └── ingest.yaml             # pose model, fps, thresholds, paths
├── data/                       # gitignored
│   ├── raw/<video_id>/         # video.mp4, source.json (yt-dlp info)
│   ├── interim/<video_id>/     # shots.json, debug overlay mp4
│   ├── processed/<video_id>/   # poses.npz, meta.json   <-- dataset contract
│   ├── labels/                 # written by tool 02 (labeling)
│   └── manifest.parquet        # one row per video + status of each stage
├── src/k1score/
│   ├── core/                   # shared by ALL tools
│   │   ├── schema.py           # dataset format + version, COCO17 joint names
│   │   ├── paths.py            # the only place that knows folder layout
│   │   ├── manifest.py         # read/write manifest, stage status
│   │   ├── config.py           # load yaml -> dataclasses
│   │   └── io.py               # load_poses(video_id), load_meta(video_id)
│   ├── cli.py                  # `k1 <tool> <command>`
│   └── tools/
│       ├── ingest/             # TOOL 01
│       │   ├── download.py     # yt-dlp
│       │   ├── probe.py        # fps, resolution, duration (OpenCV)
│       │   ├── shots.py        # camera-cut detection
│       │   ├── pose.py         # YOLO-pose + ByteTrack
│       │   ├── select.py       # pick 2 fighters, drop referee/crowd
│       │   ├── quality.py      # per-segment quality score
│       │   ├── export.py       # write poses.npz + meta.json
│       │   ├── debug.py        # skeleton overlay video
│       │   └── pipeline.py     # runs stages in order, skips finished ones
│       ├── label/              # TOOL 02 (later)
│       ├── train/              # TOOL 03 (later)
│       └── infer/              # TOOL 04 (later)
└── tests/
```

Rules: tools import from `core/`, never from each other. Only `core/paths.py` knows where files live. Only `core/io.py` reads the dataset, so labeling/training use the same loader.

## Pipeline stages

Each stage reads the previous stage's files, writes its own, and records `done/failed` in the manifest. Re-running skips finished stages (`--force` to redo one).

| # | Stage | Input → Output | Notes |
|---|-------|----------------|-------|
| 1 | `download` | URL → `raw/<id>/video.mp4`, `source.json` | `yt-dlp`, ≤1080p, mp4. `video_id` = YouTube id (stable, idempotent). Supports a URL list / playlist. |
| 2 | `probe` | video → fps, w, h, n_frames, duration | OpenCV. Reject if too low-res or fps < 24. |
| 3 | `shots` | video → `interim/<id>/shots.json` | Cut detection (HSV histogram diff, or `PySceneDetect`). Classify each shot as `ring_wide` / `other` (close-up, replay, crowd) by "2 large persons visible, wide framing". Only `ring_wide` shots go forward. |
| 4 | `pose` | video + shots → raw tracks | YOLO-pose + ByteTrack, run only on good shots. Keep all tracks with confidences. |
| 5 | `select` | tracks → 2 fighter tracks per shot | Heuristics: largest/most central tracks, drop referee (stays between/behind, less motion). Assign `fighter_0/1` (stable inside a shot; corner color mapping is a later, optional step). |
| 6 | `quality` | fighters → per-frame quality | Fraction of joints with conf > thr, track continuity, occlusion flag. Produces `valid` mask. |
| 7 | `export` | all → `processed/<id>/poses.npz` + `meta.json` | The dataset contract below. |
| 8 | `debug` (optional) | → overlay mp4 | Skeleton + IDs drawn, so a human can verify quickly. |

## Dataset contract (what the future model relies on)

`processed/<video_id>/poses.npz` (all arrays indexed by **original video frame number**, so labels can use frame ranges directly):

| Array | Shape | Meaning |
|-------|-------|---------|
| `keypoints` | `(N, 2, 17, 3)` | frame × fighter × COCO joint × (x, y, conf). x,y in **pixels**; NaN where fighter absent. |
| `valid` | `(N, 2)` bool | fighter pose usable in this frame |
| `shot_id` | `(N,)` int | shot index, `-1` if frame skipped (not a ring shot) |
| `frame_idx` | `(N,)` int | original frame number (supports fps subsampling later) |

`processed/<video_id>/meta.json`:
```json
{
  "schema_version": "1.0.0",
  "video_id": "abc123",
  "source_url": "...", "title": "...",
  "fps": 30.0, "width": 1920, "height": 1080, "n_frames": 54000,
  "joint_names": ["nose", "...COCO17..."],
  "pose_model": {"name": "yolo11m-pose", "version": "..."},
  "tracker": "bytetrack",
  "shots": [{"id": 0, "start": 0, "end": 812, "type": "ring_wide"}],
  "quality": {"valid_ratio": 0.83},
  "ingest_config_hash": "...",
  "created_at": "..."
}
```

Why this is future-proof:
- Pixel coordinates + frame size + fps → any normalization can be derived later.
- `schema_version` + `pose_model` recorded → we can migrate, or re-extract with a better pose model without touching labels (labels reference only `video_id` + frame ranges).
- Original video kept in `raw/` → the RGB-crop branch, labeling UI, and new pose models all remain possible.
- Labels live in a separate file, never inside `poses.npz`.

`manifest.parquet` columns: `video_id, url, status_download, status_shots, status_pose, status_export, schema_version, n_frames, valid_ratio, error`.

## CLI

```
k1 ingest add <url|file.txt>        # download + register
k1 ingest run [--video ID|--all] [--stage pose] [--force]
k1 ingest status                    # table from manifest
k1 ingest debug <video_id>          # write overlay video
k1 ingest validate                  # check every processed/ against schema
```

## Build order (each step testable on its own)

1. Skeleton: `pyproject.toml`, `core/` (paths, schema, manifest, config), CLI stub.
2. `download` + `probe` — test with 1 short video.
3. `pose` (no tracking yet) + `debug` overlay — eyeball quality.
4. Add ByteTrack + `select` (2 fighters, referee removal).
5. `shots` + ring-shot filter.
6. `quality` + `export` + `core/io.py` loader + `validate`.
7. Tests: schema round-trip, a tiny 5-s fixture clip, idempotent re-run.

## Open decisions
- Pose model size (`yolo11n` fast vs `yolo11m/l` accurate): benchmark in step 3.
- Process every frame, or every 2nd? Start with **every frame** (strikes are fast); subsample later via `frame_idx`.
- Corner color (red/blue) assignment: skip in v1.

---

## Implementation status (v0.1, implemented)

All 7 build steps are done; `pytest` passes (14 tests) and the pipeline ran end-to-end on a real K-1 broadcast clip
(YouTube download -> shots -> YOLO11s-pose + ByteTrack -> select -> export -> validate -> debug overlay).

Differences from the design above:
- `quality` is computed inside `export` (not a separate stage); `debug` is a command, not a stage.
  Stages in the manifest: `download, probe, shots, pose, select, export`.
- Pose weights live in `models/`. Raw intermediate detections are kept in `interim/<id>/tracks.npz`, so `select`/`export`
  can be re-tuned without re-running the pose model.
- `select` also drops **static overlays** (broadcast portrait graphics were selected as "fighters" in the first real run)
  and people smaller than `min_fighter_h`.
- Ring-shot detection uses the pose model's person boxes on 5 sampled frames per shot.
- Local video files are supported (`k1 ingest add file.mp4`), useful for cutting test clips with ffmpeg.

## Known limits / next steps
1. **Referee as fighter**: when the pose model misses a fighter (hunched guard, ropes/corner post occlusion),
   the referee can fill that slot. First real clip: valid_ratio 0.77, but part of that is referee frames.
   Ideas: bigger model (`yolo11m-pose`, `imgsz: 1280`, lower `conf`), pair-selection by co-presence/closeness,
   referee detection by clothing/position.
2. **Late seed**: if a fighter's track id fragments evenly, the part before the seed track starts is empty.
3. Slots are left/right by position at shot start, not red/blue corner (needs a later corner-assignment step).
4. Shot classification thresholds (`configs/ingest.yaml` -> `shots`) are untuned beyond one video; check `shots.json`.
5. Always eyeball a new video with `k1 ingest debug` before labeling it.

---

## Update v0.2 - identities (fighters and referee told apart, stable for the whole video)

Problem found on real videos: the referee filled a fighter slot when a fighter was missed, and fighter 0/1
swapped between shots (slots came from left/right position and from tracker ids).
ByteTrack ids are not trustworthy: one track can follow fighter A, then the referee, then fighter B in a clinch.

New stage `identify` (between `pose` and `select`), decided **per detection from appearance**:
1. measure torso (skin share, darkness) and shorts (colour) from the pose keypoints, smooth along the track;
2. learn roles once per video: referee = shirted cluster (little bare skin), fighters = the two shorts-colour clusters;
3. per frame, assign detections to roles with the Hungarian algorithm (each role at most once per frame; fighters need a
   bare torso, the referee a shirt; detections that fit no role stay `other`: crowd, photographers, overlays).

Order is deterministic: `fighter_0` = the fighter with less saturated shorts (e.g. black), so re-runs agree.
To match red/blue corner, check `identity_sheet.png` and run `k1 ingest identify <id> --swap`.

Schema 1.1.0 (backward compatible, 1.0 files still load): new optional array `referee` (N,17,3) and
`meta["identities"]` (shorts colour per role, `swapped` flag). Stages: `download, probe, shots, pose, identify, select, export`.

Files: `interim/<id>/identities.json|npz` (role of every detection), `identity_crops/`, `identity_sheet.png` (shown at the top of
`preview/index.html`).

Result on the 10-minute Muay Thai test video: both fighters usable at once in 72% of ring frames (was 30%), referee tracked
in 11.3k frames, fighters never swap.

Known limits of v0.2
- Needs visibly different shorts and a shirted referee. Same-colour shorts -> clusters merge; the identity sheet shows it.
- Stray detections can still take the referee role (e.g. a photographer in black when the referee is not detected).
- Post-fight celebrations with many people in the ring are not handled (extra people may become fighters).
- A fighter hidden behind the other is simply absent (NaN) in those frames.
