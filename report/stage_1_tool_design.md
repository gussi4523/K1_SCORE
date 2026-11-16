# Stage 1 - Tool 01 design (`ingest`)

## Request
"Build the first tool; we will have many tools, so make a clear folder structure. The first tool turns YouTube videos into datasets that will be fully usable by our model in the future."

## Principle
**Store facts, not features.** The dataset keeps raw keypoints and metadata. Normalization, velocities, windowing and augmentation are computed later in model code, so the model can change without re-processing videos.

## Folder structure (one package, many tools)

```
K1_SCRORE/
|-- ARCHITECTURE.md, README.md, pyproject.toml
|-- configs/ingest.yaml          settings per tool
|-- data/                        (git-ignored)
|   |-- raw/<id>/                video + source.json
|   |-- interim/<id>/            intermediate files, pictures
|   |-- processed/<id>/          THE DATASET: poses.npz + meta.json
|   |-- labels/                  written by tool 02 later
|   |-- manifest.parquet         one row per video, status of every stage
|-- models/                      downloaded pose weights
|-- src/k1score/
|   |-- core/                    shared by ALL tools: schema, paths, config, manifest, io
|   |-- cli.py                   `k1 <tool> <command>`
|   |-- tools/
|       |-- ingest/              TOOL 01 (built)
|       |-- label/  train/  infer/   later tools (empty placeholders)
|-- tests/   docs/   report/
```

Rules: tools import only from `core/`, never from each other; only `core/paths.py` knows where files live; only `core/io.py` reads the dataset (so labeling, training and inference all load data the same way).

## Pipeline stages (each resumable; status stored in the manifest)

| Stage | Input -> output |
|---|---|
| download | URL -> `raw/<id>/video.mp4` (video id = YouTube id, so adding twice is safe) |
| probe | video -> fps, size (rejects low-res / low-fps) |
| shots | video -> camera cuts, keep only wide ring shots (drop replays, close-ups) |
| pose | video + shots -> pose + tracks for every person |
| select | tracks -> the two fighters |
| export | -> `processed/<id>/poses.npz` + `meta.json` |

(In stage 5 a new `identify` stage was inserted before `select`.)

## Dataset contract (`poses.npz`, indexed by original frame number)

| Array | Shape | Meaning |
|---|---|---|
| `keypoints` | (N, 2, 17, 3) | frame x fighter x COCO joint x (x, y, confidence), pixels, NaN = absent |
| `valid` | (N, 2) bool | fighter pose usable in this frame |
| `shot_id` | (N,) | shot index, -1 = frame skipped (not a ring shot) |
| `frame_idx` | (N,) | original frame numbers |

`meta.json` stores fps, size, shots, pose model name, quality stats, schema version, config hash. Labels live in a **separate** file that refers to `video_id` + frame ranges, so re-extracting poses never breaks labels.

## Why this is future-proof
- pixel coordinates + frame size + fps allow any later normalization
- `schema_version` and `pose_model` are recorded, so data can be migrated or re-extracted with a better pose model
- original video is kept for the labeling UI and a future RGB branch

Design document: `../docs/tools/01_ingest.md`.
