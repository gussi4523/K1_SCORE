# Stage 2 - Implementation of tool 01 (v0.1)

## Environment
- Linux, Python 3.13, `uv` virtual environment in `.venv/`
- GPU: NVIDIA GTX 1650 (4 GB) - used automatically for pose
- Dependencies (`pyproject.toml`): numpy, pandas, pyarrow, pyyaml, opencv-python, yt-dlp, ultralytics (brings PyTorch), lap (tracker); `scipy` added later in stage 5
- Pose weights `yolo11s-pose.pt` (20 MB) download on first use into `models/`

## What was built

| File | Role |
|---|---|
| `core/schema.py` | dataset contract, COCO-17 joint names, validation |
| `core/paths.py`, `core/config.py`, `core/manifest.py`, `core/io.py` | shared plumbing: file locations, yaml config with defaults, per-video stage status (parquet), dataset loader/saver |
| `tools/ingest/download.py` | yt-dlp download (video only, H.264 preferred, up to 1080p); also accepts local files, playlists, `.txt` lists |
| `tools/ingest/probe.py` | fps / resolution check |
| `tools/ingest/shots.py` | camera-cut detection (HSV histogram distance), classification of each shot as ring shot or not (>=2 fighter-sized people, not a crowd) |
| `tools/ingest/pose.py` | YOLO-pose + ByteTrack on ring shots only; writes a flat detection table |
| `tools/ingest/select.py` | pick the two fighters (heuristic, replaced in stage 5) |
| `tools/ingest/quality.py` | per-frame usability mask (enough confidently visible joints) |
| `tools/ingest/export.py` | assemble and validate the dataset files |
| `tools/ingest/debug.py` | skeleton overlay video |
| `tools/ingest/pipeline.py` | runs stages in order, skips finished ones, invalidates later stages when one is redone |
| `cli.py` | `k1 ingest add / run / status / debug / validate` |

## Behaviours worth knowing
- **Resumable:** every stage records done / failed in `manifest.parquet`; a second run skips finished stages; `--stage X` redoes one stage and marks the later ones stale.
- **Raw detections are kept** (`interim/<id>/tracks.npz`), so selection and export can be re-tuned in seconds without re-running the pose model.
- **Validation:** `k1 ingest validate` re-checks every dataset against the schema; saving an invalid dataset is refused.

## Tests
13 automated tests at this point (no GPU, no network): schema round trip and validation, manifest, config override, cut detection on a synthetic video, ring-shot classification, fighter selection, quality mask, and an end-to-end run with a fake pose model.

Result: 13 passed.
