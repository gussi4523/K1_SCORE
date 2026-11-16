# K1 Score

Tools for building a K-1 kickboxing scoring model. Plan: `ARCHITECTURE.md`. Tool designs: `docs/tools/`.

## Setup
```bash
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest            # 20 tests, no GPU / network needed
```
Pose weights (`yolo11s-pose.pt`) download automatically into `models/` on first use.

## Tool 01 — `ingest` (video -> pose dataset)
```bash
k1 ingest add "https://www.youtube.com/watch?v=<id>" --run   # download + full pipeline
k1 ingest add urls.txt                                       # many URLs / playlists, one per line
k1 ingest add path/to/clip.mp4 --run                         # local file
k1 ingest status                                             # per-stage status of every video
k1 ingest run --all                                          # resume anything unfinished
k1 ingest run --video <id> --stage select                    # redo one stage (later stages invalidated)
k1 ingest debug <id> --start 600 --seconds 10                # skeleton overlay -> data/interim/<id>/debug.mp4
k1 ingest preview <id>                                       # pictures + index.html (who-is-who sheet, frames, timeline)
k1 ingest view <id>                                          # interactive frame-by-frame viewer
k1 ingest identify <id> [--swap]                             # who-is-who sheet path; --swap exchanges fighter 0 and 1
k1 ingest validate                                           # check datasets against the schema
```
Fighters and referee are told apart by appearance (shorts colour, shirt): fighter 0 / fighter 1 are the same people in every
shot, and the referee is stored separately.

Output per video: `data/processed/<id>/poses.npz` + `meta.json`. Load it from any tool with
`k1score.core.io.load_poses(video_id)`. Config: `configs/ingest.yaml`.

Only download videos you are allowed to use; keep raw videos private.
