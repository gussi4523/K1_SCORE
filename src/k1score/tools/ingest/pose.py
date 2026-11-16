"""Stage 4: pose estimation + multi-person tracking on ring shots only.

Output (interim/<id>/tracks.npz) is a flat table of detections:
  frame (M,), shot (M,), track_id (M,), box (M,4 xyxy px), kpts (M,17,3)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from k1score.core.schema import N_JOINTS, SHOT_RING_WIDE


def load_model(pcfg: dict, models_dir: Path | None = None):
    """Weights live in <project>/models/ (downloaded on first use), not in the current directory."""
    from ultralytics import YOLO

    name = Path(pcfg["model"])
    if models_dir is not None and not name.is_absolute() and not name.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        name = models_dir / name
    return YOLO(str(name))


def _device(pcfg: dict):
    if pcfg.get("device") is not None:
        return pcfg["device"]
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def make_person_detector(model, pcfg: dict):
    """Detector used by the shot classifier (reuses the pose model's person boxes)."""
    dev = _device(pcfg)

    def detect(frames: list[np.ndarray]) -> list[np.ndarray]:
        res = model.predict(frames, conf=pcfg["conf"], imgsz=pcfg["imgsz"], device=dev, verbose=False)
        return [r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4)) for r in res]

    return detect


def _reset_trackers(model) -> None:
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", None) or []:
        t.reset()


def run_pose(video: Path, shots: list[dict], n_frames: int, pcfg: dict, model=None, log=lambda s: None) -> dict[str, np.ndarray]:
    model = model or load_model(pcfg)
    dev = _device(pcfg)
    stride = max(1, int(pcfg["frame_stride"]))

    ring = np.full(n_frames, -1, dtype=np.int32)  # shot id of ring frames, else -1
    starts = set()
    for s in shots:
        if s["type"] == SHOT_RING_WIDE:
            ring[s["start"]:s["end"]] = s["id"]
            starts.add(s["start"])
    todo = int((ring >= 0).sum())

    frames_o, shot_o, id_o, box_o, kp_o = [], [], [], [], []
    cap = cv2.VideoCapture(str(video))
    done = 0
    for fn in range(n_frames):
        if ring[fn] < 0 or (fn - _shot_start(shots, ring[fn])) % stride:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if fn in starts:
            _reset_trackers(model)
        r = model.track(frame, persist=True, tracker=pcfg["tracker"], conf=pcfg["conf"],
                        imgsz=pcfg["imgsz"], device=dev, verbose=False)[0]
        done += 1
        if done % 300 == 0:
            log(f"  pose {done}/{todo} frames")
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue
        ids = r.boxes.id.int().cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()
        for i in range(len(ids)):
            frames_o.append(fn); shot_o.append(ring[fn]); id_o.append(ids[i])
            box_o.append(boxes[i]); kp_o.append(kpts[i])
    cap.release()

    return {
        "frame": np.asarray(frames_o, dtype=np.int32),
        "shot": np.asarray(shot_o, dtype=np.int32),
        "track_id": np.asarray(id_o, dtype=np.int32),
        "box": np.asarray(box_o, dtype=np.float32).reshape(-1, 4),
        "kpts": np.asarray(kp_o, dtype=np.float32).reshape(-1, N_JOINTS, 3),
    }


def _shot_start(shots: list[dict], shot_id: int) -> int:
    return shots[shot_id]["start"]
