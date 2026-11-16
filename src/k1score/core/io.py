"""Reading/writing the processed dataset. Every tool loads data through here."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import Paths
from .schema import validate_poses


@dataclass
class Poses:
    keypoints: np.ndarray   # (N, 2, 17, 3) float32: x, y (pixels), conf. NaN = fighter absent
    valid: np.ndarray       # (N, 2) bool
    shot_id: np.ndarray     # (N,) int32, -1 = frame not in a usable ring shot
    frame_idx: np.ndarray   # (N,) int32, original video frame numbers
    meta: dict
    referee: np.ndarray | None = None   # (N, 17, 3) float32, NaN = referee not detected (schema >= 1.1)

    @property
    def fps(self) -> float:
        return self.meta["fps"]

    @property
    def size(self) -> tuple[int, int]:
        return self.meta["width"], self.meta["height"]


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_meta(video_id: str, paths: Paths | None = None) -> dict:
    return read_json((paths or Paths()).meta_json(video_id))


def load_poses(video_id: str, paths: Paths | None = None, check: bool = True) -> Poses:
    paths = paths or Paths()
    meta = load_meta(video_id, paths)
    with np.load(paths.poses_npz(video_id)) as z:
        arrays = {k: z[k] for k in z.files}
    if check:
        errors = validate_poses(arrays, meta)
        if errors:
            raise ValueError(f"{video_id}: invalid dataset: " + "; ".join(errors))
    return Poses(meta=meta, referee=arrays.get("referee"),
                 **{k: arrays[k] for k in ("keypoints", "valid", "shot_id", "frame_idx")})


def save_poses(video_id: str, arrays: dict[str, np.ndarray], meta: dict, paths: Paths | None = None) -> None:
    paths = paths or Paths()
    errors = validate_poses(arrays, meta)
    if errors:
        raise ValueError("refusing to write invalid dataset: " + "; ".join(errors))
    paths.processed(video_id).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.poses_npz(video_id), **arrays)
    write_json(paths.meta_json(video_id), meta)
