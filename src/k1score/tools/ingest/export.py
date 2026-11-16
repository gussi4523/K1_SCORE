"""Stage 7: assemble the dataset contract (processed/<id>/poses.npz + meta.json)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from k1score.core.config import config_hash
from k1score.core.io import save_poses
from k1score.core.paths import Paths
from k1score.core.schema import COCO17_JOINTS, SCHEMA_VERSION

from .quality import compute_valid, valid_ratio


def export_dataset(video_id: str, paths: Paths, cfg: dict, source: dict, probe: dict, shots_info: dict,
                   kpts: np.ndarray, referee: np.ndarray, shot_id: np.ndarray, pose_model_name: str,
                   identities: dict) -> dict:
    n = shots_info["n_frames"]
    valid = compute_valid(kpts, cfg["quality"])
    valid &= (shot_id >= 0)[:, None]
    ratio = valid_ratio(valid, shot_id)

    arrays = {
        "keypoints": kpts.astype(np.float32),
        "referee": referee.astype(np.float32),
        "valid": valid,
        "shot_id": shot_id.astype(np.int32),
        "frame_idx": np.arange(n, dtype=np.int32),
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "source_url": source.get("url"),
        "title": source.get("title"),
        "fps": probe["fps"], "width": probe["width"], "height": probe["height"], "n_frames": n,
        "joint_names": COCO17_JOINTS,
        "pose_model": {"name": pose_model_name},
        "tracker": cfg["pose"]["tracker"],
        "frame_stride": cfg["pose"]["frame_stride"],
        "shots": shots_info["shots"],
        "quality": {
            "valid_ratio": round(ratio, 4),
            "ring_frames": int((shot_id >= 0).sum()),
            "valid_frames_per_fighter": [int(valid[:, 0].sum()), int(valid[:, 1].sum())],
        },
        "identities": {"roles": identities["roles"], "swapped": identities["swapped"]},
        "referee_frames": int((~np.isnan(referee[..., 0]).all(axis=-1)).sum()),
        "ingest_config_hash": config_hash(cfg),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_poses(video_id, arrays, meta, paths)
    return meta
