"""Dataset contract shared by every tool. Bump SCHEMA_VERSION on any breaking change."""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "1.1.0"   # 1.1: added optional `referee` array + meta["identities"]

N_FIGHTERS = 2
N_JOINTS = 17
COCO17_JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Stages of the ingest pipeline, in execution order.
INGEST_STAGES = ["download", "probe", "shots", "pose", "identify", "select", "export"]

# Roles assigned by the `identify` stage (appearance-based, stable for the whole video).
ROLE_FIGHTER_0, ROLE_FIGHTER_1, ROLE_REFEREE, ROLE_OTHER = "fighter_0", "fighter_1", "referee", "other"
FIGHTER_ROLES = [ROLE_FIGHTER_0, ROLE_FIGHTER_1]

# Shot types written by the `shots` stage.
SHOT_RING_WIDE = "ring_wide"
SHOT_OTHER = "other"

POSES_ARRAYS = {
    # name: (dtype kind, trailing shape after N)
    "keypoints": ("f", (N_FIGHTERS, N_JOINTS, 3)),
    "valid": ("b", (N_FIGHTERS,)),
    "shot_id": ("i", ()),
    "frame_idx": ("i", ()),
}

OPTIONAL_ARRAYS = {"referee": ("f", (N_JOINTS, 3))}   # added in schema 1.1

REQUIRED_META_KEYS = [
    "schema_version", "video_id", "fps", "width", "height", "n_frames",
    "joint_names", "pose_model", "tracker", "shots", "quality",
]


def validate_poses(arrays: dict[str, np.ndarray], meta: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors: list[str] = []
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            errors.append(f"meta.json missing key '{key}'")
    if errors:
        return errors

    if meta["schema_version"].split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        errors.append(f"schema major mismatch: file {meta['schema_version']} vs code {SCHEMA_VERSION}")
    if meta["joint_names"] != COCO17_JOINTS:
        errors.append("joint_names differ from COCO17")

    n = meta["n_frames"]
    for name, (kind, tail) in POSES_ARRAYS.items():
        if name not in arrays:
            errors.append(f"poses.npz missing array '{name}'")
            continue
        arr = arrays[name]
        if arr.shape != (n, *tail):
            errors.append(f"{name}: shape {arr.shape}, expected {(n, *tail)}")
        if arr.dtype.kind != kind:
            errors.append(f"{name}: dtype {arr.dtype}, expected kind '{kind}'")
    for name, (kind, tail) in OPTIONAL_ARRAYS.items():
        if name in arrays and (arrays[name].shape != (n, *tail) or arrays[name].dtype.kind != kind):
            errors.append(f"{name}: shape {arrays[name].shape} / dtype {arrays[name].dtype} invalid")
    if errors:
        return errors

    kp, valid, shot_id = arrays["keypoints"], arrays["valid"], arrays["shot_id"]
    if np.any(valid & np.isnan(kp[..., 0]).any(axis=-1)):
        errors.append("valid=True on a fighter with NaN keypoints")
    n_shots = len(meta["shots"])
    if shot_id.size and (shot_id.max() >= n_shots or shot_id.min() < -1):
        errors.append("shot_id out of range of meta['shots']")
    if not np.array_equal(arrays["frame_idx"], np.arange(n)):
        errors.append("frame_idx must equal arange(n_frames) in schema 1.x")
    return errors
