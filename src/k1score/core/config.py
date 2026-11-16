"""YAML config -> plain nested dict with defaults (no heavy dependencies)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_INGEST: dict = {
    "download": {
        "max_height": 1080,
        "cookies_from_browser": None,  # e.g. "firefox" if YouTube asks for sign-in
    },
    "probe": {"min_height": 480, "min_fps": 24.0},
    "shots": {
        "cut_threshold": 0.35,      # Bhattacharyya distance between HSV histograms of consecutive frames
        "min_shot_len": 15,         # frames; shorter shots are never used
        "samples_per_shot": 5,      # frames sampled to classify a shot
        "min_person_h": 0.25,       # fighter box height / frame height lower bound
        "max_person_h": 0.95,       # ... upper bound (above = close-up)
        "min_ring_ratio": 0.6,      # share of samples that must look like a ring shot
        "max_large_persons": 5,     # fighters + referee + corner staff
    },
    "pose": {
        "model": "yolo11s-pose.pt",
        "conf": 0.25,
        "imgsz": 960,
        "device": None,             # None = auto (cuda if available)
        "tracker": "bytetrack.yaml",
        "frame_stride": 1,          # 1 = every frame
    },
    "identify": {
        "n_roles": 3,               # fighter 0, fighter 1, referee
        "samples_per_track": 10,    # frames sampled per track to describe its appearance
        "min_track_len": 5,         # tracks with fewer detections are ignored
        "min_joint_conf": 0.3,      # shoulders + hips must be visible to describe a person
        "referee_max_skin": 0.40,   # referee cluster = shirted: mean torso skin share below this ...
        "referee_min_gap": 0.25,    # ... and at least this far below the fighters' skin share
        "referee_max_skin_assign": 0.45,  # per detection: referee only if torso skin share <= this
        "fighter_min_skin": 0.40,         # per detection: fighter only if torso skin share >= this
        "other_ratio": 3.0,         # farther than this * cluster median distance = not one of the roles
        "other_floor": 2.0,         # ... but never stricter than this (distance in z-scored feature units)
        "smooth_window": 9,         # detections; rolling median of appearance along a track
        "max_cluster_samples": 30000,
        "seed": 0,
        "min_fighter_h": 0.25,      # fighter box height / frame height; smaller = crowd/staff
        "static_min_len": 60,       # detections (~2 s) before a track can be judged static; short fragments never are
        "static_motion_ratio": 0.015,  # center std on the quieter axis below this * box height ...
        "static_max_ratio": 0.1,       # ... and on the busier axis below this * box height = overlay graphic
    },
    "quality": {
        "joint_conf": 0.3,          # joint counted as visible above this
        "min_visible_joints": 10,
        "min_mean_conf": 0.5,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_ingest_config(path: Path | str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_INGEST)
    if path and Path(path).exists():
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def ns(d: dict) -> SimpleNamespace:
    """dict -> attribute access (cfg.pose.model)."""
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})
