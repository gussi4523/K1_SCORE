"""Helpers for the flat detection table written by the `pose` stage.

Tracker ids restart at every shot, so a track is identified by (shot, track_id).
"""
from __future__ import annotations

import numpy as np

KEY_MUL = 1_000_000


def track_keys(tracks: dict[str, np.ndarray]) -> np.ndarray:
    return tracks["shot"].astype(np.int64) * KEY_MUL + tracks["track_id"].astype(np.int64)


def key_str(key: int) -> str:
    return f"{int(key) // KEY_MUL}:{int(key) % KEY_MUL}"


def group_tracks(tracks: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """track key -> row indices of that track's detections."""
    keys = track_keys(tracks)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    bounds = np.flatnonzero(np.diff(sk)) + 1
    return {int(sk[g[0]]): order[g] for g in np.split(np.arange(len(sk)), bounds) if len(g)}


def eligible_track(box: np.ndarray, frame_h: int, cfg: dict) -> bool:
    """Fighter-sized and not a static broadcast overlay (portrait graphic, logo)."""
    h = float((box[:, 3] - box[:, 1]).mean())
    if h < cfg["min_fighter_h"] * frame_h:
        return False
    if len(box) >= cfg["static_min_len"]:
        cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
        sx, sy = cx.std(), cy.std()
        # overlay: almost no motion on one axis AND little overall (a fighter pacing along one axis
        # still has a large std on the other)
        if min(sx, sy) < cfg["static_motion_ratio"] * h and max(sx, sy) < cfg["static_max_ratio"] * h:
            return False
    return True
