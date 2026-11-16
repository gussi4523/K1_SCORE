"""Stage 6 (select): apply the per-detection roles from `identify` to the raw detections.

Slot 0 is the same person for the whole video (fighter_0), slot 1 the other fighter; the referee has his own
array and can never occupy a fighter slot. `identify` already guarantees at most one detection per role per
frame, so this is a plain scatter.
"""
from __future__ import annotations

import numpy as np

from k1score.core.schema import N_FIGHTERS, N_JOINTS, SHOT_RING_WIDE

from .identify import CODE, ROLE_FIGHTER_0, ROLE_FIGHTER_1, ROLE_REFEREE


def select_by_identity(tracks: dict[str, np.ndarray], roles: np.ndarray, shots: list[dict], n_frames: int):
    """Returns (fighters (N,2,17,3), referee (N,17,3), shot_id (N,)); NaN where a person is absent."""
    fighters = np.full((n_frames, N_FIGHTERS, N_JOINTS, 3), np.nan, dtype=np.float32)
    referee = np.full((n_frames, N_JOINTS, 3), np.nan, dtype=np.float32)
    shot_id = np.full(n_frames, -1, dtype=np.int32)
    for s in shots:
        if s["type"] == SHOT_RING_WIDE:
            shot_id[s["start"]:s["end"]] = s["id"]
    fr = tracks["frame"]
    for slot, role in enumerate((ROLE_FIGHTER_0, ROLE_FIGHTER_1)):
        m = roles == CODE[role]
        fighters[fr[m], slot] = tracks["kpts"][m]
    m = roles == CODE[ROLE_REFEREE]
    referee[fr[m]] = tracks["kpts"][m]
    return fighters, referee, shot_id
