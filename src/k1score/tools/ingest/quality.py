"""Stage 6 (computed inside export): per-frame, per-fighter usability mask."""
from __future__ import annotations

import numpy as np


def compute_valid(kpts: np.ndarray, qcfg: dict) -> np.ndarray:
    """kpts (N,2,17,3) -> valid (N,2) bool: enough confidently visible joints."""
    conf = np.nan_to_num(kpts[..., 2], nan=0.0)
    visible = conf >= qcfg["joint_conf"]
    n_vis = visible.sum(axis=-1)
    mean_conf = np.where(n_vis > 0, (conf * visible).sum(-1) / np.maximum(n_vis, 1), 0.0)
    present = ~np.isnan(kpts[..., 0]).any(axis=-1)
    return present & (n_vis >= qcfg["min_visible_joints"]) & (mean_conf >= qcfg["min_mean_conf"])


def valid_ratio(valid: np.ndarray, shot_id: np.ndarray) -> float:
    """Share of ring-shot frames where BOTH fighters are usable."""
    ring = shot_id >= 0
    if not ring.any():
        return 0.0
    return float(valid[ring].all(axis=1).mean())
