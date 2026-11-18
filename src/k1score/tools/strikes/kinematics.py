"""Clean, gap-filled joint tracks per continuous ring segment (input of the strike finder)."""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd

from k1score.core.io import Poses

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmean without the 'Mean of empty slice' warning (all-NaN stays NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


@dataclass
class Track:
    """One fighter inside one segment. Positions in pixels; NaN = unknown."""
    xy: np.ndarray        # (T, 17, 2) smoothed, gaps <= max_gap interpolated
    scale: np.ndarray     # (T,) torso length in pixels (shoulder centre -> hip centre), rolling median
    valid: np.ndarray     # (T,) bool, raw stage-1 validity (NOT interpolated)


def segments(poses: Poses) -> list[tuple[int, int]]:
    """Index ranges [a, b) of consecutive frames in the same ring shot."""
    shot = poses.shot_id
    out, a = [], None
    for i in range(len(shot)):
        if shot[i] < 0 or (a is not None and shot[i] != shot[a]):
            if a is not None:
                out.append((a, i))
                a = None
        if shot[i] >= 0 and a is None:
            a = i
    if a is not None:
        out.append((a, len(shot)))
    return out


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs of at most max_gap samples that have a good neighbour on both sides."""
    x = x.copy()
    bad = np.isnan(x)
    if not bad.any() or bad.all():
        return x
    edges = np.diff(np.r_[0, bad.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s == 0 or e == len(x) or e - s > max_gap:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """NaN-aware centred moving average; positions that were NaN stay NaN."""
    if w <= 1:
        return x
    ok = ~np.isnan(x)
    k = np.ones(w)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    out = num / np.maximum(den, 1e-9)
    out[~ok] = np.nan
    return out


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median along axis 0 for a (T,) or (T, k) array; tolerant to NaN."""
    return pd.DataFrame(x.reshape(len(x), -1)).rolling(window, center=True, min_periods=3).median().to_numpy().reshape(x.shape)


def build_track(kp: np.ndarray, valid: np.ndarray, fps: float, cfg: dict) -> Track:
    """kp: (T, 17, 3) of one fighter in one segment."""
    xy = kp[..., :2].astype(np.float64).copy()
    xy[(kp[..., 2] < cfg["min_joint_conf"]) | ~valid[:, None]] = np.nan
    for j in range(xy.shape[1]):
        for c in range(2):
            xy[:, j, c] = smooth(fill_gaps(xy[:, j, c], cfg["max_gap"]), cfg["smooth"])
    shoulders = nanmean(xy[:, [L_SHOULDER, R_SHOULDER]], axis=1)
    hips = nanmean(xy[:, [L_HIP, R_HIP]], axis=1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    win = max(3, int(cfg["scale_window_s"] * fps) | 1)
    scale = pd.Series(rolling_median(torso, win)).ffill().bfill().to_numpy()
    return Track(xy=xy, scale=scale, valid=valid.copy())


def center(xy: np.ndarray) -> np.ndarray:
    """(T, 2) body centre = mean of shoulders and hips."""
    return nanmean(xy[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]], axis=1)
