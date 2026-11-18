"""Strike finder (v0.0.1, rule-based): a hand / foot / knee that leaves its "rest" position fast, toward the opponent.

For every limb we measure how far its tip is from where it normally sits relative to its own body (the rolling
median over ~2 s), in torso lengths. Body translation, camera pan and zoom cancel out. A strike is a fast peak of
that deviation, moving toward the opponent, while the opponent is within reach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .kinematics import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, Track, center, nanmean, rolling_median

# limb name -> (tip joint, root joint, knee joint or None)
HANDS = {"left_hand": (9, L_SHOULDER), "right_hand": (10, R_SHOULDER)}
LEGS = {"left_leg": (15, L_HIP, 13), "right_leg": (16, R_HIP, 14)}


@dataclass
class Event:
    attacker: int
    limb: str
    strike_type: str         # punch | kick_* | knee  (kick height is set later, once the opponent is known)
    start: int               # local indices inside the segment
    peak: int
    end: int
    tip: int                 # joint index of the striking tip
    strength: float


def _peak_events(sig: np.ndarray, fps: float, c: dict) -> list[tuple[int, int, int]]:
    """(start, peak, end) of fast peaks in a deviation signal. NaN counts as 'no motion' but a peak whose rise is
    mostly NaN is rejected."""
    s = np.nan_to_num(sig, nan=0.0)
    peaks, _ = find_peaks(s, height=c["min_dev"], prominence=c["min_prominence"],
                          distance=max(1, int(c["min_gap_s"] * fps)))
    out = []
    for p in peaks:
        floor = c["start_frac"] * s[p]
        a = p
        while a > 0 and s[a] > floor and p - a < int(c["rise_max_s"] * fps):
            a -= 1
        b = p
        while b < len(s) - 1 and s[b] > floor and b - p < int(c["fall_max_s"] * fps):
            b += 1
        if np.isnan(sig[a:p + 1]).mean() > 0.2:
            continue
        rise = np.diff(s[a:p + 1]) * fps
        if rise.size == 0 or rise.max() < c["min_speed"]:
            continue
        out.append((a, int(p), b))
    return out


def _rest_dev(rel: np.ndarray, fps: float, cfg: dict) -> np.ndarray:
    """Deviation vector of `rel` (T, 2) from its rolling-median rest position."""
    win = max(5, int(cfg["rest_window_s"] * fps) | 1)
    return rel - rolling_median(rel, win)


def find_strikes(att: Track, opp: Track, a_idx: int, fps: float, cfg: dict) -> list[Event]:
    T = len(att.xy)
    L = att.scale
    c_att, c_opp = center(att.xy), center(opp.xy)
    sx = np.sign(c_opp[:, 0] - c_att[:, 0])            # +1: opponent is to the right of the attacker
    close = np.linalg.norm(c_opp - c_att, axis=1) / L <= cfg["max_reach"]
    events: list[Event] = []

    for limb, (tip, root) in HANDS.items():
        d = _rest_dev(att.xy[:, tip] - att.xy[:, root], fps, cfg)
        dev = np.linalg.norm(d, axis=1) / L
        for a, p, b in _peak_events(dev, fps, cfg["hand"]):
            fwd = d[p, 0] * sx[p] / max(np.linalg.norm(d[p]), 1e-9)
            if close[p] and fwd >= cfg["hand"]["min_forward"]:
                events.append(Event(a_idx, limb, "punch", a, p, b, tip, float(dev[p])))

    lc = cfg["leg"]
    for limb, (ankle, hip, knee) in LEGS.items():
        da = _rest_dev(att.xy[:, ankle] - att.xy[:, hip], fps, cfg)
        dev_a = np.linalg.norm(da, axis=1) / L
        lift = (att.xy[:, hip, 1] - att.xy[:, knee, 1])      # knee above hip: positive (image y points down)
        dev_k = np.clip(_rest_dev(lift, fps, cfg) / L, 0, None)
        sig = np.fmax(dev_a, lc["knee_gain"] * dev_k)
        for a, p, b in _peak_events(sig, fps, lc):
            if not close[p]:
                continue
            shin = att.xy[p, ankle] - att.xy[p, knee]
            horiz = abs(shin[0]) / max(np.linalg.norm(shin), 1e-9)
            knee_like = lc["knee_gain"] * dev_k[p] >= dev_a[p] and horiz < lc["knee_shin_horizontal"]
            if knee_like:
                events.append(Event(a_idx, limb.replace("_leg", "_knee"), "knee", a, p, b, knee, float(sig[p])))
            else:
                fwd = da[p, 0] * sx[p] / max(np.linalg.norm(da[p]), 1e-9)
                if fwd >= lc["min_forward"]:
                    events.append(Event(a_idx, limb, "kick", a, p, b, ankle, float(sig[p])))
    return merge(events, fps, cfg["merge_s"])


def merge(events: list[Event], fps: float, merge_s: float) -> list[Event]:
    """Two detections of the same fighter closer than merge_s are one strike (body rotation can trigger both hands)."""
    events = sorted(events, key=lambda e: -e.strength)
    kept: list[Event] = []
    for e in events:
        if all(k.attacker != e.attacker or abs(k.peak - e.peak) > merge_s * fps for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.peak)


def kick_height(tip_y: float, opp_xy: np.ndarray) -> str:
    """Compare the foot's height with the opponent's shoulders and hips (image y grows downward)."""
    shoulder_y = nanmean(opp_xy[[L_SHOULDER, R_SHOULDER], 1], axis=0)
    hip_y = nanmean(opp_xy[[L_HIP, R_HIP], 1], axis=0)
    if np.isnan(shoulder_y) or np.isnan(hip_y):
        return "kick_mid"
    if tip_y < shoulder_y:
        return "kick_high"
    if tip_y < hip_y:
        return "kick_mid"
    return "kick_low"
