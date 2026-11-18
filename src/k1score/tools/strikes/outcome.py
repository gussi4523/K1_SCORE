"""Outcome (v0.0.1, rule-based): did the strike hit, was it blocked, or did it miss?

At the moment of closest approach we compare the striking tip with the defender's body parts, in the defender's
torso lengths:
  * target  = head, body (and legs for kicks): tip within `hit_dist` of it = touching
  * guard   = forearms and upper arms, plus a raised shin (a "check": thigh far from vertical)
A guard that the tip touches and that sits (almost) in front of the target = blocked.
This is a weak baseline: 2D has no depth, and the glove is not part of the skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, nanmean

HEAD = [0, 1, 2, 3, 4]
FOREARMS = [(7, 9), (8, 10)]
UPPER_ARMS = [(5, 7), (6, 8)]
THIGHS = [(11, 13), (12, 14)]
SHINS = [(13, 15), (14, 16)]


@dataclass
class Outcome:
    outcome: str
    p_hit: float
    p_blocked: float
    p_miss: float
    target: str
    d_target: float
    d_guard: float
    frame: int            # local index of the closest approach

    @property
    def confidence(self) -> float:
        return max(self.p_hit, self.p_blocked, self.p_miss)


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment a-b (inf if anything is missing)."""
    if np.isnan(p).any() or np.isnan(a).any() or np.isnan(b).any():
        return np.inf
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _min_dist(p, xy, segs) -> float:
    return min((seg_dist(p, xy[i], xy[j]) for i, j in segs), default=np.inf)


def raised(xy: np.ndarray, hip: int, knee: int, deg: float) -> bool:
    """Thigh far from vertical (knee lifted toward hip height) = leg raised."""
    d = xy[knee] - xy[hip]
    if np.isnan(d).any():
        return False
    return bool(np.degrees(np.arctan2(abs(d[0]), d[1])) >= deg)


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def distances(tip: np.ndarray, opp: np.ndarray, scale: float, strike_type: str, cfg: dict) -> tuple[str, float, float]:
    """(target name, distance to target, distance to guard), both in defender torso lengths."""
    head = nanmean(opp[HEAD], axis=0) if not np.isnan(opp[HEAD, 0]).all() else np.array([np.nan, np.nan])
    shoulders, hips = nanmean(opp[[L_SHOULDER, R_SHOULDER]], axis=0), nanmean(opp[[L_HIP, R_HIP]], axis=0)
    cand = {"head": float(np.linalg.norm(tip - head)) if not np.isnan(head).any() else np.inf,
            "body": seg_dist(tip, shoulders, hips)}
    guard_segs = FOREARMS + UPPER_ARMS
    leg_segs: list[tuple[int, int]] = []
    if strike_type != "punch":
        for (h, k), (_, a) in zip(THIGHS, SHINS):
            if raised(opp, h, k, cfg["raised_leg_deg"]):
                guard_segs = guard_segs + [(k, a)]          # raised shin = check
                leg_segs.append((h, k))
            else:
                leg_segs += [(h, k), (k, a)]
        cand["leg"] = _min_dist(tip, opp, leg_segs)
    target = min(cand, key=cand.get)
    return target, cand[target] / scale, _min_dist(tip, opp, guard_segs) / scale


def classify(tip_xy: np.ndarray, opp_xy: np.ndarray, valid: np.ndarray, opp_scale: np.ndarray,
             strike_type: str, peak: int, cfg: dict) -> Outcome:
    """tip_xy (T, 2) of the striking tip, opp_xy (T, 17, 2), valid (T,) = both fighters raw-valid, all local to the
    segment. Looks at peak +- window and keeps the frame where the tip is closest to a target."""
    best = None
    for j in range(max(0, peak - cfg["window"]), min(len(tip_xy), peak + cfg["window"] + 1)):
        if not valid[j] or np.isnan(tip_xy[j]).any() or np.isnan(opp_scale[j]):
            continue
        target, d_t, d_g = distances(tip_xy[j], opp_xy[j], opp_scale[j], strike_type, cfg)
        if np.isfinite(d_t) and (best is None or d_t < best[1]):
            best = (target, d_t, d_g, j)
    if best is None:
        return Outcome("unknown", 0.0, 0.0, 0.0, "", np.nan, np.nan, peak)
    target, d_t, d_g, j = best
    s = cfg["softness"]
    touch = _sigmoid((cfg["hit_dist"] - d_t) / s)
    guard = _sigmoid((cfg["guard_dist"] - d_g) / s) if np.isfinite(d_g) else 0.0
    in_front = _sigmoid((d_t + cfg["guard_margin"] - d_g) / s) if np.isfinite(d_g) else 0.0
    p_block = guard * in_front
    p_hit = touch * (1.0 - p_block)
    p_miss = max(0.0, 1.0 - p_hit - p_block)
    probs = {"hit": p_hit, "blocked": p_block, "miss": p_miss}
    return Outcome(max(probs, key=probs.get), p_hit, p_block, p_miss, target, d_t, d_g, j)
