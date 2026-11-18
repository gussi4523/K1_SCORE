"""Stage 2 driver: poses.npz -> strikes.parquet + frames.parquet (model v0.0.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from k1score.core.io import Poses, load_poses
from k1score.core.paths import Paths
from k1score.core.schema import MODEL_VERSION, STRIKE_COLUMNS

from .finder import find_strikes, kick_height
from .kinematics import build_track, segments
from .outcome import classify


def detect(poses: Poses, video_id: str, cfg: dict) -> pd.DataFrame:
    """Strike table for one video (rows sorted by peak frame)."""
    fps = poses.fps
    rows = []
    for a, b in segments(poses):
        shot = int(poses.shot_id[a])
        tracks = [build_track(poses.keypoints[a:b, f], poses.valid[a:b, f], fps, cfg) for f in (0, 1)]
        both = tracks[0].valid & tracks[1].valid
        for att in (0, 1):
            opp = 1 - att
            for ev in find_strikes(tracks[att], tracks[opp], att, fps, cfg):
                stype = ev.strike_type
                if stype == "kick":
                    stype = kick_height(tracks[att].xy[ev.peak, ev.tip, 1], tracks[opp].xy[ev.peak])
                res = classify(tracks[att].xy[:, ev.tip], tracks[opp].xy, both, tracks[opp].scale, stype, ev.peak,
                               cfg["outcome"])
                window = slice(ev.start, ev.end + 1)
                rows.append({
                    "video_id": video_id, "shot_id": shot,
                    "start_frame": a + ev.start, "peak_frame": a + ev.peak, "end_frame": a + ev.end,
                    "attacker": f"fighter_{att}", "limb": ev.limb, "strike_type": stype, "target": res.target,
                    "outcome": res.outcome, "p_hit": res.p_hit, "p_blocked": res.p_blocked, "p_miss": res.p_miss,
                    "confidence": res.confidence, "d_target": res.d_target, "d_guard": res.d_guard,
                    "pose_quality": float(both[window].mean()),
                    "label_source": "auto", "model_version": MODEL_VERSION,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=STRIKE_COLUMNS)
    df = df.sort_values("peak_frame").reset_index(drop=True)
    df.insert(0, "strike_id", np.arange(len(df)))
    for k in ("start", "peak", "end"):
        df[f"t_{k}"] = df[f"{k}_frame"] / fps
    return df[STRIKE_COLUMNS]


def frame_table(poses: Poses, strikes: pd.DataFrame) -> pd.DataFrame:
    """One row per frame: which strike each fighter is performing (lookup / display only; scoring uses strikes)."""
    n = len(poses.shot_id)
    cols = {f"f{f}_{k}": np.full(n, None, dtype=object) for f in (0, 1) for k in ("strike_id", "outcome")}
    for r in strikes.sort_values("confidence").itertuples():     # stronger strikes overwrite weaker overlaps
        f = r.attacker[-1]
        cols[f"f{f}_strike_id"][r.start_frame:r.end_frame + 1] = r.strike_id
        cols[f"f{f}_outcome"][r.start_frame:r.end_frame + 1] = r.outcome
    df = pd.DataFrame(cols)
    df.insert(0, "frame", poses.frame_idx)
    df.insert(1, "shot_id", poses.shot_id)
    df["valid_f0"], df["valid_f1"] = poses.valid[:, 0], poses.valid[:, 1]
    return df


def run_strikes(video_id: str, paths: Paths, cfg: dict) -> pd.DataFrame:
    poses = load_poses(video_id, paths)
    strikes = detect(poses, video_id, cfg)
    out = paths.events(video_id)
    out.mkdir(parents=True, exist_ok=True)
    strikes.to_parquet(paths.strikes_parquet(video_id), index=False)
    frame_table(poses, strikes).astype({"f0_strike_id": "Int64", "f1_strike_id": "Int64"}).to_parquet(
        paths.frames_parquet(video_id), index=False)
    return strikes


def summary(strikes: pd.DataFrame) -> str:
    if strikes.empty:
        return "no strikes found"
    t = pd.crosstab([strikes["attacker"], strikes["strike_type"]], strikes["outcome"])
    return f"{len(strikes)} strikes\n{t.to_string()}"
