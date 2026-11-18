"""Look at the strikes: annotated video and sample pictures of single strikes."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from k1score.core.io import Poses
from k1score.core.schema import LIMBS
from k1score.tools.ingest.debug import COLORS, draw_frame

OUTCOME_COLOR = {"hit": (60, 200, 60), "blocked": (0, 200, 230), "miss": (170, 170, 170), "unknown": (90, 90, 90)}
SHOW_AFTER_PEAK = 12   # frames the label stays on screen after the peak


def _annotate(frame: np.ndarray, poses: Poses, strikes: pd.DataFrame, n: int) -> np.ndarray:
    draw_frame(frame, poses, n)
    active = strikes[(strikes.start_frame <= n) & (n <= strikes.peak_frame + SHOW_AFTER_PEAK)]
    for k, r in enumerate(active.itertuples()):
        color = OUTCOME_COLOR[r.outcome]
        f = int(r.attacker[-1])
        tip = poses.keypoints[min(n, r.peak_frame), f, LIMBS[r.limb]]
        if tip[2] > 0.3:
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 14, color, 3)
        text = f"{r.attacker[-1]}: {r.strike_type} {r.outcome.upper()} {r.confidence:.2f}"
        cv2.putText(frame, text, (12, 60 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[f], 4)
        cv2.putText(frame, text, (12, 60 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def render_video(video: Path, poses: Poses, strikes: pd.DataFrame, out: Path, start: int = 0,
                 seconds: float = 30.0) -> Path:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = poses.size
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), poses.fps, (w, h))
    for n in range(start, min(len(poses.shot_id), start + int(seconds * poses.fps))):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(_annotate(frame, poses, strikes, n))
    cap.release()
    writer.release()
    return out


def render_samples(video: Path, poses: Poses, strikes: pd.DataFrame, out_dir: Path, n: int = 12, seed: int = 0) -> list[Path]:
    """Peak-frame pictures of n random strikes (confident ones first-class citizens of the sample, all outcomes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("strike_*.jpg"):
        old.unlink()
    rows = strikes.sample(min(n, len(strikes)), random_state=seed).sort_values("peak_frame")
    cap = cv2.VideoCapture(str(video))
    paths = []
    for r in rows.itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.peak_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _annotate(frame, poses, strikes, r.peak_frame)
        path = out_dir / f"strike_{r.strike_id:04d}_{r.outcome}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    cap.release()
    return paths
