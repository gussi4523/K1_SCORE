"""Optional: render the dataset back onto the video so a human can verify it quickly."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from k1score.core.io import Poses

# COCO17 skeleton edges
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLORS = [(60, 60, 255), (255, 120, 40)]  # BGR: fighter 0 red, fighter 1 blue
REF_COLOR = (60, 190, 60)                  # referee green
GREY = (150, 150, 150)                     # present but low quality


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, color, thick: int = 2, radius: int = 3) -> None:
    for a, b in EDGES:
        if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
            cv2.line(frame, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), color, thick)
    for j in range(kp.shape[0]):
        if kp[j, 2] > 0.3:
            cv2.circle(frame, tuple(map(int, kp[j, :2])), radius, color, -1)


def draw_frame(frame: np.ndarray, poses: Poses, n: int) -> np.ndarray:
    in_ring = poses.shot_id[n] >= 0
    label = f"frame {n}  shot {poses.shot_id[n]}" if in_ring else f"frame {n}  (skipped: not a ring shot)"
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for f in range(poses.keypoints.shape[1]):
        kp = poses.keypoints[n, f]
        if not np.isnan(kp[:, 0]).all():
            draw_skeleton(frame, kp, COLORS[f] if poses.valid[n, f] else GREY)
    if poses.referee is not None and not np.isnan(poses.referee[n, :, 0]).all():
        draw_skeleton(frame, poses.referee[n], REF_COLOR)
    return frame


def render_debug(video: Path, poses: Poses, out: Path, start: int = 0, seconds: float | None = None) -> Path:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = poses.size
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), poses.fps, (w, h))
    end = poses.keypoints.shape[0] if seconds is None else min(poses.keypoints.shape[0], start + int(seconds * poses.fps))
    for n in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_frame(frame, poses, n))
    cap.release()
    writer.release()
    return out


def view_interactive(video: Path, poses: Poses, start: int = 0) -> None:
    """Window with a slider. Keys: space play/pause, d/a (or ->/<-) +-1 frame, w/s +-1 s, [ ] +-10 s, q quit."""
    cap = cv2.VideoCapture(str(video))
    n_total = poses.keypoints.shape[0]
    win = "k1 viewer (space=play, a/d=frame, w/s=1s, [ ]=10s, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    state = {"n": max(0, min(start, n_total - 1)), "play": False, "shown": -1}
    cv2.createTrackbar("frame", win, state["n"], n_total - 1, lambda v: state.update(n=v))
    fps = poses.fps
    while True:
        n = state["n"]
        if n != state["shown"]:
            if n != state["shown"] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            if not ok:
                state["play"] = False
            else:
                cv2.imshow(win, draw_frame(frame, poses, n))
                cv2.setTrackbarPos("frame", win, n)
                state["shown"] = n
        key = cv2.waitKey(int(1000 / fps) if state["play"] else 30) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 or key == ord("q"):
            break
        step = {ord("d"): 1, 83: 1, ord("a"): -1, 81: -1, ord("w"): int(fps), ord("s"): -int(fps),
                ord("]"): int(10 * fps), ord("["): -int(10 * fps)}.get(key)
        if key == ord(" "):
            state["play"] = not state["play"]
        elif step:
            state["n"] = max(0, min(n_total - 1, state["n"] + step))
        elif state["play"]:
            state["n"] = min(n_total - 1, state["n"] + 1)
            if state["n"] == n_total - 1:
                state["play"] = False
    cap.release()
    cv2.destroyAllWindows()
