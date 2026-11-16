"""Stage 3: cut detection + ring-shot classification.

Only wide ring shots (two fighters fully visible) are used downstream; replays, close-ups and
crowd shots are marked `other`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from k1score.core.schema import SHOT_OTHER, SHOT_RING_WIDE

# detector(frames) -> per frame an (n, 4) array of person boxes (xyxy, pixels)
PersonDetector = Callable[[list[np.ndarray]], list[np.ndarray]]


def _hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)


def detect_cuts(video: Path, threshold: float) -> tuple[list[int], int]:
    """Single sequential pass. Returns (frame numbers where a new shot starts, true frame count)."""
    cap = cv2.VideoCapture(str(video))
    cuts = [0]
    prev = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h = _hist(frame)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > threshold:
            cuts.append(n)
        prev = h
        n += 1
    cap.release()
    return cuts, n


def cuts_to_shots(cuts: list[int], n_frames: int) -> list[dict]:
    bounds = cuts + [n_frames]
    return [{"id": i, "start": bounds[i], "end": bounds[i + 1]} for i in range(len(cuts)) if bounds[i + 1] > bounds[i]]


def _read_frames(video: Path, frame_numbers: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = []
    for fn in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def looks_like_ring(boxes: np.ndarray, frame_h: int, sc: dict) -> bool:
    """>=2 person boxes of fighter-like size, and not a crowd of big people."""
    if len(boxes) == 0:
        return False
    heights = (boxes[:, 3] - boxes[:, 1]) / frame_h
    large = (heights >= sc["min_person_h"]) & (heights <= sc["max_person_h"])
    return 2 <= int(large.sum()) <= sc["max_large_persons"]


def classify_shots(video: Path, shots: list[dict], frame_h: int, sc: dict, detector: PersonDetector) -> list[dict]:
    for shot in shots:
        length = shot["end"] - shot["start"]
        if length < sc["min_shot_len"]:
            shot["type"], shot["ring_ratio"] = SHOT_OTHER, 0.0
            continue
        k = sc["samples_per_shot"]
        picks = sorted({int(shot["start"] + (i + 0.5) * length / k) for i in range(k)})
        frames = _read_frames(video, picks)
        if not frames:
            shot["type"], shot["ring_ratio"] = SHOT_OTHER, 0.0
            continue
        hits = sum(looks_like_ring(b, frame_h, sc) for b in detector(frames))
        shot["ring_ratio"] = round(hits / len(frames), 3)
        shot["type"] = SHOT_RING_WIDE if shot["ring_ratio"] >= sc["min_ring_ratio"] else SHOT_OTHER
    return shots
