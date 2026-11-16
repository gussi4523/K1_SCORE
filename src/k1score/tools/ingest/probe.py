"""Stage 2: read basic video properties and reject unusable videos."""
from __future__ import annotations

from pathlib import Path

import cv2

from .download import IngestError


def probe_video(video: Path, cfg: dict) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IngestError(f"OpenCV cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    info = {
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "n_frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["n_frames_reported"] / fps if fps > 0 else 0.0

    pc = cfg["probe"]
    if fps < pc["min_fps"]:
        raise IngestError(f"fps {fps:.1f} < min_fps {pc['min_fps']}")
    if info["height"] < pc["min_height"]:
        raise IngestError(f"height {info['height']} < min_height {pc['min_height']}")
    return info
