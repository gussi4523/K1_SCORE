"""The only module that knows where files live."""
from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def find_root() -> Path:
    """Project root: $K1_ROOT, else the nearest parent of cwd containing pyproject.toml + configs/."""
    env = os.environ.get("K1_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "pyproject.toml").exists() and (p / "configs").is_dir():
            return p
    return Path(__file__).resolve().parents[3]


class Paths:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).resolve() if root else find_root()
        self.configs = self.root / "configs"
        self.data = self.root / "data"
        self.manifest = self.data / "manifest.parquet"
        self.labels = self.data / "labels"

    # --- per-video directories -------------------------------------------------
    def raw(self, video_id: str) -> Path:
        return self.data / "raw" / video_id

    def interim(self, video_id: str) -> Path:
        return self.data / "interim" / video_id

    def processed(self, video_id: str) -> Path:
        return self.data / "processed" / video_id

    # --- files -----------------------------------------------------------------
    def video(self, video_id: str) -> Path | None:
        d = self.raw(video_id)
        for ext in VIDEO_EXTS:
            p = d / f"video{ext}"
            if p.exists():
                return p
        return None

    def source_json(self, video_id: str) -> Path:
        return self.raw(video_id) / "source.json"

    def probe_json(self, video_id: str) -> Path:
        return self.interim(video_id) / "probe.json"

    def shots_json(self, video_id: str) -> Path:
        return self.interim(video_id) / "shots.json"

    def tracks_npz(self, video_id: str) -> Path:
        return self.interim(video_id) / "tracks.npz"

    def identities_json(self, video_id: str) -> Path:
        return self.interim(video_id) / "identities.json"

    def identities_npz(self, video_id: str) -> Path:
        return self.interim(video_id) / "identities.npz"

    def identity_crops(self, video_id: str) -> Path:
        return self.interim(video_id) / "identity_crops"

    def identity_sheet(self, video_id: str) -> Path:
        return self.interim(video_id) / "identity_sheet.png"

    def fighters_npz(self, video_id: str) -> Path:
        return self.interim(video_id) / "fighters.npz"

    def debug_video(self, video_id: str) -> Path:
        return self.interim(video_id) / "debug.mp4"

    def poses_npz(self, video_id: str) -> Path:
        return self.processed(video_id) / "poses.npz"

    def meta_json(self, video_id: str) -> Path:
        return self.processed(video_id) / "meta.json"

    def processed_ids(self) -> list[str]:
        base = self.data / "processed"
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if (p / "meta.json").exists())
