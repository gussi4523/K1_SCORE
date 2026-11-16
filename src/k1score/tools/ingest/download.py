"""Stage 1: get a video into data/raw/<video_id>/ (YouTube URL or local file)."""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from k1score.core.io import write_json
from k1score.core.paths import VIDEO_EXTS, Paths


class IngestError(RuntimeError):
    pass


def is_local_video(src: str) -> bool:
    p = Path(src).expanduser()
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def expand_sources(src: str, cfg: dict | None = None) -> list[str]:
    """A URL / local video / .txt list of either -> flat list of single-video sources."""
    p = Path(src).expanduser()
    if p.is_file() and p.suffix.lower() == ".txt":
        out: list[str] = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.extend(expand_sources(line, cfg))
        return out
    if is_local_video(src):
        return [str(p.resolve())]
    # URL: expand playlists/channels without downloading
    import yt_dlp

    opts = _ydl_base(cfg) | {"extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(src, download=False)
    if info and info.get("entries"):
        return [e["url"] if str(e.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={e['id']}"
                for e in info["entries"] if e]
    return [src]


def _ydl_base(cfg: dict | None) -> dict:
    opts: dict = {"quiet": True, "no_warnings": True, "noprogress": True}
    browser = (cfg or {}).get("download", {}).get("cookies_from_browser")
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:40] or "video"


def add_local(path: str, paths: Paths) -> dict:
    src = Path(path).resolve()
    digest = hashlib.sha1(str(src).encode()).hexdigest()[:6]
    video_id = f"local-{_slug(src.stem)}-{digest}"
    raw = paths.raw(video_id)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / f"video{src.suffix.lower()}"
    if not dst.exists():
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    info = {"video_id": video_id, "url": str(src), "title": src.stem, "source": "local"}
    write_json(paths.source_json(video_id), info)
    return info


def download(url: str, paths: Paths, cfg: dict) -> dict:
    """Download one YouTube video. video_id = YouTube id, so re-adding is idempotent."""
    import yt_dlp

    max_h = cfg["download"]["max_height"]
    probe_opts = _ydl_base(cfg) | {"skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            meta = ydl.extract_info(url, download=False)
    except Exception as e:  # yt-dlp raises DownloadError with a readable message
        raise IngestError(f"cannot read {url}: {e}") from e

    video_id = meta["id"]
    if paths.video(video_id) and paths.source_json(video_id).exists():
        return {"video_id": video_id, "url": url, "title": meta.get("title", video_id), "source": "youtube"}

    raw = paths.raw(video_id)
    raw.mkdir(parents=True, exist_ok=True)
    # Video-only (audio is not needed), H.264 preferred: OpenCV decodes it everywhere.
    fmt = (f"bv*[height<={max_h}][vcodec^=avc1]/bv*[height<={max_h}][ext=mp4]/"
           f"bv*[height<={max_h}]/b[height<={max_h}]")
    opts = _ydl_base(cfg) | {
        "format": fmt,
        "outtmpl": str(raw / "video.%(ext)s"),
        "noplaylist": True,
        "overwrites": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise IngestError(f"download failed for {url}: {e}") from e
    if not paths.video(video_id):
        raise IngestError(f"download finished but no video file found in {raw}")

    info = {
        "video_id": video_id,
        "url": meta.get("webpage_url", url),
        "title": meta.get("title", video_id),
        "uploader": meta.get("uploader"),
        "duration": meta.get("duration"),
        "upload_date": meta.get("upload_date"),
        "license": meta.get("license"),
        "source": "youtube",
    }
    write_json(paths.source_json(video_id), info)
    return info
