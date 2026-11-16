"""Human-friendly preview of a processed video: annotated frames, contact sheet, coverage timeline, index.html."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from k1score.core.io import Poses

from .debug import COLORS, GREY, REF_COLOR, draw_skeleton

NAMES = ["Fighter 0 (red)", "Fighter 1 (blue)"]


def _fmt_t(n: int, fps: float) -> str:
    s = int(n / fps)
    return f"{s // 60:02d}:{s % 60:02d}"


def _status(poses: Poses, n: int, f: int) -> str:
    if np.isnan(poses.keypoints[n, f, :, 0]).all():
        return "NOT DETECTED"
    return "ok" if poses.valid[n, f] else "low quality"


def annotate(frame: np.ndarray, poses: Poses, n: int) -> np.ndarray:
    """Skeletons + a name tag above each fighter + a caption bar with time and per-fighter status."""
    h, w = frame.shape[:2]
    people = [(poses.keypoints[n, f], COLORS[f] if poses.valid[n, f] else GREY, f"Fighter {f}") for f in range(2)]
    if poses.referee is not None:
        people.append((poses.referee[n], REF_COLOR, "Referee"))
    for kp, color, label in people:
        if np.isnan(kp[:, 0]).all():
            continue
        draw_skeleton(frame, kp, color, thick=max(2, w // 480), radius=max(3, w // 400))
        pts = kp[kp[:, 2] > 0.3, :2]
        if len(pts):
            x, y = int(pts[:, 0].mean()), int(pts[:, 1].min()) - 14
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            x = int(np.clip(x - tw // 2, 4, w - tw - 4)); y = int(np.clip(y, th + 8, h - 8))
            cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), color, -1)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    bar = np.zeros((70, w, 3), np.uint8)
    cv2.putText(bar, f"time {_fmt_t(n, poses.fps)}   frame {n}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for f in range(2):
        st = _status(poses, n, f)
        col = (80, 200, 80) if st == "ok" else ((60, 160, 255) if st == "low quality" else (90, 90, 230))
        cv2.putText(bar, f"{NAMES[f]}: {st}", (14 + f * (w // 2), 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return np.vstack([bar, frame])


def pick_frames(poses: Poses, n_good: int = 9, n_bad: int = 3, seed: int = 0) -> list[tuple[int, str]]:
    """Evenly spaced frames where both fighters are fine, plus examples where something is wrong."""
    ring = poses.shot_id >= 0
    both = poses.valid.all(axis=1) & ring
    bad = ring & ~both
    picks: list[tuple[int, str]] = []
    idx = np.flatnonzero(both)
    if len(idx):
        picks += [(int(idx[int(i)]), "good") for i in np.linspace(0, len(idx) - 1, n_good)]
    idx = np.flatnonzero(bad)
    if len(idx):
        rng = np.random.default_rng(seed)
        picks += [(int(i), "problem") for i in sorted(rng.choice(idx, size=min(n_bad, len(idx)), replace=False))]
    return picks


def timeline(poses: Poses, width: int = 1800) -> np.ndarray:
    """Rows: shot type, fighter 0, fighter 1. Brighter = more usable frames in that time slice."""
    n = poses.keypoints.shape[0]
    edges = np.linspace(0, n, width + 1).astype(int)
    rows = [("ring shot", (poses.shot_id >= 0).astype(float)),
            ("Fighter 0", poses.valid[:, 0].astype(float)), ("Fighter 1", poses.valid[:, 1].astype(float))]
    row_colors = [(170, 170, 170), COLORS[0], COLORS[1]]
    left, rh, gap, top = 110, 46, 10, 40
    img = np.full((top + 3 * (rh + gap) + 40, left + width, 3), 255, np.uint8)
    for r, ((name, arr), col) in enumerate(zip(rows, row_colors)):
        y = top + r * (rh + gap)
        cv2.putText(img, name, (6, y + rh // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        for x in range(width):
            a, b = edges[x], max(edges[x + 1], edges[x] + 1)
            frac = float(arr[a:b].mean())
            c = tuple(int(255 - frac * (255 - v)) for v in col)
            img[y:y + rh, left + x] = c
    total_s = n / poses.fps
    step = 60 if total_s > 240 else 30
    for t in range(0, int(total_s) + 1, step):
        x = left + int(t / total_s * width)
        cv2.line(img, (x, top - 8), (x, top + 3 * (rh + gap)), (200, 200, 200), 1)
        cv2.putText(img, f"{t // 60}:{t % 60:02d}", (x - 14, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(img, "Usable-pose coverage over time (darker colour = more usable frames; white = none)",
                (left, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    return img


def contact_sheet(images: list[np.ndarray], cols: int = 3, tile_w: int = 640) -> np.ndarray:
    tiles = []
    for im in images:
        s = tile_w / im.shape[1]
        tiles.append(cv2.resize(im, (tile_w, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA))
    th = max(t.shape[0] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [np.pad(t, ((0, th - t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in tiles[i:i + cols]]
        while len(row) < cols:
            row.append(np.full((th, tile_w + 4, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


HTML = """<!doctype html><meta charset=utf-8><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1300px;background:#fafafa;color:#222}}
img{{max-width:100%;border:1px solid #ccc;border-radius:6px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.good{{color:#157f3b}} .problem{{color:#c0392b}} code{{background:#eee;padding:2px 5px;border-radius:4px}}</style>
<h1>{title}</h1>
<p>{summary}</p>
<h2>How to read the images</h2>
<ul><li><b style="color:#ff3c3c">Red skeleton</b> = Fighter 0, <b style="color:#2878ff">blue skeleton</b> = Fighter 1, <b style="color:#2e9e2e">green skeleton</b> = Referee.
Identities are decided once per video from appearance (shorts colour, bare torso vs shirt), so Fighter 0 is the same person in every shot.</li>
<li><b>Grey skeleton</b> = a pose was found but is low quality, so it is excluded from the dataset.</li>
<li>The bar above each frame shows the time and whether each fighter is <i>ok</i>, <i>low quality</i> or <i>NOT DETECTED</i>.</li></ul>
{identity}<h2>Coverage over time</h2><img src="timeline.png">
<h2>Overview</h2><a href="contact_sheet.png"><img src="contact_sheet.png"></a>
<h2>Frames</h2><div class="grid">{cards}</div>"""


def render_preview(video: Path, poses: Poses, out_dir: Path, title: str, identity_sheet: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):   # previous run's frames (names include the frame number)
        old.unlink()
    picks = pick_frames(poses)
    cap = cv2.VideoCapture(str(video))
    images, cards = [], []
    for k, (n, kind) in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            continue
        im = annotate(frame, poses, n)
        im = cv2.resize(im, (1280, int(im.shape[0] * 1280 / im.shape[1])), interpolation=cv2.INTER_AREA)
        name = f"frame_{k:02d}_{kind}_{n}.png"
        cv2.imwrite(str(out_dir / name), im)
        images.append(im)
        label = "typical frame" if kind == "good" else "problem example (a fighter missing or low quality)"
        cards.append(f'<figure><img src="{name}"><figcaption class="{kind}">{label} - {_fmt_t(n, poses.fps)}</figcaption></figure>')
    cap.release()
    cv2.imwrite(str(out_dir / "timeline.png"), timeline(poses))
    if images:
        cv2.imwrite(str(out_dir / "contact_sheet.png"), contact_sheet(images))

    ring = poses.shot_id >= 0
    both = float((poses.valid.all(axis=1) & ring).sum() / max(1, ring.sum()))
    ref = ""
    if poses.referee is not None:
        ref = f" Referee visible in {(~np.isnan(poses.referee[..., 0]).all(axis=-1)).mean():.0%} of frames."
    summary = (f"{poses.keypoints.shape[0]} frames, {poses.fps:.0f} fps, {len(poses.meta['shots'])} shots. "
               f"Fighter 0 usable in {poses.valid[:, 0].mean():.0%} of frames, fighter 1 in {poses.valid[:, 1].mean():.0%}, "
               f"both at once in {both:.0%} of ring frames." + ref)
    identity = ""
    if identity_sheet is not None and identity_sheet.exists():
        cv2.imwrite(str(out_dir / "identity_sheet.png"), cv2.imread(str(identity_sheet)))
        identity = ('<h2>Who is who (check this first)</h2><p>Sample crops of the people the tool assigned to each role. '
                    'If a row shows the wrong person, run <code>k1 ingest identify &lt;id&gt; --swap</code> (fighters swapped) '
                    'or tell me.</p><img src="identity_sheet.png">')
    (out_dir / "index.html").write_text(HTML.format(title=title, summary=summary, cards="".join(cards), identity=identity))
    return out_dir / "index.html"
