"""`k1` command line. Each tool registers its own sub-command group."""
from __future__ import annotations

import argparse
import sys

from k1score.core.config import load_ingest_config
from k1score.core.io import load_poses
from k1score.core.paths import Paths
from k1score.core.schema import INGEST_STAGES


def _ingest_parser(sub) -> None:
    p = sub.add_parser("ingest", help="tool 01: YouTube video -> pose dataset")
    p.add_argument("--config", default=None, help="yaml config (default: configs/ingest.yaml)")
    isub = p.add_subparsers(dest="cmd", required=True)

    a = isub.add_parser("add", help="register a URL / playlist / local video / .txt list")
    a.add_argument("source")
    a.add_argument("--run", action="store_true", help="process right away")

    r = isub.add_parser("run", help="run pipeline stages")
    r.add_argument("--video", action="append", help="video id (repeatable)")
    r.add_argument("--all", action="store_true")
    r.add_argument("--stage", choices=INGEST_STAGES[1:], help="run only this stage (forced)")
    r.add_argument("--force", action="store_true", help="redo every stage")

    ident = isub.add_parser("identify", help="show / fix who is fighter 0, fighter 1, referee")
    ident.add_argument("video_id")
    ident.add_argument("--swap", action="store_true", help="exchange fighter 0 and fighter 1, then rebuild select+export")

    isub.add_parser("status", help="show manifest")

    d = isub.add_parser("debug", help="render skeleton overlay video")
    d.add_argument("video_id")
    d.add_argument("--start", type=int, default=0, help="start frame")
    d.add_argument("--seconds", type=float, default=30.0)

    vw = isub.add_parser("view", help="interactive frame-by-frame viewer with skeletons (needs a display)")
    vw.add_argument("video_id")
    vw.add_argument("--start", type=int, default=0, help="start frame")

    pv = isub.add_parser("preview", help="annotated sample frames + coverage timeline + index.html")
    pv.add_argument("video_id")

    v = isub.add_parser("validate", help="check processed datasets against the schema")
    v.add_argument("video_id", nargs="*")


def _run_ingest(args) -> int:
    from k1score.tools.ingest.debug import render_debug
    from k1score.tools.ingest.pipeline import IngestPipeline

    paths = Paths()
    cfg = load_ingest_config(args.config or paths.configs / "ingest.yaml")
    pipe = IngestPipeline(paths, cfg)

    if args.cmd == "add":
        ids = pipe.add(args.source)
        if args.run:
            return 0 if all([pipe.run(v) for v in ids]) and ids else 1
        return 0 if ids else 1

    if args.cmd == "run":
        ids = pipe.manifest.ids() if args.all else (args.video or [])
        if not ids:
            print("give --video ID or --all", file=sys.stderr)
            return 2
        return 0 if all([pipe.run(v, only_stage=args.stage, force=args.force) for v in ids]) else 1

    if args.cmd == "identify":
        if args.swap:
            pipe.swap(args.video_id)
            ok = pipe.run(args.video_id)
        else:
            ok = True
        print(paths.identity_sheet(args.video_id))
        return 0 if ok else 1

    if args.cmd == "status":
        df = pipe.manifest.table()
        if df.empty:
            print("manifest is empty")
            return 0
        df = df.rename(columns=lambda c: c.replace("status_", ""))
        print(df.drop(columns=["url", "schema_version"]).to_string(index=False, max_colwidth=28))
        return 0

    if args.cmd == "debug":
        poses = load_poses(args.video_id, paths)
        out = render_debug(paths.video(args.video_id), poses, paths.debug_video(args.video_id),
                           start=args.start, seconds=args.seconds)
        print(out)
        return 0

    if args.cmd == "view":
        from k1score.tools.ingest.debug import view_interactive

        view_interactive(paths.video(args.video_id), load_poses(args.video_id, paths), start=args.start)
        return 0

    if args.cmd == "preview":
        from k1score.tools.ingest.preview import render_preview

        poses = load_poses(args.video_id, paths)
        print(render_preview(paths.video(args.video_id), poses, paths.interim(args.video_id) / "preview",
                             poses.meta.get("title") or args.video_id, paths.identity_sheet(args.video_id)))
        return 0

    if args.cmd == "validate":
        problems = pipe.validate(args.video_id or None)
        bad = {k: v for k, v in problems.items() if v}
        for vid, errs in problems.items():
            print(f"{'FAIL' if errs else 'ok  '} {vid}" + (f": {errs[0]}" if errs else ""))
        return 1 if bad else 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k1", description="K-1 kickboxing scoring toolkit")
    sub = parser.add_subparsers(dest="tool", required=True)
    _ingest_parser(sub)
    args = parser.parse_args(argv)
    if args.tool == "ingest":
        return _run_ingest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
