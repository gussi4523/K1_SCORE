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


def _stage2_parsers(sub) -> None:
    p = sub.add_parser("strikes", help="stage 2 (model v0.0.1): skeletons -> strikes with hit / blocked / miss")
    p.add_argument("--config", default=None, help="yaml overrides (default: configs/strikes.yaml)")
    ssub = p.add_subparsers(dest="cmd", required=True)
    r = ssub.add_parser("run", help="find strikes, write data/events/<id>/strikes.parquet + frames.parquet")
    r.add_argument("video_id", nargs="*")
    r.add_argument("--all", action="store_true")
    sm = ssub.add_parser("samples", help="peak-frame pictures of random strikes")
    sm.add_argument("video_id")
    sm.add_argument("-n", type=int, default=12)
    rv = ssub.add_parser("render", help="video with strikes and outcomes drawn on it")
    rv.add_argument("video_id")
    rv.add_argument("--start", type=int, default=0, help="start frame")
    rv.add_argument("--seconds", type=float, default=30.0)

    s = sub.add_parser("score", help="stage 3 (model v0.0.1): strikes -> rounds -> decision")
    s.add_argument("video_id")
    s.add_argument("--config", default=None, help="yaml overrides (default: configs/scoring.yaml)")
    s.add_argument("--round-seconds", type=float, default=None, help="split video time into rounds of this length")
    s.add_argument("--redo", action="store_true", help="recompute strikes first")


def _run_strikes(args) -> int:
    from k1score.core.config import DEFAULT_STRIKES, load_config
    from k1score.tools.strikes.pipeline import run_strikes, summary
    from k1score.tools.strikes.render import render_samples, render_video

    paths = Paths()
    cfg = load_config(DEFAULT_STRIKES, args.config or paths.configs / "strikes.yaml")
    if args.cmd == "run":
        ids = paths.processed_ids() if args.all else args.video_id
        if not ids:
            print("give a video id or --all", file=sys.stderr)
            return 2
        for vid in ids:
            print(f"== {vid}\n{summary(run_strikes(vid, paths, cfg))}\n-> {paths.strikes_parquet(vid)}")
        return 0

    import pandas as pd

    poses = load_poses(args.video_id, paths)
    if not paths.strikes_parquet(args.video_id).exists():
        run_strikes(args.video_id, paths, cfg)
    strikes = pd.read_parquet(paths.strikes_parquet(args.video_id))
    if args.cmd == "samples":
        for path in render_samples(paths.video(args.video_id), poses, strikes, paths.events(args.video_id) / "samples", args.n):
            print(path)
        return 0
    print(render_video(paths.video(args.video_id), poses, strikes, paths.events(args.video_id) / "strikes.mp4",
                       start=args.start, seconds=args.seconds))
    return 0


def _run_score(args) -> int:
    import pandas as pd

    from k1score.core.config import DEFAULT_SCORING, DEFAULT_STRIKES, load_config
    from k1score.core.io import load_meta, write_json
    from k1score.tools.score.score import format_report, score_video
    from k1score.tools.strikes.pipeline import run_strikes

    paths = Paths()
    vid = args.video_id
    if args.redo or not paths.strikes_parquet(vid).exists():
        run_strikes(vid, paths, load_config(DEFAULT_STRIKES, paths.configs / "strikes.yaml"))
    cfg = load_config(DEFAULT_SCORING, args.config or paths.configs / "scoring.yaml")
    if args.round_seconds:
        cfg["round_seconds"] = args.round_seconds
    result = score_video(pd.read_parquet(paths.strikes_parquet(vid)), load_meta(vid, paths), cfg)
    write_json(paths.scores_json(vid), result)
    (paths.scores(vid) / "report.txt").write_text(format_report(result) + "\n")
    print(format_report(result))
    print(f"\n-> {paths.scores_json(vid)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k1", description="K-1 kickboxing scoring toolkit")
    sub = parser.add_subparsers(dest="tool", required=True)
    _ingest_parser(sub)
    _stage2_parsers(sub)
    args = parser.parse_args(argv)
    if args.tool == "ingest":
        return _run_ingest(args)
    if args.tool == "strikes":
        return _run_strikes(args)
    if args.tool == "score":
        return _run_score(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
