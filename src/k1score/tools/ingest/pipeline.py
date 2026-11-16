"""Runs the ingest stages in order, resumable via the manifest."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

from k1score.core.io import load_poses, read_json, write_json
from k1score.core.manifest import STATUS_DONE, STATUS_FAILED, Manifest
from k1score.core.paths import Paths
from k1score.core.schema import INGEST_STAGES, SCHEMA_VERSION, validate_poses

from . import download as dl
from .export import export_dataset
from .pose import load_model, make_person_detector, run_pose
from .probe import probe_video
from .identify import run_identify, swap_fighters
from .select import select_by_identity
from .shots import classify_shots, cuts_to_shots, detect_cuts


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class IngestPipeline:
    def __init__(self, paths: Paths, cfg: dict, log=_log):
        self.paths, self.cfg, self.log = paths, cfg, log
        self.manifest = Manifest(paths.manifest)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self.log(f"loading pose model {self.cfg['pose']['model']} ...")
            self._model = load_model(self.cfg["pose"], self.paths.root / "models")
        return self._model

    # ------------------------------------------------------------------ add
    def add(self, src: str) -> list[str]:
        """Register videos (URL / playlist / local file / .txt list). Returns new video ids."""
        ids = []
        for s in dl.expand_sources(src, self.cfg):
            try:
                info = dl.add_local(s, self.paths) if dl.is_local_video(s) else dl.download(s, self.paths, self.cfg)
            except dl.IngestError as e:
                self.log(f"[add] FAILED {s}: {e}")
                continue
            vid = info["video_id"]
            self.manifest.upsert(vid, url=info["url"], title=info["title"], schema_version=SCHEMA_VERSION)
            self.manifest.set_status(vid, "download", STATUS_DONE)
            self.log(f"[add] {vid}  {info['title']}")
            ids.append(vid)
        return ids

    # ------------------------------------------------------------------ run
    def _outputs(self, vid: str) -> dict[str, list[Path]]:
        p = self.paths
        return {
            "download": [p.source_json(vid)],
            "probe": [p.probe_json(vid)],
            "shots": [p.shots_json(vid)],
            "pose": [p.tracks_npz(vid)],
            "identify": [p.identities_json(vid), p.identities_npz(vid), p.identity_sheet(vid)],
            "select": [p.fighters_npz(vid)],
            "export": [p.poses_npz(vid), p.meta_json(vid)],
        }

    def run(self, vid: str, only_stage: str | None = None, force: bool = False) -> bool:
        if self.manifest.get(vid) is None:
            self.log(f"[{vid}] unknown video id (use `k1 ingest add` first)")
            return False
        stages = [only_stage] if only_stage else INGEST_STAGES[1:]  # `download` happens in add()
        for st in stages:
            fresh = (self.manifest.status(vid, st) == STATUS_DONE
                     and all(f.exists() for f in self._outputs(vid)[st]))
            if fresh and not (force or only_stage):
                continue
            self.manifest.invalidate_from(vid, st)  # downstream results are now stale
            self.log(f"[{vid}] stage: {st}")
            try:
                getattr(self, f"_stage_{st}")(vid)
            except Exception as e:
                self.manifest.set_status(vid, st, STATUS_FAILED, error=f"{type(e).__name__}: {e}")
                self.log(f"[{vid}] FAILED at {st}: {e}")
                if not isinstance(e, dl.IngestError):
                    self.log(traceback.format_exc())
                return False
            self.manifest.set_status(vid, st, STATUS_DONE)
        return True

    # --------------------------------------------------------------- stages
    def _video(self, vid: str) -> Path:
        video = self.paths.video(vid)
        if video is None:
            raise dl.IngestError(f"video file missing for {vid} (deleted, or a dangling symlink to a moved file?) in {self.paths.raw(vid)}")
        return video

    def _stage_download(self, vid: str) -> None:
        url = self.manifest.get(vid)["url"]
        if dl.is_local_video(url):
            dl.add_local(url, self.paths)
        else:
            dl.download(url, self.paths, self.cfg)

    def _stage_probe(self, vid: str) -> None:
        video = self.paths.video(vid)
        if video is None:
            raise dl.IngestError("video file missing in data/raw")
        info = probe_video(video, self.cfg)
        write_json(self.paths.probe_json(vid), info)

    def _stage_shots(self, vid: str) -> None:
        video, probe = self._video(vid), read_json(self.paths.probe_json(vid))
        sc = self.cfg["shots"]
        cuts, n_frames = detect_cuts(video, sc["cut_threshold"])
        shots = cuts_to_shots(cuts, n_frames)
        self.log(f"  {n_frames} frames, {len(shots)} shots; classifying ...")
        detector = make_person_detector(self.model, self.cfg["pose"])
        shots = classify_shots(video, shots, probe["height"], sc, detector)
        ring = sum(s["end"] - s["start"] for s in shots if s["type"] == "ring_wide")
        self.log(f"  ring frames: {ring}/{n_frames}")
        write_json(self.paths.shots_json(vid), {"n_frames": n_frames, "shots": shots})

    def _stage_pose(self, vid: str) -> None:
        info = read_json(self.paths.shots_json(vid))
        tracks = run_pose(self._video(vid), info["shots"], info["n_frames"], self.cfg["pose"],
                          model=self.model, log=self.log)
        self.paths.interim(vid).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.paths.tracks_npz(vid), **tracks)

    def _stage_identify(self, vid: str) -> None:
        info, probe = read_json(self.paths.shots_json(vid)), read_json(self.paths.probe_json(vid))
        with np.load(self.paths.tracks_npz(vid)) as z:
            tracks = {k: z[k] for k in z.files}
        out = run_identify(self._video(vid), tracks, info["n_frames"], probe["height"], self.cfg["identify"],
                           self.paths, vid, log=self.log)
        if not out["found"]:
            self.log("  WARNING: could not separate fighters/referee (too little data?)")
        self.log("  roles: " + ", ".join(f"{r} (in {v['n_frames']} frames)" for r, v in out["roles"].items()))
        self.log(f"  check: {self.paths.identity_sheet(vid)}")

    def swap(self, vid: str) -> None:
        """Manual fix after reviewing identity_sheet.png: exchange fighter_0 / fighter_1."""
        swap_fighters(self.paths, vid)
        self.manifest.invalidate_from(vid, "select")

    def _stage_select(self, vid: str) -> None:
        info = read_json(self.paths.shots_json(vid))
        with np.load(self.paths.tracks_npz(vid)) as z:
            tracks = {k: z[k] for k in z.files}
        with np.load(self.paths.identities_npz(vid)) as z:
            roles = z["role"]
        fighters, referee, shot_id = select_by_identity(tracks, roles, info["shots"], info["n_frames"])
        np.savez_compressed(self.paths.fighters_npz(vid), keypoints=fighters, referee=referee, shot_id=shot_id)

    def _stage_export(self, vid: str) -> None:
        with np.load(self.paths.fighters_npz(vid)) as z:
            kpts, referee, shot_id = z["keypoints"], z["referee"], z["shot_id"]
        meta = export_dataset(
            vid, self.paths, self.cfg, read_json(self.paths.source_json(vid)),
            read_json(self.paths.probe_json(vid)), read_json(self.paths.shots_json(vid)),
            kpts, referee, shot_id, self.cfg["pose"]["model"], read_json(self.paths.identities_json(vid)),
        )
        self.manifest.upsert(vid, n_frames=meta["n_frames"], valid_ratio=meta["quality"]["valid_ratio"],
                             schema_version=meta["schema_version"])

    # ------------------------------------------------------------- validate
    def validate(self, ids: list[str] | None = None) -> dict[str, list[str]]:
        problems: dict[str, list[str]] = {}
        for vid in ids or self.paths.processed_ids():
            try:
                load_poses(vid, self.paths, check=True)
                problems[vid] = []
            except Exception as e:
                problems[vid] = [str(e)]
        return problems


__all__ = ["IngestPipeline", "validate_poses"]
