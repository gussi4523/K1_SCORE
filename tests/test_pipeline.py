import numpy as np
from conftest import FakeModel

from k1score.cli import main
from k1score.core.config import load_ingest_config
from k1score.core.io import load_poses
from k1score.tools.ingest.pipeline import IngestPipeline


def _pipe(paths):
    cfg = load_ingest_config(None)
    pipe = IngestPipeline(paths, cfg, log=lambda s: None)
    pipe._model = FakeModel()
    return pipe


def test_end_to_end_and_idempotent(paths, people_video):
    pipe = _pipe(paths)
    (vid,) = pipe.add(str(people_video))
    assert pipe.run(vid)
    p = load_poses(vid, paths)
    assert p.keypoints.shape == (120, 2, 17, 3) and p.referee.shape == (120, 17, 3)
    assert p.valid.all() and (p.shot_id >= 0).all()      # fake model: every shot looks like a ring shot
    assert pipe.manifest.status(vid, "export") == "done"

    # identity is by appearance: fighter 0 (black shorts) stays the same person although the fighters swap
    # sides in the middle shot, and the referee never takes a fighter slot
    x0 = p.keypoints[:, 0, :, 0].mean(axis=1); x1 = p.keypoints[:, 1, :, 0].mean(axis=1); xr = p.referee[:, :, 0].mean(axis=1)
    left = np.r_[np.ones(40, bool), np.zeros(40, bool), np.ones(40, bool)]      # fighter A is on the left in shots 0, 2
    inner = np.r_[np.arange(5, 35), np.arange(45, 75), np.arange(85, 115)]       # ignore frames next to cuts
    assert ((x0 < x1) == left)[inner].mean() > 0.95
    assert (np.abs(xr - 315) < 40)[inner].mean() > 0.95                           # referee array holds the middle person
    assert not np.isnan(p.referee[inner]).any()
    assert pipe.validate() == {vid: []}

    # second run: nothing recomputed
    mtime = paths.poses_npz(vid).stat().st_mtime_ns
    assert pipe.run(vid)
    assert paths.poses_npz(vid).stat().st_mtime_ns == mtime

    # forcing one stage invalidates and rebuilds downstream
    assert pipe.run(vid, only_stage="identify")
    assert pipe.manifest.status(vid, "select") == "pending" and pipe.manifest.status(vid, "export") == "pending"
    assert pipe.run(vid)
    assert pipe.manifest.status(vid, "export") == "done"


def test_probe_rejects_low_res(paths, tmp_path):
    from conftest import make_video
    import cv2, numpy as np
    small = tmp_path / "small.mp4"
    w = cv2.VideoWriter(str(small), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for _ in range(10):
        w.write(np.zeros((120, 160, 3), np.uint8))
    w.release()
    pipe = _pipe(paths)
    (vid,) = pipe.add(str(small))
    assert not pipe.run(vid)
    assert pipe.manifest.status(vid, "probe") == "failed"
    assert "height" in pipe.manifest.get(vid)["error"]


def test_cli_status_and_validate(paths, video, monkeypatch, capsys):
    monkeypatch.setenv("K1_ROOT", str(paths.root))
    pipe = _pipe(paths)
    (vid,) = pipe.add(str(video))
    pipe.run(vid)
    assert main(["ingest", "status"]) == 0
    assert vid in capsys.readouterr().out
    assert main(["ingest", "validate"]) == 0


def test_swap_fighters_flips_slots(paths, people_video):
    pipe = _pipe(paths)
    (vid,) = pipe.add(str(people_video))
    assert pipe.run(vid)
    before = load_poses(vid, paths)
    pipe.swap(vid)
    assert pipe.run(vid)
    after = load_poses(vid, paths)
    assert np.allclose(before.keypoints[:, 0], after.keypoints[:, 1], equal_nan=True)
    assert np.allclose(before.keypoints[:, 1], after.keypoints[:, 0], equal_nan=True)
    assert after.meta["identities"]["swapped"] is True
    assert paths.identity_sheet(vid).exists()


def test_missing_video_fails_loudly_instead_of_producing_empty_data(paths, video):
    pipe = _pipe(paths)
    (vid,) = pipe.add(str(video))
    assert pipe.run(vid, only_stage="probe")
    video.unlink()                                   # raw/<id>/video.mp4 is now a dangling symlink
    assert not pipe.run(vid, only_stage="shots")
    assert pipe.manifest.status(vid, "shots") == "failed"
    assert "video file missing" in pipe.manifest.get(vid)["error"]
