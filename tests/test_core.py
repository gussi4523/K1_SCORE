import numpy as np
import pytest

from k1score.core.config import DEFAULT_INGEST, load_ingest_config
from k1score.core.io import load_poses, save_poses
from k1score.core.manifest import Manifest
from k1score.core.schema import COCO17_JOINTS, SCHEMA_VERSION, validate_poses


def _arrays(n=10):
    kp = np.random.rand(n, 2, 17, 3).astype(np.float32)
    return {"keypoints": kp, "valid": np.ones((n, 2), bool), "shot_id": np.zeros(n, np.int32),
            "frame_idx": np.arange(n, dtype=np.int32)}


def _meta(n=10):
    return {"schema_version": SCHEMA_VERSION, "video_id": "v", "fps": 30.0, "width": 640, "height": 480,
            "n_frames": n, "joint_names": COCO17_JOINTS, "pose_model": {"name": "x"}, "tracker": "t",
            "shots": [{"id": 0, "start": 0, "end": n, "type": "ring_wide"}], "quality": {}}


def test_roundtrip(paths):
    a, m = _arrays(), _meta()
    save_poses("v", a, m, paths)
    p = load_poses("v", paths)
    assert np.array_equal(p.keypoints, a["keypoints"]) and p.fps == 30.0 and p.size == (640, 480)
    assert paths.processed_ids() == ["v"]


def test_validate_catches_problems():
    a, m = _arrays(), _meta()
    assert validate_poses(a, m) == []
    a["keypoints"][0, 0, 0, 0] = np.nan  # valid=True but NaN keypoint
    assert any("NaN" in e for e in validate_poses(a, m))
    assert any("shape" in e for e in validate_poses(_arrays(5), m))
    m2 = _meta(); m2["schema_version"] = "2.0.0"
    assert any("major" in e for e in validate_poses(_arrays(), m2))


def test_save_refuses_invalid(paths):
    a = _arrays(); a["valid"] = a["valid"].astype(np.float32)
    with pytest.raises(ValueError):
        save_poses("v", a, _meta(), paths)


def test_manifest(paths):
    m = Manifest(paths.manifest)
    m.upsert("a", url="u", title="t")
    m.set_status("a", "probe", "done")
    m.set_status("a", "shots", "done")
    m.invalidate_from("a", "shots")
    m2 = Manifest(paths.manifest)  # reload from parquet
    assert m2.status("a", "probe") == "done" and m2.status("a", "shots") == "pending"
    m2.upsert("a", n_frames=120, valid_ratio=0.5)
    assert Manifest(paths.manifest).get("a")["n_frames"] == 120


def test_config_override(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("pose:\n  imgsz: 640\n")
    cfg = load_ingest_config(f)
    assert cfg["pose"]["imgsz"] == 640 and cfg["pose"]["model"] == DEFAULT_INGEST["pose"]["model"]


def test_referee_array_is_optional_but_checked():
    a, m = _arrays(), _meta()
    assert validate_poses(a, m) == []                                   # schema 1.0 style file (no referee) still valid
    a["referee"] = np.zeros((10, 17, 3), np.float32)
    assert validate_poses(a, m) == []
    a["referee"] = np.zeros((10, 5, 3), np.float32)
    assert any("referee" in e for e in validate_poses(a, m))


def test_manifest_loads_rows_saved_before_a_stage_existed(paths):
    import pandas as pd
    from k1score.core.manifest import COLUMNS
    df = pd.DataFrame([{c: None for c in COLUMNS if c != "status_identify"} | {"video_id": "old", "status_select": "done"}])
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.manifest, index=False)
    m = Manifest(paths.manifest)
    assert m.status("old", "identify") == "pending" and m.status("old", "select") == "done"
