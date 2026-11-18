"""Stage 2 / 3 on synthetic skeletons: a fighter A throws one punch at fighter B, with B open, guarding or far."""
import numpy as np
import pandas as pd
import pytest

from k1score.core.config import DEFAULT_SCORING, DEFAULT_STRIKES, load_config
from k1score.core.io import Poses
from k1score.core.schema import COCO17_JOINTS, MODEL_VERSION, STRIKE_COLUMNS
from k1score.tools.score.score import make_rounds, round_score, score_video
from k1score.tools.strikes.pipeline import detect, frame_table
from k1score.tools.strikes.outcome import raised

FPS, N, PEAK = 30, 90, 40


def stance(cx, face):
    """Standing fighter, torso length 100 px, guard up. face=+1 looks right, -1 looks left."""
    kp = np.zeros((17, 3), np.float32)
    kp[:, 2] = 0.9
    pts = {0: (5, 60), 1: (3, 55), 2: (7, 55), 3: (-5, 60), 4: (15, 60),
           5: (-15, 100), 6: (15, 100), 7: (-10, 140), 8: (20, 140), 9: (30, 120), 10: (35, 120),
           11: (-12, 200), 12: (12, 200), 13: (-12, 290), 14: (12, 290), 15: (-14, 380), 16: (14, 380)}
    for j, (x, y) in pts.items():
        kp[j, :2] = (cx + face * x, y)
    return kp


def make_poses(gap=130, guard=False, n=N):
    """A (left, faces right) throws a right-hand punch at PEAK; B stands `gap` px away facing left."""
    kp = np.zeros((n, 2, 17, 3), np.float32)
    for t in range(n):
        kp[t, 0] = stance(100, +1)
        kp[t, 1] = stance(100 + gap, -1)
        if guard:   # B's gloves up in front of his face
            kp[t, 1, 9, :2] = (100 + gap - 25, 62)
            kp[t, 1, 10, :2] = (100 + gap - 30, 70)
            kp[t, 1, 7, :2] = (100 + gap - 15, 110)
            kp[t, 1, 8, :2] = (100 + gap - 22, 112)
        u = np.clip(1 - abs(t - PEAK) / 4.0, 0, 1)      # punch out and back in 8 frames
        kp[t, 0, 10, :2] = np.array([100 + 35, 120]) * (1 - u) + np.array([100 + 120, 62]) * u
    meta = {"video_id": "syn", "fps": float(FPS), "width": 640, "height": 480, "n_frames": n,
            "joint_names": COCO17_JOINTS}
    return Poses(keypoints=kp, valid=np.ones((n, 2), bool), shot_id=np.zeros(n, np.int32),
                 frame_idx=np.arange(n, dtype=np.int32), meta=meta)


@pytest.fixture
def cfg():
    return load_config(DEFAULT_STRIKES)


def test_no_strike_when_standing_still(cfg):
    poses = make_poses()
    poses.keypoints[:, 0, 10, :2] = poses.keypoints[0, 0, 10, :2]
    assert detect(poses, "syn", cfg).empty


def test_one_punch_found_with_right_fields(cfg):
    df = detect(make_poses(), "syn", cfg)
    assert list(df.columns) == STRIKE_COLUMNS
    assert len(df) == 1
    r = df.iloc[0]
    assert r.attacker == "fighter_0" and r.limb == "right_hand" and r.strike_type == "punch"
    assert abs(r.peak_frame - PEAK) <= 1 and r.start_frame < r.peak_frame < r.end_frame
    assert r.model_version == MODEL_VERSION and r.label_source == "auto"
    assert r.pose_quality == 1.0 and abs(r.p_hit + r.p_blocked + r.p_miss - 1.0) < 1e-6


def test_hit_block_miss(cfg):
    hit = detect(make_poses(gap=130), "syn", cfg).iloc[0]
    miss = detect(make_poses(gap=300), "syn", cfg).iloc[0]
    block = detect(make_poses(gap=140, guard=True), "syn", cfg).iloc[0]
    assert hit.outcome == "hit" and hit.target == "head"
    assert miss.outcome == "miss"
    assert block.outcome == "blocked"


def test_unknown_when_defender_missing_at_peak(cfg):
    poses = make_poses()
    poses.valid[PEAK - 2:PEAK + 3, 1] = False      # short gap: skeleton is interpolated, but not trusted for the outcome
    df = detect(poses, "syn", cfg)
    assert len(df) == 1 and df.iloc[0].outcome == "unknown" and df.iloc[0].pose_quality < 1.0


def test_defender_missing_for_long_means_no_strike(cfg):
    poses = make_poses()
    poses.valid[PEAK - 10:PEAK + 10, 1] = False     # long gap: direction / reach unknown, so nothing is claimed
    assert detect(poses, "syn", cfg).empty


def test_defender_far_away_is_not_a_strike(cfg):
    assert detect(make_poses(gap=900), "syn", cfg).empty


def test_frame_table_marks_strike_frames(cfg):
    poses = make_poses()
    st = detect(poses, "syn", cfg)
    ft = frame_table(poses, st)
    assert len(ft) == N and ft.f0_strike_id.notna().sum() == st.iloc[0].end_frame - st.iloc[0].start_frame + 1
    assert ft.f1_strike_id.isna().all()


def test_raised_leg_uses_thigh_angle():
    xy = np.zeros((17, 2))
    xy[11], xy[13] = (0, 0), (0, 90)           # vertical thigh
    assert not raised(xy, 11, 13, 35)
    xy[13] = (80, 30)                          # thigh ~70 deg from vertical
    assert raised(xy, 11, 13, 35)


# --- stage 3 ---------------------------------------------------------------------------------------------

def _strikes(rows):
    base = {"outcome": "hit", "strike_type": "punch", "confidence": 0.9, "pose_quality": 1.0}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["strike_id"] = range(len(df))
    return df


META = {"video_id": "syn", "fps": 30.0, "n_frames": 30 * 400}


def test_round_score_ten_point_must():
    cfg = DEFAULT_SCORING
    assert round_score(5, 3, cfg) == (10, 9)
    assert round_score(3, 5, cfg) == (9, 10)
    assert round_score(4, 4, cfg) == (10, 10)
    assert round_score(30, 3, cfg) == (10, 8)


def test_make_rounds():
    cfg = dict(DEFAULT_SCORING)
    assert make_rounds(400, cfg) == [(0.0, 400)]
    cfg["round_seconds"] = 180
    assert make_rounds(400, cfg) == [(0, 180), (180, 360), (360, 400)]
    cfg["round_seconds"] = 20
    assert len(make_rounds(60.03, cfg)) == 3             # a 0.03 s tail does not become a round
    cfg["round_boundaries"] = [[0, 100], [160, 260]]
    assert make_rounds(400, cfg) == [(0.0, 100.0), (160.0, 260.0)]


def test_scoring_counts_only_hits_and_skips_unknown():
    cfg = dict(DEFAULT_SCORING)
    df = _strikes([
        {"attacker": "fighter_0", "t_peak": 10.0},
        {"attacker": "fighter_0", "t_peak": 20.0, "strike_type": "kick_high"},   # 2 points
        {"attacker": "fighter_1", "t_peak": 30.0},
        {"attacker": "fighter_1", "t_peak": 40.0, "outcome": "blocked"},         # 0 points
        {"attacker": "fighter_1", "t_peak": 50.0, "outcome": "miss"},
        {"attacker": "fighter_1", "t_peak": 60.0, "outcome": "unknown"},
        {"attacker": "fighter_1", "t_peak": 70.0, "pose_quality": 0.1},          # low quality, skipped
    ])
    s = score_video(df, META, cfg)
    r = s["rounds"][0]
    assert r["fighter_0"]["points"] == 3.0 and r["fighter_1"]["points"] == 1.0
    assert r["fighter_0"]["hits_by_type"]["kick_high"] == 1
    assert r["round_score"] == [10, 9] and s["decision"] == "fighter_0"
    assert s["strikes_not_scored"] == 2 and s["total"]["score"] == {"fighter_0": 10, "fighter_1": 9}


def test_scoring_splits_rounds_and_can_draw():
    cfg = dict(DEFAULT_SCORING, round_seconds=200)
    df = _strikes([{"attacker": "fighter_0", "t_peak": 10.0}, {"attacker": "fighter_1", "t_peak": 300.0}])
    s = score_video(df, META, cfg)
    assert [r["round_score"] for r in s["rounds"]] == [[10, 9], [9, 10]]
    assert s["decision"] == "draw"


def test_scoring_with_no_strikes():
    s = score_video(_strikes([]).reindex(columns=STRIKE_COLUMNS), META, dict(DEFAULT_SCORING))
    assert s["decision"] == "draw" and s["rounds"][0]["round_score"] == [10, 10]
