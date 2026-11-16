import numpy as np

from k1score.core.config import DEFAULT_INGEST
from k1score.tools.ingest.quality import compute_valid, valid_ratio
from k1score.tools.ingest import identify as I
from k1score.tools.ingest.select import select_by_identity
from k1score.tools.ingest.tracks import eligible_track, group_tracks
from k1score.tools.ingest.shots import classify_shots, cuts_to_shots, detect_cuts, looks_like_ring

SC = DEFAULT_INGEST["shots"]


def test_cut_detection(video):
    cuts, n = detect_cuts(video, SC["cut_threshold"])
    assert n == 120 and cuts == [0, 40, 80]
    shots = cuts_to_shots(cuts, n)
    assert [(s["start"], s["end"]) for s in shots] == [(0, 40), (40, 80), (80, 120)]


def test_ring_classification(video):
    shots = cuts_to_shots([0, 40, 80], 120)
    two = np.array([[0, 100, 50, 300], [200, 100, 250, 300]], dtype=float)
    seq_flat = iter([two, np.zeros((0, 4)), two[:1]])  # shot0: ring, shot1: empty, shot2: one person
    out = classify_shots(video, shots, 480, {**SC, "samples_per_shot": 1}, lambda fr: [next(seq_flat) for _ in fr])
    assert [s["type"] for s in out] == ["ring_wide", "other", "other"]
    assert looks_like_ring(two, 480, SC) and not looks_like_ring(two[:1], 480, SC)


def test_quality():
    kp = np.full((3, 2, 17, 3), np.nan, np.float32)
    kp[0, :, :, :] = 0.9          # frame 0: both good
    kp[1, 0, :, :] = 0.9          # frame 1: only fighter 0
    kp[2, :, :, :] = 0.9; kp[2, 1, :, 2] = 0.1  # frame 2: fighter 1 low confidence
    v = compute_valid(kp, DEFAULT_INGEST["quality"])
    assert v.tolist() == [[True, True], [True, False], [True, False]]
    assert valid_ratio(v, np.array([0, 0, -1])) == 0.5


# ------------------------------------------------------------------ identify / select
ICFG = DEFAULT_INGEST["identify"]


def _rows(rng, n, skin, tdark, cr, cg, sat, sdark, sv):
    base = np.array([skin, tdark, cr, cg, sat, sdark, sv, 30, 30, 30], float)
    return base + rng.normal(0, 0.02, (n, 10))


def _synthetic_appearance(n=300, seed=1):
    rng = np.random.default_rng(seed)
    ref = _rows(rng, n, 0.05, 0.9, 0.34, 0.33, 0.10, 0.9, 0.15)      # dark shirt, no skin
    a = _rows(rng, n, 0.70, 0.10, 0.34, 0.33, 0.15, 0.8, 0.20)       # bare torso, black shorts
    b = _rows(rng, n, 0.70, 0.10, 0.60, 0.25, 0.70, 0.1, 0.50)       # bare torso, red shorts
    return ref, a, b


def test_learn_roles_separates_referee_and_both_fighters():
    ref, a, b = _synthetic_appearance()
    model = I.learn_roles(np.vstack([ref, a, b]), ICFG)
    assert model is not None and set(model["centroids"]) == {"fighter_0", "fighter_1", "referee"}
    Fz = lambda F: (F[:, :I.N_F] - model["mu"]) / model["sd"] * I.COL_WEIGHT
    frames = np.arange(3)
    rows = np.array([a[0], b[0], ref[0]])
    roles, _ = I.assign_frames(Fz(rows), np.zeros(3, int), model, rows[:, 0], ICFG)
    assert roles.tolist() == [I.CODE["fighter_0"], I.CODE["fighter_1"], I.CODE["referee"]]   # black shorts = fighter_0


def test_assign_frames_one_role_per_frame_and_clothing_gate():
    ref, a, b = _synthetic_appearance()
    model = I.learn_roles(np.vstack([ref, a, b]), ICFG)
    Fz = lambda F: (F[:, :I.N_F] - model["mu"]) / model["sd"] * I.COL_WEIGHT
    # a frame with two shirted people (referee + a photographer in black): only one may become referee,
    # the other must NOT fall into a fighter slot even though both fighter slots are free
    rows = np.array([ref[0], ref[1]])
    roles, _ = I.assign_frames(Fz(rows), np.zeros(2, int), model, rows[:, 0], ICFG)
    assert sorted(roles.tolist()) == [0, I.CODE["referee"]]


def test_learn_roles_without_referee_returns_two_fighters():
    _, a, b = _synthetic_appearance()
    model = I.learn_roles(np.vstack([a, b]), ICFG)
    assert model is not None and set(model["centroids"]) == {"fighter_0", "fighter_1"}


def test_describe_measures_skin_and_shorts(people_video):
    import cv2
    cap = cv2.VideoCapture(str(people_video)); ok, frame = cap.read(); cap.release()
    from conftest import BOXES
    def kp_for(box):
        k = np.zeros((17, 3), np.float32)
        k[:, 0] = np.linspace(box[0], box[2], 17); k[:, 1] = np.linspace(box[1], box[3], 17); k[:, 2] = 0.9
        return k
    fa, ref = I.describe(frame, kp_for(BOXES[0]), 0.3), I.describe(frame, kp_for(BOXES[2]), 0.3)
    assert fa[0] > 0.8 and ref[0] < 0.1          # skin share: fighter high, referee ~0
    assert ref[1] > 0.9                          # referee torso is dark
    assert fa[5] > 0.9                           # fighter A (black shorts) shorts are dark


def test_static_overlay_is_not_eligible():
    cfg = ICFG
    n = 80
    overlay = np.tile([520.0, 200, 620, 480], (n, 1)); overlay[:, [0, 2]] += ((np.arange(n) % 3) * 4)[:, None]
    moving = np.tile([100.0, 100, 200, 400], (n, 1)) + np.arange(n)[:, None] * np.array([1, 1, 1, 1])
    assert not eligible_track(overlay, 480, cfg)
    assert eligible_track(moving, 480, cfg)
    assert not eligible_track(np.tile([0.0, 0, 20, 50], (n, 1)) + np.arange(n)[:, None], 480, cfg)   # too small (crowd)


def test_select_by_identity_scatters_roles():
    kp = np.zeros((5, 17, 3), np.float32)
    for i in range(5):
        kp[i] = i + 1
    tracks = {"frame": np.array([0, 0, 0, 1, 1]), "shot": np.zeros(5, int), "track_id": np.arange(5),
              "box": np.zeros((5, 4), np.float32), "kpts": kp}
    roles = np.array([I.CODE["fighter_1"], I.CODE["fighter_0"], I.CODE["referee"], 0, I.CODE["fighter_0"]], np.int8)
    shots = [{"id": 0, "start": 0, "end": 3, "type": "ring_wide"}]
    f, r, sid = select_by_identity(tracks, roles, shots, 3)
    assert f[0, 0, 0, 0] == 2 and f[0, 1, 0, 0] == 1 and r[0, 0, 0] == 3     # fighter_0 -> slot 0, referee separate
    assert f[1, 0, 0, 0] == 5 and np.isnan(f[1, 1]).all() and np.isnan(r[1]).all()
    assert np.isnan(f[2]).all() and sid.tolist() == [0, 0, 0]
