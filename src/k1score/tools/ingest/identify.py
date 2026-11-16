"""Stage 5 (identify): decide WHO is who, once per video, from appearance - per detection, not per track.

Tracker ids restart at every camera cut and jump between people in clinches (one track can follow fighter A,
then the referee, then fighter B), and "left/right" changes when fighters switch sides. So identity is decided
for every single detection in every frame from what the person wears:

    referee               - shirt: little bare skin on the torso, torso mostly dark
    fighter_0, fighter_1  - bare torso; told apart by shorts colour (saturation, chromaticity, darkness)
    other                 - crowd, staff, broadcast overlays, anything that fits no role

Steps
  1. measure appearance of each detection (torso + shorts regions found from the pose keypoints), smooth it
     along its track (short rolling median, removes occlusion noise)
  2. learn the roles once for the whole video: 2-means referee-vs-fighters, then 2-means fighter-vs-fighter
  3. per frame, assign detections to roles with the Hungarian algorithm: each role is used at most once per
     frame, and a detection far from every role stays `other`
Result: interim/<id>/identities.npz (role of every detection row), identities.json (summary),
identity_crops/ and identity_sheet.png (pictures to verify by eye).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from k1score.core.io import read_json, write_json
from k1score.core.paths import Paths
from k1score.core.schema import ROLE_FIGHTER_0, ROLE_FIGHTER_1, ROLE_OTHER, ROLE_REFEREE

from .tracks import eligible_track, group_tracks

SKIN_LO, SKIN_HI = (0, 40, 70), (25, 200, 255)   # HSV range of (light..tanned) skin
DARK_V = 60                                      # V below this = black cloth / deep shadow

# role codes stored in identities.npz
CODE = {ROLE_OTHER: 0, ROLE_FIGHTER_0: 1, ROLE_FIGHTER_1: 2, ROLE_REFEREE: 3}
NAME = {v: k for k, v in CODE.items()}

FEATURES = ["skin", "torso_dark", "shorts_chroma_r", "shorts_chroma_g", "shorts_sat", "shorts_dark", "shorts_v"]
N_F = len(FEATURES)                    # columns 0..6; columns 7..9 hold the mean shorts colour (B, G, R)
REF_COLS, FIGHT_COLS = [0, 1], [2, 3, 4, 5, 6]
COL_WEIGHT = np.array([2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0])   # torso cues separate referee from black-shorted fighter


# --------------------------------------------------------------------------- appearance measurements
def person_regions(kp: np.ndarray, shape: tuple[int, int], jconf: float):
    """Torso and shorts rectangles (x0, y0, x1, y1) from keypoints, or None if shoulders/hips not visible."""
    idx = [5, 6, 11, 12]
    if (kp[idx, 2] < jconf).any() or np.isnan(kp[idx, :2]).any():
        return None
    sh_y, hip_y = kp[[5, 6], 1].mean(), kp[[11, 12], 1].mean()
    th = hip_y - sh_y
    if th < 12:
        return None
    xs = kp[idx, 0]
    cx, w = (xs.min() + xs.max()) / 2, max(xs.max() - xs.min(), 0.4 * th)   # side views: shoulders ~ hips in x
    h_img, w_img = shape

    def clip(r):
        x0, y0, x1, y1 = int(max(r[0], 0)), int(max(r[1], 0)), int(min(r[2], w_img)), int(min(r[3], h_img))
        return (x0, y0, x1, y1) if (x1 - x0) * (y1 - y0) >= 16 and x1 - x0 >= 3 and y1 - y0 >= 3 else None

    torso = clip((cx - 0.35 * w, sh_y + 0.10 * th, cx + 0.35 * w, hip_y - 0.05 * th))
    shorts = clip((cx - 0.55 * w, hip_y, cx + 0.55 * w, hip_y + 0.45 * th))
    return (torso, shorts) if torso and shorts else None


def describe(frame: np.ndarray, kp: np.ndarray, jconf: float) -> np.ndarray | None:
    """10 numbers: the 7 FEATURES then mean shorts colour B, G, R. None if the body regions are not visible."""
    regions = person_regions(kp, frame.shape[:2], jconf)
    if regions is None:
        return None
    (tx0, ty0, tx1, ty1), (sx0, sy0, sx1, sy1) = regions
    t_hsv = cv2.cvtColor(frame[ty0:ty1, tx0:tx1], cv2.COLOR_BGR2HSV)
    s_roi = frame[sy0:sy1, sx0:sx1]
    s_hsv = cv2.cvtColor(s_roi, cv2.COLOR_BGR2HSV)
    bgr = s_roi.reshape(-1, 3).astype(np.float64).mean(axis=0)
    chroma = bgr / max(bgr.sum(), 1.0)                       # brightness-independent colour
    return np.array([
        cv2.inRange(t_hsv, SKIN_LO, SKIN_HI).mean() / 255.0,
        (t_hsv[..., 2] < DARK_V).mean(),
        chroma[2], chroma[1],
        s_hsv[..., 1].mean() / 255.0,
        (s_hsv[..., 2] < DARK_V).mean(),
        s_hsv[..., 2].mean() / 255.0,
        *bgr,
    ])


# --------------------------------------------------------------------------- clustering
def weighted_kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 8, iters: int = 60):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(X[rng.choice(len(X), p=p)])
        C = np.array(centers)
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        inertia = float(d[np.arange(len(X)), lab].sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def learn_roles(F: np.ndarray, icfg: dict) -> dict | None:
    """F: (n, 10) appearance rows (no NaN). Returns role centroids/cutoffs in z-scored, weighted feature space."""
    if len(F) < 20:
        return None
    X = F[:, :N_F]
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
    Z = (X - mu) / sd
    Zw = Z * COL_WEIGHT
    rng = np.random.default_rng(icfg["seed"])
    sub = rng.choice(len(X), size=min(len(X), icfg["max_cluster_samples"]), replace=False)

    # step 1: referee = shirted cluster (little skin), only if clearly separated from bare-torso fighters
    lab, C = weighted_kmeans(Zw[sub][:, REF_COLS], 2, icfg["seed"])
    skin = np.array([X[sub][lab == j, 0].mean() if (lab == j).any() else 9.0 for j in range(2)])
    ref_j = int(skin.argmin())
    has_ref = skin[ref_j] < icfg["referee_max_skin"] and (skin[1 - ref_j] - skin[ref_j]) > icfg["referee_min_gap"]
    member = {r: None for r in (ROLE_REFEREE, ROLE_FIGHTER_0, ROLE_FIGHTER_1)}
    fight_pool = np.arange(len(sub))
    if has_ref:
        member[ROLE_REFEREE] = sub[lab == ref_j]
        fight_pool = np.flatnonzero(lab != ref_j)

    # step 2: the two fighters by shorts appearance (among everything that is not referee)
    if len(fight_pool) < 20:
        return None
    flab, _ = weighted_kmeans(Zw[sub[fight_pool]][:, FIGHT_COLS], 2, icfg["seed"])
    sat = [X[sub[fight_pool][flab == j], 4].mean() if (flab == j).any() else 9.0 for j in range(2)]
    order = sorted(range(2), key=lambda j: sat[j])        # deterministic: less saturated shorts = fighter_0
    member[ROLE_FIGHTER_0] = sub[fight_pool][flab == order[0]]
    member[ROLE_FIGHTER_1] = sub[fight_pool][flab == order[1]]

    centroids, cutoff, info = {}, {}, {}
    for role, idx in member.items():
        if idx is None or len(idx) < 10:
            continue
        c = Zw[idx].mean(0)
        d = np.sqrt(((Zw[idx] - c) ** 2).sum(1))
        centroids[role] = c
        cutoff[role] = max(icfg["other_ratio"] * float(np.median(d)), icfg["other_floor"])
        bgr = F[idx, N_F:].mean(0)
        info[role] = {"samples": int(len(idx)), "torso_skin": round(float(X[idx, 0].mean()), 3),
                      "shorts_sat": round(float(X[idx, 4].mean()), 3),
                      "shorts_rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])]}
    if ROLE_FIGHTER_0 not in centroids or ROLE_FIGHTER_1 not in centroids:
        return None
    return {"mu": mu, "sd": sd, "centroids": centroids, "cutoff": cutoff, "info": info}


def assign_frames(Fz: np.ndarray, frames: np.ndarray, model: dict, skin: np.ndarray,
                  icfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, give each role to at most one detection (Hungarian). Fz: z-scored weighted features (NaN = unknown).
    `skin` (raw torso skin share per row) gates roles: fighters need a bare torso, the referee must be shirted.
    Returns (role code per row, distance to its role)."""
    roles = list(model["centroids"])
    Cm = np.array([model["centroids"][r] for r in roles])
    cut = np.array([model["cutoff"][r] for r in roles])
    is_ref = np.array([r == ROLE_REFEREE for r in roles])
    out = np.zeros(len(Fz), np.int8)
    dist = np.full(len(Fz), np.nan, np.float32)
    valid = ~np.isnan(Fz).any(axis=1)
    order = np.argsort(frames, kind="stable")
    sf = frames[order]
    bounds = np.flatnonzero(np.diff(sf)) + 1
    for grp in np.split(order, bounds):
        rows = grp[valid[grp]]
        if len(rows) == 0:
            continue
        D = np.sqrt(((Fz[rows][:, None, :] - Cm[None]) ** 2).sum(-1))       # (n_det, n_roles)
        ok_skin = np.where(is_ref[None], skin[rows][:, None] <= icfg["referee_max_skin_assign"],
                           skin[rows][:, None] >= icfg["fighter_min_skin"])
        cost = np.where((D <= cut[None]) & ok_skin, D, 1e6)                  # too far / wrong clothing = not that role
        r_idx, c_idx = linear_sum_assignment(cost)
        for i, j in zip(r_idx, c_idx):
            if cost[i, j] < 1e5:
                out[rows[i]] = CODE[roles[j]]
                dist[rows[i]] = D[i, j]
    return out, dist


# --------------------------------------------------------------------------- the stage
def run_identify(video: Path, tracks: dict[str, np.ndarray], n_frames: int, frame_h: int, icfg: dict,
                 paths: Paths, video_id: str, log=lambda s: None) -> dict:
    M = len(tracks["frame"])
    groups = group_tracks(tracks)
    eligible = {k: rows for k, rows in groups.items()
                if len(rows) >= icfg["min_track_len"] and eligible_track(tracks["box"][rows], frame_h, icfg)}
    elig_rows = np.sort(np.concatenate(list(eligible.values()))) if eligible else np.array([], int)
    log(f"  tracks: {len(groups)} total, {len(eligible)} fighter-sized and moving ({len(elig_rows)} detections)")

    # 1. appearance of every eligible detection (one sequential pass over the video)
    F = np.full((M, N_F + 3), np.nan)
    by_frame: dict[int, list[int]] = {}
    for r in elig_rows:
        by_frame.setdefault(int(tracks["frame"][r]), []).append(int(r))
    cap = cv2.VideoCapture(str(video))
    fn = 0
    for target in sorted(by_frame):
        while fn < target:
            if not cap.grab():
                break
            fn += 1
        ok, frame = cap.read()
        fn += 1
        if not ok:
            break
        for r in by_frame[target]:
            d = describe(frame, tracks["kpts"][r], icfg["min_joint_conf"])
            if d is not None:
                F[r] = d
    cap.release()

    # smooth along each track: rolling median over neighbouring detections (fills gaps, removes occlusion noise)
    win = icfg["smooth_window"]
    for rows in eligible.values():
        rows = rows[np.argsort(tracks["frame"][rows])]
        F[rows] = pd.DataFrame(F[rows]).rolling(win, center=True, min_periods=1).median().to_numpy()

    # 2. learn roles once for the whole video
    have = ~np.isnan(F[elig_rows]).any(axis=1)
    model = learn_roles(F[elig_rows][have], icfg)
    roles_arr = np.zeros(M, np.int8)
    dist_arr = np.full(M, np.nan, np.float32)
    if model is not None:
        # 3. assign per frame
        Fz = (F[elig_rows, :N_F] - model["mu"]) / model["sd"] * COL_WEIGHT
        r_el, d_el = assign_frames(Fz, tracks["frame"][elig_rows], model, F[elig_rows, 0], icfg)
        roles_arr[elig_rows], dist_arr[elig_rows] = r_el, d_el

    summary = {"swapped": False, "found": model is not None, "roles": {}}
    for role in (ROLE_FIGHTER_0, ROLE_FIGHTER_1, ROLE_REFEREE):
        m = roles_arr == CODE[role]
        if m.any() and model is not None:
            summary["roles"][role] = {**model["info"][role], "n_detections": int(m.sum()),
                                      "n_frames": int(len(np.unique(tracks["frame"][m])))}
    write_json(paths.identities_json(video_id), summary)
    np.savez_compressed(paths.identities_npz(video_id), role=roles_arr, dist=dist_arr)
    save_crops(video, tracks, roles_arr, dist_arr, paths, video_id)
    build_sheet(paths, video_id)
    return summary


def save_crops(video: Path, tracks: dict, roles_arr: np.ndarray, dist_arr: np.ndarray, paths: Paths,
               video_id: str, per_role: int = 8) -> None:
    """Typical crops per role, spread over the whole video (for the verification sheet)."""
    crop_dir = paths.identity_crops(video_id)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for old in crop_dir.glob("*.jpg"):
        old.unlink()
    rng = np.random.default_rng(0)
    picks: dict[int, list[tuple[str, int]]] = {}
    for code, role in NAME.items():
        rows = np.flatnonzero(roles_arr == code)
        if role == ROLE_OTHER:
            big = np.flatnonzero(tracks["box"][:, 3] - tracks["box"][:, 1] > 0)   # any detection
            rows = rng.choice(big, size=min(per_role, len(big)), replace=False) if len(big) and roles_arr.any() else []
            chosen = list(rows)
        elif len(rows):
            rows = rows[np.argsort(tracks["frame"][rows])]
            conf = rows[dist_arr[rows] <= np.nanpercentile(dist_arr[rows], 60)]      # most typical 60%
            conf = conf if len(conf) else rows
            chosen = [int(conf[int(i)]) for i in np.linspace(0, len(conf) - 1, min(per_role, len(conf)))]
        else:
            chosen = []
        for i, r in enumerate(chosen):
            picks.setdefault(int(tracks["frame"][r]), []).append((f"{role}_{i}", int(r)))
    cap = cv2.VideoCapture(str(video))
    for f in sorted(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        for name, r in picks[f]:
            x0, y0, x1, y1 = [int(v) for v in tracks["box"][r]]
            crop = frame[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)]
            if crop.size:
                cv2.imwrite(str(crop_dir / f"{name}.jpg"),
                            cv2.resize(crop, (max(1, int(160 * crop.shape[1] / crop.shape[0])), 160)))
    cap.release()


def swap_fighters(paths: Paths, video_id: str) -> None:
    """Manual fix: exchange fighter_0 and fighter_1 (e.g. to match red/blue corner)."""
    z = np.load(paths.identities_npz(video_id))
    role = z["role"].copy()
    a, b = role == CODE[ROLE_FIGHTER_0], role == CODE[ROLE_FIGHTER_1]
    role[a], role[b] = CODE[ROLE_FIGHTER_1], CODE[ROLE_FIGHTER_0]
    np.savez_compressed(paths.identities_npz(video_id), role=role, dist=z["dist"])
    data = read_json(paths.identities_json(video_id))
    r = data["roles"]
    if ROLE_FIGHTER_0 in r and ROLE_FIGHTER_1 in r:
        r[ROLE_FIGHTER_0], r[ROLE_FIGHTER_1] = r[ROLE_FIGHTER_1], r[ROLE_FIGHTER_0]
    data["swapped"] = not data["swapped"]
    write_json(paths.identities_json(video_id), data)
    cd = paths.identity_crops(video_id)
    for p in cd.glob(f"{ROLE_FIGHTER_0}_*.jpg"):
        p.rename(cd / p.name.replace(ROLE_FIGHTER_0, "TMP"))
    for p in cd.glob(f"{ROLE_FIGHTER_1}_*.jpg"):
        p.rename(cd / p.name.replace(ROLE_FIGHTER_1, ROLE_FIGHTER_0))
    for p in cd.glob("TMP_*.jpg"):
        p.rename(cd / p.name.replace("TMP", ROLE_FIGHTER_1))
    build_sheet(paths, video_id)


# --------------------------------------------------------------------------- picture for humans
ROLE_TITLES = {ROLE_FIGHTER_0: "FIGHTER 0", ROLE_FIGHTER_1: "FIGHTER 1", ROLE_REFEREE: "REFEREE",
               ROLE_OTHER: "NOT ASSIGNED (random detections the tool ignored: crowd / staff / unclear)"}
ROLE_COLORS = {ROLE_FIGHTER_0: (60, 60, 255), ROLE_FIGHTER_1: (255, 120, 40), ROLE_REFEREE: (60, 190, 60),
               ROLE_OTHER: (130, 130, 130)}


def build_sheet(paths: Paths, video_id: str) -> Path:
    data = read_json(paths.identities_json(video_id))
    crop_dir = paths.identity_crops(video_id)
    rows = []
    for role in [ROLE_FIGHTER_0, ROLE_FIGHTER_1, ROLE_REFEREE, ROLE_OTHER]:
        crops = [c for c in (cv2.imread(str(p)) for p in sorted(crop_dir.glob(f"{role}_*.jpg"))) if c is not None]
        info = data["roles"].get(role)
        if info:
            title = f"{ROLE_TITLES[role]} - seen in {info['n_frames']} frames (typical crops across the video)"
        elif role == ROLE_OTHER:
            title = ROLE_TITLES[role]
        else:
            title = f"{ROLE_TITLES[role]}: NOT FOUND in this video"
        strip = np.full((200, 1500, 3), 255, np.uint8)
        x = 10
        for c in crops:
            if x + c.shape[1] > 1490:
                break
            strip[34:34 + c.shape[0], x:x + c.shape[1]] = c
            x += c.shape[1] + 8
        cv2.rectangle(strip, (0, 0), (1500, 28), ROLE_COLORS[role], -1)
        cv2.putText(strip, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if info:
            r, g, b = info["shorts_rgb"]
            cv2.rectangle(strip, (1380, 4), (1400, 24), (b, g, r), -1)
            cv2.putText(strip, "shorts", (1405, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(strip)
    sheet = np.vstack(rows)
    cv2.imwrite(str(paths.identity_sheet(video_id)), sheet)
    return paths.identity_sheet(video_id)
