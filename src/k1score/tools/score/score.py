"""Stage 3 (v0.0.1): strikes.parquet + scoring config -> points per round, 10-point-must round scores, decision.

The result is an AUTOMATED ESTIMATE from weak rule-based strikes, not an official score. Weights are placeholders.
"""
from __future__ import annotations

import pandas as pd

from k1score.core.schema import MODEL_VERSION, STRIKE_TYPES

DISCLAIMER = ("Automated estimate from rule-based strike detection (model %s). Not an official score: damage, "
              "knockdowns, fouls and ring control are not seen, and point weights are placeholders." % MODEL_VERSION)
FIGHTERS = ("fighter_0", "fighter_1")


def make_rounds(duration_s: float, cfg: dict) -> list[tuple[float, float]]:
    if cfg["round_boundaries"]:
        return [(float(a), float(b)) for a, b in cfg["round_boundaries"]]
    if cfg["round_seconds"]:
        rs = cfg["round_seconds"]
        n = max(1, int(-(-(duration_s - 1.0) // rs)))          # a tail shorter than 1 s is not a round of its own
        return [(i * rs, duration_s if i == n - 1 else (i + 1) * rs) for i in range(n)]
    return [(0.0, duration_s)]


def strike_points(row, cfg: dict) -> float:
    if row.outcome == "hit":
        return float(cfg["points_per_hit"].get(row.strike_type, 0.0))
    if row.outcome == "blocked":
        return float(cfg["points_per_blocked"])
    return 0.0


def round_score(p0: float, p1: float, cfg: dict) -> tuple[int, int]:
    """10-point must: winner 10, loser 9 (8 when the gap reaches margin_for_10_8); equal points = even round."""
    if p0 == p1:
        return (10, 10) if cfg["allow_even_round"] else (10, 9)
    loser = 8 if abs(p0 - p1) >= cfg["margin_for_10_8"] else 9
    return (10, loser) if p0 > p1 else (loser, 10)


def score_video(strikes: pd.DataFrame, meta: dict, cfg: dict) -> dict:
    duration = meta["n_frames"] / meta["fps"]
    usable = strikes[(strikes.outcome != "unknown")
                     & (strikes.confidence >= cfg["min_confidence"])
                     & (strikes.pose_quality >= cfg["min_pose_quality"])]
    skipped = len(strikes) - len(usable)
    rounds, totals, total_pts = [], {f: 0 for f in FIGHTERS}, {f: 0.0 for f in FIGHTERS}

    for i, (a, b) in enumerate(make_rounds(duration, cfg), start=1):
        in_round = usable[(usable.t_peak >= a) & (usable.t_peak < b + (1e-9 if b >= duration else 0))]
        entry: dict = {"round": i, "start_s": round(a, 2), "end_s": round(b, 2)}
        pts = {}
        for f in FIGHTERS:
            s = in_round[in_round.attacker == f]
            p = float(sum(strike_points(r, cfg) for r in s.itertuples()))
            pts[f] = p
            entry[f] = {"strikes": int(len(s)), "hits": int((s.outcome == "hit").sum()),
                        "blocked": int((s.outcome == "blocked").sum()), "misses": int((s.outcome == "miss").sum()),
                        "hits_by_type": {t: int(((s.outcome == "hit") & (s.strike_type == t)).sum())
                                         for t in STRIKE_TYPES}, "points": p}
        entry["round_score"] = list(round_score(pts[FIGHTERS[0]], pts[FIGHTERS[1]], cfg))
        best = in_round[in_round.outcome == "hit"].assign(
            pts=[strike_points(r, cfg) for r in in_round[in_round.outcome == "hit"].itertuples()]
        ).sort_values(["pts", "confidence"], ascending=False).head(5)
        entry["top_strikes"] = [{"strike_id": int(r.strike_id), "t": round(float(r.t_peak), 2), "attacker": r.attacker,
                                 "type": r.strike_type, "confidence": round(float(r.confidence), 2)}
                                for r in best.itertuples()]
        rounds.append(entry)
        for k, f in enumerate(FIGHTERS):
            totals[f] += entry["round_score"][k]
            total_pts[f] += pts[f]

    if totals[FIGHTERS[0]] == totals[FIGHTERS[1]]:
        decision = "draw"
    else:
        decision = FIGHTERS[0] if totals[FIGHTERS[0]] > totals[FIGHTERS[1]] else FIGHTERS[1]
    shorts = {f: meta.get("identities", {}).get("roles", {}).get(f, {}).get("shorts_rgb") for f in FIGHTERS}
    return {"video_id": meta["video_id"], "model_version": MODEL_VERSION, "disclaimer": DISCLAIMER,
            "fighter_shorts_rgb": shorts, "strikes_total": int(len(strikes)), "strikes_not_scored": int(skipped),
            "rounds": rounds,
            "total": {"score": totals, "points": total_pts},
            "decision": decision}


def format_report(s: dict) -> str:
    lines = [f"K1 Score {s['model_version']} - {s['video_id']}", ""]
    lines.append(f"{'round':>5} {'time':>13} | {'F0 str/hit/blk/miss':>20} {'pts':>5} | {'F1 str/hit/blk/miss':>20} {'pts':>5} | score")
    for r in s["rounds"]:
        def cell(f):
            d = r[f]
            return f"{d['strikes']:>3}/{d['hits']:>3}/{d['blocked']:>3}/{d['misses']:>3}".rjust(20), f"{d['points']:>5.1f}"
        (c0, p0), (c1, p1) = cell("fighter_0"), cell("fighter_1")
        lines.append(f"{r['round']:>5} {r['start_s']:>6.0f}-{r['end_s']:<6.0f} | {c0} {p0} | {c1} {p1} | "
                     f"{r['round_score'][0]}-{r['round_score'][1]}")
    t = s["total"]["score"]
    lines += ["", f"total {t['fighter_0']}-{t['fighter_1']}  ->  decision: {s['decision']}",
              f"({s['strikes_not_scored']} of {s['strikes_total']} strikes not scored: unknown outcome / low quality)",
              "", s["disclaimer"]]
    return "\n".join(lines)
