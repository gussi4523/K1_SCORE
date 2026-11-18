# Model v0.0.1 - rule-based baseline (stages 2 and 3)

Date: 2026-11-18. Purpose: a weak but complete end-to-end base (skeletons -> strikes -> points -> decision) to
measure every later, better model against. **No training, no labels, no neural network yet.**
The result is an automated estimate, not an official score.

```
k1 strikes run <video_id>      # data/events/<id>/strikes.parquet + frames.parquet
k1 strikes samples <video_id>  # peak-frame pictures of random strikes (check by eye)
k1 strikes render <video_id>   # video with strikes + outcomes drawn on it
k1 score <video_id> [--round-seconds 180]   # data/scores/<id>/scores.json + report.txt
```

## Stage 2a - strike finder (`tools/strikes/finder.py`)

All distances are in **torso lengths** (shoulder centre to hip centre), so zoom does not matter.

1. Per ring shot, per fighter: drop joints below confidence 0.3, interpolate gaps up to 5 frames, smooth 3 frames.
2. For each limb tip (wrist, ankle) compute its position relative to its own root (shoulder / hip) and subtract its
   "rest" position (rolling median over 2 s). Body movement and camera pan cancel out.
3. A **strike** = a fast peak of that deviation (hand >= 0.55 torso at >= 3 torso/s; leg >= 1.2 torso at >= 4 torso/s),
   moving toward the opponent, while the opponent's centre is within 5 torso lengths.
4. **Knee** = knee lifted toward hip height with the shin hanging down; **kick** otherwise. Kick height from the foot
   vs the opponent's shoulders / hips: `kick_high`, `kick_mid`, `kick_low`. Punches are one class, `punch`.
5. Two detections of the same fighter within 0.2 s are merged (the stronger stays).

## Stage 2b - outcome (`tools/strikes/outcome.py`)

At the closest approach (peak +- 2 frames) of the tip to the defender:

| Quantity | Definition |
|---|---|
| target | nearest of head (mean of face joints), body (shoulder-hip line), legs (kicks only) |
| guard | forearms + upper arms, plus a **raised shin** (thigh more than 35 degrees from vertical = a check) |
| hit | tip within 0.5 torso of the target and not in front of a guard |
| blocked | tip within 0.35 torso of a guard that is (almost) in front of the target |
| miss | otherwise |

Distances become probabilities (`p_hit`, `p_blocked`, `p_miss`) with a sigmoid; `confidence` = the largest.
`unknown` = attacker or defender skeleton was not valid around the peak; such strikes are not scored.

## Strike table (`strikes.parquet`)

Columns are `STRIKE_COLUMNS` in `core/schema.py`: ids and frames, `attacker`, `limb`, `strike_type`, `target`,
`outcome`, `p_hit`, `p_blocked`, `p_miss`, `confidence`, `d_target`, `d_guard`, `pose_quality`, `label_source`
(`auto`), `model_version` (`v0.0.1`).
Differences from the plan in `pipeline_v1_scoring.md`: probabilities are three columns, there is no `type_probs`
(the type is a rule, not a model), and the type set is coarser (no jab / cross / hook / uppercut yet).

## Stage 3 - scoring (`tools/score/score.py`)

- Points per **hit**: punch 1, low kick 1, mid kick 2, high kick 2, knee 2; blocked 0 (placeholders in `DEFAULT_SCORING`).
- Strikes with `unknown` outcome or `pose_quality < 0.5` are not scored (the report says how many).
- Rounds: one round = whole video, unless `--round-seconds N` or `round_boundaries` is given (round detection is manual).
- 10-point must: more points = 10, the other 9 (8 when the gap is 15 points or more - a guess); equal = 10-10.
- Decision = higher total of round scores, else `draw`.

## First numbers (not accuracy - there are no labels yet)

| Video | Strikes found | Hit / blocked / miss / unknown |
|---|---|---|
| K-1 clip, 60 s (Takeru) | 24 | 7 / 10 / 7 / 0 |
| Muay Thai fight, 10 min | 224 | 43 / 96 / 82 / 3 |

A contact sheet of 12 random strikes from the K-1 clip showed the marks sitting on real punches. Whether the
outcomes are right is **not measured**: that needs a hand-checked set (the next step).

## Known weaknesses

- Blocked looks too common (43% of strikes on the Muay Thai fight): the defender's arms are almost always near the
  head, so the guard rule fires easily. Needs checking against hand labels.
- 2D has no depth; the glove is not in the skeleton; no reaction cue (head snap) and no sound yet.
- Punch types are not separated; clinch strikes are unreliable; strikes toward the camera give little motion.
- Thresholds are first guesses, tuned on nothing. Scoring weights and the 10-8 rule are placeholders.
- Strikes are dropped when the defender's skeleton is missing for more than 5 frames around the strike.

## Next

1. Hand-check about 50 strikes with a small viewer; measure agreement per outcome and type.
2. Tune thresholds on that set, then train a PyTorch model that must beat v0.0.1.
3. Return to stage 1 and raise the share of frames with both fighters usable.
