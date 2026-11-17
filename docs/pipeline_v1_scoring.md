# Scoring pipeline v1 - plan (3 stages)

Date: 2026-11-17. Status: plan, nothing of stages 2 and 3 is built yet.

## Decisions taken (from the planning questions)

| Topic | Decision |
|---|---|
| Scoring rules | **K-1 rules**, implemented as a configurable file (see section 5) |
| Stage 2 output | **one record per strike** (start/end frame, attacker, type, outcome, confidence) **plus a per-frame lookup table** derived from it |
| Datasets | **BoxingVI** (punch types from skeletons) **+ our own automatic outcome labels** (hit / blocked / miss), checked on a small human-verified set |
| Strike types in v1 | punches (jab, cross, hook, uppercut), kicks (low, middle, high), knees |

## 1. Overview

```
 video
   |
   v
 STAGE 1  ingest (built)          video -> skeleton frames (fighter 0, fighter 1, referee, quality)
   |         data/processed/<id>/poses.npz + meta.json
   v
 STAGE 2  strike analysis         skeleton frames -> list of strikes: when, who, type, hit / blocked / miss
   |         data/events/<id>/strikes.parquet  (+ frames.parquet derived)
   v
 STAGE 3  scoring                 strikes + rules -> points per round, per fighter, winner
             data/scores/<id>/scores.json  (+ readable report)
```

Each stage reads only the previous stage's files, so stages can be developed, tested and replaced independently (same principle as tool 01).

## 2. Stage 2 - strike analysis

Why one record per strike: a strike lasts about 10-30 frames. Labelling every frame would make stage 3 rebuild strikes from frames and risk counting one strike many times.

Three parts (they can share one model with several outputs later):

| Part | Task | v1 method |
|---|---|---|
| 2a Strike finder | *when* does a strike happen, which limb | automatic from skeleton motion: fast wrist / ankle / knee moving toward the opponent; no training needed, so it also produces candidates for labeling |
| 2b Type classifier | jab / cross / hook / uppercut / low-mid-high kick / knee | PyTorch model on a window of skeleton frames; punches pretrained on BoxingVI, kicks and knees from our own labels |
| 2c Outcome classifier | hit / blocked / miss | PyTorch model on a window of **both** fighters' skeletons (distance to head and body, opponent's guard, opponent's reaction); trained on automatic labels plus a human-verified set |

### 2.1 Strike record (`strikes.parquet`, one row per strike)

| Column | Meaning |
|---|---|
| `strike_id` | running number inside the video |
| `video_id`, `shot_id` | source |
| `start_frame`, `peak_frame`, `end_frame` | `peak_frame` = moment of full extension / contact |
| `t_start`, `t_peak`, `t_end` | the same in seconds |
| `attacker` | `fighter_0` / `fighter_1` |
| `limb` | `left_hand`, `right_hand`, `left_leg`, `right_leg`, `left_knee`, `right_knee` |
| `strike_type` | `jab, cross, hook, uppercut, kick_low, kick_mid, kick_high, knee` |
| `type_probs` | probabilities per type (list) |
| `outcome` | `hit` / `blocked` / `miss` |
| `outcome_probs` | `p_hit, p_blocked, p_miss` |
| `confidence` | max probability after calibration |
| `pose_quality` | share of valid skeleton frames in the window (low = unreliable) |
| `label_source` | `auto`, `model`, `human` |
| `model_version` | which model produced the row |

### 2.2 Per-frame table (`frames.parquet`, derived, one row per frame)

| Column | Meaning |
|---|---|
| `frame`, `shot_id` | position |
| `f0_strike_id`, `f1_strike_id` | strike the fighter is performing in this frame, or null |
| `f0_outcome`, `f1_outcome` | outcome of that strike, or null |
| `valid_f0`, `valid_f1` | from stage 1 |

This table is only for lookup and display (overlay video, timeline). **Scoring uses `strikes.parquet`.**

## 3. Stage 3 - scoring

Input: `strikes.parquet`, a scoring config, and round boundaries.

Output `scores.json`:
```
rounds: [ { round, start_frame, end_frame,
            fighter_0: { strikes, hits, blocked, misses, hits_by_type, points },
            fighter_1: { ... },
            round_score: [10, 9], notes } ],
total: { fighter_0: ..., fighter_1: ... },
decision: winner | draw (extra round needed) ,
explanation: list of the strikes that drove each round
```
The result is an **automated estimate**, not an official score. Real judges also weigh damage, knockdowns, aggression and ring control, which a skeleton cannot fully see.

## 4. Datasets

| Source | Role in v1 | To check first |
|---|---|---|
| [BoxingVI](https://arxiv.org/abs/2511.16524) | punch-type pretraining (6,915 clips from 20 YouTube videos, 2D keypoints from AlphaPose) | licence, download location, **the exact punch classes** (search summaries disagree: "six types" vs four named: jab, cross, hook, uppercut - read the paper), keypoint format (needs a converter to our COCO-17 layout), single boxer only, sparring/shadow-boxing domain vs real fights |
| Our fights (stage 1 output) | kicks, knees, outcomes, real-fight domain | amount of verified labels |
| [MMA pose dataset](https://data.mendeley.com/datasets/c456bnk8bm) | optional: fine-tune the pose model to raise skeleton coverage | licence |
| BoxMAC | maybe block labels | availability unconfirmed |

No public dataset with hit / blocked / miss for kickboxing was found, so **outcome labels come from us**: automatic first guess (distance of limb to head/body at the peak, whether the opponent's hands are up, the opponent's reaction, optionally an impact sound from the audio) plus a human-verified set from fights not used for training. Without that verified set the model's accuracy cannot be known.

## 5. Scoring rules (K-1), as configuration

What is well supported by sources: 3-round bouts and the 10-point must system (round winner 10, loser 9 or less, even round possible). Knockdown scoring (10-8) and extra-round procedure come from other promotions or other rulesets and **must be checked against the current official K-1 rulebook** before use.

`configs/scoring.yaml` (planned):
```
rounds: 3
round_seconds: 180
points_per_hit: {jab: 1, cross: 1, hook: 1, uppercut: 1, kick_low: 1, kick_mid: 2, kick_high: 2, knee: 2}   # starting guess, editable
points_per_blocked: 0
margin_for_10_8: 15         # point difference that makes a 10-8 round (starting guess)
allow_even_round: true
extra_round_on_draw: true   # verify against official rules
```
Weights are placeholders to tune; they are not official values.

## 6. Open problems (not solved by the plan)

| Problem | Note |
|---|---|
| **Round boundaries** | the video may be a full fight or highlights; v1: round start/end frames given in a config; later detect from the clock graphic or from rest periods |
| **Knockdowns** | cannot be derived from strikes alone; later detect "fighter on canvas" from pose, or enter manually |
| **Fouls / point deductions** | not detectable; manual input |
| **Clinch strikes (knees)** | skeletons overlap and tracking suffers; expect lower quality |
| **2D single camera** | hit vs blocked may not separate from skeletons alone; an image-crop branch is the fallback |
| **Fighter hidden behind the other** | skeleton missing, strike `pose_quality` low |

## 7. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Strike finder + `strikes.parquet` / `frames.parquet` schema + tests | the table is produced for our two fights and the strikes look right in the viewer |
| M2 | Dataset check and BoxingVI import with converter | licence confirmed, punch-type baseline trained |
| M3 | Automatic outcome labels + labeling viewer + human-verified set | verified set exists on held-out fights |
| M4 | Stage 2 model (type + outcome) | metrics on held-out fights (macro-F1, confusion matrix) |
| M5 | Stage 3 scoring + `scores.json` + readable report | round table and decision produced from `strikes.parquet` |
| M6 | One command: video to decision (`k1 score <video>`) | end to end on a full fight |

## 8. Folder plan

```
src/k1score/tools/
|-- ingest/     stage 1 (built)
|-- strikes/    stage 2: finder, features, models, labeling viewer
|-- score/      stage 3: rules, rounds, report
data/events/<id>/   strikes.parquet, frames.parquet
data/scores/<id>/   scores.json
configs/scoring.yaml
```
