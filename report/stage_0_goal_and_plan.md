# Stage 0 - Goal and plan

## Request
Build a computer-vision project that scores K-1 kickboxing:
1. use **OpenCV** to extract a skeleton ("mesh") from the frames of fight videos,
2. build a **labeled dataset** from videos found on the web,
3. train a **PyTorch** classifier on the skeleton points (arms, legs, ...) that decides, for each strike, whether it was a **hit**, was **blocked**, or **missed** (a model with at least 4 layers).

## Decisions made at this stage

| Topic | Decision | Why |
|---|---|---|
| Skeleton | OpenCV for video reading and drawing, a **pose model** for keypoints | OpenCV alone cannot produce skeletons. YOLO-pose (Ultralytics) chosen: multi-person and has built-in tracking. MediaPipe was rejected (single person only, fails in clinches) |
| Classes | `hit`, `blocked`, `miss`, `none` | the three outcomes asked for, plus "no strike" background |
| Model | Temporal model on skeleton sequences: 1D-CNN + BiGRU (the "4 layers"), compared with a baseline, a small Transformer and ST-GCN | skeleton sequences are time series; start simple |
| Labeling | automatic strike *candidates* (velocity peaks toward the opponent), human labels each with a key press | much cheaper than labeling whole videos |
| Evaluation | split by **fight**, never by random window; report macro-F1 and confusion matrix | avoids leakage; classes are imbalanced |
| Legal | keep raw videos private, share only derived data | web videos are copyrighted |

## Pipeline from the plan

```
YouTube video -> frames (OpenCV) -> pose + tracking (2 fighters) -> normalized sequences
   -> candidate strikes -> human labels -> dataset -> PyTorch classifier -> hit / blocked / miss -> score
```

## Milestones from the plan

| # | Milestone | Status today |
|---|---|---|
| 1 | Environment + download videos | done |
| 2 | Pose extraction + overlay | done |
| 3 | Features + candidate strike detector | **not started** |
| 4 | Labeling tool + first ~500 labels | **not started** |
| 5 | Baseline + first model | **not started** |
| 6 | Scale labels (2-3k), try Transformer / ST-GCN | **not started** |
| 7 | Inference pipeline + scoring | **not started** |
| 8 | (If needed) RGB branch for hit vs blocked | **not started** |

## Main risks listed in the plan

- pose quality in clinches, occlusion and blur (this turned out to be the dominant problem so far, see stages 3-5)
- different camera angles across web videos
- subjective labels (hit vs blocked) and class imbalance
- small dataset
- 2D skeletons from one camera may not be enough to tell hit from blocked; an RGB branch may be needed

Full text: `../ARCHITECTURE.md`.
