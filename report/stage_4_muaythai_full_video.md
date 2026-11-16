# Stage 4 - Full 10-minute test video and tools to look at the result

## The test video
`https://youtu.be/sqrSVvNe8wY` - Muay Thai, Shimon vs Johan Ghazali, 598 s, 1080p, 25 fps, **14,943 frames**. Run with:

```bash
k1 ingest add "https://youtu.be/sqrSVvNe8wY" --run
```
Pose stage ran at about 25 frames per second, so the whole video took about 10 minutes on the GTX 1650.

## Result of v0.1 (before identities)

| Item | Value |
|---|---|
| Shots found | 22 (11 classified as ring shots, covering 14,927 of 14,943 frames) |
| Fighter 0 present / usable | 74% / 72% of frames |
| Fighter 1 present / usable | 45% / 41% of frames |
| Both fighters usable at once | **30%** |

Two causes were found:
1. frames rejected by the quality check had on average only about 7 visible joints (threshold was 10) - legs are often out of frame or hidden in tight shots;
2. fighter 1 was missing in about half of the frames because slot selection failed (this was later traced to the selection logic, not only to the pose model - see stage 5).

## Tools added so a human can look at the data
Because "it ran" says nothing about quality, three viewers were added:

| Command | What it gives |
|---|---|
| `k1 ingest debug <id> --start N --seconds S` | video file with skeletons drawn (`data/interim/<id>/debug.mp4`) |
| `k1 ingest preview <id>` | folder `data/interim/<id>/preview/` with `index.html`: 9 typical frames, 3 problem frames, contact sheet, **coverage timeline** (when is each fighter usable), summary numbers |
| `k1 ingest view <id>` | interactive window: slider, play/pause, step by frame / second / 10 s |

Colours: red = fighter 0, blue = fighter 1, grey = pose found but low quality. (Green = referee, added in stage 5.)

## Outcome of the human review
The user reviewed the pictures and reported the two problems that drive stage 5:
1. the referee must be defined separately from the fighters;
2. fighter labels must not change (0 <-> 1) from shot to shot.
