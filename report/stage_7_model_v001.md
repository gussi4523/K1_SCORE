# Stage 7 - Model v0.0.1: rule-based baseline (2026-11-18)

**Goal:** a complete but weak end-to-end base: skeletons to strikes to points to decision, to measure every later model against.

**Method (short):** strikes are fast deviations of a hand, foot or knee from its normal position relative to the body, directed at an opponent within reach. Hit / blocked / miss comes from the distance between the striking tip and the opponent's head, body, legs and guard (forearms, upper arms, raised shin). Hits are turned into round points and a decision with the 10-point must system. Full description: `docs/model_v0.0.1.md`.

**Result on real fights (counts, not accuracy):**

| Video | Strikes | Hit / blocked / miss / unknown |
|---|---|---|
| K-1 clip, 60 s | 24 | 7 / 10 / 7 / 0 |
| Muay Thai, 10 min | 224 | 43 / 96 / 82 / 3 |

**Status:** 34 tests pass. Accuracy is unknown, because no strike has been checked by hand. About 43% of strikes are labelled blocked, which is probably too many.

**Next:** hand-check about 50 strikes, tune the rules, train the first neural model that has to beat this baseline. Daily detail: `../daily_reports/2026-11-18.md`.
