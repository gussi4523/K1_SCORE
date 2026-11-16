# LinkedIn post drafts

## Main post (about 220 words)

🥊 Building an AI that scores kickboxing fights - day 1

I started a project: teach a computer to watch a K-1 / Muay Thai fight and decide, for every strike, whether it was a hit, a block or a miss.

Day 1 goal: turn a YouTube fight into clean training data.

What I built: a pipeline that downloads a fight, finds the camera cuts, runs pose estimation (YOLO11-pose) and outputs the skeletons of both fighters and the referee, frame by frame.

What went wrong (the interesting part):
1️⃣ The TV graphic with each fighter's portrait was detected as a "person" and picked as a fighter.
2️⃣ The referee kept taking a fighter's place.
3️⃣ My first fix, grouping tracks by clothing, failed. The tracker's ID is not a person: in clinches it jumps, so one track was the referee, then fighter A, then fighter B.
4️⃣ What worked: decide identity per detection, per frame (bare torso vs shirt, shorts colour), and give each role to at most one person per frame.

Result on a 10-minute fight: both fighters usable in 72% of frames (was 30%), referee tracked separately, fighters never swap. 21 automated tests.

Lesson I almost missed: I blamed the pose model for a "missing" fighter. He was detected all along; my selection logic had given his slot to the referee. Look at the raw data before blaming the model.

Next: a labeling tool, then a PyTorch classifier for hit / block / miss.

#ComputerVision #PyTorch #MachineLearning #PoseEstimation #BuildInPublic #MuayThai #Kickboxing

---

## Short version (about 90 words)

🥊 Day 1 of building an AI that scores kickboxing: from a YouTube fight to clean skeleton data.

Biggest lesson: the tracker's ID is not a person. In clinches it jumped between the referee and both fighters, so grouping tracks by clothing failed. Deciding identity per frame (bare torso vs shirt, shorts colour) worked: both fighters usable in 72% of frames, up from 30%.

Also: I blamed the model for a "missing" fighter, but my own selection logic had given his slot to the referee. Check the raw data first.

Next: labeling + a PyTorch hit/block/miss model.

#ComputerVision #PyTorch #BuildInPublic

---

## Picture suggestions (carousel)

1. `images/s5_timeline_after_identity.png` - the coverage chart (it is your own chart, no broadcast footage in it)
2. A skeleton-only picture (skeletons on a black background, no video) - can be generated
3. Before / after of the same frame: `images/s3_before_referee_in_fighter_slot.png` and `images/s5_after_same_frame_with_identities.png`
4. A simple text slide with the numbers: 30% -> 72%, 13 -> 21 tests

Check before posting: the fight frames are from broadcasts (K-1, ONE Championship) and show identifiable athletes. Broadcast footage is copyrighted; showing screenshots publicly can be a problem. The skeleton-only picture and the chart avoid this.
