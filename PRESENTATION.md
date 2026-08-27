## The test clips

Six videos are used throughout. Every number in this document names the clip it came from.

| clip            | what it is                                                            |
| --------------- | --------------------------------------------------------------------- |
| `messi.mp4`   | 337 frames, 24 shots — live football footage with burned-in captions |
| `anime.mp4`   | 437 frames, 7 shots — animated football                              |
| `patrick.mp4` | 326 frames, 1 shot — handheld street video, crowded pavement         |
| `thanos.mp4`  | 292 frames, 4 shots                                                   |
| `cat1.mp4`    | 617 frames, 1 shot — a cat, and a human hand entering frame          |
| `cat2.mp4`    | 696 frames, 4 shots — a cat, no people at all                        |

Most examples below come from **`messi.mp4`**.

---

## 01 — Sampling

**What it does.** Turns the video into individual picture files, one per frame.

**Models and libraries.** OpenCV only.

**Input.** The `.mp4` file.
**Output.** A folder of images, plus a note of how many there were.

**Status:** nothing else in the pipeline uses this. Module 02 opens the video itself,
which was measured **9x** than reading these images back — on
`patrick.mp4`, 0.31s against 2.9s — *and* more accurate, because saving to JPEG loses a little
detail that changes what the detectors find later. This module survives only as something a
human can flick through.

---

## 02 — Segmentation

**What it does.** Finds the cuts — the points where the camera switches to a different shot —
and picks a few representative still frames from each shot.

**Models and libraries.**

- **OmniShotCut** (`uva-cv-lab/OmniShotCut`) — finds the cuts.
- **MobileCLIP-S2** (Apple, via the `open_clip` library) — turns each frame into a numeric
  summary, so frames can be compared to each other.
- **OpenCV** — decoding and sharpness measurement.

**Input.** The video.
**Output.** A list of shots with start and end frames, and roughly three chosen still images
per shot ("keyframes").

**On `messi.mp4`:** 24 shots, 66 keyframes.

**Why keyframes matter.** Almost everything later works on these few images rather than all
337 frames. Choosing them well is therefore important. The module splits each shot into equal
time slices and takes one image from each, preferring the sharpest and most typical.

---

## 03 — Profiler

**What it does.** Takes a quick, cheap look at each shot and answers four yes/no questions:
is there a **person**? a **face**? any **text**? any **object**?

**Models and libraries.**

- **YOLO11** (via the `ultralytics` library) — finds people and 80 kinds of object. RT-DETR is
  available as an alternative.
- **YuNet** (from OpenCV's model zoo) — finds faces.
- **PP-OCRv3's text detector** — finds *where* text is, not what it says. Used through
  OpenCV.

**Input.** The keyframes from 02.
**Output.** Four answers per shot, plus supporting detail — how many people, which object
types, and how many of the keyframes agreed.

**On `messi.mp4`:** person yes, face yes, text yes, object yes.
**On `cat2.mp4`:** no people at all — the four answers are what let later modules skip it.

**How it uses 02.** It only ever looks at 02's keyframes, never the video, so it costs almost
nothing. Faces are searched for *only inside* the boxes where people were found, which is what
lets it tell "a face is visible" apart from "a body is visible" without a second sweep of the whole image.

Those four answers are the entire handoff to the next module. Deliberately no object *names* —
naming things there would mean writing an arbitrary rulebook mapping 80 object types to
analyses, which is brittle. The names are kept alongside as evidence a human can inspect.

---

## 04 — Router

**What it does.** Decides which analyses to run on each shot. This is the module that saves
all the resources.

**Models and libraries.** **None.** Plain Python rules.

**Input.** The four answers from 03.
**Output.** A per-shot list of which analyses are switched on, which are switched off, **the
rule that decided each one, and the evidence behind it.**

**On `messi.mp4`:** 14 analyses. The scene-level ones fire on all 24 shots, the people-related
ones on 23, object masks on only 12.

**Why it is written down so carefully.** Every decision — including the *negatives* — is
recorded with its reason. We can open the file months later and see exactly why a shot skipped
face analysis, without re-running anything. Running it twice on the same input gives an identical file.

A shot with no people skips the entire human pipeline: no face analysis, no body analysis, no
tracking. On `anime.mp4`, shot 3 is exactly that case. That is the saving the whole
architecture exists to produce.

---

## 05 — Global experts (the always-on)

**What it does.** Describes the *scene*, regardless of who is in it. Eight analyses:

| analysis                                                  | model                                             |
| --------------------------------------------------------- | ------------------------------------------------- |
| depth — how far away each part of the image is           | **Depth Anything V2 Small**                 |
| surface angles — which way surfaces face                 | derived from the depth, no model                  |
| edges — the outline drawing of the image                 | **PiDiNet** (via `controlnet_aux`)        |
| place type — "soccer field", "kitchen", "street"         | **Places365 ResNet-18**                     |
| tags — free labels like "referee", "penalty kick"        | **MobileCLIP** against RAM's 4,585-tag list |
| semantic regions — which pixels are sky, road, person    | **SegFormer-B1** (ADE20K)                   |
| object regions — separated object shapes                 | **Mask2Former** (Swin-Tiny, COCO)           |
| classical measures — brightness, contrast, colourfulness | OpenCV and NumPy, no model                        |

**Input.** 02's keyframes.
**Output.** Picture-shaped maps for the visual ones, plus numbers and labels.

**On `messi.mp4`:** all 66 keyframes, five maps each.

**How it uses the router.** It doesn't — and that is the point. These describe the scene, not
people, so there is nothing to gate them on. They run everywhere, always.

**A note on the tags.** RAM++ was the obvious model here and was rejected: it is 2.87 GB,
larger than everything else combined. Since 02 already produced a numeric summary of each
keyframe, tagging became a matrix multiplication against a list of words — no extra model, no
extra download.

---

## 06 — Detection and tracking (the first gated module)

**What it does.** Follows each person through a shot and gives them a number, so that
everything measured later can be attached to the right individual.

**Models and libraries.**

- **YOLO11** (`ultralytics`) — finds people, frame by frame. *Reuses 03's downloaded weights.*
- **ByteTrack** — joins those detections into continuous tracks. BoT-SORT is available as an
  alternative.
- **YuNet** — finds a face inside each tracked body. *Also reused from 03.*

**Input.** The router's decisions, plus the video itself.
**Output.** For every person, a box on every frame, plus a handful of the best-quality close-up
crops of their face and body.

**On `messi.mp4`:** 88 people-tracks, from reading only 176 of the 337 frames.

**How it uses the router.** This is the **first** module that reads the router's file. It skips
un-gated shots entirely — it does not even decode those frames.

**Three things worth knowing:**

*Faces come from inside bodies.* Rather than run a separate face-finder, it looks for a face
inside each already-tracked body. The face therefore automatically belongs to that person, and
no separate step is needed to work out whose face it is.

*The tracker restarts at every cut.* Following someone across a cut is meaningless — the camera
has jumped somewhere else. So person "1" in shot 3 and person "1" in shot 17 of `messi.mp4` are
unrelated people.

*It only looks every other frame.* Detection runs on every second frame and positions in
between are filled in by drawing a straight line. Where the gap is bigger than that — meaning
the tracker genuinely lost someone — **no position is invented at all.** The file records those
gaps rather than papering over them.

---

## 07 — Conditional experts

**What it does.** The module that actually looks at people closely. Face shape, which way the
head is turned, expression, emotion, body posture, and reading text on screen.

**Models and libraries.**

| what                                       | model                                                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| face shape, head direction, expression     | **MediaPipe FaceLandmarker** — one pass gives 478 face points, 52 expression values, and head direction together |
| body posture                               | **MediaPipe PoseLandmarker** — 33 skeleton joints                                                                |
| emotion, and positive/negative feeling     | **HSEmotion** (EfficientNet-B0)                                                                                   |
| face fingerprint                           | **ArcFace** from InsightFace's `buffalo_l` release                                                              |
| approximate age and apparent gender        | InsightFace's`genderage` model                                                                                        |
| appearance attributes (glasses, hair, hat) | **MobileCLIP**, reusing 02's download                                                                             |
| reading text                               | **CRNN** from OpenCV's model zoo                                                                                  |

Run through `mediapipe`, `onnxruntime` and OpenCV.

**Input.** 06's tracks and crops, plus the router's decisions.
**Output.** For each person, a series of measurements over time, plus a numeric fingerprint of
their face.

**On `messi.mp4`:** 140 measurements over time, 77 face fingerprints, 313 pieces of text read.
**On `cat1.mp4`:** 8 body-posture measurements and **zero** face measurements — the only
"person" there is a hand, so there is never a face to measure.

**It works at two speeds, on purpose:**

- Things that *change* — head direction, expression, posture — are measured repeatedly through
  the shot, because the change is the interesting part.
- Things that *don't change* — approximate age, apparent gender, whether someone wears glasses,
  their face fingerprint — are measured only on the handful of best crops 06 already picked.

**The face fingerprint** is the important output for later. It is 512 numbers describing a
face. On `messi.mp4`, two crops of the same person score high similarity score and two crops of
different people score low similarity score. That gap is what will eventually let the system say "this is
the same person as in the earlier shot".

**Reading text needed a workaround.** The text reader handles about ten characters at a time,
but `messi.mp4`'s captions are full sentences. Fed a whole line it produced `'conitasks'`. Read
in chunks it produces `'yourw orldcu pcare erlsso'` — recognisably "Your World Cup Career Is
So".

---

## 08 — Aggregation

**What it does.** Turns hundreds of individual measurements into a few statements per person.

**Models and libraries.** **None.** NumPy arithmetic only..

**Input.** 07's measurements and 05's scene descriptions.
**Output.** One record per person per shot, plus one record for the shot itself.

**On `messi.mp4`:** 53 person records.

Instead of "at frame 29 happy, at frame 33 neutral, at frame 37 neutral, at frame 41 neutral"
you get "mostly neutral, agreement 75%, head turned from −15° to +3°" — a real record from
shot 1 of `messi.mp4`.

**The genuinely new number is variance** — how *much* something moved. A photograph can tell
you someone looks calm. Only video can tell you whether they *stayed* calm or swung wildly.
Same average, completely different person. That measure does not exist until something looks
across time, and this is that something.

**It refuses to invent confidence.** If a person was only measured twice, variance is reported
as "unknown" rather than as a number. On `messi.mp4`, 8 of the 53 records are that thin, and
the file says so.

---

## 09 — Fusion

**What it does.** Describes the relationships *between* people — facts that no single person's
record can contain.

**Models and libraries.** **None.** NumPy arithmetic only.

**Input.** 08's records and 06's positions.
**Output.** For every pair of people on screen together: who is where relative to whom, who is
facing whom, and whether their emotions move together.

**On `messi.mp4`:** 71 pairs on screen together, 51 with a directional "attends to" link, 3 of
them mutual.

**Two things are named honestly:**

*It is "attention", not "gaze".* The system measures **which way the head is turned**, not
where the eyes are pointing. Those are different things and there is no eye-tracking model
here, so it is named for what it actually measures.

*Emotional synchrony usually refuses to answer.* Correlating two people's emotional ups and
downs needs enough measurements of both. On `messi.mp4` **no pair has enough**, so the system
reports "not available" 71 times rather than producing 71 meaningless numbers. On `anime.mp4`,
whose shots are longer, it produces **13 real ones**. This is a limitation of short, fast-cut
footage, and the file states it rather than hiding it.

---

## How the modules connect

```
video
  └─ 02 shots + keyframes          OmniShotCut, MobileCLIP
       └─ 03 four yes/no answers   YOLO11, YuNet, PP-OCRv3
            └─ 04 which analyses to run          (no models)
                 ├─ 05 scene description         Depth Anything, PiDiNet,
                 │                               Places365, SegFormer, Mask2Former
                 └─ 06 who is where              YOLO11 + ByteTrack, YuNet
                      └─ 07 what they look like  MediaPipe, HSEmotion,
                           │                     ArcFace, CRNN
                           └─ 08 summarise each person      (no models)
                                └─ 09 relate people          (no models)
```

Each module writes one file and reads the files before it. Any module can be re-run on its own
without redoing the whole pipeline, and every intermediate result can be opened and checked.
