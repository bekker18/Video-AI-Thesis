# Project State

Current state of the codebase. Read this first for full context.

## What this is

A modular video processing pipeline for a thesis. Input is `.mp4`. The pipeline is a
sequence of numbered modules; each writes to `data/processed/<video_name>/` and later
modules read what earlier ones wrote.

Planned modules: `01-sampling`, `02-segmentation`, `03-profiler`, `04-router`,
`05-global-experts`, `06-detection-tracking`, `07-conditional-experts`, `08-consolidation`,
`09-aggregation`, `10-fusion`, `11-vllm-synthesis`, `12-video-identikit`.

An `03-association` module (detection + tracking) was specified, designed, then dropped
before any code was written. The remaining modules were renumbered to close the gap, so
what HISTORY.md calls `04-profiler` is now `03-profiler`. Detection and tracking returned
later as `06-detection-tracking`, on the gated side of the router rather than ahead of it —
which is the placement that makes it affordable.

**Implemented: 01 through 10.** Everything from 11-vllm-synthesis onward is unwritten; see
[PLANNED.md](PLANNED.md) for what those modules are meant to do.

**Consolidation was specified as 10 and built as 08.** It resolves segment-local track ids
into global person ids, and it needs only 06's tracks and 07's embeddings — nothing from
aggregation. Putting it there lets 09 and 10 work at person scope directly, so every statistic
is computed once, from raw samples, over whole people. See HISTORY.md for the alternative that
was rejected and why.

04-router is the only consumer of `profile.json`'s `flags`, which is why that block is a
flat predicate. **06-detection-tracking is the first consumer of `routing.json`**; 05 does
not read it, because the plastic experts are ungated by definition.

## Conventions

- **Python 3.10, no local interpreter — everything runs in Docker.**
- `pyright` at standard rules must report no errors. It is a dev dependency in
  `pyproject.toml`, so `docker compose run --rm --entrypoint pyright pipeline` just works.
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; the image runs
  `uv sync --frozen`. There is no requirements.txt.
- Code style: simple, minimal comments, small docs. No speculative abstraction.
- **Frame indices are 0-based everywhere.** `frames/000033.jpg` is the frame the json
  calls `33`, and is the 34th frame of the video. This was a bug once; do not reintroduce it.

## Layout

```
data/raw/<video>.mp4                input
data/processed/<video>/frames/      decoded frames            (01)
data/processed/<video>/shots/       first frame of each shot  (02)
data/processed/<video>/keyframes/   representative frames     (02)
data/processed/<video>/*.json       per-stage metadata
data/processed/<video>/routing.json active expert set per shot   (04)
data/processed/<video>/global/      dense maps per keyframe      (05)
data/processed/<video>/global.json  scene, tags, descriptors     (05)
data/processed/<video>/tracks/      best crops per track          (06)
data/processed/<video>/tracks.json  per-frame track table         (06)
data/processed/<video>/conditional.json  per-track series + text  (07)
data/processed/<video>/face_embeddings.npy  ArcFace, one row per face crop (07)
data/processed/<video>/body_embeddings.npy  ReID, one row per body crop   (07)
data/processed/<video>/consolidated.json  global person ids            (08)
data/processed/<video>/aggregated.json  per-person + per-segment records (09)
data/processed/<video>/fused.json   persons and their relations   (10)
models/<stage>/                     weights for that stage
models/torch/                       torch hub cache (TORCH_HOME)
src/<stage>/<name>.py               stage implementation, exposes run(cfg)
config.py                           Config dataclass passed to every stage
download_models.py                  weight registry + downloader
main.py                             orchestrator
preview.py                          renders 06, 07, 08 and 10 back over the video
```

## How the orchestrator works

`main.py` holds `STAGES` in pipeline order. For each requested stage it builds a frozen
`Config`, calls `download_models.ensure(stage)`, then loads the stage module **by file
path** with `importlib` and calls `run(cfg)`.

Stage files are named after the stage minus its number, with hyphens turned into
underscores: `01-sampling` → `src/01-sampling/sampling.py`, `06-global-experts` →
`src/06-global-experts/global_experts.py`. This is deliberate — directory names like
`01-sampling` are not valid Python module names, and eleven files all called `module.py`
would collide when checked together.

Requested stages are always sorted into pipeline order, and unknown names are rejected
up front.

To add a module: create `src/<stage>/<name>.py` exposing `run(cfg: Config) -> dict[str, Any]`,
add the stage to `STAGES`, add any config fields to `config.py`, and register weights in
`download_models.py`.

## Models

Weights live in `models/<stage>/`. `download_models.py` has the registry and skips anything
already on disk. `models/` is bind-mounted, so weights survive rebuilds.

Two kinds of weights are handled:

- **`MODELS`** — Hugging Face repos, fetched with `hf_hub_download` (single file) or
  `snapshot_download` (whole repo).
- **`BACKBONES`** — weights the model libraries fetch themselves via torch hub.
  `TORCH_HOME=/app/models/torch` is set in the Dockerfile so these persist too.
  Without it, OmniShotCut re-downloaded resnet18 (44.7 MB) on every single run.

Stages are given `cfg.model_dir` and pass explicit paths / `cache_dir` to the libraries.
`HF_HOME` is deliberately **not** used — explicit paths avoid ambiguity about where things land.

## 01-sampling

Decodes the video into frames with OpenCV and writes `frames/` + `sampling.json`.

`--fps` drops frames by **stride**, not timestamp, so the effective rate is
`source_fps / round(source_fps / target)`. It uses `grab()` to skip frames without
paying for colour conversion.

**Nothing currently consumes its output.** 02 decodes the video itself. 01 survives only
as an inspectable artifact. Whether to delete it, or make it opt-in, is an open decision —
it was originally justified by dense-frame tracking in 03-association, which no longer exists.

## 02-segmentation

Reads the video **directly**; does not need 01.

1. Loads OmniShotCut and reads its working resolution (128×96).
2. **One decode pass** (`_scan`). Each frame is downscaled once to the MobileCLIP crop size,
   and that single intermediate feeds three consumers: a 128×96 array for OmniShotCut,
   a MobileCLIP-S2 embedding, and a Laplacian sharpness score. Full frames are never
   accumulated, so memory stays flat regardless of video length.
3. Shot detection on the accumulated array.
4. `_normalise` makes ranges disjoint and in-bounds — the model reports a *shared* frame
   between neighbours (`[0,33]` then `[33,108]`) and can run one index past the end.
5. Keyframe selection per shot (below).
6. **Second decode pass** (`_write_frames`) writes only the ~25 selected frames at full
   resolution. It exists because you cannot know which frames to keep until the embeddings
   exist, and `_scan` has discarded the pixels by then.

**Keyframe selection.** Split the shot into `--keyframes` equal time spans, take one frame
from each. Within a span, rank candidates by cosine similarity to that span's mean
embedding, then pick the **sharpest** of the top 25%, skipping anything with cosine > 0.98
to an already-chosen keyframe. The temporal split is what stops all keyframes landing on
one instant; sharpness avoids motion-blurred picks. Fewer than `k` keyframes is normal and
correct for short or static shots.

**Outputs:** `shots/0000.jpg`, `keyframes/0000_000123.jpg` (`<shot>_<frame>`),
`segmentation.json`, `keyframe_embeddings.npy`.

### Non-obvious things about 02

- **`--shot-overlap` defaults to 30, not the model's own 20.** Feeding a decoded array
  instead of a file path makes OmniShotCut miss a real cut at overlap 20, because OpenCV
  decodes 437 frames where ffmpeg's raw pipe emits 438, which shifts window alignment.
  Overlap 30 recovers it. Verified across three clips. Do not lower this without re-testing.
- **`--shot-mode` defaults to `default`, not `clean_shot`.** `clean_shot` silently keeps
  only ranges labelled `general` and discards dissolves, wipes and fades, under-reporting
  shot count with no warning.
- **`_preprocess_spec` reads crop size and normalisation out of open_clip's transform**
  rather than hardcoding them, because preprocessing is reimplemented in OpenCV for speed.
  If the CLIP model is swapped, this keeps working.
- MobileCLIP-S2's `Normalize` happens to be identity (mean 0, std 1) — do not rely on that.

## 03-profiler

Reads `segmentation.json` + `keyframes/` from 02. **Decodes no video** — the keyframes are
already on disk at full resolution.

Produces the presence flags 04-router branches on. Four cheap always-on detectors:

1. **Objects and persons** — one ultralytics forward pass over *all* keyframes batched
   together, 80 COCO classes. `--detector` switches between YOLO11 n/s/m and RT-DETR-L;
   both load through ultralytics so the comparison is fair.
2. **Faces** — YuNet, run **only inside returned person boxes**, never on the full frame.
   This is what makes face presence distinguishable from body presence without a second
   full-frame inference. Boxes are mapped back to frame coordinates.
3. **Text** — PP-OCRv3's DB **detection stage only**, via `cv2.dnn_TextDetectionModel_DB`.
   No recognition head, so it answers "is there text" not "what does it say".
4. **Scene** — zero-shot against Places365's 365 category names, using the MobileCLIP
   embeddings 02 already saved in `keyframe_embeddings.npy`. Costs one text-encoder pass
   over the vocabulary and a matrix multiply. No scene model is downloaded.

**Aggregation:** a flag fires if **any** keyframe saw it. `counts` carries the maximum per
keyframe and `agreement` the fraction of keyframes that agreed, so the router can threshold
differently later without re-running anything.

**`flags` is the entire contract 04-router reads** — four booleans, no class names:

```json
"flags":     { "person": true, "face": true, "text": true, "object": true },
"counts":    { "person_max": 3, "face_max": 2, "object_max": 1,
               "objects": { "banana": 1, "baseball glove": 1 } },
"agreement": { "person": 1.0, "face": 1.0, "text": 1.0, "object": 1.0,
               "objects": { "banana": 0.333, "baseball glove": 0.667 } }
```

The router must not branch on object class identity — that would need an arbitrary
80-class → expert mapping and is brittle against false positives. Class names stay in
`counts` / `agreement.objects` / per-keyframe boxes as **diagnostics**, deliberately kept
rather than dropped: a false positive there is exactly what makes `flags.object` fire
wrongly, and that is invisible from the boolean alone.

**Output** `profile.json`: per shot `flags` / `counts` / `agreement` / `scenes` plus full
per-keyframe boxes (so later experts can crop without re-detecting), and a `video_flags`
union plus `object_totals` histogram.

### Non-obvious things about 03

- Scene classification reuses **02's** MobileCLIP cache via
  `download_models.stage_dir("02-segmentation")`, deliberately, to avoid a second 380 MB copy.
- `text_ppocrv3.onnx` is PaddleOCR's detector exported to ONNX. It gives the PaddleOCR
  detection stage that was originally specified **without** the paddlepaddle dependency,
  which would have risked cuDNN conflicts against torch's CUDA 12.4 runtime.
- The DB text model needs an input size that is a multiple of 32.
- `YOLO_CONFIG_DIR` and `YOLO_OFFLINE` are set in the Dockerfile so ultralytics writes its
  settings into `models/` and does not phone home.
- Detector weights are fetched **on demand** by `download_models.detector()`, so switching
  `--detector` does not pull every variant.
- ultralytics is **AGPL-3.0** — fine for a thesis, relevant if the code is published.
- **ultralytics must be installed with `--no-deps`.** It requires `opencv-python` (the GUI
  build), which installs a second `cv2` alongside `opencv-python-headless` — at one point
  the image held opencv-python 5.0.0.93 and headless 4.10.0.84 at the same time, with
  import order deciding the winner. Its remaining dependencies are in `requirements.txt`.
- Faces on stylised animation do get detected (YuNet scores 0.88–0.92 on `anime.mp4`),
  better than expected. Do not assume it fails there without testing.
- **Object false positives are the known weak point.** At the current defaults
  (`yolo11n`, `--det-conf 0.25`) about a third of object detections are implausible for
  their clip, and `flags.object` fires on 19 of 36 test shots. Measured sweep:

  | detector | conf | shots triggered | detections | implausible | time |
  | --- | --- | --- | --- | --- | --- |
  | yolo11n | 0.25 | 19/36 | 60 | 33% | 5.2s |
  | yolo11n | 0.40 | 13/36 | 33 | 36% | 1.0s |
  | yolo11n | 0.55 | 8/36 | 17 | 35% | 1.0s |
  | yolo11s | 0.25 | 15/36 | 47 | 23% | 2.2s |
  | yolo11s | 0.40 | 13/36 | 29 | 17% | 1.2s |
  | yolo11s | 0.55 | 7/36 | 19 | 11% | 1.2s |

  **Raising the threshold does not help on yolo11n** — the implausible share stays ~35%,
  it just detects less of everything. **Model size is the lever**, and yolo11s costs
  essentially nothing extra. `yolo11s --det-conf 0.4` looked best (same trigger count as
  the current defaults, half the junk) but defaults were left unchanged pending a decision.
  "Implausible" here means the class is absent from a hand-written plausible set per clip —
  a transparent proxy, not a precision metric.
- A `--min-agreement` filter was considered and **not** added: single-frame detections
  outnumber recurring ones even for plausible classes, so it would suppress real brief
  objects too. `agreement.objects` is recorded so this can be revisited with data.

## 04-router

Reads `profile.json`, writes `routing.json`. **Pure function — no models, no GPU, no new
dependencies.** Maps presence flags to the active expert set, deterministically: the same
input always produces a byte-identical file (verified).

| expert | kind | fires when |
| --- | --- | --- |
| `depth`, `edges`, `scene`, `tags` | plastic | always |
| `detection_tracking` | conditional | a person is present, **or** an object needing stable identity |
| `face`, `body` | conditional | a person is present |
| `pairwise_relations` | conditional | two or more people co-occur in one profiled keyframe |
| `ocr` | conditional | text is present |
| `object_masks` | conditional | objects are present |

Every expert is recorded with `fired`, the `rule` in words, and the `evidence` that decided
it — including the ones that did **not** fire. That is the auditability requirement: a
routing decision can be explained from `routing.json` alone, without re-running anything.

The "camera on a table" case works as specified: a shot with no people gets
`depth, edges, scene, tags, ocr, object_masks` and skips `detection_tracking, face, body,
pairwise_relations` — the entire human pipeline, including tracking that would otherwise
have run at full frame rate for nothing.

**`anime.mp4` shot 4 is that case.** Frames 199–239, `person: False`, `face: False`, so it
receives the eight plastic experts plus `ocr` and `object_masks` and skips the whole human
branch. `messi.mp4` shot 23 (frames 317–336) is the purest version: all four flags false,
plastic experts only, nothing conditional at all.

The failure mode that matters most in a gated pipeline is the opposite one, and it is
asymmetric: a false **positive** in 03 wastes compute, while a false **negative** silently
deletes the whole human branch for that shot, and nothing downstream can recover it. It also
compounds, because 03 only runs YuNet *inside* person boxes — no person box means the face
flag cannot fire either, however plain the face is.

**`anime.mp4` shot 3 was the standing example of that, and the `yolo11s` default fixed it.**
Under `yolo11n` the shot — a full-frame close-up of a character's face — profiled as
`person: False`, `face: False` plus a `clock` false positive, and lost its entire human
branch. Under `yolo11s` it profiles as `person: True`, `face: True` with no spurious object,
and produces one track whose body box covers 80–87% of the frame, a face in 3 of its 5
frames, and two series samples carrying head pose and emotion. The router was always correct
on the input it was given; the input improved. The lesson stands even though the example no
longer does — nothing in `routing.json` could have revealed the miss.

### Non-obvious things about 04

- `IDENTITY_CLASSES` is the one place the router looks at object class names, because the
  spec requires "an object that needs a stable identity downstream" and that is inherently
  class-dependent. It is an explicit, auditable frozenset (vehicles, animals, sports
  equipment). Everything else routes on booleans only.
- `--router-agreement` is the optional soft gate from the spec: a flag must hold in at least
  that fraction of a shot's keyframes. Default 0.0 = off. It is an effective lever on the
  object false positives — on `messi.mp4` it cuts `object_masks` from 9 shots to 4 at 0.5,
  because one-keyframe detections like `banana` stop counting. (Those figures were 12 and 3
  under `yolo11n`; the detector switch removed a third of the false triggers on its own.)
- Cost-aware top-k routing under a compute budget is **not** implemented; it needs a cost
  model per expert, which does not exist until 05/06 are written.
- `person_max` is reported as 0 when the person flag is gated off, so evidence never
  contradicts the decision.

## 05-global-experts

The plastic, always-on branch. Reads `segmentation.json` for the keyframe list and
`keyframe_embeddings.npy` for tags; **ignores `routing.json` entirely**, because these
experts are ungated by definition. Writes dense maps under `data/processed/<video>/global/`
and everything else to `global.json`.

| expert | model | consumes |
| --- | --- | --- |
| depth | `depth-anything/Depth-Anything-V2-Small-hf` | keyframe images |
| normals | derived — no model | the depth map |
| edges | PiDiNet via `controlnet_aux` | keyframe images |
| scene | Places365 ResNet18 | keyframe images |
| tags | MobileCLIP zero-shot over RAM's 4,585-tag list | **02's embeddings, no pixels** |
| semantic | SegFormer-B1 ADE20K, 150 classes | keyframe images |
| panoptic | Mask2Former swin-tiny COCO-panoptic, 133 classes | keyframe images |
| classical | OpenCV/numpy — no model | keyframe images |

Models load one at a time and are freed before the next, so six never sit on the GPU at
once. Keyframes are decoded once and shared. Depth must run before normals; nothing else
is ordered.

### Non-obvious things about 05

- **Panoptic `segment_index` is per-keyframe geometry, never identity.** It is not matched
  across frames and must not be read as a track id — stable identity comes only from the
  gated branch. The field is named `segment_index` rather than `instance_id` for this reason,
  and `global.json` carries a `note` saying so.
- **Two label spaces on purpose**: `ade20k` for semantic, `coco_panoptic` for panoptic. There
  is no ADE20K panoptic checkpoint at tiny size, so unifying them is not possible. Each
  output records its own `label_space`; downstream must never conflate the two id ranges.
- **Normals are unprojected, not gradient-based.** Taking Sobel gradients of normalised
  disparity produces a map dominated by depth discontinuities where flat surfaces read as
  featureless. The working version converts disparity to a bounded pseudo-distance, unprojects
  to camera space with an assumed focal length of `max(H, W)`, and takes the cross product of
  neighbouring surface vectors. Depth Anything is relative, so normals are directionally
  right but not physically calibrated.
- Scene was **moved here from 03-profiler**, which no longer emits `scenes`. 03 used zero-shot
  CLIP; 05 uses a trained Places365 head. Two differing answers in two files would only have
  confused the merge in 07/08.
- Dense maps are PNG at `--map-size` longest side (512), roughly 10x smaller than `.npy` and
  directly viewable. Depth is 16-bit with `min`/`max` recorded so the original range is
  recoverable. 66 keyframes produce 330 files, ~52 MB.
- `controlnet_aux` needs `--no-deps` (it wants `opencv-python`) **and** `scikit-image`, which
  its package `__init__` imports transitively via the anyline detector.
- `transformers==4.46.3` requires `huggingface_hub<1.0`, which is why that pin is a range.

## 06-detection-tracking

The first gated stage, and the first consumer of `routing.json`. Runs only on shots where
04-router fired `detection_tracking`, and establishes the identities every later conditional
expert reads. Writes `tracks.json` plus the best crops per track under `tracks/`.

**Downloads no new weights.** The detector and YuNet both come from `03-profiler`'s
directory, the same cross-stage reuse 05 does with 02's CLIP cache.

| piece | how |
| --- | --- |
| bodies | one ultralytics pass per detection frame, `person` class only |
| tracking | ByteTrack (default) or BoT-SORT, rebuilt per shot |
| faces | YuNet inside each tracked body box, so a face inherits the body's track id |
| objects | temporal union per class, never identity |
| crops | best `--crops` face crops **and** best `--crops` body crops per track |

Tracks are renumbered per shot from 1 in order of first appearance, after short tracks are
dropped, so ids are dense and readable. ByteTrack's own id is kept as `tracker_id`.
Interpolation, filtering and renumbering all happen before crops are written, so no crop
directory exists for a track that is not in the output.

**Faces have no tracker of their own.** Running YuNet inside the tracked body box extends
03-profiler's pattern and means face and body share one key. A separate face tracker would
need its own weights, a second tracker, and a face↔body association step, and would break
every time a head turns. The cost is that a face whose body the detector misses is lost.

**Track ids are scoped to one shot.** A fresh tracker is built at every boundary, so ids
restart at 1 and the only global key is `(shot, track_id)`. `tracks.json` carries a `note`
saying so. Cross-shot re-identification is a separate problem and is not attempted here.

**Detection runs on `--det-stride` (default 2), tracking covers every frame.** Frames
between detections are filled by linear interpolation — offline, so interpolation beats
forward prediction — and every such row is marked `"interpolated": true`. Only pixels for
detection frames are ever read, so the stride saves decoding as well as inference.

**Interpolation is bounded by the stride.** A wider hole means the tracker lost the object
and re-acquired it; a straight line across that is fabrication, so those frames get no row.
Each track records `gaps` and `widest_gap` so the discontinuity stays visible. Before this
bound, `patrick.mp4` would carry 427 further fabricated rows on top of 993 real observations,
some bridging holes of nearly a second.

**`--track-low-conf` (0.1) sits well below 03's `--det-conf` (0.25) on purpose.** ByteTrack's
second association pass is the reason the module uses ByteTrack at all, and feeding it only
detections above 0.25 would silently disable it. Measured on `anime.mp4`, with the second
stage off (`--track-low-conf 0.25`) against on:

| | tracks | observations | track breaks |
| --- | --- | --- | --- |
| second stage off | 40 | 1162 | 37 |
| second stage on | 41 | 1242 | **15** |

80 more real observations and **breaks cut by 59%**. That is exactly the
recovery of blurred and briefly occluded people the design calls for.

### Non-obvious things about 06

- **BoT-SORT ReID does not exist in ultralytics 8.3.40.** Its source says "Haven't supported
  BoT-SORT(reid) yet" and sets `self.encoder = None`; every use site is guarded, so
  `with_reid` is a silent no-op. `--tracker botsort` therefore buys camera-motion
  compensation only. This is not a configuration mistake — the feature is absent upstream.
- **`track_id` is renumbered; `tracker_id` is ByteTrack's own.** ByteTrack burns an id the
  moment a candidate track is created, but only returns tracks matched a second time, so its
  raw ids are very sparse - `patrick.mp4` issues 603 of them and 510 belong to candidates
  that never confirm, giving 1, 3, 66, 67, 121, 146 for the first six survivors. Tracks are
  therefore renumbered per shot from 1, in order of first appearance, after `--min-track`
  filtering. The raw id is kept as `tracker_id` so a track can still be traced back to the
  tracker. `track_id` counts surviving tracks in a shot; `tracker_id` counts nothing.
- **Never build the tracker with `dict.setdefault`.** Python evaluates the default eagerly,
  so `trackers.setdefault(shot, _make_tracker(...))` constructs a tracker on every frame, and
  `BYTETracker.__init__` calls `reset_id()`, which zeroes the **class-level** `BaseTrack._count`.
  Track ids then restart from 1 on every frame, distinct people collide under one id, and the
  per-shot observation table silently merges them. The tracker is now built with an explicit
  `get`-then-insert.
- **The trackers are driven directly, not through `model.track()`.** `BYTETracker` and
  `BOTSORT` are fed a small `Detections` shim carrying `xywh`/`conf`/`cls`. That is what
  lets one detector pass serve both the tracker and the object union, and lets the tracker
  be rebuilt per shot. Their `update()` returns `[x1, y1, x2, y2, track_id, score, cls, idx]`.
- **YuNet is letterboxed into a constant 320×320 input.** Calling `setInputSize` per body
  box reallocates its internal buffers, after which it returns phantom faces scoring ~1.0
  that differ between runs on byte-identical pixels — verified in isolation: ten fresh
  detectors on one 32×105 crop gave `1.0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 1.0`. Letterboxed at a
  fixed size the same crop gives 0 ten times out of ten, and the whole stage becomes
  reproducible. **Do not go back to per-crop `setInputSize`.**
- **YuNet can also return `inf` box coordinates**, which raised `OverflowError` on a real
  clip. Non-finite rows are dropped. `03-profiler` has the same unguarded conversion in
  `_faces` and is exposed to the same crash; it has not been hit there only because it runs
  on far fewer crops.
- **Face boxes are clamped to the body box**, not to the frame. YuNet regresses coordinates
  that can run past the crop it was given, and a face escaping its own body box breaks the
  one invariant that makes the shared track id meaningful.
- **Face and body crops are ranked separately.** They feed two different conditional experts,
  and a single mixed pool of five can leave either with nothing. Ranking is sharpness × area
  × frontality × detector score, normalised within the track, then split into equal time
  spans with the best taken from each — 02's keyframe lesson, since ranking a whole track at
  once returns near-identical neighbouring frames.
- **Frontality comes from YuNet's five landmarks**, which 03-profiler discards. A frontal
  face puts the nose on the eye midpoint; profile views push it towards one eye.
- **Objects get a temporal union, not identity** (`--track-objects` is not implemented, and
  the union is what the identikit actually consumes). It is nearly free: the same forward
  pass already returns all 80 classes. Objects keep 03's `--det-conf`, not the tracker's low
  threshold, so the union is not flooded with junk.
- `cat2.mp4` is the clean demonstration of gating without people: all four shots fire
  `detection_tracking` because a cat is an identity class, and all four produce **zero
  tracks**, because there is nobody to track.

## 07-conditional-experts

The gated figurative and enunciative experts. Reads `routing.json` for the gate, `tracks.json`
for identities and crops, and `profile.json` for text regions. Writes `conditional.json` plus
`face_embeddings.npy` and `body_embeddings.npy`.

**Two sampling rates, because the consumers differ.**

| rate | experts | source |
| --- | --- | --- |
| every `--series-stride` observed rows | head pose, blendshapes, emotion + valence/arousal, body pose | decoded video |
| best crops per track | ArcFace identity, body appearance, age, gender, zero-shot attributes | 06's crop JPEGs, no decoding |
| text-triggered | OCR recognition | 02's keyframes, no decoding |

Decoding happens only for the series; everything else reads files 06 and 02 already wrote.

| expert | model | note |
| --- | --- | --- |
| mesh + head pose + blendshapes | MediaPipe FaceLandmarker | one pass, three outputs |
| body pose | MediaPipe PoseLandmarker | 33 keypoints plus world coordinates |
| emotion, valence, arousal | HSEmotion EfficientNet-B0 | eight classes and V/A together |
| identity | ArcFace `w600k_r50` | 512-d, L2-normalised, 08's input |
| body appearance | YouTu ReID `person_reid_youtu_2021nov` | 768-d, L2-normalised, 08's other input |
| age, gender | InsightFace `genderage` | same release zip as ArcFace |
| attributes | MobileCLIP zero-shot | reuses 02's cache, no new weights |
| OCR | OpenCV zoo CRNN | the recognition half 03 left out |

### Non-obvious things about 07

- **Head pose is not a separate model.** It is the rotation block of FaceLandmarker's
  facial transformation matrix, which the same call returns alongside 478 landmarks and 52
  blendshapes. This is why 6DRepNet and solvePnP were both dropped - solvePnP would also
  need camera intrinsics the pipeline does not have.
- **Emotion runs only where the mesh confirms a face.** HSEmotion has no "not a face" answer;
  it classifies whatever it is handed. Ungated it produced an emotion for the hand in
  `cat1.mp4` and fired on 257 of `anime.mp4`'s crops against 148 real faces. Gated behind
  FaceLandmarker the counts match exactly.
- **The CRNN is single-channel and reads about ten characters.** Feeding it BGR throws
  inside the first convolution. 03's detector returns whole caption lines at a median 13:1
  aspect, which squash to nothing at the model's 100x32 input, so crops are normalised to
  32px high and read in chunks. Whole-line reads gave `'conitasks'`; chunked they give
  `'yourw orldcu pcare erlsso'` for "Your World Cup Career Is So".
- **Text crops are never padded.** `_crop`'s margin is a fraction of the *longest* side, so
  padding a 443x34 caption line inflates its height too, halving the chunk count and undoing
  the fix above.
- **ArcFace needs aligned faces.** The five template points come from FaceLandmarker's
  landmarks, ordered by image x rather than by index, so the template's left/right assignment
  holds without depending on which eye MediaPipe calls "left".
- **`buffalo_l`'s two models want different input ranges.** ArcFace (`w600k_r50`) takes
  `(x-127.5)/127.5`; `genderage` takes raw 0-255. Feeding genderage ArcFace's range does not
  fail, it collapses: every crop came back 34-35 years old at 0.54 gender confidence, and the
  0.54 was mistaken for an honest answer on low-resolution faces. Corrected, the same 464
  crops span ages 20-67 at 0.69-0.99 confidence, and `messi.mp4` reads 47 of 77 crops male
  instead of 3. A model fed the wrong range answers with a constant, and a constant survives
  every downstream check - 09 reported `variance: 0.0` at `n: 5` and was right to.
- **onnxruntime runs on CPU deliberately.** Its CUDA build would have to agree with torch's
  on cuDNN, which is the conflict that kept paddlepaddle out of 03. These models are small.
- **mediapipe needs `libGLESv2.so.2` and `libEGL.so.1`.** Its Tasks API is a C library loaded
  by ctypes, so the import succeeds and `create_from_options` is what fails without them.
- **mediapipe is pinned below 0.10.30.** 0.10.35 removed the legacy `mp.solutions` namespace,
  and `controlnet_aux`'s package `__init__` calls `mp.solutions.drawing_utils` - so installing
  a current mediapipe silently broke **05-global-experts**, which had worked only because
  mediapipe was absent. 0.10.21 has both `solutions` and the Tasks API. Do not raise this pin
  without re-running 05.
- **mediapipe also wants `opencv-contrib-python`**, a third cv2 build. Neutralised the same
  way as ultralytics' `opencv-python`, with a `sys_platform == 'never'` override in
  `pyproject.toml`.
- **The two embeddings are not interchangeable, and the body one carries the stage.** ArcFace
  identifies a person; the ReID vector identifies an outfit under one lighting. Coverage is
  what forces the pairing: a face embedding exists for 17 of `anime.mp4`'s 42 tracks, 24 of
  `messi.mp4`'s 88 and **4 of `patrick.mp4`'s 69**, while a body crop exists for nearly all of
  them. Face-only consolidation would reach 6% of the clip that most needs it.
- **The body model is gated on `body`, the face one on `face`.** They are different router
  flags, so a track can have one descriptor and not the other, and 08 must cope with that.
- **Embedding rows carry the frame of the crop they came from.** Not every crop yields a
  vector — FaceLandmarker fails, or `estimateAffinePartial2D` returns nothing — so row order
  alone does not identify a crop, and 08 gates on per-crop quality and needs the mapping.
- `object_masks` is **not implemented**; `conditional.json` records it under
  `not_implemented` rather than silently ignoring the gate.

## 08-consolidation

Resolves 06's segment-local track ids into global person ids, and writes `consolidated.json`.
A **pure function** like 04, 09 and 10: no models, no GPU, byte-identical across runs. 07 does
the embedding, this does the clustering, which is what makes retuning a threshold cost a second.

**This is the pipeline's main identity resolver, not a repair step.** Within-segment
association is motion-only and survives about two seconds, so one busy segment fragments one
person into many tracks — 69 surviving tracks in a single shot of `patrick.mp4`. Most of the
identity in a video is established here, not in 06.

**Two passes, because the priors differ.**

| pass | pairs offered | constraint | threshold |
| --- | --- | --- | --- |
| within segment | tracks in the same shot | frame sets must be disjoint | strict: 0.55 face, 0.70 body |
| across segments | merged units in different shots | satisfied by construction | looser: 0.45 face, 0.60 body |

Inside a shot lighting and pose are stable, so a high threshold costs nothing and buys
precision. Across a cut the same person genuinely scores lower, so keeping the high threshold
would find nothing. Same-segment pairs are deliberately **not** re-offered in pass two: a merge
the strict threshold rejected must not be laundered by the loose one.

**Descriptors are per-track means of gated crop vectors.** A crop contributes only if its box
area clears `--consol-min-area` and, for a face, its frontality clears `--consol-min-front`.
One bad crop inside the mean is exactly what produces a wrong link.

**Abstention is permitted, and it fires.** A track with no crop above the floors gets no
descriptor, is never a candidate for a link, and comes out as a singleton with
`"abstained": true`. A wrong link merges two people's demographics, emotion distributions and
gaze statistics into one fictitious person, and nothing downstream can detect that it happened.

Measured across the test clips:

| clip | tracks | persons | within merges | across merges | abstained |
| --- | --- | --- | --- | --- | --- |
| patrick | 69 | **42** | 27 | 0 | 3 |
| messi | 88 | 80 | 2 | 6 | 22 |
| anime | 42 | 32 | 1 | 9 | 3 |
| cat1 | 11 | 8 | 3 | 0 | 0 |
| thanos | 5 | 5 | 0 | 0 | 3 |
| cat2 | 0 | 0 | 0 | 0 | 0 |

The two clips have opposite profiles, which is the argument for splitting the passes:
`patrick.mp4` is one shot with 2,346 same-shot pairs and no cross-shot ones, `messi.mp4` is
23 shots with 220 same-shot pairs and 3,608 cross-shot ones.

### Non-obvious things about 08

- **Complete linkage, not single linkage.** A unit joins a cluster only if it clears the
  threshold against **every** member. Single linkage chains distinct people together through
  one weak intermediate, and it also breaks the frame-disjointness constraint at cluster scope:
  A-B disjoint and B-C disjoint does not make A-C disjoint. Merging in descending similarity
  makes the result independent of the order tracks arrive in.
- **Two tracks sharing a frame are never one person**, whatever they look like. That hard
  constraint removes 18% of `patrick.mp4`'s same-shot pairs before any similarity is computed,
  and it is what makes 10's merged per-frame table well defined.
- **Sharpness is not a usable quality floor and was removed.** It is Laplacian variance, and
  its scale is a property of the footage: the median crop runs from 10 on `thanos.mp4` to 1139
  on `messi.mp4`. A floor of 20 discarded every crop in two clips while passing everything in
  the other three. Area is pixels and frontality is a normalised ratio; both mean the same
  thing in every clip, so those are the floors that survived.
- **Body appearance does nearly all the work, and the split is not close.** A face descriptor
  survives gating for 4 of `patrick.mp4`'s 69 tracks and 66 have a body one.
- **Cross-shot face linking barely fires on this footage.** On `messi.mp4` the cosine between
  cross-shot face descriptors has median 0.061 and maximum 0.526, so at 0.45 exactly one pair
  links. ArcFace on a sports broadcast — motion blur, varied angles, small faces — is a harder
  problem than ArcFace on portraits, and the numbers say so rather than the thresholds hiding it.
- **`thanos.mp4` abstains on 3 of 5 tracks and that is correct.** FaceLandmarker finds a face
  in only 5 of its 24 face crops, because the character is CGI with non-human proportions, and
  06 kept body crops for one track. With no descriptor there is no evidence, and the stage says
  so instead of guessing.
- **The output stamps the settings that produced the ids.** `(segment, track_id)` depends on
  the detector, `--det-stride` and `--min-track`, and 06 renumbers survivors densely, so
  changing any of them shifts every id. `source` records them; nothing yet checks the stamp.

## 09-aggregation

Collapses 06's per-frame table and 07's per-sample series into per-person records. A **pure
function** like 04-router: no models, no GPU, no new dependencies, byte-identical across runs.
Reads `consolidated.json`, `conditional.json`, `tracks.json`, `global.json`, `routing.json`,
`segmentation.json`; writes `aggregated.json`.

**Two scopes, and the person one is now video-wide:**

| scope | key | contents |
| --- | --- | --- |
| person | `person_id` | head pose, emotion, valence/arousal, blendshapes, pose, demographics, attributes |
| segment | `segment` | plastic timeline from 05, deduplicated text, object extents from 06 |

08 having run first is what makes the person scope available. Every statistic is computed once,
from the raw samples, over whole people — pooling 08's per-fragment summaries instead would
have been exact for means and variances and wrong for medians, MADs and modal agreement, which
is the argument that put consolidation ahead of this stage.

### Non-obvious things about 09

- **Variance is the deliverable, and it is null below two samples.** It is the video-only
  dynamism signal, so reporting 0.0 for a single-sample track would be a lie. Every statistic
  carries `n`, and `thin_records` counts persons with fewer than two series samples - **36 of
  80** on `messi.mp4` and **30 of 42** on `patrick.mp4`. Thin records are the majority case on
  those clips, not an edge case: since 08 gives every track a person, body-only tracks reach
  this stage with no series at all. 11-vllm-synthesis needs a rule for them.
- **Pose statistics drop joints mediapipe predicted outside the crop.** MediaPipe emits
  coordinates for occluded and off-frame joints; 22% of `messi.mp4` keypoints fall outside
  `[0,1]`. Aggregating them would do to pose exactly what aggregating interpolated boxes
  would do to position. `--pose-vis` sets the visibility floor on top of that.
- **Movement is measured in crop-normalised coordinates**, so it reads as change of posture
  rather than change of camera distance. A person walking towards the lens does not register
  as movement simply because their box grew.
- **Categoricals keep their distribution.** Emotion is the mean of the per-sample probability
  vectors plus a modal label and the fraction of samples that agreed with it - the same
  argument that kept `agreement` in 03 rather than collapsing to booleans.
- **Continuous values carry mean, variance, median and MAD.** The mean and variance are what
  the thesis wants; the median is what survives one bad crop.
- **Label lists sort by support first, then mean score.** A tag offered by three keyframes
  outranks one offered by a single keyframe with a marginally higher score, so printed scores
  are deliberately not monotonic.
- **Text dedup is exact-string and therefore weak.** OCR reads the same caption slightly
  differently on each keyframe, so `messi.mp4` collapses 313 readings to only 259 groups.
  `anime.mp4` does better, 85 to 52. Fuzzy matching was not invented; the counts are recorded
  so the decision can be made against data.

## 10-fusion

Turns 09's per-person records into the edges between them. A **pure function** like 04, 08 and
09. Reads `aggregated.json`, `conditional.json`, `tracks.json`, `segmentation.json`; writes
`fused.json`. A pair can now be co-present in several segments and their evidence accumulates
across all of them; `co_present.segments` records which.

| edge | derived from | gate |
| --- | --- | --- |
| co-presence | overlapping frame sets | prerequisite for every other edge |
| placement | box centroids and area ratio | any co-present pair |
| attention | head orientation vs. direction to the other person | both need head pose |
| synchrony | Pearson r of valence and arousal | needs `--sync-min` shared samples |

Measured across the test clips:

| clip | persons | relations | attention | mutual | synchrony | cross-segment |
| --- | --- | --- | --- | --- | --- | --- |
| anime | 32 | 184 | 130 | 1 | 14 | 6 |
| messi | 80 | 200 | 74 | 3 | 0 | 1 |
| patrick | 42 | 274 | 38 | 0 | 0 | 0 |
| cat1 | 8 | 3 | 0 | 0 | 0 | 0 |
| thanos | 5 | 1 | 0 | 0 | 0 | 0 |
| cat2 | 0 | 0 | 0 | 0 | 0 | 0 |

These are not comparable to the pre-consolidation figures. The person count is no longer a
track count: every track now reaches this stage, because a body descriptor exists for tracks
that had no series and no face attributes and so never appeared in `conditional.json` before.
`patrick.mp4` reads 69 tracks → 42 people; the old "18 entities" counted only the tracks 07
had something to say about.

### Non-obvious things about 10

- **It is called attention, not gaze, because that is what it measures.** There is no gaze
  model in the pipeline; the ray is the outward axis of 07's head-pose rotation projected
  into the image plane. Naming it `gaze` would overclaim by a wide margin.
- **Synchrony abstains rather than guessing.** Below `--sync-min` (5) it emits
  `{"available": false, "reason": ...}` and no number. That costs almost everything:
  `messi.mp4` and `patrick.mp4` produce **zero** synchrony edges, `anime.mp4` thirteen.
  Correlating three-point series would have produced a full graph of meaningless numbers.
- **Placement uses box area, not depth.** 05 writes depth at keyframes only, so a per-frame
  depth ordering does not exist. `size_ratio` is an honest proxy and `nearer` has a
  `similar` band rather than forcing a call.
- **Face-to-body binding is validated, not performed.** 06 already clamps a face inside its
  body box and gives both one track id, so this stage only counts violations - zero on every
  clip - rather than associating anything.
- **Text binds to entities, never to raw tracks.** 06 emits more tracks than 07 keeps, so
  binding against `tracks.json` attaches captions to ids that have no person record. Caught
  by verification: 79 failures on the first run.
- **Co-presence is not implied by a shared segment.** `cat1.mp4` has 8 persons and 28
  possible pairs, and only **3** are ever co-present - the other 25 share the one segment but
  never a frame, being sequential fragments of the same hand.

## Performance facts (measured, RTX 4070 laptop, 22 cores)

- Decoding is **~4%** of 02's runtime. Models and preprocessing dominate.
- 02's scan: 65.7s → 34.2s over 3260 frames after optimisation, with identical shot
  boundaries and embeddings at cosine 0.996.
- OpenCV and ffmpeg decode at the same speed (1.9s vs 1.7s / 3260 frames) — OpenCV *is*
  libavcodec underneath.
- ffmpeg beats OpenCV at *writing* JPEGs: 1.5× faster and 30–40% smaller at equal quality.
  This is why 01 originally used ffmpeg. It now uses OpenCV anyway, by request, for consistency.
- **NVDEC is slower than CPU decoding here** at 720p, 1080p and 4K. JPEG encoding is the
  bottleneck and every frame must be copied back to host memory. Do not "optimise" by
  adding GPU decoding.
- Writing to the bind-mounted `data/` is ~2× slower than container-local disk.
- PNG is ~5× slower and ~9× larger than JPEG.
- **Decoding the video beats reading 01-sampling's JPEGs, by ~9×.** Measured in 06 over the
  frames it actually wants (`--frame-source video` vs `frames`, three runs each):

  | clip | frames read | video | 01-sampling frames |
  | --- | --- | --- | --- |
  | patrick | 164 of 326 | 0.48–0.54s | 4.15–5.08s |
  | messi | 176 of 337 | 0.28–0.32s | 2.51–3.70s |

  This holds even on `messi.mp4`, where routing skips a shot and the frames path does pure
  random access while the video path must still `grab()` past everything. JPEG decode simply
  costs more than H.264 decode plus a skip. The frames path is also **lossy**: re-encoding at
  q95 perturbs the detector enough to change association — `patrick.mp4` gives 69 tracks over
  1888 rows from the video and 68 over 1875 from the JPEGs. `--frame-source video` is the
  default for both reasons.

## Known open items

1. **01-sampling still has no consumer, and now has none coming.** 06 was its last candidate
   and measured it as ~9× slower and lossy against decoding the video. `--frame-source frames`
   keeps the comparison reproducible, but nothing uses it by default. Delete, or make opt-in.
2. **Two decodes in 02.** Could be one via `CAP_PROP_POS_FRAMES` seeking, but OpenCV seeking
   on H.264 is off-by-one-prone and correctness matters more here.
3. **MobileCLIP embeds every frame** — ~60% of remaining runtime. A `--keyframe-stride`
   would nearly halve it at the cost of a smaller candidate pool.
4. **`clean_shot` vs `default` boundaries** disagree slightly on a short ambiguous segment
   in `anime.mp4` (192–198 vs 192–195). Not investigated.
5. **ffmpeg is still in the image** — only because `omnishotcut` imports `ffmpeg-python`
   at module load. The pipeline never shells out to it.
6. **Detector default is now `yolo11s`; `--det-conf` is still `0.25`.** Switching the model
   was the half of the sweep that mattered — on `cat1.mp4` it relabelled a pair of plush toys
   from `person` to `teddy bear`, dropping `person_max` from 2 to 1 and correctly killing the
   `pairwise_relations` branch. Raising `--det-conf` to 0.4 remains unadopted, and `cat1.mp4`
   is the argument against it: all three person detections there sit between 0.267 and 0.276,
   so 0.4 would discard the two correct ones along with the false one. The switch also moved
   numbers elsewhere, which is why several figures in this file changed with it: `anime.mp4`
   recovered shot 3's face (41 tracks to 42, 8 cross-segment merges to 9), and `messi.mp4`'s
   `object_masks` fell from 12 shots to 9.
7. **Cost-aware top-k routing** is unimplemented — it needs per-expert cost figures. 05 and
   06 now exist, so most of that data is finally available.
8. **`--router-agreement` defaults to off.** It suppresses object false positives well
   (12 shots to 3 on `messi.mp4` at 0.5) but has not been validated for false negatives.
9. **The `face` expert fires on person presence, not `flags.face`.** `flags.face` and
   `face_max` are computed by 03 and read by nobody. On `cat1.mp4` — where the only "person"
   is a hand — this runs YuNet over the whole shot for nothing, and it produced one false
   face crop. Gating `face` on `flags.face` would make the flag load-bearing; the risk is a
   shot whose keyframes all catch the subject in profile losing its face branch entirely.
10. **Re-identification exists, in 08, and is not tracking.** 06 still hands over a new
   track id whenever motion association fails, and across a shot boundary every identity is
   new by construction — that is unchanged and correct, because Kalman and IoU mean nothing
   across a cut. 08 links those fragments afterwards by appearance, which is the separate
   algorithm the dropped 03-association design predicted would be needed. What remains open
   is that its thresholds have no ground truth behind them: the merge counts are measured,
   the merge *correctness* is inspected in `consolidation_preview.mp4` by eye. A hand-labelled
   identity set on one clip would turn the thresholds from defensible into calibrated.
11. **Low-quality face crops are still kept by 06, and now gated by 08.** Ranking in 06 is
   relative within a track, so `patrick.mp4` track 3 contributes five blurred non-faces that
   all cleared `--face-conf 0.6`. They remain in `tracks.json` — 06 has no clip-independent
   scale to reject them on — but 08 refuses them a descriptor via `--consol-min-area` and
   `--consol-min-front`. Sharpness was tried as a third floor and removed; see 08 above.
12. **A full-frame face can still be missed by 03, and the router cannot know.** The standing
   example — `anime.mp4` shot 3, a close-up filling the frame that profiled as `person: False`
   plus a `clock` — was fixed by the `yolo11s` default and now profiles correctly. What is
   unresolved is the general case: 03 sees only keyframes, runs YuNet only inside person
   boxes, and a missed person box deletes the human branch with nothing downstream able to
   notice. A keyframe-level sanity check, or a fallback full-frame face pass when a shot has
   no person, is still the fix; it needs a decision about how much 03 is allowed to cost.
13. **`tracks.json` and `conditional.json` grow with video length × track count.** Fine for
   these clips, not for a feature film; 09 and 10 exist partly to compress them.
14. **07 runs the face branch wherever a person is present**, because the router gates `face`
   on person presence (item 9). On `cat1.mp4` FaceLandmarker runs across the whole shot and
   correctly finds nothing; on `patrick.mp4` only 4 of 69 tracks yield an identity embedding.
   Gating `face` on `flags.face` would cut that work directly.
15. **Ethnicity is specified but unimplemented.** FairFace has no stable direct weight URL,
   and the output needs framing rather than a number — see [PLANNED.md](PLANNED.md).
16. **Consolidation did not rescue synchrony, and the reason is structural.** Running fusion
   after consolidation was half the argument for the reorder, and it did lengthen the series
   it was meant to: median samples per record went from 5 to 9 on `anime.mp4`. But
   `messi.mp4` and `patrick.mp4` are unchanged at zero synchrony edges, because the tracks
   that merge there are body-only — no face, therefore no affect series — while the tracks
   that carry affect are face-bearing and rarely merge. Synchrony also needs a *pair* to
   co-occur repeatedly, which is strictly harder than one person appearing twice. The
   ordering is still right; the expectation was too optimistic.
17. **OCR is chunked, not line-level.** The CRNN reads ~10 characters, so caption lines are
   split and rejoined with spaces at arbitrary boundaries — `'legend arythat youre getting'`.
   Readable, but the word boundaries are not real. PP-OCRv4's recognition head exported to
   ONNX would read a whole line at 48×320 and is the upgrade if this matters.
18. **Synchrony is barely computable on shot-heavy footage**, still. At `--sync-min 5`,
   `messi.mp4` and `patrick.mp4` yield zero edges and `anime.mp4` fourteen — one more than
   before consolidation. Lowering the floor produces numbers, not measurements. What would
   actually move it is affect on more tracks, which means a face branch that reaches them
   (item 9) rather than a different sample floor.
19. **Attention is head orientation, not gaze.** No gaze model is installed. L2CS-Net would
   give true gaze angles in the head frame, still needing composition with head pose; a
   gaze-target model would answer "who looks at whom" directly. Neither was added.
20. **Depth ordering is unavailable to 10.** 05 writes depth at keyframes only, so placement
   falls back to box-area ratio. Settling the per-expert frequency table would resolve it.
21. **Nothing checks 08's `source` stamp.** It records the detector, stride, tracker
   thresholds and `--min-track` that produced the track ids, so a downstream file keyed to a
   stale 06 run is *detectable* — but 09 and 10 do not compare it against `tracks.json`, so
   detection is still manual.
22. **`--consol-*` defaults are chosen from measured similarity distributions, not tuned.**
   Body within 0.70 sits near the 96th percentile of `patrick.mp4`'s candidate pairs and body
   across 0.60 above the 99th of `messi.mp4`'s cross-shot pairs, which is deliberately
   conservative: a missing link costs a duplicate person, a wrong one invents a person who
   does not exist. No sweep has been run against a labelled set.

## Environment

- Docker with the NVIDIA Container Toolkit; `gpus: all` in `docker-compose.yml`.
  Note: `deploy.resources.reservations.devices` did **not** pass the GPU through on this
  host and silently fell back to CPU — `gpus: all` is what works.
- Base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, ~13 GB built.
- `main.py`, `config.py`, `download_models.py` and `src/` are bind-mounted, so code edits
  apply without rebuilding. **Changing `pyproject.toml` or the `Dockerfile` needs a rebuild**,
  and changing dependencies also needs `uv lock` re-run before building.
- **Dockerfile layer order matters.** `ENV` lines near the top invalidate every layer below
  them, including the ~10-minute torch install — that is why `YOLO_*` sits *after* the pip
  layers. Add new env vars low. Pip layers use BuildKit cache mounts, so editing
  `requirements.txt` reuses downloaded wheels instead of pulling ~2.5 GB again.
- There is **one** requirements file. Packages that would drag in conflicting dependencies
  (OmniShotCut, ultralytics) are installed `--no-deps` in the Dockerfile with their real
  dependencies declared in `requirements.txt`. Do not split it per stage: pip would resolve
  each file independently and could silently upgrade a pinned base package.
- OmniShotCut must be installed with upgraded pip/setuptools and `--no-deps`; see HISTORY.md.

## Test clips

`data/raw/`: `anime.mp4` (437 frames, 24fps, 7 shots, animated), `messi.mp4` (337, 30fps,
24 shots, live footage with caption overlays), `thanos.mp4` (292, 30fps, 4 shots),
`patrick.mp4` (326, 30fps, 1 shot, live street footage), `cat1.mp4` (617, 1 shot),
`cat2.mp4` (696, 4 shots).

`anime.mp4` is the shot-detection regression case — it must produce **7 shots** with a
boundary at frame 240. Six shots means the input path has regressed.

Profiler sanity checks: `messi.mp4` should give `scene=soccer` on every shot and
`text=True` (it has burned-in captions and a watermark); `patrick.mp4` should give
`scene=crosswalk` with `car`/`traffic light` objects and `text=False`.

`cat1.mp4` is the detector-confidence case. The only human in it is a hand entering from
the top right; at `yolo11n` it read as two people (the second being a pair of plush toys),
at `yolo11s` as one person plus a `teddy bear`. All three person detections sit between
0.267 and 0.276, and one keyframe shows the hand plainly and gets nothing — so the true
positives and the false positive are inseparable by threshold, and raising `--det-conf` to
0.4 would discard all of them.

06 invariants worth re-checking after any change, all verified across the six clips:
no track leaves its shot's frame range; a shot is tracked if and only if the router fired
`detection_tracking` for it; every hole in a track is wider than `--det-stride`; face boxes
lie inside their body box; crop frames are observations, never interpolated rows; and two
runs of the same command produce identical output apart from the timing fields.

08 invariants, also verified across the six clips: every track from `tracks.json` appears in
exactly one person and none appears twice; no person holds two tracks that share a frame;
`09`'s `person_records` and `10`'s `person_count` both equal `08`'s `person_count`; every
recorded link's similarity is at least the threshold it names; and `consolidated.json`,
`aggregated.json` and `fused.json` are byte-identical across two runs.

There is no ground truth for identity on these clips, so correctness of the *merges* is
checked by eye against `consolidation_preview.mp4`, which draws 08's person id over the same
boxes `tracks_preview.mp4` labels with 06's track id.
