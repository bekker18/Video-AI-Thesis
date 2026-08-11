# Project State

Current state of the codebase. Read this first for full context.

## What this is

A modular video processing pipeline for a thesis. Input is `.mp4`. The pipeline is a
sequence of numbered modules; each writes to `data/processed/<video_name>/` and later
modules read what earlier ones wrote.

Planned modules: `01-sampling`, `02-segmentation`, `03-profiler`, `04-router`,
`05-global-experts`, `06-conditional-experts`, `07-aggregation`, `08-fusion`,
`09-consolidation`, `10-vllm-synthesis`.

An `03-association` module (detection + tracking) was specified, designed, then dropped
before any code was written. The remaining modules were renumbered to close the gap, so
what HISTORY.md calls `04-profiler` is now `03-profiler`.

**Implemented: 01, 02 and 03.** Everything from 04-router onward is unwritten.

04-router will fire experts based on 03-profiler's presence flags. That is the only
consumer of `profile.json`, so its `flags` block is designed to be a flat predicate.

## Conventions

- **Python 3.10, no local interpreter — everything runs in Docker.**
- `mypy --strict` must pass with no `type: ignore`.
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
models/<stage>/                     weights for that stage
models/torch/                       torch hub cache (TORCH_HOME)
src/<stage>/<name>.py               stage implementation, exposes run(cfg)
config.py                           Config dataclass passed to every stage
download_models.py                  weight registry + downloader
main.py                             orchestrator
```

## How the orchestrator works

`main.py` holds `STAGES` in pipeline order. For each requested stage it builds a frozen
`Config`, calls `download_models.ensure(stage)`, then loads the stage module **by file
path** with `importlib` and calls `run(cfg)`.

Stage files are named after the stage minus its number, with hyphens turned into
underscores: `01-sampling` → `src/01-sampling/sampling.py`, `06-global-experts` →
`src/06-global-experts/global_experts.py`. This is deliberate — directory names like
`01-sampling` are not valid Python module names, and eleven files all called `module.py`
would collide under mypy.

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

## Known open items

1. **01-sampling has no consumer.** Delete, or make opt-in.
2. **Two decodes in 02.** Could be one via `CAP_PROP_POS_FRAMES` seeking, but OpenCV seeking
   on H.264 is off-by-one-prone and correctness matters more here.
3. **MobileCLIP embeds every frame** — ~60% of remaining runtime. A `--keyframe-stride`
   would nearly halve it at the cost of a smaller candidate pool.
4. **`clean_shot` vs `default` boundaries** disagree slightly on a short ambiguous segment
   in `anime.mp4` (192–198 vs 192–195). Not investigated.
5. **ffmpeg is still in the image** — only because `omnishotcut` imports `ffmpeg-python`
   at module load. The pipeline never shells out to it.
6. **Object detector defaults not tuned.** The sweep says `yolo11s --det-conf 0.4` roughly
   halves false positives at no runtime cost. Deliberately deferred — defaults are still
   `yolo11n` / `0.25`. Worth settling before 04-router leans on `flags.object`.
7. **04-router is next.** It reads `flags` and nothing else.

## Environment

- Docker with the NVIDIA Container Toolkit; `gpus: all` in `docker-compose.yml`.
  Note: `deploy.resources.reservations.devices` did **not** pass the GPU through on this
  host and silently fell back to CPU — `gpus: all` is what works.
- Base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, ~13 GB built.
- `main.py`, `config.py`, `download_models.py` and `src/` are bind-mounted, so code edits
  apply without rebuilding. **Changing `requirements.txt` or the `Dockerfile` needs a rebuild.**
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
24 shots, live footage with caption overlays), `tanos.mp4` (292, 30fps, 4 shots),
`patrick.mp4` (326, 30fps, 1 shot, live street footage).

`anime.mp4` is the shot-detection regression case — it must produce **7 shots** with a
boundary at frame 240. Six shots means the input path has regressed.

Profiler sanity checks: `messi.mp4` should give `scene=soccer` on every shot and
`text=True` (it has burned-in captions and a watermark); `patrick.mp4` should give
`scene=crosswalk` with `car`/`traffic light` objects and `text=False`.
