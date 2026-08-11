# History

What was built, what was measured, and why decisions went the way they did.
For the current state see [STATE.md](STATE.md).

## Starting point

The repo was cleared to `data/raw/*.mp4` and a `.gitignore`. Everything here was written
from scratch. Requirements set up front: Python, minimal dependencies, Docker only (no
local interpreter), simple code, minimal comments, small README. `mypy --strict` was added
as a requirement partway through and applied to everything.

Module list given: 01-sampling, 02-segmentation, 03-association, 04-profiler, 05-router,
06-global-experts, 07-conditional-experts, 08-aggregation, 09-fusion, 10-consolidation,
11-vllm-synthesis.

## 01-sampling, first version (ffmpeg)

Built with ffmpeg driven through `subprocess`, chosen over OpenCV because a single native
process pipelines decode and encode, while OpenCV pays a Python round-trip per frame.

**GPU decoding was requested and turned out to be a loss.** NVDEC was benchmarked against
CPU at three resolutions:

| | 720p | 1080p | 4K |
| --- | --- | --- | --- |
| CPU | 2.7s | 5.2s | 18.9s |
| NVDEC | 4.2s | 7.4s | 23.8s |

CPU won everywhere. JPEG *encoding* is the bottleneck, not decoding, and NVDEC additionally
has to copy every frame back to host memory. `--decoder auto` was set to pick CPU, with
`--decoder cuda` available to force it. This contradicted the original instruction to use
the GPU where possible; the measurements were reported rather than the instruction followed
silently.

Related finding: writing 3260 JPEGs to the bind-mounted `data/` took 19.1s versus 8.8s to
container-local disk — the single most expensive operation in the pipeline at that point.

A GPU passthrough problem was also found: `deploy.resources.reservations.devices` in
`docker-compose.yml` silently did **not** pass the GPU through, so the container decoded on
CPU while claiming a GPU was requested. `gpus: all` works.

## Structural decisions

`argparse.Namespace` was replaced with a frozen `Config` dataclass, because `Namespace`
types every attribute as `Any` and mypy could say nothing about it.

Stage files were renamed from a uniform `module.py` to `<stage-name>.py`, because directory
names like `01-sampling` are not valid Python module names and eleven identical basenames
collide under mypy.

`download_models.py` was added on request: a stage→weights registry that skips anything
already present. This replaced an earlier plan to juggle `HF_HOME`, and is better — stages
now receive explicit paths and `cache_dir` arguments, so there is no ambiguity about where
weights land.

## 02-segmentation

Models specified: OmniShotCut for shot detection, MobileCLIP ("ClipMobile") for embeddings.
Both were verified to exist and their source was read rather than trusting the docs —
which mattered, because the published API description was incomplete.

**Two packaging problems.** `pip install git+…OmniShotCut` silently produced an empty wheel
named `UNKNOWN-0.0.0`: Ubuntu's pip 22 is too old for the repo's PEP 621 metadata. Upgrading
pip/setuptools before installing fixes it. It then failed on `import cv2`; installing with
`--no-deps` alongside `opencv-python-headless` avoids pulling in `libGL`.

### First version, and why it was wrong

The first version read 01-sampling's extracted JPEGs. It produced two visible defects:

1. **Keyframes were consecutive** — `21,22,23`, `57,58,59`. Selection took the top-k frames
   by cosine similarity to the shot's mean embedding, and the k frames nearest a centroid
   are inevitably adjacent. One instant sampled three times.
2. **A real cut was missed.** Independent ffmpeg scene detection found a cut at frame 240.
   OmniShotCut found it when given the video file (7 shots) and missed it when given the
   extracted frames (6 shots). Tested at 256px and native 1276px — **identical results**,
   so resolution and JPEG quality were not the cause.

Also found: shots shared a boundary frame (`0-33` then `33-108`), the model's decoder saw
438 frames where ffmpeg extracted 437, and `clean_shot` mode was silently discarding any
shot not labelled `general`.

### Reworked

Rewritten to read the video directly. Keyframe selection replaced with: split the shot into
k equal time spans, take one per span, ranking by similarity to the span mean and then
picking the sharpest of the top 25%, skipping near-duplicates. `--shot-mode` default changed
to `default` so transitions are no longer silently dropped.

Result on `anime.mp4`: 7 shots, cut at 240 recovered, keyframes `5,17,28` instead of
`21,22,23`. Verified visually — three genuinely different moments from the same shot.

A bug was introduced and caught in the same pass: `_write_frames` counted distinct frame
indices rather than output paths, so a frame that was both a shot's first frame and a
keyframe caused the loop to exit one write early. 19 keyframes recorded, 18 on disk.

## 03-association: designed, then dropped

Specified as person/face/object detection and tracking with YOLO or RT-DETR plus ByteTrack
or BoT-SORT. Design notes produced but no code written. Key points raised:

- **The tracker must reset at every shot boundary.** Kalman + IoU association is meaningless
  across a cut; a tracker run over a whole video will confidently carry IDs across cuts.
- The module is really two stages: within-shot tracking producing tracklets, then cross-shot
  association by appearance embedding producing stable identities. The second is what later
  modules actually need.
- **Keyframes-only tracking is structurally impossible**, not merely worse — with seconds
  between keyframes, IoU is zero and the motion model has nothing to work with. It would be
  appearance clustering, a different algorithm.
- COCO has `person` but no `face`; faces need a separate detector and embedding.

The module was then removed from the pipeline entirely. This mattered beyond itself: dense
frames for tracking were the main remaining justification for 01-sampling.

## ffmpeg vs OpenCV, measured

| | ffmpeg | OpenCV |
| --- | --- | --- |
| decode only (3260 frames) | 1.73s | 1.91s |
| extract to JPEG, size-matched | 5.7s / 419MB | 8.4s / 411MB |
| same fidelity, file size | 159KB @ 44.5dB | 226KB @ 44.5dB |
| subsample every 15th | 1.82s | 1.74s |
| extract + resize to 640 | 3.4s | 8.1s |

Decode speed is a tie because OpenCV wraps FFmpeg's libavcodec. ffmpeg wins clearly at
writing JPEGs. OpenCV wins at subsampling, because `grab()` skips colour conversion on
discarded frames, and — more importantly — hands frames over as numpy arrays.

**One correction was issued during this work.** An initial measurement showed ffmpeg at
33 dB against OpenCV's 41.8 dB at equal file size, which was reported as a large encoder
quality gap. That was wrong: it was an artifact of the video→JPEG colour path, with a
systematic `-1.23` brightness bias. Encoding identical PNG input, the two encoders are
within 0.1 dB. The cause of the bias was never pinned down.

## Everything moved to OpenCV

On request, ffmpeg was dropped entirely and OmniShotCut is now fed a decoded array rather
than a file path.

**This reintroduced the missed cut**, at any resolution — confirming the cause was never
resolution but the frame count: OpenCV decodes 437 frames, ffmpeg's raw pipe emits 438,
shifting the inference window alignment. Rather than accept the regression, the `overlap`
parameter was swept:

| overlap | anime | tanos | messi |
| --- | --- | --- | --- |
| 20 (model default) | 6 shots, misses 240 | match | match |
| 30 | **7 shots, finds 240** | match | match |
| 40 | 7 shots, finds 240 | match | match |

`--shot-overlap` now defaults to 30. Padding the array with a duplicate final frame did
*not* recover the cut, confirming window alignment rather than the missing frame itself.

CLI consequences: `--quality` changed from ffmpeg's inverted 2–31 scale to OpenCV's 0–100,
and `--decoder` was removed — the standard OpenCV wheel has no NVDEC path.

## The frame-numbering bug

01-sampling wrote files starting at `000001` while every frame index in the pipeline is
0-based, so `000034.jpg` was frame 33. This caused a real misreading of shot boundaries —
a report that "frame 33 is still part of the 0-32 shot" was, correctly, a symptom of this.
Files now start at `000000`.

The boundary itself was verified visually and was correct: frame 32 is a close-up, 33 and
34 are the wide shot, so the cut sits between 32 and 33.

## Optimising 02

Profiled before changing anything:

| | time | share |
| --- | --- | --- |
| MobileCLIP forward | 26.8s | 41.5% |
| PIL preprocessing | 18.6s | 28.9% |
| sharpness | 8.7s | 13.6% |
| resize for shot detection | 7.0s | 10.9% |
| **decode** | **2.5s** | **3.9%** |

Decoding — the subject of most of the project's benchmarking — was 4% of this module.

Three changes: one downscale per frame reused by all three consumers instead of three
separate resizes; OpenCV preprocessing replacing PIL, with crop size and normalisation read
out of open_clip's transform rather than hardcoded; and fp16 autocast with batch 128.

**65.7s → 34.2s**, with identical shot boundaries, embeddings at cosine 0.996, and
sharpness rank correlation 0.977. fp32-optimised scored 0.99625 and fp16 scored 0.99626,
so the small difference comes from cv2-vs-PIL resampling, not from fp16.

## 03-profiler

Specified as cheap always-on trigger detectors feeding a router: one multi-class detector
for persons and objects, a light face detector run only inside person boxes, PaddleOCR's
detection stage for text presence, and Places365 for scene. Keyframes only.

Two of the four proposed models were replaced with cheaper equivalents, both accepted:

- **Scene** became zero-shot MobileCLIP against the Places365 category names, reusing the
  embeddings 02 already saves. No scene model is downloaded at all.
- **Text** became OpenCV's DB detector — which turned out not to be an alternative to
  PaddleOCR but the *same thing*: the model in OpenCV's zoo is PP-OCRv3's detection stage
  exported to ONNX. That gives the specified detector without the paddlepaddle dependency,
  which would have risked cuDNN conflicts against torch's CUDA 12.4 runtime.

Modules were renumbered at this point to close the gap left by 03-association.

**A dependency conflict was found and fixed.** ultralytics requires `opencv-python` (the
GUI build), so installing it normally put opencv-python 5.0.0.93 alongside
opencv-python-headless 4.10.0.84 — two `cv2` installations of different major versions,
with import order deciding which one wins. It is now installed `--no-deps`, the same
treatment OmniShotCut already needed.

**A requirements split was tried and reverted.** Rebuilds were taking ten minutes, so
per-stage packages were briefly moved to a second requirements file. The real cause was
different: a `YOLO_CONFIG_DIR` env var had been added to the `ENV` block at the top of the
Dockerfile, invalidating every layer beneath it including the torch install. Moving the env
var below the pip layers fixed it, the second file was deleted, and BuildKit cache mounts
were added so editing `requirements.txt` no longer re-downloads 2.5 GB.

**One bug fixed during verification.** Faces are searched inside each person box
independently, so overlapping person boxes could return the same face twice. Deduplicated
by IoU.

Results on `messi.mp4`: 24 shots, `scene=soccer` on every one, text correctly flagged (the
clip has burned-in captions and a "TURZO HD" watermark, verified by eye). `patrick.mp4`
gives `scene=crosswalk` with cars and a traffic light — correct, and worth noting because
it was wrongly assumed to be animated; it is live street footage. Faces on `anime.mp4` were
predicted to fail and did not, scoring 0.88–0.92.

Object false positives (`banana`, `surfboard` in a soccer clip) are visible at
`--det-conf 0.25` with `yolo11n`.

### Splitting the routing surface from the record

The proposal was made to drop object class names entirely, on the grounds that the router
fires conditional experts regardless of which class was detected. Half of that was adopted.

Routing on class identity would indeed be bad — it needs an arbitrary 80-class → expert
mapping and is brittle. So `flags` became four plain booleans and the router never sees a
class name.

Deleting the classes was argued against, and the argument was data-led: on `messi.mp4`,
roughly half the object-triggered shots fire on `surfboard`, `banana`, `baseball glove` or
`tie`. Removing the names removes **no** false trigger — `banana` still sets
`object=true` — it only removes the ability to notice. So class names stayed in `counts`
and `agreement.objects` as diagnostics.

One assumption was checked and turned out wrong: the object flag was expected to be true
almost everywhere and therefore useless. It fires on 19 of 36 shots, so it does discriminate.

A detector sweep followed. The notable result is that **raising the confidence threshold
does not reduce the false-positive share on yolo11n** — implausible detections stay around
35% from conf 0.25 to 0.55, it simply detects less of everything. Model size is the lever:
yolo11s cuts the implausible share to 23% at conf 0.25 and 17% at 0.4, for about 0.2s extra
across all four clips. Defaults were left unchanged by request, to be settled later.

A `--min-agreement` filter was also considered and rejected: single-frame detections
outnumber recurring ones even for plausible classes, so it would suppress real brief
objects. Per-class agreement is recorded so the idea can be revisited against data.

## Weight caching

OmniShotCut pulls a resnet18 ImageNet backbone through torch hub, which cached to
`/root/.cache/torch` inside the container and was therefore re-downloaded (44.7 MB) on every
run. `TORCH_HOME=/app/models/torch` now points it at the mounted `models/` directory, and
`download_models.py` gained a `BACKBONES` registry so it can be prefetched.

The two real checkpoints — OmniShotCut (156.5 MB) and MobileCLIP-S2 (380 MB) — were already
cached correctly and were not the cause.
