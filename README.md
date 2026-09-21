# Video AI Thesis

A modular video processing pipeline.

## Local venv

```bash
uv sync
```

Installs everything into `.venv/` on Python 3.10. Point your editor at it, or prefix
commands with `uv run`. The full pipeline still wants the container, for ffmpeg and CUDA.

## Build

```bash
docker compose build
```

## Run

```bash
# whole pipeline
docker compose run --rm pipeline data/raw/messi.mp4

# one module on its own
docker compose run --rm pipeline data/raw/messi.mp4 --stages 02-segmentation
```

Output lands in `data/processed/<video_name>/`:

```
json/            every stage's metadata
embeddings/      keyframe, face and body vectors (.npy)
previews/        rendered inspection videos
frames/          decoded frames        (01)
shots/           first frame per shot  (02)
keyframes/       representative frames (02)
global/          dense maps per keyframe (05)
tracks/          best crops per track  (06)
```

`json/`, `embeddings/` and `previews/` are created on first use.

## Options

| flag                        | default        | meaning                                                       |
| --------------------------- | -------------- | ------------------------------------------------------------- |
| `--stages <name> ...`       | all            | run a subset, always in pipeline order                        |
| `--device cpu\|cuda\|auto`  | `auto`         | where models run                                              |
| `--fps 5`                   | every frame    | sample rate for 01-sampling                                   |
| `--width 640`               | source         | resize frames, keeping aspect ratio                           |
| `--quality 95`              | `95`           | jpeg quality, 0 (worst) to 100 (best)                         |
| `--image-format png`        | `jpg`          | frame file format                                             |
| `--keyframes 3`             | `3`            | representative frames kept per shot                           |
| `--shot-mode clean_shot`    | `default`      | `clean_shot` drops transitions, keeping only cuts             |
| `--shot-overlap 30`         | `30`           | frames shared between adjacent detection windows              |
| `--detector yolo11m`        | `yolo11s`      | object detector: yolo11n/s/m or rtdetr-l                      |
| `--det-conf 0.4`            | `0.25`         | detection confidence threshold                                |
| `--face-conf 0.6`           | `0.6`          | face detection confidence threshold                           |
| `--text-conf 0.5`           | `0.5`          | text region confidence threshold                              |
| `--scenes 3`                | `3`            | scene labels kept per shot                                    |
| `--router-agreement 0.5`    | `0.0`          | soft gate: fraction of a shot's keyframes a flag must hold in |
| `--map-size 512`            | `512`          | longest side of every dense map written by 05                 |
| `--seg-model segformer-b1`  | `segformer-b1` | semantic model: segformer-b0/b1/b2                            |
| `--seg-top 10`              | `10`           | classes or segments kept per keyframe                         |
| `--tags 15`                 | `15`           | zero-shot tags kept per keyframe                              |
| `--expert-batch 8`          | `8`            | keyframes per forward pass in 05                              |
| `--tracker botsort`         | `bytetrack`    | tracker used by 06                                            |
| `--frame-source frames`     | `video`        | `frames` reads 01-sampling's output instead of decoding       |
| `--det-stride 1`            | `2`            | detect every Nth frame; the rest are interpolated             |
| `--track-low-conf 0.1`      | `0.1`          | floor for ByteTrack's second association pass                 |
| `--track-high-conf 0.25`    | `0.25`         | first association pass takes detections above this            |
| `--track-new-conf 0.25`     | `0.25`         | a new track starts only above this                            |
| `--track-buffer 30`         | `30`           | frames a lost track survives before it is dropped             |
| `--track-match 0.8`         | `0.8`          | IoU distance a match must beat                                |
| `--track-batch 16`          | `16`           | frames per detector forward pass in 06                        |
| `--crops 5`                 | `5`            | best crops kept per track, per kind                           |
| `--crop-pad 0.1`            | `0.1`          | fraction of the box added as margin                           |
| `--min-track 3`             | `3`            | tracks shorter than this are dropped                          |
| `--series-stride 2`         | `2`            | observed rows between series samples in 07                    |
| `--blendshapes 10`          | `10`           | blendshape coefficients kept per sample                       |
| `--attributes 8`            | `8`            | zero-shot face attributes kept per crop                       |
| `--consol-face-within 0.55` | `0.55`         | face cosine a link needs inside a segment                     |
| `--consol-face-across 0.45` | `0.45`         | face cosine a link needs across segments                      |
| `--consol-body-within 0.70` | `0.70`         | body cosine a link needs inside a segment                     |
| `--consol-body-across 0.60` | `0.60`         | body cosine a link needs across segments                      |
| `--consol-min-area 1600`    | `1600`         | crop pixel area below which a descriptor is not trusted       |
| `--consol-min-front 0.15`   | `0.15`         | frontality a face crop needs to contribute a descriptor       |
| `--agg-top 10`              | `10`           | labels kept per distribution in 09                            |
| `--pose-vis 0.5`            | `0.5`          | visibility a joint must reach to count in 09                  |
| `--attention-deg 30`        | `30`           | cone width within which a head counts as attending someone    |
| `--sync-min 5`              | `5`            | shared affect samples a synchrony edge needs                  |

## Preview the output

Draws what 06, 07, 08 and 10 produced back over the source video, for inspection.

```bash
# all four views
docker compose run --rm --entrypoint python3 pipeline preview.py messi

# just one
docker compose run --rm --entrypoint python3 pipeline preview.py messi --views fusion
```

Writes `data/processed/<video>/previews/<view>_preview.mp4`.

| view            | shows                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `tracks`        | 06 — person boxes, solid where detected and dashed where interpolated, faces in white, and the frames chosen as crops |
| `conditional`   | 07 — pose skeleton, head-pose axes, emotion with valence/arousal, recognised text                                     |
| `consolidation` | 08 — global person ids over the same boxes as`tracks`, so what merged is visible against it                           |
| `fusion`        | 10 — the relation graph: amber attends, green mutual, cyan dashed synchrony                                           |

## Download model weights

Runs automatically before each stage; anything already in `models/` is reused.
To prefetch everything up front:

```bash
docker compose run --rm --entrypoint python3 pipeline download_models.py
```
