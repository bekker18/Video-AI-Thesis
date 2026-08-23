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

Output lands in `data/processed/<video_name>/`.

## Options

| flag                         | default          | meaning                                                       |
| ---------------------------- | ---------------- | ------------------------------------------------------------- |
| `--stages <name> ...`      | all              | run a subset, always in pipeline order                        |
| `--device cpu\|cuda\|auto`   | `auto`         | where models run                                              |
| `--fps 5`                  | every frame      | sample rate for 01-sampling                                   |
| `--width 640`              | source           | resize frames, keeping aspect ratio                           |
| `--quality 95`             | `95`           | jpeg quality, 0 (worst) to 100 (best)                         |
| `--image-format png`       | `jpg`          | frame file format                                             |
| `--keyframes 3`            | `3`            | representative frames kept per shot                           |
| `--shot-mode clean_shot`   | `default`      | `clean_shot` drops transitions, keeping only cuts           |
| `--shot-overlap 30`        | `30`           | frames shared between adjacent detection windows              |
| `--detector yolo11s`       | `yolo11n`      | object detector: yolo11n/s/m or rtdetr-l                      |
| `--det-conf 0.4`           | `0.25`         | detection confidence threshold                                |
| `--router-agreement 0.5`   | `0.0`          | soft gate: fraction of a shot's keyframes a flag must hold in |
| `--map-size 512`           | `512`          | longest side of every dense map written by 05                 |
| `--seg-model segformer-b1` | `segformer-b1` | semantic model: segformer-b0/b1/b2                            |
| `--seg-top 10`             | `10`           | classes or segments kept per keyframe                         |
| `--tags 15`                | `15`           | zero-shot tags kept per keyframe                              |
| `--expert-batch 8`         | `8`            | keyframes per forward pass in 05                              |
| `--tracker botsort`        | `bytetrack`    | tracker used by 06                                            |
| `--frame-source frames`    | `video`        | `frames` reads 01-sampling's output instead of decoding       |
| `--det-stride 1`           | `2`            | detect every Nth frame; the rest are interpolated             |
| `--track-low-conf 0.1`     | `0.1`          | floor for ByteTrack's second association pass                 |
| `--track-high-conf 0.25`   | `0.25`         | first association pass takes detections above this            |
| `--track-new-conf 0.25`    | `0.25`         | a new track starts only above this                            |
| `--track-buffer 30`        | `30`           | frames a lost track survives before it is dropped             |
| `--track-match 0.8`        | `0.8`          | IoU distance a match must beat                                |
| `--track-batch 16`         | `16`           | frames per detector forward pass in 06                        |
| `--crops 5`                | `5`            | best crops kept per track, per kind                           |
| `--crop-pad 0.1`           | `0.1`          | fraction of the box added as margin                           |
| `--min-track 3`            | `3`            | tracks shorter than this are dropped                          |

## Download model weights

Runs automatically before each stage; anything already in `models/` is reused.
To prefetch everything up front:

```bash
docker compose run --rm --entrypoint python3 pipeline download_models.py
```
