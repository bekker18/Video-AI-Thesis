# Video AI Thesis

A modular video processing pipeline. See [STATE.md](STATE.md) for how it works and
[HISTORY.md](HISTORY.md) for how it got here.

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

| flag                       | default     | meaning                                                       |
| -------------------------- | ----------- | ------------------------------------------------------------- |
| `--stages <name> ...`    | all         | run a subset, always in pipeline order                        |
| `--device cpu\|cuda\|auto` | `auto`    | where models run                                              |
| `--fps 5`                | every frame | sample rate for 01-sampling                                   |
| `--width 640`            | source      | resize frames, keeping aspect ratio                           |
| `--quality 95`           | `95`      | jpeg quality, 0 (worst) to 100 (best)                         |
| `--image-format png`     | `jpg`     | frame file format                                             |
| `--keyframes 3`          | `3`       | representative frames kept per shot                           |
| `--shot-mode clean_shot` | `default` | `clean_shot` drops transitions, keeping only cuts           |
| `--shot-overlap 30`      | `30`      | frames shared between adjacent detection windows              |
| `--detector yolo11s`     | `yolo11n` | object detector: yolo11n/s/m or rtdetr-l                      |
| `--det-conf 0.4`         | `0.25`    | detection confidence threshold                                |
| `--router-agreement 0.5` | `0.0`     | soft gate: fraction of a shot's keyframes a flag must hold in |

## Download model weights

Runs automatically before each stage; anything already in `models/` is reused.
To prefetch everything up front:

```bash
docker compose run --rm --entrypoint python3 pipeline download_models.py
```

## Type check

```bash
docker compose run --rm --entrypoint mypy pipeline
```
