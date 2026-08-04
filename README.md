# Video-AI-Thesis

Build the image:

```bash
docker compose build
```

Pass a full path to decode a video into frames (output to `data/processed/<video>/frame/`):

```bash
docker compose run --rm sample data/raw/clips/example.mp4
```

Segment sampled frames into shots + keyframes with OmniShotCut and MobileCLIP
(output to `data/processed/<video>/segmentation/`). Requires an NVIDIA GPU:

```bash
docker compose run --rm segment data/processed/example/frames
```

Use `--mode default` to also label transitions (dissolve, wipe, fade) instead of
keeping general cuts only.

Track faces, bodies and objects and associate them into stable identity ids
(output to `data/processed/<video>/association/`):

```bash
docker compose run --rm associate data/processed/example/frames
```

Detection runs every second frame (`--detect-every`), tracking on every frame:
in between detections the Kalman motion model carries each box forward. Two
profiles are available:

| `--profile`            | detector            | tracker       |
| ------------------------ | ------------------- | ------------- |
| `affordable` (default) | YOLO11 + YOLO-face  | ByteTrack     |
| `offline`              | RT-DETR + YOLO-face | BoT-SORT-ReID |

Trackers reset at every shot boundary from `segments.json`, since a cut breaks
motion continuity. The `offline` profile re-links people across those
boundaries using ReID embeddings; `affordable` keeps identities shot-local.

Outputs:

- `tracks.json` — shots with a per-shot headcount (`num_people`,
  `max_people_in_frame`), the identity list, and every track's lifetime.
- `tracks.jsonl` — one record per box per frame, each stamped with its
  `identity_id` and whether the box was `detected` or `predicted`.

Faces and bodies are bound into a single `person_*` identity by containment
voting over each shot, so a person seen through both streams is counted once
and fusion has one key to bind its outputs to. Objects get `object_*` ids.

## Visualizing the tracks

Add `--visualize` to the association run, or render from a finished run without
tracking again (output to `data/processed/<video>/visualization/tracks.mp4`):

```bash
docker compose run --rm associate data/processed/example/frames --visualize-only
```

Boxes are coloured by identity rather than by track, so a face and the body it
was bound to share a colour; frames carried by the motion model between
detections are drawn dashed. The overlay reads `frame`, `shot` and the number
of people in that frame.

| flag                        | effect                                              |
| --------------------------- | --------------------------------------------------- |
| `--visualize-streams`     | Comma-separated streams to draw, e.g.`face,body`. |
| `--visualize-fps`         | Playback rate override.                             |
| `--visualize-scores`      | Label each box with its confidence.                 |
| `--hide-predicted`        | Detected boxes only.                                |
| `--save-annotated-frames` | Also write the annotated frames as images.          |
