# Video-AI-Thesis

Build the image:

```bash
docker compose build
```

Pass a full path to decode a video into frames (output to `data/processed/<video>/frame/`):

```bash
docker compose run --rm decode data/raw/clips/example.mp4
```

Segment sampled frames into shots + keyframes (output to `data/processed/<video>/segmentation/`):

```bash
docker compose run --rm segment data/processed/example/frames
```
