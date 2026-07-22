# Video-AI-Thesis

Pass a full path to decode a video into frames:

```bash
docker compose run --rm decode data/raw/clips/example.mp4 - data/processed/example
```

Segment sampled frames into shots + keyframes (output to `data/processed/<video>/segmentation/`):

```bash
docker compose run --rm segment data/processed/input_example
```
