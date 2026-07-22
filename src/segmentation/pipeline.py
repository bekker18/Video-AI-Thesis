"""End-to-end shot segmentation over sampled frames."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .keyframes import KeyframeSelector
from .shots import ShotDetector, load_frames

# All model weights live locally under models/segmentation/ and are loaded offline.
DEFAULT_TRANSNET_WEIGHTS = Path("models/segmentation/transnetv2-pytorch-weights.pth")
DEFAULT_MOBILECLIP_WEIGHTS = Path("models/segmentation/mobileclip_s2.pt")


def segment(
    frames_dir: Path,
    output_dir: Path | None = None,
    transnet_weights: Path = DEFAULT_TRANSNET_WEIGHTS,
    mobileclip_weights: Path = DEFAULT_MOBILECLIP_WEIGHTS,
    threshold: float = 0.5,
    device: str = "cpu",
) -> list[dict]:
    """Segment sampled frames into shots and select keyframes."""

    frames_dir = Path(frames_dir)
    # Frames live in data/processed/<video>/frame; write segmentation output to
    # the sibling data/processed/<video>/segmentation directory.
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else frames_dir.parent / "segmentation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, paths = load_frames(frames_dir)

    detector = ShotDetector(transnet_weights, device=device)
    scenes = detector.detect(frames, threshold=threshold)

    selector = KeyframeSelector(mobileclip_weights, device=device)
    keyframes = selector.select(paths, scenes)

    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(exist_ok=True)

    shots: list[dict] = []
    for shot_id, ((start, end), keyframe) in enumerate(zip(scenes, keyframes)):
        source = paths[keyframe]
        saved_as = keyframe_dir / f"shot_{shot_id:04d}_{source.name}"
        shutil.copyfile(source, saved_as)
        shots.append(
            {
                "shot_id": shot_id,
                "start_frame": int(start),
                "end_frame": int(end),
                "keyframe_index": int(keyframe),
                "keyframe_file": source.name,
                "keyframe_saved_as": saved_as.name,
            }
        )

    manifest = {
        "frames_dir": str(frames_dir),
        "num_frames": len(paths),
        "num_shots": len(shots),
        "threshold": threshold,
        "shots": shots,
    }
    (output_dir / "segments.json").write_text(json.dumps(manifest, indent=2))
    return shots
