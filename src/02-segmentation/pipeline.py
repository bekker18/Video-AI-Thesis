"""End-to-end shot segmentation over sampled frames."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .keyframes import KeyframeSelector
from .shots import DEFAULT_OVERLAP, ShotDetector, load_frames

# All model weights live locally under models/segmentation/ and are loaded offline.
DEFAULT_OMNISHOTCUT_WEIGHTS = Path("models/segmentation/OmniShotCut_ckpt.pth")
DEFAULT_MOBILECLIP_WEIGHTS = Path("models/segmentation/mobileclip_s2.pt")


def segment(
    frames_dir: Path,
    output_dir: Path | None = None,
    omnishotcut_weights: Path = DEFAULT_OMNISHOTCUT_WEIGHTS,
    mobileclip_weights: Path = DEFAULT_MOBILECLIP_WEIGHTS,
    mode: str = "clean_shot",
    overlap: int = DEFAULT_OVERLAP,
    device: str = "cpu",
) -> list[dict]:
    """Segment sampled frames into shots and select keyframes."""

    frames_dir = Path(frames_dir)
    # Frames live in data/processed/<video>/frame; write segmentation output to the sibling data/processed/<video>/segmentation directory.
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else frames_dir.parent / "segmentation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ShotDetector(omnishotcut_weights, mode=mode, overlap=overlap)

    frames, paths = load_frames(frames_dir, size=detector.process_size)
    scenes, intra_labels, inter_labels = detector.detect(frames)

    selector = KeyframeSelector(mobileclip_weights, device=device)
    keyframes = selector.select(paths, scenes)

    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(exist_ok=True)

    shots: list[dict] = []
    for shot_id, ((start, end), keyframe) in enumerate(zip(scenes, keyframes)):
        source = paths[keyframe]
        saved_as = keyframe_dir / f"shot_{shot_id:04d}_{source.name}"
        shutil.copyfile(source, saved_as)
        shot = {
            "shot_id": shot_id,
            "start_frame": int(start),
            "end_frame": int(end),
            "keyframe_index": int(keyframe),
            "keyframe_file": source.name,
            "keyframe_saved_as": saved_as.name,
        }
        # Relation labels are only produced in "default" mode.
        if shot_id < len(intra_labels):
            shot["intra_label"] = intra_labels[shot_id]
        if shot_id < len(inter_labels):
            shot["inter_label"] = inter_labels[shot_id]
        shots.append(shot)

    manifest = {
        "frames_dir": str(frames_dir),
        "num_frames": len(paths),
        "num_shots": len(shots),
        "mode": mode,
        "overlap": overlap,
        "shots": shots,
    }
    (output_dir / "segments.json").write_text(json.dumps(manifest, indent=2))
    return shots
