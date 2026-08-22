"""Config passed to every stage's run()."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    video: Path
    out_root: Path  # data/processed/<video_name>
    model_dir: Path  # models/<stage>, created by the stage on first download
    device: str  # auto | cpu | cuda

    # 01-sampling
    fps: float | None  # None keeps every frame
    image_format: str  # jpg | png
    quality: int  # jpeg quality, 0 (worst) - 100 (best)
    width: int | None  # resize to this width, keeping aspect ratio

    # 02-segmentation
    shot_mode: str  # default | clean_shot
    shot_overlap: int  # frames shared between adjacent inference windows
    keyframes: int  # representative frames kept per shot

    # 03-profiler
    detector: str  # yolo11n | yolo11s | yolo11m | rtdetr-l
    det_conf: float  # detection confidence threshold
    face_conf: float  # face detection confidence threshold
    text_conf: float  # text region confidence threshold
    scenes: int  # scene labels kept per shot

    # 04-router
    router_agreement: (
        float  # soft gate: fraction of keyframes a flag must hold in; 0 = off
    )

    # 05-global-experts
    map_size: int  # longest side of every dense map written
    expert_batch: int  # keyframes per forward pass
    seg_model: str  # segformer-b0 | segformer-b1 | segformer-b2
    seg_top: int  # classes or segments kept per keyframe
    tags: int  # zero-shot tags kept per keyframe
