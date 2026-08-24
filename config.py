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

    # 06-detection-tracking
    tracker: str  # bytetrack | botsort
    frame_source: str  # video | frames (01-sampling's output)
    det_stride: int  # detect every Nth frame; the rest are interpolated
    track_low_conf: float  # floor for ByteTrack's second association pass
    track_high_conf: float  # first pass takes detections above this
    track_new_conf: float  # a new track starts only above this
    track_buffer: int  # frames a lost track survives before it is dropped
    track_match: float  # IoU distance a match must beat
    track_batch: int  # frames per detector forward pass
    crops: int  # best crops kept per track
    crop_pad: float  # fraction of the box added as margin
    min_track: int  # tracks shorter than this are dropped

    # 07-conditional-experts
    series_stride: int  # observed rows between series samples
    blendshapes: int  # blendshape coefficients kept per sample
    attributes: int  # zero-shot attributes kept per crop
