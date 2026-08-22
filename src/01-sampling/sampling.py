"""01-sampling: decode a video into frames with OpenCV."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from config import Config


def _codec(capture: cv2.VideoCapture) -> str:
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()


def _reset(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _params(cfg: Config) -> list[int]:
    if cfg.image_format == "png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    return [cv2.IMWRITE_JPEG_QUALITY, cfg.quality]


def run(cfg: Config) -> dict[str, Any]:
    frames_dir = _reset(cfg.out_root / "frames")

    capture = cv2.VideoCapture(str(cfg.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {cfg.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    codec = _codec(capture)
    params = _params(cfg)

    # Frames are dropped by stride rather than timestamp, so --fps is approximate.
    stride = max(1, round(source_fps / cfg.fps)) if cfg.fps and source_fps else 1

    decoded = written = 0
    size = (0, 0)
    while True:
        if not capture.grab():  # cheap: no colour conversion for frames we skip
            break
        if decoded % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                break
            if cfg.width and frame.shape[1] > cfg.width:
                height = round(frame.shape[0] * cfg.width / frame.shape[1])
                frame = cv2.resize(
                    frame, (cfg.width, height), interpolation=cv2.INTER_AREA
                )
            size = (frame.shape[1], frame.shape[0])
            # Numbered from 0 so a filename matches the frame index other stages report.
            cv2.imwrite(
                str(frames_dir / f"{written:06d}.{cfg.image_format}"), frame, params
            )
            written += 1
        decoded += 1
    capture.release()

    if not written:
        raise RuntimeError(f"decoded no frames from {cfg.video}")

    meta = {
        "video": str(cfg.video),
        "codec": codec,
        "source_fps": round(source_fps, 3),
        "decoded_frames": decoded,
        "frames_dir": frames_dir.name,  # relative to this stage's output root
        "frame_count": written,
        "frame_width": size[0],
        "frame_height": size[1],
        "stride": stride,
        "sampled_fps": round(source_fps / stride, 3) if source_fps else 0.0,
        "image_format": cfg.image_format,
    }
    (cfg.out_root / "sampling.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
