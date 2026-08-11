"""02-segmentation: split a video into shots with OmniShotCut, then pick representative
keyframes per shot with MobileCLIP. Reads the video directly; 01-sampling is not required."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

from config import Config

SHOT_REPO = "uva-cv-lab/OmniShotCut"
SHOT_CKPT = "OmniShotCut_ckpt.pth"
CLIP_MODEL = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

BATCH = 128
DUPLICATE = 0.98  # cosine above this counts as the same picture
SHARP_POOL = 0.25  # fraction of the most representative frames to pick the sharpest from


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_shot_model(checkpoint: Path) -> tuple[Any, tuple[int, int]]:
    """Also reports the resolution the model works at, so frames are only resized once."""
    import omnishotcut

    model = omnishotcut.load(str(checkpoint))
    args = getattr(model, "_model_args", None)
    size = (int(getattr(args, "process_width", 128)), int(getattr(args, "process_height", 96)))
    return model, size


def _detect_shots(model: Any, frames: Array, mode: str, overlap: int) -> tuple[list[list[int]], list[str], list[str]]:
    result = model.inference(frames, mode=mode, overlap=overlap)
    if mode == "clean_shot":
        return list(result), [], []
    ranges, intra, inter = result
    return list(ranges), list(intra), list(inter)


def _normalise(ranges: list[list[int]], frame_count: int) -> list[tuple[int, int]]:
    """OmniShotCut reports a shared frame between neighbours ([0,33] then [33,108]) and can
    run one past the end. Make the ranges disjoint and in bounds."""
    shots: list[tuple[int, int]] = []
    ordered = sorted((int(s), int(e)) for s, e in ranges)
    for i, (start, end) in enumerate(ordered):
        if i + 1 < len(ordered):
            end = min(end, ordered[i + 1][0] - 1)
        start, end = max(0, start), min(end, frame_count - 1)
        if end >= start:
            shots.append((start, end))
    return shots or [(0, frame_count - 1)]


def _sharpness(frame: Array) -> float:
    """Variance of the Laplacian; low means blurred or mid-motion."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


def _preprocess_spec(preprocess: Any) -> tuple[int, list[float], list[float]]:
    """Read the crop size and normalisation out of open_clip's transform, so the same
    preprocessing can be done with OpenCV instead of PIL."""
    crop, mean, std = 256, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    for step in getattr(preprocess, "transforms", []):
        size = getattr(step, "size", None)
        if isinstance(size, (tuple, list)):
            crop = int(size[0])
        if hasattr(step, "mean") and hasattr(step, "std"):
            mean, std = list(step.mean), list(step.std)
    return crop, mean, std


def _scan(video: Path, device: str, cache_dir: Path, shot_size: tuple[int, int]) -> tuple[Array, Array, Array, float]:
    """Single decode pass. Each frame feeds three things: the small array OmniShotCut runs on,
    a MobileCLIP embedding, and a sharpness score. Full frames are never accumulated, so
    memory stays flat regardless of video length."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, cache_dir=str(cache_dir))
    model = model.to(device).eval()
    crop, mean, std = _preprocess_spec(preprocess)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0

    offset = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    scale = torch.tensor(std, device=device).view(1, 3, 1, 1)

    shot_frames: list[Array] = []
    vectors: list[Array] = []
    sharp: list[float] = []
    batch: list[Array] = []

    def flush() -> None:
        if not batch:
            return
        stacked = torch.from_numpy(np.stack(batch)).to(device)
        tensor = stacked.permute(0, 3, 1, 2).float().div_(255).sub_(offset).div_(scale)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            features = model.encode_image(tensor)
        features = features.float()
        features /= features.norm(dim=-1, keepdim=True)
        vectors.append(features.cpu().numpy())
        batch.clear()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # One downscale of the full frame, reused by all three consumers below.
        height, width = frame.shape[:2]
        ratio = crop / min(height, width)
        mid = cv2.resize(frame, (round(width * ratio), round(height * ratio)),
                         interpolation=cv2.INTER_AREA)

        top, left = (mid.shape[0] - crop) // 2, (mid.shape[1] - crop) // 2
        square = mid[top : top + crop, left : left + crop]
        batch.append(np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB)))

        shot_frames.append(cv2.cvtColor(cv2.resize(mid, shot_size, interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB))
        sharp.append(_sharpness(mid))

        if len(batch) == BATCH:
            flush()
    flush()
    capture.release()

    if not shot_frames:
        raise RuntimeError(f"decoded no frames from {video}")

    return (np.asarray(shot_frames, dtype=np.uint8), np.concatenate(vectors),
            np.asarray(sharp), fps)


def _select(embeddings: Array, sharp: Array, start: int, end: int, k: int) -> list[int]:
    """Split the shot into k spans and take one frame from each, so the keyframes cover the
    whole shot instead of clustering on a single instant. Within a span, choose the sharpest
    of the frames that best match that span's average appearance."""
    length = end - start + 1
    count = max(1, min(k, length))
    bounds = np.linspace(0, length, count + 1).astype(int)

    chosen: list[int] = []
    for i in range(count):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue

        span = embeddings[start + lo : start + hi]
        centre = span.mean(axis=0)
        norm = float(np.linalg.norm(centre))
        if norm:
            centre = centre / norm

        order = np.argsort(-(span @ centre))
        pool = order[: max(1, round(len(order) * SHARP_POOL))]
        ranked = sorted(pool, key=lambda j: -sharp[start + lo + int(j)])

        for j in list(ranked) + list(order):
            frame = start + lo + int(j)
            if all(float(embeddings[frame] @ embeddings[c]) < DUPLICATE for c in chosen):
                chosen.append(frame)
                break

    return chosen or [start]


def _write_frames(video: Path, wanted: dict[int, list[Path]]) -> None:
    """Second decode pass, writing only the frames that were selected."""
    capture = cv2.VideoCapture(str(video))
    index = 0
    remaining = sum(len(paths) for paths in wanted.values())  # a frame can have two names
    while remaining:
        ok, frame = capture.read()
        if not ok:
            break
        for path in wanted.get(index, ()):
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            remaining -= 1
        index += 1
    capture.release()


def _reset(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run(cfg: Config) -> dict[str, Any]:
    device = _device(cfg.device)
    shot_model, shot_size = _load_shot_model(cfg.model_dir / SHOT_CKPT)

    shot_frames, embeddings, sharp, fps = _scan(cfg.video, device, cfg.model_dir, shot_size)
    frame_count = len(shot_frames)

    ranges, intra, inter = _detect_shots(shot_model, shot_frames, cfg.shot_mode, cfg.shot_overlap)
    detected = len(ranges)
    shots_range = _normalise(ranges, frame_count)
    del shot_frames

    shots_dir = _reset(cfg.out_root / "shots")
    keys_dir = _reset(cfg.out_root / "keyframes")

    wanted: dict[int, list[Path]] = {}
    shots: list[dict[str, Any]] = []
    keep: list[int] = []

    for index, (start, end) in enumerate(shots_range):
        first = shots_dir / f"{index:04d}.jpg"
        wanted.setdefault(start, []).append(first)

        keyframes: list[dict[str, Any]] = []
        for frame in _select(embeddings, sharp, start, end, cfg.keyframes):
            path = keys_dir / f"{index:04d}_{frame:06d}.jpg"
            wanted.setdefault(frame, []).append(path)
            keyframes.append({
                "frame": frame,
                "time": round(frame / fps, 3),
                "path": f"{keys_dir.name}/{path.name}",
                "sharpness": round(float(sharp[frame]), 1),
            })
            keep.append(frame)

        shot: dict[str, Any] = {
            "index": index,
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round((end + 1) / fps, 3),
            "first_frame": f"{shots_dir.name}/{first.name}",
            "keyframes": keyframes,
        }
        if index < len(intra):
            shot["intra_label"] = intra[index]
            shot["inter_label"] = inter[index]
        shots.append(shot)

    _write_frames(cfg.video, wanted)
    np.save(cfg.out_root / "keyframe_embeddings.npy", embeddings[keep])

    meta = {
        "video": str(cfg.video),
        "shot_count": len(shots),
        "keyframe_count": len(keep),
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "shot_mode": cfg.shot_mode,
        "shot_overlap": cfg.shot_overlap,
        "shot_model": SHOT_REPO,
        "embed_model": CLIP_MODEL,
        "device": device,
        "shots": shots,
    }
    if detected != len(shots):
        meta["dropped_ranges"] = detected - len(shots)
    (cfg.out_root / "segmentation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {k: v for k, v in meta.items() if k != "shots"}
