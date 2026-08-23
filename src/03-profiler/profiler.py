"""03-profiler: cheap always-on detectors over each shot's keyframes, producing the
presence flags 04-router branches on.

One detector pass gives persons and objects. Faces are found only inside person boxes, so
body presence and face presence stay distinct without a second full-frame inference. Text
presence comes from PP-OCRv3's detection stage alone. Scene belongs to 05-global-experts,
which runs a trained Places365 head; keeping a second answer here would only invite the
two to disagree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

import download_models
from config import Config

STAGE = "03-profiler"
FACE_MODEL = "face_yunet.onnx"
TEXT_MODEL = "text_ppocrv3.onnx"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

PERSON = "person"
FACE_INPUT = (320, 320)  # YuNet runs on the person crop, not the whole frame
TEXT_INPUT = (736, 736)  # multiple of 32, as the DB model requires
TEXT_MEAN = (122.68, 116.78, 103.94)


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_segmentation(out_root: Path) -> dict[str, Any]:
    path = out_root / "segmentation.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run 02-segmentation first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _detect(
    weights: Path, images: list[Array], conf: float, device: str
) -> list[list[dict[str, Any]]]:
    """One forward pass over every keyframe; persons and objects come out together."""
    from ultralytics import RTDETR, YOLO

    model = RTDETR(str(weights)) if "rtdetr" in weights.stem else YOLO(str(weights))
    results = model.predict(images, conf=conf, device=device, verbose=False)

    frames: list[list[dict[str, Any]]] = []
    for result in results:
        boxes: list[dict[str, Any]] = []
        names = result.names
        for box in result.boxes or []:
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            boxes.append(
                {
                    "label": str(names[int(box.cls[0])]),
                    "score": round(float(box.conf[0]), 3),
                    "box": xyxy,
                }
            )
        frames.append(boxes)
    return frames


def _iou(a: list[int], b: list[int]) -> float:
    wide = min(a[2], b[2]) - max(a[0], b[0])
    tall = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or tall <= 0:
        return 0.0
    overlap = wide * tall
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def _dedupe(
    boxes: list[dict[str, Any]], threshold: float = 0.4
) -> list[dict[str, Any]]:
    """Person boxes overlap, so the same face can be found in two crops."""
    kept: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda b: -float(b["score"])):
        if all(_iou(box["box"], other["box"]) < threshold for other in kept):
            kept.append(box)
    return kept


def _faces(
    detector: Any, image: Array, persons: list[dict[str, Any]], conf: float
) -> list[dict[str, Any]]:
    """YuNet inside each person box only, so a face is always tied to a body."""
    height, width = image.shape[:2]
    found: list[dict[str, Any]] = []

    for person in persons:
        x1, y1, x2, y2 = person["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue

        crop = image[y1:y2, x1:x2]
        detector.setInputSize((crop.shape[1], crop.shape[0]))
        _, detections = detector.detect(crop)
        if detections is None:
            continue

        for row in detections:
            score = float(row[14])
            if score < conf:
                continue
            fx, fy, fw, fh = (int(v) for v in row[:4])
            found.append(
                {
                    "score": round(score, 3),
                    "box": [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh],
                }
            )
    return _dedupe(found)


def _text(detector: Any, image: Array, conf: float) -> list[dict[str, Any]]:
    """DB detection stage only: where text is, not what it says."""
    boxes, scores = detector.detect(image)
    regions: list[dict[str, Any]] = []
    for quad, score in zip(boxes, scores):
        if float(score) < conf:
            continue
        points = np.asarray(quad).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        regions.append({"score": round(float(score), 3), "box": [x, y, x + w, y + h]})
    return regions


def _summarise(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """`flags` is the whole contract 04-router reads: four booleans, no class names. Class
    identity stays in `counts` as a diagnostic, because a false positive there is what makes
    a flag fire wrongly, and you cannot see that from the boolean alone."""
    total = max(1, len(keyframes))
    persons = [k["person_count"] for k in keyframes]
    faces = [k["face_count"] for k in keyframes]

    classes: dict[str, int] = {}
    seen: dict[str, int] = {}
    for keyframe in keyframes:
        for label, count in keyframe["object_counts"].items():
            classes[label] = max(classes.get(label, 0), count)
            seen[label] = seen.get(label, 0) + 1

    text_hits = sum(1 for k in keyframes if k["text_count"])
    object_hits = sum(1 for k in keyframes if k["object_counts"])
    return {
        "flags": {
            "person": any(persons),
            "face": any(faces),
            "text": bool(text_hits),
            "object": bool(classes),
        },
        "counts": {
            "person_max": max(persons, default=0),
            "face_max": max(faces, default=0),
            "object_max": max(
                (sum(k["object_counts"].values()) for k in keyframes), default=0
            ),
            "objects": dict(sorted(classes.items())),
        },
        "agreement": {
            "person": round(sum(1 for p in persons if p) / total, 3),
            "face": round(sum(1 for f in faces if f) / total, 3),
            "text": round(text_hits / total, 3),
            "object": round(object_hits / total, 3),
            # Per class, so a one-frame false positive is distinguishable from a real object.
            "objects": {
                label: round(n / total, 3) for label, n in sorted(seen.items())
            },
        },
    }


def run(cfg: Config) -> dict[str, Any]:
    segmentation = _read_segmentation(cfg.out_root)
    device = _device(cfg.device)

    weights = download_models.detector(STAGE, cfg.detector)
    face_detector = cv2.FaceDetectorYN.create(
        str(cfg.model_dir / FACE_MODEL), "", FACE_INPUT, cfg.face_conf, 0.3, 5000
    )
    text_detector = cv2.dnn.TextDetectionModel_DB(str(cfg.model_dir / TEXT_MODEL))
    text_detector.setBinaryThreshold(0.3).setPolygonThreshold(cfg.text_conf)
    text_detector.setInputParams(1.0 / 255.0, TEXT_INPUT, TEXT_MEAN, True)

    # Every keyframe of every shot goes through the detector in one batch.
    paths: list[Path] = []
    owners: list[int] = []
    for shot in segmentation["shots"]:
        for keyframe in shot["keyframes"]:
            paths.append(cfg.out_root / keyframe["path"])
            owners.append(int(shot["index"]))

    images = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError("some keyframes referenced by segmentation.json are missing")
    detections = _detect(weights, images, cfg.det_conf, device)

    per_frame: list[dict[str, Any]] = []
    for i, (image, boxes) in enumerate(zip(images, detections)):
        persons = [b for b in boxes if b["label"] == PERSON]
        objects = [b for b in boxes if b["label"] != PERSON]

        counts: dict[str, int] = {}
        for box in objects:
            counts[box["label"]] = counts.get(box["label"], 0) + 1

        faces = _faces(face_detector, image, persons, cfg.face_conf)
        text = _text(text_detector, image, cfg.text_conf)
        per_frame.append(
            {
                "frame": int(paths[i].stem.split("_")[1]),
                "person_count": len(persons),
                "face_count": len(faces),
                "object_counts": counts,
                "text_count": len(text),
                "persons": persons,
                "objects": objects,
                "faces": faces,
                "text": text,
            }
        )

    shots: list[dict[str, Any]] = []
    for shot in segmentation["shots"]:
        index = int(shot["index"])
        picked = [i for i, owner in enumerate(owners) if owner == index]
        keyframes = [per_frame[i] for i in picked]
        summary = _summarise(keyframes)
        shots.append(
            {
                "index": index,
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "start_time": shot["start_time"],
                "end_time": shot["end_time"],
                **summary,
                "keyframes": keyframes,
            }
        )

    every_class: dict[str, int] = {}
    for shot_data in shots:
        for label, count in shot_data["counts"]["objects"].items():
            every_class[label] = every_class.get(label, 0) + count

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "keyframe_count": len(per_frame),
        "detector": cfg.detector,
        "device": device,
        "video_flags": {
            "person": any(s["flags"]["person"] for s in shots),
            "face": any(s["flags"]["face"] for s in shots),
            "text": any(s["flags"]["text"] for s in shots),
            "object": any(s["flags"]["object"] for s in shots),
        },
        "object_totals": dict(sorted(every_class.items(), key=lambda kv: -kv[1])),
        "shots": shots,
    }
    (cfg.out_root / "profile.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "shots"}
