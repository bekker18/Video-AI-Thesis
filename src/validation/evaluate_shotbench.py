"""12's shot-size rule on ShotBench's shot-size questions (Liu et al., NeurIPS 2025), answered as multiple choice.
After python3 -m src.tools.download_shotbench: evaluate_shotbench.py detect, features, then score --out <report>"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main import load_stage, parse_args

AGG: Any = load_stage("09-aggregation")  # its framing measures
KIT: Any = load_stage("12-identikit")  # its shot-size rule

RAW = ROOT / "data" / "raw" / "shotbench"
CACHE = ROOT / "data" / "validation" / "shot"
DETECTIONS = CACHE / "shotbench_detections.json"
FEATURES = CACHE / "shotbench_features.json"
# ShotBench's sizes from tightest to widest, in the order of 12's SIZES
SIZES = (
    "Extreme Close Up",
    "Close Up",
    "Medium Close Up",
    "Medium",
    "Medium Wide",
    "Wide",
    "Extreme Wide",
)
# shot size accuracy (%), ShotBench's Table 2 and project page
PUBLISHED = {
    "ShotVL-7B (trained on ShotQA)": 81.2,
    "ShotVL-3B (trained on ShotQA)": 77.9,
    "Qwen2.5-VL-72B-Instruct": 75.1,
    "GPT-4o": 69.3,
    "Qwen2.5-VL-7B-Instruct": 69.1,
    "InternVL3-38B": 68.0,
    "Qwen2.5-VL-32B-Instruct": 62.3,
    "Gemini-2.5-flash-preview-04-17": 57.7,
    "Qwen2.5-VL-3B-Instruct": 54.6,
    "Gemini-2.0-flash": 48.9,
    "random choice of 4": 25.0,
}


def _settings() -> argparse.Namespace:
    """main.py's defaults, so 06 runs here as the pipeline runs it."""
    argv = sys.argv
    sys.argv = ["main.py", "clip.mp4"]
    try:
        return parse_args()
    finally:
        sys.argv = argv


def _questions() -> list[dict[str, Any]]:
    csv.field_size_limit(10**8)
    with (RAW / "test.tsv").open(encoding="utf-8") as f:
        rows = [
            r for r in csv.DictReader(f, delimiter="\t") if r["category"] == "shot size"
        ]
    return [
        {
            "index": r["index"],
            "image": json.loads(r["path"].replace("'", '"'))[0],
            "options": json.loads(r["options"]),
            "answer": r["answer"],
        }
        for r in rows
    ]


def detect() -> None:
    """06's detector and face pass on each still: a person counts where 06 would start a track."""
    from ultralytics import RTDETR, YOLO

    from src.tools import download_models

    det: Any = load_stage("06-detection-tracking")
    settings = _settings()
    device = det._device(settings.device)
    weights = download_models.detector(det.PROFILER, settings.detector)
    model = RTDETR(str(weights)) if "rtdetr" in weights.stem else YOLO(str(weights))
    det._start_yunet(
        str(download_models.stage_dir(det.PROFILER) / det.FACE_MODEL),
        settings.face_conf,
    )
    found: dict[str, Any] = {}
    for q in _questions():
        image = cv2.imread(str(RAW / q["image"]))
        height, width = image.shape[:2]
        result = model.predict(
            [image], conf=settings.track_low_conf, device=device, verbose=False
        )[0]
        persons, _ = det._split(result.boxes, result.names, settings.det_conf)
        rows: list[dict[str, Any]] = []
        for (cx, cy, w, h), score in zip(persons.xywh, persons.conf):
            if float(score) < settings.track_new_conf:
                continue
            body = [
                round(float(cx - w / 2)),
                round(float(cy - h / 2)),
                round(float(cx + w / 2)),
                round(float(cy + h / 2)),
            ]
            crop = det._face_crop(image, body)
            face = (
                det._face_from(det._yunet(crop[0]), crop, settings.face_conf)
                if crop
                else None
            )
            rows.append(
                {
                    "body": body,
                    "body_score": round(float(score), 3),
                    "face": face["box"] if face else None,
                }
            )
        found[q["image"]] = {"size": [width, height], "rows": rows}
    CACHE.mkdir(parents=True, exist_ok=True)
    DETECTIONS.write_text(json.dumps(found), encoding="utf-8")
    print(
        f"{len(found)} images, {sum(1 for v in found.values() if v['rows'])} with a person",
        flush=True,
    )


def _vector(
    model: Any,
    image: np.ndarray[Any, Any],
    crop: int,
    mean: list[float],
    std: list[float],
    device: str,
) -> np.ndarray[Any, Any]:
    """MobileCLIP's image vector, preprocessed as 02's _scan does: short side to the crop, centre square, normalised."""
    import torch

    height, width = image.shape[:2]
    ratio = crop / min(height, width)
    mid = cv2.resize(
        image,
        (round(width * ratio), round(height * ratio)),
        interpolation=cv2.INTER_AREA,
    )
    top, left = (mid.shape[0] - crop) // 2, (mid.shape[1] - crop) // 2
    square = cv2.cvtColor(mid[top : top + crop, left : left + crop], cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(square))
        .to(device)[None]
        .permute(0, 3, 1, 2)
        .float()
        .div_(255)
    )
    tensor = tensor.sub_(torch.tensor(mean, device=device).view(1, 3, 1, 1)).div_(
        torch.tensor(std, device=device).view(1, 3, 1, 1)
    )
    with torch.no_grad():
        found = model.encode_image(tensor).float()
    return (found / found.norm(dim=-1, keepdim=True)).cpu().numpy()[0]


def features() -> None:
    """On each still, what 12's rule reads: face and body heights through 09, 07's pose of the largest
    person, and MobileCLIP's nearest size."""
    import open_clip

    from src.tools import download_models

    seg: Any = load_stage("02-segmentation")
    con: Any = load_stage("07-conditional-experts")
    device = seg._device("auto")
    model, _, preprocess = open_clip.create_model_and_transforms(
        seg.CLIP_MODEL, cache_dir=str(download_models.stage_dir("02-segmentation"))
    )
    model = model.to(device).eval()
    crop, mean, std = seg._preprocess_spec(preprocess)
    texts = KIT._size_texts(download_models.stage_dir("12-identikit"))
    landmarker = con._pose_landmarker(
        download_models.stage_dir("07-conditional-experts")
    )
    found: dict[str, Any] = json.loads(DETECTIONS.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for name, entry in found.items():
        image = cv2.imread(str(RAW / name))
        size = tuple(entry["size"])
        face = body = None
        samples: list[list[float]] = []
        if entry["rows"]:
            shot = {
                "tracks": [
                    {
                        "frames": [
                            {"frame": 0, "interpolated": False, **r}
                            for r in entry["rows"]
                        ]
                    }
                ]
            }
            framing = AGG._framing(shot, size)
            face, body = (
                float(framing["face_height"]["median"]),
                float(framing["body_height"]["median"]),
            )
            main = max(entry["rows"], key=lambda r: r["body"][3] - r["body"][1])["body"]
            pose = con._pose_pass(landmarker, con._crop(image, main))
            if pose:
                samples.append(KIT._parts(pose["keypoints"], main, size))
        vector = _vector(model, image, crop, mean, std, device)
        out[name] = {
            "face": face,
            "body": body,
            "samples": samples,
            "nearest": int(np.argmax(texts @ vector)),
        }
    FEATURES.write_text(json.dumps(out), encoding="utf-8")
    print(
        f"{len(out)} stills, {sum(1 for v in out.values() if v['samples'])} with a pose",
        flush=True,
    )


def _band(value: float, low: float, high: float) -> bool:
    return (value - low) / max(high - low, 1e-9) >= 0.5


def _size(entry: dict[str, Any]) -> tuple[int, bool]:
    """12's size, and whether its measure sits in the tighter half of the band that decided it."""
    if entry["face"] is None:
        return entry["nearest"], False
    face, body = entry["face"], entry["body"]
    found, source = KIT._size(face, body, entry["samples"])
    if source == "body cut":
        return found, False
    bands = {
        ("face", 0): (face, KIT.EXTREME_CLOSE, 1.0),
        ("face", 1): (face, KIT.CLOSE, KIT.EXTREME_CLOSE),
        ("face", 3): (face, KIT.MEDIUM_FACE, KIT.CLOSE),
        ("body", 3): (body, KIT.MEDIUM_BODY, 1.0),
        ("body", 5): (body, KIT.LONG, KIT.MEDIUM_BODY),
        ("body", 6): (body, 0.0, KIT.LONG),
    }
    return found, _band(*bands[(source, found)])


def _choose(options: dict[str, str], here: int, tight: bool) -> str:
    """The option nearest our size on ShotBench's scale; a two-size option is as near as its nearer size.
    On a tie, the tighter option if our measure sits in the tighter half of its band, else the wider."""
    places = {
        letter: [SIZES.index(s.strip()) for s in text.split(",")]
        for letter, text in options.items()
    }
    distance = {letter: min(abs(p - here) for p in ps) for letter, ps in places.items()}
    near = [letter for letter, d in distance.items() if d == min(distance.values())]
    if len(near) == 1:
        return near[0]
    return (min if tight else max)(near, key=lambda letter: min(places[letter]))


def score(out: Path) -> None:
    found: dict[str, Any] = json.loads(FEATURES.read_text(encoding="utf-8"))
    questions = _questions()
    answers: dict[str, str] = {}
    exact = 0
    for q in questions:
        here, tight = _size(found[q["image"]])
        answers[q["index"]] = _choose(q["options"], here, tight)
        exact += any(
            SIZES[here] in [s.strip() for s in t.split(",")]
            for t in q["options"].values()
        )
    right = sum(1 for q in questions if answers[q["index"]] == q["answer"])
    report = {
        "about": (
            "ShotBench (Liu et al., NeurIPS 2025) asks, for 485 film stills, which of four shot sizes each one is. "
            "Stage 12's shot-size rule answers: 06's person detector and face finder run on the still, 09 turns "
            "them into the largest face and body heights, 07's pose landmarker reads where the frame cuts the "
            "largest body, and with nobody found MobileCLIP picks the nearest size; the option nearest 12's size "
            "on ShotBench's seven-step scale is chosen. The rule was fitted on AVE film scenes, never on ShotBench."
        ),
        "rule": {
            k: getattr(KIT, k)
            for k in (
                "EXTREME_CLOSE",
                "CLOSE",
                "MEDIUM_FACE",
                "MEDIUM_BODY",
                "LONG",
                "VISIBLE",
            )
        },
        "questions": len(questions),
        "accuracy": round(100 * right / len(questions), 1),
        "our_size_among_options": exact,
        "published_shot_size_accuracy": PUBLISHED,
        "answers": answers,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("answers", "about", "published_shot_size_accuracy")
            },
            indent=1,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Score 12's shot-size rule on ShotBench")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("detect", help="06's persons and faces on every shot-size image")
    sub.add_parser("features", help="the measures 12's rule reads, on every image")
    rate = sub.add_parser("score", help="12's sizes as answers, and the accuracy")
    rate.add_argument("--out", type=Path, default=CACHE / "shotbench.json")
    args = p.parse_args()
    if args.command == "detect":
        detect()
    elif args.command == "features":
        features()
    else:
        score(args.out)


if __name__ == "__main__":
    main()
