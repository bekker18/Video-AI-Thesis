"""Scores 10's looks_at rule on UCO-LAEO's pairs of people looking at each other: python3 src/validation/evaluate_laeo.py"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_stage  # noqa: E402

RELATIONS: Any = load_stage("10-relations")

# LAEO-Net++'s partition 1 (datasets/ln_ucolaeoConfig.py); settings are chosen on every other video
TEST = (
    "got05", "got06", "got08", "mr04", "mr05", "mr06", "mr10", "sv03", "sv06", "sv10",
    "twd03", "twd04", "twd05", "twd06", "twd07",
)  # fmt: skip
CUTS = (0.2, 0.3, 0.4, 0.5)  # Gaze-LLE in-frame probabilities swept
LIMITS = (10.0, 15.0, 20.0, 30.0)  # degrees from the line of sight swept

Heads: TypeAlias = dict[str, dict[int, list[list[int]]]]
Row: TypeAlias = tuple[bool, bool]  # LAEO label, both looks land on the other


def _blocks(path: Path) -> dict[int, list[list[int]]]:
    """A frame number on a line of its own, then a line per head (box, then a facing word) or pair."""
    blocks: dict[int, list[list[int]]] = {}
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 1:
            rows = blocks.setdefault(int(parts[0]), [])
        elif parts:
            rows.append([int(p) for p in parts if p.lstrip("-").isdigit()])
    return blocks


def _heads(db: Path) -> tuple[Heads, list[str]]:
    """Frames with 2+ heads by frame file (annotations count from 1); skips label 9 and mismatched videos."""
    notes = db / "annotations"
    heads: Heads = {}
    skipped: list[str] = []
    for path in sorted((notes / "annotations_head").glob("*_heads.txt")):
        video = path.name.removesuffix("_heads.txt")
        blocks = _blocks(path)
        files = len(list((db / "frames" / video).glob("*.jpg")))
        if abs(len(blocks) - files) > 1:
            skipped.append(video)
            continue
        lines = (notes / "annotations_frame" / f"{video}.txt").read_text(encoding="utf-8")
        labels = dict(line.split()[:2] for line in lines.splitlines() if line.strip())
        heads[video] = {
            k - 1: boxes
            for k, boxes in blocks.items()
            if len(boxes) >= 2 and k - 1 < files and "9" not in labels.get(str(k), "9")
        }
    return heads, skipped


def _face(head: list[int]) -> list[float]:
    """The face box 07 grows into this head box, so the annotated head reaches Gaze-LLE unchanged."""
    fw, fh = (head[2] - head[0]) / 1.4, (head[3] - head[1]) / 1.5
    return [head[0] + 0.2 * fw, head[1] + 0.4 * fh, head[0] + 1.2 * fw, head[1] + 1.4 * fh]


def _gazes(db: Path, heads: Heads, cache: Path, device: str) -> dict[str, Any]:
    """07's Gaze-LLE pass on every annotated head, cached as "video/frame/head" -> gaze or None."""
    gazes: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = [(v, f) for v, frames in heads.items() for f in frames if f"{v}/{f}/0" not in gazes]
    if not missing:
        return gazes
    experts: Any = load_stage("07-conditional-experts")
    model = experts._gaze_model(device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for done, (video, frame) in enumerate(missing, start=1):
        image = cv2.imread(str(db / "frames" / video / f"{frame:06d}.jpg"))
        height, width = image.shape[:2]
        faces = [_face(b[:4]) for b in heads[video][frame]]
        keep = [i for i, f in enumerate(faces) if experts._readable(experts._head_box(f, width, height))]
        found = experts._gaze_pass(model, image, [faces[i] for i in keep], device) if keep else []
        looks = dict(zip(keep, found))
        for i in range(len(faces)):
            gazes[f"{video}/{frame}/{i}"] = looks.get(i)
        if done % 500 == 0 or done == len(missing):  # an interrupted run keeps what it measured
            cache.write_text(json.dumps(gazes), encoding="utf-8")
    return gazes


def _mutual(db: Path, heads: Heads, gazes: dict[str, Any], cut: float, limit: float) -> dict[str, list[Row]]:
    """Per pair of heads, whether each one's look lands on the other by 10's rule, split as tune and test."""
    rows: dict[str, list[Row]] = {"tune": [], "test": []}
    for video, frames in heads.items():
        laeo = _blocks(db / "annotations" / "annotations_pair" / f"pair_{video}.txt")
        split = "test" if video in TEST else "tune"
        for frame, boxes in frames.items():
            faces = [_face(b[:4]) for b in boxes]
            spans = [RELATIONS._head_of({"face": f}) for f in faces]
            landed = []
            for i, face in enumerate(faces):
                gaze = gazes.get(f"{video}/{frame}/{i}")
                others = {j: c for j, c in enumerate(spans) if j != i}
                landed.append(RELATIONS._landed(gaze, face, others, cut, limit) if gaze else None)
            positive = {tuple(sorted(p[:2])) for p in laeo.get(frame + 1, []) if p[0] > 0}
            for i, j in combinations(range(len(boxes)), 2):
                rows[split].append(((i + 1, j + 1) in positive, landed[i] == j and landed[j] == i))
    return rows


def _score(rows: list[Row]) -> dict[str, Any]:
    said = [y for y, mutual in rows if mutual]
    positives = sum(1 for y, _ in rows if y)
    precision = sum(said) / len(said) if said else 0.0
    recall = sum(said) / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def main() -> None:
    p = argparse.ArgumentParser(description="Score 10's looks_at against UCO-LAEO")
    p.add_argument("--db", type=Path, default=ROOT / "data" / "raw" / "ucolaeo" / "ucolaeodb")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "validation" / "laeo.json")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="for Gaze-LLE")
    args = p.parse_args()

    heads, skipped = _heads(args.db)
    gazes = _gazes(args.db, heads, args.out.with_name("laeo_gaze.json"), args.device)
    sweep = [
        {"cut": cut, "degrees": limit, **_score(_mutual(args.db, heads, gazes, cut, limit)["tune"])}
        for cut in CUTS
        for limit in LIMITS
    ]
    chosen = max(sweep, key=lambda s: s["f1"])
    current = _mutual(args.db, heads, gazes, RELATIONS.IN_FRAME, RELATIONS.LOOK_DEG)
    best = _mutual(args.db, heads, gazes, chosen["cut"], chosen["degrees"])
    report = {
        "videos_skipped": skipped,
        "pairs": {
            split: {"pairs": len(rows), "laeo": sum(1 for y, _ in rows if y)}
            for split, rows in current.items()
        },
        "tune_sweep": sweep,
        "chosen": {"cut": chosen["cut"], "degrees": chosen["degrees"]},
        "test": {
            "current": {"cut": RELATIONS.IN_FRAME, "degrees": RELATIONS.LOOK_DEG, **_score(current["test"])},
            "chosen": {"cut": chosen["cut"], "degrees": chosen["degrees"], **_score(best["test"])},
        },
        "note": "heads are UCO-LAEO's annotated boxes, not 06's face boxes",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
