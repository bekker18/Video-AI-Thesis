"""Scores 10's looks_at rule on UCO-LAEO's pairs of people looking at each other: python3 src/validation/evaluate_laeo.py;
with --protocol mtgs, on VSGaze's UCO-LAEO test pairs as MTGS (NeurIPS 2024) scores them, AP and F1"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_stage

RELATIONS: Any = load_stage("10-relations")

# LAEO-Net++'s partition 1 (datasets/ln_ucolaeoConfig.py); settings are chosen on every other video
TEST = (
    "got05", "got06", "got08", "mr04", "mr05", "mr06", "mr10", "sv03", "sv06", "sv10",
    "twd03", "twd04", "twd05", "twd06", "twd07",
)  # fmt: skip
VSGAZE = (
    ROOT / "data" / "raw" / "vsgaze" / "VSGaze" / "uco_laeo_test.json"
)  # MTGS's UCO-LAEO test pairs
# AP on UCO-LAEO, %. MTGS's Table 3(d): MTGS fine-tuned on UCO-LAEO (dagger: starting from VSGaze), on these
# test videos by its code; the earlier models' figures there are copied from their own papers, on their own pairs.
# The rows from CSGaze's table state no protocol. Only MTGS's are on exactly these pairs.
PUBLISHED_AP = {
    "MTGS, fine-tuned on UCO-LAEO from VSGaze (NeurIPS 2024)": 97.4,
    "MTGS, fine-tuned on UCO-LAEO (NeurIPS 2024)": 96.3,
    "Gupta et al. (FG 2024), CSGaze's table, protocol unstated": 95.7,
    "CSGaze (2025), protocol unstated": 88.7,
    "LAEO-Net++ (TPAMI 2021), its own pairs": 86.7,
    "GP-Static (2023), CSGaze's table, protocol unstated": 80.3,
    "LAEO-Net (CVPR 2019), its own pairs": 79.5,
    "Doosti et al. (2021), its own pairs": 76.2,
}
# F1 on these UCO-LAEO test pairs, %: MTGS's Table 2, every model trained on VSGaze; "from gaze points" decides
# LAEO by each gaze point falling in the other's head box, the counterpart of 10's rule, which uses angles
PUBLISHED_F1 = {
    "MTGS trained without gaze following, social-gaze output": 93.9,
    "MTGS, social-gaze output": 88.9,
    "MTGS, from gaze points": 87.3,
    "Gupta et al. (CVPR 2022), from gaze points": 85.6,
    "Chong et al., static, from gaze points": 81.1,
    "Chong et al., temporal, from gaze points": 77.9,
}
BOOT = 2000
SEED = 0
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
        lines = (notes / "annotations_frame" / f"{video}.txt").read_text(
            encoding="utf-8"
        )
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
    return [
        head[0] + 0.2 * fw,
        head[1] + 0.4 * fh,
        head[0] + 1.2 * fw,
        head[1] + 1.4 * fh,
    ]


def _gazes(db: Path, heads: Heads, cache: Path, device: str) -> dict[str, Any]:
    """07's Gaze-LLE pass on every annotated head, cached as "video/frame/head" -> gaze or None."""
    gazes: dict[str, Any] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    missing = [
        (v, f)
        for v, frames in heads.items()
        for f in frames
        if f"{v}/{f}/0" not in gazes
    ]
    if not missing:
        return gazes
    experts: Any = load_stage("07-conditional-experts")
    model = experts._gaze_model(device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    for done, (video, frame) in enumerate(missing, start=1):
        image = cv2.imread(str(db / "frames" / video / f"{frame:06d}.jpg"))
        height, width = image.shape[:2]
        faces = [_face(b[:4]) for b in heads[video][frame]]
        keep = [
            i
            for i, f in enumerate(faces)
            if experts._readable(experts._head_box(f, width, height))
        ]
        found = (
            experts._gaze_pass(model, image, [faces[i] for i in keep], device)
            if keep
            else []
        )
        looks = dict(zip(keep, found))
        for i in range(len(faces)):
            gazes[f"{video}/{frame}/{i}"] = looks.get(i)
        if done % 500 == 0 or done == len(
            missing
        ):  # an interrupted run keeps what it measured
            cache.write_text(json.dumps(gazes), encoding="utf-8")
    return gazes


def _mutual(
    db: Path, heads: Heads, gazes: dict[str, Any], cut: float, limit: float
) -> dict[str, list[Row]]:
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
                landed.append(
                    RELATIONS._landed(gaze, face, others, cut, limit) if gaze else None
                )
            positive = {
                tuple(sorted(p[:2])) for p in laeo.get(frame + 1, []) if p[0] > 0
            }
            for i, j in combinations(range(len(boxes)), 2):
                rows[split].append(
                    ((i + 1, j + 1) in positive, landed[i] == j and landed[j] == i)
                )
    return rows


def _score(rows: list[Row]) -> dict[str, Any]:
    said = [y for y, mutual in rows if mutual]
    positives = sum(1 for y, _ in rows if y)
    precision = sum(said) / len(said) if said else 0.0
    recall = sum(said) / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _direction(
    gaze: dict[str, Any] | None, face: list[float], other: tuple[float, float, float]
) -> float:
    """How surely a look lands on the other head, by 10's geometry: in-frame probability times a falloff of
    the angle between the sight line and the head, on LOOK_DEG's scale; fixed before scoring."""
    if not gaze:
        return 0.0
    x, y = RELATIONS._head_centre(face)
    tx, ty = float(gaze["target"][0]), float(gaze["target"][1])
    if math.hypot(tx - x, ty - y) < 1.0:
        return 0.0
    hx, top, bottom = other
    hy = min(max(ty, top), bottom)
    if math.hypot(hx - x, hy - y) < (face[3] - face[1]) / 2:
        return 0.0
    turn = abs(math.atan2(hy - y, hx - x) - math.atan2(ty - y, tx - x))
    degrees = math.degrees(min(turn, 2 * math.pi - turn))
    return float(gaze["in_frame"]) * math.exp(-((degrees / RELATIONS.LOOK_DEG) ** 2))


def _vsgaze(db: Path) -> tuple[list[dict[str, Any]], Heads]:
    """VSGaze's UCO-LAEO test frames, their head boxes in pixels (the zero placeholder dropped)."""
    rows: list[dict[str, Any]] = json.loads(VSGAZE.read_text(encoding="utf-8"))
    heads: Heads = {}
    sizes: dict[str, tuple[int, int]] = {}
    for r in rows:
        _, video, name = r["path"].split("/")
        if video not in sizes:
            height, width = cv2.imread(str(db / "frames" / video / name)).shape[:2]
            sizes[video] = (width, height)
        width, height = sizes[video]
        r["video"], r["frame"] = video, int(Path(name).stem)
        heads.setdefault(video, {})[r["frame"]] = [
            [
                round(b[0] * width),
                round(b[1] * height),
                round(b[2] * width),
                round(b[3] * height),
            ]
            for b in r["head_bboxes"][1:]
        ]
    return rows, heads


def _mtgs_rows(
    rows: list[dict[str, Any]], heads: Heads, gazes: dict[str, Any]
) -> list[tuple[str, int, float, float, bool]]:
    """Per labelled ordered pair: video, label, the pair score kept by MTGS's per-person maximum, the raw score,
    and 10's own yes or no (each look lands on the other)."""
    out = []
    for r in rows:
        boxes = heads[r["video"]][r["frame"]]
        faces = [_face(b) for b in boxes]
        spans = [RELATIONS._head_of({"face": f}) for f in faces]
        looks = [gazes.get(f"{r['video']}/{r['frame']}/{i}") for i in range(len(boxes))]
        index = {pid: i - 1 for i, pid in enumerate(r["person_ids"]) if i > 0}
        landed = [
            RELATIONS._landed(g, f, {j: c for j, c in enumerate(spans) if j != i})
            if g
            else None
            for i, (g, f) in enumerate(zip(looks, faces))
        ]
        raw: list[float | None] = []
        for a, b in r["pairs"]:
            if a == -1 or b == -1:
                raw.append(None)
                continue
            i, j = index[a], index[b]
            raw.append(
                min(
                    _direction(looks[i], faces[i], spans[j]),
                    _direction(looks[j], faces[j], spans[i]),
                )
            )
        # MTGS's compute_metrics: per person in second place, only the best-scoring partner keeps its score
        kept = [0.0] * len(raw)
        for person in set(b for _, b in r["pairs"] if b != -1):
            group = [
                k for k, (a, b) in enumerate(r["pairs"]) if b == person and a != -1
            ]
            if group:
                best = max(group, key=lambda k: raw[k] or 0.0)
                kept[best] = raw[best] or 0.0
        for k, ((a, b), label) in enumerate(zip(r["pairs"], r["laeo_pairs"])):
            if label == -1 or a == -1 or b == -1:
                continue
            i, j = index[a], index[b]
            out.append(
                (
                    r["video"],
                    int(label),
                    kept[k],
                    raw[k] or 0.0,
                    landed[i] == j and landed[j] == i,
                )
            )
    return out


def mtgs(db: Path, out: Path, device: str) -> dict[str, Any]:
    """10's looks_at on MTGS's UCO-LAEO test protocol: VSGaze's pairs and heads, AP after MTGS's per-person
    maximum, F1 from 10's yes or no as MTGS's notebook takes it from its gaze points."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rows, heads = _vsgaze(db)
    gazes = _gazes(db, heads, out.with_name("laeo_gaze_vsgaze.json"), device)
    pairs = _mtgs_rows(rows, heads, gazes)
    y = np.array([p[1] for p in pairs])
    kept, raw = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    said = np.array([p[4] for p in pairs])
    by_video: dict[str, list[int]] = {}
    for k, p in enumerate(pairs):
        by_video.setdefault(p[0], []).append(k)
    videos = sorted(by_video)
    rng = np.random.default_rng(SEED)
    aps, f1s = [], []
    for _ in range(BOOT):
        idx = np.array(
            [
                k
                for v in rng.choice(len(videos), len(videos))
                for k in by_video[videos[v]]
            ]
        )
        aps.append(average_precision_score(y[idx], kept[idx]))
        f1s.append(f1_score(y[idx], said[idx]))
    interval = lambda v: [
        round(100 * float(np.percentile(v, q)), 1) for q in (2.5, 97.5)
    ]
    return {
        "about": (
            "10's looks_at on the same UCO-LAEO test pairs MTGS (Gupta et al., NeurIPS 2024) is scored on: VSGaze's "
            "annotation file, its head boxes given to 07's Gaze-LLE as in MTGS's evaluation. AP follows MTGS's "
            "compute_metrics.py: ordered pairs, and per person only the best-scoring partner keeps its score. Our pair "
            "score, fixed before scoring, is the weaker of the two looks, each Gaze-LLE's in-frame probability times "
            "exp(-(angle/20 degrees)^2), the angle between the sight line and the other head as 10 measures it. F1 is "
            "10's own yes or no from gaze, the counterpart of MTGS's 'from gaze points' row. Nothing was fitted on these "
            "pairs; 10's two constants were set on UCO-LAEO's other videos, and Gaze-LLE's checkpoint was trained on "
            "GazeFollow and VideoAttentionTarget. MTGS was trained on VSGaze, which includes UCO-LAEO's training videos. "
            "Only MTGS's figures are on exactly these pairs; earlier models' AP comes from their own papers' pairs."
        ),
        "test_videos": videos,
        "frames": len(rows),
        "ordered_pairs": len(y),
        "laeo": int(y.sum()),
        "chance_ap": round(100 * float(y.mean()), 1),
        "ap": round(100 * float(average_precision_score(y, kept)), 1),
        "ap_95": interval(aps),
        "ap_without_per_person_maximum": round(
            100 * float(average_precision_score(y, raw)), 1
        ),
        "roc_auc": round(100 * float(roc_auc_score(y, kept)), 1),
        "f1": round(100 * float(f1_score(y, said)), 1),
        "f1_95": interval(f1s),
        "precision": round(100 * float(precision_score(y, said)), 1),
        "recall": round(100 * float(recall_score(y, said)), 1),
        "published_ap": PUBLISHED_AP,
        "published_f1": PUBLISHED_F1,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Score 10's looks_at against UCO-LAEO")
    p.add_argument(
        "--db", type=Path, default=ROOT / "data" / "raw" / "ucolaeo" / "ucolaeodb"
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "laeo.json"
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="for Gaze-LLE"
    )
    p.add_argument(
        "--protocol",
        choices=["laeo-net", "mtgs"],
        default="laeo-net",
        help="whose test pairs and scoring",
    )
    args = p.parse_args()
    if args.protocol == "mtgs":
        report = mtgs(args.db, args.out, args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("about", "test_videos")},
                indent=1,
            )
        )
        return

    heads, skipped = _heads(args.db)
    gazes = _gazes(args.db, heads, args.out.with_name("laeo_gaze.json"), args.device)
    sweep = [
        {
            "cut": cut,
            "degrees": limit,
            **_score(_mutual(args.db, heads, gazes, cut, limit)["tune"]),
        }
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
            "current": {
                "cut": RELATIONS.IN_FRAME,
                "degrees": RELATIONS.LOOK_DEG,
                **_score(current["test"]),
            },
            "chosen": {
                "cut": chosen["cut"],
                "degrees": chosen["degrees"],
                **_score(best["test"]),
            },
        },
        "note": "heads are UCO-LAEO's annotated boxes, not 06's face boxes",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
