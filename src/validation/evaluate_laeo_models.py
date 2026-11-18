"""The models behind the looks-at rule, scored on UCO-LAEO's annotated heads before any rule: YOLO's person
boxes, YuNet inside them as 06 runs it, MediaPipe's face mesh as 07 runs it. Then 10's rule end to end from
those boxes, 10's HEAD_REACH against the annotated heads, and 12's listing of a relation per shot:
python3 src/validation/evaluate_laeo_models.py --out <report>"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "validation"))
import evaluate_laeo as laeo

from main import load_stage
from src.tools import download_models

TRACKING: Any = load_stage("06-detection-tracking")
EXPERTS: Any = load_stage("07-conditional-experts")
RELATIONS: Any = laeo.RELATIONS

DB = ROOT / "data" / "raw" / "ucolaeo" / "ucolaeodb"
DET_CONF, LOW_CONF, FACE_CONF, CROP_PAD = 0.25, 0.1, 0.6, 0.1  # main.py's defaults
SIZES = (
    ("under 60 px", 0, 60),
    ("60-120 px", 60, 120),
    ("120-240 px", 120, 240),
    ("240 px and up", 240, 10**6),
)
SHARES = (0.3, 0.4, 0.5, 0.6, 0.7)  # 12's relation_share swept; 0.5 is the pipeline's
MINIMUMS = (1, 3, 5)  # 12's relation_min swept; 3 is the pipeline's
BATCH = 32


def _facings(video: str) -> dict[int, list[str]]:
    """The facing word of each head line, in the order of evaluate_laeo's boxes."""
    out: dict[int, list[str]] = {}
    rows: list[str] = []
    for line in (
        (DB / "annotations" / "annotations_head" / f"{video}_heads.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        parts = line.split()
        if len(parts) == 1:
            rows = out.setdefault(int(parts[0]) - 1, [])
        elif parts:
            words = [p for p in parts if not p.lstrip("-").isdigit()]
            rows.append(words[0] if words else "unknown")
    return out


def _centre(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _inside(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _upper(box: list[float]) -> list[float]:
    """The top half of a body box, where its head sits."""
    return [box[0], box[1], box[2], box[1] + (box[3] - box[1]) / 2.0]


def _own_head(body: list[float], heads: list[list[int]]) -> int | None:
    """The annotated head a body box belongs to: centre in the box's top half, nearest its top midpoint."""
    top = ((body[0] + body[2]) / 2.0, body[1])
    found = [i for i, h in enumerate(heads) if _inside(_centre(h), _upper(body))]
    return (
        min(
            found,
            key=lambda i: np.hypot(
                _centre(heads[i])[0] - top[0], _centre(heads[i])[1] - top[1]
            ),
        )
        if found
        else None
    )


def _head_of_face(face: list[float], heads: list[list[int]]) -> int | None:
    """The smallest annotated head holding a face's centre."""
    found = [i for i, h in enumerate(heads) if _inside(_centre(face), h)]
    return (
        min(
            found,
            key=lambda i: (heads[i][2] - heads[i][0]) * (heads[i][3] - heads[i][1]),
        )
        if found
        else None
    )


def _size(head: list[int]) -> str:
    height = head[3] - head[1]
    return next(name for name, low, high in SIZES if low <= height < high)


def _detect(frames: list[tuple[str, int]], cache: Path) -> dict[str, Any]:
    """Per frame: YOLO's persons from 0.1 with their scores, YuNet's face in each, the mesh on each face, and
    Gaze-LLE on every readable face of the frame at once, as 06 and 07 run them."""
    found: dict[str, Any] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    todo = [(v, f) for v, f in frames if f"{v}/{f}" not in found]
    if not todo:
        return found
    from ultralytics import YOLO

    download_models.ensure("03-profiler")
    yolo = YOLO(
        str(download_models.detector("03-profiler", download_models.DEFAULT_DETECTOR))
    )
    TRACKING._start_yunet(
        str(download_models.stage_dir("03-profiler") / TRACKING.FACE_MODEL), FACE_CONF
    )
    download_models.ensure("07-conditional-experts")
    mesh = EXPERTS._face_landmarker(ROOT / "models" / "07-conditional-experts")
    gaze = EXPERTS._gaze_model("cuda")
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        images = [cv2.imread(str(DB / "frames" / v / f"{f:06d}.jpg")) for v, f in chunk]
        results = yolo.predict(
            images, conf=LOW_CONF, classes=[0], device="cuda", verbose=False
        )
        for (video, frame), image, result in zip(chunk, images, results):
            height, width = image.shape[:2]
            persons = []
            for box, score in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            ):
                body = [round(float(v)) for v in box]
                crop = TRACKING._face_crop(image, body)
                face = (
                    TRACKING._face_from(TRACKING._yunet(crop[0]), crop, FACE_CONF)
                    if crop
                    else None
                )
                meshed = None
                if face:
                    patch = EXPERTS._crop(image, face["box"], CROP_PAD)
                    meshed = bool(patch.size and EXPERTS._face_pass(mesh, patch, 10))
                persons.append(
                    {
                        "body": body,
                        "score": round(float(score), 3),
                        "face": face["box"] if face else None,
                        "mesh": meshed,
                    }
                )
            readable = [
                i
                for i, p in enumerate(persons)
                if p["face"]
                and p["score"] >= DET_CONF
                and EXPERTS._readable(EXPERTS._head_box(p["face"], width, height))
            ]
            looks = (
                EXPERTS._gaze_pass(
                    gaze, image, [persons[i]["face"] for i in readable], "cuda"
                )
                if readable
                else []
            )
            for i, look in zip(readable, looks):
                persons[i]["gaze"] = look
            found[f"{video}/{frame}"] = persons
        if (start // BATCH) % 20 == 0:
            cache.write_text(json.dumps(found), encoding="utf-8")
            print(f"{start + len(chunk)}/{len(todo)} frames", flush=True)
    cache.write_text(json.dumps(found), encoding="utf-8")
    return found


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def _models(
    heads: Any, facings: dict[str, Any], found: dict[str, Any]
) -> dict[str, Any]:
    """Per annotated head: a person box over it, a YuNet face on it, the mesh on that face; by size and facing.
    Per face found: whether its body box is its own person's; per body: where its head ends."""
    tally: dict[str, dict[str, list[int]]] = {}
    wrong = crowded_wrong = faces_placed = crowded_placed = 0
    reach: list[float] = []
    covered = 0

    def add(group: str, key: str, hit: bool) -> None:
        row = tally.setdefault(group, {}).setdefault(key, [0, 0])
        row[0] += int(hit)
        row[1] += 1

    for video, frames in heads.items():
        for frame, boxes in frames.items():
            persons = [p for p in found[f"{video}/{frame}"] if p["score"] >= DET_CONF]
            low = found[f"{video}/{frame}"]
            facing = facings[video].get(frame, [])
            face_heads = {}
            for p in persons:
                if p["face"]:
                    face_heads[id(p)] = _head_of_face(p["face"], boxes)
            for i, head in enumerate(boxes):
                body = any(_inside(_centre(head), _upper(p["body"])) for p in persons)
                body_low = any(_inside(_centre(head), _upper(p["body"])) for p in low)
                face = [p for p in persons if p["face"] and face_heads[id(p)] == i]
                keys = ("all", _size(head), facing[i] if i < len(facing) else "unknown")
                for k, group in zip(keys, ("all", "size", "facing")):
                    add(f"person box, {group}", k, body)
                    add(f"person box from 0.1, {group}", k, body_low)
                    add(f"face, {group}", k, bool(face))
                    if face:
                        add(f"mesh on the face, {group}", k, bool(face[0]["mesh"]))
            for p in persons:
                own = _own_head(p["body"], boxes)
                if own is not None:
                    width = p["body"][2] - p["body"][0]
                    reach.append((boxes[own][3] - p["body"][1]) / width)
                    _, hy = _centre(boxes[own])
                    covered += int(
                        p["body"][1]
                        <= hy
                        <= p["body"][1] + RELATIONS.HEAD_REACH * width
                    )
                if p["face"] and own is not None and face_heads[id(p)] is not None:
                    bad = face_heads[id(p)] != own
                    faces_placed += 1
                    wrong += int(bad)
                    if len(boxes) >= 3:
                        crowded_placed += 1
                        crowded_wrong += int(bad)
    return {
        "recall": {
            g: {k: {"rate": _rate(*v), "heads": v[1]} for k, v in sorted(rows.items())}
            for g, rows in tally.items()
        },
        "face_in_wrong_body": {
            "faces": faces_placed,
            "rate": _rate(wrong, faces_placed),
            "faces_in_frames_with_3_heads_or_more": crowded_placed,
            "crowded_rate": _rate(crowded_wrong, crowded_placed),
        },
        "head_reach": {
            "pipeline": RELATIONS.HEAD_REACH,
            "bodies": len(reach),
            "median_head_bottom_in_box_widths": round(float(np.median(reach)), 3)
            if reach
            else None,
            "quartiles": [round(float(np.percentile(reach, q)), 3) for q in (25, 75)]
            if reach
            else None,
            "head_centre_within_span": _rate(covered, len(reach)),
        },
    }


def _end_to_end(heads: Any, found: dict[str, Any]) -> dict[str, Any]:
    """Each annotated pair: looking at each other when the persons over the two heads, from our own boxes and
    faces, each land on the other by 10's rule. A head no person covers cannot be found."""
    rows: dict[str, list[tuple[bool, bool]]] = {"tune": [], "test": []}
    for video, frames in heads.items():
        pairs = laeo._blocks(
            DB / "annotations" / "annotations_pair" / f"pair_{video}.txt"
        )
        split = "test" if video in laeo.TEST else "tune"
        for frame, boxes in frames.items():
            persons = [p for p in found[f"{video}/{frame}"] if p["score"] >= DET_CONF]
            spans = {
                k: RELATIONS._head_of({"face": p["face"], "body": p["body"]})
                for k, p in enumerate(persons)
            }
            landed = {
                k: RELATIONS._landed(
                    p["gaze"], p["face"], {j: s for j, s in spans.items() if j != k}
                )
                for k, p in enumerate(persons)
                if p.get("gaze")
            }
            owner: dict[int, int] = {}
            for k, p in enumerate(persons):
                head = (
                    _head_of_face(p["face"], boxes)
                    if p["face"]
                    else _own_head(p["body"], boxes)
                )
                if head is not None and head not in owner:
                    owner[head] = k
            positive = {
                tuple(sorted(p[:2])) for p in pairs.get(frame + 1, []) if p[0] > 0
            }
            for i, j in combinations(range(len(boxes)), 2):
                a, b = owner.get(i), owner.get(j)
                said = (
                    a is not None
                    and b is not None
                    and landed.get(a) == b
                    and landed.get(b) == a
                )
                rows[split].append(((i + 1, j + 1) in positive, said))
    return {
        split: laeo._score(r) | {"pairs": len(r), "positives": sum(y for y, _ in r)}
        for split, r in rows.items()
    }


def _tracks(frames: dict[int, list[list[int]]]) -> dict[int, list[int]]:
    """Annotated heads linked over a shot by overlap with the last box of each track: frame -> track per head."""
    last: list[tuple[int, list[int]]] = []
    out: dict[int, list[int]] = {}
    for frame in sorted(frames):
        ids = []
        used: set[int] = set()
        for box in frames[frame]:
            best, score = None, 0.3
            for t, (seen, prev) in enumerate(last):
                if t in used or frame - seen > 5:
                    continue
                overlap = laeo_iou(box, prev)
                if overlap > score:
                    best, score = t, overlap
            if best is None:
                last.append((frame, box))
                best = len(last) - 1
            else:
                last[best] = (frame, box)
            used.add(best)
            ids.append(best)
        out[frame] = ids
    return out


def laeo_iou(a: list[int], b: list[int]) -> float:
    wide, tall = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or tall <= 0:
        return 0.0
    both = wide * tall
    return both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both)


def _per_shot(heads: Any, gazes: dict[str, Any]) -> dict[str, Any]:
    """12's listing on whole shots, from the annotated heads: a pair is listed as looking at each other when
    each one's looks land on the other in at least `share` of at least `minimum` samples. The truth: the pair
    is labelled as looking at each other in at least half the frames where both are annotated."""
    pairs_by_split: dict[str, list[tuple[bool, dict[tuple[float, int], bool]]]] = {
        "tune": [],
        "test": [],
    }
    for video, frames in heads.items():
        labels = laeo._blocks(
            DB / "annotations" / "annotations_pair" / f"pair_{video}.txt"
        )
        split = "test" if video in laeo.TEST else "tune"
        ids = _tracks(frames)
        looks: dict[tuple[int, int], list[bool]] = {}
        together: dict[tuple[int, int], list[bool]] = {}
        for frame, boxes in frames.items():
            faces = [laeo._face(b[:4]) for b in boxes]
            spans = [RELATIONS._head_of({"face": f}) for f in faces]
            positive = {
                tuple(sorted(p[:2])) for p in labels.get(frame + 1, []) if p[0] > 0
            }
            for i in range(len(boxes)):
                gaze = gazes.get(f"{video}/{frame}/{i}")
                hit = (
                    RELATIONS._landed(
                        gaze, faces[i], {j: s for j, s in enumerate(spans) if j != i}
                    )
                    if gaze
                    else None
                )
                for j in range(len(boxes)):
                    if j != i and gaze:
                        looks.setdefault((ids[frame][i], ids[frame][j]), []).append(
                            hit == j
                        )
            for i, j in combinations(range(len(boxes)), 2):
                key = tuple(sorted((ids[frame][i], ids[frame][j])))
                together.setdefault(key, []).append((i + 1, j + 1) in positive)
        for (a, b), labelled in together.items():
            truth = sum(labelled) >= len(labelled) / 2
            said = {}
            for share in SHARES:
                for minimum in MINIMUMS:
                    ab, ba = looks.get((a, b), []), looks.get((b, a), [])
                    said[(share, minimum)] = (
                        len(ab) >= minimum
                        and len(ba) >= minimum
                        and sum(ab) >= share * len(ab)
                        and sum(ba) >= share * len(ba)
                    )
            pairs_by_split[split].append((truth, said))
    out: dict[str, Any] = {}
    for split, rows in pairs_by_split.items():
        out[split] = {
            "pairs": len(rows),
            "looking_at_each_other": sum(t for t, _ in rows),
        }
        for share in SHARES:
            for minimum in MINIMUMS:
                out[split][f"share {share}, at least {minimum}"] = laeo._score(
                    [(t, s[(share, minimum)]) for t, s in rows]
                )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    heads, skipped = laeo._heads(DB)
    facings = {v: _facings(v) for v in heads}
    frames = [(v, f) for v, fs in heads.items() for f in fs]
    found = _detect(frames, args.out.with_name(args.out.stem + "_detections.json"))
    gazes = json.loads(
        (ROOT / "data" / "runs" / "validation" / "laeo" / "laeo_gaze.json").read_text(
            encoding="utf-8"
        )
    )
    report = {
        "about": __doc__,
        "videos": len(heads),
        "skipped": skipped,
        "frames": len(frames),
        "heads": sum(len(b) for fs in heads.values() for b in fs.values()),
        "settings": {
            "det_conf": DET_CONF,
            "low_conf": LOW_CONF,
            "face_conf": FACE_CONF,
            "crop_pad": CROP_PAD,
            "in_frame": RELATIONS.IN_FRAME,
            "look_deg": RELATIONS.LOOK_DEG,
        },
        "models": _models(heads, facings, found),
        "looks_at_from_our_boxes": _end_to_end(heads, found),
        "looks_at_from_annotated_heads": {
            "test": "evaluate_laeo.py: F1 0.895 at in-frame 0.3, 20 degrees"
        },
        "per_shot": _per_shot(heads, gazes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("videos", "frames", "heads")}))
    print(
        json.dumps(report["models"]["face_in_wrong_body"]),
        json.dumps(report["models"]["head_reach"]),
    )
    print(json.dumps(report["looks_at_from_our_boxes"]))
    print(
        json.dumps(
            {
                s: report["per_shot"][s]["share 0.5, at least 3"]
                for s in ("tune", "test")
            }
        )
    )


if __name__ == "__main__":
    main()
