"""Scores the pipeline against AVE's per-shot labels, over every AVE scene in data/processed/: python3 src/validation/evaluate_ave.py"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Row: TypeAlias = tuple[set[str], str | None]

MOTION = {
    "locked": "static",
    "handheld": "handheld",
    "pan": "pan",
    "tilt": "tilt",
    "zoom": "zoom",
}
SIZE = {
    "extreme-wide": "extreme long",
    "wide": "long",
    "medium": "medium",
    "close-up": "close-up",
    "extreme-close-up": "extreme close-up",
}
LOCATION = {"int": "indoor", "ext": "outdoor"}
PEOPLE = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5+"}
SHOT_TYPE = {
    "single": "1",
    "two-shot": "2",
    "three-shot": "3",
    "group": "4+",
    "group-shot": "4+",
}
FLAGS = {"human": "person", "face": "face", "text": "text"}  # shot-subject -> 03's flag
RECALL_ONLY = {"face", "text"}

COUNTS = tuple(PEOPLE.values())
TYPE_COUNTS = ("0", "1", "2", "3", "4+")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _values(shot: dict[str, Any], field: str, mapping: dict[str, str]) -> set[str]:
    return {mapping[v] for v in shot.get(field, []) if v in mapping}


def _bin(count: int, order: tuple[str, ...]) -> str:
    return order[min(count, len(order) - 1)]


def _ratio(hits: float, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if overlap <= 0:
        return 0.0
    return overlap / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - overlap)


def _align(
    truth: list[tuple[int, int]], found: list[tuple[int, int]], floor: float
) -> list[tuple[int, int]]:
    """Shot and segment pairs that are each other's best overlap and clear `floor`."""
    if not truth or not found:
        return []
    table = np.array([[_iou(a, b) for b in found] for a in truth])
    pairs: list[tuple[int, int]] = []
    for i in range(len(truth)):
        j = int(table[i].argmax())
        if table[i, j] >= floor and int(table[:, j].argmax()) == i:
            pairs.append((i, j))
    return pairs


def _cut_hits(truth: list[int], found: list[int], tolerance: int) -> int:
    free = sorted(found)
    hits = 0
    for cut in truth:
        near = [f for f in free if abs(f - cut) <= tolerance]
        if near:
            free.remove(min(near, key=lambda f: abs(f - cut)))
            hits += 1
    return hits


def _categorical(
    rows: list[Row], order: tuple[str, ...], ordinal: bool
) -> dict[str, Any]:
    """A prediction counts if it is any of the shot's labels; `None` is an abstention."""
    labelled = [(t, p) for t, p in rows if t]
    answered = [(t, p) for t, p in labelled if p is not None]
    out: dict[str, Any] = {
        "n": len(labelled),
        "answered": len(answered),
        "coverage": _ratio(len(answered), len(labelled)),
    }
    if not answered:
        return out
    classes = [c for c in order if any(c in t for t, _ in answered)]
    recalls = [
        sum(1 for t, p in answered if c in t and p in t)
        / sum(1 for t, _ in answered if c in t)
        for c in classes
    ]
    confusion: dict[str, Counter[str]] = {}
    for t, p in answered:
        confusion.setdefault(p if p in t else min(t, key=order.index), Counter())[
            p
        ] += 1
    out["accuracy"] = _ratio(sum(1 for t, p in answered if p in t), len(answered))
    out["balanced_accuracy"] = round(float(np.mean(recalls)), 3)
    out["majority_baseline"] = _ratio(
        max(sum(1 for t, _ in answered if c in t) for c in classes), len(answered)
    )
    if ordinal:
        near = sum(
            1
            for t, p in answered
            if min(abs(order.index(p) - order.index(c)) for c in t) <= 1
        )
        out["within_one"] = _ratio(near, len(answered))
    out["confusion"] = {
        row: {c: confusion[row][c] for c in order if confusion[row][c]}
        for row in order
        if row in confusion
    }
    return out


def _binary(rows: list[tuple[bool, bool]], negatives: bool) -> dict[str, Any]:
    present = [p for t, p in rows if t]
    absent = [p for t, p in rows if not t]
    return {
        "n": len(rows),
        "labelled_present": len(present),
        "recall": _ratio(sum(present), len(present)),
        "false_positive_rate": _ratio(sum(absent), len(absent)) if negatives else None,
    }


def _auc(rows: list[tuple[set[str], float]]) -> float | None:
    """Chance that an exterior shot has a higher outdoor probability than an interior one."""
    outside = [p for t, p in rows if t == {"outdoor"}]
    inside = [p for t, p in rows if t == {"indoor"}]
    if not outside or not inside:
        return None
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0 for a in outside for b in inside
    )
    return round(wins / (len(outside) * len(inside)), 3)


def _consistency(items: list[tuple[str, int, Array]]) -> dict[str, Any]:
    """Is a segment's nearest other segment, by 02's keyframe embedding, from the same scene?
    Adjacent segments of one scene are left out, being trivially alike."""
    scenes = np.array([s for s, _, _ in items])
    if len(set(scenes.tolist())) < 2:
        return {"scenes": len(set(scenes.tolist()))}
    index = np.array([i for _, i, _ in items])
    vectors = np.stack([v for _, _, v in items])
    sims = vectors @ vectors.T
    same = np.equal.outer(scenes, scenes)
    allowed = ~(same & (np.abs(np.subtract.outer(index, index)) <= 1))
    hits: list[bool] = []
    chance: list[float] = []
    for a in range(len(items)):
        if not allowed[a].any():
            continue
        hits.append(bool(same[a, int(np.where(allowed[a], sims[a], -np.inf).argmax())]))
        chance.append(float(same[a][allowed[a]].mean()))
    return {
        "segments": len(items),
        "scenes": len(set(scenes.tolist())),
        "nearest_same_scene": _ratio(sum(hits), len(hits)),
        "chance": round(float(np.mean(chance)), 3),
        "cosine_within_scene": round(float(sims[same & allowed].mean()), 4),
        "cosine_between_scenes": round(float(sims[~same].mean()), 4),
    }


def _most_at_once(shot: dict[str, Any]) -> int:
    """Most people 06 observed in one frame of the segment; an untracked segment is 0."""
    frames = Counter(
        int(row["frame"])
        for track in shot.get("tracks", [])
        for row in track["frames"]
        if not row["interpolated"]
    )
    return max(frames.values(), default=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Score the pipeline against AVE's labels")
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "raw" / "ave" / "annotations.json",
    )
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "ave.json"
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="overlap a shot and a segment need to be compared",
    )
    p.add_argument(
        "--tolerance", type=int, default=2, help="frames a detected cut may be off by"
    )
    args = p.parse_args()

    labels: dict[str, Any] = json.loads(args.labels.read_text(encoding="utf-8"))
    scenes = sorted(
        s
        for s in labels
        if (args.processed / s / "json" / "segmentation.json").exists()
    )
    if not scenes:
        sys.exit(f"no AVE scene in {args.processed} has been through 02-segmentation")

    rows: dict[str, list[Row]] = {
        k: []
        for k in (
            "camera_motion",
            "people_visible",
            "people_tracked",
            "people_followed",
            "people_by_shot_type",
            "shot_size",
            "location",
        )
    }
    presence: dict[str, list[tuple[bool, bool]]] = {
        k: [] for k in ("person", "face", "text", "animal")
    }
    outdoor: list[tuple[set[str], float]] = []
    items: list[tuple[str, int, Array]] = []
    stages = Counter[str]()
    total_shots = total_segments = matched = 0
    truth_cuts = found_cuts = cut_hits = 0

    for scene in scenes:
        root = args.processed / scene
        segmentation = _load(root / "json" / "segmentation.json") or {}
        profile = _load(root / "json" / "profile.json")
        tracks = _load(root / "json" / "tracks.json")
        kit = _load(root / "identikit.json")
        stages.update(
            k
            for k, v in (("profile", profile), ("tracks", tracks), ("identikit", kit))
            if v
        )
        fps = float(segmentation["fps"])

        truth = sorted(
            (
                (
                    round(float(s["start-time"][0]) * fps),
                    round(float(s["end-time"][0]) * fps),
                    s,
                )
                for s in labels[scene].values()
            ),
            key=lambda t: t[0],
        )
        first, last = truth[0][0], max(end for _, end, _ in truth)
        segments = [
            s
            for s in segmentation["shots"]
            if s["end_frame"] >= first and s["start_frame"] <= last
        ]
        clipped = [
            (max(int(s["start_frame"]), first), min(int(s["end_frame"]), last))
            for s in segments
        ]
        total_shots += len(truth)
        total_segments += len(segments)

        cuts = [start for start, _, _ in truth[1:]]
        found = [
            int(s["start_frame"])
            for s in segments
            if first < int(s["start_frame"]) <= last
        ]
        truth_cuts += len(cuts)
        found_cuts += len(found)
        cut_hits += _cut_hits(cuts, found, args.tolerance)

        embeddings: Array = np.load(root / "embeddings" / "keyframe_embeddings.npy")
        row = 0
        for shot in segmentation["shots"]:
            count = len(shot["keyframes"])
            if shot in segments:
                vector = embeddings[row : row + count].mean(axis=0)
                items.append(
                    (scene, int(shot["index"]), vector / np.linalg.norm(vector))
                )
            row += count

        profiled = {int(s["index"]): s for s in (profile or {}).get("shots", [])}
        tracked = {int(s["index"]): s for s in (tracks or {}).get("shots", [])}
        enunciative = {
            int(s["segment"]): s
            for s in (kit or {}).get("enunciative", {}).get("segments", [])
        }
        figurative = {
            int(s["segment"]): s
            for s in (kit or {}).get("figurative", {}).get("segments", [])
        }
        animals = {
            int(i)
            for s in (kit or {}).get("figurative", {}).get("subjects", [])
            for i in s["segments"]
        }

        for i, j in _align([(a, b) for a, b, _ in truth], clipped, args.iou):
            label, segment = truth[i][2], segments[j]
            index = int(segment["index"])
            matched += 1
            people = _values(label, "num-people", PEOPLE)
            movement = (segment.get("camera") or {}).get("movement")
            rows["camera_motion"].append(
                (
                    _values(label, "shot-motion", MOTION),
                    None if movement in (None, "unknown") else movement,
                )
            )
            subjects = set(label.get("shot-subject", []))

            if index in profiled:
                shot = profiled[index]
                visible = int(shot["counts"]["person_max"])
                rows["people_visible"].append((people, _bin(visible, COUNTS)))
                rows["people_by_shot_type"].append(
                    (_values(label, "shot-type", SHOT_TYPE), _bin(visible, TYPE_COUNTS))
                )
                if people and not ("0" in people and len(people) > 1):
                    presence["person"].append(
                        ("0" not in people, bool(shot["flags"]["person"]))
                    )
                if subjects:
                    for subject, flag in FLAGS.items():
                        if flag != "person":
                            presence[flag].append(
                                (subject in subjects, bool(shot["flags"][flag]))
                            )
            if index in tracked:
                rows["people_tracked"].append(
                    (people, _bin(_most_at_once(tracked[index]), COUNTS))
                )
            if index in enunciative:
                scene_shot = enunciative[index]
                rows["people_followed"].append(
                    (people, _bin(int(scene_shot["followed"]), COUNTS))
                )
                distance = scene_shot.get("camera_distance")
                rows["shot_size"].append(
                    (
                        _values(label, "shot-size", SIZE),
                        distance["scale"] if distance else None,
                    )
                )
            if index in figurative:
                io = figurative[index].get("indoor_outdoor")
                place = _values(label, "shot-location", LOCATION)
                rows["location"].append((place, io["label"] if io else None))
                if io:
                    outdoor.append((place, float(io["outdoor"])))
                if subjects:
                    presence["animal"].append(("animal" in subjects, index in animals))

    precision = _ratio(cut_hits, found_cuts)
    recall = _ratio(cut_hits, truth_cuts)
    fields = {
        "camera_motion": _categorical(
            rows["camera_motion"], tuple(MOTION.values()), False
        ),
        "people_visible": _categorical(rows["people_visible"], COUNTS, True),
        "people_tracked": _categorical(rows["people_tracked"], COUNTS, True),
        "people_followed": _categorical(rows["people_followed"], COUNTS, True),
        "people_by_shot_type": _categorical(
            rows["people_by_shot_type"], TYPE_COUNTS, True
        ),
        "shot_size": _categorical(rows["shot_size"], tuple(SIZE.values()), True),
        "location": {
            **_categorical(rows["location"], tuple(LOCATION.values()), False),
            "auc": _auc(outdoor),
        },
    }
    report = {
        "scenes": len(scenes),
        "scenes_with": {"segmentation": len(scenes), **stages},
        "alignment": {
            "iou_floor": args.iou,
            "ave_shots": total_shots,
            "segments": total_segments,
            "matched": matched,
        },
        "cuts": {
            "tolerance_frames": args.tolerance,
            "ave": truth_cuts,
            "pipeline": found_cuts,
            "matched": cut_hits,
            "precision": precision,
            "recall": recall,
            "f1": round(2 * precision * recall / (precision + recall), 3)
            if precision and recall
            else None,
        },
        "fields": fields,
        "presence": {
            k: _binary(v, k not in RECALL_ONLY) for k, v in presence.items() if v
        },
        "scene_consistency": _consistency(items),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"{len(scenes)} scenes; {matched} of {total_shots} AVE shots matched a segment of {total_segments}"
    )
    print(f"cuts: precision {precision}, recall {recall}, f1 {report['cuts']['f1']}")
    for name, s in fields.items():
        print(
            f"{name:20} n {s['n']:5}  coverage {s['coverage']}  accuracy {s.get('accuracy')}  "
            f"balanced {s.get('balanced_accuracy')}  baseline {s.get('majority_baseline')}"
        )
    for name, s in report["presence"].items():
        print(
            f"{name:20} n {s['n']:5}  recall {s['recall']}  false positives {s['false_positive_rate']}"
        )
    print(f"scene consistency: {report['scene_consistency']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
