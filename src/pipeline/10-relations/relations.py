"""10-relations: placement, gaze targets and affect synchrony between 09's persons."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from config import Config

STAGE = "10-relations"
IN_FRAME = 0.3  # in-frame probability a look needs to land; tuned on UCO-LAEO
LOOK_DEG = 20.0  # degrees a head may sit off the line of sight; tuned on UCO-LAEO
HEAD_REACH = 0.58  # head bottom below body top, in box widths (median, 191,824 faces)

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Head: TypeAlias = tuple[float, float, float]  # x, top, bottom


def _read(json_dir: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = json_dir / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _centre(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _overlaps(a: list[int], b: list[int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _head_centre(face: list[int]) -> tuple[float, float]:
    """Centre of the face box grown to a head, as 07 grows it for Gaze-LLE."""
    return (face[0] + face[2]) / 2.0, (face[1] + face[3]) / 2.0 - 0.15 * (
        face[3] - face[1]
    )


def _head_of(row: dict[str, Any]) -> Head:
    """A vertical span: the face's head centre, else down from the body box's top."""
    if row["face"]:
        x, y = _head_centre(row["face"])
        return x, y, y
    x1, y1, x2, y2 = row["body"]
    return (x1 + x2) / 2.0, y1, min(y2, y1 + HEAD_REACH * (x2 - x1))


def _landed(
    gaze: dict[str, Any],
    face: list[int],
    heads: dict[int, Head],
    cut: float = IN_FRAME,
    limit: float = LOOK_DEG,
) -> int | None:
    """Of heads within `limit` degrees of the sight line, the one nearest the point."""
    if float(gaze["in_frame"]) < cut:
        return None
    x, y = _head_centre(face)
    tx, ty = float(gaze["target"][0]), float(gaze["target"][1])
    if math.hypot(tx - x, ty - y) < 1.0:
        return None
    own = (face[3] - face[1]) / 2
    best: tuple[float, int] | None = None
    for pid, (hx, top, bottom) in heads.items():
        hy = min(max(ty, top), bottom)
        # a head on the looker's own is a duplicate detection
        if math.hypot(hx - x, hy - y) < own:
            continue
        turn = abs(math.atan2(hy - y, hx - x) - math.atan2(ty - y, tx - x))
        if math.degrees(min(turn, 2 * math.pi - turn)) > limit:
            continue
        distance = math.hypot(hx - tx, hy - ty)
        if best is None or distance < best[0]:
            best = (distance, pid)
    return best[1] if best else None


def _looks_at(
    looks: list[tuple[int, int | None, set[int]]], other: int
) -> dict[str, Any]:
    """Share of looks landing on the other while on screen, whole and per segment."""
    shared = [
        (segment, landed == other)
        for segment, landed, present in looks
        if other in present
    ]
    if not shared:
        return {
            "n": 0,
            "available": False,
            "reason": "no shared frame with gaze",
        }
    hits = sum(hit for _, hit in shared)
    by_segment: list[dict[str, int]] = []
    for segment in sorted({s for s, _ in shared}):
        mine = [hit for s, hit in shared if s == segment]
        by_segment.append({"segment": segment, "n": len(mine), "hits": sum(mine)})
    return {
        "n": len(shared),
        "available": True,
        "share": round(hits / len(shared), 3),
        "hits": hits,
        "by_segment": by_segment,
    }


def _synchrony(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], floor: int
) -> dict[str, Any]:
    """Correlation of two affect series; abstains below the sample floor."""
    shared = sorted(set(a) & set(b))
    usable = [
        f
        for f in shared
        if a[f].get("emotion", {}).get("valence") is not None
        and b[f].get("emotion", {}).get("valence") is not None
    ]
    if len(usable) < floor:
        return {
            "n": len(usable),
            "available": False,
            "reason": f"needs {floor} shared samples",
        }

    out: dict[str, Any] = {
        "n": len(usable),
        "available": True,
        "frames": [usable[0], usable[-1]],
    }
    for axis in ("valence", "arousal"):
        left = np.asarray([float(a[f]["emotion"][axis]) for f in usable])
        right = np.asarray([float(b[f]["emotion"][axis]) for f in usable])
        if left.std() < 1e-9 or right.std() < 1e-9:
            out[axis] = None  # a flat series correlates with nothing
        else:
            out[axis] = round(float(np.corrcoef(left, right)[0, 1]), 4)
    return out


def _placement(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Relative position over shared frames; box size ratio stands in for depth."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}

    dx: list[float] = []
    dy: list[float] = []
    ratio: list[float] = []
    for frame in shared:
        ax, ay = _centre(a[frame]["body"])
        bx, by = _centre(b[frame]["body"])
        dx.append(ax - bx)
        dy.append(ay - by)
        area_b = _area(b[frame]["body"])
        if area_b > 0:
            ratio.append(_area(a[frame]["body"]) / area_b)

    mean_dx, mean_dy = float(np.mean(dx)), float(np.mean(dy))
    mean_ratio = float(np.mean(ratio)) if ratio else 1.0
    return {
        "n": len(shared),
        "frames": [shared[0], shared[-1]],
        "horizontal": "left" if mean_dx < 0 else "right",
        "vertical": "above" if mean_dy < 0 else "below",
        "horizontal_px": round(mean_dx, 1),
        "vertical_px": round(mean_dy, 1),
        "size_ratio": round(mean_ratio, 3),
        "nearer": "a"
        if mean_ratio > 1.15
        else ("b" if mean_ratio < 0.87 else "similar"),
    }


def run(cfg: Config) -> dict[str, Any]:
    segmentation = _read(cfg.json_dir, "segmentation.json", "02-segmentation")
    tracks_meta = _read(cfg.json_dir, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.json_dir, "conditional.json", "07-conditional-experts")
    aggregated = _read(cfg.json_dir, "aggregated.json", "09-aggregation")

    rows: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for shot in tracks_meta["shots"]:
        if not shot["tracked"]:
            continue
        for track in shot["tracks"]:
            rows[(int(shot["index"]), int(track["track_id"]))] = {
                int(r["frame"]): r for r in track["frames"]
            }

    series: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    raw_text: dict[int, list[dict[str, Any]]] = {}
    for shot in conditional["shots"]:
        raw_text[int(shot["index"])] = shot["text"]
        for track in shot["tracks"]:
            series[(int(shot["index"]), int(track["track_id"]))] = {
                int(s["frame"]): s for s in track["series"]
            }

    # One table per person; frames cannot collide, as 08 never links tracks sharing one.
    person_rows: dict[int, dict[int, dict[str, Any]]] = {}
    person_series: dict[int, dict[int, dict[str, Any]]] = {}
    segment_of: dict[int, dict[int, int]] = {}
    for person in aggregated["persons"]:
        pid = int(person["person_id"])
        person_rows[pid] = {}
        person_series[pid] = {}
        segment_of[pid] = {}
        for track in person["tracks"]:
            key = (int(track["segment"]), int(track["track_id"]))
            person_rows[pid].update(rows.get(key, {}))
            person_series[pid].update(series.get(key, {}))
            for frame in rows.get(key, {}):
                segment_of[pid][frame] = key[0]

    heads: dict[int, dict[int, Head]] = {}
    for pid, table in person_rows.items():
        for frame, row in table.items():
            heads.setdefault(frame, {})[pid] = _head_of(row)
    looks: dict[int, list[tuple[int, int | None, set[int]]]] = {}
    for pid, samples in person_series.items():
        for frame, sample in sorted(samples.items()):
            face = person_rows[pid].get(frame, {}).get("face")
            if "gaze" not in sample or not face:
                continue
            others = {p: h for p, h in heads[frame].items() if p != pid}
            landed = _landed(sample["gaze"], face, others)
            looks.setdefault(pid, []).append(
                (segment_of[pid][frame], landed, set(others))
            )

    face_outside_body = sum(
        1
        for rows in person_rows.values()
        for r in rows.values()
        if r["face"] and not _overlaps(r["face"], r["body"])
    )

    relations: list[dict[str, Any]] = []
    synchrony_edges = looks_at_edges = 0
    for left, right in itertools.combinations(
        sorted(aggregated["persons"], key=lambda p: int(p["person_id"])), 2
    ):
        a_id, b_id = int(left["person_id"]), int(right["person_id"])
        a_rows, b_rows = person_rows[a_id], person_rows[b_id]
        together = sorted(set(a_rows) & set(b_rows))
        if not together:
            continue

        a_series, b_series = person_series[a_id], person_series[b_id]
        sync = _synchrony(a_series, b_series, cfg.sync_min)
        a_sees_b = _looks_at(looks.get(a_id, []), b_id)
        b_sees_a = _looks_at(looks.get(b_id, []), a_id)
        if a_sees_b.get("hits") or b_sees_a.get("hits"):
            looks_at_edges += 1

        if sync["available"]:
            synchrony_edges += 1
        relations.append(
            {
                "a": a_id,
                "b": b_id,
                "co_present": {
                    "n": len(together),
                    "frames": [together[0], together[-1]],
                    "segments": sorted({segment_of[a_id][f] for f in together}),
                },
                "placement": _placement(a_rows, b_rows),
                "looks_at": {
                    "a_to_b": a_sees_b,
                    "b_to_a": b_sees_a,
                    "mutual": bool(a_sees_b.get("hits") and b_sees_a.get("hits")),
                },
                "synchrony": sync,
            }
        )

    shots: list[dict[str, Any]] = []
    for shot in aggregated["shots"]:
        index = int(shot["index"])
        present = [int(p) for p in shot["person_ids"]]
        text: list[dict[str, Any]] = []
        for entry in raw_text.get(index, []):
            frame = int(entry["frame"])
            touching = [
                pid
                for pid in present
                if frame in person_rows.get(pid, {})
                and _overlaps(entry["box"], person_rows[pid][frame]["body"])
            ]
            text.append({**entry, "persons": sorted(touching)})
        shots.append(
            {
                "index": index,
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "person_ids": sorted(present),
                "text": text,
            }
        )

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "person_count": len(aggregated["persons"]),
        "relation_count": len(relations),
        "synchrony_edges": synchrony_edges,
        "looks_at_edges": looks_at_edges,
        "cross_segment_relations": sum(
            1 for r in relations if len(r["co_present"]["segments"]) > 1
        ),
        "look_degrees": LOOK_DEG,
        "look_in_frame": IN_FRAME,
        "synchrony_min_samples": cfg.sync_min,
        "face_outside_body": face_outside_body,
        "note": (
            "a look lands on the head nearest Gaze-LLE's point, of those within look_degrees of "
            "its line of sight, when its in-frame probability reaches look_in_frame; "
            "synchrony abstains below the sample floor; relations are between global persons "
            "and may span segments"
        ),
        "relations": relations,
        "shots": shots,
    }
    (cfg.json_dir / "relations.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k not in {"relations", "shots"}}
