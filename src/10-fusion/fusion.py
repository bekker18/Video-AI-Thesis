"""10-fusion: bind experts onto entities and compute the relations between them.

09 produced the nodes: one record per global person. This produces the edges, which are the
facts no single person's record can hold - who attends whom, whose affect moves together,
who stands where relative to whom. Every edge carries the frames and sample counts behind it,
so a relation can be argued with rather than only read.

A person's frames are the union of the frames of the tracks 08 resolved them from,
and 08 guarantees those are disjoint - a person is never two boxes in one frame - which is what makes
the merged per-frame table well defined. A pair can now be co-present in several segments and
their evidence accumulates across all of them. That is what makes synchrony computable on
shot-heavy footage, where a segment-local series was two or three points and it abstained.

Attention is derived from head orientation, not from a gaze model, so it is named for what it measures.
Synchrony still abstains below a sample floor rather than correlating short series.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from config import Config

STAGE = "10-fusion"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]


def _read(out_root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = out_root / name
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


def _forward(pose: dict[str, float]) -> tuple[float, float]:
    """Image-plane direction the head faces, from the same Euler angles 07 recorded.
    Rebuilds R = Rz(roll) Ry(yaw) Rx(pitch) and projects its outward axis."""
    yaw, pitch, roll = (math.radians(float(pose[k])) for k in ("yaw", "pitch", "roll"))
    rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    ry = np.array(
        [
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)],
        ]
    )
    rz = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1],
        ]
    )
    vector = (rz @ ry @ rx) @ np.array([0.0, 0.0, -1.0])
    norm = float(np.linalg.norm(vector[:2]))
    return (
        (0.0, 0.0)
        if norm < 1e-6
        else (float(vector[0]) / norm, float(vector[1]) / norm)
    )


def _attends(
    samples: dict[int, dict[str, Any]],
    boxes: dict[int, dict[str, Any]],
    other: dict[int, dict[str, Any]],
    limit: float,
) -> dict[str, Any]:
    """Fraction of shared frames where the head points within `limit` degrees of the other person.
    Head orientation, not gaze - there is no gaze model in the pipeline."""
    hits = 0
    angles: list[float] = []
    for frame, sample in sorted(samples.items()):
        row, target = boxes.get(frame), other.get(frame)
        if not row or not target or "head_pose" not in sample or not row["face"]:
            continue
        fx, fy = _centre(row["face"])
        tx, ty = _centre(target["face"] or target["body"])
        towards = np.array([tx - fx, ty - fy])
        distance = float(np.linalg.norm(towards))
        if distance < 1e-6:
            continue

        facing = np.array(_forward(sample["head_pose"]))
        cosine = float(np.clip(np.dot(facing, towards / distance), -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        angles.append(angle)
        if angle <= limit:
            hits += 1

    if not angles:
        return {"n": 0, "available": False, "reason": "no shared frame with head pose"}
    return {
        "n": len(angles),
        "available": True,
        "share": round(hits / len(angles), 3),
        "min_angle": round(min(angles), 1),
        "mean_angle": round(float(np.mean(angles)), 1),
    }


def _synchrony(
    a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], floor: int
) -> dict[str, Any]:
    """Correlation of two affect series. Below the floor it abstains:
    correlating a handful of points produces a number with no meaning, and a number invites belief."""
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
    """Relative position over the frames both are present. Size ratio stands in for depth ordering:
    05 writes depth at keyframes only, so a per-frame ordering does not exist."""
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
    segmentation = _read(cfg.out_root, "segmentation.json", "02-segmentation")
    tracks_meta = _read(cfg.out_root, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.out_root, "conditional.json", "07-conditional-experts")
    aggregated = _read(cfg.out_root, "aggregated.json", "09-aggregation")

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

    # One table per person, the union over its tracks. Frames cannot collide:
    # 08 refuses to link two tracks that share one.
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

    entities: list[dict[str, Any]] = []
    for person in aggregated["persons"]:
        pid = int(person["person_id"])
        # 06 clamps a face to its body box, so this validates the binding rather than performing it:
        # face and body already share one track id.
        escaped = sum(
            1
            for r in person_rows[pid].values()
            if r["face"] and not _overlaps(r["face"], r["body"])
        )
        emotion = person.get("emotion") or {}
        entities.append(
            {
                "person_id": pid,
                "tracks": person["tracks"],
                "segments": person["segments"],
                "first_frame": person["first_frame"],
                "last_frame": person["last_frame"],
                "linked": person["linked"],
                "abstained": person["abstained"],
                "has_face": bool(person["support"]["face"]),
                "has_pose": bool(person["support"]["pose"]),
                "modal_emotion": emotion.get("modal"),
                "demographics": person.get("demographics"),
                "binding": {
                    "face_frames": person["support"]["face"],
                    "face_outside_body": escaped,
                },
            }
        )

    relations: list[dict[str, Any]] = []
    attention_edges = synchrony_edges = 0
    for left, right in itertools.combinations(
        sorted(aggregated["persons"], key=lambda p: int(p["person_id"])), 2
    ):
        a_id, b_id = int(left["person_id"]), int(right["person_id"])
        a_rows, b_rows = person_rows[a_id], person_rows[b_id]
        together = sorted(set(a_rows) & set(b_rows))
        if not together:
            continue

        a_series, b_series = person_series[a_id], person_series[b_id]
        a_to_b = _attends(a_series, a_rows, b_rows, cfg.attention_deg)
        b_to_a = _attends(b_series, b_rows, a_rows, cfg.attention_deg)
        sync = _synchrony(a_series, b_series, cfg.sync_min)

        if a_to_b["available"] or b_to_a["available"]:
            attention_edges += 1
        if sync["available"]:
            synchrony_edges += 1
        relations.append(
            {
                "a": a_id,
                "b": b_id,
                "co_present": {
                    "n": len(together),
                    "frames": [together[0], together[-1]],
                    # Two people can meet in more than one segment now,
                    # and how many is a different fact from how long.
                    "segments": sorted({segment_of[a_id][f] for f in together}),
                },
                "placement": _placement(a_rows, b_rows),
                "attention": {
                    "a_to_b": a_to_b,
                    "b_to_a": b_to_a,
                    "mutual": bool(
                        a_to_b.get("share", 0) > 0 and b_to_a.get("share", 0) > 0
                    ),
                },
                "synchrony": sync,
            }
        )

    # Text is bound to whichever people its region actually overlaps.
    # Full-width captions overlap everyone, which the counts make visible rather than hide.
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
        "person_count": len(entities),
        "relation_count": len(relations),
        "attention_edges": attention_edges,
        "synchrony_edges": synchrony_edges,
        "cross_segment_relations": sum(
            1 for r in relations if len(r["co_present"]["segments"]) > 1
        ),
        "attention_degrees": cfg.attention_deg,
        "synchrony_min_samples": cfg.sync_min,
        "note": (
            "attention is head orientation, not gaze; synchrony abstains below the sample "
            "floor; relations are between global persons and may span segments"
        ),
        "entities": entities,
        "relations": relations,
        "shots": shots,
    }
    (cfg.out_root / "fused.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {
        k: v for k, v in meta.items() if k not in {"entities", "relations", "shots"}
    }
