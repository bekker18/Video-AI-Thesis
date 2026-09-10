"""09-aggregation: collapse per-frame and per-sample outputs into per-person records.

Aggregates at two scopes: one record per global person from 08, over every track that person
was resolved from, and one record per segment carrying the plastic timeline, text and object extents.
08 having run first is what makes the person scope available - before it,
the only key was (segment, track_id) and the same person in two segments was two strangers.

Every statistic is computed once, from the raw samples, over whole people.
Pooling 08's summaries instead would have been exact for means and variances and wrong for medians,
MADs and modal agreement, which is the argument that put consolidation ahead of this stage.

Every statistic carries its support count, and variance is null below two samples rather than
zero - most supports are still small, and saying so is the point.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from config import Config

STAGE = "09-aggregation"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

# Joints mediapipe places outside the crop are predictions, not observations.
IN_CROP = (0.0, 1.0)
CLASSICAL_KEYS = (
    "brightness",
    "contrast_rms",
    "saturation",
    "colourfulness",
    "focus",
    "edge_density",
    "horizontal_symmetry",
)


def _read(out_root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = out_root / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _stats(values: list[float]) -> dict[str, Any]:
    """Mean and variance for the dynamism signal, median and MAD because one bad crop moves
    a mean and not a median. Variance needs two samples; below that it is null, not zero."""
    n = len(values)
    if not n:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    out: dict[str, Any] = {
        "n": n,
        "mean": round(float(arr.mean()), 4),
        "median": round(median, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
    out["variance"] = round(float(arr.var(ddof=1)), 4) if n >= 2 else None
    out["mad"] = round(float(np.median(np.abs(arr - median))), 4) if n >= 2 else None
    return out


def _distribution(
    rows: list[dict[str, float]], labels: tuple[str, ...]
) -> dict[str, Any]:
    """Mean of the per-sample probability vectors, not a vote. The modal label plus
    the fraction of samples that agreed with it, so disagreement stays visible."""
    if not rows:
        return {"n": 0}
    matrix = np.asarray([[float(r.get(k, 0.0)) for k in labels] for r in rows])
    mean = matrix.mean(axis=0)
    modal = labels[int(np.argmax(mean))]
    per_sample = [labels[int(np.argmax(row))] for row in matrix]
    return {
        "n": len(rows),
        "modal": modal,
        "agreement": round(per_sample.count(modal) / len(per_sample), 3),
        "distribution": {k: round(float(v), 4) for k, v in zip(labels, mean)},
    }


def _labelled(entries: list[list[dict[str, Any]]], top: int) -> list[dict[str, Any]]:
    """Sparse label lists (tags, attributes, scenes) aggregated by mean score
    and how many of the sources actually offered the label."""
    totals: dict[str, list[float]] = {}
    for row in entries:
        for item in row:
            totals.setdefault(str(item["label"]), []).append(float(item["score"]))
    ranked = sorted(
        totals.items(), key=lambda kv: (-len(kv[1]), -float(np.mean(kv[1])), kv[0])
    )
    return [
        {"label": name, "score": round(float(np.mean(scores)), 4), "n": len(scores)}
        for name, scores in ranked[:top]
    ]


def _pose_summary(samples: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    """Movement is measured in crop-normalised coordinates, so it reads as change of posture
    rather than of camera distance. Joints predicted outside the crop are dropped first.

    A person's samples now span several segments, and the displacement between the last sample
    of one and the first of the next is a cut, not a movement. Consecutive pairs are taken only inside a segment.
    """
    usable: list[tuple[int, int, Array, Array]] = []
    for sample in samples:
        pose = sample.get("pose")
        if not pose:
            continue
        joints = np.asarray(pose["keypoints"], dtype=np.float64)
        keep = (
            (joints[:, 2] >= floor)
            & (joints[:, 0] >= IN_CROP[0])
            & (joints[:, 0] <= IN_CROP[1])
            & (joints[:, 1] >= IN_CROP[0])
            & (joints[:, 1] <= IN_CROP[1])
        )
        usable.append(
            (int(sample["segment"]), int(sample["frame"]), joints[:, :2], keep)
        )

    if not usable:
        return {"n": 0}

    usable.sort(key=lambda u: (u[0], u[1]))
    counts = [int(keep.sum()) for _, _, _, keep in usable]
    spread: list[float] = []
    stacked = np.stack([xy for _, _, xy, _ in usable])
    masks = np.stack([keep for _, _, _, keep in usable])
    for j in range(stacked.shape[1]):
        seen = stacked[masks[:, j], j]
        if len(seen) >= 2:
            spread.append(float(seen.var(axis=0).sum()))

    movement: list[float] = []
    for (segment, _, a, ka), (other, _, b, kb) in itertools.pairwise(usable):
        both = ka & kb
        if segment == other and both.any():
            movement.append(float(np.linalg.norm(b[both] - a[both], axis=1).mean()))

    return {
        "n": len(usable),
        "joints_used": _stats([float(c) for c in counts]),
        "movement": _stats(movement),
        "joint_spread": round(float(np.mean(spread)), 5) if spread else None,
    }


def _person(
    person: dict[str, Any],
    series: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    attrs: dict[str, list[Any]],
    fps: float,
    cfg: Config,
) -> dict[str, Any]:
    face = [s for s in series if "head_pose" in s]
    emotions = [s["emotion"] for s in series if "emotion" in s]
    # Summed, not last minus first: a person's segments are separated by everything that happened between them,
    # and that is not time they were on screen.
    frames = sum(int(s["frame_count"]) for s in spans)

    record: dict[str, Any] = {
        "person_id": person["person_id"],
        "tracks": person["tracks"],
        "segments": person["segments"],
        "linked": person["linked"],
        "abstained": person["abstained"],
        "first_frame": min(int(s["first_frame"]) for s in spans) if spans else None,
        "last_frame": max(int(s["last_frame"]) for s in spans) if spans else None,
        "duration_frames": frames,
        "duration_seconds": round(frames / fps, 3) if fps else 0.0,
        "support": {
            "segments": len(person["segments"]),
            "tracks": len(spans),
            "series": len(series),
            "face": len(face),
            "pose": sum(1 for s in series if "pose" in s),
            "crops": len(attrs.get("embedding_rows", [])),
            "body_crops": len(attrs.get("body_rows", [])),
            "observed_frames": sum(int(s["observed_frames"]) for s in spans),
        },
    }

    if face:
        record["head_pose"] = {
            axis: _stats([float(s["head_pose"][axis]) for s in face])
            for axis in ("yaw", "pitch", "roll")
        }
        shapes: dict[str, list[float]] = {}
        for sample in face:
            for name, value in sample.get("blendshapes", {}).items():
                shapes.setdefault(name, []).append(float(value))
        ranked = sorted(shapes.items(), key=lambda kv: (-float(np.mean(kv[1])), kv[0]))
        record["blendshapes"] = {
            name: _stats(values) for name, values in ranked[: cfg.agg_top]
        }

    if emotions:
        labels = tuple(sorted(emotions[0]["scores"]))
        record["emotion"] = _distribution([e["scores"] for e in emotions], labels)
        for axis in ("valence", "arousal"):
            present = [float(e[axis]) for e in emotions if e.get(axis) is not None]
            if present:
                record[axis] = _stats(present)

    pose = _pose_summary(series, cfg.pose_vis)
    if pose["n"]:
        record["pose"] = pose

    if attrs.get("gender_age"):
        ages = [float(g["age"]) for g in attrs["gender_age"]]
        votes = Counter(str(g["gender"]) for g in attrs["gender_age"])
        winner, count = votes.most_common(1)[0]
        confidence = [
            float(g["gender_score"])
            for g in attrs["gender_age"]
            if g["gender"] == winner
        ]
        record["demographics"] = {
            "age": _stats(ages),
            "gender": {
                "label": winner,
                "agreement": round(count / sum(votes.values()), 3),
                "confidence": round(float(np.mean(confidence)), 4),
                "n": sum(votes.values()),
            },
        }
    if attrs.get("clip"):
        record["attributes"] = _labelled(attrs["clip"], cfg.agg_top)

    return record


def _plastic(keyframes: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    """The always-on branch has no tracks, so it aggregates to the segment alone."""
    if not keyframes:
        return {"n": 0}

    out: dict[str, Any] = {"n": len(keyframes)}
    out["scene"] = _labelled(
        [k["scene"] for k in keyframes if k.get("scene")], cfg.agg_top
    )
    out["tags"] = _labelled(
        [k["tags"] for k in keyframes if k.get("tags")], cfg.agg_top
    )
    out["visual"] = {
        key: _stats(
            [float(k["classical"][key]) for k in keyframes if key in k["classical"]]
        )
        for key in CLASSICAL_KEYS
    }

    depth = [k["depth"] for k in keyframes if k.get("depth")]
    if depth:
        out["depth"] = {
            "min": _stats([float(d["min"]) for d in depth]),
            "max": _stats([float(d["max"]) for d in depth]),
            "relative": True,
        }
    edges = [float(k["edges"]["density"]) for k in keyframes if k.get("edges")]
    if edges:
        out["edge_density"] = _stats(edges)

    coverage: dict[str, list[float]] = {}
    for keyframe in keyframes:
        for name, value in keyframe.get("semantic", {}).get("coverage", {}).items():
            coverage.setdefault(str(name), []).append(float(value))
    if coverage:
        ranked = sorted(
            coverage.items(), key=lambda kv: (-float(np.mean(kv[1])), kv[0])
        )
        out["semantic"] = {
            "label_space": "ade20k",
            "coverage": {
                name: {"mean": round(float(np.mean(v)), 4), "n": len(v)}
                for name, v in ranked[: cfg.agg_top]
            },
        }
    counts = [
        float(k["panoptic"]["segment_count"]) for k in keyframes if k.get("panoptic")
    ]
    if counts:
        out["panoptic_segments"] = _stats(counts)
    return out


def _text(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same caption is detected on every keyframe of a shot, so readings are deduplicated
    and carry the range they were seen over."""
    grouped: dict[str, list[int]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["text"]), []).append(int(entry["frame"]))
    return [
        {
            "text": reading,
            "n": len(frames),
            "first_frame": min(frames),
            "last_frame": max(frames),
        }
        for reading, frames in sorted(
            grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
    ]


def run(cfg: Config) -> dict[str, Any]:
    segmentation = _read(cfg.out_root, "segmentation.json", "02-segmentation")
    routing = _read(cfg.out_root, "routing.json", "04-router")
    tracks_meta = _read(cfg.out_root, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.out_root, "conditional.json", "07-conditional-experts")
    global_meta = _read(cfg.out_root, "global.json", "05-global-experts")
    consolidated = _read(cfg.out_root, "consolidated.json", "08-consolidation")

    fps = float(segmentation["fps"])
    gates = {int(s["index"]): sorted(s["experts"]) for s in routing["shots"]}
    spans = {
        (int(s["index"]), int(t["track_id"])): t
        for s in tracks_meta["shots"]
        if s["tracked"]
        for t in s["tracks"]
    }
    tracks_by_key = {
        (int(s["index"]), int(t["track_id"])): t
        for s in conditional["shots"]
        for t in s["tracks"]
    }
    by_shot: dict[int, list[dict[str, Any]]] = {}
    for keyframe in global_meta["keyframes"]:
        by_shot.setdefault(int(keyframe["shot"]), []).append(keyframe)

    persons: list[dict[str, Any]] = []
    thin = 0
    for person in consolidated["persons"]:
        keys = [(int(t["segment"]), int(t["track_id"])) for t in person["tracks"]]
        present = [k for k in keys if k in spans]
        if not present:
            continue

        # Every sample carries the segment it came from: the statistics are order-free,
        # but pose movement is not, and neither is anything a later stage may want to split back.
        series: list[dict[str, Any]] = []
        attrs: dict[str, list[Any]] = {
            "embedding_rows": [],
            "body_rows": [],
            "gender_age": [],
            "clip": [],
        }
        for key in present:
            source = tracks_by_key.get(key)
            if source is None:
                continue
            series += [{**sample, "segment": key[0]} for sample in source["series"]]
            for field, values in source.get("attributes", {}).items():
                attrs.setdefault(field, []).extend(values)
        series.sort(key=lambda s: (int(s["segment"]), int(s["frame"])))

        record = _person(person, series, [spans[k] for k in present], attrs, fps, cfg)
        if record["support"]["series"] < 2:
            thin += 1
        persons.append(record)

    by_segment: dict[int, list[int]] = {}
    for record in persons:
        for segment in record["segments"]:
            by_segment.setdefault(int(segment), []).append(int(record["person_id"]))

    shots: list[dict[str, Any]] = []
    for shot in conditional["shots"]:
        index = int(shot["index"])
        source = next(s for s in tracks_meta["shots"] if int(s["index"]) == index)
        shots.append(
            {
                "index": index,
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "start_time": round(int(shot["start_frame"]) / fps, 3) if fps else 0.0,
                "end_time": round((int(shot["end_frame"]) + 1) / fps, 3)
                if fps
                else 0.0,
                "experts": gates.get(index, []),
                "tracked": bool(source["tracked"]),
                "person_ids": sorted(by_segment.get(index, [])),
                "plastic": _plastic(by_shot.get(index, []), cfg),
                "text": _text(shot["text"]),
                "objects": source.get("objects", []),
            }
        )

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "person_records": len(persons),
        "thin_records": thin,
        "multi_segment_persons": sum(1 for p in persons if len(p["segments"]) > 1),
        "pose_vis_floor": cfg.pose_vis,
        "top_k": cfg.agg_top,
        "note": (
            "records are video-scoped: one per global person from 08, over every track it was resolved from"
        ),
        "persons": persons,
        "shots": shots,
    }
    (cfg.out_root / "aggregated.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k not in {"persons", "shots"}}
