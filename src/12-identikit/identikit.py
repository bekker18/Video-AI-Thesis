"""12-identikit: the terminal artifact, assembled from what 03, 08, 09, 10 and 11 wrote.
Measures nothing, and never opens tracks.json or conditional.json."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from config import Config

STAGE = "12-identikit"
INPUTS = ("aggregated.json", "relations.json", "identity.json", "profile.json")

# 07 scores attributes one at a time, so a person can rank high on both halves of a pair.
EXCLUSIVE: tuple[tuple[str, ...], ...] = (
    ("a person with a beard", "a clean-shaven person"),
    ("a person with a moustache", "a clean-shaven person"),
    ("a person with long hair", "a person with short hair", "a bald person"),
    ("a person with curly hair", "a person with straight hair", "a bald person"),
    ("a person with blond hair", "a person with dark hair", "a person with grey hair", "a bald person"),
    ("a person looking at the camera", "a person looking away"),
    ("a person looking at the camera", "a person in profile view"),
    ("a person wearing eyeglasses", "a person wearing sunglasses"),
    ("a person wearing a hat", "a person wearing a cap", "a person wearing a headscarf"),
)
CONTRADICTS = {
    frozenset(pair) for group in EXCLUSIVE for pair in itertools.combinations(group, 2)
}

# Banded on how many 03 saw at once, not how many 08 followed: crowdedness describes the frame.
CROWD = ((0, "none"), (1, "single"), (2, "couple"), (5, "group"))

PENDING: dict[str, str] = {
    "plastic.video.grayscale": "05 does not compute it",
    "plastic.video.colour_histogram": "05 does not compute it",
    "plastic.segments[].palette": "05 does not compute it",
    "plastic.segments[].centralities": "09 does not carry 05's panoptic boxes forward",
    "plastic.segments[].depth": "05 writes depth maps that no stage reads back",
    "figurative.segments[].indoor_outdoor": "Places365's indoor/outdoor labels are not fetched",
    "figurative.persons[].by_segment": "09 pools a person's samples across segments",
    "enunciative.persons[].gaze": "no gaze model is installed",
    "enunciative.segments[].camera_distance": "05 writes depth maps that no stage reads back",
    "enunciative.segments[].face_background_ratio": "09 does not carry face box areas forward",
    "enunciative.segments[].portrait_scene": "follows from face_background_ratio",
}

EXCLUDED: dict[str, str] = {
    "ethnicity": "left out deliberately: a perceived category against a fixed taxonomy, "
    "with error rates that differ across groups",
}

PROVENANCE: dict[str, str] = {
    "timeline": "02 shot boundaries and 08 person ids, via 09",
    "plastic.segments[].visual": "05 OpenCV and NumPy measures, via 09",
    "plastic.segments[].coverage": "05 SegFormer-B1 on ADE20K, via 09",
    "figurative.segments[].tags": "05 MobileCLIP against RAM's tag list, via 09",
    "figurative.segments[].place": "05 Places365 ResNet-18, via 09",
    "figurative.segments[].objects": "06 YOLO11 temporal union, via 09",
    "figurative.segments[].text": "07 OpenCV CRNN, via 09; persons bound by 10",
    "figurative.segments[].crowdedness": "03 YOLO11 person count over keyframes",
    "figurative.persons[].links": "08 ArcFace and YouTu ReID cosine",
    "figurative.persons[].age": "07 InsightFace genderage, via 09",
    "figurative.persons[].gender": "07 InsightFace genderage, via 09",
    "figurative.persons[].attributes": "07 MobileCLIP zero-shot, via 09",
    "figurative.persons[].emotion": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].valence": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].arousal": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].movement": "07 MediaPipe PoseLandmarker, via 09",
    "enunciative.persons[].head_pose": "07 MediaPipe FaceLandmarker, via 09",
    "enunciative.relations": "10, from 06 boxes and 07 head pose and affect",
    "enunciative.segments[].visible_at_once": "03 YOLO11 person count over keyframes",
    "enunciative.segments[].followed": "08 person ids, via 09",
}

CAVEATS: tuple[str, ...] = (
    "attention is head orientation, not gaze",
    "placement 'nearer' is a box-area proxy, not depth",
    "place comes from Places365 and is weak on broadcast and animated footage; tags are steadier",
    "age and gender are a perceived estimate from one small model, not a property of a person",
    "text is read by a CRNN in ten-character chunks, so word boundaries are not real",
    "visible_at_once is the maximum over a segment's keyframes, so it is itself a lower bound",
    "confidence is banded on time-series samples; crop-based fields carry their own n",
    "the caption is written by a VLM; `unsupported` lists objects it names that no detector found",
)


def _read(json_dir: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = json_dir / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _bytes(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _measure(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    """09's statistics cut to what a reader uses; sd is null below two samples."""
    if not stats or not int(stats.get("n", 0)):
        return None
    variance = stats.get("variance")
    return {
        "mean": stats["mean"],
        "sd": round(math.sqrt(float(variance)), 4) if variance is not None else None,
        "median": stats["median"],
        "n": stats["n"],
    }


def _weighted(values: list[tuple[float, float]]) -> dict[str, Any] | None:
    """Duration-weighted over segments, so a 0.3 s shot does not count as much as a 5 s one."""
    total = sum(weight for _, weight in values)
    if not values or total <= 0:
        return None
    mean = sum(value * weight for value, weight in values) / total
    sd = (
        math.sqrt(sum(weight * (value - mean) ** 2 for value, weight in values) / total)
        if len(values) >= 2
        else None
    )
    return {
        "mean": round(mean, 4),
        "sd": round(sd, 4) if sd is not None else None,
        "n": len(values),
    }


def _crowd(count: int) -> str:
    for ceiling, label in CROWD:
        if count <= ceiling:
            return label
    return "crowd"


def _assess(at_once: int, followed: int) -> str:
    """Only more visible than followed is evidence; turnover legitimately raises followed."""
    if not at_once and not followed:
        return "empty"
    if at_once > followed:
        return "under_tracked"
    if at_once and followed > 2 * at_once:
        return "turnover_or_fragmented"
    return "consistent"


def _attributes(
    ranked: list[dict[str, Any]], keep: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """The better-ranked label wins each contradiction among the displayed; losers are returned."""
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for item in ranked:
        if len(kept) == keep:
            break
        label = str(item["label"])
        if any(frozenset((label, str(k["label"]))) in CONTRADICTS for k in kept):
            dropped.append(label)
        else:
            kept.append({"label": label, "score": item["score"], "n": item["n"]})
    return kept, dropped


def _text(
    readings: list[dict[str, Any]], bound: list[dict[str, Any]], keep: int
) -> list[dict[str, Any]]:
    """09's deduplicated readings, with the people 10 found each one overlapping."""
    persons: dict[str, set[int]] = {}
    for entry in bound:
        persons.setdefault(str(entry["text"]), set()).update(
            int(p) for p in entry.get("persons", [])
        )
    return [
        {**reading, "persons": sorted(persons.get(str(reading["text"]), set()))}
        for reading in readings[:keep]
    ]


def _evidenced(relation: dict[str, Any], floor: float) -> bool:
    """Listed when a head points at the other often enough, or affect moves together."""
    attention = relation["attention"]
    share = max(float(attention[side].get("share") or 0.0) for side in ("a_to_b", "b_to_a"))
    return share >= floor or bool(relation["synchrony"]["available"])


def _relation(relation: dict[str, Any]) -> dict[str, Any]:
    attention = relation["attention"]
    synchrony = relation["synchrony"]
    placement = relation["placement"]
    heads = attention["a_to_b"].get("available") or attention["b_to_a"].get("available")
    return {
        "pair": [relation["a"], relation["b"]],
        "co_present": {
            "frames": relation["co_present"]["n"],
            "segments": relation["co_present"]["segments"],
        },
        "placement": {
            k: placement[k] for k in ("horizontal", "vertical", "nearer", "size_ratio", "n")
        },
        "attention": {
            "a_to_b": attention["a_to_b"].get("share"),
            "b_to_a": attention["b_to_a"].get("share"),
            "mutual": attention["mutual"],
        }
        if heads
        else None,
        "synchrony": {k: synchrony.get(k) for k in ("valence", "arousal", "n")}
        if synchrony["available"]
        else None,
    }


def _person(
    person: dict[str, Any], links: list[dict[str, Any]], keep: int
) -> tuple[dict[str, Any], list[str]]:
    demographics = person.get("demographics") or {}
    gender = demographics.get("gender")
    emotion = person.get("emotion") or {}
    attributes, dropped = _attributes(person.get("attributes") or [], keep)
    series = int(person["support"]["series"])
    record = {
        "person_id": person["person_id"],
        "segments": person["segments"],
        "tracks": person["tracks"],
        "seconds": person["duration_seconds"],
        "linked": person["linked"],
        "abstained": person["abstained"],
        "links": links,
        "age": _measure(demographics.get("age")),
        "gender": {k: gender[k] for k in ("label", "agreement", "confidence", "n")}
        if gender
        else None,
        "attributes": attributes,
        "emotion": {k: emotion[k] for k in ("modal", "agreement", "n", "distribution")}
        if emotion.get("n")
        else None,
        "valence": _measure(person.get("valence")),
        "arousal": _measure(person.get("arousal")),
        "movement": _measure((person.get("pose") or {}).get("movement")),
        "by_segment": None,
        "support": person["support"],
        "confidence": "good" if series >= 10 else ("moderate" if series >= 2 else "thin"),
    }
    return record, dropped


def _head_pose(person: dict[str, Any]) -> dict[str, Any] | None:
    pose = person.get("head_pose")
    if not pose:
        return None
    return {axis: _measure(pose.get(axis)) for axis in ("yaw", "pitch", "roll")}


def _caption(json_dir: Path, shots: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """11 runs a VLM, so a missing or stale caption is a gap, not an error."""
    path = json_dir / "caption.json"
    if not path.exists():
        return None, "11-caption has not run"
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if meta["segments"] != [[int(s["start_frame"]), int(s["end_frame"])] for s in shots]:
        return None, "caption.json was written for other segments; re-run 11-caption"
    return meta, ""


def run(cfg: Config) -> dict[str, Any]:
    aggregated = _read(cfg.json_dir, "aggregated.json", "09-aggregation")
    relations = _read(cfg.json_dir, "relations.json", "10-relations")
    identity = _read(cfg.json_dir, "identity.json", "08-identity")
    profile = _read(cfg.json_dir, "profile.json", "03-profiler")

    keep = cfg.identikit_labels
    pending, provenance = dict(PENDING), dict(PROVENANCE)
    caption_meta, caption_gap = _caption(cfg.json_dir, aggregated["shots"])
    caption: dict[str, Any] | None = None
    if caption_meta is None:
        pending["figurative.caption"] = caption_gap
    else:
        caption = {
            "text": caption_meta["caption"],
            **{k: caption_meta[k] for k in ("words", "clip_score", "unsupported")},
        }
        provenance["figurative.caption"] = (
            f"11 {caption_meta['model']} on 02 keyframes, with hints from 03, 05, 06, 09 and 10"
        )
    shots = aggregated["shots"]
    at_once = {int(s["index"]): int(s["counts"]["person_max"]) for s in profile["shots"]}
    bound_text = {int(s["index"]): s["text"] for s in relations["shots"]}
    links_of = {
        int(p["person_id"]): [
            {k: link[k] for k in ("modality", "similarity", "threshold", "margin")}
            for link in p["links"]
        ]
        for p in identity["persons"]
    }

    duration = max(float(s["end_time"]) for s in shots)
    frames = max(int(s["end_frame"]) for s in shots) + 1
    lengths = [float(s["end_time"]) - float(s["start_time"]) for s in shots]

    # Described means at least one expert value reached the record.
    described: list[dict[str, Any]] = []
    heads: list[dict[str, Any]] = []
    tracked_only: list[int] = []
    conflicts: list[dict[str, Any]] = []
    for person in aggregated["persons"]:
        pid = int(person["person_id"])
        record, dropped = _person(person, links_of.get(pid, []), keep)
        head_pose = _head_pose(person)
        measured = ("age", "gender", "attributes", "emotion", "valence", "arousal", "movement")
        if not any(record[k] for k in measured) and head_pose is None:
            tracked_only.append(pid)
            continue
        described.append(record)
        if head_pose is not None:
            heads.append({"person_id": pid, "head_pose": head_pose, "gaze": None})
        if dropped:
            conflicts.append({"person_id": pid, "dropped": dropped})

    timeline_segments: list[dict[str, Any]] = []
    plastic_segments: list[dict[str, Any]] = []
    figurative_segments: list[dict[str, Any]] = []
    enunciative_segments: list[dict[str, Any]] = []
    weighted: dict[str, list[tuple[float, float]]] = {}
    for shot, seconds in zip(shots, lengths):
        index = int(shot["index"])
        plastic = shot["plastic"]
        followed = len(shot["person_ids"])
        timeline_segments.append(
            {
                "segment": index,
                "start": shot["start_time"],
                "end": shot["end_time"],
                "seconds": round(seconds, 3),
                "person_ids": shot["person_ids"],
            }
        )

        visual = {
            key: stats["mean"]
            for key, stats in (plastic.get("visual") or {}).items()
            if int(stats.get("n", 0))
        }
        for key, value in visual.items():
            weighted.setdefault(key, []).append((float(value), seconds))
        coverage = ((plastic.get("semantic") or {}).get("coverage")) or {}
        plastic_segments.append(
            {
                "segment": index,
                "visual": visual,
                "coverage": {name: c["mean"] for name, c in coverage.items()},
                "palette": None,
                "centralities": None,
                "depth": None,
            }
        )

        figurative_segments.append(
            {
                "segment": index,
                "tags": [t["label"] for t in (plastic.get("tags") or [])[:keep]],
                "place": [
                    {"label": p["label"], "score": p["score"]}
                    for p in (plastic.get("scene") or [])[:keep]
                ],
                "indoor_outdoor": None,
                "objects": [
                    {"label": o["label"], "frames": o["frame_count"]}
                    for o in shot.get("objects", [])
                ],
                "text": _text(shot["text"], bound_text.get(index, []), keep),
                "text_readings": len(shot["text"]),
                "crowdedness": _crowd(at_once.get(index, 0)),
            }
        )

        enunciative_segments.append(
            {
                "segment": index,
                "visible_at_once": at_once.get(index, 0),
                "followed": followed,
                "assessment": _assess(at_once.get(index, 0), followed),
                "camera_distance": None,
                "face_background_ratio": None,
                "portrait_scene": None,
            }
        )

    kept = [
        _relation(r)
        for r in relations["relations"]
        if _evidenced(r, cfg.identikit_attention)
    ]
    persons = aggregated["persons"]

    record: dict[str, Any] = {
        "video": aggregated["video"],
        "generated": {
            "stage": STAGE,
            "inputs": list(INPUTS) + (["caption.json"] if caption_meta else []),
            "tracking": identity["source"],
        },
        "summary": {
            "duration_s": round(duration, 3),
            "frames": frames,
            "segments": len(shots),
            "persons": {
                "resolved": len(persons),
                "described": len(described),
                "tracked_only": len(tracked_only),
                "abstained": sum(1 for p in persons if p["abstained"]),
                "thin": aggregated["thin_records"],
            },
            "relations": {
                "co_present": relations["relation_count"],
                "with_evidence": len(kept),
                "synchrony": relations["synchrony_edges"],
                "attention_floor": cfg.identikit_attention,
            },
        },
        "timeline": {
            "segments": timeline_segments,
            "rhythm": {
                "shots": len(shots),
                "mean_s": round(sum(lengths) / len(lengths), 3),
                "min_s": round(min(lengths), 3),
                "max_s": round(max(lengths), 3),
                "shots_per_minute": round(60 * len(shots) / duration, 2) if duration else 0.0,
            },
            "presence": [
                {
                    "person_id": p["person_id"],
                    "segments": p["segments"],
                    "first_frame": p["first_frame"],
                    "last_frame": p["last_frame"],
                    "seconds": p["duration_seconds"],
                }
                for p in persons
            ],
        },
        "plastic": {
            "video": {
                **{key: _weighted(values) for key, values in weighted.items()},
                "grayscale": None,
                "colour_histogram": None,
            },
            "segments": plastic_segments,
        },
        "figurative": {
            "caption": caption,
            "segments": figurative_segments,
            "persons": described,
            "tracked_only": {"count": len(tracked_only), "ids": tracked_only},
        },
        "enunciative": {
            "persons": heads,
            "relations": kept,
            "segments": enunciative_segments,
        },
        "provenance": provenance,
        "gaps": {
            "pending": pending,
            "excluded": EXCLUDED,
            "abstained": [int(p["person_id"]) for p in persons if p["abstained"]],
            "thin": [int(p["person_id"]) for p in persons if int(p["support"]["series"]) < 2],
            "segments_without_conditional_branch": [
                int(s["index"]) for s in shots if not s["tracked"]
            ],
            "under_tracked_segments": [
                e["segment"] for e in enunciative_segments if e["assessment"] == "under_tracked"
            ],
            "attribute_conflicts": conflicts,
            "face_outside_body": relations["face_outside_body"],
            "caveats": list(CAVEATS),
        },
    }
    inputs = _bytes(aggregated) + _bytes(relations) + _bytes(identity) + _bytes(profile)
    inputs += _bytes(caption_meta) if caption_meta else 0
    size = _bytes(record)
    record["budget"] = {
        "inputs_bytes": inputs,
        "record_bytes": size,
        "reduction": round(1 - size / inputs, 3) if inputs else 0.0,
    }

    (cfg.out_root / "identikit.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return {
        "video": record["video"],
        "segments": len(shots),
        **{f"persons_{k}": v for k, v in record["summary"]["persons"].items()},
        "relations_listed": len(kept),
        "budget": record["budget"],
    }
