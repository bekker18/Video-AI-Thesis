"""12-identikit: the terminal artifact, assembled from 03, 08, 09, 10 and 11; measures nothing."""

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

# Shot scale on the median largest face, else body, height; first match wins, else "extreme long".
SCALE: tuple[tuple[str, str, float], ...] = (
    ("extreme close-up", "face_height", 0.75),
    ("close-up", "face_height", 0.30),
    ("medium", "face_height", 0.12),
    ("medium", "body_height", 0.75),
    ("long", "body_height", 0.25),
)
GRAYSCALE = 0.01  # weighted chroma; a grey keyframe re-encoded as JPEG reads 0.0, the dullest colour clip 0.03
PORTRAIT = 0.05  # median face area against the rest of the frame; close-ups on the six clips sit above it
WARM = 5.0  # mean b* beyond which a segment reads warm, or below its negative cool
ANIMALS = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)

PENDING: dict[str, str] = {}

EXCLUDED: dict[str, str] = {}

PROVENANCE: dict[str, str] = {
    "timeline": "02 shot boundaries and 08 person ids, via 09",
    "plastic.segments[].visual": "05 OpenCV and NumPy measures, via 09",
    "plastic.segments[].coverage": "05 SegFormer-B1 on ADE20K, via 09",
    "plastic.segments[].palette": "05 fixed colour cells, via 09",
    "plastic.segments[].centralities": "05 Mask2Former panoptic boxes, via 09",
    "plastic.segments[].depth": "05 Depth Anything V2 under Mask2Former segments, via 09",
    "plastic.video.colour_histogram": "05 RGB histograms, via 09, duration-weighted",
    "plastic.video.grayscale": "05 chroma, via 09, duration-weighted, against GRAYSCALE",
    "plastic.segments[].medium": "05 MobileCLIP zero-shot on 02's keyframe embeddings, via 09",
    "plastic.video.medium": "05 MobileCLIP zero-shot, via 09, duration-weighted",
    "plastic.segments[].temperature": "05 mean Lab b* per keyframe, via 09, against WARM",
    "figurative.subjects": "06 YOLO11 temporal union of animal classes, via 09, against --subject-min",
    "figurative.persons[].perceived_ethnicity": "07 FairFace ResNet-34 on live-action face crops "
    "clearing 08's floors, via 09",
    "figurative.persons[].demographic_check": "07 InsightFace genderage against FairFace, via 09",
    "enunciative.persons[].address": "07 MediaPipe head pose and Gaze-LLE, via 09",
    "enunciative.segments[].address": "07 MediaPipe head pose and Gaze-LLE, via 09",
    "enunciative.segments[].camera": "02 LK flow on corners and a RANSAC similarity, via 09",
    "figurative.segments[].indoor_outdoor": "05 Places365 ResNet-18 with Places365's IO list, via 09",
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
    "figurative.persons[].by_segment": "07 HSEmotion and PoseLandmarker samples split by segment, via 09",
    "enunciative.persons[].head_pose": "07 MediaPipe FaceLandmarker, via 09",
    "enunciative.relations": "10, from 06 boxes and 07 head pose and affect",
    "enunciative.relations[].looks_at": "10, from 07 Gaze-LLE look-at points in 06 body boxes",
    "enunciative.persons[].gaze": "07 Gaze-LLE with DINOv2 ViT-B on the frame and head box, via 09",
    "enunciative.segments[].visible_at_once": "03 YOLO11 person count over keyframes",
    "enunciative.segments[].followed": "08 person ids, via 09",
    "enunciative.segments[].camera_distance": "06 YuNet face and YOLO11 body boxes, via 09",
    "enunciative.segments[].face_background_ratio": "06 YuNet face boxes, via 09",
    "enunciative.segments[].portrait_scene": "face_background_ratio against PORTRAIT",
}

CAVEATS: tuple[str, ...] = (
    "attention is head orientation, not gaze; looks_at is gaze",
    "gaze comes from an enlarged YuNet face box and the scene: faces YuNet misses have none, and "
    "the model was trained on real video, not animation or CGI",
    "placement 'nearer' is a box-area proxy, not depth",
    "place comes from Places365 and is weak on broadcast and animated footage; tags are steadier",
    "depth is relative disparity within each keyframe, 1 nearest: an order and ratios, not distances",
    "centralities and depth are per label: panoptic segments are not matched across keyframes",
    "age and gender are a perceived estimate from one small model, not a property of a person",
    "text is read by a CRNN in ten-character chunks, so word boundaries are not real",
    "visible_at_once is the maximum over a segment's keyframes, so it is itself a lower bound",
    "camera_distance is a shot scale from the largest followed person, not a distance; faces YuNet misses read as wider shots",
    "confidence is banded on time-series samples; crop-based fields carry their own n",
    "the caption is written by a VLM; `unsupported` lists objects it names that no detector found",
    "perceived_ethnicity is FairFace's reading against its own 7 groups, not a property of a person; "
    "it is read on live-action faces only, and photoreal CGI passes that gate",
    "medium separates 2D animation well; photoreal CGI reads as live action",
    "address is the head turned to the lens with the gaze off the frame, not a measured eye contact",
    "camera movement is one global similarity per frame pair: a large moving subject can read as a pan",
    "temperature is the keyframes' mean colour, so grading counts as much as light",
    "subjects are animal classes the detector saw often enough, neither tracked nor told apart",
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
    looks = relation["looks_at"]
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
        "looks_at": {
            "a_to_b": looks["a_to_b"].get("share"),
            "b_to_a": looks["b_to_a"].get("share"),
            "mutual": looks["mutual"],
        }
        if looks["a_to_b"].get("available") or looks["b_to_a"].get("available")
        else None,
        "synchrony": {k: synchrony.get(k) for k in ("valence", "arousal", "n")}
        if synchrony["available"]
        else None,
    }


def _gaze(person: dict[str, Any]) -> dict[str, Any] | None:
    gaze = person.get("gaze")
    if not gaze:
        return None
    return {
        "in_frame_share": gaze["in_frame_share"],
        "in_frame": _measure(gaze["in_frame"]),
        "n": gaze["n"],
    }


def _counted(found: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """The named keys of a 09 block, or null when it rests on no samples."""
    if not found or not found.get("n"):
        return None
    return {k: found[k] for k in keys}


def _temperature(warmth: float | None) -> str | None:
    if warmth is None:
        return None
    return "warm" if warmth > WARM else ("cool" if warmth < -WARM else "neutral")


def _medium(mediums: list[tuple[dict[str, float], float]]) -> dict[str, Any] | None:
    """Segment distributions weighted by duration."""
    total = sum(seconds for _, seconds in mediums)
    if not mediums or total <= 0:
        return None
    mix = {
        k: round(sum(d[k] * seconds for d, seconds in mediums) / total, 4)
        for k in mediums[0][0]
    }
    return {"label": max(mix.items(), key=lambda kv: kv[1])[0], "distribution": mix}


def _subjects(
    shots: list[dict[str, Any]], lengths: list[float], floor: float
) -> list[dict[str, Any]]:
    """Animals seen in at least `floor` of a segment's detection frames; seconds are share x duration."""
    found: dict[str, dict[str, Any]] = {}
    for shot, seconds in zip(shots, lengths):
        for o in shot.get("objects", []):
            if o["label"] in ANIMALS and float(o["share"]) >= floor:
                entry = found.setdefault(o["label"], {"label": o["label"], "segments": [], "seconds": 0.0})
                entry["segments"].append(int(shot["index"]))
                entry["seconds"] += float(o["share"]) * seconds
    return sorted(
        ({**e, "seconds": round(e["seconds"], 2)} for e in found.values()),
        key=lambda e: (-e["seconds"], e["label"]),
    )


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
        "perceived_ethnicity": _counted(
            person.get("perceived_ethnicity"),
            ("taxonomy", "modal", "agreement", "distribution", "n"),
        ),
        "demographic_check": _counted(
            demographics.get("cross_check"), ("gender_agreement", "age_in_bracket", "n")
        ),
        "attributes": attributes,
        "emotion": {k: emotion[k] for k in ("modal", "agreement", "n", "distribution")}
        if emotion.get("n")
        else None,
        "valence": _measure(person.get("valence")),
        "arousal": _measure(person.get("arousal")),
        "movement": _measure((person.get("pose") or {}).get("movement")),
        "by_segment": [
            {
                "segment": s["segment"],
                "seconds": s["duration_seconds"],
                "emotion": {k: s["emotion"][k] for k in ("modal", "agreement", "n")}
                if (s.get("emotion") or {}).get("n")
                else None,
                "valence": _measure(s.get("valence")),
                "arousal": _measure(s.get("arousal")),
                "movement": _measure(s.get("movement")),
            }
            for s in person.get("by_segment") or []
        ],
        "support": person["support"],
        "confidence": "good" if series >= 10 else ("moderate" if series >= 2 else "thin"),
    }
    return record, dropped


def _head_pose(person: dict[str, Any]) -> dict[str, Any] | None:
    pose = person.get("head_pose")
    if not pose:
        return None
    return {axis: _measure(pose.get(axis)) for axis in ("yaw", "pitch", "roll")}


def _histogram(histograms: list[tuple[dict[str, list[float]], float]]) -> dict[str, Any] | None:
    """Segment histograms weighted by duration, like the other video-level plastic values."""
    total = sum(seconds for _, seconds in histograms)
    if not histograms or total <= 0:
        return None
    return {
        channel: [
            round(sum(h[channel][i] * seconds for h, seconds in histograms) / total, 4)
            for i in range(len(histograms[0][0][channel]))
        ]
        for channel in ("r", "g", "b")
    }


def _band(value: float, names: tuple[str, str, str]) -> str:
    return names[0] if value < 1 / 3 else (names[2] if value > 2 / 3 else names[1])


def _centralities(plastic: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": t["label"],
            "horizontal": _band(float(t["x"]), ("left", "centre", "right")),
            "vertical": _band(float(t["y"]), ("top", "middle", "bottom")),
            **{k: t[k] for k in ("x", "y", "centrality", "area", "spread", "n")},
        }
        for t in plastic.get("things") or []
    ]


def _depth(plastic: dict[str, Any]) -> dict[str, Any] | None:
    things = plastic.get("things") or []
    background = (plastic.get("background_disparity") or {}).get("mean")
    if not things and background is None:
        return None
    return {
        "background": background,
        "objects": [
            {
                "label": t["label"],
                "disparity": t["disparity"],
                "ratio": round(float(t["disparity"]) / background, 3) if background else None,
            }
            for t in things
        ],
        "order": [t["label"] for t in sorted(things, key=lambda t: (-float(t["disparity"]), t["label"]))],
    }


def _indoor_outdoor(plastic: dict[str, Any]) -> dict[str, Any] | None:
    stats = plastic.get("outdoor") or {}
    if not stats.get("n"):
        return None
    label = "outdoor" if float(stats["mean"]) >= 0.5 else "indoor"
    return {"label": label, "outdoor": stats["mean"], "n": stats["n"]}


def _framing(framing: dict[str, Any]) -> dict[str, Any]:
    """Null where nobody was followed: there is no one to frame."""
    if not framing.get("frames"):
        return {"camera_distance": None, "face_background_ratio": None, "portrait_scene": None}
    medians = {k: float(framing[k]["median"]) for k in ("face_height", "body_height")}
    scale = next((label for label, key, floor in SCALE if medians[key] >= floor), "extreme long")
    ratio = float(framing["face_background"]["median"])
    return {
        "camera_distance": {"scale": scale, **medians, "frames": framing["frames"]},
        "face_background_ratio": ratio,
        "portrait_scene": "portrait" if ratio >= PORTRAIT else "scene",
    }


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
    ethnicity_withheld: list[dict[str, Any]] = []
    for person in aggregated["persons"]:
        pid = int(person["person_id"])
        record, dropped = _person(person, links_of.get(pid, []), keep)
        head_pose = _head_pose(person)
        measured = (
            "age",
            "gender",
            "perceived_ethnicity",
            "attributes",
            "emotion",
            "valence",
            "arousal",
            "movement",
        )
        if not any(record[k] for k in measured) and head_pose is None:
            tracked_only.append(pid)
            continue
        described.append(record)
        if head_pose is not None:
            heads.append(
                {
                    "person_id": pid,
                    "head_pose": head_pose,
                    "gaze": _gaze(person),
                    "address": _counted(person.get("address"), ("share", "n")),
                }
            )
        if dropped:
            conflicts.append({"person_id": pid, "dropped": dropped})
        withheld = person.get("perceived_ethnicity")
        if withheld and not withheld["n"]:
            ethnicity_withheld.append({"person_id": pid, **withheld["withheld"]})

    timeline_segments: list[dict[str, Any]] = []
    plastic_segments: list[dict[str, Any]] = []
    figurative_segments: list[dict[str, Any]] = []
    enunciative_segments: list[dict[str, Any]] = []
    weighted: dict[str, list[tuple[float, float]]] = {}
    histograms: list[tuple[dict[str, list[float]], float]] = []
    mediums: list[tuple[dict[str, float], float]] = []
    moves: dict[str, float] = {}
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
        if plastic.get("histogram"):
            histograms.append((plastic["histogram"], seconds))
        medium = plastic.get("medium") or {}
        if medium.get("n"):
            mediums.append((medium["distribution"], seconds))
        camera = shot.get("camera") or {}
        if camera.get("movement"):
            moves[camera["movement"]] = moves.get(camera["movement"], 0.0) + seconds
        coverage = ((plastic.get("semantic") or {}).get("coverage")) or {}
        plastic_segments.append(
            {
                "segment": index,
                "medium": _counted(medium, ("modal", "agreement", "n")),
                "temperature": _temperature(visual.get("warmth")),
                "visual": visual,
                "coverage": {name: c["mean"] for name, c in coverage.items()},
                "palette": [
                    {"hex": "#{:02x}{:02x}{:02x}".format(*c["rgb"]), "share": c["share"]}
                    for c in plastic.get("palette") or []
                ],
                "centralities": _centralities(plastic),
                "depth": _depth(plastic),
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
                "indoor_outdoor": _indoor_outdoor(plastic),
                "objects": [
                    {"label": o["label"], "frames": o["frame_count"], "share": o["share"]}
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
                **_framing(shot.get("framing") or {}),
                "address": _counted(shot.get("address"), ("share", "n")),
                "camera": {k: camera[k] for k in ("movement", "direction")} if camera else None,
            }
        )

    video_visual = {key: _weighted(values) for key, values in weighted.items()}
    chroma = video_visual.get("chroma")
    warmth = video_visual.get("warmth")
    video_medium = _medium(mediums)
    subjects = _subjects(shots, lengths, cfg.subject_min)
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
            "fps": round(frames / duration, 2) if duration else None,
            "segments": len(shots),
            "medium": video_medium["label"] if video_medium else None,
            "subjects": len(subjects),
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
                "attention": relations["attention_edges"],
                "mutual": sum(1 for r in relations["relations"] if r["attention"]["mutual"]),
                "looks_at": relations["looks_at_edges"],
                "synchrony": relations["synchrony_edges"],
                "cross_segment": relations["cross_segment_relations"],
                "attention_floor": cfg.identikit_attention,
            },
            "text": {
                "readings": sum(int(t["n"]) for s in shots for t in s["text"]),
                "distinct": sum(len(s["text"]) for s in shots),
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
                **video_visual,
                "grayscale": float(chroma["mean"]) < GRAYSCALE if chroma else None,
                "temperature": _temperature(float(warmth["mean"])) if warmth else None,
                "medium": video_medium,
                "colour_histogram": _histogram(histograms),
            },
            "segments": plastic_segments,
        },
        "figurative": {
            "caption": caption,
            "segments": figurative_segments,
            "subjects": subjects,
            "persons": described,
            "tracked_only": {"count": len(tracked_only), "ids": tracked_only},
        },
        "enunciative": {
            "persons": heads,
            "relations": kept,
            "segments": enunciative_segments,
            # share of running time per camera movement
            "camera": {
                k: round(v / duration, 3)
                for k, v in sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))
            }
            if duration
            else {},
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
            "ethnicity_withheld": ethnicity_withheld,
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
