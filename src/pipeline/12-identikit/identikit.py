"""12-identikit: the terminal artifact, assembled from 03, 08, 09, 10 and 11; measures nothing."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from config import Config

STAGE = "12-identikit"
INPUTS = (
    "aggregated.json",
    "relations.json",
    "identity.json",
    "profile.json",
    "tracks.json",
    "conditional.json",
)

# 07 scores each attribute alone, so a person can rank high on both of a pair
EXCLUSIVE: tuple[tuple[str, ...], ...] = (
    ("a person with a beard", "a clean-shaven person"),
    ("a person with a moustache", "a clean-shaven person"),
    ("a person with long hair", "a person with short hair", "a bald person"),
    ("a person with curly hair", "a person with straight hair", "a bald person"),
    (
        "a person with blond hair",
        "a person with dark hair",
        "a person with grey hair",
        "a bald person",
    ),
    ("a person looking at the camera", "a person looking away"),
    ("a person looking at the camera", "a person in profile view"),
    ("a person wearing eyeglasses", "a person wearing sunglasses"),
    (
        "a person wearing a hat",
        "a person wearing a cap",
        "a person wearing a headscarf",
    ),
)
CONTRADICTS = {
    frozenset(pair) for group in EXCLUSIVE for pair in itertools.combinations(group, 2)
}

# on how many 03 saw at once, not how many 08 followed
CROWD = ((0, "none"), (1, "single"), (2, "couple"), (5, "group"))

# Shot size, fitted on AVE scenes 51-100 (2026-11-05): the median largest face height first, then where the
# frame cuts the largest followed body (07's pose landmarks), then face and body heights; with nobody
# followed, MobileCLIP's nearest size. Medium long is cinematography's medium wide.
SIZES = (
    "extreme close-up",
    "close-up",
    "medium close-up",
    "medium",
    "medium long",
    "long",
    "extreme long",
)
EXTREME_CLOSE = 0.75  # face height
CLOSE = 0.33  # face height
MEDIUM_FACE = 0.125  # face height
MEDIUM_BODY = 1.0  # body height, where neither a face nor the body cut decides
LONG = 0.25  # body height; below it extreme long
VISIBLE = 0.3  # pose landmark visibility from which a body part counts as in frame
# MediaPipe's landmarks, top to bottom: face, shoulders, elbows, hips, knees, ankles and feet
BODY_PARTS = (
    tuple(range(11)),
    (11, 12),
    (13, 14),
    (23, 24),
    (25, 26),
    (27, 28, 29, 30, 31, 32),
)
CUT = (1, 2, 3, 4, 4, 5)  # the lowest part in frame -> index into SIZES
CLIP_MODEL = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"  # 02's keyframe embedder
SIZE_PROMPTS = tuple(  # as fitted; wide is long here
    (f"a film still: {n} shot", f"a {n} shot of a scene")
    for n in (
        "extreme close-up",
        "close-up",
        "medium close-up",
        "medium",
        "medium wide",
        "wide",
        "extreme wide",
    )
)
GRAYSCALE = 0.01  # weighted chroma below which a segment reads grayscale
PORTRAIT = 0.05  # median face area against the rest of the frame, for a portrait
WARM = 5.0  # mean b* above which a segment reads warm, below its negative cool
OUTDOOR = 0.168  # mean Places365 outdoor probability from which a segment reads outdoor; the lowest that keeps 95% of AVE scenes 51-100's interiors indoor
# The values a person's record can show; head pose and gaze are under enunciative.
MEASURED = (
    "age",
    "gender",
    "perceived_ethnicity",
    "emotion",
    "valence",
    "arousal",
    "movement",
)
# On the median samples behind a person's values; first match wins, else "low".
CONFIDENCE = ((10, "good"), (3, "moderate"))
ANIMALS = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
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
    "enunciative.persons[].address": "07 MediaPipe head pose and Gaze-LLE, via 09",
    "enunciative.segments[].address": "07 MediaPipe head pose and Gaze-LLE, via 09",
    "enunciative.segments[].camera": "09 LK flow on corners off 06's person boxes, a RANSAC similarity per frame pair",
    "figurative.segments[].indoor_outdoor": "05 Places365 ResNet-18 with Places365's IO list, via 09",
    "figurative.segments[].tags": "05 MobileCLIP against RAM's tag list, via 09",
    "figurative.segments[].place": "05 Places365 ResNet-18, via 09",
    "figurative.segments[].objects": "06 YOLO11 temporal union, via 09, against --object-min and --object-frames",
    "figurative.segments[].text": "07 OpenCV CRNN, via 09, near-identical readings grouped; persons bound by 10",
    "figurative.video.tags": "05 MobileCLIP against RAM's tag list, via 09, duration-weighted",
    "figurative.video.place": "05 Places365 ResNet-18, via 09, duration-weighted",
    "figurative.video.indoor_outdoor": "05 Places365 with Places365's IO list, via 09, duration-weighted",
    "figurative.video.objects": "06 YOLO11 temporal union, via 09; the classes listed in some segment",
    "figurative.segments[].crowdedness": "03 YOLO11 person count over keyframes",
    "figurative.persons[].links": "08 ArcFace and YouTu ReID cosine; same_face: one face box under two body boxes",
    "figurative.persons[].age": "07 FairFace ResNet-34, 9 age brackets, on face crops clearing 08's floors, via 09",
    "figurative.persons[].gender": "07 FairFace ResNet-34 on face crops clearing 08's floors, via 09",
    "figurative.persons[].attributes": "07 MobileCLIP zero-shot, via 09",
    "figurative.persons[].emotion": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].valence": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].arousal": "07 HSEmotion EfficientNet-B0, via 09",
    "figurative.persons[].movement": "07 MediaPipe PoseLandmarker, via 09",
    "figurative.persons[].by_segment": "07 HSEmotion and PoseLandmarker samples split by segment, via 09",
    "enunciative.persons[].head_pose": "07 MediaPipe FaceLandmarker, via 09",
    "enunciative.relations": "10, from 06 boxes and 07 gaze and affect",
    "enunciative.relations[].looks_at": "10, from 07 Gaze-LLE's sight line and point to heads from 06 boxes",
    "enunciative.persons[].gaze": "07 Gaze-LLE with DINOv2 ViT-B on the frame and head box, via 09",
    "enunciative.segments[].visible_at_once": "03 YOLO11 person count over keyframes",
    "enunciative.segments[].followed": "08 person ids, via 09",
    "enunciative.segments[].camera_distance": (
        "06 YuNet face and YOLO11 body boxes via 09, and 07 MediaPipe pose landmarks; with nobody followed, "
        "02's MobileCLIP keyframe embeddings against SIZE_PROMPTS"
    ),
    "enunciative.segments[].face_background_ratio": "06 YuNet face boxes, via 09",
    "enunciative.segments[].portrait_scene": "face_background_ratio against PORTRAIT",
    "plastic.segments[].keyframes": "05 OpenCV measures per keyframe, via 09",
    "figurative.segments[].keyframes": "07 HSEmotion samples of everyone on screen and 06 tracks, per keyframe's part of "
    "the segment, via 09",
    "enunciative.segments[].keyframes": "camera_distance's and camera's rules over each keyframe's part of the segment",
}

CAVEATS: tuple[str, ...] = (
    "gaze comes from an enlarged YuNet face box and the scene: faces YuNet misses have none, and "
    "the model was trained on real video, not animation or CGI",
    "placement 'nearer' is a box-area proxy, not depth",
    "place comes from Places365 and is weak on broadcast and animated footage; tags are steadier",
    "depth is relative disparity within each keyframe, 1 nearest: an order and ratios, not distances",
    "centralities and depth are per label: panoptic segments are not matched across keyframes",
    "gender and age are FairFace's perceived estimate, age as one of 9 brackets, not a property of "
    "a person; they are read on face crops that clear 08's floors",
    "text is read by a CRNN in ten-character chunks, so word boundaries are not real",
    "visible_at_once is the maximum over a segment's keyframes, so it is itself a lower bound",
    "camera_distance is a shot size from the largest followed person, not a distance, its cut-offs fitted on AVE's "
    "films; faces YuNet misses read as wider shots, and with nobody followed it is MobileCLIP's nearest size",
    "a person's confidence is banded on the median samples behind their values, good from 10, "
    "moderate from 3; a glimpsed person's every value rests on one sample",
    "a relation is listed when, in some segment, --relation-share of one person's looks on "
    "--relation-min samples or more land on the other's head (of the heads within 20 degrees of the "
    "line of sight, the one nearest Gaze-LLE's point)",
    "the caption is written by a VLM; `unsupported` lists objects it names that no detector found",
    "perceived_ethnicity is FairFace's reading against its own 7 groups, not a property of a person; "
    "it is read on live-action faces only, and photoreal CGI passes that gate",
    "medium separates 2D animation well; photoreal CGI reads as live action",
    "address is the head turned to the lens with the gaze off the frame, not a measured eye contact",
    "camera movement is one similarity per frame pair fitted off detected people, so an animal or "
    "object filling the frame can still read as a move; its thresholds were tuned on AVE's films, and "
    "a smooth broadcast or drawn camera can still read as a handheld rig",
    "temperature is the keyframes' mean colour, so grading counts as much as light",
    "subjects are animal classes the detector saw often enough, neither tracked nor told apart",
    "a keyframe's emotion, valence and arousal pool everyone on screen in its part of the segment, not one person",
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
    """09's grouped readings, with the people 10 found any variant of each overlapping."""
    persons: dict[str, set[int]] = {}
    for entry in bound:
        persons.setdefault(str(entry["text"]), set()).update(
            int(p) for p in entry.get("persons", [])
        )
    out: list[dict[str, Any]] = []
    for reading in readings[:keep]:
        touching: set[int] = set()
        for variant in reading["variants"]:
            touching |= persons.get(str(variant), set())
        out.append(
            {
                **{k: reading[k] for k in ("text", "n", "first_frame", "last_frame")},
                "persons": sorted(touching),
            }
        )
    return out


def _peak(side: dict[str, Any], minimum: int) -> dict[str, Any] | None:
    """A direction's best segment, on `minimum` samples or more."""
    found = [s for s in side.get("by_segment", []) if int(s["n"]) >= minimum]
    if not found:
        return None
    best = max(found, key=lambda s: (s["hits"] / s["n"], s["n"]))
    return {
        "segment": best["segment"],
        "share": round(best["hits"] / best["n"], 3),
        "n": best["n"],
    }


def _evidenced(relation: dict[str, Any], floor: float, minimum: int) -> bool:
    """Listed when one segment's looks reach `floor`, or affect moves together."""
    looks = relation["looks_at"]
    peaks = [_peak(looks[side], minimum) for side in ("a_to_b", "b_to_a")]
    return bool(relation["synchrony"]["available"]) or any(
        p is not None and p["share"] >= floor for p in peaks
    )


def _side(found: dict[str, Any], minimum: int) -> dict[str, Any] | None:
    """One direction's share, samples and best segment; null where unmeasured."""
    if not found.get("available"):
        return None
    return {"share": found["share"], "n": found["n"], "peak": _peak(found, minimum)}


def _relation(relation: dict[str, Any], minimum: int) -> dict[str, Any]:
    synchrony = relation["synchrony"]
    placement = relation["placement"]
    looks = relation["looks_at"]
    return {
        "pair": [relation["a"], relation["b"]],
        "co_present": {
            "frames": relation["co_present"]["n"],
            "segments": relation["co_present"]["segments"],
        },
        "placement": {
            k: placement[k]
            for k in ("horizontal", "vertical", "nearer", "size_ratio", "n")
        },
        "looks_at": {
            "a_to_b": _side(looks["a_to_b"], minimum),
            "b_to_a": _side(looks["b_to_a"], minimum),
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


def _counted(
    found: dict[str, Any] | None, keys: tuple[str, ...]
) -> dict[str, Any] | None:
    """The named keys of a 09 block, or null when it rests on no samples."""
    if not found or not found.get("n"):
        return None
    return {k: found[k] for k in keys}


def _shares(seconds: dict[str, float], duration: float) -> dict[str, float]:
    ranked = sorted(seconds.items(), key=lambda kv: (-kv[1], kv[0]))
    return {k: round(v / duration, 3) for k, v in ranked}


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
                entry = found.setdefault(
                    o["label"], {"label": o["label"], "segments": [], "seconds": 0.0}
                )
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
    emotion = person.get("emotion") or {}
    attributes, dropped = _attributes(person.get("attributes") or [], keep)
    record = {
        "person_id": person["person_id"],
        "segments": person["segments"],
        "tracks": person["tracks"],
        "seconds": person["duration_seconds"],
        "linked": person["linked"],
        "abstained": person["abstained"],
        "links": links,
        "age": _counted(
            demographics.get("age"), ("modal", "agreement", "distribution", "n")
        ),
        "gender": _counted(
            demographics.get("gender"), ("modal", "agreement", "distribution", "n")
        ),
        "perceived_ethnicity": _counted(
            person.get("perceived_ethnicity"),
            ("taxonomy", "modal", "agreement", "distribution", "n"),
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
    }
    return record, dropped


def _samples(
    record: dict[str, Any],
    head_pose: dict[str, Any] | None,
    gaze: dict[str, Any] | None,
) -> list[int]:
    """The samples behind each value a person's record shows; empty when it shows none."""
    counts = [int(record[k]["n"]) for k in MEASURED if record[k]]
    if record["attributes"]:
        counts.append(max(int(a["n"]) for a in record["attributes"]))
    if head_pose and head_pose.get("yaw"):
        counts.append(int(head_pose["yaw"]["n"]))
    if gaze:
        counts.append(int(gaze["n"]))
    return counts


def _confidence(counts: list[int]) -> str:
    middle = statistics.median(counts)
    return next((label for floor, label in CONFIDENCE if middle >= floor), "low")


def _head_pose(person: dict[str, Any]) -> dict[str, Any] | None:
    pose = person.get("head_pose")
    if not pose:
        return None
    return {axis: _measure(pose.get(axis)) for axis in ("yaw", "pitch", "roll")}


def _histogram(
    histograms: list[tuple[dict[str, list[float]], float]],
) -> dict[str, Any] | None:
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
                "ratio": round(float(t["disparity"]) / background, 3)
                if background
                else None,
            }
            for t in things
        ],
        "order": [
            t["label"]
            for t in sorted(things, key=lambda t: (-float(t["disparity"]), t["label"]))
        ],
    }


def _io(outdoor: float) -> dict[str, Any]:
    return {"label": "outdoor" if outdoor >= OUTDOOR else "indoor", "outdoor": outdoor}


def _indoor_outdoor(plastic: dict[str, Any]) -> dict[str, Any] | None:
    stats = plastic.get("outdoor") or {}
    if not stats.get("n"):
        return None
    return {**_io(float(stats["mean"])), "n": stats["n"]}


def _parts(
    points: list[list[float]], box: list[int], size: tuple[int, int]
) -> list[float]:
    """Per body part, the best visibility among its landmarks inside the frame; 07's crop is the body box."""
    width, height = size
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(width, box[2]), min(height, box[3])
    return [
        round(
            float(
                max(
                    (
                        points[i][2]
                        for i in part
                        if 0 <= x1 + points[i][0] * (x2 - x1) < width
                        and 0 <= y1 + points[i][1] * (y2 - y1) < height
                    ),
                    default=0.0,
                )
            ),
            3,
        )
        for part in BODY_PARTS
    ]


def _cut(samples: list[list[float]]) -> int | None:
    """The lowest body part in frame, the lower median over the person's samples."""
    levels = sorted(
        next((g for g in range(len(BODY_PARTS) - 1, -1, -1) if s[g] >= VISIBLE), -1)
        for s in samples
    )
    levels = [x for x in levels if x >= 0]
    return levels[(len(levels) - 1) // 2] if levels else None


def _size(face: float, body: float, samples: list[list[float]]) -> tuple[int, str]:
    """An index into SIZES, and the measure that decided it."""
    if face >= EXTREME_CLOSE:
        return 0, "face"
    if face >= CLOSE:
        return 1, "face"
    cut = _cut(samples)
    if cut is not None:
        found = CUT[cut]
        return (6 if found == 5 and body < LONG else found), "body cut"
    if face >= MEDIUM_FACE:
        return 3, "face"
    if body >= MEDIUM_BODY:
        return 3, "body"
    return (5 if body >= LONG else 6), "body"


def _body_samples(
    tracks: dict[str, Any], conditional: dict[str, Any], size: tuple[int, int]
) -> dict[int, list[tuple[int, list[float]]]]:
    """Per segment, the body-part visibilities of its largest followed person, frame by frame, over 07's pose samples."""
    poses = {
        (int(s["index"]), int(t["track_id"])): [
            r for r in t.get("series", []) if r.get("pose")
        ]
        for s in conditional["shots"]
        for t in s.get("tracks", [])
    }
    out: dict[int, list[tuple[int, list[float]]]] = {}
    for shot in tracks["shots"]:
        index = int(shot["index"])
        bodies = {
            int(t["track_id"]): {
                int(r["frame"]): r["body"] for r in t["frames"] if not r["interpolated"]
            }
            for t in shot.get("tracks", [])
        }
        bodies = {k: v for k, v in bodies.items() if v}
        if not bodies:
            continue
        main = max(
            bodies,
            key=lambda t: statistics.median(
                (b[3] - b[1]) / size[1] for b in bodies[t].values()
            ),
        )
        out[index] = [
            (
                int(r["frame"]),
                _parts(r["pose"]["keypoints"], bodies[main][int(r["frame"])], size),
            )
            for r in poses.get((index, main), [])
            if int(r["frame"]) in bodies[main]
        ]
    return out


def _size_texts(model_dir: Path) -> np.ndarray[Any, np.dtype[Any]]:
    """MobileCLIP's text vector per size, two prompts averaged; encoded once and kept in model_dir."""
    path = model_dir / "shot_size_texts.npy"
    if path.exists():
        cached: np.ndarray[Any, np.dtype[Any]] = np.load(path)
        return cached
    import open_clip
    import torch

    from src.tools import download_models

    cache = str(download_models.stage_dir("02-segmentation"))
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, cache_dir=cache)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL, cache_dir=cache)
    with torch.no_grad():
        text = (
            model.eval()
            .encode_text(tokenizer([p for pair in SIZE_PROMPTS for p in pair]))
            .float()
        )
    text = (text / text.norm(dim=-1, keepdim=True)).numpy()
    centres = text.reshape(len(SIZES), 2, -1).mean(axis=1)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, centres)
    return centres


def _nearest_sizes(cfg: Config, wanted: set[int]) -> dict[int, int]:
    """For segments with nobody followed, MobileCLIP's nearest size over 02's keyframe embeddings."""
    if not wanted:
        return {}
    segmentation = _read(cfg.json_dir, "segmentation.json", "02-segmentation")
    embeddings = np.load(cfg.embeddings_dir / "keyframe_embeddings.npy")
    texts = _size_texts(cfg.model_dir)
    out: dict[int, int] = {}
    row = 0
    for shot in segmentation["shots"]:
        count = len(shot["keyframes"])
        if int(shot["index"]) in wanted and count:
            vector = embeddings[row : row + count].mean(axis=0)
            out[int(shot["index"])] = int(
                np.argmax(texts @ (vector / np.linalg.norm(vector)))
            )
        row += count
    return out


def _framing(
    framing: dict[str, Any], samples: list[list[float]], nearest: int | None
) -> dict[str, Any]:
    """With nobody followed, MobileCLIP's size and no face ratio: there is no one to frame."""
    if not framing.get("frames"):
        return {
            "camera_distance": None
            if nearest is None
            else {"scale": SIZES[nearest], "from": "MobileCLIP"},
            "face_background_ratio": None,
            "portrait_scene": None,
        }
    medians = {k: float(framing[k]["median"]) for k in ("face_height", "body_height")}
    found, source = _size(medians["face_height"], medians["body_height"], samples)
    ratio = float(framing["face_background"]["median"])
    return {
        "camera_distance": {
            "scale": SIZES[found],
            "from": source,
            **medians,
            "frames": framing["frames"],
        },
        "face_background_ratio": ratio,
        "portrait_scene": "portrait" if ratio >= PORTRAIT else "scene",
    }


def _keyframes(shot: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """One level's share of a segment's per-keyframe entries."""
    return [{k: e[k] for k in ("start", "end", *keys)} for e in shot["keyframes"]]


def _keyframe_size(
    keyframe: dict[str, Any],
    samples: list[tuple[int, list[float]]],
    segment: dict[str, Any] | None,
) -> str | None:
    """The shot size over one keyframe's part of a segment; with nobody seen there, the segment's."""
    seen = [
        parts
        for frame, parts in samples
        if keyframe["start_frame"] <= frame <= keyframe["end_frame"]
    ]
    found = _framing(keyframe["framing"], seen, None)["camera_distance"] or segment
    return found["scale"] if found else None


def _caption(
    json_dir: Path, shots: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    """11 runs a VLM, so a missing or stale caption is a gap, not an error."""
    path = json_dir / "caption.json"
    if not path.exists():
        return None, "11-caption has not run"
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if meta["segments"] != [
        [int(s["start_frame"]), int(s["end_frame"])] for s in shots
    ]:
        return None, "caption.json was written for other segments; re-run 11-caption"
    return meta, ""


def run(cfg: Config) -> dict[str, Any]:
    aggregated = _read(cfg.json_dir, "aggregated.json", "09-aggregation")
    relations = _read(cfg.json_dir, "relations.json", "10-relations")
    identity = _read(cfg.json_dir, "identity.json", "08-identity")
    profile = _read(cfg.json_dir, "profile.json", "03-profiler")
    if "whole_video" not in aggregated:
        raise RuntimeError(
            "aggregated.json predates whole_video; re-run 09-aggregation"
        )
    if any("keyframes" not in s for s in aggregated["shots"]):
        raise RuntimeError(
            "aggregated.json predates the keyframe lists; re-run 09-aggregation"
        )
    samples = _body_samples(
        _read(cfg.json_dir, "tracks.json", "06-detection-tracking"),
        _read(cfg.json_dir, "conditional.json", "07-conditional-experts"),
        tuple(aggregated["frame_size"]),
    )
    nearest = _nearest_sizes(
        cfg,
        {
            int(s["index"])
            for s in aggregated["shots"]
            if not (s.get("framing") or {}).get("frames")
        },
    )

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
            f"11 {caption_meta['model']} on 02 keyframes, with hints from 02, 03, 05, 06, 07, 09 and 10"
        )
    shots = aggregated["shots"]
    at_once = {
        int(s["index"]): int(s["counts"]["person_max"]) for s in profile["shots"]
    }
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

    # Described: some value rests on two samples or more. Glimpsed: values, each from one sample.
    shown: list[dict[str, Any]] = []
    heads: list[dict[str, Any]] = []
    tracked_only: list[int] = []
    conflicts: list[dict[str, Any]] = []
    demographics_withheld: list[dict[str, Any]] = []
    ethnicity_withheld: list[dict[str, Any]] = []
    for person in aggregated["persons"]:
        pid = int(person["person_id"])
        record, dropped = _person(person, links_of.get(pid, []), keep)
        head_pose, gaze = _head_pose(person), _gaze(person)
        counts = _samples(record, head_pose, gaze)
        if not counts:
            tracked_only.append(pid)
            continue
        record["status"] = "described" if max(counts) > 1 else "glimpsed"
        record["confidence"] = _confidence(counts)
        shown.append(record)
        if head_pose is not None or gaze is not None:
            heads.append(
                {
                    "person_id": pid,
                    "head_pose": head_pose,
                    "gaze": gaze,
                    "address": _counted(person.get("address"), ("share", "n")),
                }
            )
        if dropped:
            conflicts.append({"person_id": pid, "dropped": dropped})
        demographics = person.get("demographics")
        if demographics and not demographics["gender"]["n"]:
            demographics_withheld.append({"person_id": pid, **demographics["withheld"]})
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
    turns: dict[str, float] = {}
    listed_objects: set[str] = set()
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
        if camera.get("move"):
            turns[camera["move"]] = turns.get(camera["move"], 0.0) + seconds
        coverage = ((plastic.get("semantic") or {}).get("coverage")) or {}
        plastic_segments.append(
            {
                "segment": index,
                "medium": _counted(medium, ("modal", "agreement", "n")),
                "temperature": _temperature(visual.get("warmth")),
                "visual": visual,
                "coverage": {name: c["mean"] for name, c in coverage.items()},
                "palette": [
                    {
                        "hex": "#{:02x}{:02x}{:02x}".format(*c["rgb"]),
                        "share": c["share"],
                    }
                    for c in plastic.get("palette") or []
                ],
                "centralities": _centralities(plastic),
                "depth": _depth(plastic),
                "keyframes": _keyframes(shot, ("brightness", "warmth")),
            }
        )

        # A class seen in a frame or two is mostly the detector's noise: counted, not listed.
        objects = shot.get("objects", [])
        listed = [
            o
            for o in objects
            if float(o["share"]) >= cfg.object_min
            and int(o["frame_count"]) >= cfg.object_frames
        ]
        listed_objects |= {str(o["label"]) for o in listed}
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
                    {
                        "label": o["label"],
                        "frames": o["frame_count"],
                        "share": o["share"],
                    }
                    for o in listed
                ],
                "objects_withheld": len(objects) - len(listed),
                "text": _text(shot["text"], bound_text.get(index, []), keep),
                "text_readings": len(shot["text"]),
                "crowdedness": _crowd(at_once.get(index, 0)),
                "keyframes": _keyframes(
                    shot, ("emotion", "valence", "arousal", "people")
                ),
            }
        )

        poses = samples.get(index, [])
        framing = _framing(
            shot.get("framing") or {}, [p for _, p in poses], nearest.get(index)
        )
        enunciative_segments.append(
            {
                "segment": index,
                "visible_at_once": at_once.get(index, 0),
                "followed": followed,
                "assessment": _assess(at_once.get(index, 0), followed),
                **framing,
                "address": _counted(shot.get("address"), ("share", "n")),
                "camera": {
                    k: camera.get(k) for k in ("movement", "rig", "move", "direction")
                }
                if camera
                else None,
                "keyframes": [
                    {
                        "start": k["start"],
                        "end": k["end"],
                        "shot_size": _keyframe_size(
                            k, poses, framing["camera_distance"]
                        ),
                        "camera": k["camera"],
                    }
                    for k in shot["keyframes"]
                ],
            }
        )

    video_visual = {key: _weighted(values) for key, values in weighted.items()}
    chroma = video_visual.get("chroma")
    warmth = video_visual.get("warmth")
    video_medium = _medium(mediums)
    subjects = _subjects(shots, lengths, cfg.subject_min)
    kept = [
        _relation(r, cfg.relation_min)
        for r in relations["relations"]
        if _evidenced(r, cfg.relation_share, cfg.relation_min)
    ]
    persons = aggregated["persons"]
    whole = aggregated["whole_video"]
    video_objects = [o for o in whole["objects"] if o["label"] in listed_objects]

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
            "fps": round(frames / duration, 2),
            "segments": len(shots),
            "medium": video_medium["label"] if video_medium else None,
            "subjects": len(subjects),
            "persons": {
                "resolved": len(persons),
                "described": sum(1 for r in shown if r["status"] == "described"),
                "glimpsed": sum(1 for r in shown if r["status"] == "glimpsed"),
                "tracked_only": len(tracked_only),
                "abstained": sum(1 for p in persons if p["abstained"]),
                "thin": aggregated["thin_records"],
            },
            "relations": {
                "co_present": relations["relation_count"],
                "with_evidence": len(kept),
                "mutual": sum(
                    1 for r in relations["relations"] if r["looks_at"]["mutual"]
                ),
                "looks_at": relations["looks_at_edges"],
                "synchrony": relations["synchrony_edges"],
                "cross_segment": relations["cross_segment_relations"],
                "share_floor": cfg.relation_share,
                "relation_min": cfg.relation_min,
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
                "shots_per_minute": round(60 * len(shots) / duration, 2),
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
            "video": {
                "tags": whole["tags"],
                "place": whole["place"],
                "indoor_outdoor": _io(float(whole["outdoor"]))
                if whole["outdoor"] is not None
                else None,
                "objects": video_objects,
                "objects_withheld": len(whole["objects"]) - len(video_objects),
            },
            "segments": figurative_segments,
            "subjects": subjects,
            "persons": shown,
            "tracked_only": {"count": len(tracked_only), "ids": tracked_only},
        },
        "enunciative": {
            "persons": heads,
            "relations": kept,
            "segments": enunciative_segments,
            # shares of running time: one label per segment, and the move whatever the rig
            "camera": _shares(moves, duration),
            "camera_moves": _shares(turns, duration),
        },
        "provenance": provenance,
        "gaps": {
            "pending": pending,
            "excluded": EXCLUDED,
            "abstained": [int(p["person_id"]) for p in persons if p["abstained"]],
            "thin": [
                int(p["person_id"]) for p in persons if int(p["support"]["series"]) < 2
            ],
            "glimpsed": [
                int(r["person_id"]) for r in shown if r["status"] == "glimpsed"
            ],
            "segments_without_conditional_branch": [
                int(s["index"]) for s in shots if not s["tracked"]
            ],
            "under_tracked_segments": [
                e["segment"]
                for e in enunciative_segments
                if e["assessment"] == "under_tracked"
            ],
            "attribute_conflicts": conflicts,
            "demographics_withheld": demographics_withheld,
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
        "reduction": round(1 - size / inputs, 3),
    }

    # compact: indentation doubled the file
    (cfg.out_root / "identikit.json").write_text(
        json.dumps(record, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "video": record["video"],
        "segments": len(shots),
        **{f"persons_{k}": v for k, v in record["summary"]["persons"].items()},
        "relations_listed": len(kept),
        "budget": record["budget"],
    }
