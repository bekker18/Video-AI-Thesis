"""09-aggregation: per-person and per-segment records from raw samples; every statistic carries its n."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, TypeAlias

import cv2
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
    "chroma",
    "focus",
    "edge_density",
    "horizontal_symmetry",
)
PALETTE = 6  # colours kept per segment
IN_FRAME = 0.5  # Gaze-LLE probability at which a look counts as inside the frame
LIVE = "live action"  # the only medium perceived ethnicity is read in
TAXONOMY = "FairFace-7"


def _read(json_dir: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = json_dir / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _stats(values: list[float]) -> dict[str, Any]:
    """Mean and variance, plus median and MAD for robustness; variance is null below two samples."""
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
    """Mean probability vector, its modal label, and the share of samples that agreed with it."""
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
    """Label lists ranked by how many sources offered each label, then by mean score."""
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
    """Crop-normalised movement between consecutive samples, never across a cut."""
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


def _address(samples: list[dict[str, Any]], cone: float) -> dict[str, Any]:
    """Share of samples with the head within `cone` of the camera axis and the gaze off the frame."""
    looks = [s for s in samples if "head_pose" in s and "gaze" in s]
    if not looks:
        return {"n": 0}
    hits = sum(
        1
        for s in looks
        if abs(float(s["head_pose"]["yaw"])) <= cone
        and abs(float(s["head_pose"]["pitch"])) <= cone
        and float(s["gaze"]["in_frame"]) < IN_FRAME
    )
    return {"n": len(looks), "share": round(hits / len(looks), 3)}


def _top(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _in_bracket(age: float, bracket: str) -> bool:
    low, _, high = bracket.rstrip("+").partition("-")
    return float(low) <= age < (float(high) + 1 if high else float("inf"))


def _ethnicity(
    entries: list[dict[str, Any]],
    crops: dict[int, tuple[int, dict[str, Any]]],
    live: set[int],
    cfg: Config,
) -> dict[str, Any]:
    """FairFace's perceived groups, from face crops that clear 08's floors in live-action segments."""
    kept: list[dict[str, float]] = []
    footage = quality = 0
    for entry in entries:
        segment, crop = crops[int(entry["frame"])]
        if segment not in live:
            footage += 1
        elif float(crop["area"]) < cfg.identity_min_area or float(
            crop["frontality"] or 0.0
        ) < cfg.identity_min_front:
            quality += 1
        else:
            kept.append(entry["ethnicity"])
    found = _distribution(kept, tuple(kept[0])) if kept else {"n": 0}
    return {
        "taxonomy": TAXONOMY,
        **found,
        "withheld": {"not_live_action": footage, "low_quality": quality},
    }


def _cross_check(
    gender_age: list[dict[str, Any]], fairface: list[dict[str, Any]]
) -> dict[str, Any]:
    """InsightFace genderage against FairFace's gender and age heads, crop by crop."""
    theirs = {int(f["frame"]): f for f in fairface}
    pairs = [(g, theirs[int(g["frame"])]) for g in gender_age if int(g["frame"]) in theirs]
    if not pairs:
        return {"n": 0}
    gender = sum(1 for g, f in pairs if g["gender"] == _top(f["gender"]))
    age = sum(1 for g, f in pairs if _in_bracket(float(g["age"]), _top(f["age"])))
    return {
        "n": len(pairs),
        "gender_agreement": round(gender / len(pairs), 3),
        "age_in_bracket": round(age / len(pairs), 3),
    }


def _affect(emotions: list[dict[str, Any]]) -> dict[str, Any]:
    if not emotions:
        return {}
    labels = tuple(sorted(emotions[0]["scores"]))
    out: dict[str, Any] = {"emotion": _distribution([e["scores"] for e in emotions], labels)}
    for axis in ("valence", "arousal"):
        present = [float(e[axis]) for e in emotions if e.get(axis) is not None]
        if present:
            out[axis] = _stats(present)
    return out


def _by_segment(
    series: list[dict[str, Any]],
    spans: list[tuple[int, dict[str, Any]]],
    fps: float,
    cfg: Config,
) -> list[dict[str, Any]]:
    """The person's affect and movement per segment, so a change across a cut is not pooled away."""
    out: list[dict[str, Any]] = []
    for segment in sorted({seg for seg, _ in spans}):
        frames = sum(int(s["frame_count"]) for seg, s in spans if seg == segment)
        mine = [s for s in series if int(s["segment"]) == segment]
        entry: dict[str, Any] = {
            "segment": segment,
            "duration_seconds": round(frames / fps, 3) if fps else 0.0,
            "series": len(mine),
            **_affect([s["emotion"] for s in mine if "emotion" in s]),
        }
        movement = _pose_summary(mine, cfg.pose_vis).get("movement", {"n": 0})
        if movement["n"]:
            entry["movement"] = movement
        out.append(entry)
    return out


def _person(
    person: dict[str, Any],
    series: list[dict[str, Any]],
    spans: list[tuple[int, dict[str, Any]]],
    attrs: dict[str, list[Any]],
    crops: dict[int, tuple[int, dict[str, Any]]],
    live: set[int],
    fps: float,
    cfg: Config,
) -> dict[str, Any]:
    face = [s for s in series if "head_pose" in s]
    emotions = [s["emotion"] for s in series if "emotion" in s]
    # Summed, not last minus first: the time between a person's segments is not screen time.
    frames = sum(int(s["frame_count"]) for _, s in spans)

    record: dict[str, Any] = {
        "person_id": person["person_id"],
        "tracks": person["tracks"],
        "segments": person["segments"],
        "linked": person["linked"],
        "abstained": person["abstained"],
        "first_frame": min(int(s["first_frame"]) for _, s in spans) if spans else None,
        "last_frame": max(int(s["last_frame"]) for _, s in spans) if spans else None,
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
            "observed_frames": sum(int(s["observed_frames"]) for _, s in spans),
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

    record.update(_affect(emotions))
    gazes = [s["gaze"] for s in series if "gaze" in s]
    if gazes:
        inside = [float(g["in_frame"]) for g in gazes]
        record["gaze"] = {
            "n": len(gazes),
            "in_frame": _stats(inside),
            "in_frame_share": round(sum(1 for p in inside if p >= IN_FRAME) / len(inside), 3),
        }
    address = _address(series, cfg.address_deg)
    if address["n"]:
        record["address"] = address

    pose = _pose_summary(series, cfg.pose_vis)
    if pose["n"]:
        record["pose"] = pose
    record["by_segment"] = _by_segment(series, spans, fps, cfg)

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
            "cross_check": _cross_check(attrs["gender_age"], attrs.get("fairface", [])),
        }
    if attrs.get("fairface"):
        record["perceived_ethnicity"] = _ethnicity(attrs["fairface"], crops, live, cfg)
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
    # 8-bit Lab centres a*, b* on 128; warmth is b* (+yellow), tint a* (+magenta)
    lab = [k["classical"]["mean_lab"] for k in keyframes]
    out["visual"]["warmth"] = _stats([float(v[2]) - 128.0 for v in lab])
    out["visual"]["tint"] = _stats([float(v[1]) - 128.0 for v in lab])
    mediums = [k["medium"] for k in keyframes if k.get("medium")]
    if mediums:
        out["medium"] = _distribution(mediums, tuple(mediums[0]))

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
    outdoor = [float(k["outdoor"]) for k in keyframes if "outdoor" in k]
    if outdoor:
        out["outdoor"] = _stats(outdoor)
    out.update(_colour(keyframes))
    out.update(_things(keyframes, cfg.agg_top))
    return out


def _colour(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean histogram, and 05's colour cells merged across keyframes by share."""
    hists = [k["classical"]["histogram"] for k in keyframes if "histogram" in k["classical"]]
    cells: dict[int, list[tuple[float, list[int]]]] = {}
    for keyframe in keyframes:
        for entry in keyframe["classical"].get("palette", []):
            cells.setdefault(int(entry["cell"]), []).append((float(entry["share"]), entry["rgb"]))
    palette: list[dict[str, Any]] = []
    for parts in cells.values():
        weight = sum(share for share, _ in parts)
        rgb = [round(sum(share * c[i] for share, c in parts) / weight) for i in range(3)]
        palette.append({"rgb": rgb, "share": round(weight / len(keyframes), 4)})
    palette.sort(key=lambda p: (-float(p["share"]), p["rgb"]))
    out: dict[str, Any] = {"palette": palette[:PALETTE]}
    if hists:
        out["histogram"] = {
            c: [round(float(v), 4) for v in np.mean([h[c] for h in hists], axis=0)]
            for c in ("r", "g", "b")
        }
    return out


def _things(keyframes: list[dict[str, Any]], top: int) -> dict[str, Any]:
    """05's panoptic objects per label; unmatched across keyframes, so a label pools its instances."""
    found: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for keyframe in keyframes:
        width, height = keyframe["size"]
        for segment in keyframe["panoptic"]["segments"]:
            if not segment.get("thing"):
                continue
            x1, y1, x2, y2 = segment["bbox"]
            x, y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            centrality = 1.0 - float(np.hypot(x - 0.5, y - 0.5) / np.hypot(0.5, 0.5))
            found.setdefault(str(segment["label"]), []).append(
                (x, y, centrality, float(segment["area"]), float(segment["disparity"]))
            )
    things: list[dict[str, Any]] = []
    for label, rows in found.items():
        arr = np.asarray(rows)
        mean = arr.mean(axis=0)
        spread = (
            round(float(np.sqrt(arr[:, 0].var(ddof=1) + arr[:, 1].var(ddof=1))), 4)
            if len(rows) >= 2
            else None
        )
        things.append(
            {
                "label": label,
                "n": len(rows),
                **{k: round(float(v), 4) for k, v in zip(("x", "y", "centrality", "area", "disparity"), mean)},
                "spread": spread,
            }
        )
    things.sort(key=lambda t: (-float(t["area"]), str(t["label"])))
    backgrounds = [
        float(k["panoptic"]["background_disparity"])
        for k in keyframes
        if k["panoptic"].get("background_disparity") is not None
    ]
    return {"things": things[:top], "background_disparity": _stats(backgrounds)}


def _text(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Readings deduplicated per shot, each with the frame range it was seen over."""
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


def _frame_size(cfg: Config, tracks_meta: dict[str, Any]) -> tuple[int, int]:
    """The pixels 06's boxes are in: 01's frames if 06 read those, else one decoded frame."""
    if tracks_meta.get("frame_source") == "frames":
        sampling = _read(cfg.json_dir, "sampling.json", "01-sampling")
        return int(sampling["frame_width"]), int(sampling["frame_height"])
    capture = cv2.VideoCapture(str(cfg.video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read a frame from {cfg.video}")
    return int(frame.shape[1]), int(frame.shape[0])


def _framing(shot: dict[str, Any], size: tuple[int, int]) -> dict[str, Any]:
    """Per observed frame: largest face and body as a share of frame height, faces against the rest."""
    width, height = size
    rows_at: dict[int, list[dict[str, Any]]] = {}
    for track in shot.get("tracks", []):
        for row in track["frames"]:
            if not row["interpolated"]:
                rows_at.setdefault(int(row["frame"]), []).append(row)

    def tall(box: list[int]) -> float:
        return max(0, min(height, box[3]) - max(0, box[1])) / height

    def area(box: list[int]) -> int:
        return max(0, min(width, box[2]) - max(0, box[0])) * max(0, min(height, box[3]) - max(0, box[1]))

    face_height: list[float] = []
    body_height: list[float] = []
    face_background: list[float] = []
    for frame in sorted(rows_at):
        rows = rows_at[frame]
        faces = [r["face"] for r in rows if r["face"]]
        covered = sum(area(f) for f in faces)
        face_height.append(max((tall(f) for f in faces), default=0.0))
        body_height.append(max(tall(r["body"]) for r in rows))
        face_background.append(covered / max(1, width * height - covered))
    return {
        "frames": len(rows_at),
        "face_frames": sum(1 for h in face_height if h),
        "face_height": _stats(face_height),
        "body_height": _stats(body_height),
        "face_background": _stats(face_background),
    }


def run(cfg: Config) -> dict[str, Any]:
    segmentation = _read(cfg.json_dir, "segmentation.json", "02-segmentation")
    routing = _read(cfg.json_dir, "routing.json", "04-router")
    tracks_meta = _read(cfg.json_dir, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.json_dir, "conditional.json", "07-conditional-experts")
    global_meta = _read(cfg.json_dir, "global.json", "05-global-experts")
    identity = _read(cfg.json_dir, "identity.json", "08-identity")
    if any("size" not in k for k in global_meta["keyframes"]):
        raise RuntimeError("global.json predates 05's object depth; re-run 05-global-experts")

    fps = float(segmentation["fps"])
    size = _frame_size(cfg, tracks_meta)
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
    plastic = {int(s["index"]): _plastic(by_shot.get(int(s["index"]), []), cfg) for s in conditional["shots"]}
    live = {i for i, p in plastic.items() if (p.get("medium") or {}).get("modal") == LIVE}

    persons: list[dict[str, Any]] = []
    everyone: list[dict[str, Any]] = []
    thin = 0
    for person in identity["persons"]:
        keys = [(int(t["segment"]), int(t["track_id"])) for t in person["tracks"]]
        present = [k for k in keys if k in spans]
        if not present:
            continue

        # Samples keep their segment: movement and the per-segment breakdown need it.
        series: list[dict[str, Any]] = []
        attrs: dict[str, list[Any]] = {
            "embedding_rows": [],
            "body_rows": [],
            "gender_age": [],
            "fairface": [],
            "clip": [],
        }
        crops: dict[int, tuple[int, dict[str, Any]]] = {}  # frames are unique within a person
        for key in present:
            crops.update({int(c["frame"]): (key[0], c) for c in spans[key]["crops"]["face"]})
            source = tracks_by_key.get(key)
            if source is None:
                continue
            series += [{**sample, "segment": key[0]} for sample in source["series"]]
            for field, values in source.get("attributes", {}).items():
                attrs.setdefault(field, []).extend(values)
        series.sort(key=lambda s: (int(s["segment"]), int(s["frame"])))
        everyone += series

        record = _person(
            person, series, [(k[0], spans[k]) for k in present], attrs, crops, live, fps, cfg
        )
        if record["support"]["series"] < 2:
            thin += 1
        persons.append(record)

    by_segment: dict[int, list[int]] = {}
    for record in persons:
        for segment in record["segments"]:
            by_segment.setdefault(int(segment), []).append(int(record["person_id"]))

    cameras = {int(s["index"]): s.get("camera") for s in segmentation["shots"]}
    stride = max(1, int(tracks_meta["det_stride"]))
    shots: list[dict[str, Any]] = []
    for shot in conditional["shots"]:
        index = int(shot["index"])
        source = next(s for s in tracks_meta["shots"] if int(s["index"]) == index)
        start, end = int(shot["start_frame"]), int(shot["end_frame"])
        detected = len(range(start, end + 1, stride)) + (1 if (end - start) % stride else 0)
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
                "plastic": plastic[index],
                "camera": cameras.get(index),
                "text": _text(shot["text"]),
                # share of the detection frames 06 ran on that saw the class
                "objects": [
                    {**o, "share": round(int(o["frame_count"]) / detected, 3)}
                    for o in source.get("objects", [])
                ],
                "framing": _framing(source, size),
                "address": _address(
                    [s for s in everyone if int(s["segment"]) == index], cfg.address_deg
                ),
            }
        )

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "person_records": len(persons),
        "thin_records": thin,
        "multi_segment_persons": sum(1 for p in persons if len(p["segments"]) > 1),
        "frame_size": list(size),
        "pose_vis_floor": cfg.pose_vis,
        "address_degrees": cfg.address_deg,
        "top_k": cfg.agg_top,
        "note": (
            "records are video-scoped: one per global person from 08, over every track it was resolved from"
        ),
        "persons": persons,
        "shots": shots,
    }
    (cfg.json_dir / "aggregated.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k not in {"persons", "shots"}}
