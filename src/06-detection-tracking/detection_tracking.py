"""06-detection-tracking: the first gated stage.

Runs only on shots where 04-router fired detection_tracking, and establishes the identities
every later conditional expert depends on. Bodies are detected and tracked; faces are found
with YuNet inside each tracked body box, so a face inherits its body's track id instead of
needing a second tracker.

Track ids are scoped to one shot - the tracker is rebuilt at every boundary, because Kalman
and IoU association carry no meaning across a cut. The key is (shot, track_id).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

import download_models
from config import Config

STAGE = "06-detection-tracking"
PROFILER = "03-profiler"  # detector and YuNet weights are reused from there
FACE_MODEL = "face_yunet.onnx"
PERSON_CLASS = 0
FACE_INPUT = (320, 320)

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Detections:
    """What ultralytics' trackers read off a results object."""

    xywh: Array
    conf: Array
    cls: Array

    def __len__(self) -> int:
        return int(self.conf.shape[0])


@dataclass(frozen=True)
class TrackerArgs:
    """ByteTrack splits detections at track_high_thresh and runs a second association pass
    over everything down to track_low_thresh. That pass is what recovers blurred or occluded
    people, so track_low_thresh must sit well below 03-profiler's detection threshold."""

    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = False


def _read(out_root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = out_root / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _wanted(shots: list[dict[str, Any]], stride: int) -> dict[int, int]:
    """Frame index -> shot index, for the frames that get a real detection pass. Skipped
    frames are filled in by interpolation, so no pixels are needed for them. A shot's last
    frame is always included, to anchor the interpolation at both ends."""
    picked: dict[int, int] = {}
    for shot in shots:
        start, end = int(shot["start_frame"]), int(shot["end_frame"])
        index = int(shot["index"])
        for frame in range(start, end + 1, stride):
            picked[frame] = index
        picked[end] = index
    return picked


def _from_video(video: Path, wanted: dict[int, int]) -> Iterator[tuple[int, Array]]:
    """Sequential decode. grab() skips unwanted frames without paying for colour conversion,
    but every frame before the last wanted one still has to be walked past."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    last = max(wanted)
    index = 0
    while index <= last:
        if not capture.grab():
            break
        if index in wanted:
            ok, frame = capture.retrieve()
            if not ok:
                break
            yield index, frame
        index += 1
    capture.release()


def _from_frames(
    frames_dir: Path, wanted: dict[int, int], suffix: str
) -> Iterator[tuple[int, Array]]:
    """Random access into 01-sampling's output: only the wanted files are ever touched."""
    for index in sorted(wanted):
        path = frames_dir / f"{index:06d}.{suffix}"
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"missing {path}; run 01-sampling with no --fps first")
        yield index, image


def _source(
    cfg: Config, wanted: dict[int, int], frame_count: int
) -> Iterator[tuple[int, Array]]:
    if cfg.frame_source == "video":
        return _from_video(cfg.video, wanted)

    meta = _read(cfg.out_root, "sampling.json", "01-sampling")
    if int(meta["stride"]) != 1:
        raise RuntimeError(
            "01-sampling was run with --fps; its frames are not one-to-one"
        )
    if int(meta["frame_count"]) != frame_count:
        raise RuntimeError(
            f"01-sampling wrote {meta['frame_count']} frames but 02 saw {frame_count}"
        )
    return _from_frames(
        cfg.out_root / str(meta["frames_dir"]), wanted, str(meta["image_format"])
    )


def _make_tracker(name: str, cfg: Config, fps: float) -> Any:
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = TrackerArgs(
        track_high_thresh=cfg.track_high_conf,
        track_low_thresh=cfg.track_low_conf,
        new_track_thresh=cfg.track_new_conf,
        track_buffer=cfg.track_buffer,
        match_thresh=cfg.track_match,
    )
    rate = max(1, round(fps))
    if name == "botsort":
        return BOTSORT(args, frame_rate=rate)
    return BYTETracker(args, frame_rate=rate)


def _split(
    boxes: Any, names: dict[int, str], object_conf: float
) -> tuple[Detections, list[dict[str, Any]]]:
    """One forward pass gives persons and objects. Persons go to the tracker at the low
    threshold ByteTrack needs; objects only need a temporal union, so they keep 03's
    threshold rather than drowning the union in low-confidence junk."""
    xywh = boxes.xywh.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    people = cls == PERSON_CLASS
    persons = Detections(xywh[people], conf[people], cls[people])

    keep = (~people) & (conf >= object_conf)
    objects = [
        {
            "label": names[int(c)],
            "score": round(float(s), 3),
            "box": [int(v) for v in b],
        }
        for b, s, c in zip(xyxy[keep], conf[keep], cls[keep])
    ]
    return persons, objects


def _frontality(landmarks: Array) -> float:
    """YuNet returns right eye, left eye, nose and both mouth corners. A frontal face puts
    the nose on the midpoint between the eyes; profile views push it towards one of them."""
    right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    span = float(np.linalg.norm(left_eye - right_eye))
    if span < 1e-6:
        return 0.0
    centre = (right_eye + left_eye) / 2.0
    return float(max(0.0, 1.0 - 2.0 * abs(float(nose[0]) - float(centre[0])) / span))


def _letterbox(crop: Array, size: int) -> tuple[Array, float]:
    """Body boxes are all different shapes, but YuNet must always see the same input size.
    Calling setInputSize per crop reallocates its buffers and it then returns phantom faces
    scoring ~1.0, differing between runs on identical pixels. Fixed size, padded, is stable."""
    height, width = crop.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    canvas: Array = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:new_height, :new_width] = cv2.resize(
        crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale


def _face_in(
    detector: Any, image: Array, box: list[int], conf: float
) -> dict[str, Any] | None:
    """YuNet inside the tracked body box only, so a face always belongs to a body track."""
    height, width = image.shape[:2]
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(width, box[2]), min(height, box[3])
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None

    canvas, scale = _letterbox(image[y1:y2, x1:x2], FACE_INPUT[0])
    _, detections = detector.detect(canvas)
    if detections is None or not len(detections):
        return None

    best = max(detections, key=lambda row: float(row[14]))
    score = float(best[14])
    if score < conf:
        return None

    # YuNet occasionally returns inf for a box on marginal input.
    raw = np.asarray(best[:14], dtype=np.float64)
    if not bool(np.isfinite(raw).all()):
        return None

    fx, fy, fw, fh = (round(float(v) / scale) for v in raw[:4])
    if fw <= 0 or fh <= 0:
        return None
    landmarks = raw[4:14].reshape(5, 2) / scale
    # YuNet regresses coordinates that can run past the crop, so clamp to the body box:
    # a face belongs to its track's body by construction and must not escape it.
    return {
        "box": [
            min(max(x1 + fx, x1), x2),
            min(max(y1 + fy, y1), y2),
            min(max(x1 + fx + fw, x1), x2),
            min(max(y1 + fy + fh, y1), y2),
        ],
        "score": round(score, 3),
        "frontality": round(_frontality(landmarks), 3),
    }


def _interpolate(
    observed: dict[int, dict[str, Any]], max_gap: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Detection runs on a stride; frames in between are filled from the observations either
    side. This is offline, so interpolation beats forward prediction.

    Only the stride is filled. A longer gap means the tracker lost the object and re-acquired
    it, and a straight line across that is fabrication - those frames get no row at all."""
    frames = sorted(observed)
    out: list[dict[str, Any]] = []
    gaps = 0
    widest = 0
    for i, frame in enumerate(frames):
        out.append({"frame": frame, **observed[frame], "interpolated": False})
        if i + 1 >= len(frames):
            continue

        nxt = frames[i + 1]
        gap = nxt - frame
        if gap > max_gap:
            gaps += 1
            widest = max(widest, gap)
            continue

        start_box = np.asarray(observed[frame]["body"], dtype=np.float64)
        end_box = np.asarray(observed[nxt]["body"], dtype=np.float64)
        for step in range(1, gap):
            box = start_box + (end_box - start_box) * (step / gap)
            out.append(
                {
                    "frame": frame + step,
                    "body": [round(float(v)) for v in box],
                    "body_score": observed[frame]["body_score"],
                    "face": None,
                    "face_score": None,
                    "interpolated": True,
                }
            )
    return sorted(out, key=lambda row: int(row["frame"])), gaps, widest


def _pick_crops(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Split the track into equal spans and take the best of each. Ranking a whole track at
    once returns near-identical neighbouring frames - the trap 02 hit with keyframes."""
    if not candidates or count <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: int(c["frame"]))
    spans = np.linspace(0, len(ordered), min(count, len(ordered)) + 1).astype(int)

    picked: list[dict[str, Any]] = []
    for i in range(len(spans) - 1):
        lo, hi = int(spans[i]), int(spans[i + 1])
        if hi > lo:
            picked.append(max(ordered[lo:hi], key=lambda c: float(c["score"])))
    return picked


def _score_pool(pool: list[dict[str, Any]]) -> None:
    """Sharpness and area are relative to the best in the same track, so a distant track is
    not punished for being small. Frontality only exists for face crops."""
    sharpest = max(float(c["sharpness"]) for c in pool) or 1.0
    largest = max(float(c["area"]) for c in pool) or 1.0
    for candidate in pool:
        frontal = candidate["frontality"]
        candidate["score"] = round(
            float(candidate["sharpness"])
            / sharpest
            * float(candidate["area"])
            / largest
            * (float(frontal) if frontal is not None else 0.5)
            * float(candidate["quality"]),
            5,
        )


def _objects(seen: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Temporal union rather than identity: the identikit consumes object classes and how
    long they last, so per-class extent is enough without stable object ids."""
    union: dict[str, dict[str, Any]] = {}
    per_frame: dict[tuple[str, int], int] = {}
    for frame, box in seen:
        label = str(box["label"])
        key = (label, frame)
        per_frame[key] = per_frame.get(key, 0) + 1
        entry = union.get(label)
        if entry is None:
            union[label] = {
                "label": label,
                "first_frame": frame,
                "last_frame": frame,
                "frames": {frame},
                "max_per_frame": per_frame[key],
            }
            continue
        entry["first_frame"] = min(int(entry["first_frame"]), frame)
        entry["last_frame"] = max(int(entry["last_frame"]), frame)
        entry["frames"].add(frame)
        entry["max_per_frame"] = max(int(entry["max_per_frame"]), per_frame[key])

    out: list[dict[str, Any]] = []
    for entry in union.values():
        frames = entry.pop("frames")
        entry["frame_count"] = len(frames)
        out.append(entry)
    return sorted(out, key=lambda o: (-int(o["frame_count"]), str(o["label"])))


def _write_crops(
    cfg: Config,
    crop_pool: dict[int, dict[int, list[dict[str, Any]]]],
    wanted: dict[int, int],
    frame_count: int,
) -> None:
    """Second pass over the source, re-reading only the frames that won a crop slot."""
    needed: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    for shot, tracks in crop_pool.items():
        for track_id, picks in tracks.items():
            for candidate in picks:
                needed.setdefault(int(candidate["frame"]), []).append(
                    (shot, track_id, candidate)
                )
    if not needed:
        return

    root = cfg.out_root / "tracks"
    for index, image in _source(cfg, {f: wanted[f] for f in needed}, frame_count):
        for shot, track_id, candidate in needed[index]:
            x1, y1, x2, y2 = candidate["box"]
            pad = round(cfg.crop_pad * max(x2 - x1, y2 - y1))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
            crop = image[y1:y2, x1:x2]
            if not crop.size:
                continue
            directory = root / f"{shot:04d}_{track_id:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            name = f"{candidate['kind']}_{index:06d}.jpg"
            cv2.imwrite(str(directory / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            candidate["path"] = f"tracks/{directory.name}/{name}"


def run(cfg: Config) -> dict[str, Any]:
    from ultralytics import RTDETR, YOLO

    device = _device(cfg.device)
    segmentation = _read(cfg.out_root, "segmentation.json", "02-segmentation")
    routing = _read(cfg.out_root, "routing.json", "04-router")
    fps = float(segmentation["fps"])
    frame_count = int(segmentation["frame_count"])

    download_models.ensure(PROFILER)
    weights = download_models.detector(PROFILER, cfg.detector)
    model = RTDETR(str(weights)) if "rtdetr" in weights.stem else YOLO(str(weights))
    face_detector = cv2.FaceDetectorYN.create(
        str(download_models.stage_dir(PROFILER) / FACE_MODEL),
        "",
        FACE_INPUT,
        cfg.face_conf,
        0.3,
        5000,
    )

    gated = [s for s in routing["shots"] if "detection_tracking" in s["experts"]]
    faces_wanted = {int(s["index"]) for s in routing["shots"] if "face" in s["experts"]}
    wanted = _wanted(gated, max(1, cfg.det_stride)) if gated else {}

    crops_root = cfg.out_root / "tracks"
    if crops_root.exists():
        shutil.rmtree(crops_root)

    observations: dict[int, dict[int, dict[int, dict[str, Any]]]] = {}
    seen_objects: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    crop_pool: dict[int, dict[int, list[dict[str, Any]]]] = {}
    trackers: dict[int, Any] = {}
    batch: list[tuple[int, Array]] = []
    read_seconds = 0.0
    total_seconds = 0.0
    frames_read = 0

    def flush() -> None:
        if not batch:
            return
        results = model.predict(
            [image for _, image in batch],
            conf=cfg.track_low_conf,
            device=device,
            verbose=False,
        )
        for (index, image), result in zip(batch, results):
            shot = wanted[index]
            persons, objects = _split(result.boxes, result.names, cfg.det_conf)
            if objects:
                seen_objects.setdefault(shot, []).extend(
                    (index, box) for box in objects
                )

            # A fresh tracker per shot: association across a cut is meaningless.
            # Built once per shot, never with setdefault: that evaluates its default every
            # call, and each tracker constructed resets the global track id counter.
            tracker = trackers.get(shot)
            if tracker is None:
                tracker = _make_tracker(cfg.tracker, cfg, fps)
                trackers[shot] = tracker
            for row in tracker.update(persons, image):
                x1, y1, x2, y2, track_id = (round(float(v)) for v in row[:5])
                body = [x1, y1, x2, y2]
                score = round(float(row[5]), 3)
                face = (
                    _face_in(face_detector, image, body, cfg.face_conf)
                    if shot in faces_wanted
                    else None
                )
                observations.setdefault(shot, {}).setdefault(track_id, {})[index] = {
                    "body": body,
                    "body_score": score,
                    "face": face["box"] if face else None,
                    "face_score": face["score"] if face else None,
                }

                region = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if region.size:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    crop_pool.setdefault(shot, {}).setdefault(track_id, []).append(
                        {
                            "frame": index,
                            "kind": "face" if face else "body",
                            "box": face["box"] if face else body,
                            "path": None,
                            "quality": face["score"] if face else score,
                            "frontality": face["frontality"] if face else None,
                            "sharpness": round(
                                float(cv2.Laplacian(gray, cv2.CV_32F).var()), 1
                            ),
                            "area": float((y2 - y1) * (x2 - x1)),
                        }
                    )
        batch.clear()

    if wanted:
        started = time.perf_counter()
        stream = _source(cfg, wanted, frame_count)
        while True:
            # Timed around the pull alone, so read_seconds compares frame sources rather
            # than the detector that dominates the rest of the loop.
            clock = time.perf_counter()
            item = next(stream, None)
            read_seconds += time.perf_counter() - clock
            if item is None:
                break
            frames_read += 1
            batch.append(item)
            if len(batch) >= cfg.track_batch:
                flush()
        flush()
        total_seconds = time.perf_counter() - started

    # Face and body crops are ranked separately: they feed the two different conditional
    # experts, and one pool of five mixed kinds can leave either expert with nothing.
    # Interpolate and drop short tracks before anything else, so crops are never written for
    # a track that will not appear, and so ids can be renumbered densely. ByteTrack's own ids
    # are sparse: it burns one per candidate track and most never confirm.
    kept: dict[int, list[dict[str, Any]]] = {}
    for index, by_id in observations.items():
        survivors: list[dict[str, Any]] = []
        for tracker_id, observed in by_id.items():
            rows, gaps, widest = _interpolate(observed, max(1, cfg.det_stride))
            if len(rows) < cfg.min_track:
                continue
            survivors.append(
                {
                    "tracker_id": tracker_id,
                    "rows": rows,
                    "gaps": gaps,
                    "widest_gap": widest,
                    "observed": len(observed),
                }
            )
        survivors.sort(key=lambda s: (int(s["rows"][0]["frame"]), int(s["tracker_id"])))
        for dense, track in enumerate(survivors, start=1):
            track["track_id"] = dense
        kept[index] = survivors

    # Crops are keyed by the dense id from here on, including their directory names.
    picks: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for index, survivors in kept.items():
        for track in survivors:
            pool = crop_pool.get(index, {}).get(int(track["tracker_id"]), [])
            chosen: list[dict[str, Any]] = []
            for kind in ("face", "body"):
                same = [c for c in pool if c["kind"] == kind]
                if same:
                    _score_pool(same)
                    chosen += _pick_crops(same, cfg.crops)
            picks.setdefault(index, {})[int(track["track_id"])] = chosen
    _write_crops(cfg, picks, wanted, frame_count)

    shots: list[dict[str, Any]] = []
    total_tracks = 0
    for shot in routing["shots"]:
        index = int(shot["index"])
        base = {
            "index": index,
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
        }
        if "detection_tracking" not in shot["experts"]:
            shots.append(
                {**base, "tracked": False, "reason": "detection_tracking did not fire"}
            )
            continue

        tracks: list[dict[str, Any]] = []
        for track in kept.get(index, []):
            rows = track["rows"]
            picked = picks.get(index, {}).get(int(track["track_id"]), [])
            tracks.append(
                {
                    "track_id": track["track_id"],
                    # ByteTrack's own id, kept so a track can be traced back to the tracker.
                    "tracker_id": track["tracker_id"],
                    "first_frame": rows[0]["frame"],
                    "last_frame": rows[-1]["frame"],
                    "frame_count": len(rows),
                    "observed_frames": track["observed"],
                    "face_frames": sum(1 for r in rows if r["face"]),
                    "gaps": track["gaps"],
                    "widest_gap": track["widest_gap"],
                    "crops": {
                        kind: [c for c in picked if c["kind"] == kind]
                        for kind in ("face", "body")
                    },
                    "frames": rows,
                }
            )
        total_tracks += len(tracks)
        shots.append(
            {
                **base,
                "tracked": True,
                "track_count": len(tracks),
                "tracks": tracks,
                "objects": _objects(seen_objects.get(index, [])),
            }
        )

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "tracked_shots": len(gated),
        "skipped_shots": len(shots) - len(gated),
        "track_count": total_tracks,
        "detector": cfg.detector,
        "tracker": cfg.tracker,
        "device": device,
        "det_stride": cfg.det_stride,
        "frame_source": cfg.frame_source,
        "frames_total": frame_count,
        "frames_read": frames_read,
        "read_seconds": round(read_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "thresholds": {
            "track_low": cfg.track_low_conf,
            "track_high": cfg.track_high_conf,
            "new_track": cfg.track_new_conf,
            "face": cfg.face_conf,
        },
        "note": "track ids are scoped to one shot; the key is (shot, track_id)",
        "shots": shots,
    }
    (cfg.out_root / "tracks.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "shots"}
