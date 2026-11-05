"""09's camera rule on CameraBench's test clips (arXiv 2504.15376), scored as the paper scores published methods.
After python3 -m src.tools.download_camerabench: evaluate_camera.py shots, boxes, gate, pairs, roll, score"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_ave import Row, _categorical, _ratio

from main import load_stage, parse_args

AGG: Any = load_stage(
    "09-aggregation"
)  # its constants, mask and rule, so the fits are 09's own

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

RAW = ROOT / "data" / "raw" / "camerabench"
CACHE = ROOT / "data" / "validation" / "camera" / "camerabench"
TESTS = ROOT / "data" / "processed" / "tests"
TUNED = (
    ROOT / "data" / "validation" / "camera" / "pairs" / "tests"
)  # tune_camera.py's cache of 09's fits
WORKERS = 4  # clips at once, each fitting on AGG.MOTION_THREADS threads
BOOT = 1000  # resamples of the clips behind the means' 95% intervals
SEED = 0
METHOD = "video-ai-thesis 09, B+G1+F 2026-11-04"
FIT = ("tx", "ty", "scale", "roll")
VARIANTS = {
    "masked": "",
    "unmasked": "whole_",
}  # off 06's people as 09 fits, or the whole frame

# The paper's Table 2 columns in order, each scored by one of 09's per-second quantities, signed by 09's
# directions: content sliding left is a pan right, sliding down a tilt up, growing a zoom in. A positive
# rotation angle turns the picture clockwise on screen, which is the camera rolling counterclockwise.
TASKS: dict[str, tuple[str, int]] = {
    "Move_In": ("zoom", 1),
    "Move_Out": ("zoom", -1),
    "Move_Up": ("tilt", 1),
    "Move_Down": ("tilt", -1),
    "Move_Right": ("pan", -1),
    "Move_Left": ("pan", 1),
    "Zoom_In": ("zoom", 1),
    "Zoom_Out": ("zoom", -1),
    "Pan_Right": ("pan", -1),
    "Pan_Left": ("pan", 1),
    "Tilt_Up": ("tilt", 1),
    "Tilt_Down": ("tilt", -1),
    "Roll_Clockwise": ("roll", -1),
    "Roll_Counterclockwise": ("roll", 1),
    "Static": ("static", 1),
}
OWN = "09's own seven: pan, tilt, zoom, static"
GROUPS = {  # the paper's column groups; a flat image transform sees only rotation and zoom
    OWN: (
        "Pan_Right",
        "Pan_Left",
        "Tilt_Up",
        "Tilt_Down",
        "Zoom_In",
        "Zoom_Out",
        "Static",
    ),
    "pan and tilt": ("Pan_Right", "Pan_Left", "Tilt_Up", "Tilt_Down"),
    "zoom": ("Zoom_In", "Zoom_Out"),
    "roll": ("Roll_Clockwise", "Roll_Counterclockwise"),
    "dolly, pedestal and truck": (
        "Move_In",
        "Move_Out",
        "Move_Up",
        "Move_Down",
        "Move_Right",
        "Move_Left",
    ),
    "static": ("Static",),
}
# AP per column as TASKS orders them, then the published mean: Table 2 of the NeurIPS 2025 version
PAPER: dict[str, tuple[float, ...]] = {
    "Random chance": (
        29.3,
        9.7,
        6.7,
        8.6,
        15.8,
        11.5,
        11.1,
        10.2,
        15.0,
        15.4,
        12.7,
        7.7,
        8.9,
        10.2,
        9.7,
        12.2,
    ),
    "COLMAP": (
        36.2,
        13.1,
        11.9,
        19.7,
        34.1,
        30.0,
        13.9,
        14.2,
        43.9,
        46.4,
        28.3,
        19.1,
        42.1,
        48.7,
        7.5,
        27.3,
    ),
    "VGGSfM": (
        56.6,
        28.9,
        28.7,
        38.2,
        48.9,
        35.3,
        21.7,
        17.3,
        60.9,
        58.7,
        46.6,
        43.3,
        61.4,
        55.5,
        16.7,
        41.3,
    ),
    "DUSt3R": (
        58.9,
        24.0,
        30.7,
        18.0,
        38.3,
        26.9,
        18.2,
        24.6,
        59.4,
        63.8,
        32.9,
        27.3,
        61.0,
        57.9,
        13.1,
        37.0,
    ),
    "MASt3R": (
        47.5,
        21.1,
        23.5,
        40.2,
        38.7,
        38.1,
        42.2,
        46.6,
        66.6,
        58.0,
        63.2,
        40.3,
        50.4,
        53.5,
        15.7,
        43.1,
    ),
    "CUT3R": (
        68.9,
        50.4,
        24.7,
        34.2,
        37.0,
        27.6,
        15.9,
        21.3,
        59.1,
        65.0,
        65.0,
        47.5,
        60.7,
        66.2,
        15.1,
        42.7,
    ),
    "MegaSaM": (
        73.8,
        43.9,
        24.2,
        29.1,
        45.3,
        44.2,
        11.1,
        10.2,
        79.5,
        82.2,
        73.8,
        65.3,
        71.5,
        75.8,
        22.0,
        50.1,
    ),
    "GPT-4o": (
        66.3,
        29.2,
        21.1,
        38.2,
        38.0,
        21.9,
        41.7,
        39.3,
        44.7,
        42.1,
        43.6,
        35.5,
        24.0,
        28.7,
        32.0,
        36.4,
    ),
    "Qwen2.5-VL-7B": (
        56.0,
        14.9,
        18.7,
        30.5,
        34.5,
        27.6,
        29.8,
        43.4,
        62.7,
        66.7,
        54.5,
        34.1,
        18.8,
        24.2,
        19.8,
        35.8,
    ),
    "Qwen2.5-VL-72B": (
        58.0,
        16.8,
        20.6,
        32.5,
        36.4,
        29.5,
        31.7,
        45.4,
        64.7,
        68.6,
        56.4,
        36.0,
        20.6,
        26.1,
        21.6,
        37.7,
    ),
    "Qwen2.5-VL-72B fine-tuned on CameraBench": (
        86.8,
        41.3,
        30.5,
        50.6,
        70.7,
        52.6,
        57.1,
        78.5,
        81.9,
        86.6,
        79.1,
        70.4,
        35.0,
        43.8,
        76.6,
        62.8,
    ),
}
# rows the project's leaderboard (sy77777en.github.io/CameraBench/leaderboard) gives differently, rerun since
LEADERBOARD: dict[str, tuple[float, ...]] = {
    "MegaSaM": (
        87.0,
        58.3,
        43.0,
        48.4,
        59.1,
        58.0,
        11.1,
        10.2,
        77.9,
        82.4,
        75.6,
        57.7,
        67.4,
        76.9,
        60.1,
        58.2,
    ),
    "CUT3R": (
        88.0,
        65.5,
        38.7,
        54.6,
        42.5,
        36.5,
        15.9,
        21.3,
        59.1,
        65.0,
        65.0,
        47.5,
        60.7,
        66.2,
        37.6,
        50.9,
    ),
    "MASt3R": (
        65.4,
        34.3,
        35.1,
        59.6,
        43.7,
        38.1,
        42.2,
        46.6,
        66.6,
        58.0,
        63.2,
        40.3,
        50.4,
        53.5,
        45.6,
        49.5,
    ),
    "DUSt3R": (
        70.3,
        37.3,
        41.7,
        30.2,
        41.5,
        35.6,
        18.2,
        24.6,
        59.4,
        63.8,
        32.9,
        27.3,
        61.0,
        57.9,
        42.6,
        43.0,
    ),
}
SHAKES = (
    "static",
    "no-shaking",
    "minimal-shaking",
    "unsteady",
    "very-unsteady",
)  # test.jsonl's levels
SHAKY = ("unsteady", "very-unsteady")
MOVES = {  # CameraBench's primitives grouped as AVE groups them, with the direction 09 would name
    "pan": {
        "pan-left": "left",
        "truck-left": "left",
        "pan-right": "right",
        "truck-right": "right",
    },
    "tilt": {
        "tilt-up": "up",
        "pedestal-up": "up",
        "tilt-down": "down",
        "pedestal-down": "down",
    },
    "zoom": {"zoom-in": "in", "dolly-in": "in", "zoom-out": "out", "dolly-out": "out"},
}
STILL = ("static", "no-motion")
UNGROUPED = ("roll-CW", "roll-CCW", "arc-CW", "arc-CCW")  # moves 09 has no class for


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _npz(path: Path) -> dict[str, Array]:
    with np.load(path) as found:
        return {k: found[k] for k in found.files}


def _settings() -> argparse.Namespace:
    """main.py's defaults, so 02 and 06 run here as the pipeline runs them."""
    argv = sys.argv
    sys.argv = ["main.py", "clip.mp4"]
    try:
        return parse_args()
    finally:
        sys.argv = argv


def _videos() -> list[Path]:
    return sorted((RAW / "videos").glob("*.mp4"))


def shots() -> None:
    """02's OmniShotCut on every clip: the rule assumes one shot, so a clip with a cut is scored on its longest."""
    seg: Any = load_stage("02-segmentation")
    settings = _settings()
    model, size = seg._load_shot_model(
        ROOT / "models" / "02-segmentation" / seg.SHOT_CKPT
    )
    path = CACHE / "shots.json"
    found: dict[str, Any] = _load(path) if path.exists() else {}
    videos = [v for v in _videos() if v.name not in found]
    for done, video in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
        frames: list[Array] = []
        height = width = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        capture.release()
        ranges: list[tuple[int, int]] = []
        if frames:
            detected, _, _ = seg._detect_shots(
                model,
                np.asarray(frames, dtype=np.uint8),
                settings.shot_mode,
                settings.shot_overlap,
            )
            ranges = seg._normalise(detected, len(frames))
        found[video.name] = {
            "fps": fps,
            "width": width,
            "height": height,
            "frames": len(frames),
            "shots": [list(r) for r in ranges],
        }
        if done % 50 == 0 or done == len(videos):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(found, indent=1), encoding="utf-8")
        print(f"[{done}/{len(videos)}] {video.name}: {len(ranges)} shots", flush=True)


def _track(
    model: Any,
    det: Any,
    settings: argparse.Namespace,
    device: str,
    video: Path,
    clip: dict[str, Any],
) -> tuple[dict[int, list[list[int]]], int]:
    """06's body tracking on every shot: its detector, floors, ByteTrack, stride and interpolation; no faces."""
    shots = [
        {"index": i, "start_frame": s, "end_frame": e}
        for i, (s, e) in enumerate(clip["shots"])
    ]
    stride = max(1, settings.det_stride)
    wanted: dict[int, int] = det._wanted(shots, stride)
    trackers: dict[int, Any] = {}
    observed: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}

    def flush(batch: list[tuple[int, Array]]) -> None:
        results = model.predict(
            [image for _, image in batch],
            conf=settings.track_low_conf,
            device=device,
            verbose=False,
        )
        for (index, image), result in zip(batch, results):
            shot = wanted[index]
            persons, _ = det._split(result.boxes, result.names, settings.det_conf)
            tracker = trackers.get(shot)
            if tracker is None:  # one per shot, as in 06
                tracker = det._make_tracker(settings.tracker, settings, clip["fps"])
                trackers[shot] = tracker
            for row in tracker.update(persons, image):
                x1, y1, x2, y2, track_id = (round(float(v)) for v in row[:5])
                observed.setdefault((shot, track_id), {})[index] = {
                    "body": [x1, y1, x2, y2],
                    "body_score": round(float(row[5]), 3),
                }

    batch: list[tuple[int, Array]] = []
    for item in det._from_video(video, wanted):
        batch.append(item)
        if len(batch) >= settings.track_batch:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    boxes: dict[int, list[list[int]]] = {}
    tracks = 0
    for rows_by_frame in observed.values():
        rows, _, _ = det._interpolate(rows_by_frame, stride)
        if len(rows) < settings.min_track:
            continue
        tracks += 1
        for row in rows:
            boxes.setdefault(int(row["frame"]), []).append(row["body"])
    return boxes, tracks


def _detector() -> tuple[Any, Any, argparse.Namespace, str]:
    """06's detector, module, settings and device, as main.py runs it."""
    from ultralytics import RTDETR, YOLO

    from src.tools import download_models

    det: Any = load_stage("06-detection-tracking")
    settings = _settings()
    weights = download_models.detector(det.PROFILER, settings.detector)
    model = RTDETR(str(weights)) if "rtdetr" in weights.stem else YOLO(str(weights))
    return model, det, settings, det._device(settings.device)


def boxes() -> None:
    """06's person boxes on every shot of every clip; in the pipeline 06 runs only where 03 saw a person."""
    model, det, settings, device = _detector()
    clips: dict[str, Any] = _load(CACHE / "shots.json")
    out = CACHE / "boxes"
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        n
        for n, c in clips.items()
        if c["shots"] and not (out / f"{Path(n).stem}.json").exists()
    ]
    for done, name in enumerate(todo, start=1):
        found, tracks = _track(
            model, det, settings, device, RAW / "videos" / name, clips[name]
        )
        (out / f"{Path(name).stem}.json").write_text(
            json.dumps(
                {
                    "width": clips[name]["width"],
                    "tracks": tracks,
                    "frames": {str(k): v for k, v in sorted(found.items())},
                }
            ),
            encoding="utf-8",
        )
        print(f"[{done}/{len(todo)}] {name}: {tracks} tracks", flush=True)


def _fit(
    before: Array, after: Array, mask: Array | None
) -> tuple[float, float, float, float] | None:
    """09's _motion, also keeping the similarity's rotation angle, which 09 fits but drops."""
    points = cv2.goodFeaturesToTrack(before, AGG.CORNERS, 0.01, 8, mask=mask)
    if points is None or len(points) < AGG.MIN_POINTS:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, np.empty(0))
    good = status.ravel() == 1
    if int(good.sum()) < AGG.MIN_POINTS:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        points[good], moved[good], method=cv2.RANSAC
    )
    if matrix is None:
        return None
    height, width = before.shape
    centre = np.array([width / 2, height / 2])
    shift = matrix @ np.append(centre, 1.0) - centre
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    angle = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return float(shift[0]) / width, float(shift[1]) / width, math.log(scale), angle


def _step(before: Array, after: Array, mask: Array | None) -> list[float]:
    """09's _step (off people, else the whole frame), the whole frame alone, and which one 09 used:
    1 off people, 0 nobody in the frame, 2 whole frame after the masked fit failed, -1 no fit."""
    whole = _fit(before, after, None)
    masked = _fit(before, after, mask) if mask is not None else None
    code = 1.0 if masked else -1.0 if whole is None else 0.0 if mask is None else 2.0
    blank = [math.nan] * len(FIT)
    return [*(masked or whole or blank), *(whole or blank), code]


def _decode(
    video: Path, boxes: dict[int, list[list[int]]], box_width: int
) -> dict[str, Array]:
    """Every pair of consecutive frames, read, resized and masked as 09's _motions does."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    steps: list[Any] = []
    starts: list[int] = []
    previous: tuple[Array, Array | None] | None = None
    with ThreadPoolExecutor(AGG.MOTION_THREADS) as pool:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            ratio = AGG.MOTION_SIDE / min(height, width)
            small = cv2.resize(
                frame,
                (round(width * ratio), round(height * ratio)),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mask = AGG._mask(
                gray.shape, boxes.get(index, []), ratio * width / box_width
            )
            if previous is not None:
                steps.append(pool.submit(_step, previous[0], gray, previous[1]))
                starts.append(index - 1)
            previous, index = (gray, mask), index + 1
    capture.release()
    names = (*FIT, *(f"whole_{n}" for n in FIT), "mask")
    rows = np.asarray([s.result() for s in steps], dtype=np.float64).reshape(
        -1, len(names)
    )
    return {
        "start": np.asarray(starts, dtype=np.int64),
        **{n: rows[:, i] for i, n in enumerate(names)},
    }


def gate() -> None:
    """On the test clips, 06's stored boxes and tune_camera.py's cache of 09's fits must come out again unchanged."""
    model, det, settings, device = _detector()
    differ = 0
    for cached in sorted(TUNED.glob("*.npz")):
        root = TESTS / cached.stem / "json"
        segmentation = _load(root / "segmentation.json")
        video = ROOT / segmentation["video"]
        stored: dict[int, list[list[int]]] = {}
        for shot in _load(root / "tracks.json")["shots"]:
            for track in shot.get("tracks", []):
                for row in track["frames"]:
                    stored.setdefault(int(row["frame"]), []).append(row["body"])
        routed = [
            [s["start_frame"], s["end_frame"]]
            for s in _load(root / "routing.json")["shots"]
            if "detection_tracking" in s["experts"]
        ]
        clip = {"shots": routed, "fps": float(segmentation["fps"])}
        tracked = _track(model, det, settings, device, video, clip)[0] if routed else {}
        frames = set(stored) | set(tracked)
        same_boxes = sum(
            1 for f in frames if sorted(stored.get(f, [])) == sorted(tracked.get(f, []))
        )
        fits = _decode(
            video, stored, int(_load(root / "aggregated.json")["frame_size"][0])
        )
        old = _npz(cached)
        same_fits = all(
            np.array_equal(fits[k], old[k], equal_nan=old[k].dtype.kind == "f")
            for k in ("start", "tx", "ty", "scale", "mask")
        )
        differ += same_boxes < len(frames) or not same_fits
        print(
            f"{cached.stem}: boxes identical on {same_boxes} of {len(frames)} frames; "
            f"{len(fits['start'])} pairs {'identical' if same_fits else 'DIFFERENT'}",
            flush=True,
        )
    if differ:
        sys.exit(f"{differ} test clips differ from 06's boxes or 09's fits")


def _pairs(job: tuple[Path, Path, Path]) -> str:
    video, boxes_path, out = job
    found = _load(boxes_path)
    fits = _decode(
        video, {int(k): v for k, v in found["frames"].items()}, int(found["width"])
    )
    np.savez_compressed(out, **fits)
    return f"{out.name}: {len(fits['start'])} pairs"


def pairs(workers: int) -> None:
    out = CACHE / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (RAW / "videos" / f"{path.stem}.mp4", path, out / f"{path.stem}.npz")
        for path in sorted((CACHE / "boxes").glob("*.json"))
        if not (out / f"{path.stem}.npz").exists()
    ]
    print(f"{len(jobs)} clips to fit", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for done, line in enumerate(pool.map(_pairs, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)


def _longest(clip: dict[str, Any]) -> tuple[int, int]:
    first, last = max(clip["shots"], key=lambda s: s[1] - s[0])
    return int(first), int(last)


def _covered(clip: dict[str, Any]) -> float:
    first, last = _longest(clip)
    return (last - first + 1) / clip["frames"]


def _measure(
    fits: dict[str, Array], prefix: str, shot: tuple[int, int], fps: float
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """09's camera on the shot, and the unrounded per-second quantities its decisions come from."""
    first, last = shot
    keep = (fits["start"] >= first) & (fits["start"] < last)
    tx, ty, scale, roll = (fits[prefix + n][keep] for n in FIT)
    camera: dict[str, Any] = AGG._camera(
        [None if math.isnan(x) else (x, y, s) for x, y, s in zip(tx, ty, scale)], fps
    )
    known = ~np.isnan(tx)
    if int(known.sum()) < 2:
        return camera, None
    seconds = int(known.sum()) / fps
    pan, tilt, zoom, turn = (
        float(v[known].sum()) / seconds for v in (tx, ty, scale, roll)
    )
    jitter = (AGG._spread(tx[known]) + AGG._spread(ty[known])) * math.sqrt(fps)
    stored = (camera["pan"], camera["tilt"], camera["zoom"], camera["jitter"])
    if stored != (round(pan, 4), round(tilt, 4), round(zoom, 4), round(jitter, 5)):
        raise RuntimeError("the scores drifted from 09's camera")
    static = -max(
        abs(pan) / AGG.STILL,
        abs(tilt) / AGG.STILL,
        abs(zoom) / AGG.STILL,
        jitter / AGG.SHAKE,
    )
    return camera, {
        "pan": pan,
        "tilt": tilt,
        "zoom": zoom,
        "roll": turn,
        "jitter": jitter,
        "static": static,
    }


def _measured(
    clips: dict[str, Any],
) -> dict[str, dict[str, tuple[dict[str, Any], dict[str, float] | None]]]:
    """Per variant, every clip's camera and quantities on its longest shot."""
    out: dict[str, dict[str, tuple[dict[str, Any], dict[str, float] | None]]] = {
        v: {} for v in VARIANTS
    }
    for name, clip in clips.items():
        path = CACHE / "pairs" / f"{Path(name).stem}.npz"
        if not clip["shots"] or not path.exists():
            continue
        fits = _npz(path)
        for variant, prefix in VARIANTS.items():
            out[variant][name] = _measure(fits, prefix, _longest(clip), clip["fps"])
    return out


def _official(folder: Path, files: list[Path]) -> dict[str, dict[str, Any]]:
    """CameraBench's own scorer, run on our score files, by split name."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(RAW / "binary_classification_evaluation.py"),
            *map(str, files),
            "--output_dir",
            str(folder),
            "--output_file",
            str(folder / "official.json"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.exit(result.stderr)
    return {
        r["split_name"]: r
        for r in _load(folder / "official.json")["results_by_split"].values()
    }


def _interval(
    ranked: dict[str, tuple[list[str], list[bool], list[float]]],
) -> dict[str, list[float]]:
    """95% intervals for the mean AP and each group's, resampling clips with replacement across all questions."""
    clips = sorted({n for names, _, _ in ranked.values() for n in names})
    index = {n: i for i, n in enumerate(clips)}
    rows = {
        t: (np.asarray([index[n] for n in names]), truth, scores)
        for t, (names, truth, scores) in ranked.items()
    }
    rng = np.random.default_rng(SEED)
    means: dict[str, list[float]] = {"mean": [], **{g: [] for g in GROUPS}}
    for _ in range(BOOT):
        counts = np.bincount(
            rng.integers(0, len(clips), len(clips)), minlength=len(clips)
        )
        ap = {
            t: 100
            * float(average_precision_score(truth, scores, sample_weight=counts[at]))
            for t, (at, truth, scores) in rows.items()
        }
        means["mean"].append(float(np.mean(list(ap.values()))))
        for group, tasks in GROUPS.items():
            means[group].append(float(np.mean([ap[t] for t in tasks])))
    return {
        k: [
            round(float(np.percentile(v, 2.5)), 1),
            round(float(np.percentile(v, 97.5)), 1),
        ]
        for k, v in means.items()
    }


def _classification(
    variant: str,
    tasks: dict[str, list[dict[str, Any]]],
    measured: dict[str, tuple[dict[str, Any], dict[str, float] | None]],
) -> dict[str, Any]:
    """The 15 score files in the official format, scored by the official scorer and checked against scikit-learn."""
    folder = CACHE / "scores" / variant
    folder.mkdir(parents=True, exist_ok=True)
    model = f"video-ai-thesis-09-{variant}"
    files: list[Path] = []
    ours: dict[str, tuple[float, float, int]] = {}
    ranked: dict[str, tuple[list[str], list[bool], list[float]]] = {}
    for task, (quantity, sign) in TASKS.items():
        items = tasks[task]
        values: dict[str, float] = {}
        for item in items:
            found = measured.get(item["image"], ({}, None))[1]
            if found is not None:
                values[item["image"]] = sign * found[quantity]
        neutral = float(
            np.median(list(values.values()))
        )  # an unmeasured clip ranks mid-list, 09's "unknown"
        scores = [values.get(i["image"], neutral) for i in items]
        truth = [i["label"].lower() == "yes" for i in items]
        ranked[task] = ([i["image"] for i in items], truth, scores)
        ours[task] = (
            float(average_precision_score(truth, scores)),
            float(roc_auc_score(truth, scores)),
            len(items) - len(values),
        )
        path = folder / f"classification_scores_{task}.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "method_type": "camera_rule",
                        "model_name": model,
                        "checkpoint": "",
                        "split_name": task,
                        "method": METHOD,
                    },
                    "scores": [
                        {
                            "sample_id": str(k),
                            "video_path": item["image"],
                            "question": item["question"],
                            "ground_truth_label": item["label"],
                            "method": model,
                            "score": score,
                            "error": None,
                        }
                        for k, (item, score) in enumerate(zip(items, scores))
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        files.append(path)

    official = _official(folder, files)
    per_task: dict[str, Any] = {}
    for column, task in enumerate(TASKS):
        found = official[task]
        ap, auc, unmeasured = ours[task]
        if (
            abs(found["average_precision"] - ap) > 1e-9
            or abs(found["roc_auc"] - auc) > 1e-9
        ):
            raise RuntimeError(f"{task}: the official scorer and scikit-learn disagree")
        chance = round(100 * found["num_positive"] / found["num_samples"], 1)
        if chance != PAPER["Random chance"][column]:
            raise RuntimeError(
                f"{task}: {chance}% yes, the paper's chance row says {PAPER['Random chance'][column]}"
            )
        per_task[task] = {
            "ap": round(100 * ap, 1),
            "roc_auc": round(100 * auc, 1),
            "yes": found["num_positive"],
            "no": found["num_negative"],
            "unmeasured": unmeasured,
        }
    aps = [100 * ours[t][0] for t in TASKS]
    interval = _interval(ranked)
    return {
        "mean_ap": round(float(np.mean(aps)), 1),
        "mean_ap_95": interval.pop("mean"),
        "mean_roc_auc": round(100 * float(np.mean([ours[t][1] for t in TASKS])), 1),
        "groups": {
            g: round(float(np.mean([100 * ours[t][0] for t in ts])), 1)
            for g, ts in GROUPS.items()
        },
        "groups_95": interval,
        "per_task": per_task,
        "official_scores": str(folder.relative_to(ROOT)),
    }


def _groups(row: tuple[float, ...]) -> dict[str, float]:
    tasks = list(TASKS)
    return {
        g: round(float(np.mean([row[tasks.index(t)] for t in ts])), 1)
        for g, ts in GROUPS.items()
    }


def _decisions(
    measured: dict[str, tuple[dict[str, Any], dict[str, float] | None]],
    labels: dict[str, list[str]],
    static: list[dict[str, Any]],
) -> dict[str, Any]:
    """09's rig, move and static decisions against CameraBench's labels, at the thresholds fitted on AVE."""
    levels: dict[str, list[bool]] = {}
    rig: list[tuple[bool, float, bool]] = []  # shaky, jitter, called handheld
    for name, tags in labels.items():
        level = next((s for s in SHAKES if s in tags), None)
        camera, values = measured.get(name, ({}, None))
        if level is None or values is None:
            continue
        handheld = camera["rig"] == "handheld"
        levels.setdefault(level, []).append(handheld)
        if level != "minimal-shaking":
            rig.append((level in SHAKY, values["jitter"], handheld))
    called = [shaky for shaky, _, handheld in rig if handheld]
    shaky = sum(1 for s, _, _ in rig if s)

    moves: list[Row] = []
    directions: list[bool] = []
    for name, tags in labels.items():
        camera, values = measured.get(name, ({}, None))
        if not camera or any(t in tags for t in UNGROUPED):
            continue
        truth = {
            group for group, members in MOVES.items() if any(t in tags for t in members)
        }
        if not truth and any(t in tags for t in STILL):
            truth = {"none"}
        moves.append((truth, camera["move"]))
        if len(truth) == 1 and camera["move"] in truth and camera["move"] != "none":
            wanted = {d for t, d in MOVES[camera["move"]].items() if t in tags}
            if len(wanted) == 1:
                directions.append(camera["direction"] in wanted)

    still = [
        (i["label"].lower() == "yes", measured[i["image"]][0]["movement"] == "static")
        for i in static
        if i["image"] in measured
    ]
    said = [truth for truth, called_static in still if called_static]
    return {
        "rig": {
            "jitter_auc": round(
                float(roc_auc_score([s for s, _, _ in rig], [j for _, j, _ in rig])), 3
            ),
            "handheld_precision": _ratio(sum(called), len(called)),
            "handheld_recall": _ratio(sum(called), shaky),
            "clips": len(rig),
            "shaky": shaky,
            "handheld_share_by_level": {
                level: {
                    "clips": len(levels[level]),
                    "handheld": _ratio(sum(levels[level]), len(levels[level])),
                }
                for level in SHAKES
                if level in levels
            },
        },
        "move": {
            **_categorical(moves, ("none", "pan", "tilt", "zoom"), False),
            "direction_accuracy": _ratio(sum(directions), len(directions)),
            "direction_clips": len(directions),
        },
        "static": {
            "precision": _ratio(sum(said), len(said)),
            "recall": _ratio(sum(said), sum(1 for t, _ in still if t)),
            "called_static": len(said),
            "clips": len(still),
        },
    }


def score(out: Path) -> None:
    clips: dict[str, Any] = _load(CACHE / "shots.json")
    labels = {
        Path(r["path"]).name: list(r["labels"]) for r in _jsonl(RAW / "test.jsonl")
    }
    tasks = {t: _jsonl(RAW / "tasks" / f"{t}.jsonl") for t in TASKS}
    measured = _measured(clips)
    named = {i["image"] for items in tasks.values() for i in items}
    fps = [c["fps"] for c in clips.values()]
    cut = [_covered(c) for c in clips.values() if len(c["shots"]) > 1]
    frames = [c["frames"] for c in clips.values()]
    results = {v: _classification(v, tasks, measured[v]) for v in VARIANTS}
    masked = results["masked"]
    fits = [
        _npz(CACHE / "pairs" / f"{Path(n).stem}.npz")["mask"]
        for n in measured["masked"]
    ]
    codes = np.concatenate(fits) if fits else np.empty(0)
    columns = [*TASKS, "mean"]
    report = {
        "about": (
            "CameraBench (Lin et al., NeurIPS 2025) asks 15 yes/no questions about how the camera moves in 1,071 "
            "internet clips labelled by experts. A method ranks the clips for each question and is scored by average "
            "precision (AP: 100 means every 'yes' clip ranked above every 'no'; random ranking gives "
            f"{PAPER['Random chance'][-1]}). Here stage 09's camera rule, with its thresholds fitted on AVE films and "
            "nothing fitted on CameraBench, ranks the clips by its own per-second pan, tilt, zoom and roll and its "
            "shake, and CameraBench's own code scores it. Over all 15 questions our mean AP is "
            f"{masked['mean_ap']} (95% interval {masked['mean_ap_95'][0]}-{masked['mean_ap_95'][1]}); the best 3D "
            f"reconstruction, MegaSaM, has {PAPER['MegaSaM'][-1]} in the paper and {LEADERBOARD['MegaSaM'][-1]} on the "
            f"later leaderboard. On the seven questions 09 itself answers (pan, tilt, zoom, static) ours is "
            f"{masked['groups'][OWN]}, MegaSaM's {_groups(PAPER['MegaSaM'])[OWN]} "
            f"({_groups(LEADERBOARD['MegaSaM'])[OWN]} on the leaderboard). We lead on zoom and stillness, where 3D "
            "reconstruction is weak, and trail on pan and tilt. A flat image transform cannot tell a pan from a "
            "sideways move (truck), a tilt from a vertical move (pedestal) or a zoom from a forward move (dolly), so "
            "those six questions reuse the pan, tilt and zoom scores."
        ),
        "source": _load(RAW / "source.json"),
        "method": METHOD,
        "clips": {
            "videos": len(clips),
            "named_in_tasks": len(named),
            "missing_from_videos": sorted(named - set(clips)),
            "with_a_cut": len(cut),
            "longest_shot_share_of_cut_clips": {
                "median": round(float(np.median(cut)), 3) if cut else None,
                "under_half": sum(1 for c in cut if c < 0.5),
            },
            "measured": {
                v: sum(1 for _, q in m.values() if q is not None)
                for v, m in measured.items()
            },
            "fps": {
                "min": round(min(fps), 2),
                "median": round(float(np.median(fps)), 2),
                "max": round(max(fps), 2),
            },
            "frames": {
                "min": min(frames),
                "median": int(np.median(frames)),
                "max": max(frames),
            },
            "pairs_fitted": {
                "off_people": _ratio(int((codes == 1).sum()), codes.size),
                "nobody_in_frame": _ratio(int((codes == 0).sum()), codes.size),
                "whole_frame_after_mask_failed": _ratio(
                    int((codes == 2).sum()), codes.size
                ),
                "no_fit": _ratio(int((codes == -1).sum()), codes.size),
            },
        },
        "binary_classification": results,
        "comparison": {
            "columns": columns,
            "ours": {
                v: [r["per_task"][t]["ap"] for t in TASKS] + [r["mean_ap"]]
                for v, r in results.items()
            },
            "paper_table_2": {name: list(row) for name, row in PAPER.items()},
            "leaderboard_reruns": {
                name: list(row) for name, row in LEADERBOARD.items()
            },
            "paper_methods_under_ours_masked": [
                name
                for name, row in PAPER.items()
                if name != "Random chance" and row[-1] < results["masked"]["mean_ap"]
            ],
            "groups": {
                **{f"ours {v}": r["groups"] for v, r in results.items()},
                **{name: _groups(row) for name, row in PAPER.items()},
                **{
                    f"{name} (leaderboard)": _groups(row)
                    for name, row in LEADERBOARD.items()
                },
            },
        },
        "decisions": {
            v: _decisions(measured[v], labels, tasks["Static"]) for v in VARIANTS
        },
        "notes": [
            "Each clip is scored on its longest shot by 02's OmniShotCut; per question, a clip with fewer than two "
            "fitted frame pairs gets that question's median score.",
            "Person boxes come from 06's detector, ByteTrack and settings run on every shot; in the pipeline 06 runs "
            "only on shots where 03 saw a person, so the pipeline would mask somewhat fewer frames.",
            "Rig: shaky = unsteady or very-unsteady, steady = no-shaking or static; minimal-shaking is left out of "
            "the AUC, precision and recall.",
            "Move: clips whose primitives fall in one of AVE's groups (pan with truck, tilt with pedestal, zoom with "
            "dolly), or static/no-motion as none; clips with roll or arc are left out.",
            "The leaderboard reran some 3D reconstruction methods after the paper; both are listed.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for variant, result in results.items():
        print(f"{variant}: mean AP {result['mean_ap']}, groups {result['groups']}")
    print(f"wrote {out}")


def roll(out: Path) -> None:
    """Five frames of labelled roll clips that 09 sees turn 20-90 degrees, to check the roll sign by eye."""
    clips: dict[str, Any] = _load(CACHE / "shots.json")
    measured = _measured(clips)["masked"]
    rows: list[Array] = []
    for task in ("Roll_Clockwise", "Roll_Counterclockwise"):
        yes = [
            i["image"]
            for i in _jsonl(RAW / "tasks" / f"{task}.jsonl")
            if i["label"].lower() == "yes"
        ]
        turned: list[tuple[str, float]] = []
        for name in yes:
            values = measured.get(name, ({}, None))[1]
            if values is not None:
                first, last = _longest(clips[name])
                turned.append(
                    (
                        name,
                        math.degrees(
                            values["roll"] * (last - first) / clips[name]["fps"]
                        ),
                    )
                )
        for name, degrees in sorted(
            (t for t in turned if 20 <= abs(t[1]) <= 90), key=lambda t: -abs(t[1])
        )[:3]:
            first, last = _longest(clips[name])
            capture = cv2.VideoCapture(str(RAW / "videos" / name))
            pictures: list[Array] = []
            for index in np.linspace(first, last, 5).round().astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                _, frame = capture.read()
                height, width = frame.shape[:2]
                pictures.append(cv2.resize(frame, (round(width * 160 / height), 160)))
                pictures.append(np.zeros((160, 6, 3), np.uint8))
            capture.release()
            row = np.vstack(
                [
                    np.zeros((30, sum(p.shape[1] for p in pictures), 3), np.uint8),
                    np.hstack(pictures),
                ]
            )
            turn = "clockwise" if degrees > 0 else "counterclockwise"
            text = f"label {task}: {name}   fits: picture turns {turn} {abs(degrees):.0f} deg"
            cv2.putText(
                row, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack(
        [np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Score 09's camera rule on CameraBench")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("shots", help="02's shot cuts in every clip")
    sub.add_parser("boxes", help="06's person boxes on every shot")
    sub.add_parser(
        "gate", help="the box and fit code against 06's and 09's own on the test clips"
    )
    cache = sub.add_parser("pairs", help="every frame pair's fit, masked and unmasked")
    cache.add_argument("--workers", type=int, default=WORKERS)
    sheet = sub.add_parser(
        "roll",
        help="labelled roll clips across their shot, to check the roll sign by eye",
    )
    sheet.add_argument("--out", type=Path, default=CACHE / "roll.png")
    rate = sub.add_parser(
        "score", help="the official scores, the comparison and 09's decisions"
    )
    rate.add_argument("--out", type=Path, default=CACHE.parent / "camerabench.json")
    args = p.parse_args()
    if args.command == "shots":
        shots()
    elif args.command == "boxes":
        boxes()
    elif args.command == "gate":
        gate()
    elif args.command == "pairs":
        pairs(args.workers)
    elif args.command == "roll":
        roll(args.out)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
