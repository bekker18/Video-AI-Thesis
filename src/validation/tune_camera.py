"""09's camera rule and changes to it, scored against AVE's shot-motion labels from cached frame-pair fits:
python3 src/validation/tune_camera.py pairs [--stride 2]   then   python3 src/validation/tune_camera.py evaluate"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_ave import MOTION, MOVE, RIG, _categorical, _values, matched

from main import load_stage

AGG: Any = load_stage(
    "09-aggregation"
)  # its constants, mask and rule, so the fits are 09's own

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

COLLECTIONS = {"ave": "ave_processed", "tests": "tests", "reels": "reels"}
CACHE = ROOT / "data" / "validation" / "camera" / "pairs"
WORKERS = 4  # videos at once, each fitting on AGG.MOTION_THREADS threads
FIT = ("tx", "ty", "scale", "points", "tracked", "inliers")
# mask: 1 fit off people, 0 whole frame with nobody in it, 2 whole frame after the masked fit failed, -1 no fit


def _read(root: Path, name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(
        (root / "json" / name).read_text(encoding="utf-8")
    )
    return found


def _fit(before: Array, after: Array, mask: Array | None) -> tuple[float, ...] | None:
    """09's _motion, also returning the corners found, tracked and kept by RANSAC."""
    points = cv2.goodFeaturesToTrack(before, AGG.CORNERS, 0.01, 8, mask=mask)
    if points is None or len(points) < AGG.MIN_POINTS:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, np.empty(0))
    good = status.ravel() == 1
    if int(good.sum()) < AGG.MIN_POINTS:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[good], moved[good], method=cv2.RANSAC
    )
    if matrix is None:
        return None
    height, width = before.shape
    centre = np.array([width / 2, height / 2])
    shift = matrix @ np.append(centre, 1.0) - centre
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    return (
        float(shift[0]) / width,
        float(shift[1]) / width,
        math.log(scale),
        float(len(points)),
        float(good.sum()),
        float(inliers.sum()),
    )


def _step(before: Array, after: Array, mask: Array | None) -> list[float]:
    """09's _step: off people when the frame has any, else, or if too few corners are left, the whole frame."""
    found = _fit(before, after, mask) if mask is not None else None
    if found is not None:
        return [*found, 1.0]
    found = _fit(before, after, None)
    if found is None:
        return [math.nan] * len(FIT) + [-1.0]
    return [*found, 0.0 if mask is None else 2.0]


def _pairs(job: tuple[Path, Path, int]) -> str:
    """Every pair of frames `stride` apart, decoded and masked as 09 does."""
    root, out, stride = job
    segmentation = _read(root, "segmentation.json")
    tracks = _read(root, "tracks.json")
    size = _read(root, "aggregated.json")["frame_size"]
    boxes: dict[int, list[list[int]]] = {}
    for shot in tracks["shots"]:
        for track in shot.get("tracks", []):
            for row in track["frames"]:
                boxes.setdefault(int(row["frame"]), []).append(row["body"])

    capture = cv2.VideoCapture(str(ROOT / segmentation["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {segmentation['video']}")
    steps: list[Any] = []
    starts: list[int] = []
    previous: tuple[Array, Array | None, int] | None = None
    shape = (0, 0)
    with ThreadPoolExecutor(AGG.MOTION_THREADS) as pool:
        index = 0
        while True:
            if index % stride:
                if not capture.grab():
                    break
                index += 1
                continue
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
            shape = gray.shape
            mask = AGG._mask(gray.shape, boxes.get(index, []), ratio * width / size[0])
            if previous is not None:
                steps.append(pool.submit(_step, previous[0], gray, previous[1]))
                starts.append(previous[2])
            previous, index = (gray, mask, index), index + 1
    capture.release()

    rows = np.asarray([s.result() for s in steps], dtype=np.float64).reshape(
        -1, len(FIT) + 1
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        start=np.asarray(starts, dtype=np.int64),
        **{name: rows[:, i] for i, name in enumerate((*FIT, "mask"))},
        fps=float(segmentation["fps"]),
        stride=stride,
        width=shape[1],
        height=shape[0],
    )
    return f"{out.parent.name}/{out.name}: {len(starts)} pairs"


def pairs(collections: list[str], stride: int, workers: int) -> None:
    jobs: list[tuple[Path, Path, int]] = []
    for name in collections:
        folder = ROOT / "data" / "processed" / COLLECTIONS[name]
        for root in sorted(folder.iterdir()):
            out = (
                CACHE
                / (name if stride == 1 else f"{name}_s{stride}")
                / f"{root.name}.npz"
            )
            if (root / "json" / "aggregated.json").exists() and not out.exists():
                jobs.append((root, out, stride))
    print(f"{len(jobs)} videos to fit at stride {stride}")
    with ProcessPoolExecutor(workers) as pool:
        for done, line in enumerate(pool.map(_pairs, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)


def _load(path: Path) -> dict[str, Any]:
    with np.load(path) as found:
        return {k: found[k] for k in found.files}


def reproduce(collections: list[str]) -> dict[str, Any]:
    """09's own rule on the cached fits must give every segment's stored camera."""
    out: dict[str, Any] = {}
    for name in collections:
        shots = mismatched = 0
        examples: list[str] = []
        for path in sorted((CACHE / name).glob("*.npz")):
            fits = _load(path)
            root = ROOT / "data" / "processed" / COLLECTIONS[name] / path.stem
            motions = [
                None if math.isnan(x) else (x, y, s)
                for x, y, s in zip(fits["tx"], fits["ty"], fits["scale"])
            ]
            for shot in _read(root, "aggregated.json")["shots"]:
                first, last = int(shot["start_frame"]), int(shot["end_frame"])
                again = AGG._camera(motions[first:last], float(fits["fps"]))
                shots += 1
                if again != shot["camera"]:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append(f"{path.stem} segment {shot['index']}")
        out[name] = {"segments": shots, "mismatched": mismatched, "examples": examples}
    return out


# Scoring rules on AVE: fit on scenes 51-100, report scenes 1-50

LABELS = ROOT / "data" / "raw" / "ave" / "annotations.json"
TEST = ROOT / "data" / "processed" / "ave"  # scenes 1-50, the reported half
IOU = 0.5  # evaluate_ave's floor
TASKS = {  # AVE's values as evaluate_ave maps them, and the classes in its order
    "one": (MOTION, ("static", "handheld", "pan", "tilt", "zoom")),
    "rig": (RIG, ("steady", "handheld")),
    "move": (MOVE, ("none", "pan", "tilt", "zoom")),
}
FEATURES = (
    "n",
    "pan",
    "tilt",
    "zoom",
    "pan_peak",
    "tilt_peak",
    "zoom_peak",
    "jitter_std",
    "jitter_trend",
    "jitter_mad",
    "snr_x",
    "snr_y",
    "snr_z",
    "fps",
    "aspect",
)
INLIERS = (
    0.0,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)  # share of tracked corners RANSAC must keep
SPIKE = 5  # pairs in the median filter that removes one- or two-pair jumps
STILLS = np.geomspace(1e-4, 0.3, 70)
RATIOS = {"snr": np.geomspace(0.05, 30.0, 60), "strength": np.geomspace(1.0, 300.0, 60)}
FOLDS = 5
BOOT = 2000
SEED = 0
CURRENT: dict[str, Any] = {
    "shake": AGG.SHAKE,
    "still_pan": AGG.STILL,
    "still_zoom": AGG.STILL,
    "ratio": math.inf,
    "promote": "",  # snr or strength: what lets a move outrank a handheld rig in the one label
    "inliers": 0.0,
    "tilt_height": False,
    "jitter": "std",  # std, trend or mad
    "move": "net",  # net over the shot, or peak 1 s window
    "spikes": False,
    "fallback": True,  # fit the whole frame when the masked fit fails
    "power": 0.0,  # shake scales with fps^power
}
# each is fitted the same way, so the refit baseline is the first row's structure
CANDIDATES: dict[str, dict[str, Any]] = {
    "refit": {},
    "A zoom threshold": {"split": True},
    "A tilt in heights": {"tilt_height": True},
    "B move outranks rig, steadiness": {"promote": "snr"},
    "B move outranks rig, strength": {"promote": "strength"},
    "C move in 1 s windows": {"move": "peak"},
    "D inlier floor": {"filter": True},
    "E jitter about a trend": {"jitter": "trend"},
    "G1 jitter from the median spread": {"jitter": "mad"},
    "G2 no whole-frame fallback": {"fallback": False},
    "G4 spikes removed": {"spikes": True},
}
SETTINGS = ("tilt_height", "promote", "jitter", "move", "spikes", "fallback", "power")


def _slice(fits: dict[str, Any], first: int, last: int) -> dict[str, Array]:
    """The pairs whose two frames both lie inside [first, last]."""
    inside = (fits["start"] >= first) & (fits["start"] + int(fits["stride"]) <= last)
    return {k: fits[k][inside] for k in (*FIT, "mask")}


def _shots(
    collection: str, scenes: list[str], labels: dict[str, Any] | None, folder: str = ""
) -> list[dict[str, Any]]:
    """A row per segment: matched AVE shots with their labels, or every segment of an unlabelled video."""
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        fits = _load(CACHE / (folder or collection) / f"{scene}.npz")
        root = ROOT / "data" / "processed" / COLLECTIONS[collection] / scene
        segmentation = _read(root, "segmentation.json")
        pairs = (
            matched(labels[scene], segmentation, IOU)
            if labels is not None
            else [(None, s) for s in segmentation["shots"]]
        )
        fps = float(fits["fps"]) / int(fits["stride"])
        for label, segment in pairs:
            first, last = int(segment["start_frame"]), int(segment["end_frame"])
            truth = {}
            for task, (mapping, _) in TASKS.items():
                found = _values(label, "shot-motion", mapping) if label else set()
                truth[task] = next(iter(found)) if len(found) == 1 else ""
            rows.append(
                {
                    "scene": scene,
                    "video": segmentation["video"],
                    "first": first,
                    "last": last,
                    "segment": int(segment["index"]),
                    "seconds": (last - first + 1) / float(fits["fps"]),
                    "fps": fps,
                    "aspect": float(fits["width"]) / float(fits["height"]),
                    "truth": truth,
                    "fits": _slice(fits, first, last),
                }
            )
    return rows


def _residual(values: Array) -> float:
    """Spread about a quadratic in time: a pan that speeds up and slows down leaves little."""
    if len(values) < 4:
        return float(values.std())
    t = np.arange(len(values), dtype=np.float64)
    return float((values - np.polyval(np.polyfit(t, values, 2), t)).std())


def _mad(values: Array) -> float:
    """Spread about the median, scaled to a normal's sd: one jump barely moves it."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def _peak(values: Array, fps: float) -> float:
    """The strongest 1 s of movement, per second: a brief or out-and-back move is not averaged away."""
    width = max(2, round(fps))
    if len(values) <= width:
        return float(values.sum()) * fps / len(values)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    sums = cumulative[width:] - cumulative[:-width]
    return float(sums[int(np.argmax(np.abs(sums)))]) * fps / width


def _features(
    shots: list[dict[str, Any]], inliers: float, spikes: bool, fallback: bool
) -> dict[str, Array]:
    """What every rule reads, per shot; dropped pairs go first, then spikes if asked."""
    from scipy.ndimage import median_filter

    rows: list[list[float]] = []
    for shot in shots:
        f = shot["fits"]
        fps = shot["fps"]
        keep = ~np.isnan(f["tx"])
        if inliers:
            keep &= f["inliers"] >= inliers * f["tracked"]
        if not fallback:
            keep &= f["mask"] != 2.0
        n = int(keep.sum())
        if n < 2:
            rows.append([n] + [math.nan] * (len(FEATURES) - 3) + [fps, shot["aspect"]])
            continue
        x, y, z = (f[k][keep] for k in ("tx", "ty", "scale"))
        if spikes:
            x, y, z = (median_filter(v, size=SPIKE, mode="nearest") for v in (x, y, z))
        seconds = n / fps
        snr = [
            abs(float(v.mean())) / float(v.std()) if v.std() > 0 else math.inf
            for v in (x, y, z)
        ]
        rows.append(
            [
                n,
                float(x.sum()) / seconds,
                float(y.sum()) / seconds,
                float(z.sum()) / seconds,
                _peak(x, fps),
                _peak(y, fps),
                _peak(z, fps),
                float(x.std() + y.std()),
                _residual(x) + _residual(y),
                _mad(x) + _mad(y),
                *snr,
                fps,
                shot["aspect"],
            ]
        )
    table = np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURES))
    return {name: table[:, i] for i, name in enumerate(FEATURES)}


def _label(f: dict[str, Array], p: dict[str, Any]) -> dict[str, Array]:
    """09's rule with the candidate changes; "" is an abstention, as 09's unknown."""
    known = f["n"] >= 2
    jitter = f[f"jitter_{p['jitter']}"] * f["fps"] ** p["power"]
    rig = np.where(jitter > p["shake"], "handheld", "steady")
    suffix = "_peak" if p["move"] == "peak" else ""
    tilt = f["tilt" + suffix] * (f["aspect"] if p["tilt_height"] else 1.0)
    a = np.abs(f["pan" + suffix]) / p["still_pan"]
    b = np.abs(tilt) / p["still_pan"]
    c = np.abs(f["zoom" + suffix]) / p["still_zoom"]
    with np.errstate(invalid="ignore"):
        strength = np.maximum(np.maximum(a, b), c)
        move = np.select(
            [strength < 1, c >= np.maximum(a, b), a >= b],
            ["none", "zoom", "pan"],
            "tilt",
        )
        signal = (
            np.select(
                [move == "pan", move == "tilt", move == "zoom"],
                [f["snr_x"], f["snr_y"], f["snr_z"]],
                0.0,
            )
            if p["promote"] == "snr"
            else strength
        )
        promote = (rig == "handheld") & (move != "none") & (signal >= p["ratio"])
    one = np.where(
        rig == "handheld", "handheld", np.where(move == "none", "static", move)
    )
    one = np.where(promote, move, one)
    return {
        "one": np.where(known, one, ""),
        "rig": np.where(known, rig, ""),
        "move": np.where(known, move, ""),
    }


def _truth(shots: list[dict[str, Any]], task: str) -> Array:
    return np.asarray([s["truth"][task] for s in shots])


def _balanced(
    truth: Array, pred: Array, classes: tuple[str, ...], weight: Array | None = None
) -> tuple[float, float]:
    """evaluate_ave's balanced accuracy and accuracy, over single-label shots the rule answered."""
    w = np.ones(len(truth)) if weight is None else weight
    answered = (truth != "") & (pred != "")
    total = float(w[answered].sum())
    if not total:
        return math.nan, math.nan
    right = answered & (truth == pred)
    recalls = [
        float(w[right & (truth == c)].sum()) / float(w[answered & (truth == c)].sum())
        for c in classes
        if w[answered & (truth == c)].sum() > 0
    ]
    return float(np.mean(recalls)), float(w[right].sum()) / total


def _best(values: Any, scores: list[float]) -> Any:
    """The median of the values that reach the best score, so a plateau is not cut at its edge."""
    top = np.nanmax(scores)
    tied = [v for v, s in zip(values, scores) if s >= top - 1e-12]
    return tied[len(tied) // 2]


def _tune(
    shots: list[dict[str, Any]],
    change: dict[str, Any],
    cache: dict[Any, dict[str, Array]],
) -> dict[str, Any]:
    """Each number on its own question: shake on the rig, the stills on the move, the ratio and the
    inlier floor on the one label."""
    truths = {task: _truth(shots, task) for task in TASKS}

    def score(f: dict[str, Array], p: dict[str, Any], task: str) -> float:
        return _balanced(truths[task], _label(f, p)[task], TASKS[task][1])[0]

    found: list[tuple[float, dict[str, Any]]] = []
    for inliers in INLIERS if change.get("filter") else (0.0,):
        p = {
            **CURRENT,
            "inliers": inliers,
            **{k: change[k] for k in SETTINGS if k in change},
        }
        key = (inliers, p["spikes"], p["fallback"])
        if key not in cache:
            cache[key] = _features(shots, *key)
        f = cache[key]
        jitter = f[f"jitter_{p['jitter']}"] * f["fps"] ** p["power"]
        grid = np.unique(np.nanquantile(jitter, np.linspace(0.005, 0.995, 200)))
        p["shake"] = float(
            _best(grid, [score(f, {**p, "shake": s}, "rig") for s in grid])
        )
        if change.get("split"):
            pairs = [(a, b) for a in STILLS for b in STILLS]
            a, b = _best(
                pairs,
                [
                    score(f, {**p, "still_pan": a, "still_zoom": b}, "move")
                    for a, b in pairs
                ],
            )
            p["still_pan"], p["still_zoom"] = float(a), float(b)
        else:
            s = _best(
                STILLS,
                [
                    score(f, {**p, "still_pan": s, "still_zoom": s}, "move")
                    for s in STILLS
                ],
            )
            p["still_pan"] = p["still_zoom"] = float(s)
        if p["promote"]:
            grid = [*RATIOS[p["promote"]], math.inf]
            p["ratio"] = float(
                _best(grid, [score(f, {**p, "ratio": r}, "one") for r in grid])
            )
        found.append((score(f, p, "one"), p))
    return max(found, key=lambda x: x[0])[1]


def _predict(shots: list[dict[str, Any]], p: dict[str, Any]) -> dict[str, Array]:
    return _label(_features(shots, p["inliers"], p["spikes"], p["fallback"]), p)


def _cross(shots: list[dict[str, Any]], change: dict[str, Any]) -> dict[str, float]:
    """5-fold over the fitting scenes: what a change gains out of sample, before the test half is seen."""
    scenes = sorted({s["scene"] for s in shots})
    order = np.random.default_rng(SEED).permutation(len(scenes))
    folds = [{scenes[i] for i in order[k::FOLDS]} for k in range(FOLDS)]
    pred = {task: np.full(len(shots), "", dtype=object) for task in TASKS}
    for fold in folds:
        train = [s for s in shots if s["scene"] not in fold]
        held = [i for i, s in enumerate(shots) if s["scene"] in fold]
        p = _tune(train, change, {})
        found = _predict([shots[i] for i in held], p)
        for task in TASKS:
            pred[task][held] = found[task]
    return {
        task: round(
            _balanced(_truth(shots, task), pred[task].astype(str), classes)[0], 4
        )
        for task, (_, classes) in TASKS.items()
    }


def _scores(shots: list[dict[str, Any]], p: dict[str, Any]) -> dict[str, Any]:
    pred = _predict(shots, p)
    out: dict[str, Any] = {}
    for task, (_, classes) in TASKS.items():
        rows = [
            ({s["truth"][task]} if s["truth"][task] else set(), str(v) or None)
            for s, v in zip(shots, pred[task])
        ]
        found = _categorical(rows, classes, False)
        out[task] = {
            k: found.get(k)
            for k in (
                "n",
                "answered",
                "accuracy",
                "balanced_accuracy",
                "majority_baseline",
                "per_class",
            )
        }
    return out


def _paired(
    shots: list[dict[str, Any]], p: dict[str, Any], base: dict[str, Any]
) -> dict[str, Any]:
    """Bootstrap over whole test scenes, the same resample for both rules: the gain and its 95% interval."""
    scenes = sorted({s["scene"] for s in shots})
    index = np.asarray([scenes.index(s["scene"]) for s in shots])
    new, old = _predict(shots, p), _predict(shots, base)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(scenes), (BOOT, len(scenes)))
    out: dict[str, Any] = {}
    for task, (_, classes) in TASKS.items():
        truth = _truth(shots, task)
        gains = []
        for row in picks:
            w = np.bincount(row, minlength=len(scenes))[index].astype(np.float64)
            gains.append(
                _balanced(truth, new[task], classes, w)[0]
                - _balanced(truth, old[task], classes, w)[0]
            )
        gain = (
            _balanced(truth, new[task], classes)[0]
            - _balanced(truth, old[task], classes)[0]
        )
        low, high = np.nanpercentile(gains, [2.5, 97.5])
        out[task] = {
            "gain": round(gain, 4),
            "ci": [round(float(low), 4), round(float(high), 4)],
            "clear": bool(low > 0 or high < 0),
        }
    return out


def _shares(
    shots: list[dict[str, Any]], p: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Each video's share of running time per one label, for videos without labels."""
    pred = _predict(shots, p)["one"]
    out: dict[str, dict[str, float]] = {}
    for shot, label in zip(shots, pred):
        entry = out.setdefault(shot["scene"], {})
        key = str(label) or "unknown"
        entry[key] = entry.get(key, 0.0) + shot["seconds"]
    for scene, entry in out.items():
        total = sum(entry.values())
        out[scene] = {
            k: round(v / total, 3)
            for k, v in sorted(entry.items(), key=lambda kv: -kv[1])
        }
    return out


def _fps_check(fit: list[dict[str, Any]], scenes: list[str]) -> dict[str, Any]:
    """The rig at 24 fps against every second frame (12 fps): a rule that does not depend on frame rate
    gives the same answer. Shake is scaled by fps^power; AVE's 24 fps cannot choose the power."""
    full = _shots("ave", scenes, None)
    half = _shots("ave", scenes, None, folder="ave_s2")
    out: dict[str, Any] = {}
    for power in (0.0, 0.5, 1.0):
        p = _tune(fit, {"power": power}, {})
        a, b = _predict(full, p)["rig"], _predict(half, p)["rig"]
        both = (a != "") & (b != "")
        out[str(power)] = {
            "shake": round(p["shake"], 6),
            "same_rig_at_12_fps": round(float((a[both] == b[both]).mean()), 4),
            "handheld_at_24": round(float((a[both] == "handheld").mean()), 4),
            "handheld_at_12": round(float((b[both] == "handheld").mean()), 4),
        }
    return out


def _rounded(p: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v)
        for k, v in p.items()
    }


def _split() -> tuple[dict[str, Any], list[str], list[str]]:
    """AVE's labels, the fitting scenes (51-100) and the reported ones (1-50, in data/processed/ave)."""
    labels: dict[str, Any] = json.loads(LABELS.read_text(encoding="utf-8"))
    scenes = sorted(p.stem for p in (CACHE / "ave").glob("*.npz") if p.stem in labels)
    test_names = {d.name for d in TEST.iterdir() if (d / "identikit.json").exists()}
    return (
        labels,
        [s for s in scenes if s not in test_names],
        [s for s in scenes if s in test_names],
    )


def evaluate(out: Path) -> None:
    labels, fit_scenes, test_scenes = _split()
    scenes = fit_scenes + test_scenes
    fit = _shots("ave", fit_scenes, labels)
    test = _shots("ave", test_scenes, labels)
    tests = _shots(
        "tests", sorted(p.stem for p in (CACHE / "tests").glob("*.npz")), None
    )
    reels = _shots(
        "reels", sorted(p.stem for p in (CACHE / "reels").glob("*.npz")), None
    )
    print(
        f"fit {len(fit_scenes)} scenes, {len(fit)} shots; test {len(test_scenes)} scenes, {len(test)} shots"
    )

    found: dict[str, Any] = {}
    params: dict[str, dict[str, Any]] = {"current": dict(CURRENT)}
    for name, change in CANDIDATES.items():
        params[name] = _tune(fit, change, {})
        found[name] = {"cross_validated_on_fit": _cross(fit, change)}
        print(
            f"{name:34} cv {found[name]['cross_validated_on_fit']}  {_rounded(params[name])}",
            flush=True,
        )

    # every change whose cross-validated one label beats the refit's; of two settings of one key, the better
    base_cv = found["refit"]["cross_validated_on_fit"]["one"]
    gaining = {
        name: found[name]["cross_validated_on_fit"]["one"]
        for name in CANDIDATES
        if name != "refit" and found[name]["cross_validated_on_fit"]["one"] > base_cv
    }
    combined: dict[str, Any] = {}
    for name in sorted(gaining, key=lambda n: gaining[n]):
        combined |= CANDIDATES[name]
    params["combined"] = _tune(fit, combined, {})
    found["combined"] = {
        "changes": combined,
        "cross_validated_on_fit": _cross(fit, combined),
    }
    print(
        f"{'combined':34} {combined} cv {found['combined']['cross_validated_on_fit']}",
        flush=True,
    )

    rules: dict[str, Any] = {}
    for name, p in params.items():
        rules[name] = {
            **found.get(name, {}),
            "params": _rounded(p),
            "test": _scores(test, p),
            "gain_over_refit_on_test": _paired(test, p, params["refit"])
            if name != "refit"
            else None,
            "tests_clips": _shares(tests, p),
            "reels_share": {
                k: round(v, 3) for k, v in _pooled(_shares(reels, p), reels).items()
            },
        }
    report = {
        "fit_scenes": len(fit_scenes),
        "test_scenes": len(test_scenes),
        "fit_shots": len(fit),
        "test_shots": len(test),
        "objective": "one label: balanced accuracy (AVE's average per-class accuracy); shake fitted on the rig's, "
        "stills on the move's, ratio and inlier floor on the one label's",
        "combined_rule": "every change whose 5-fold cross-validated one-label balanced accuracy on the fit half "
        "beats the refit's",
        "reproduce": reproduce(["ave", "tests", "reels"]),
        "rules": rules,
        "fps_check": _fps_check(fit, scenes) if (CACHE / "ave_s2").exists() else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, rule in rules.items():
        t = rule["test"]
        gain = rule["gain_over_refit_on_test"] or {}
        print(
            f"{name:34} one {t['one']['balanced_accuracy']} ({t['one']['accuracy']})  "
            f"rig {t['rig']['balanced_accuracy']}  move {t['move']['balanced_accuracy']} ({t['move']['accuracy']})  "
            f"gain {gain.get('one')}"
        )
    print(f"-> {out}")


def _pooled(
    shares: dict[str, dict[str, float]], shots: list[dict[str, Any]]
) -> dict[str, float]:
    """Shares over a whole collection, each video weighted by its running time."""
    seconds: dict[str, float] = {}
    for s in shots:
        seconds[s["scene"]] = seconds.get(s["scene"], 0.0) + s["seconds"]
    total = sum(seconds.values())
    out: dict[str, float] = {}
    for scene, entry in shares.items():
        for k, v in entry.items():
            out[k] = out.get(k, 0.0) + v * seconds[scene] / total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


SHEET_ROWS = 8


def _frame(video: str, index: int) -> Array:
    capture = cv2.VideoCapture(str(ROOT / video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return np.zeros((90, 160, 3), dtype=np.uint8)
    small = cv2.resize(
        frame,
        (320, round(320 * frame.shape[0] / frame.shape[1])),
        interpolation=cv2.INTER_AREA,
    )
    out: Array = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return out


def _quality(fits: dict[str, Array]) -> str:
    ok = ~np.isnan(fits["tx"])
    if not ok.any():
        return "no fit"
    whole = float(np.isin(fits["mask"][ok], (0.0, 2.0)).mean())
    kept = float(np.median(fits["inliers"][ok] / fits["tracked"][ok]))
    return f"whole-frame {whole:.0%} inliers {kept:.0%}"


def _draw(rows: list[tuple[dict[str, Any], str]], path: Path) -> None:
    """A row per shot: first, middle and last frame, and the summed motion over the shot."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(18, 2.9 * len(rows)), dpi=80)
    for row, (shot, caption) in enumerate(rows):
        frames = (shot["first"], (shot["first"] + shot["last"]) // 2, shot["last"])
        for col, index in enumerate(frames):
            ax = fig.add_subplot(len(rows), 4, row * 4 + col + 1)
            ax.imshow(_frame(shot["video"], index))
            ax.set_axis_off()
            ax.set_title(
                f"{shot['scene'][:20]} seg {shot['segment']} frame {index}", fontsize=8
            )
        ax = fig.add_subplot(len(rows), 4, row * 4 + 4)
        fits = shot["fits"]
        ok = ~np.isnan(fits["tx"])
        for key, colour in (
            ("tx", "tab:blue"),
            ("ty", "tab:orange"),
            ("scale", "tab:green"),
        ):
            ax.plot(np.cumsum(np.where(ok, fits[key], 0.0)), color=colour, label=key)
        ax.set_title(caption, fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path)


def sheets(folder: Path) -> None:
    """The refit rule's errors on the fitting scenes, drawn. The reported scenes are never drawn."""
    labels, fit_scenes, _ = _split()
    fit = _shots("ave", fit_scenes, labels)
    p = _tune(fit, {}, {})
    f = _features(fit, 0.0, False, True)
    pred = _label(f, p)
    t = {task: _truth(fit, task) for task in TASKS}
    moves = ("pan", "tilt", "zoom")
    categories = {
        "moves_read_as_none": np.isin(t["move"], moves) & (pred["move"] == "none"),
        "locked_read_as_zoom": (t["move"] == "none") & (pred["move"] == "zoom"),
        "locked_read_as_pan_or_tilt": (t["move"] == "none")
        & np.isin(pred["move"], ("pan", "tilt")),
        "moves_read_as_handheld": np.isin(t["one"], moves)
        & (pred["one"] == "handheld"),
        "locked_read_as_handheld": (t["rig"] == "steady") & (pred["rig"] == "handheld"),
        "handheld_read_as_steady": (t["rig"] == "handheld") & (pred["rig"] == "steady"),
    }
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"refit rule on the fitting scenes: {_rounded(p)}")
    for name, chosen in categories.items():
        found = np.flatnonzero(chosen)
        print(f"{name}: {len(found)} shots")
        if not len(found):
            continue
        picks = sorted(
            rng.choice(found, min(SHEET_ROWS, len(found)), replace=False).tolist()
        )
        rows = [
            (
                fit[i],
                f"AVE {fit[i]['truth']['one'] or fit[i]['truth']['move'] or fit[i]['truth']['rig']} | "
                f"read {pred['one'][i]}, move {pred['move'][i]}\n"
                f"pan {f['pan'][i]:+.3f} tilt {f['tilt'][i]:+.3f} zoom {f['zoom'][i]:+.3f} "
                f"jitter {f['jitter_std'][i]:.5f} {_quality(fit[i]['fits'])}",
            )
            for i in picks
        ]
        _draw(rows, folder / f"{name}.png")


# The rules compared on the test clips, which have no labels: each segment drawn with every rule's answer
CLIP_RULES: dict[str, dict[str, Any] | None] = {
    "current": None,
    "B+G1+F": {"promote": "snr", "jitter": "mad", "power": 0.5},
    "fixed in advance+F": {
        "promote": "snr",
        "jitter": "mad",
        "filter": True,
        "fallback": False,
        "power": 0.5,
    },
}
CLIP_ROWS = 12  # segments per page


def clips(folder: Path) -> None:
    labels, fit_scenes, test_scenes = _split()
    fit = _shots("ave", fit_scenes, labels)
    test = _shots("ave", test_scenes, labels)
    names = sorted(p.stem for p in (CACHE / "tests").glob("*.npz"))
    shots = _shots("tests", names, None)
    params = {
        name: dict(CURRENT) if change is None else _tune(fit, change, {})
        for name, change in CLIP_RULES.items()
    }
    preds = {name: _predict(shots, p) for name, p in params.items()}
    table: list[dict[str, Any]] = []
    for i, shot in enumerate(shots):
        table.append(
            {
                "clip": shot["scene"],
                "segment": shot["segment"],
                "frames": [shot["first"], shot["last"]],
                "seconds": round(shot["seconds"], 2),
                **{
                    name: [str(preds[name][task][i]) or "unknown" for task in TASKS]
                    for name in params
                },
            }
        )
    report = {
        "rules": {name: _rounded(p) for name, p in params.items()},
        "columns": "per rule: one label, rig, move",
        "ave_test": {
            name: {
                task: (s["balanced_accuracy"], s["accuracy"])
                for task, s in _scores(test, p).items()
            }
            for name, p in params.items()
        },
        "clip_shares": {name: _shares(shots, p) for name, p in params.items()},
        "segments": table,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "clips.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    f = _features(shots, 0.0, False, True)
    for clip in names:
        rows = []
        for i, shot in enumerate(shots):
            if shot["scene"] != clip:
                continue
            answers = "\n".join(
                f"{name}: {preds[name]['one'][i] or 'unknown'} (rig {preds[name]['rig'][i] or '-'}, "
                f"move {preds[name]['move'][i] or '-'})"
                for name in params
            )
            rows.append(
                (
                    shot,
                    f"{shot['seconds']:.1f} s  pan {f['pan'][i]:+.3f} tilt {f['tilt'][i]:+.3f} "
                    f"zoom {f['zoom'][i]:+.3f} {_quality(shot['fits'])}\n{answers}",
                )
            )
        for page, start in enumerate(range(0, len(rows), CLIP_ROWS)):
            _draw(rows[start : start + CLIP_ROWS], folder / f"{clip}_{page}.png")
    for name, p in params.items():
        print(f"{name}: {_rounded(p)}")
        print(f"   AVE scenes 1-50 (balanced, accuracy): {report['ave_test'][name]}")
    print(f"-> {folder}")


def main() -> None:
    p = argparse.ArgumentParser(description="09's camera rule against AVE's labels")
    sub = p.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("pairs", help="fit every frame pair once and cache it")
    cache.add_argument(
        "--collections", nargs="+", choices=list(COLLECTIONS), default=list(COLLECTIONS)
    )
    cache.add_argument(
        "--stride", type=int, default=1, help="frames between a pair's two frames"
    )
    cache.add_argument("--workers", type=int, default=WORKERS)
    check = sub.add_parser(
        "reproduce", help="09's rule on the cache against the stored cameras"
    )
    check.add_argument(
        "--collections", nargs="+", choices=list(COLLECTIONS), default=list(COLLECTIONS)
    )
    draw = sub.add_parser(
        "sheets",
        help="the refit rule's errors on the fitting scenes, as contact sheets",
    )
    draw.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "camera" / "sheets"
    )
    score = sub.add_parser(
        "evaluate", help="fit every rule on scenes 51-100, report scenes 1-50"
    )
    score.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "camera" / "camera.json",
    )
    compare = sub.add_parser(
        "clips", help="the test clips' segments under each rule in CLIP_RULES, drawn"
    )
    compare.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "camera" / "clips"
    )
    args = p.parse_args()
    if args.command == "pairs":
        pairs(args.collections, args.stride, args.workers)
    elif args.command == "reproduce":
        print(json.dumps(reproduce(args.collections), indent=2))
    elif args.command == "sheets":
        sheets(args.out)
    elif args.command == "clips":
        clips(args.out)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
