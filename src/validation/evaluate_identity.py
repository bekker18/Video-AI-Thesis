"""08's identity rule on VPCD (Brown et al., ICCVW 2021), beside MuHPC reimplemented without voice:
python3 src/validation/evaluate_identity.py --out <report>"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import load_stage

IDENTITY: Any = load_stage("08-identity")

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

RAW = ROOT / "data" / "raw" / "vpcd" / "VPCD_Release1"
CACHE = ROOT / "data" / "validation" / "identity" / "vpcd_tracks.npz"
SERIES = ("TBBT", "Buffy", "Sherlock", "Friends", "Hidden_Figures", "About_Last_Night")
TUNING = {f"Friends/episode0{i}" for i in range(1, 6)}  # VPCD's validation set
MIN_AREA = 1600  # 08's --identity-min-area
# 08's thresholds, and the share of different-people pairs passing them on our videos (below, for the veto)
RATES = {
    "face_within": 0.0888,
    "face_across": 0.0445,
    "body_within": 0.2577,
    "body_across": 0.3806,
    "face_veto": 0.9194,
}
OURS = {
    "face_within": 0.311,
    "face_across": 0.449,
    "body_within": 0.575,
    "body_across": 0.512,
    "face_veto": 0.331,
}
DUPLICATE = 0.5  # IoU of one person boxed twice
TIGHT, BACK, RATIO = 0.48, 0.4, 0.9  # MuHPC's settings
BOOT, SEED = 2000, 0

# VPCD Table 2: WCP, NMI, CP, CR
PUBLISHED = {
    "MuHPC, face + body + voice": {
        "TBBT": (96.9, 92.8, 80.4, 79.6),
        "Buffy": (85.8, 76.4, 68.4, 67.2),
        "Sherlock": (84.8, 60.0, 55.2, 57.2),
        "Friends": (90.8, 83.1, 87.7, 86.6),
        "Hidden_Figures": (77.6, 70.3, 59.0, 52.0),
        "About_Last_Night": (96.0, 90.2, 98.3, 86.4),
    },
    "MuHPC, face + body": {
        "TBBT": (96.9, 92.8, 80.4, 79.6),
        "Buffy": (85.7, 75.6, 68.1, 67.9),
        "Sherlock": (84.1, 52.9, 51.7, 54.3),
        "Friends": (89.5, 81.3, 84.6, 82.4),
        "Hidden_Figures": (77.6, 70.3, 59.0, 52.0),
        "About_Last_Night": (95.7, 89.4, 98.2, 86.3),
    },
    "B-C1C, face + body": {
        "TBBT": (87.7, 69.2, 39.4, 50.6),
        "Buffy": (73.6, 58.2, 34.6, 41.6),
        "Sherlock": (77.7, 41.6, 29.3, 43.6),
        "Friends": (85.3, 77.1, 69.5, 70.8),
        "Hidden_Figures": (76.2, 69.8, 55.2, 50.3),
        "About_Last_Night": (94.4, 85.8, 68.0, 76.8),
    },
    "B-ReID, body": {
        "TBBT": (80.5, 69.7, 49.6, 55.0),
        "Buffy": (65.0, 60.9, 52.7, 46.8),
        "Sherlock": (61.2, 28.9, 43.6, 44.3),
        "Friends": (70.9, 60.4, 71.0, 56.3),
        "Hidden_Figures": (32.6, 23.4, 36.8, 19.6),
        "About_Last_Night": (41.0, 14.1, 37.4, 32.6),
    },
}
# RAD-Net Table II: WCP, NMI, CF
RADNET = {
    "TBBT": (96.62, 93.12, 77.33),
    "Buffy": (90.27, 76.56, 77.33),
    "Sherlock": (93.59, 61.84, 68.74),
    "Friends": (93.61, 85.07, 86.66),
    "Hidden_Figures": (92.28, 78.69, 63.99),
}
# VPCD Table 3, faces per frame: WCP, NMI, CP, CR, clusters
PUBLISHED_FACES = {
    "TBBT": (99.4, 97.8, 87.8, 88.6, 168),
    "Buffy": (96.1, 92.8, 85.6, 85.5, 223),
    "Sherlock": (86.7, 60.3, 79.1, 71.2, 96),
    "Friends": (98.7, 94.9, 98.1, 94.0, 543),
}
EXACT = ("TBBT", "Buffy", "Hidden_Figures")  # release matches the paper


class _Safe(pickle.Unpickler):
    """No code from the pickle runs."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        return super().find_class(module, name)


def _rows(feats: Any) -> Array:
    rows = np.asarray(
        np.stack(feats) if isinstance(feats, list) else feats, dtype=np.float32
    )
    out: Array = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6)
    return out


def _describe(rows: Array, usable: Array) -> Array | None:
    entries = [{"row": i, "frame": i} for i in range(len(rows))]
    found: Array | None = IDENTITY._descriptor(
        entries, set(np.flatnonzero(usable).tolist()), rows
    )
    return found


def _area(boxes: Array) -> Array:
    out: Array = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    return out


def _iou(a: Array, b: Array) -> float:
    wide = min(a[2], b[2]) - max(a[0], b[0])
    tall = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or tall <= 0:
        return 0.0
    overlap = wide * tall
    return float(
        overlap
        / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    )


def _episode(ep: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, list[Any]] = {
        k: []
        for k in (
            "label",
            "back",
            "frames",
            "faces",
            "boxes",
            "face",
            "body",
            "face_crops",
            "body_crops",
            "mean_face",
            "mean_body",
            "face_detections",
        )
    }
    feats = {
        kind: {k: v for k, v in ep[f"{kind}_feats"].items() if not isinstance(v, dict)}
        for kind in ("face", "body", "back")
    }
    tracks = [(k, False) for k in sorted(set(ep["face"]) | set(ep["body"]))] + [
        (k, True) for k in sorted(ep["back"])
    ]
    for key, back in tracks:
        face_boxes = None if back else ep["face"].get(key)
        body_boxes = ep["back"][key] if back else ep["body"].get(key)
        body_rows = feats["back" if back else "body"].get(key)
        frames: set[int] = set()
        faces: dict[int, Array] = {}
        bodies: dict[int, Array] = {}
        for boxes, store in ((face_boxes, faces), (body_boxes, bodies)):
            if boxes is not None and len(boxes):
                for row in np.asarray(boxes, dtype=np.float64):
                    store[int(row[0])] = row[1:]
                frames |= set(store)
        face = mean_face = body = mean_body = None
        face_n = body_n = 0
        if face_boxes is not None and len(face_boxes):
            rows = _rows(feats["face"][key])
            usable = _area(np.asarray(face_boxes, dtype=np.float64)) >= MIN_AREA
            face, face_n = _describe(rows, usable), int(usable.sum())
            mean_face = _describe(rows, np.ones(len(rows), dtype=bool))
        if body_rows is not None and len(body_rows):
            rows = _rows(body_rows)
            boxes = (
                np.asarray(body_boxes, dtype=np.float64)
                if body_boxes is not None
                else np.zeros((0, 5))
            )
            # shared body tracks: boxes cover only this track
            usable = (
                _area(boxes) >= MIN_AREA
                if len(boxes) == len(rows)
                else np.ones(len(rows), dtype=bool)
            )
            body, body_n = _describe(rows, usable), int(usable.sum())
            mean_body = _describe(rows, np.ones(len(rows), dtype=bool))
        label = (ep["back_GT"] if back else ep["GT"])[key]
        for name, value in (
            ("label", str(label)),
            ("back", back),
            ("frames", sorted(frames)),
            ("faces", faces),
            ("boxes", bodies),
            ("face", face),
            ("body", body),
            ("face_crops", face_n),
            ("body_crops", body_n),
            ("mean_face", mean_face),
            ("mean_body", mean_body),
            ("face_detections", 0 if face_boxes is None else len(face_boxes)),
        ):
            out[name].append(value)
    spans = [(min(f), max(f)) if f else (0, -1) for f in out["frames"]]
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    pairs, double = [], []
    sets = [set(f) for f in out["frames"]]
    for n, i in enumerate(order):
        for j in order[n + 1 :]:
            if spans[j][0] > spans[i][1]:
                break
            shared = sets[i] & sets[j]
            if not shared:
                continue
            a, b = sorted((i, j))
            face = any(
                f in out["faces"][a]
                and f in out["faces"][b]
                and _iou(out["faces"][a][f], out["faces"][b][f]) > DUPLICATE
                for f in shared
            )
            both = [f for f in shared if f in out["boxes"][a] and f in out["boxes"][b]]
            body = (
                bool(both)
                and float(
                    np.median(
                        [_iou(out["boxes"][a][f], out["boxes"][b][f]) for f in both]
                    )
                )
                > DUPLICATE
            )
            pairs.append((a, b))
            double.append(face or body)
    out["pairs"], out["double"] = pairs, double
    return out


def _vectors(values: list[Array | None]) -> Array:
    out: Array = np.stack(
        [
            np.full(256, np.nan, dtype=np.float32)
            if v is None
            else v.astype(np.float32)
            for v in values
        ]
    )
    return out


def _build() -> None:
    store: dict[str, list[Any]] = {
        k: []
        for k in (
            "episode",
            "label",
            "back",
            "frames",
            "offsets",
            "face",
            "body",
            "face_crops",
            "body_crops",
            "mean_face",
            "mean_body",
            "face_detections",
            "pairs",
            "double",
        )
    }
    names: list[str] = []
    start = 0
    for series in SERIES:
        t = time.time()
        with (RAW / f"{series}.pk").open("rb") as f:
            data = _Safe(f, encoding="latin1").load()
        for name in sorted(data):
            ep = _episode(data[name])
            n = len(ep["label"])
            store["episode"].append(np.full(n, len(names)))
            names.append(f"{series}/{name}")
            for key in ("label", "back", "face_crops", "body_crops", "face_detections"):
                store[key].append(np.asarray(ep[key]))
            for key in ("face", "body", "mean_face", "mean_body"):
                store[key].append(_vectors(ep[key]))
            store["frames"].append(
                np.concatenate([np.asarray(f, dtype=np.int32) for f in ep["frames"]])
            )
            store["offsets"].append(
                np.cumsum([0] + [len(f) for f in ep["frames"]])[:-1]
                + sum(len(x) for x in store["frames"][:-1])
            )
            store["pairs"].append(
                np.asarray(ep["pairs"], dtype=np.int64).reshape(-1, 2) + start
            )
            store["double"].append(np.asarray(ep["double"], dtype=bool))
            start += n
        print(
            f"{series}: {len(data)} episodes cached in {time.time() - t:.0f} s",
            flush=True,
        )
        del data
    arrays = {k: np.concatenate(v) for k, v in store.items()}
    arrays["lengths"] = np.diff(np.append(arrays["offsets"], len(arrays["frames"])))
    arrays["names"] = np.asarray(names)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **arrays)


def _cosines(data: dict[str, Array], kind: str, pairs: Array) -> Array:
    a, b = data[kind][pairs[:, 0]], data[kind][pairs[:, 1]]
    keep = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    out: Array = np.einsum("ij,ij->i", a[keep], b[keep])
    return out


def _transfer(data: dict[str, Array]) -> dict[str, Any]:
    """08's thresholds at the same rates on VPCD's different-people pairs."""
    pairs = data["pairs"][~data["double"]]
    out: dict[str, Any] = {"rates": RATES, "ours": OURS}
    for kind in ("face", "body"):
        different = _cosines(data, kind, pairs)
        out[f"{kind}_pairs"] = len(different)
        for scope in ("within", "across"):
            out[f"{kind}_{scope}"] = round(
                float(np.quantile(different, 1 - RATES[f"{kind}_{scope}"])), 4
            )
        if kind == "face":
            out["face_veto"] = round(
                float(np.quantile(different, RATES["face_veto"])), 4
            )
    return out


def _segments(starts: Array, ends: Array) -> Array:
    """A cut wherever no track continues."""
    order = np.argsort(starts, kind="stable")
    segment = np.empty(len(starts), dtype=np.int64)
    current, reach = -1, -np.inf
    for i in order:
        if starts[i] > reach:
            current += 1
        reach = max(reach, ends[i])
        segment[i] = current
    return segment


def _run08(units: list[dict[str, Any]], limits: dict[str, Any]) -> Array:
    """08's own grouping after pass zero; abstained tracks stay alone."""
    active = [u for u in units if u["face"] is not None or u["body"] is not None]
    cfg = SimpleNamespace(**{f"identity_{k}": limits[k] for k in RATES if k != "face_veto"})
    final = IDENTITY._resolve(active, cfg)[0]
    pred = np.full(len(units), -1, dtype=np.int64)
    for c, unit in enumerate(final):
        for key in unit["keys"]:
            pred[key[1]] = c
    alone = pred < 0
    pred[alone] = len(final) + np.arange(int(alone.sum()))
    return pred


def _blocked(frames: list[set[int]]) -> Array:
    spans = [(min(f), max(f)) for f in frames]
    order = sorted(range(len(frames)), key=lambda i: spans[i][0])
    out = np.zeros((len(frames), len(frames)), dtype=bool)
    for n, i in enumerate(order):
        for j in order[n + 1 :]:
            if spans[j][0] > spans[i][1]:
                break
            if frames[i] & frames[j]:
                out[i, j] = out[j, i] = True
    return out


def _first_neighbours(features: Array, blocked: Array) -> Array:
    """MuHPC stage 1, as its equation (1) reads."""
    label = np.arange(len(features))
    while True:
        ids, label = np.unique(label, return_inverse=True)
        onehot = np.zeros((len(features), len(ids)))
        onehot[np.arange(len(features)), label] = 1
        means = onehot.T @ features
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-6)
        distance = 1 - means @ means.T
        distance[(onehot.T @ blocked @ onehot) > 0] = np.inf
        np.fill_diagonal(distance, np.inf)
        nearest = distance.argmin(axis=1)
        gap = distance[np.arange(len(ids)), nearest]
        parent = list(range(len(ids)))

        def root(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = False
        for a in np.flatnonzero(gap <= TIGHT):
            ra, rb = root(int(a)), root(int(nearest[a]))
            if ra != rb:
                parent[rb] = ra
                merged = True
        if not merged:
            return label
        label = np.array([root(c) for c in range(len(ids))])[label]


def _muhpc(
    face: Array, body: Array, back: Array, segment: Array, blocked: Array
) -> Array:
    """MuHPC without voice: stages 1 and 3."""
    people = np.flatnonzero(~back)
    label = np.full(len(back), -1, dtype=np.int64)
    label[people] = _first_neighbours(
        face[people], blocked[np.ix_(people, people)].astype(float)
    )
    has = np.flatnonzero(~np.isnan(body[:, 0]))
    distance = 1 - body[has] @ body[has].T
    np.fill_diagonal(distance, np.inf)
    two = np.sort(distance, axis=1)[:, :2]
    ratio = np.divide(two[:, 0], two[:, 1], out=np.zeros(len(has)), where=two[:, 1] > 0)
    distinct = ratio <= RATIO
    row = {u: r for r, u in enumerate(has.tolist())}
    faced = np.flatnonzero(distinct & ~back[has])
    nxt = int(label.max()) + 1
    for b in np.flatnonzero(back):
        chosen = -1
        if b in row and distinct[row[b]]:
            near = faced[np.abs(segment[has[faced]] - segment[b]) <= 1]
            gaps = distance[row[b], near]
            for k in np.argsort(gaps, kind="stable"):
                if gaps[k] > BACK:
                    break
                u = has[near[k]]
                if not blocked[b, label == label[u]].any():
                    chosen = int(label[u])
                    break
        if chosen >= 0:
            label[b] = chosen
        else:
            label[b], nxt = nxt, nxt + 1
    return label


def _scores(pred: Array, truth: Array, weight: Array | None = None) -> dict[str, float]:
    _, p = np.unique(pred, return_inverse=True)
    _, t = np.unique(truth, return_inverse=True)
    m = np.zeros((p.max() + 1, t.max() + 1))
    np.add.at(m, (p, t), np.ones(len(p)) if weight is None else weight.astype(float))
    joint = m / m.sum()
    pk, pc = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    mutual = float((joint[nz] * np.log(joint[nz] / np.outer(pk, pc)[nz])).sum())
    entropy = sum(-float((q[q > 0] * np.log(q[q > 0])).sum()) for q in (pk, pc))
    who, where = linear_sum_assignment(-m.T)
    cp, cr = np.zeros(m.shape[1]), np.zeros(m.shape[1])
    cp[who] = m.T[who, where] / m.sum(axis=1)[where]
    cr[who] = m.T[who, where] / m.sum(axis=0)[who]
    return {
        "wcp": 100 * float(m.max(axis=1).sum() / m.sum()),
        "nmi": 100 * (1.0 if entropy == 0 else 2 * mutual / entropy),
        "cp": 100 * float(cp.mean()),
        "cr": 100 * float(cr.mean()),
        "clusters": int(m.shape[0]),
        "identities": int(m.shape[1]),
    }


METRICS = ("wcp", "nmi", "cp", "cr")


def _average(per_episode: dict[str, dict[str, float]], keep: Any) -> dict[str, Any]:
    """Mean per series, then over series."""
    by_series: dict[str, list[dict[str, float]]] = {}
    for name, scores in per_episode.items():
        if keep(name):
            by_series.setdefault(name.split("/")[0], []).append(scores)
    if not by_series:
        return {}
    series = {
        s: {m: round(float(np.mean([e[m] for e in eps])), 1) for m in METRICS}
        | {
            "clusters": int(sum(e["clusters"] for e in eps)),
            "identities": int(sum(e["identities"] for e in eps)),
            "episodes": len(eps),
        }
        for s, eps in by_series.items()
    }
    mean = {
        m: round(float(np.mean([v[m] for v in series.values()])), 1) for m in METRICS
    }
    return {"series": series, "average": mean}


def _interval(
    per_episode: dict[str, dict[str, float]], keep: Any
) -> dict[str, list[float]]:
    """95% interval, resampling episodes within series."""
    by_series: dict[str, list[dict[str, float]]] = {}
    for name, scores in per_episode.items():
        if keep(name):
            by_series.setdefault(name.split("/")[0], []).append(scores)
    rng = np.random.default_rng(SEED)
    out: dict[str, list[float]] = {}
    if not by_series:
        return out
    for m in METRICS:
        values = [np.array([e[m] for e in eps]) for eps in by_series.values()]
        draws = [
            np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in values])
            for _ in range(BOOT)
        ]
        out[m] = [round(float(np.percentile(draws, q)), 1) for q in (2.5, 97.5)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rebuild", action="store_true", help="read VPCD's pickles again")
    p.add_argument("--series", nargs="*", default=list(SERIES), choices=SERIES)
    args = p.parse_args()
    if args.rebuild or not CACHE.exists():
        _build()
    data = dict(np.load(CACHE, allow_pickle=False))
    limits = _transfer(data)
    IDENTITY.FACE_VETO = limits["face_veto"]
    print(json.dumps(limits), flush=True)

    methods = ("08", "MuHPC face + body, ours")
    results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in methods}
    faces_only: dict[str, dict[str, float]] = {}
    shots: dict[str, dict[str, float]] = {}
    counts = {"tracks": 0, "abstained": 0, "face_descriptors": 0, "body_descriptors": 0}
    for e, name in enumerate(data["names"].tolist()):
        if name.split("/")[0] not in args.series:
            continue
        t = time.time()
        idx = np.flatnonzero(data["episode"] == e)
        frames = [
            set(data["frames"][o : o + n].tolist())
            for o, n in zip(data["offsets"][idx], data["lengths"][idx])
        ]
        starts = np.array([min(f) for f in frames])
        ends = np.array([max(f) for f in frames])
        segment = _segments(starts, ends)
        truth = data["label"][idx]
        units = []
        for i, g in enumerate(idx):
            face = (
                None
                if np.isnan(data["face"][g, 0])
                else data["face"][g].astype(np.float64)
            )
            body = (
                None
                if np.isnan(data["body"][g, 0])
                else data["body"][g].astype(np.float64)
            )
            units.append(
                {
                    "keys": [[int(segment[i]), i]],
                    "frames": frames[i],
                    "links": [],
                    "face": face,
                    "body": body,
                    "face_crops": int(data["face_crops"][g]),
                    "body_crops": int(data["body_crops"][g]),
                }
            )
        counts["tracks"] += len(units)
        counts["abstained"] += sum(
            u["face"] is None and u["body"] is None for u in units
        )
        counts["face_descriptors"] += sum(u["face"] is not None for u in units)
        counts["body_descriptors"] += sum(u["body"] is not None for u in units)
        blocked = _blocked(frames)
        back = data["back"][idx]
        results["08"][name] = _scores(_run08(units, limits), truth)
        mean_face, mean_body = (
            data["mean_face"][idx].astype(np.float64),
            data["mean_body"][idx].astype(np.float64),
        )
        results["MuHPC face + body, ours"][name] = _scores(
            _muhpc(mean_face, mean_body, back, segment, blocked), truth
        )
        people = np.flatnonzero(~back)
        stage1 = _first_neighbours(
            mean_face[people], blocked[np.ix_(people, people)].astype(float)
        )
        faces_only[name] = _scores(
            stage1, truth[people], data["face_detections"][idx][people]
        )
        span = int(ends.max() - starts.min() + 1)
        shots[name] = {
            "segments": int(segment.max() + 1),
            "frames_per_segment": round(span / (segment.max() + 1), 1),
        }
        print(
            f"{name}: {len(units)} tracks, {shots[name]['segments']} segments, "
            f"08 NMI {results['08'][name]['nmi']:.1f}, MuHPC NMI "
            f"{results['MuHPC face + body, ours'][name]['nmi']:.1f}, {time.time() - t:.0f} s",
            flush=True,
        )

    def tuning(n: str) -> bool:
        return n in TUNING

    def test(n: str) -> bool:
        return n not in TUNING

    by_series: dict[str, list[dict[str, float]]] = {}
    for n, v in shots.items():
        by_series.setdefault(n.split("/")[0], []).append(v)

    report: dict[str, Any] = {
        "about": __doc__,
        "settings": {
            "min_area": MIN_AREA,
            "frontality": "not in VPCD: off",
            "pass_zero": "off: one track per person per frame",
            "thresholds": limits,
            "muhpc": {"tight": TIGHT, "back": BACK, "ratio": RATIO},
        },
        "tracks": counts,
        "segments": {
            s: {
                "episodes": len(v),
                "segments_per_episode": round(
                    float(np.mean([x["segments"] for x in v]))
                ),
                "frames_per_segment": round(
                    float(np.mean([x["frames_per_segment"] for x in v])), 1
                ),
            }
            for s, v in by_series.items()
        },
        "results": {
            m: {
                "all_episodes": _average(r, lambda n: True),
                "test": _average(r, test) | {"interval_95": _interval(r, test)},
                "tuning": _average(r, tuning),
                "per_episode": r,
            }
            for m, r in results.items()
        },
        "reproduction": {
            "faces_only_per_frame": {
                s: {
                    "ours": _average(faces_only, lambda n, s=s: n.startswith(s + "/"))[
                        "series"
                    ].get(s),
                    "published": dict(
                        zip(("wcp", "nmi", "cp", "cr", "clusters"), PUBLISHED_FACES[s])
                    ),
                    "release_matches_paper": s in EXACT,
                }
                for s in PUBLISHED_FACES
                if any(n.startswith(s + "/") for n in faces_only)
            },
            "face_and_body": {
                s: {
                    "ours": _average(
                        results["MuHPC face + body, ours"],
                        lambda n, s=s: n.startswith(s + "/"),
                    )["series"][s],
                    "published": dict(zip(METRICS, PUBLISHED["MuHPC, face + body"][s])),
                    "release_matches_paper": s in EXACT[1:],
                }
                for s in args.series
            },
        },
        "published": PUBLISHED,
        "published_radnet_wcp_nmi_cf": RADNET,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for m in methods:
        r = report["results"][m]
        print(
            m,
            "all",
            r["all_episodes"]["average"],
            "test",
            r["test"]["average"],
            r["test"]["interval_95"],
        )
    print(json.dumps(report["reproduction"], indent=1))


if __name__ == "__main__":
    main()
