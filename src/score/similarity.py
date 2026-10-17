"""FRESCO-style distance between videos, from their identikits: one distance per field, calibrated over
the collection, averaged per group and level, for every pair; each video's nearest, and clusters.

python3 -m src.score.similarity [video ...] [--calibration scale|rank] [--clusters 4] [--explain A B]"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Kit: TypeAlias = dict[str, Any]
Mix: TypeAlias = dict[str, float]

LEVELS = ("plastic", "figurative", "enunciative")
QUANTILES = np.linspace(0.005, 0.995, 100)  # where two distributions of values are compared
SCALE_Q = 0.9  # --calibration scale: the pair distance at this quantile of a field maps to 1
UNBOUNDED = {"scalar", "samples"}  # kinds whose raw distance has units, not a 0-1 scale
CLUSTERS = 4  # k-medoids clusters, by default
RESTARTS = 20  # k-medoids runs, the cheapest kept
SEED = 0
TOP = 5  # nearest videos listed
HELD_BY = 3  # fields listed for what a cluster shares


def _mix(weights: dict[str, float]) -> Mix | None:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else None


def _by_time(pairs: list[tuple[Any, float]]) -> Mix | None:
    """Labels weighted by the seconds they hold; None labels are left out."""
    weights: dict[str, float] = {}
    for label, seconds in pairs:
        if label is not None:
            weights[str(label)] = weights.get(str(label), 0.0) + seconds
    return _mix(weights)


def _plastic_mean(key: str) -> Callable[[Kit], float | None]:
    def read(kit: Kit) -> float | None:
        stats = kit["plastic"]["video"].get(key)
        return float(stats["mean"]) if stats else None

    return read


def _medium(kit: Kit) -> Mix | None:
    medium = kit["plastic"]["video"].get("medium")
    return _mix(medium["distribution"]) if medium else None


def _histogram(kit: Kit) -> list[Mix] | None:
    found = kit["plastic"]["video"].get("colour_histogram")
    if not found:
        return None
    mixes = [_mix({str(i): float(v) for i, v in enumerate(found[c])}) for c in ("r", "g", "b")]
    return [m for m in mixes if m] or None


def _seconds(kit: Kit) -> dict[int, float]:
    return {int(s["segment"]): float(s["seconds"]) for s in kit["timeline"]["segments"]}


def _coverage(kit: Kit) -> Mix | None:
    seconds = _seconds(kit)
    weights: dict[str, float] = {}
    for s in kit["plastic"]["segments"]:
        for name, share in (s.get("coverage") or {}).items():
            key = str(name).strip()
            weights[key] = weights.get(key, 0.0) + float(share) * seconds[int(s["segment"])]
    total = sum(seconds.values())
    return {k: v / total for k, v in weights.items()} if total > 0 and weights else None


def _shot_lengths(kit: Kit) -> tuple[Array, Array] | None:
    lengths = [float(s["seconds"]) for s in kit["timeline"]["segments"] if float(s["seconds"]) > 0]
    if not lengths:
        return None
    return np.log(np.asarray(lengths)), np.ones(len(lengths))


def _video_set(key: str, value: str) -> Callable[[Kit], Mix | None]:
    def read(kit: Kit) -> Mix | None:
        found = (kit["figurative"].get("video") or {}).get(key)
        if found is None:
            return None
        return {str(x["label"]): float(x[value]) for x in found}

    return read


def _outdoor(kit: Kit) -> float | None:
    found = (kit["figurative"].get("video") or {}).get("indoor_outdoor")
    return float(found["outdoor"]) if found else None


def _animals(kit: Kit) -> Mix:
    duration = float(kit["summary"]["duration_s"]) or 1.0
    return {str(s["label"]): float(s["seconds"]) / duration for s in kit["figurative"]["subjects"]}


def _people_time(kit: Kit) -> float | None:
    seconds = sum(float(s["seconds"]) for s in kit["timeline"]["segments"])
    with_people = sum(float(s["seconds"]) for s in kit["timeline"]["segments"] if s["person_ids"])
    return with_people / seconds if seconds > 0 else None


def _crowd(kit: Kit) -> Mix | None:
    seconds = _seconds(kit)
    return _by_time([(s["crowdedness"], seconds[int(s["segment"])]) for s in kit["figurative"]["segments"]])


def _described(kit: Kit) -> list[dict[str, Any]]:
    return [p for p in kit["figurative"]["persons"] if p.get("status", "described") == "described"]


def _women(kit: Kit) -> float | None:
    gendered = [(p["gender"]["label"], float(p["seconds"])) for p in _described(kit) if p.get("gender")]
    total = sum(s for _, s in gendered)
    return sum(s for g, s in gendered if g == "female") / total if total > 0 else None


def _ages(kit: Kit) -> tuple[Array, Array] | None:
    found = [(float(p["age"]["mean"]), float(p["seconds"])) for p in _described(kit) if p.get("age")]
    found = [(a, s) for a, s in found if s > 0]
    if not found:
        return None
    return np.asarray([a for a, _ in found]), np.asarray([s for _, s in found])


def _emotion(kit: Kit) -> Mix | None:
    weights: dict[str, float] = {}
    for p in _described(kit):
        if p.get("emotion"):
            for name, share in p["emotion"]["distribution"].items():
                weights[name] = weights.get(name, 0.0) + float(share) * float(p["seconds"])
    return _mix(weights)


def _affect(axis: str) -> Callable[[Kit], float | None]:
    def read(kit: Kit) -> float | None:
        found = [(float(p[axis]["mean"]), float(p["seconds"])) for p in _described(kit) if p.get(axis)]
        total = sum(s for _, s in found)
        return sum(v * s for v, s in found) / total if total > 0 else None

    return read


def _shot_scale(kit: Kit) -> Mix | None:
    seconds = _seconds(kit)
    return _by_time(
        [((s.get("camera_distance") or {}).get("scale"), seconds[int(s["segment"])]) for s in kit["enunciative"]["segments"]]
    )


def _portrait(kit: Kit) -> float | None:
    seconds = _seconds(kit)
    framed = [(s["portrait_scene"], seconds[int(s["segment"])]) for s in kit["enunciative"]["segments"] if s.get("portrait_scene")]
    total = sum(t for _, t in framed)
    return sum(t for f, t in framed if f == "portrait") / total if total > 0 else None


def _camera(kit: Kit) -> Mix | None:
    return _mix({k: float(v) for k, v in kit["enunciative"]["camera"].items() if k != "unknown"})


def _per_sample(
    key: str, value: Callable[[dict[str, Any]], float], count: Callable[[dict[str, Any]], int] = lambda x: int(x["n"])
) -> Callable[[Kit], float | None]:
    """A person-level share or angle, pooled over every person by the samples behind it."""

    def read(kit: Kit) -> float | None:
        found = [(value(p[key]), count(p[key])) for p in kit["enunciative"]["persons"] if p.get(key)]
        total = sum(n for _, n in found)
        return sum(v * n for v, n in found) / total if total > 0 else None

    return read


def _looks(kit: Kit) -> float | None:
    relations = kit["summary"]["relations"]
    pairs = int(relations["co_present"])
    return int(relations["with_evidence"]) / pairs if pairs else None


# name, level, group, kind, reader. Perceived ethnicity is left out: it feeds nothing downstream.
FIELDS: tuple[tuple[str, str, str, str, Callable[[Kit], Any]], ...] = (
    ("medium", "plastic", "technical", "mix", _medium),
    ("colour histogram", "plastic", "chromatic", "histogram", _histogram),
    ("brightness", "plastic", "chromatic", "scalar", _plastic_mean("brightness")),
    ("saturation", "plastic", "chromatic", "scalar", _plastic_mean("saturation")),
    ("chroma", "plastic", "chromatic", "scalar", _plastic_mean("chroma")),
    ("warmth", "plastic", "chromatic", "scalar", _plastic_mean("warmth")),
    ("tint", "plastic", "chromatic", "scalar", _plastic_mean("tint")),
    ("contrast", "plastic", "eidetic", "scalar", _plastic_mean("contrast_rms")),
    ("edge density", "plastic", "eidetic", "scalar", _plastic_mean("edge_density")),
    ("symmetry", "plastic", "eidetic", "scalar", _plastic_mean("horizontal_symmetry")),
    ("region coverage", "plastic", "topological", "set", _coverage),
    ("shot lengths", "plastic", "rhythm", "samples", _shot_lengths),
    ("tags", "figurative", "general", "set", _video_set("tags", "share")),
    ("place", "figurative", "setting", "set", _video_set("place", "score")),
    ("outdoor", "figurative", "setting", "share", _outdoor),
    ("objects", "figurative", "content", "set", _video_set("objects", "share")),
    ("animals", "figurative", "content", "set", _animals),
    ("people on screen", "figurative", "people", "share", _people_time),
    ("crowdedness", "figurative", "people", "mix", _crowd),
    ("share of women", "figurative", "people", "share", _women),
    ("age", "figurative", "people", "samples", _ages),
    ("emotion", "figurative", "people", "mix", _emotion),
    ("valence", "figurative", "people", "scalar", _affect("valence")),
    ("arousal", "figurative", "people", "scalar", _affect("arousal")),
    ("shot scale", "enunciative", "watcher", "mix", _shot_scale),
    ("portrait share", "enunciative", "watcher", "share", _portrait),
    ("camera movement", "enunciative", "watcher", "mix", _camera),
    ("looks into the camera", "enunciative", "observer", "share", _per_sample("address", lambda a: float(a["share"]))),
    ("gaze in frame", "enunciative", "observer", "share", _per_sample("gaze", lambda g: float(g["in_frame_share"]))),
    ("head turn", "enunciative", "observer", "scalar", _per_sample("head_pose", lambda h: abs(float(h["yaw"]["mean"])), lambda h: int(h["yaw"]["n"]))),
    ("looks between people", "enunciative", "relations", "share", _looks),
)


LEVEL_OF = {name: level for name, level, _, _, _ in FIELDS}
KIND_OF = {name: kind for name, _, _, kind, _ in FIELDS}
LOGGED = {"shot lengths"}  # fields read as logarithms


def _blank(values: list[Any], out: Array) -> Array:
    """No distance where either video lacks the field."""
    missing = np.array([v is None for v in values])
    out[missing, :] = np.nan
    out[:, missing] = np.nan
    return out


def _hellinger_rows(values: list[Mix | None]) -> Array:
    """|sqrt p - sqrt q| / sqrt 2: exactly 0 for equal mixes, where sqrt(1 - sum sqrt pq) is not."""
    keys = sorted({k for v in values if v for k in v})
    roots = np.array([[math.sqrt(v.get(k, 0.0)) if v else 0.0 for k in keys] for v in values]).reshape(len(values), len(keys))
    return np.clip(cdist(roots, roots) / math.sqrt(2.0), 0.0, 1.0)


def _mix_distance(values: list[Any]) -> Array:
    return _blank(values, _hellinger_rows(values))


def _histogram_distance(values: list[Any]) -> Array:
    channels = [_hellinger_rows([v[c] if v and len(v) > c else None for v in values]) for c in range(3)]
    return _blank(values, np.mean(channels, axis=0))


def _set_distance(values: list[Any]) -> Array:
    """Weighted Jaccard; two empty sets carry no evidence either way."""
    keys = sorted({k for v in values if v for k in v})
    vec = np.array([[v.get(k, 0.0) if v else 0.0 for k in keys] for v in values]).reshape(len(values), len(keys))
    out = np.full((len(values), len(values)), np.nan)
    for i in range(len(values)):
        low = np.minimum(vec[i], vec).sum(axis=1)
        high = np.maximum(vec[i], vec).sum(axis=1)
        out[i] = np.where(high > 0, 1.0 - low / np.where(high > 0, high, 1.0), np.nan)
    return _blank(values, out)


def _difference(values: list[Any]) -> Array:
    x = np.array([np.nan if v is None else float(v) for v in values])
    return np.abs(x[:, None] - x[None, :])


def _wasserstein(values: list[Any]) -> Array:
    """1-Wasserstein between weighted value distributions, from their quantile functions."""
    rows = []
    for v in values:
        if v is None:
            rows.append(np.full(len(QUANTILES), np.nan))
            continue
        x, w = v
        order = np.argsort(x)
        cum = (np.cumsum(w[order]) - 0.5 * w[order]) / w.sum()
        rows.append(np.interp(QUANTILES, cum, x[order]))
    q = np.stack(rows)
    return np.abs(q[:, None, :] - q[None, :, :]).mean(axis=2)


DISTANCES: dict[str, Callable[[list[Any]], Array]] = {
    "mix": _mix_distance,
    "histogram": _histogram_distance,
    "set": _set_distance,
    "share": _difference,
    "scalar": _difference,
    "samples": _wasserstein,
}


def _calibrate(raw: Array, kind: str, method: str) -> Array:
    """rank: the share of the field's measured pairs that are strictly closer, so identical videos stay at 0.
    scale: units divided by the field's SCALE_Q quantile, for the kinds with units; bounded distances
    are kept as they are."""
    n = len(raw)
    upper = np.triu_indices(n, 1)
    values = raw[upper]
    measured = ~np.isnan(values)
    scaled = values.copy()
    if measured.any():
        if method == "rank":
            scaled[measured] = (rankdata(values[measured], method="min") - 1) / measured.sum()
        elif kind in UNBOUNDED:
            top = float(np.quantile(values[measured], SCALE_Q))
            if top > 0:
                scaled[measured] = np.clip(values[measured] / top, 0.0, 1.0)
    out = np.full_like(raw, np.nan)
    out[upper] = scaled
    out.T[upper] = scaled
    present = ~np.isnan(np.diag(raw))
    out[np.diag_indices(n)] = np.where(present, 0.0, np.nan)
    return out


def _nanmean(matrices: list[Array]) -> Array:
    stack = np.stack(matrices)
    have = ~np.isnan(stack)
    count = have.sum(axis=0)
    total = np.where(have, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def score(kits: list[Kit], method: str = "scale") -> dict[str, Any]:
    """Per field raw and calibrated matrices, then the group, level and overall averages."""
    raw: dict[str, Array] = {}
    fields: dict[str, Array] = {}
    values: dict[str, list[Any]] = {}
    for name, _, _, kind, read in FIELDS:
        values[name] = [read(kit) for kit in kits]
        raw[name] = DISTANCES[kind](values[name])
        fields[name] = _calibrate(raw[name], kind, method)
    groups: dict[tuple[str, str], Array] = {}
    for level, group in dict.fromkeys((lv, g) for _, lv, g, _, _ in FIELDS):
        groups[(level, group)] = _nanmean([fields[n] for n, lv, g, _, _ in FIELDS if (lv, g) == (level, group)])
    levels = {level: _nanmean([m for (lv, _), m in groups.items() if lv == level]) for level in LEVELS}
    return {
        "values": values,
        "raw": raw,
        "fields": fields,
        "groups": groups,
        "levels": levels,
        "overall": _nanmean(list(levels.values())),
    }


def _nearest(matrix: Array, names: list[str], top: int) -> dict[str, list[list[Any]]]:
    out: dict[str, list[list[Any]]] = {}
    for i, name in enumerate(names):
        order = [j for j in np.argsort(np.where(np.isnan(matrix[i]), np.inf, matrix[i])) if j != i]
        out[name] = [[names[j], round(float(matrix[i, j]), 4)] for j in order[:top] if not np.isnan(matrix[i, j])]
    return out


def _kmedoids(d: Array, k: int) -> Array:
    """k-medoids, k-medoids++ seeding, the cheapest of RESTARTS seeded runs; returns each video's cluster."""
    rng = np.random.default_rng(SEED)
    n = len(d)
    best: tuple[float, Array] | None = None
    for _ in range(RESTARTS):
        medoids = [int(rng.integers(n))]
        while len(medoids) < k:
            near = d[:, medoids].min(axis=1) ** 2
            medoids.append(int(rng.choice(n, p=near / near.sum())) if near.sum() > 0 else int(rng.integers(n)))
        for _ in range(100):
            labels = d[:, medoids].argmin(axis=1)
            moved = [
                int(np.flatnonzero(labels == c)[d[np.ix_(labels == c, labels == c)].sum(axis=1).argmin()])
                if (labels == c).any()
                else medoids[c]
                for c in range(k)
            ]
            if moved == medoids:
                break
            medoids = moved
        labels = d[:, medoids].argmin(axis=1)
        cost = float(d[np.arange(n), np.asarray(medoids)[labels]].sum())
        if best is None or cost < best[0]:
            best = (cost, labels)
    assert best is not None
    return best[1]


def _typical(name: str, values: list[Any]) -> Any:
    """A cluster's value of a field: the mean of a number, the leading labels of a mix or set, the
    median of a distribution, in seconds for shot lengths."""
    present = [v for v in values if v is not None]
    kind = KIND_OF[name]
    if not present or kind == "histogram":
        return None
    if kind in ("share", "scalar"):
        return round(float(np.mean(present)), 3)
    if kind == "samples":
        middle = float(np.median([np.median(x) for x, _ in present]))
        return round(math.exp(middle) if name in LOGGED else middle, 2)
    total: dict[str, float] = {}
    for v in present:
        for label, weight in v.items():
            total[label] = total.get(label, 0.0) + weight / len(present)
    return {k: round(w, 2) for k, w in sorted(total.items(), key=lambda kv: -kv[1])[:3]}


def _held_by(
    fields: dict[str, Array], values: dict[str, list[Any]], members: list[int], others: list[int]
) -> list[dict[str, Any]]:
    """The fields whose distances are smallest inside the cluster against outside it, with the
    cluster's value of each: what it shares."""
    found = []
    for name, m in fields.items():
        inside = m[np.ix_(members, members)][np.triu_indices(len(members), 1)]
        outside = m[np.ix_(members, others)].ravel()
        if np.isnan(inside).all() or np.isnan(outside).all():
            continue
        found.append(
            {
                "field": name,
                "typical": _typical(name, [values[name][i] for i in members]),
                "inside": round(float(np.nanmean(inside)), 3),
                "outside": round(float(np.nanmean(outside)), 3),
                "gap": float(np.nanmean(outside) - np.nanmean(inside)),
            }
        )
    found.sort(key=lambda f: -f["gap"])
    return [{k: v for k, v in f.items() if k != "gap"} for f in found[:HELD_BY]]


def _clusters(
    matrix: Array, fields: dict[str, Array], values: dict[str, list[Any]], names: list[str], k: int
) -> dict[str, Any] | None:
    """k-medoids into k clusters, each named by its medoid, a real video, with the fields it shares.
    Average linkage was tried and split off single outliers; silhouette is reported, not used to pick k,
    being low on a collection that is a continuum rather than separate kinds."""
    n = len(names)
    k = min(k, n - 1)
    if k < 2:
        return None
    filled = np.where(np.isnan(matrix), np.nanmean(matrix), matrix)
    np.fill_diagonal(filled, 0.0)
    labels = _kmedoids(filled, k)
    groups = []
    for label in sorted(set(labels.tolist())):
        members = [i for i in range(n) if labels[i] == label]
        others = [i for i in range(n) if labels[i] != label]
        medoid = min(members, key=lambda i: float(filled[i, members].sum()))
        groups.append(
            {
                "medoid": names[medoid],
                "members": sorted(names[i] for i in members),
                "shares": _held_by(fields, values, members, others) if len(members) > 1 and others else [],
            }
        )
    groups.sort(key=lambda g: (-len(g["members"]), g["medoid"]))
    fit = float(silhouette_score(filled, labels, metric="precomputed")) if len(groups) > 1 else None
    return {"k": len(groups), "silhouette": round(fit, 3) if fit is not None else None, "groups": groups}


def _text(typical: Any) -> str:
    if isinstance(typical, dict):
        return ", ".join(f"{k} {v:g}" for k, v in typical.items())
    return "-" if typical is None else f"{typical:g}"


def _round(matrix: Array) -> list[list[float | None]]:
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in matrix]


def explain(result: dict[str, Any], names: list[str], a: str, b: str) -> str:
    """Every field's raw and calibrated distance for one pair, under its group and level."""
    i, j = names.index(a), names.index(b)

    def cell(m: Array) -> str:
        return "-" if np.isnan(m[i, j]) else f"{m[i, j]:.3f}"

    lines = [f"{a} vs {b}: overall {cell(result['overall'])}"]
    for level in LEVELS:
        lines.append(f"  {level} {cell(result['levels'][level])}")
        for (lv, group), matrix in result["groups"].items():
            if lv != level:
                continue
            lines.append(f"    {group} {cell(matrix)}")
            for name, flv, fg, _, _ in FIELDS:
                if (flv, fg) == (lv, group):
                    lines.append(f"      {name:24} raw {cell(result['raw'][name]):>6}  calibrated {cell(result['fields'][name])}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="FRESCO-style distance between processed videos")
    p.add_argument("videos", nargs="*", help="names under --processed; default every one with an identikit")
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "score" / "similarity.json")
    p.add_argument(
        "--calibration",
        choices=["rank", "scale"],
        default="scale",
        help="scale: units over each field's 90th percentile; rank: each field's pair percentile",
    )
    p.add_argument("--clusters", type=int, default=CLUSTERS, help="k-medoids clusters per scope")
    p.add_argument("--top", type=int, default=TOP, help="nearest videos listed per video")
    p.add_argument("--explain", nargs=2, metavar=("A", "B"), help="print one pair's distances field by field")
    args = p.parse_args()

    names = args.videos or sorted(d.name for d in args.processed.iterdir() if (d / "identikit.json").exists())
    if len(names) < 2:
        sys.exit("needs at least two processed videos")
    kits = [json.loads((args.processed / n / "identikit.json").read_text(encoding="utf-8")) for n in names]
    result = score(kits, args.calibration)

    report = {
        "videos": names,
        "calibration": args.calibration,
        "fields": [
            {
                "name": name,
                "level": level,
                "group": group,
                "kind": kind,
                "pairs": int((~np.isnan(result["fields"][name][np.triu_indices(len(names), 1)])).sum()),
            }
            for name, level, group, kind, _ in FIELDS
        ],
        "distance": {**{lv: _round(m) for lv, m in result["levels"].items()}, "overall": _round(result["overall"])},
        "nearest": {
            scope: _nearest(m, names, args.top)
            for scope, m in (("overall", result["overall"]), *result["levels"].items())
        },
        "clusters": {
            scope: _clusters(
                m,
                {n: f for n, f in result["fields"].items() if scope in ("overall", LEVEL_OF[n])},
                result["values"],
                names,
                args.clusters,
            )
            for scope, m in (("overall", result["overall"]), *result["levels"].items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")

    print(f"{len(names)} videos, {len(names) * (len(names) - 1) // 2} pairs, calibration {args.calibration}")
    for scope, found in report["clusters"].items():
        if found:
            print(f"{scope}: {found['k']} clusters, silhouette {found['silhouette']}")
            for g in found["groups"]:
                shares = "; ".join(f"{f['field']} {_text(f['typical'])}" for f in g["shares"]) or "-"
                print(f"  {g['medoid']:14} {len(g['members']):3} videos, shares {shares}")
    if args.explain:
        print(explain(result, names, *args.explain))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
