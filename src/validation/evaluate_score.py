"""The similarity score against AVE's labels on scenes it never saw: Mantel tests per level against profiles
built from the labels, nearest neighbours, clusters and weights, beside two black-box embeddings:
python3 src/validation/evaluate_score.py --processed data/processed/ave_processed"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.score import similarity as sim

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Mix: TypeAlias = dict[str, float]
Shot: TypeAlias = tuple[float, float, dict[str, Any]]

LEVELS = ("plastic", "figurative", "enunciative")
PROFILES = {
    "camera": ("shot-size", "shot-angle", "shot-motion", "shot-type"),
    "content": ("shot-location", "num-people", "shot-subject"),
    "order": (
        "shot-size",
        "shot-motion",
        "num-people",
    ),  # labels the arcs have a counterpart for
}
NEAREST = ("shot-location", "num-people", "shot-motion", "shot-size")
MOBILECLIP = "MobileCLIP keyframes"
CLIP = "CLIP ViT-L/14 keyframes"
CLIP_CACHE = ROOT / "models" / "validation"  # OpenAI's ViT-L/14, as consistency.py
# (candidate, baseline, reference): the difference in Spearman, paired over the same resamples
GAINS = (
    ("figurative", MOBILECLIP, "content"),
    ("figurative", CLIP, "content"),
    ("enunciative", MOBILECLIP, "camera"),
    ("enunciative", CLIP, "camera"),
    ("overall", MOBILECLIP, "content"),
    ("overall", CLIP, "content"),
    ("figurative", "figurative, no arcs", "content"),
    ("enunciative", "enunciative, no arcs", "camera"),
    ("overall", "overall, no arcs", "order"),
    ("order", "overall, no arcs", "order"),
    ("figurative", "figurative, no rare-label weights", "content"),
)
CLUSTERED = ("plastic", "figurative", "enunciative", "overall", "order")
CLUSTERS = 4
PERMUTATIONS = 999
BOOT = 1000
SUBSAMPLES, SUBSAMPLE_SHARE = 100, 0.8
DRAWS = (
    100  # random field weights, each e^U(-1, 1), as the weighting study of 2026-10-17
)
TOP = 5
BATCH = 64
SEED = 0


def _shots(scene: dict[str, Any]) -> list[Shot]:
    return sorted(
        (
            (float(s["start-time"][0]), float(s["end-time"][0]), s)
            for s in scene.values()
        ),
        key=lambda t: t[0],
    )


def _labels(shot: dict[str, Any], field: str) -> list[str]:
    return [str(v) for v in shot.get(field, []) if v != "other"]


def _profile(shots: list[Shot], field: str) -> Mix | None:
    """A field's labels weighted by the seconds of the shots carrying them; a shot with two labels splits."""
    weights: dict[str, float] = {}
    for start, end, shot in shots:
        found = _labels(shot, field)
        for label in found:
            weights[label] = weights.get(label, 0.0) + (end - start) / len(found)
    return sim._mix(weights)


def _order(shots: list[Shot], field: str) -> Mix | None:
    """The mix of label changes from one shot to the next, weighted by the two shots' seconds, as an arc
    weighs its keyframes; shots with two labels or none are left out."""
    seq = [
        (end - start, found[0])
        for start, end, shot in shots
        if len(found := _labels(shot, field)) == 1
    ]
    weights: dict[str, float] = {}
    for (d0, a), (d1, b) in pairwise(seq):
        key = f"{a} > {b}"
        weights[key] = weights.get(key, 0.0) + d0 + d1
    return sim._mix(weights)


def _reference(shots: list[list[Shot]], name: str) -> Array:
    read = _order if name == "order" else _profile
    return sim._nanmean(
        [sim._mix_distance([read(s, field) for s in shots]) for field in PROFILES[name]]
    )


def _cosine(vectors: Array) -> Array:
    out = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    np.fill_diagonal(out, 0.0)
    return out


def _unit(rows: Array) -> Array:
    mean = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).mean(axis=0)
    return mean / np.linalg.norm(mean)


def _mobileclip(processed: Path, names: list[str]) -> Array:
    """02's MobileCLIP keyframe embeddings, averaged per scene."""
    return _cosine(
        np.stack(
            [
                _unit(np.load(processed / n / "embeddings" / "keyframe_embeddings.npy"))
                for n in names
            ]
        )
    )


def _clip(processed: Path, names: list[str], cache: Path) -> Array:
    """CLIP ViT-L/14 image embeddings of the same keyframes, averaged per scene, cached beside the report."""
    found: dict[str, Array] = dict(np.load(cache)) if cache.exists() else {}
    missing = [n for n in names if n not in found]
    if missing:
        import open_clip
        import torch
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model: Any
        preprocess: Any
        # OpenAI trained these weights with QuickGELU; plain GELU bends the vectors
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=str(CLIP_CACHE),
            force_quick_gelu=True,
        )
        model = model.to(device).eval()
        for name in missing:
            seg = json.loads(
                (processed / name / "json" / "segmentation.json").read_text(
                    encoding="utf-8"
                )
            )
            paths = [
                processed / name / k["path"]
                for s in seg["shots"]
                for k in s["keyframes"]
            ]
            rows: list[Array] = []
            with torch.no_grad():
                for start in range(0, len(paths), BATCH):
                    batch = torch.stack(
                        [
                            preprocess(Image.open(p).convert("RGB"))
                            for p in paths[start : start + BATCH]
                        ]
                    ).to(device)
                    rows.append(model.encode_image(batch).float().cpu().numpy())
            found[name] = _unit(np.concatenate(rows))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **found)
    return _cosine(np.stack([found[n] for n in names]))


def _spearman(x: Array, y: Array) -> float:
    keep = ~(np.isnan(x) | np.isnan(y))
    return float(np.corrcoef(rankdata(x[keep]), rankdata(y[keep]))[0, 1])


def _mantel(
    a: Array,
    b: Array,
    perms: list[Array],
    draws: list[tuple[Array, Array]],
) -> dict[str, Any]:
    """Spearman over the scene pairs; p from permuting b's scenes; interval from resampling scenes."""
    rows, cols = np.triu_indices(len(a), 1)
    rho = _spearman(a[rows, cols], b[rows, cols])
    above = sum(_spearman(a[rows, cols], b[p][:, p][rows, cols]) >= rho for p in perms)
    boot = [_spearman(a[i, j], b[i, j]) for i, j in draws]
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "rho": round(rho, 4),
        "p": round((1 + above) / (1 + len(perms)), 4),
        "ci": [round(float(low), 4), round(float(high), 4)],
    }


def _gain(
    a: Array, b: Array, ref: Array, draws: list[tuple[Array, Array]]
) -> dict[str, Any]:
    rows, cols = np.triu_indices(len(a), 1)
    gain = _spearman(a[rows, cols], ref[rows, cols]) - _spearman(
        b[rows, cols], ref[rows, cols]
    )
    boot = [
        _spearman(a[i, j], ref[i, j]) - _spearman(b[i, j], ref[i, j]) for i, j in draws
    ]
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"gain": round(gain, 4), "ci": [round(float(low), 4), round(float(high), 4)]}


def _resamples(n: int, rng: np.random.Generator) -> list[tuple[Array, Array]]:
    """Scene resamples as pair indices; a scene drawn twice is never paired with itself."""
    rows, cols = np.triu_indices(n, 1)
    out: list[tuple[Array, Array]] = []
    for _ in range(BOOT):
        idx = rng.integers(n, size=n)
        i, j = idx[rows], idx[cols]
        keep = i != j
        out.append((i[keep], j[keep]))
    return out


def _dominant(mix: Mix | None) -> str | None:
    return max(mix, key=lambda k: mix[k]) if mix else None


def _nearest(matrix: Array, labels: list[str | None]) -> float:
    """Share of scenes whose nearest scene has the same dominant label."""
    hits: list[bool] = []
    for i, label in enumerate(labels):
        row = np.where(np.isnan(matrix[i]), np.inf, matrix[i])
        row[i] = np.inf
        j = int(row.argmin())
        if label is not None and labels[j] is not None:
            hits.append(labels[j] == label)
    return round(float(np.mean(hits)), 4)


def _chance(labels: list[str | None]) -> float:
    found = [x for x in labels if x is not None]
    counts = np.array([found.count(x) for x in set(found)], dtype=float)
    n = len(found)
    return round(float((counts * (counts - 1)).sum() / (n * (n - 1))), 4)


def _clusters(matrix: Array, rng: np.random.Generator) -> dict[str, Any]:
    """k-medoids as the score's report, its silhouette, and how well each cluster comes back on 80% subsamples."""
    d = np.nan_to_num(matrix, nan=float(np.nanmean(matrix)))
    np.fill_diagonal(d, 0.0)
    labels = sim._kmedoids(d, CLUSTERS)
    n = len(d)
    found: list[list[float]] = [[] for _ in range(CLUSTERS)]
    for _ in range(SUBSAMPLES):
        sub = np.sort(rng.choice(n, size=round(n * SUBSAMPLE_SHARE), replace=False))
        again = sim._kmedoids(d[np.ix_(sub, sub)], CLUSTERS)
        for c in range(CLUSTERS):
            members = set(sub[labels[sub] == c].tolist())
            if not members:
                continue
            found[c].append(
                max(
                    len(members & (other := set(sub[again == k].tolist())))
                    / len(members | other)
                    for k in range(CLUSTERS)
                )
            )
    return {
        "sizes": sorted(int((labels == c).sum()) for c in range(CLUSTERS)),
        "silhouette": round(
            float(silhouette_score(d, labels, metric="precomputed")), 4
        ),
        "jaccard": sorted(round(float(np.mean(x)), 3) for x in found if x),
    }


def _weighted(
    filled: dict[str, Array],
    group_of: dict[str, tuple[str, str]],
    weight: dict[str, float],
) -> dict[str, Array]:
    """The score's tree with a weight per field: fields to groups, groups to levels, levels to overall."""
    groups: dict[tuple[str, str], Array] = {}
    for key in dict.fromkeys(group_of.values()):
        names = [n for n, g in group_of.items() if g == key]
        stack = np.stack([filled[n] for n in names])
        w = np.array([weight[n] for n in names])[:, None, None]
        have = ~np.isnan(stack)
        total = np.where(have, w, 0.0).sum(axis=0)
        groups[key] = np.where(
            total > 0,
            np.where(have, stack * w, 0.0).sum(axis=0) / np.maximum(total, 1e-12),
            np.nan,
        )
    levels = {
        lv: sim._nanmean([m for (level, _), m in groups.items() if level == lv])
        for lv in LEVELS
    }
    return {**levels, "overall": sim._nanmean(list(levels.values()))}


def _top(matrix: Array) -> list[set[int]]:
    out: list[set[int]] = []
    for i in range(len(matrix)):
        row = np.where(np.isnan(matrix[i]), np.inf, matrix[i])
        row[i] = np.inf
        out.append(set(np.argsort(row, kind="stable")[:TOP].tolist()))
    return out


def _weights(result: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """How many of each video's five nearest survive random field weights."""
    filled = {
        n: sim._impute(m, sim._lacks(result["values"][n]))
        for n, m in result["fields"].items()
    }
    group_of: dict[str, tuple[str, str]] = result["group_of"]
    equal = _weighted(filled, group_of, {n: 1.0 for n in filled})
    for lv in LEVELS:
        assert np.allclose(equal[lv], result["levels"][lv], equal_nan=True)
    base = {scope: _top(m) for scope, m in equal.items()}
    kept: dict[str, list[float]] = {scope: [] for scope in base}
    for _ in range(DRAWS):
        weight = {n: float(np.exp(rng.uniform(-1.0, 1.0))) for n in filled}
        for scope, m in _weighted(filled, group_of, weight).items():
            kept[scope].append(
                float(np.mean([len(a & b) / TOP for a, b in zip(base[scope], _top(m))]))
            )
    return {
        scope: {"mean": round(float(np.mean(x)), 4), "min": round(float(np.min(x)), 4)}
        for scope, x in kept.items()
    }


def _page(path: Path, report: dict[str, Any]) -> None:
    rows: list[tuple[str, ...]] = [
        ("matrix", "camera labels", "content labels", "label order")
    ]
    for name, found in report["mantel"].items():
        rows.append(
            (
                name,
                *(
                    f"{found[r]['rho']:.3f} [{found[r]['ci'][0]:.3f}, {found[r]['ci'][1]:.3f}]"
                    for r in PROFILES
                ),
            )
        )
    near: list[tuple[str, ...]] = [("matrix", *NEAREST)]
    near.append(("chance", *(f"{report['nearest']['chance'][f]:.2f}" for f in NEAREST)))
    for name, found in report["nearest"]["share"].items():
        near.append((name, *(f"{found[f]:.2f}" for f in NEAREST)))
    fig = Figure(
        figsize=(16, 2.6 + 0.3 * (len(rows) + len(near))), dpi=100, facecolor="#f9f9f7"
    )
    fig.text(
        0.02,
        0.97,
        "The score against AVE's labels",
        fontsize=22,
        weight="bold",
        va="top",
    )
    fig.text(
        0.02,
        0.9,
        f"{report['scenes']} AVE scenes, {report['pairs']:,} pairs. Mantel: Spearman between distance matrices, "
        f"95% interval from {report['resamples']:,} resamples of scenes. Nearest: the share of scenes whose "
        "nearest scene has the same dominant label.",
        fontsize=10.5,
        color="#555555",
        va="top",
    )
    y = 0.8
    step = 0.6 / (len(rows) + len(near) + 2)
    for table, at in (
        (rows, (0.02, 0.33, 0.55, 0.77)),
        (near, (0.02, 0.33, 0.48, 0.63, 0.78)),
    ):
        for i, row in enumerate(table):
            for text, x in zip(row, at):
                fig.text(
                    x,
                    y,
                    text,
                    fontsize=10.5,
                    color="#888888" if i == 0 else "#111111",
                    va="top",
                )
            y -= step
        y -= step
    fig.savefig(path, facecolor="#f9f9f7", bbox_inches="tight", pad_inches=0.3)


def main() -> None:
    p = argparse.ArgumentParser(description="The similarity score against AVE's labels")
    p.add_argument(
        "--processed", type=Path, default=ROOT / "data" / "processed" / "ave_processed"
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "raw" / "ave" / "annotations.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "score" / "score_ave.json",
        help="the report; its page and the CLIP cache are written beside it",
    )
    args = p.parse_args()

    ave: dict[str, Any] = json.loads(args.labels.read_text(encoding="utf-8"))
    names = sorted(
        d.name
        for d in args.processed.iterdir()
        if (d / "identikit.json").exists() and d.name in ave
    )
    if len(names) < 20:
        sys.exit(f"needs 20 labelled AVE scenes under {args.processed}")
    kits = [
        json.loads((args.processed / n / "identikit.json").read_text(encoding="utf-8"))
        for n in names
    ]
    shots = [_shots(ave[n]) for n in names]
    references = {r: _reference(shots, r) for r in PROFILES}

    result = sim.score(kits)
    groups: dict[tuple[str, str], Array] = result["groups"]
    no_arcs = {
        lv: sim._nanmean(
            [m for (level, g), m in groups.items() if level == lv and g != "arc"]
        )
        for lv in LEVELS
    }
    rare = sim.RARE
    sim.RARE = set()
    plain = sim.score(kits)
    sim.RARE = rare
    candidates = {
        **{lv: result["levels"][lv] for lv in LEVELS},
        "overall": result["overall"],
        "order": result["order"],
        "figurative, no arcs": no_arcs["figurative"],
        "enunciative, no arcs": no_arcs["enunciative"],
        "overall, no arcs": sim._nanmean(list(no_arcs.values())),
        "figurative, no rare-label weights": plain["levels"]["figurative"],
        MOBILECLIP: _mobileclip(args.processed, names),
        CLIP: _clip(args.processed, names, args.out.with_name("clip_vitl14.npz")),
    }

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(names)) for _ in range(PERMUTATIONS)]
    draws = _resamples(len(names), rng)
    dominant = {f: [_dominant(_profile(s, f)) for s in shots] for f in NEAREST}
    report: dict[str, Any] = {
        "scenes": len(names),
        "pairs": len(names) * (len(names) - 1) // 2,
        "references": {r: list(f) for r, f in PROFILES.items()},
        "permutations": PERMUTATIONS,
        "resamples": BOOT,
        "seed": SEED,
        "mantel": {
            name: {r: _mantel(m, references[r], perms, draws) for r in PROFILES}
            for name, m in candidates.items()
        },
        "gains": [
            {
                "matrix": a,
                "against": b,
                "reference": r,
                **_gain(candidates[a], candidates[b], references[r], draws),
            }
            for a, b, r in GAINS
        ],
        "nearest": {
            "chance": {f: _chance(dominant[f]) for f in NEAREST},
            "share": {
                name: {f: _nearest(m, dominant[f]) for f in NEAREST}
                for name, m in candidates.items()
            },
        },
        "clusters": {
            name: _clusters(candidates[name], np.random.default_rng([SEED, i]))
            for i, name in enumerate(CLUSTERED)
        },
        "weights": _weights(result, np.random.default_rng([SEED, 99])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = args.out.with_suffix(".png")
    _page(page, report)

    for name, found in report["mantel"].items():
        print(
            f"{name:36}"
            + "  ".join(
                f"{r} {x['rho']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] p {x['p']:.3f}"
                for r, x in found.items()
            )
        )
    for g in report["gains"]:
        print(
            f"gain {g['matrix']} over {g['against']} on {g['reference']}: {g['gain']:+.3f} {g['ci']}"
        )
    print("nearest, chance:", report["nearest"]["chance"])
    for name, found in report["nearest"]["share"].items():
        print(f"nearest {name:36} {found}")
    for name, found in report["clusters"].items():
        print(f"clusters {name:12} {found}")
    print("weights:", report["weights"])
    print(f"-> {args.out}, page {page.name}")


if __name__ == "__main__":
    main()
