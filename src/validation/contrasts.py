"""Groups AVE's labels tell apart, against the pipeline's values there: python3 src/validation/contrasts.py"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from evaluate_ave import matched
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Row: TypeAlias = dict[str, Any]

MIN_N = 20  # shots per group needed to judge
IOU = 0.5  # evaluate_ave's floor for matching an AVE shot to a segment

# value, what it is, AVE field, label expected higher, label expected lower; plausibility.py's, set before its first run
CONTRASTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("outdoor", "outdoor probability (05 Places365)", "shot-location", "ext", "int"),
    ("sky", "sky coverage (05 SegFormer)", "shot-location", "ext", "int"),
    ("face_height", "largest face height (12)", "shot-size", "close-up", "wide"),
    ("face_height", "largest face height (12)", "shot-size", "close-up", "medium"),
    (
        "face_background",
        "face to background ratio (12)",
        "shot-size",
        "close-up",
        "wide",
    ),
    ("steady", "share read steady (09)", "shot-motion", "locked", "handheld"),
    ("visible", "people visible (03)", "num-people", "1", "0"),
    ("visible", "people visible (03)", "num-people", "2", "1"),
    ("followed", "people followed (08)", "num-people", "2", "1"),
    ("animal", "animal subject (06, via 12)", "shot-subject", "animal", "human"),
    ("not_live", "not live action (05 medium)", "shot-subject", "cartoon", "human"),
)


def _read(root: Path, name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((root / name).read_text(encoding="utf-8"))
    return found


def _rows(root: Path, shots: dict[str, Any]) -> list[Row]:
    """One row per AVE shot matched to a segment: its labels and the pipeline's values there."""
    kit = _read(root, "identikit.json")
    segmentation = _read(root, "json/segmentation.json")
    plastic = {int(s["segment"]): s for s in kit["plastic"]["segments"]}
    figurative = {int(s["segment"]): s for s in kit["figurative"]["segments"]}
    enunciative = {int(s["segment"]): s for s in kit["enunciative"]["segments"]}
    animals = {int(i) for s in kit["figurative"]["subjects"] for i in s["segments"]}
    visible = {
        int(s["index"]): int(s["counts"]["person_max"])
        for s in _read(root, "json/profile.json")["shots"]
    }
    medium = {
        int(s["index"]): (s["plastic"].get("medium") or {}).get("distribution")
        for s in _read(root, "json/aggregated.json")["shots"]
    }
    rows: list[Row] = []
    for labels, segment in matched(shots, segmentation, IOU):
        i = int(segment["index"])
        distance = enunciative[i].get("camera_distance") or {}
        rig = (enunciative[i].get("camera") or {}).get("rig")
        live = medium[i]
        rows.append(
            {
                "scene": root.name,
                "labels": labels,
                "outdoor": (figurative[i].get("indoor_outdoor") or {}).get("outdoor"),
                "sky": float(plastic[i]["coverage"].get("sky", 0.0)),
                "face_height": distance.get("face_height"),
                "face_background": enunciative[i].get("face_background_ratio"),
                "steady": None if rig is None else float(rig == "steady"),
                "visible": float(visible[i]),
                "followed": float(enunciative[i]["followed"]),
                "animal": float(i in animals),
                "not_live": 1.0 - float(live["live action"]) if live else None,
            }
        )
    return rows


def _auc(high: Array, low: Array) -> float:
    """P(higher > lower), ties counting half."""
    if not len(high) or not len(low):
        return float("nan")
    ordered = np.sort(low)
    below = np.searchsorted(ordered, high, side="left")
    equal = np.searchsorted(ordered, high, side="right") - below
    return float((below + 0.5 * equal).sum() / (len(high) * len(low)))


def _member(labels: dict[str, Any], field: str, wanted: str, other: str) -> bool:
    """A shot with both labels is in neither group."""
    values = set(labels.get(field, []))
    return wanted in values and other not in values


def _join(arrays: list[Array]) -> Array:
    return np.concatenate(arrays) if arrays else np.empty(0)


def _contrast(
    rows: list[Row],
    check: tuple[str, str, str, str, str],
    rng: np.random.Generator,
    boot: int,
) -> Row:
    """AUC of the higher group over the lower; the interval resamples whole scenes."""
    value, name, field, higher, lower = check
    usable = [r for r in rows if r[value] is not None]

    def group(scene: str, wanted: str, other: str) -> Array:
        return np.array(
            [
                r[value]
                for r in usable
                if r["scene"] == scene and _member(r["labels"], field, wanted, other)
            ],
            dtype=float,
        )

    scenes = sorted({r["scene"] for r in usable})
    groups = [(group(s, higher, lower), group(s, lower, higher)) for s in scenes]
    high, low = _join([g[0] for g in groups]), _join([g[1] for g in groups])
    entry: Row = {
        "check": f"{value}: {higher} > {lower}",
        "value": name,
        "field": field,
        "n": [len(high), len(low)],
    }
    if min(len(high), len(low)) < MIN_N:
        return {**entry, "result": "too few"}
    stats = [
        _auc(_join([groups[i][0] for i in picks]), _join([groups[i][1] for i in picks]))
        for picks in rng.integers(0, len(groups), size=(boot, len(groups)))
    ]
    ci = [round(float(x), 3) for x in np.nanpercentile(stats, [2.5, 97.5])]
    return {
        **entry,
        "means": [round(float(high.mean()), 4), round(float(low.mean()), 4)],
        "auc": round(_auc(high, low), 3),
        "ci": ci,
        "result": "pass" if ci[0] > 0.5 else ("fail" if ci[1] < 0.5 else "weak"),
    }


# Page, in the identikit page's ink and surfaces

PAGE_SIZE = (22.4, 13.5)  # inches at 100 dpi
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
PAGE = "#f9f9f7"


def _page(path: Path, report: dict[str, Any]) -> None:
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(0.018, 0.988, "Contrasts: AVE's groups against the pipeline", fontsize=26,
             weight="bold", color=INK, va="top")  # fmt: skip
    fig.text(
        0.018,
        0.95,
        f"{report['scenes']} AVE scenes, {report['matched_shots']:,} shots matched to a segment. AUC: the chance "
        "that a shot of the higher group scores above one of the lower, ties half; pass when the 95% interval, "
        f"from {report['resamples']:,} resamples of whole scenes, sits above 0.5.",
        fontsize=12,
        color=SECONDARY,
        va="top",
    )
    at = [0.018, 0.22, 0.3, 0.44, 0.53, 0.65, 0.8]
    header = [
        "pipeline value",
        "AVE field",
        "higher > lower",
        "shots",
        "means",
        "AUC [95% CI]",
        "result",
    ]
    step = 10.5 * 1.7 / (PAGE_SIZE[1] * 72)
    y = 0.9
    for text, x in zip(header, at):
        fig.text(x, y, text, fontsize=10, color=MUTED, va="top")
    for r in [*report["checks"], *report["planted"]]:
        y -= step
        means = r.get("means")
        row = [
            r["value"],
            r["field"],
            r["check"].split(": ", 1)[1],
            f"{r['n'][0]:,} / {r['n'][1]:,}",
            "-" if means is None else f"{means[0]:.3g} / {means[1]:.3g}",
            f"{r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
            if "auc" in r
            else "-",
            r["result"]
            + (f" (planted: expected {r['expected']})" if "expected" in r else ""),
        ]
        for i, (text, x) in enumerate(zip(row, at)):
            fig.text(
                x, y, text, fontsize=10.5, color=INK if i == 0 else SECONDARY, va="top"
            )
    fig.savefig(
        path, facecolor=PAGE, bbox_inches="tight", pad_inches=0.3
    )  # one short table


def main() -> None:
    p = argparse.ArgumentParser(
        description="AVE's label groups against the pipeline's values"
    )
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
        default=ROOT / "data" / "validation" / "contrasts" / "contrasts.json",
        help="the report; its page is written beside it",
    )
    p.add_argument(
        "--boot", type=int, default=1000, help="scene resamples behind each interval"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    scenes = sorted(
        s for s in labels if (args.processed / s / "identikit.json").exists()
    )
    if not scenes:
        sys.exit(f"no AVE scene in {args.processed} has been through 12-identikit")
    rows = [r for s in scenes for r in _rows(args.processed / s, labels[s])]
    checks = [
        _contrast(rows, check, np.random.default_rng([args.seed, i]), args.boot)
        for i, check in enumerate(CONTRASTS)
    ]
    dead = _contrast(
        [{**r, "outdoor": 0.5} for r in rows],
        CONTRASTS[0],
        np.random.default_rng([args.seed, 999]),
        args.boot,
    )
    report = {
        "scenes": len(scenes),
        "matched_shots": len(rows),
        "iou_floor": IOU,
        "resamples": args.boot,
        "seed": args.seed,
        "min_n": MIN_N,
        "checks": checks,
        "planted": [
            {
                **dead,
                "value": "outdoor probability set to 0.5 everywhere",
                "expected": "not pass",
                "caught": dead["result"] != "pass",
            }
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _page(args.out.with_suffix(".png"), report)

    for r in checks:
        found = f"{r['auc']:.3f} {r['ci']}" if "auc" in r else f"n {r['n']}"
        print(f"{r['result']:8} {r['value']:36} {r['check']:34} {found}")
    for r in report["planted"]:
        print(
            f"planted: {r['value']}: {r['result']} ({'caught' if r['caught'] else 'MISSED'})"
        )
    print(f"-> {args.out}, page {args.out.with_suffix('.png').name}")


if __name__ == "__main__":
    main()
