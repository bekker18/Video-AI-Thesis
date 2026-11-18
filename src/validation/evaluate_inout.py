"""Gaze-LLE's in-frame probability, as 07 runs it, against VideoAttentionTarget's labels of looks inside and
outside the frame, on its test videos; 09 reads a look as outside below IN_FRAME (0.5):
python3 src/validation/evaluate_inout.py --out <report>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "validation"))
import evaluate_laeo as laeo

from main import load_stage

EXPERTS: Any = load_stage("07-conditional-experts")
AGGREGATION: Any = load_stage("09-aggregation")
VAT = ROOT / "data" / "raw" / "videoattentiontarget"
EVERY = 3  # every third annotated frame: neighbours are nearly the same picture
BOOT, SEED = 2000, 0


def _labels() -> dict[str, dict[str, list[tuple[list[int], bool]]]]:
    """Per clip and frame file: each annotated head box and whether its look lands inside the frame."""
    out: dict[str, dict[str, list[tuple[list[int], bool]]]] = {}
    for path in sorted((VAT / "annotations" / "test").glob("*/*/*.txt")):
        clip = f"{path.parent.parent.name}/{path.parent.name}"
        lines = [
            line.split(",")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for n, parts in enumerate(lines):
            if n % EVERY:
                continue
            name, box, gaze = (
                parts[0].strip(),
                [round(float(v)) for v in parts[1:5]],
                float(parts[5]),
            )
            out.setdefault(clip, {}).setdefault(name, []).append((box, gaze >= 0))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = _labels()
    cache = args.out.with_name(args.out.stem + "_looks.json")
    done: dict[str, list[list[Any]]] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    model = EXPERTS._gaze_model("cuda")
    for clip, frames in labels.items():
        if clip in done:
            continue
        rows: list[list[Any]] = []
        for name, heads in frames.items():
            image = cv2.imread(str(VAT / "images" / clip / name))
            if image is None:
                continue
            faces = [laeo._face(box) for box, _ in heads]
            for (box, inside), look in zip(
                heads, EXPERTS._gaze_pass(model, image, faces, "cuda")
            ):
                rows.append([float(look["in_frame"]), inside])
        done[clip] = rows
        cache.write_text(json.dumps(done), encoding="utf-8")
        print(clip, len(rows), flush=True)
    probs = [p for rows in done.values() for p, _ in rows]
    truth = [t for rows in done.values() for _, t in rows]
    clips = [c for c, rows in done.items() for _ in rows]
    y, s = np.array(truth), np.array(probs)
    cut = float(AGGREGATION.IN_FRAME)
    said = s >= cut
    by_clip: dict[str, list[int]] = {}
    for k, c in enumerate(clips):
        by_clip.setdefault(c, []).append(k)
    names = sorted(by_clip)
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(BOOT):
        idx = np.array(
            [k for c in rng.choice(len(names), len(names)) for k in by_clip[names[c]]]
        )
        if y[idx].all() or not y[idx].any():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    report = {
        "about": __doc__,
        "clips": len(names),
        "heads": len(y),
        "inside": int(y.sum()),
        "outside": int((~y).sum()),
        "auc": round(float(roc_auc_score(y, s)), 3),
        "auc_95": [round(float(np.percentile(aucs, q)), 3) for q in (2.5, 97.5)],
        "ap_inside": round(float(average_precision_score(y, s)), 3),
        "ap_outside": round(float(average_precision_score(~y, -s)), 3),
        "cut": cut,
        "accuracy_at_cut": round(float((said == y).mean()), 3),
        "inside_read_inside": round(float(said[y].mean()), 3),
        "outside_read_outside": round(float((~said[~y]).mean()), 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
