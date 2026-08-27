"""Renders the output of 06, 07 and 09 back over the source video.

    python3 preview.py messi
    python3 preview.py messi --views fusion

Writes data/processed/<video>/<view>_preview.mp4. Inspection only - no stage reads these.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Draw: TypeAlias = Callable[[Array, int], None]

PALETTE = (
    (66, 135, 245),
    (46, 204, 113),
    (231, 76, 60),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (52, 152, 219),
    (192, 57, 43),
    (127, 140, 141),
)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
ATTEND = (60, 190, 245)
MUTUAL = (110, 240, 130)
SYNC = (240, 200, 90)
FAINT = (95, 95, 95)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# mediapipe's 33-point skeleton
POSE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


def _load(root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = root / name
    if not path.exists():
        sys.exit(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _colour(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


def _centre(box: list[int]) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def _shot_map(tracks: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for shot in tracks["shots"]:
        for frame in range(int(shot["start_frame"]), int(shot["end_frame"]) + 1):
            out[frame] = int(shot["index"])
    return out


def _rows(tracks: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    """(shot, track, frame) -> the per-frame row 06 wrote."""
    out: dict[tuple[int, int, int], dict[str, Any]] = {}
    for shot in tracks["shots"]:
        if not shot["tracked"]:
            continue
        for track in shot["tracks"]:
            for row in track["frames"]:
                out[(int(shot["index"]), int(track["track_id"]), int(row["frame"]))] = (
                    row
                )
    return out


def _banner(frame: Array, left: str, right: str, live: bool) -> None:
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 26), (14, 14, 14), -1)
    cv2.putText(
        frame, left, (10, 18), FONT, 0.5, (90, 230, 140) if live else (150, 150, 150), 1
    )
    cv2.putText(
        frame,
        right,
        (max(10, width - 8 * len(right) - 10), 18),
        FONT,
        0.42,
        (150, 150, 150),
        1,
    )


def _panel(
    frame: Array, lines: list[tuple[str, tuple[int, int, int]]], width: int = 430
) -> None:
    if not lines:
        return
    height = frame.shape[0]
    top = height - 14 - 16 * len(lines)
    cv2.rectangle(frame, (8, top - 16), (width, height - 6), (18, 18, 18), -1)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (14, top + 16 * i), FONT, 0.4, colour, 1)


def _dashed(
    frame: Array,
    a: tuple[int, int],
    b: tuple[int, int],
    colour: tuple[int, int, int],
    step: int = 14,
) -> None:
    vector = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return
    unit = vector / length
    for start in range(0, int(length), step * 2):
        p = np.asarray(a) + unit * start
        q = np.asarray(a) + unit * min(start + step, length)
        cv2.line(frame, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2)


# Detection + Tracking


def _tracks_view(root: Path) -> Draw:
    tracks = _load(root, "tracks.json", "06-detection-tracking")
    shot_of = _shot_map(tracks)

    rows: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    crops: dict[int, list[tuple[int, str]]] = {}
    for shot in tracks["shots"]:
        if not shot["tracked"]:
            continue
        for track in shot["tracks"]:
            tid = int(track["track_id"])
            for row in track["frames"]:
                rows.setdefault(int(row["frame"]), []).append((tid, row))
            for kind in ("face", "body"):
                for crop in track["crops"][kind]:
                    crops.setdefault(int(crop["frame"]), []).append((tid, kind))

    def draw(frame: Array, index: int) -> None:
        shot = shot_of.get(index)
        active = rows.get(index, [])
        for tid, row in active:
            colour = _colour(tid)
            x1, y1, x2, y2 = row["body"]
            if row["interpolated"]:
                for a, b in (((x1, y1), (x2, y1)), ((x1, y2), (x2, y2))):
                    _dashed(frame, a, b, colour, 9)
                for a, b in (((x1, y1), (x1, y2)), ((x2, y1), (x2, y2))):
                    _dashed(frame, a, b, colour, 9)
                tag = f"#{tid} interp"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                tag = f"#{tid} {row['body_score']}"
            cv2.putText(frame, tag, (x1 + 3, max(16, y1 - 6)), FONT, 0.55, colour, 2)
            if row["face"]:
                fx1, fy1, fx2, fy2 = row["face"]
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), WHITE, 2)
                cv2.putText(
                    frame,
                    str(row["face_score"]),
                    (fx1, max(12, fy1 - 4)),
                    FONT,
                    0.45,
                    WHITE,
                    1,
                )

        for i, (tid, kind) in enumerate(crops.get(index, [])):
            cv2.putText(
                frame,
                f"CROP #{tid} {kind}",
                (frame.shape[1] - 230, 46 + 22 * i),
                FONT,
                0.55,
                _colour(tid),
                2,
            )

        observed = sum(1 for _, r in active if not r["interpolated"])
        tracked = shot is not None and any(
            int(s["index"]) == shot and s["tracked"] for s in tracks["shots"]
        )
        state = "tracked" if tracked else "NOT GATED - detection_tracking did not fire"
        _banner(
            frame,
            f"frame {index:5d}   shot {shot if shot is not None else '-'}   {state}",
            "solid = detected   dashed = interpolated   white = face",
            tracked,
        )
        cv2.putText(
            frame,
            f"tracks {len(active)}  observed {observed}  "
            f"interpolated {len(active) - observed}",
            (10, 44),
            FONT,
            0.45,
            (200, 200, 200),
            1,
        )

    return draw


# Conditional Experts


def _rotation(yaw: float, pitch: float, roll: float) -> Array:
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    rx = np.array(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]]
    )
    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    )
    rz = np.array(
        [[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]]
    )
    out: Array = rz @ ry @ rx
    return out


def _meter(
    frame: Array, x: int, y: int, value: float, label: str, width: int = 90
) -> None:
    cv2.rectangle(frame, (x, y), (x + width, y + 8), (70, 70, 70), -1)
    mid = x + width // 2
    end = int(mid + max(-1.0, min(1.0, value)) * width / 2)
    cv2.rectangle(
        frame,
        (min(mid, end), y),
        (max(mid, end), y + 8),
        (90, 200, 90) if value >= 0 else (90, 90, 220),
        -1,
    )
    cv2.putText(
        frame,
        f"{label} {value:+.2f}",
        (x + width + 6, y + 8),
        FONT,
        0.4,
        (220, 220, 220),
        1,
    )


def _conditional_view(root: Path) -> Draw:
    tracks = _load(root, "tracks.json", "06-detection-tracking")
    cond = _load(root, "conditional.json", "07-conditional-experts")
    shot_of = _shot_map(tracks)
    rows = _rows(tracks)

    samples: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    attrs: dict[tuple[int, int], dict[str, Any]] = {}
    text_at: dict[int, list[dict[str, Any]]] = {}
    for shot in cond["shots"]:
        index = int(shot["index"])
        for entry in shot["text"]:
            text_at.setdefault(int(entry["frame"]), []).append(entry)
        for track in shot["tracks"]:
            tid = int(track["track_id"])
            if track["attributes"]:
                attrs[(index, tid)] = track["attributes"]
            for sample in track["series"]:
                samples.setdefault(int(sample["frame"]), []).append(
                    (index, tid, sample)
                )

    def draw(frame: Array, index: int) -> None:
        height = frame.shape[0]
        shot = shot_of.get(index)
        live = samples.get(index, [])

        for shot_id, tid, sample in live:
            colour = _colour(tid)
            row = rows.get((shot_id, tid, index))
            if not row:
                continue

            pose = sample.get("pose")
            if pose and row["body"]:
                bx1, by1, bx2, by2 = row["body"]
                bw, bh = bx2 - bx1, by2 - by1
                # joints predicted outside the crop are guesses, not observations
                points = [
                    (
                        int(bx1 + kp[0] * bw),
                        int(by1 + kp[1] * bh),
                        kp[2] if 0.0 <= kp[0] <= 1.0 and 0.0 <= kp[1] <= 1.0 else 0.0,
                    )
                    for kp in pose["keypoints"]
                ]
                for a, b in POSE_EDGES:
                    if points[a][2] > 0.5 and points[b][2] > 0.5:
                        cv2.line(frame, points[a][:2], points[b][:2], colour, 2)
                for px, py, visible in points:
                    if visible > 0.5:
                        cv2.circle(frame, (px, py), 2, colour, -1)

            if row["face"]:
                fx1, fy1, fx2, fy2 = row["face"]
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), colour, 1)
                head = sample.get("head_pose")
                if head:
                    cx, cy = _centre(row["face"])
                    matrix = _rotation(head["yaw"], head["pitch"], head["roll"])
                    scale = max(18, (fx2 - fx1) // 2)
                    for vector, axis in (
                        ((1, 0, 0), (0, 0, 255)),
                        ((0, -1, 0), (0, 255, 0)),
                        ((0, 0, -1), (255, 0, 0)),
                    ):
                        end = matrix @ np.asarray(vector, dtype=np.float64) * scale
                        cv2.arrowedLine(
                            frame,
                            (cx, cy),
                            (int(cx + end[0]), int(cy + end[1])),
                            axis,
                            2,
                            tipLength=0.25,
                        )
                label = f"#{tid}"
                if "emotion" in sample:
                    label += f" {sample['emotion']['label']}"
                cv2.putText(frame, label, (fx1, max(12, fy1 - 5)), FONT, 0.5, colour, 2)

        for entry in text_at.get(index, []):
            x1, y1, x2, y2 = entry["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), MAGENTA, 1)
            cv2.putText(
                frame,
                str(entry["text"])[:46],
                (x1, min(height - 4, y2 + 14)),
                FONT,
                0.45,
                (255, 120, 255),
                1,
            )

        if live:
            shot_id, tid, sample = live[0]
            lines: list[tuple[str, tuple[int, int, int]]] = [
                (f"track #{tid}  frame {index}", (230, 230, 230))
            ]
            emotion = sample.get("emotion")
            if emotion:
                lines.append((str(emotion["label"]), (120, 220, 255)))
            head = sample.get("head_pose")
            if head:
                lines.append(
                    (
                        (
                            f"yaw {head['yaw']:+.0f}  pitch {head['pitch']:+.0f}  "
                            f"roll {head['roll']:+.0f}"
                        ),
                        (200, 200, 200),
                    )
                )
            shapes = list(sample.get("blendshapes", {}).items())[:2]
            if shapes:
                lines.append(
                    ("  ".join(f"{k} {v:.2f}" for k, v in shapes), (170, 170, 170))
                )
            found = attrs.get((shot_id, tid))
            if found and found.get("gender_age"):
                first = found["gender_age"][0]
                labels = found["clip"][0][:2] if found.get("clip") else []
                lines.append(
                    (
                        f"{first['gender']} ~{first['age']}  "
                        + ", ".join(str(x["label"])[9:] for x in labels),
                        (150, 200, 150),
                    )
                )
            _panel(frame, lines)
            if emotion and emotion.get("valence") is not None:
                base = height - 14 - 16 * len(lines)
                _meter(frame, 150, base + 8, float(emotion["valence"]), "val")
                _meter(frame, 150, base + 24, float(emotion["arousal"]), "aro")

        _banner(
            frame,
            f"frame {index:5d}  shot {shot if shot is not None else '-'}  "
            + ("SAMPLE" if live else "held"),
            "skeleton=pose  axes=head pose  magenta=OCR",
            bool(live),
        )

    return draw


# Fusion


def _fusion_view(root: Path) -> Draw:
    tracks = _load(root, "tracks.json", "06-detection-tracking")
    fused = _load(root, "fused.json", "09-fusion")
    shot_of = _shot_map(tracks)
    rows = _rows(tracks)

    edges: dict[int, list[dict[str, Any]]] = {}
    entities: dict[int, list[int]] = {}
    for shot in fused["shots"]:
        index = int(shot["index"])
        entities[index] = [int(e["track_id"]) for e in shot["entities"]]
        edges[index] = shot["relations"]

    def anchor(shot: int, tid: int, index: int) -> tuple[int, int] | None:
        row = rows.get((shot, tid, index))
        return None if row is None else _centre(row["face"] or row["body"])

    def draw(frame: Array, index: int) -> None:
        shot = shot_of.get(index)
        if shot is None:
            return
        shown: list[tuple[dict[str, Any], float, float]] = []

        for relation in edges.get(shot, []):
            a = anchor(shot, int(relation["a"]), index)
            b = anchor(shot, int(relation["b"]), index)
            if a is None or b is None:
                continue
            ab = float(relation["attention"]["a_to_b"].get("share", 0.0))
            ba = float(relation["attention"]["b_to_a"].get("share", 0.0))
            sync = relation["synchrony"]
            strong = max(ab, ba) >= 0.5
            if not strong and not sync["available"]:
                continue
            shown.append((relation, ab, ba))

            colour = (
                MUTUAL
                if relation["attention"]["mutual"]
                else (ATTEND if strong else SYNC)
            )
            vector = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
            length = max(1.0, float(np.linalg.norm(vector)))
            gap = 26.0 if length > 60 else 0.0
            unit = vector / length
            p = tuple((np.asarray(a) + unit * gap).astype(int))
            q = tuple((np.asarray(b) - unit * gap).astype(int))
            start, end = (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))

            if strong:
                thick = 1 + round(2 * max(ab, ba))
                if ab >= 0.5:
                    cv2.arrowedLine(frame, start, end, colour, thick, tipLength=0.06)
                if ba >= 0.5:
                    cv2.arrowedLine(frame, end, start, colour, thick, tipLength=0.06)
            else:
                _dashed(frame, start, end, SYNC)

            mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            tag = f"{relation['a']}-{relation['b']}"
            if sync["available"] and sync.get("valence") is not None:
                tag += f" r{float(sync['valence']):+.2f}"
            cv2.putText(frame, tag, (mid[0] + 4, mid[1] - 4), FONT, 0.4, colour, 1)

        here = 0
        for tid in entities.get(shot, []):
            row = rows.get((shot, tid, index))
            if not row:
                continue
            here += 1
            x1, y1, x2, y2 = row["body"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
            cv2.putText(
                frame, f"#{tid}", (x1 + 3, y1 + 14), FONT, 0.42, (170, 170, 170), 1
            )

        lines: list[tuple[str, tuple[int, int, int]]] = []
        for relation, ab, ba in shown[:4]:
            place = relation["placement"]
            text = (
                f"{relation['a']}-{relation['b']}  a->b {ab:.2f}  b->a {ba:.2f}  "
                f"{place['horizontal']}/{place['vertical']}  nearer {place['nearer']}"
            )
            if relation["synchrony"]["available"]:
                text += f"  sync {relation['synchrony'].get('valence')}"
            lines.append(
                (text, MUTUAL if relation["attention"]["mutual"] else (210, 210, 210))
            )
        _panel(frame, lines)

        _banner(
            frame,
            f"frame {index:5d}  shot {shot}  people here {here}  "
            f"edges drawn {len(shown)}",
            "green=mutual  amber=attends  cyan dashed=synchrony",
            bool(shown),
        )

    return draw


VIEWS: dict[str, tuple[Callable[[Path], Draw], str]] = {
    "tracks": (_tracks_view, "06-detection-tracking"),
    "conditional": (_conditional_view, "07-conditional-experts"),
    "fusion": (_fusion_view, "09-fusion"),
}


def render(video: Path, out_path: Path, draw: Draw) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    codec = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), codec, fps, size)

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        draw(frame, index)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    return index


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render pipeline output over the source video"
    )
    p.add_argument("video", help="video name under data/processed/, e.g. messi")
    p.add_argument("--views", nargs="+", choices=sorted(VIEWS), default=sorted(VIEWS))
    args = p.parse_args()

    root = ROOT / "data" / "processed" / args.video
    if not root.is_dir():
        sys.exit(f"no processed output at {root}")

    source = ROOT / str(_load(root, "segmentation.json", "02-segmentation")["video"])
    for name in args.views:
        build, produced_by = VIEWS[name]
        out_path = root / f"{name}_preview.mp4"
        frames = render(source, out_path, build(root))
        print(f"{args.video} {name}: {frames} frames -> {out_path}  ({produced_by})")


if __name__ == "__main__":
    main()
