"""Renders 06, 07, 08 and 10 over the source video, draws 12's identikit as a page, and on request a
video's nearest and farthest by the similarity score: python3 -m src.tools.preview messi [--views ...]"""

from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import os
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from config import JSON_DIR, PREVIEWS_DIR

ROOT = Path(__file__).resolve().parents[2]
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
LOOKS = (60, 190, 245)
MUTUAL = (110, 240, 130)
SYNC = (240, 200, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GAZE_HOLD = 8  # frames a gaze sample stays drawn, about two sample gaps
ARROW_MIN = 3  # samples a segment's share needs for an arrow, as 12's --relation-min
LOOK_IN_FRAME = 0.3  # as 10's IN_FRAME
HEAD_REACH = 0.58  # as 10's

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
    path = root / JSON_DIR / name
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


def _gaze_ray(
    frame: Array, face: list[int], gaze: dict[str, Any], colour: tuple[int, int, int]
) -> None:
    """Gaze-LLE's line to its point, or "out" for a look leaving the frame."""
    if float(gaze["in_frame"]) < LOOK_IN_FRAME:
        text = f"out {float(gaze['in_frame']):.2f}"
        cv2.putText(frame, text, (face[0], face[3] + 14), FONT, 0.42, colour, 1)
        return
    target = (int(gaze["target"][0]), int(gaze["target"][1]))
    cv2.line(frame, _centre(face), target, colour, 1, cv2.LINE_AA)
    cv2.circle(frame, target, 7, colour, 2)


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
                    # MediaPipe's y points up and its face along +z; the image's y points down.
                    for vector, axis in (
                        ((1, 0, 0), (0, 0, 255)),
                        ((0, 1, 0), (0, 255, 0)),
                        ((0, 0, 1), (255, 0, 0)),
                    ):
                        end = matrix @ np.asarray(vector, dtype=np.float64) * scale
                        cv2.arrowedLine(
                            frame,
                            (cx, cy),
                            (int(cx + end[0]), int(cy - end[1])),
                            axis,
                            2,
                            tipLength=0.25,
                        )
                if "gaze" in sample:
                    _gaze_ray(frame, row["face"], sample["gaze"], colour)
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
            "skeleton=pose  axes=head pose  line+ring=gaze  magenta=OCR",
            bool(live),
        )

    return draw


# Relations


def _person_tracks(aggregated: dict[str, Any]) -> dict[int, list[tuple[int, int]]]:
    """09 owns the person records; 10 writes edges only."""
    return {
        int(p["person_id"]): [
            (int(t["segment"]), int(t["track_id"])) for t in p["tracks"]
        ]
        for p in aggregated["persons"]
    }


def _head_box(face: list[int]) -> list[int]:
    """The face box grown to a head, as 07 grows it for Gaze-LLE."""
    x1, y1, x2, y2 = face
    w, h = x2 - x1, y2 - y1
    return [
        round(x1 - 0.2 * w),
        round(y1 - 0.4 * h),
        round(x2 + 0.2 * w),
        round(y2 + 0.1 * h),
    ]


def _head_span(body: list[int]) -> tuple[int, int, int]:
    """10's head span for someone without a face: down the body box's centre."""
    x1, y1, x2, y2 = body
    return (x1 + x2) // 2, y1, min(y2, y1 + round(HEAD_REACH * (x2 - x1)))


def _relations_view(root: Path) -> Draw:
    tracks = _load(root, "tracks.json", "06-detection-tracking")
    relations = _load(root, "relations.json", "10-relations")
    aggregated = _load(root, "aggregated.json", "09-aggregation")
    cond = _load(root, "conditional.json", "07-conditional-experts")
    shot_of = _shot_map(tracks)
    rows = _rows(tracks)
    owned = _person_tracks(aggregated)
    here_in = {
        int(s["index"]): [int(p) for p in s["person_ids"]] for s in relations["shots"]
    }
    gazes: dict[tuple[int, int], dict[int, dict[str, Any]]] = {
        (int(s["index"]), int(t["track_id"])): {
            int(x["frame"]): x["gaze"] for x in t["series"] if "gaze" in x
        }
        for s in cond["shots"]
        for t in s["tracks"]
    }

    # (frame, the sample's face, gaze) per person
    looked = {
        pid: sorted(
            (
                (frame, rows[(*key, frame)]["face"], gaze)
                for key in keys
                for frame, gaze in gazes.get(key, {}).items()
            ),
            key=lambda item: item[0],
        )
        for pid, keys in owned.items()
    }

    def row_at(pid: int, index: int) -> dict[str, Any] | None:
        for shot, tid in owned.get(pid, []):
            found = rows.get((shot, tid, index))
            if found is not None:
                return found
        return None

    def gaze_at(pid: int, index: int) -> tuple[list[int], dict[str, Any]] | None:
        """The latest gaze sample and its face, at most GAZE_HOLD frames old."""
        samples = looked.get(pid, [])
        i = bisect.bisect_right(samples, index, key=lambda item: item[0])
        if not i or index - samples[i - 1][0] > GAZE_HOLD:
            return None
        return samples[i - 1][1], samples[i - 1][2]

    owner = {key: pid for pid, keys in owned.items() for key in keys}
    faces_of: dict[int, list[tuple[int, list[int]]]] = {}
    for (s, t, f), r in rows.items():
        if r["face"] and (s, t) in owner:
            faces_of.setdefault(owner[(s, t)], []).append((f, r["face"]))
    for faces in faces_of.values():
        faces.sort(key=lambda item: item[0])

    def face_near(pid: int, index: int) -> list[int] | None:
        """This frame's face, else the latest within GAZE_HOLD frames."""
        faces = faces_of.get(pid, [])
        i = bisect.bisect_right(faces, index, key=lambda item: item[0])
        if i and index - faces[i - 1][0] <= GAZE_HOLD:
            return faces[i - 1][1]
        return None

    def anchor(pid: int, index: int) -> tuple[int, int] | None:
        """The face near this frame, else the middle of 10's head span."""
        row = row_at(pid, index)
        if row is None:
            return None
        face = face_near(pid, index)
        if face:
            return _centre(face)
        x, top, bottom = _head_span(row["body"])
        return x, (top + bottom) // 2

    def in_segment(side: dict[str, Any], segment: int) -> tuple[float, int]:
        """10's share of looks landing on the other in this segment, and its samples."""
        found = [s for s in side.get("by_segment", []) if s["segment"] == segment]
        if not found:
            return 0.0, 0
        hits, n = found[0]["hits"], found[0]["n"]
        return (hits / n if n >= ARROW_MIN else 0.0), n

    def draw(frame: Array, index: int) -> None:
        shot = shot_of.get(index)
        if shot is None:
            return
        shown: list[tuple[dict[str, Any], tuple[float, int, float, int]]] = []

        # Relations are video-scoped, so a pair is drawn wherever both are on screen.
        for relation in relations["relations"]:
            a = anchor(int(relation["a"]), index)
            b = anchor(int(relation["b"]), index)
            if a is None or b is None:
                continue
            # Arrows follow the segment on screen, not the whole video.
            ab, n_ab = in_segment(relation["looks_at"]["a_to_b"], shot)
            ba, n_ba = in_segment(relation["looks_at"]["b_to_a"], shot)
            sync = relation["synchrony"]
            strong = max(ab, ba) >= 0.5
            if not strong and not sync["available"]:
                continue
            shown.append((relation, (ab, n_ab, ba, n_ba)))

            colour = MUTUAL if min(ab, ba) >= 0.5 else (LOOKS if strong else SYNC)
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
        for pid in here_in.get(shot, []):
            row = row_at(pid, index)
            if not row:
                continue
            here += 1
            x1, y1, x2, y2 = row["body"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
            cv2.putText(
                frame, f"P{pid}", (x1 + 3, y1 + 14), FONT, 0.42, (170, 170, 170), 1
            )
            face = face_near(pid, index)
            if face:
                hx1, hy1, hx2, hy2 = _head_box(face)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (150, 150, 150), 1)
            else:
                x, top, bottom = _head_span(row["body"])
                cv2.line(frame, (x, top), (x, bottom), (150, 150, 150), 2)
            look = gaze_at(pid, index)
            if look is not None:
                _gaze_ray(frame, *look, WHITE)

        lines: list[tuple[str, tuple[int, int, int]]] = []
        for relation, (ab, n_ab, ba, n_ba) in shown[:4]:
            place = relation["placement"]
            text = (
                f"P{relation['a']}-P{relation['b']}  a->b {ab:.2f} (n {n_ab})  "
                f"b->a {ba:.2f} (n {n_ba})  {place['horizontal']}/{place['vertical']}"
            )
            if relation["synchrony"]["available"]:
                text += f"  sync {relation['synchrony'].get('valence')}"
            lines.append((text, MUTUAL if min(ab, ba) >= 0.5 else (210, 210, 210)))
        _panel(frame, lines)

        _banner(
            frame,
            f"frame {index:5d}  shot {shot}  people here {here}  "
            f"edges drawn {len(shown)}",
            "white=gaze point  arrows, this segment: amber=one way  green=both  cyan dashed=synchrony",
            bool(shown),
        )

    return draw


# Identity


def _identity_view(root: Path) -> Draw:
    """The tracks view's boxes labelled with 08's person id, to read against tracks_preview.mp4."""
    tracks = _load(root, "tracks.json", "06-detection-tracking")
    con = _load(root, "identity.json", "08-identity")
    shot_of = _shot_map(tracks)
    rows = _rows(tracks)

    person_of: dict[tuple[int, int], dict[str, Any]] = {}
    for person in con["persons"]:
        for track in person["tracks"]:
            person_of[(int(track["segment"]), int(track["track_id"]))] = person

    by_shot: dict[int, list[tuple[int, int]]] = {}
    for shot in tracks["shots"]:
        if not shot["tracked"]:
            continue
        for track in shot["tracks"]:
            by_shot.setdefault(int(shot["index"]), []).append(
                (int(shot["index"]), int(track["track_id"]))
            )

    def draw(frame: Array, index: int) -> None:
        shot = shot_of.get(index)
        if shot is None:
            return
        drawn = merged = 0
        for key in by_shot.get(shot, []):
            row = rows.get((key[0], key[1], index))
            if row is None:
                continue
            person = person_of.get(key)
            if person is None:
                continue
            drawn += 1
            pid = int(person["person_id"])
            colour = PALETTE[pid % len(PALETTE)]
            if person["abstained"]:
                colour = (120, 120, 120)
            elif person["linked"]:
                merged += 1
            x1, y1, x2, y2 = row["body"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            tag = f"P{pid}"
            if person["linked"]:
                tag += f" x{len(person['tracks'])}"
            if person["abstained"]:
                tag += " ?"
            cv2.putText(frame, tag, (x1 + 3, y1 + 15), FONT, 0.45, colour, 1)
            cv2.putText(
                frame,
                f"t{key[1]}",
                (x1 + 3, y2 - 5),
                FONT,
                0.36,
                (150, 150, 150),
                1,
            )

        _banner(
            frame,
            f"frame {index:5d}  shot {shot}  boxes {drawn}  merged {merged}  "
            f"{con['track_count']} tracks -> {con['person_count']} people",
            "P=person id  x N=tracks merged  ?=abstained  t=06 track id",
            bool(drawn),
        )

    return draw


# Identikit page

PAGE_VIEW = "identikit"
PAGE_SIZE = (22.4, 13.5)  # inches at 100 dpi, the size of the drawn target
ASPECT = PAGE_SIZE[0] / PAGE_SIZE[1]
LINE = 1.45 / (PAGE_SIZE[1] * 72)  # figure height of one line, per point of font size
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
PAGE, SURFACE, HAIRLINE, BASELINE = "#f9f9f7", "#fcfcfb", "#e1e0d9", "#c3c2b7"
# The reference palette's leading slots, in its order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SEGMENT_FILL = "#cde2fb"
CHECK = {
    "consistent": "ok",
    "under_tracked": "under-tracked",
    "turnover_or_fragmented": "turnover",
    "empty": "empty",
}


class Column:
    """Writes top-down in figure coordinates and stops at its floor."""

    def __init__(
        self, fig: Figure, x: float, top: float, width: float, floor: float
    ) -> None:
        self.fig, self.x, self.y, self.width, self.floor = fig, x, top, width, floor

    def room(self, height: float) -> bool:
        return self.y - height >= self.floor

    def text(
        self,
        text: str,
        size: float = 10.5,
        colour: str = INK,
        weight: str = "normal",
        indent: float = 0.0,
    ) -> None:
        chars = max(16, int((self.width - indent) * PAGE_SIZE[0] * 100 / (size * 0.8)))
        for line in textwrap.wrap(text, chars) or [""]:
            if not self.room(size * LINE):
                return
            self.fig.text(
                self.x + indent,
                self.y,
                line,
                fontsize=size,
                color=colour,
                weight=weight,
                va="top",
            )
            self.y -= size * LINE

    def heading(self, text: str) -> None:
        self.y -= 0.008
        self.text(text, size=12.5, weight="bold")
        self.y -= 0.002

    def axes(self, height: float, indent: float = 0.0) -> Axes | None:
        if not self.room(height):
            return None
        ax = self.fig.add_axes(
            (self.x + indent, self.y - height, self.width - indent, height)
        )
        _bare(ax)
        self.y -= height + 0.006
        return ax


def _bare(ax: Axes) -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _box(fig: Figure, rect: tuple[float, float, float, float], title: str) -> Column:
    x, y, w, h = rect
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.008",
            mutation_aspect=ASPECT,
            transform=fig.transFigure,
            facecolor=SURFACE,
            edgecolor=HAIRLINE,
            linewidth=1.2,
            zorder=-1,
        )
    )
    fig.text(
        x + w / 2,
        y + h - 0.012,
        title,
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
        color=INK,
    )
    fig.add_artist(
        Line2D(
            [x + 0.01, x + w - 0.01],
            [y + h - 0.047] * 2,
            transform=fig.transFigure,
            color=HAIRLINE,
            linewidth=1,
        )
    )
    return Column(fig, x + 0.012, y + h - 0.056, w - 0.024, y + 0.008)


def _legend(c: Column, items: list[tuple[str, str]], size: float = 9.5) -> None:
    x = c.x
    for colour, label in items:
        c.fig.add_artist(
            Rectangle(
                (x, c.y - 0.012),
                0.007,
                0.011,
                transform=c.fig.transFigure,
                facecolor=colour,
                edgecolor="none",
            )
        )
        c.fig.text(x + 0.010, c.y, label, fontsize=size, color=SECONDARY, va="top")
        x += 0.022 + len(label) * size * 0.8 / (PAGE_SIZE[0] * 100)
    c.y -= size * LINE + 0.004


def _sd(stats: dict[str, Any], digits: int = 1) -> str:
    return f" sd {stats['sd']:.{digits}f}" if stats.get("sd") is not None else ""


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _header(fig: Figure, kit: dict[str, Any], name: str) -> None:
    s = kit["summary"]
    fps = f" @ {s['fps']:g} fps" if s.get("fps") else ""
    fig.text(
        0.018, 0.988, "Video identikit", fontsize=26, weight="bold", color=INK, va="top"
    )
    fig.text(
        0.018,
        0.95,
        f"{name}.mp4 · {s['frames']} frames{fps} · {s['duration_s']:.1f} s · {_count(s['segments'], 'segment')}",
        fontsize=13,
        color=SECONDARY,
        va="top",
    )
    p = s["persons"]
    tiles = (
        ("persons resolved", p["resolved"]),
        ("described", p["described"]),
        ("glimpsed", p["glimpsed"]),
        ("tracked only", p["tracked_only"]),
        ("abstained", p["abstained"]),
        ("thin records", p["thin"]),
    )
    for i, (label, value) in enumerate(tiles):
        x = 0.46 + i * 0.088
        fig.text(
            x, 0.985, str(value), fontsize=28, weight="semibold", color=INK, va="top"
        )
        fig.text(x, 0.94, label, fontsize=10.5, color=MUTED, va="top")


def _timeline(fig: Figure, kit: dict[str, Any]) -> None:
    _box(fig, (0.015, 0.70, 0.97, 0.225), "Timeline")
    segments = kit["timeline"]["segments"]
    rhythm = kit["timeline"]["rhythm"]
    duration = float(kit["summary"]["duration_s"]) or 1.0
    left, width = 0.105, 0.87

    fig.text(
        0.027,
        0.874,
        "SEGMENTS",
        fontsize=10.5,
        weight="bold",
        color=SECONDARY,
        va="top",
    )
    rhythm_text = f"{_count(rhythm['shots'], 'shot')}\nmean {rhythm['mean_s']:.2f} s\n{rhythm['shots_per_minute']:.1f} per min"
    fig.text(
        0.027, 0.856, rhythm_text, fontsize=9, color=MUTED, va="top", linespacing=1.4
    )
    band = fig.add_axes((left, 0.845, width, 0.03))
    _bare(band)
    band.set_xlim(0, duration)
    band.set_ylim(0, 1)
    for s in segments:
        band.add_patch(
            Rectangle(
                (s["start"], 0),
                s["seconds"],
                1,
                facecolor=SEGMENT_FILL,
                edgecolor=SURFACE,
                linewidth=2,
            )
        )
        if s["seconds"] / duration * width * PAGE_SIZE[0] * 100 >= 18:
            band.text(
                s["start"] + s["seconds"] / 2,
                0.5,
                str(s["segment"]),
                ha="center",
                va="center",
                fontsize=9,
                color=INK,
            )
    for t in np.linspace(0, duration, 7):
        fig.text(
            left + width * float(t) / duration,
            0.84,
            f"{t:.1f} s",
            fontsize=8.5,
            color=MUTED,
            ha="center",
            va="top",
        )

    fig.text(
        0.027,
        0.812,
        "PRESENCE",
        fontsize=10.5,
        weight="bold",
        color=SECONDARY,
        va="top",
    )
    persons = sorted(
        kit["timeline"]["presence"],
        key=lambda p: (-float(p["seconds"]), int(p["person_id"])),
    )
    shown = persons[:8]
    if not shown:
        fig.text(left, 0.79, "nobody was followed", fontsize=10, color=MUTED, va="top")
        return
    ax = fig.add_axes((left, 0.722, width, 0.09))
    _bare(ax)
    ax.set_xlim(0, duration)
    ax.set_ylim(len(shown) - 0.5, -0.5)
    span = {s["segment"]: (s["start"], s["seconds"]) for s in segments}
    for row, p in enumerate(shown):
        for segment in p["segments"]:
            start, seconds = span[segment]
            ax.barh(
                row,
                seconds,
                left=start,
                height=0.55,
                color=SERIES[0],
                edgecolor=SURFACE,
                linewidth=1,
            )
        label = f"P{p['person_id']}  {p['seconds']:.1f} s"
        ax.text(
            -duration * 0.004,
            row,
            label,
            ha="right",
            va="center",
            fontsize=8.5,
            color=SECONDARY,
            clip_on=False,
        )

    notes: list[str] = []
    if len(persons) > len(shown):
        notes.append(f"{len(persons) - len(shown)} more persons not drawn")
    back = [
        f"P{p['person_id']}"
        for p in persons
        if any(b - a > 1 for a, b in itertools.pairwise(sorted(p["segments"])))
    ]
    if back:
        notes.append(
            f"{', '.join(back[:6])} reappear after an absence: one person across segments, resolved by 08"
        )
    fig.text(
        left,
        0.716,
        " · ".join(notes),
        fontsize=9,
        color=MUTED,
        style="italic",
        va="top",
    )


def _spark_row(
    c: Column, label: str, xs: list[float], ys: list[Any], span: float
) -> None:
    height = 0.03
    if not c.room(height):
        return
    c.fig.text(c.x, c.y - 0.006, label, fontsize=10.5, color=INK, va="top")
    ax = c.fig.add_axes((c.x + 0.13, c.y - height, c.width - 0.13, height))
    _bare(ax)
    points = [(x, float(y)) for x, y in zip(xs, ys) if y is not None]
    if points:
        px, py = [p[0] for p in points], [p[1] for p in points]
        ax.plot(
            px,
            py,
            color=SERIES[0],
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.plot(
            px[-1:],
            py[-1:],
            "o",
            color=SERIES[0],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        pad = (max(py) - min(py)) * 0.2 or 0.01
        ax.set_xlim(0, span)
        ax.set_ylim(min(py) - pad, max(py) + pad)
    c.y -= height + 0.006


def _plastic_column(fig: Figure, kit: dict[str, Any]) -> None:
    c = _box(fig, (0.015, 0.015, 0.315, 0.67), "Plastic level")
    segments = kit["plastic"]["segments"]
    video = kit["plastic"]["video"]
    timeline = kit["timeline"]["segments"]
    mids = [s["start"] + s["seconds"] / 2 for s in timeline]
    duration = float(kit["summary"]["duration_s"]) or 1.0

    c.heading("Chromatic categories")
    grey = video.get("grayscale")
    chroma = video.get("chroma")
    grey_text = "unknown" if grey is None else ("yes" if grey else "no")
    c.text(
        f"Grayscale: {grey_text}"
        + (f" · chroma {chroma['mean']:.3f}" if chroma else "")
    )
    medium = video.get("medium")
    warmth = video.get("warmth")
    c.text(
        f"Medium: {medium['label'] if medium else 'unknown'} · temperature {video.get('temperature') or 'unknown'}"
        + (f", b* {warmth['mean']:+.1f}" if warmth else "")
    )
    for key, label in (("brightness", "Brightness"), ("saturation", "Saturation")):
        stats = video.get(key)
        if stats:
            values = [s["visual"].get(key) for s in segments]
            _spark_row(
                c, f"{label} {stats['mean']:.3f}{_sd(stats, 3)}", mids, values, duration
            )
    c.text("Palette, per segment", colour=SECONDARY)
    ax = c.axes(0.045)
    if ax is not None:
        ax.set_xlim(0, len(segments))
        ax.set_ylim(0, 1)
        for i, s in enumerate(segments):
            colours = s.get("palette") or []
            total = sum(float(p["share"]) for p in colours) or 1.0
            top = 1.0
            for p in colours:
                h = float(p["share"]) / total
                ax.add_patch(
                    Rectangle(
                        (i, top - h),
                        1,
                        h,
                        facecolor=p["hex"],
                        edgecolor=SURFACE,
                        linewidth=1.5,
                    )
                )
                top -= h
        step = max(1, math.ceil(len(segments) / 14))
        for i in range(0, len(segments), step):
            ax.text(
                i + 0.5,
                -0.06,
                str(segments[i]["segment"]),
                ha="center",
                va="top",
                fontsize=8,
                color=MUTED,
                clip_on=False,
            )
        c.y -= 0.012

    c.heading("Rhythm and montage")
    r = kit["timeline"]["rhythm"]
    c.text(
        f"{_count(r['shots'], 'shot')} · {r['shots_per_minute']:.1f} per minute · length mean {r['mean_s']:.2f} s, "
        f"min {r['min_s']:.2f} s, max {r['max_s']:.2f} s"
    )
    ax = c.axes(0.03)
    if ax is not None:
        seconds = [float(s["seconds"]) for s in timeline]
        slot = c.width * PAGE_SIZE[0] * 100 / len(seconds)
        ax.bar(range(len(seconds)), seconds, width=min(0.7, 24 / slot), color=SERIES[0])
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_xlim(-0.6, len(seconds) - 0.4)
        ax.set_ylim(0, max(seconds) * 1.05)

    c.heading("Topological categories")
    weight: dict[str, float] = {}
    for s, t in zip(segments, timeline):
        for name, share in s["coverage"].items():
            weight[name] = weight.get(name, 0.0) + float(share) * float(t["seconds"])
    classes = [
        k
        for k, _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))[
            : len(SERIES)
        ]
    ]
    c.text("Spatial coverage per segment", colour=SECONDARY)
    ax = c.axes(min(0.075, 0.0085 * len(segments) + 0.012), indent=0.018)
    if ax is not None:
        for row, s in enumerate(segments):
            start = 0.0
            for k, name in enumerate(classes):
                share = float(s["coverage"].get(name, 0.0))
                if share:
                    ax.barh(
                        row,
                        share,
                        left=start,
                        height=0.8,
                        color=SERIES[k],
                        edgecolor=SURFACE,
                        linewidth=1,
                    )
                start += share
            ax.barh(
                row,
                max(0.0, 1.0 - start),
                left=start,
                height=0.8,
                color=BASELINE,
                edgecolor=SURFACE,
                linewidth=1,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(len(segments) - 0.5, -0.5)
        fit = max(
            1, int(ax.get_position().height * PAGE_SIZE[1] * 100 / 12)
        )  # one label per 12 px
        step = max(1, math.ceil(len(segments) / fit))
        for row in range(0, len(segments), step):
            ax.text(
                -0.01,
                row,
                str(segments[row]["segment"]),
                ha="right",
                va="center",
                fontsize=7.5,
                color=MUTED,
                clip_on=False,
            )
        _legend(
            c,
            [(SERIES[k], name.strip()) for k, name in enumerate(classes)]
            + [(BASELINE, "other")],
        )

    if not c.room(2 * 10.5 * LINE):
        return
    c.text("Objects near to far, relative depth", colour=SECONDARY)
    busy = [s for s in segments if s.get("depth") and s["depth"]["objects"]]
    busy = sorted(
        sorted(busy, key=lambda s: -len(s["depth"]["objects"]))[:4],
        key=lambda s: s["segment"],
    )
    for s in busy:
        d = s["depth"]
        background = (
            f" · background {d['background']:.2f}"
            if d["background"] is not None
            else ""
        )
        main = s["centralities"][0] if s["centralities"] else None
        where = (
            f" · {main['label']} {main['horizontal']} {main['vertical']}, {main['area']:.0%}"
            if main
            else ""
        )
        c.text(
            f"seg {s['segment']}: {' > '.join(d['order'][:4])}{background}{where}",
            size=9.5,
            indent=0.006,
        )
    if not busy:
        c.text("no objects found", size=9.5, colour=MUTED)


def _card(c: Column, p: dict[str, Any], watch: dict[str, Any] | None) -> None:
    head = (watch or {}).get("head_pose")
    gaze = (watch or {}).get("gaze")
    segments = ", ".join(str(s) for s in p["segments"])
    c.text(
        f"Person {p['person_id']} · segments {segments} · {p['seconds']:.1f} s · "
        f"{_count(len(p['tracks']), 'track')} · {_count(p['support']['series'], 'sample')}",
        size=10,
        weight="bold",
    )
    lines: list[str] = []
    if p.get("age"):
        lines.append(f"Age {p['age']['mean']:.1f}{_sd(p['age'])} (n={p['age']['n']})")
    if p.get("gender"):
        g = p["gender"]
        lines.append(
            f"Gender {g['label']}, agreement {g['agreement']:.0%} (n={g['n']})"
        )
    if p.get("perceived_ethnicity"):
        e = p["perceived_ethnicity"]
        lines.append(
            f"Perceived ethnicity {e['modal']}, agreement {e['agreement']:.0%} (n={e['n']}, {e['taxonomy']})"
        )
    if p.get("demographic_check"):
        d = p["demographic_check"]
        lines.append(
            f"FairFace check: gender agrees {d['gender_agreement']:.0%}, age in bracket {d['age_in_bracket']:.0%} (n={d['n']})"
        )
    if p.get("emotion"):
        e = p["emotion"]
        lines.append(
            f"Emotion {e['modal']}, agreement {e['agreement']:.0%} (n={e['n']})"
        )
    if p.get("movement"):
        m = p["movement"]
        lines.append(f"Body movement {m['mean']:.3f}{_sd(m, 3)} (n={m['n']})")
    yaw = (head or {}).get("yaw")
    if yaw:
        lines.append(f"Head yaw {yaw['mean']:.1f} deg{_sd(yaw)} (n={yaw['n']})")
    if gaze:
        lines.append(
            f"Gaze inside the frame {gaze['in_frame_share']:.0%} (n={gaze['n']})"
        )
    address = (watch or {}).get("address")
    if address:
        lines.append(f"Looks into the camera {address['share']:.0%} (n={address['n']})")
    per = [
        f"seg {b['segment']} {b['emotion']['modal']} {round(b['emotion']['agreement'] * b['emotion']['n'])}/{b['emotion']['n']}"
        for b in p.get("by_segment") or []
        if b.get("emotion")
    ]
    if per:
        lines.append("Per segment: " + " · ".join(per))
    for line in lines:
        c.text(line, size=9.5, indent=0.008)
    c.y -= 0.004


def _figurative_column(fig: Figure, kit: dict[str, Any]) -> None:
    c = _box(fig, (0.345, 0.015, 0.315, 0.67), "Figurative level")
    figurative = kit["figurative"]
    caption = figurative.get("caption")
    c.heading("Caption")
    if caption:
        c.text(caption["text"], size=11.5)
        unsupported = ", ".join(caption["unsupported"]) or "none"
        c.text(
            f"{caption['words']} words · CLIP score {caption['clip_score']:.2f} · named but not detected: {unsupported}",
            size=9,
            colour=MUTED,
        )
    else:
        c.text(
            f"not available: {kit['gaps']['pending'].get('figurative.caption', '')}",
            colour=MUTED,
        )

    # Place only video-wide: per segment Places365 is too often a confident wrong category.
    c.heading("Whole video")
    video = figurative["video"]
    io = video.get("indoor_outdoor")
    places = ", ".join(f"{p['label']} {p['score']:.2f}" for p in video["place"][:3])
    c.text(
        " · ".join(
            v for v in (f"{io['label']} {io['outdoor']:.2f}" if io else "", places) if v
        )
        or "place unknown",
        size=10,
    )
    tags = ", ".join(t["label"] for t in video["tags"][:5])
    objects = ", ".join(o["label"] for o in video["objects"][:5])
    c.text(
        f"Tags: {tags or 'none'} · objects: {objects or 'none'}",
        size=9.5,
        colour=SECONDARY,
    )
    # The readings themselves are not drawn: the CRNN's ten-character chunks make up words.
    totals = kit["summary"]["text"]
    c.text(
        f"Text: {totals['readings']} readings in {totals['distinct']} groups, not shown: "
        "the OCR splits words at arbitrary points"
        if totals["readings"]
        else "Text: none read",
        size=9.5,
        colour=SECONDARY,
    )

    c.heading("Per segment")
    times = {s["segment"]: s for s in kit["timeline"]["segments"]}
    segments = figurative["segments"]
    for s in segments[:2]:
        t = times[s["segment"]]
        io = s.get("indoor_outdoor")
        where = (
            f"{io['label']} {io['outdoor']:.2f}" if io else "indoor or outdoor unknown"
        )
        c.text(
            f"seg {s['segment']} · {t['start']:.1f}-{t['end']:.1f} s · {where} · {s['crowdedness']}",
            size=9.5,
            weight="bold",
        )
        c.text(
            ", ".join(s["tags"]) or "no tags", size=9.5, colour=SECONDARY, indent=0.008
        )
    if len(segments) > 2:
        c.text(f"+{len(segments) - 2} more segments", size=9, colour=MUTED)

    c.heading("Content participants")
    subjects = figurative.get("subjects") or []
    if subjects:
        c.text(
            "Non-human subjects: "
            + ", ".join(f"{s['label']} {s['seconds']:.1f} s" for s in subjects),
            size=10,
        )
    persons = sorted(
        (p for p in figurative["persons"] if p["status"] == "described"),
        key=lambda p: (-int(p["support"]["series"]), int(p["person_id"])),
    )
    heads = {h["person_id"]: h for h in kit["enunciative"]["persons"]}
    for p in persons[:2]:
        _card(c, p, heads.get(p["person_id"]))
    rest = [f"P{p['person_id']}" for p in persons[2:]]
    also = (
        f"Also described: {', '.join(rest[:8])}"
        + (f" +{len(rest) - 8} more" if len(rest) > 8 else "")
        if rest
        else ""
    )
    # Tracked-only and unlinked counts are in the header.
    glimpsed = kit["summary"]["persons"]["glimpsed"]
    notes = [also] if also else []
    if glimpsed:
        notes.append(f"{glimpsed} glimpsed, each value from one sample")
    if notes:
        c.text("; ".join(notes), size=9, colour=MUTED)


def _graph(c: Column, relations: list[dict[str, Any]], floor: float) -> None:
    if not relations:
        c.text("no relation with evidence", size=9.5, colour=MUTED)
        return
    degree: dict[int, int] = {}
    for relation in relations:
        for pid in relation["pair"]:
            degree[pid] = degree.get(pid, 0) + 1
    nodes = [
        pid for pid, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ]
    height = 0.16
    if not c.room(height + 0.03):
        return
    width = height / ASPECT
    ax = c.fig.add_axes((c.x + (c.width - width) / 2, c.y - height, width, height))
    _bare(ax)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)
    at = {pid: (float(np.cos(a)), float(np.sin(a))) for pid, a in zip(nodes, angles)}
    for relation in relations:
        a, b = relation["pair"]
        if a not in at or b not in at:
            continue
        looks = relation.get("looks_at") or {}
        ab, ba = (
            bool(side and side.get("peak") and side["peak"]["share"] >= floor)
            for side in (looks.get("a_to_b"), looks.get("b_to_a"))
        )
        # One colour per edge, strongest first: mutual, looks at, synchrony.
        if ab and ba:
            colour, src, dst, arrow = SERIES[2], a, b, "-"
        elif ab or ba:
            colour, src, dst, arrow = (
                (SERIES[0], a, b, "-|>") if ab else (SERIES[0], b, a, "-|>")
            )
        else:
            colour, src, dst, arrow = SERIES[1], a, b, "-"
        ax.annotate(
            "",
            xy=at[dst],
            xytext=at[src],
            arrowprops={
                "arrowstyle": arrow,
                "color": colour,
                "lw": 2,
                "shrinkA": 13,
                "shrinkB": 13,
            },
        )
    for pid, (x, y) in at.items():
        ax.plot(
            [x],
            [y],
            "o",
            markersize=22,
            color=SEGMENT_FILL,
            markeredgecolor=SERIES[0],
            markeredgewidth=1.2,
        )
        ax.text(x, y, f"P{pid}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    c.y -= height + 0.006
    _legend(
        c, [(SERIES[0], "looks at"), (SERIES[2], "mutual"), (SERIES[1], "synchrony")]
    )
    if len(degree) > len(nodes):
        c.text(
            f"{len(degree) - len(nodes)} more persons in these relations not drawn",
            size=9,
            colour=MUTED,
        )


def _camera_cell(camera: dict[str, Any]) -> str:
    """The label, with a direction only when the label is the move; the moves line has the rest."""
    cell = camera.get("movement") or "-"
    if camera.get("direction") and camera.get("move") in (None, cell):
        cell += f" {camera['direction']}"
    return cell


def _framing_table(c: Column, segments: list[dict[str, Any]]) -> None:
    size = 9.0
    columns = (
        ("seg", 0.0),
        ("visible", 0.022),
        ("followed", 0.062),
        ("shot", 0.105),
        ("camera", 0.16),
        ("framing", 0.195),
        ("check", 0.232),
    )
    for label, dx in columns:
        c.fig.text(c.x + dx, c.y, label, fontsize=size, color=MUTED, va="top")
    c.y -= size * LINE
    shown = 0
    for s in segments:
        if not c.room(2 * size * LINE):
            break
        distance = s.get("camera_distance")
        camera = s.get("camera") or {}
        cells = (
            str(s["segment"]),
            str(s["visible_at_once"]),
            str(s["followed"]),
            distance["scale"] if distance else "-",
            _camera_cell(camera),
            s.get("portrait_scene") or "-",
            CHECK.get(s["assessment"], s["assessment"]),
        )
        for (_, dx), cell in zip(columns, cells):
            c.fig.text(c.x + dx, c.y, str(cell), fontsize=size, color=INK, va="top")
        c.y -= size * LINE
        shown += 1
    if shown < len(segments):
        c.text(f"+{len(segments) - shown} more segments", size=size, colour=MUTED)


def _enunciative_column(fig: Figure, kit: dict[str, Any]) -> None:
    c = _box(fig, (0.675, 0.015, 0.31, 0.67), "Enunciative level")
    r = kit["summary"]["relations"]
    c.heading("Relations between persons")
    c.text(f"Co-present pairs {r['co_present']} · with evidence {r['with_evidence']}")
    c.text(
        f"Looks at {r['looks_at']} · mutual {r['mutual']} · "
        f"synchrony {r['synchrony']} · cross-segment {r['cross_segment']}"
    )
    _graph(c, kit["enunciative"]["relations"], float(r["share_floor"]))
    c.heading("Watcher-looked system")
    persons = kit["enunciative"]["persons"]
    posed = sum(1 for p in persons if p.get("head_pose"))
    c.text(f"Head pose for {_count(posed, 'person')}, mean and sd over their samples")
    gazes = [p["gaze"] for p in persons if p.get("gaze")]
    samples = sum(int(g["n"]) for g in gazes)
    if samples:
        inside = sum(float(g["in_frame_share"]) * int(g["n"]) for g in gazes) / samples
        c.text(
            f"Gaze for {_count(len(gazes), 'person')}: inside the frame in {inside:.0%} of {samples} samples"
        )
    else:
        c.text("Gaze: none measured", colour=MUTED)
    addressed = [p for p in persons if p.get("address")]
    looks = sum(int(p["address"]["n"]) for p in addressed)
    if looks:
        share = (
            sum(
                float(p["address"]["share"]) * int(p["address"]["n"]) for p in addressed
            )
            / looks
        )
        mostly = [
            f"P{p['person_id']}"
            for p in addressed
            if p["address"]["share"] >= 0.5 and p["address"]["n"] >= 2
        ]
        c.text(
            f"Looks into the camera in {share:.0%} of {looks} samples"
            + (f"; mostly {', '.join(mostly[:6])}" if mostly else "")
        )
    c.heading("Framing per segment")
    moves = kit["enunciative"].get("camera") or {}
    turns = kit["enunciative"].get("camera_moves") or {}
    if moves:
        line = "Camera: " + ", ".join(f"{k} {v:.0%}" for k, v in moves.items())
        if turns:
            line += " · moves: " + ", ".join(f"{k} {v:.0%}" for k, v in turns.items())
        c.text(line, size=9.5)
    _framing_table(c, kit["enunciative"]["segments"])


def _identikit_page(root: Path, out_path: Path) -> None:
    path = root / "identikit.json"
    if not path.exists():
        sys.exit(f"missing {path}; run 12-identikit first")
    kit: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    _header(fig, kit, root.name)
    _timeline(fig, kit)
    _plastic_column(fig, kit)
    _figurative_column(fig, kit)
    _enunciative_column(fig, kit)
    fig.savefig(out_path, facecolor=PAGE)


# Similarity page, after FRESCO's Figure 11: the reference, then for each score its nearest,
# median and farthest videos in the collection src.score.similarity compared.

SIMILARITY_VIEW = "similarity"
SCORE_REPORT = ROOT / "data" / "score" / "similarity.json"
RANKED = 8  # videos per block, as in the figure
SCOPES = (
    ("overall", "Overall score"),
    ("plastic", "Plastic score"),
    ("figurative", "Figurative score"),
    ("enunciative", "Enunciative score"),
)
BLOCKS = ("Top 8: most similar", "Median 8", "Last 8: least similar")
STRIP = 6  # the reference's keyframes over time
THUMB_WIDTH = 360  # pixels a keyframe is reduced to before drawing


def _keyframes(root: Path) -> tuple[list[Path], Array | None]:
    """Keyframe paths in time order, with their 02 embeddings when present."""
    path = root / JSON_DIR / "segmentation.json"
    if not path.exists():
        return [], None
    shots = json.loads(path.read_text(encoding="utf-8"))["shots"]
    paths = [root / k["path"] for s in shots for k in s["keyframes"]]
    vectors = root / "embeddings" / "keyframe_embeddings.npy"
    return paths, np.load(vectors) if vectors.exists() else None


def _picture(path: Path) -> Array | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (THUMB_WIDTH, max(1, round(height * THUMB_WIDTH / width))),
        interpolation=cv2.INTER_AREA,
    )
    out: Array = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return out


def _typical_frame(root: Path) -> Array | None:
    """The keyframe whose embedding is nearest the video's mean: its most typical picture."""
    paths, vectors = _keyframes(root)
    if not paths:
        return None
    pick = len(paths) // 2
    if vectors is not None and len(vectors) == len(paths):
        mean = vectors.mean(axis=0)
        pick = int(np.argmax(vectors @ (mean / (np.linalg.norm(mean) or 1.0))))
    return _picture(paths[pick])


def _thumb(
    fig: Figure, rect: tuple[float, float, float, float], image: Array | None
) -> None:
    ax = fig.add_axes(rect)
    _bare(ax)
    ax.set_anchor("C")
    if image is None:
        ax.set_facecolor(HAIRLINE)
        ax.text(
            0.5, 0.5, "no keyframe", ha="center", va="center", fontsize=8, color=MUTED
        )
    else:
        ax.imshow(image)


def _shared_prefix(names: list[str]) -> str:
    """What every name starts with, up to its last separator (a download site's tag): left off labels."""
    common = os.path.commonprefix(names) if len(names) > 1 else ""
    return common[: max(common.rfind(s) for s in "_.-") + 1]


def _label(name: str, prefix: str, width: int) -> str:
    short = name[len(prefix) :]
    return short if len(short) <= width else short[: width - 1] + "…"


def _ranked(row: list[float | None], me: int) -> list[tuple[int, float]]:
    """Every other measured video, nearest first."""
    found = [(j, float(d)) for j, d in enumerate(row) if j != me and d is not None]
    return sorted(found, key=lambda jd: (jd[1], jd[0]))


def _reference(
    fig: Figure, root: Path, name: str, prefix: str, report: dict[str, Any]
) -> None:
    c = _box(fig, (0.015, 0.03, 0.165, 0.875), "Reference")
    height = 0.2
    _thumb(fig, (c.x, c.y - height, c.width, height), _typical_frame(root))
    c.y -= height + 0.01
    c.text(_label(name, prefix, 24), size=12.5, weight="bold")
    kit_path = root / "identikit.json"
    if kit_path.exists():
        kit = json.loads(kit_path.read_text(encoding="utf-8"))
        s = kit["summary"]
        c.text(
            f"{s['duration_s']:.1f} s · {_count(s['segments'], 'segment')} · {s.get('medium') or 'medium unknown'}",
            size=10,
            colour=SECONDARY,
        )
        p = s["persons"]
        c.text(
            f"{p['described']} persons described, {p['resolved']} resolved",
            size=10,
            colour=SECONDARY,
        )
        caption = (kit["figurative"].get("caption") or {}).get("text")
        if caption:
            c.y -= 0.006
            c.text(f'"{caption}"', size=10)
    c.y -= 0.012
    c.text("Keyframes over time", size=10, colour=MUTED)
    paths, _ = _keyframes(root)
    chosen = (
        [
            paths[int(i)]
            for i in np.linspace(0, len(paths) - 1, min(STRIP, len(paths))).round()
        ]
        if paths
        else []
    )
    cell_w, cell_h = (c.width - 0.006) / 2, 0.085
    for i, path in enumerate(chosen):
        x = c.x + (i % 2) * (cell_w + 0.006)
        y = c.y - (i // 2 + 1) * (cell_h + 0.006)
        if y < c.floor:
            break
        _thumb(fig, (x, y, cell_w, cell_h), _picture(path))
    c.y -= math.ceil(len(chosen) / 2) * (cell_h + 0.006) + 0.01
    c.text(
        f"Compared with {len(report['videos']) - 1} videos, calibration {report['calibration']}. "
        "Distances run from 0, the same, to 1."
        + (f" Names leave out the {prefix} they all start with." if prefix else ""),
        size=9,
        colour=MUTED,
    )


def _similarity_page(root: Path, out_path: Path, report_path: Path) -> None:
    if not report_path.exists():
        sys.exit(f"missing {report_path}; run python3 -m src.score.similarity first")
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    names: list[str] = report["videos"]
    if root.name not in names:
        sys.exit(f"{root.name} is not in {report_path}; rerun the score with it")
    me = names.index(root.name)
    processed = root.parent
    prefix = _shared_prefix(names)

    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(
        0.018,
        0.988,
        "Video similarity",
        fontsize=26,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.018,
        0.95,
        f"{_label(root.name, prefix, 24)} against the collection, nearest to farthest by each level of the score, "
        "after FRESCO's Figure 11 (Morra et al., CVIU 2024)",
        fontsize=13,
        color=SECONDARY,
        va="top",
    )
    _reference(fig, root, root.name, prefix, report)

    left, right, top, bottom = 0.2, 0.985, 0.895, 0.03
    gap, label_h = 0.008, 0.017
    block_w = (right - left - 2 * gap) / 3
    cell_w = block_w / 4
    row_h = (top - bottom - 0.035) / len(SCOPES)
    cell_h = (row_h - 0.014) / 2 - label_h
    for b, title in enumerate(BLOCKS):
        fig.text(
            left + b * (block_w + gap) + block_w / 2,
            top,
            title,
            ha="center",
            va="top",
            fontsize=15,
            weight="bold",
            color=INK,
        )

    stills: dict[str, Array | None] = {}
    for r, (scope, label) in enumerate(SCOPES):
        y_top = top - 0.035 - r * row_h
        fig.text(
            left - 0.008,
            y_top - row_h / 2,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=12.5,
            weight="bold",
            color=INK,
        )
        ranked = _ranked(report["distance"][scope][me], me)
        start = max(0, (len(ranked) - RANKED) // 2)
        blocks = (ranked[:RANKED], ranked[start : start + RANKED], ranked[-RANKED:])
        for b, block in enumerate(blocks):
            x0 = left + b * (block_w + gap)
            fig.add_artist(
                FancyBboxPatch(
                    (x0, y_top - row_h + 0.008),
                    block_w,
                    row_h - 0.008,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    mutation_aspect=ASPECT,
                    transform=fig.transFigure,
                    facecolor=SURFACE,
                    edgecolor=HAIRLINE,
                    linewidth=1,
                    zorder=-1,
                )
            )
            for i, (j, distance) in enumerate(block):
                other = names[j]
                if other not in stills:
                    stills[other] = _typical_frame(processed / other)
                x = x0 + (i % 4) * cell_w + 0.003
                y = y_top - 0.004 - (i // 4 + 1) * (cell_h + label_h)
                _thumb(fig, (x, y + label_h, cell_w - 0.006, cell_h), stills[other])
                fig.text(
                    x + (cell_w - 0.006) / 2,
                    y + label_h - 0.002,
                    f"{_label(other, prefix, 12)} · {distance:.2f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=SECONDARY,
                )
    fig.savefig(out_path, facecolor=PAGE)


VIEWS: dict[str, tuple[Callable[[Path], Draw], str]] = {
    "tracks": (_tracks_view, "06-detection-tracking"),
    "conditional": (_conditional_view, "07-conditional-experts"),
    "identity": (_identity_view, "08-identity"),
    "relations": (_relations_view, "10-relations"),
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
    views = sorted([*VIEWS, PAGE_VIEW])
    p.add_argument(
        "--views",
        nargs="+",
        choices=[*views, SIMILARITY_VIEW],
        default=views,
        help=f"{SIMILARITY_VIEW} is drawn only when asked: it needs src.score.similarity's report",
    )
    p.add_argument(
        "--score", type=Path, default=SCORE_REPORT, help="the similarity report"
    )
    args = p.parse_args()

    root = ROOT / "data" / "processed" / args.video
    if not root.is_dir():
        sys.exit(f"no processed output at {root}")

    previews = root / PREVIEWS_DIR
    previews.mkdir(parents=True, exist_ok=True)
    for name in args.views:
        if name == PAGE_VIEW:
            out_path = previews / "identikit.png"
            _identikit_page(root, out_path)
            print(f"{args.video} {name}: page -> {out_path}  (12-identikit)")
            continue
        if name == SIMILARITY_VIEW:
            out_path = previews / "similarity.png"
            _similarity_page(root, out_path, args.score)
            print(f"{args.video} {name}: page -> {out_path}  ({args.score.name})")
            continue
        source = ROOT / str(
            _load(root, "segmentation.json", "02-segmentation")["video"]
        )
        build, produced_by = VIEWS[name]
        out_path = previews / f"{name}_preview.mp4"
        frames = render(source, out_path, build(root))
        print(f"{args.video} {name}: {frames} frames -> {out_path}  ({produced_by})")


if __name__ == "__main__":
    main()
