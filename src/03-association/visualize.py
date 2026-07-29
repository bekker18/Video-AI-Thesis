"""Render tracked boxes over the sampled frames as a video.

Reads tracks.jsonl rather than re-running detection, so a visualization costs
one decode-draw-encode pass and can be produced from any finished association run.

Boxes are coloured by identity, not by track: a face and the body it was bound
to share a colour, which is what makes the association visible. Frames carried
by the motion model alone are drawn dashed, so the cost of `--detect-every` is
visible too.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2

# Sampled clips are decoded from here, and the source is probed for its frame
# rate so the rendered video plays at the speed of the original.
RAW_DIR = Path("data/raw/clips")
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm")
DEFAULT_FPS = 25.0

FRAME_PATTERN = "frame_*.png"
OUTPUT_NAME = "tracks.mp4"

# Distinct in BGR, and distinguishable against most footage.
_PALETTE = [
    (56, 56, 255),
    (60, 180, 60),
    (255, 160, 30),
    (30, 200, 255),
    (200, 80, 220),
    (0, 140, 255),
    (220, 200, 60),
    (140, 90, 255),
    (80, 220, 180),
    (255, 110, 130),
    (100, 170, 30),
    (190, 190, 190),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45


def _frame_step(paths: list[Path]) -> int:
    """Sampling step, recovered from the frame numbering.

    Frames are named after their index in the source video, so the gap between
    consecutive files is the `--step` the clip was decoded with.
    """
    if len(paths) < 2:
        return 1
    try:
        first, second = (int(p.stem.split("_")[-1]) for p in paths[:2])
    except ValueError:
        return 1
    return max(1, second - first)


def _source_fps(frames_dir: Path) -> float | None:
    """Frame rate of the clip these frames were decoded from, if it is around."""
    name = frames_dir.parent.name
    for suffix in VIDEO_SUFFIXES:
        candidate = RAW_DIR / f"{name}{suffix}"
        if not candidate.exists():
            continue
        capture = cv2.VideoCapture(str(candidate))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        if fps and fps > 0:
            return float(fps)
    return None


def _playback_fps(frames_dir: Path, paths: list[Path]) -> float:
    """Rate at which the sampled frames play back in real time."""
    fps = _source_fps(frames_dir)
    if fps is None:
        return DEFAULT_FPS
    return max(1.0, fps / _frame_step(paths))


def _color(identity_id: str) -> tuple[int, int, int]:
    """Stable colour per identity, keyed on the id's counter.

    Ids are numbered by first appearance, so indexing the palette by that
    number gives neighbouring identities different colours.
    """
    kind, _, number = identity_id.rpartition("_")
    try:
        index = int(number)
    except ValueError:
        index = len(identity_id)
    # Offset the object series so person_0001 and object_0001 do not collide.
    if kind.endswith("object"):
        index += len(_PALETTE) // 2
    return _PALETTE[index % len(_PALETTE)]


def _text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = background
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _dashed_rectangle(image, p1, p2, color, thickness: int, dash: int = 8) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        end = min(x + dash, x2)
        cv2.line(image, (x, y1), (end, y1), color, thickness)
        cv2.line(image, (x, y2), (end, y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        end = min(y + dash, y2)
        cv2.line(image, (x1, y), (x1, end), color, thickness)
        cv2.line(image, (x2, y), (x2, end), color, thickness)


def _draw_label(image, text: str, x: int, y: int, color) -> None:
    (width, height), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, 1)
    # Boxes touching the top edge would have their label drawn off-screen.
    y = max(y, height + baseline + 2)
    x = max(0, min(x, image.shape[1] - width - 4))
    cv2.rectangle(image, (x, y - height - baseline - 2), (x + width + 4, y), color, -1)
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline),
        _FONT,
        _FONT_SCALE,
        _text_color(color),
        1,
        cv2.LINE_AA,
    )


def _draw_hud(image, text: str) -> None:
    (width, height), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, 1)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width + 16, height + baseline + 12), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(
        image,
        text,
        (8, height + 6),
        _FONT,
        _FONT_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _load_records(tracks_file: Path) -> dict[int, list[dict]]:
    """Per-frame boxes, keyed by the frame's position in the sampled sequence."""
    by_frame: defaultdict[int, list[dict]] = defaultdict(list)
    with Path(tracks_file).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_frame[int(record["frame_index"])].append(record)
    # Bodies first so the smaller face box lands on top of its owner's box.
    order = {"object": 0, "body": 1, "face": 2}
    for records in by_frame.values():
        records.sort(key=lambda r: order.get(r["stream"], 0))
    return dict(by_frame)


def _caption(record: dict, show_scores: bool, bound: bool) -> str:
    identity_id = record.get("identity_id") or record["track_key"]
    parts = [identity_id]
    # The class is the informative part for objects; for people it is implied.
    if record["stream"] == "object":
        parts.append(record["class_name"])
    elif record["stream"] == "face":
        # A face drawn alongside its own body box already carries the id on the
        # body, and repeating it crowds the frame; the shared colour is the link.
        parts = ["face"] if bound else [identity_id, "face"]
    if show_scores:
        parts.append(f"{record['score']:.2f}")
    return " ".join(parts)


def visualize(
    frames_dir: Path,
    output_dir: Path | None = None,
    tracks_file: Path | None = None,
    fps: float | None = None,
    streams: tuple[str, ...] | None = None,
    show_scores: bool = False,
    show_predicted: bool = True,
    save_frames: bool = False,
) -> dict:
    """Draw the association output over the frames and encode it as a video."""

    frames_dir = Path(frames_dir)
    # Frames live in data/processed/<video>/frames; write to the sibling
    # visualization/ directory so later stages can add their own renders there.
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else frames_dir.parent / "visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracks_file is None:
        tracks_file = frames_dir.parent / "association" / "tracks.jsonl"
    if not Path(tracks_file).exists():
        raise FileNotFoundError(
            f"Association output not found: {tracks_file}\n"
            "Run the association step first."
        )

    paths = sorted(frames_dir.glob(FRAME_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No frames matching {FRAME_PATTERN!r} in {frames_dir}")

    by_frame = _load_records(tracks_file)
    if streams is not None:
        by_frame = {
            index: [r for r in records if r["stream"] in streams]
            for index, records in by_frame.items()
        }

    # Shot ids come from the records themselves, so the render does not depend
    # on segments.json still being present.
    shot_of = {
        index: records[0]["shot_id"] for index, records in by_frame.items() if records
    }

    if fps is None:
        fps = _playback_fps(frames_dir, paths)

    frame_dir = output_dir / "frames"
    if save_frames:
        frame_dir.mkdir(exist_ok=True)

    output_path = output_dir / OUTPUT_NAME
    writer = None
    num_predicted = 0
    try:
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Failed to read frame: {path}")

            records = by_frame.get(index, [])
            drawn_bodies = {
                r.get("identity_id") for r in records if r["stream"] == "body"
            }
            people = set()
            for record in records:
                predicted = record["source"] == "predicted"
                if predicted:
                    num_predicted += 1
                    if not show_predicted:
                        continue

                identity_id = record.get("identity_id") or record["track_key"]
                if identity_id.startswith("person"):
                    people.add(identity_id)

                color = _color(identity_id)
                x1, y1, x2, y2 = (round(v) for v in record["bbox"])
                # Faces sit inside their body box, so keep their outline thin.
                thickness = 1 if record["stream"] == "face" else 2
                if predicted:
                    _dashed_rectangle(image, (x1, y1), (x2, y2), color, thickness)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                caption = _caption(
                    record, show_scores, bound=identity_id in drawn_bodies
                )
                _draw_label(image, caption, x1, y1 - 2, color)

            _draw_hud(
                image,
                f"frame {index + 1}/{len(paths)}  "
                f"shot {shot_of.get(index, 0):03d}  "
                f"people {len(people)}",
            )

            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(image)

            if save_frames:
                cv2.imwrite(str(frame_dir / path.name), image)
    finally:
        if writer is not None:
            writer.release()

    num_boxes = sum(len(records) for records in by_frame.values())
    return {
        "video": str(output_path),
        "frames_dir": str(frames_dir),
        "tracks_file": str(tracks_file),
        "num_frames": len(paths),
        "fps": round(float(fps), 3),
        "num_boxes": num_boxes,
        "num_predicted_boxes": num_predicted,
        "streams": list(streams) if streams is not None else None,
    }
