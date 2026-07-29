"""End-to-end track association over sampled frames.

Consumes the sampled frames plus the shot boundaries from segments.json,
and produces stable identity ids for faces, bodies and objects

Shots drive the loop because a cut invalidates motion continuity: trackers are
reset at every boundary and identity is re-established across boundaries by
appearance instead (offline profile only).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .identities import (
    DEFAULT_CONTAINMENT,
    DEFAULT_MIN_VOTES,
    DEFAULT_REID_SIMILARITY,
    Track,
    assign_identities,
    bind_faces_to_bodies,
    collect_votes,
    count_people,
    link_across_shots,
    merge_identities,
)
from .detect import DEFAULT_CONF, DEFAULT_IMGSZ, STREAMS, MultiStreamDetector
from .trackers import StreamTracker, TrackerConfig

# All model weights live locally under models/association/ and are loaded offline.
DEFAULT_FACE_WEIGHTS = Path("models/association/yolov11n-face.pt")
DEFAULT_YOLO_WEIGHTS = Path("models/association/yolo11n.pt")
DEFAULT_RTDETR_WEIGHTS = Path("models/association/rtdetr-l.pt")

# Detection every 1-2 frames; tracking every frame.
DEFAULT_DETECT_EVERY = 2
DEFAULT_BUFFER_FRAMES = 30

FRAME_PATTERN = "frame_*.png"


def _load_shots(frames_dir: Path, segments_file: Path | None, num_frames: int):
    """Shot ranges as [start, end] positional indices into the frame list."""
    if segments_file is None:
        segments_file = frames_dir.parent / "segmentation" / "segments.json"
        if not segments_file.exists():
            # Without segmentation output the whole clip is one shot.
            return [(0, num_frames - 1)], None
    elif not Path(segments_file).exists():
        raise FileNotFoundError(f"Segments file not found: {segments_file}")

    manifest = json.loads(Path(segments_file).read_text())
    shots = [
        (int(s["start_frame"]), min(int(s["end_frame"]), num_frames - 1))
        for s in manifest["shots"]
    ]
    return shots or [(0, num_frames - 1)], Path(segments_file)


def _track_key(shot_id: int, stream: str, track_id: int) -> str:
    return f"s{shot_id:04d}_{stream}_{track_id:04d}"


def track(
    frames_dir: Path,
    output_dir: Path | None = None,
    segments_file: Path | None = None,
    profile: str = "affordable",
    face_weights: Path = DEFAULT_FACE_WEIGHTS,
    object_weights: Path | None = None,
    detect_every: int = DEFAULT_DETECT_EVERY,
    buffer_frames: int = DEFAULT_BUFFER_FRAMES,
    containment: float = DEFAULT_CONTAINMENT,
    min_votes: int = DEFAULT_MIN_VOTES,
    reid_similarity: float = DEFAULT_REID_SIMILARITY,
    link_shots: bool = True,
    detect_objects: bool = True,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    device: str = "cpu",
) -> dict:
    """Detect, track and associate identities over the sampled frames."""

    import cv2

    if detect_every < 1:
        raise ValueError(f"detect_every must be >= 1, got {detect_every}")

    frames_dir = Path(frames_dir)
    # Frames live in data/processed/<video>/frames; write alongside them.
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else frames_dir.parent / "association"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(frames_dir.glob(FRAME_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No frames matching {FRAME_PATTERN!r} in {frames_dir}")

    shots, segments_path = _load_shots(frames_dir, segments_file, len(paths))

    # The offline path swaps the general detector for RT-DETR and the tracker
    # for BoT-SORT-ReID; faces come from the same face checkpoint either way.
    general_arch = "rtdetr" if profile == "offline" else "yolo"
    if object_weights is None:
        object_weights = (
            DEFAULT_RTDETR_WEIGHTS if profile == "offline" else DEFAULT_YOLO_WEIGHTS
        )

    detector = MultiStreamDetector(
        face_weights=face_weights,
        general_weights=object_weights,
        general_arch=general_arch,
        device=device,
        imgsz=imgsz,
        conf=conf,
        detect_objects=detect_objects,
    )
    config = TrackerConfig(
        profile=profile,
        detect_every=detect_every,
        buffer_frames=buffer_frames,
        device=device,
    )
    class_names = detector.general_model.names
    trackers = {
        "face": StreamTracker("face", config, {0: "face"}),
        "body": StreamTracker("body", config, class_names),
        "object": StreamTracker("object", config, class_names),
    }

    tracks: dict[str, Track] = {}
    votes: defaultdict[tuple[str, str], int] = defaultdict(int)
    records: list[dict] = []
    num_detection_frames = 0

    for shot_id, (start, end) in enumerate(shots):
        for tracker in trackers.values():
            tracker.reset()

        for index in range(start, end + 1):
            # Detection every detect_every frames, counted from the shot start
            # so each shot opens on a detection frame.
            is_detection_frame = (index - start) % detect_every == 0
            image = None
            if is_detection_frame:
                image = cv2.imread(str(paths[index]))
                if image is None:
                    raise RuntimeError(f"Failed to read frame: {paths[index]}")
                detections = detector.detect(image)
                num_detection_frames += 1

            frame_boxes: dict[str, list] = {}
            for stream in STREAMS:
                tracker = trackers[stream]
                if is_detection_frame:
                    boxes = tracker.update(detections.stream(stream), image)
                else:
                    boxes = tracker.predict()
                frame_boxes[stream] = boxes

                for box in boxes:
                    key = _track_key(shot_id, stream, box.track_id)
                    entry = tracks.get(key)
                    if entry is None:
                        entry = Track(
                            key=key,
                            stream=stream,
                            shot_id=shot_id,
                            track_id=box.track_id,
                            class_id=box.class_id,
                            class_name=box.class_name,
                            start_frame=index,
                            end_frame=index,
                        )
                        tracks[key] = entry
                    entry.end_frame = index
                    entry.num_frames += 1
                    entry.num_detected += int(box.detected)
                    entry.score_sum += box.score
                    records.append(
                        {
                            "frame_index": index,
                            "frame_file": paths[index].name,
                            "shot_id": shot_id,
                            "stream": stream,
                            "track_key": key,
                            "class_name": box.class_name,
                            "bbox": [round(v, 2) for v in box.xyxy],
                            "score": round(box.score, 4),
                            "source": "detected" if box.detected else "predicted",
                        }
                    )

            # Faces and bodies vote for a pairing on every frame they co-occur,
            # including predicted ones - a face is bound to a body over the
            # whole shot, not on a single frame's overlap.
            collect_votes(
                [
                    (_track_key(shot_id, "face", b.track_id), b.xyxy)
                    for b in frame_boxes["face"]
                ],
                [
                    (_track_key(shot_id, "body", b.track_id), b.xyxy)
                    for b in frame_boxes["body"]
                ],
                votes,
                threshold=containment,
            )

        # Appearance features are only available once a shot has been tracked.
        for stream, tracker in trackers.items():
            for track_id, embedding in tracker.embeddings.items():
                entry = tracks.get(_track_key(shot_id, stream, track_id))
                if entry is not None:
                    entry.embedding = embedding

    track_list = list(tracks.values())
    bindings = bind_faces_to_bodies(dict(votes), min_votes=min_votes)
    identities = assign_identities(track_list, bindings)

    if link_shots and profile == "offline":
        merges = link_across_shots(track_list, similarity=reid_similarity)
        identities = merge_identities(identities, track_list, merges)

    manifest = _build_manifest(
        frames_dir=frames_dir,
        segments_path=segments_path,
        paths=paths,
        shots=shots,
        tracks=track_list,
        identities=identities,
        bindings=bindings,
        records=records,
        profile=profile,
        detect_every=detect_every,
        num_detection_frames=num_detection_frames,
        detector=detector,
        general_arch=general_arch,
    )

    (output_dir / "tracks.json").write_text(json.dumps(manifest, indent=2))
    _write_records(output_dir / "tracks.jsonl", records, tracks)
    return manifest


def _write_records(path: Path, records: list[dict], tracks: dict[str, Track]) -> None:
    """Write per-frame boxes as JSON lines, stamped with the final identity."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            record["identity_id"] = tracks[record["track_key"]].identity_id
            handle.write(json.dumps(record) + "\n")


def _build_manifest(
    frames_dir: Path,
    segments_path: Path | None,
    paths: list[Path],
    shots: list[tuple[int, int]],
    tracks: list[Track],
    identities: dict,
    bindings: dict[str, str],
    records: list[dict],
    profile: str,
    detect_every: int,
    num_detection_frames: int,
    detector: MultiStreamDetector,
    general_arch: str,
) -> dict:
    by_key = {t.key: t for t in tracks}

    # Per-frame person counts, so the gate can use the peak in a shot rather
    # than the number of identities that merely appear somewhere in it.
    people_per_frame: defaultdict[int, set[str]] = defaultdict(set)
    for record in records:
        identity_id = by_key[record["track_key"]].identity_id
        if identity_id and identity_id.startswith("person"):
            people_per_frame[record["frame_index"]].add(identity_id)

    shot_summaries = []
    for shot_id, (start, end) in enumerate(shots):
        person_ids = count_people(identities, by_key, shot_id)
        object_ids = sorted(
            identity.identity_id
            for identity in identities.values()
            if identity.kind == "object"
            and any(by_key[k].shot_id == shot_id for k in identity.track_keys)
        )
        peak = max(
            (len(people_per_frame.get(i, ())) for i in range(start, end + 1)),
            default=0,
        )
        shot_summaries.append(
            {
                "shot_id": shot_id,
                "start_frame": start,
                "end_frame": end,
                "num_people": len(person_ids),
                "max_people_in_frame": peak,
                "person_ids": person_ids,
                "num_objects": len(object_ids),
                "object_ids": object_ids,
            }
        )

    return {
        "frames_dir": str(frames_dir),
        "segments_file": str(segments_path) if segments_path else None,
        "num_frames": len(paths),
        "num_shots": len(shots),
        "profile": profile,
        "tracker": "BoT-SORT-ReID" if profile == "offline" else "ByteTrack",
        "detector": {**detector.weights, "general_arch": general_arch},
        "detect_every": detect_every,
        "num_detection_frames": num_detection_frames,
        "boxes_file": "tracks.jsonl",
        "num_identities": {
            "person": sum(1 for i in identities.values() if i.kind == "person"),
            "object": sum(1 for i in identities.values() if i.kind == "object"),
        },
        "num_face_body_bindings": len(bindings),
        "shots": shot_summaries,
        "identities": [
            {
                "identity_id": identity.identity_id,
                "kind": identity.kind,
                "class_name": identity.class_name,
                "shots": identity.shots,
                "start_frame": identity.start_frame,
                "end_frame": identity.end_frame,
                "face_tracks": identity.face_tracks,
                "body_tracks": identity.body_tracks,
                "object_tracks": identity.object_tracks,
            }
            for identity in sorted(
                identities.values(), key=lambda i: (i.kind, i.identity_id)
            )
        ],
        "tracks": [
            {
                "track_key": t.key,
                "identity_id": t.identity_id,
                "stream": t.stream,
                "shot_id": t.shot_id,
                "track_id": t.track_id,
                "class_name": t.class_name,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "num_frames": t.num_frames,
                "num_detected": t.num_detected,
                "mean_score": round(t.mean_score, 4),
            }
            for t in sorted(tracks, key=lambda t: (t.shot_id, t.stream, t.track_id))
        ],
    }
