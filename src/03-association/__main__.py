"""CLI entry point."""

import argparse
from pathlib import Path

from .identities import (
    DEFAULT_CONTAINMENT,
    DEFAULT_FACE_SIMILARITY,
    DEFAULT_MIN_DETECTIONS,
    DEFAULT_MIN_TRACK_SCORE,
    DEFAULT_MIN_VOTES,
    DEFAULT_REID_SIMILARITY,
)
from .detect import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_NMS_IOU,
    DEFAULT_OBJECT_CONF,
    STREAMS,
)
from .download import download_models
from .pipeline import (
    DEFAULT_BUFFER_FRAMES,
    DEFAULT_DETECT_EVERY,
    DEFAULT_FACE_WEIGHTS,
    track,
)
from .trackers import DEFAULT_REID_WEIGHTS, PROFILES
from .visualize import visualize


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Track faces, bodies and objects across sampled frames and "
            "associate them into stable identity ids for gating and fusion."
        )
    )
    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Directory of sampled frames.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help=(
            "segments.json from the segmentation step. Defaults to the sibling "
            "segmentation/ directory; without it the clip is one shot."
        ),
    )
    parser.add_argument(
        "-p",
        "--profile",
        choices=PROFILES,
        default="affordable",
        help=(
            "affordable: YOLO + ByteTrack. offline: RT-DETR + BoT-SORT-ReID, "
            "which also links identities across shots."
        ),
    )
    parser.add_argument(
        "--face-weights",
        type=Path,
        default=DEFAULT_FACE_WEIGHTS,
        help="YOLO-face checkpoint.",
    )
    parser.add_argument(
        "--object-weights",
        type=Path,
        default=None,
        help="Body/object detector checkpoint. Defaults to the profile's model.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=DEFAULT_DETECT_EVERY,
        help="Run detection every n-th frame; tracking runs on every frame.",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=DEFAULT_BUFFER_FRAMES,
        help="How long an unmatched track survives before it is dropped.",
    )
    parser.add_argument(
        "--containment",
        type=float,
        default=DEFAULT_CONTAINMENT,
        help="Face-in-body overlap needed to vote for a face-body pairing.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=DEFAULT_MIN_VOTES,
        help="Frames of agreement needed before a face is bound to a body.",
    )
    parser.add_argument(
        "--reid-similarity",
        type=float,
        default=DEFAULT_REID_SIMILARITY,
        help="Body cosine similarity to merge person identities across shots.",
    )
    parser.add_argument(
        "--face-similarity",
        type=float,
        default=DEFAULT_FACE_SIMILARITY,
        help="Face cosine similarity to merge person identities across shots.",
    )
    parser.add_argument(
        "--reid-weights",
        type=Path,
        default=DEFAULT_REID_WEIGHTS,
        help="Appearance encoder used to link identities across shots.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=DEFAULT_NMS_IOU,
        help="IoU above which two boxes are the same thing detected twice.",
    )
    parser.add_argument(
        "--object-conf",
        type=float,
        default=DEFAULT_OBJECT_CONF,
        help="Confidence floor for the object stream, stricter than --conf.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=DEFAULT_MIN_DETECTIONS,
        help="Detected frames a track needs before it becomes an identity.",
    )
    parser.add_argument(
        "--min-track-score",
        type=float,
        default=DEFAULT_MIN_TRACK_SCORE,
        help="Mean score a track needs before it becomes an identity.",
    )
    parser.add_argument(
        "--no-link-shots",
        action="store_true",
        help="Keep identities shot-local instead of merging them by appearance.",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="Track faces and bodies only, skipping the object stream.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Detector input size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Detector confidence floor; kept low so ByteTrack sees weak boxes.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for detection and ReID: cpu, cuda.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Also render the tracked boxes over the frames as a video in "
            "data/processed/<video>/visualization/."
        ),
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Render from an existing tracks.jsonl instead of re-running the stage.",
    )
    parser.add_argument(
        "--visualize-streams",
        default=",".join(STREAMS),
        help="Comma-separated streams to draw: face, body, object.",
    )
    parser.add_argument(
        "--visualize-fps",
        type=float,
        default=None,
        help="Playback rate; defaults to the source clip's rate over --step.",
    )
    parser.add_argument(
        "--visualize-scores",
        action="store_true",
        help="Label each box with its confidence.",
    )
    parser.add_argument(
        "--hide-predicted",
        action="store_true",
        help="Draw detected boxes only, hiding the motion-model frames.",
    )
    parser.add_argument(
        "--save-annotated-frames",
        action="store_true",
        help="Also write the annotated frames as images next to the video.",
    )
    args = parser.parse_args()

    out = (
        args.output
        if args.output is not None
        else args.frames_dir.parent / "association"
    )
    streams = tuple(s.strip() for s in args.visualize_streams.split(",") if s.strip())
    unknown = set(streams) - set(STREAMS)
    if unknown:
        parser.error(f"Unknown stream(s): {', '.join(sorted(unknown))}")

    if args.visualize_only:
        _visualize(args, streams, tracks_file=out / "tracks.jsonl")
        return

    download_models()

    manifest = track(
        args.frames_dir,
        output_dir=args.output,
        segments_file=args.segments,
        profile=args.profile,
        face_weights=args.face_weights,
        object_weights=args.object_weights,
        detect_every=args.detect_every,
        buffer_frames=args.buffer_frames,
        containment=args.containment,
        min_votes=args.min_votes,
        reid_similarity=args.reid_similarity,
        face_similarity=args.face_similarity,
        reid_weights=args.reid_weights,
        min_detections=args.min_detections,
        min_track_score=args.min_track_score,
        link_shots=not args.no_link_shots,
        detect_objects=not args.no_objects,
        imgsz=args.imgsz,
        conf=args.conf,
        nms_iou=args.nms_iou,
        object_conf=args.object_conf,
        device=args.device,
    )

    counts = manifest["num_identities"]
    print(
        f"Tracked {manifest['num_frames']} frame(s) in {manifest['num_shots']} shot(s) "
        f"with {manifest['tracker']}; found {counts['person']} person "
        f"and {counts['object']} object identities"
    )
    print(f"Wrote {out}/tracks.json and {out}/tracks.jsonl")

    if args.visualize:
        _visualize(args, streams, tracks_file=out / "tracks.jsonl")


def _visualize(args, streams: tuple[str, ...], tracks_file: Path) -> None:
    render = visualize(
        args.frames_dir,
        tracks_file=tracks_file,
        fps=args.visualize_fps,
        streams=streams,
        show_scores=args.visualize_scores,
        show_predicted=not args.hide_predicted,
        save_frames=args.save_annotated_frames,
    )
    print(
        f"Rendered {render['num_frames']} frame(s) at {render['fps']} fps with "
        f"{render['num_boxes']} box(es)"
    )
    print(f"Wrote {render['video']}")


if __name__ == "__main__":
    main()
