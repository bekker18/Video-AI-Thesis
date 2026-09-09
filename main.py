"""Pipeline orchestrator. Each stage lives in src/<stage>/ and exposes run(cfg)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import download_models
from config import Config

ROOT = Path(__file__).resolve().parent

# Stages run in this order. Extended as modules are added.
STAGES = [
    "01-sampling",
    "02-segmentation",
    "03-profiler",
    "04-router",
    "05-global-experts",
    "06-detection-tracking",
    "07-conditional-experts",
    "08-consolidation",
    "09-aggregation",
    "10-fusion",
]


class Stage(Protocol):
    def run(self, cfg: Config) -> dict[str, Any]: ...


def stage_path(name: str) -> Path:
    """01-sampling -> src/01-sampling/sampling.py, 06-global-experts -> .../global_experts.py"""
    stem = name.split("-", 1)[1].replace("-", "_")
    return ROOT / "src" / name / f"{stem}.py"


def load_stage(name: str) -> Stage:
    path = stage_path(name)
    if not path.exists():
        sys.exit(f"stage not implemented: {name}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load stage: {path}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses in the stage can resolve their annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Stage, module)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video AI pipeline")
    p.add_argument("video", help="path to input .mp4")
    p.add_argument(
        "--stages",
        nargs="+",
        default=STAGES,
        help="subset of stages to run, e.g. --stages 02-segmentation",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="sampling rate; default keeps every frame",
    )
    p.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    p.add_argument(
        "--quality", type=int, default=95, help="jpeg quality, 0 (worst) to 100 (best)"
    )
    p.add_argument(
        "--width", type=int, default=None, help="resize frames to this width"
    )

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--shot-mode",
        choices=["default", "clean_shot"],
        default="default",
        help="clean_shot keeps only cuts and discards transitions",
    )
    p.add_argument(
        "--shot-overlap",
        type=int,
        default=30,
        help="frames shared between adjacent inference windows",
    )
    p.add_argument(
        "--keyframes", type=int, default=3, help="representative frames kept per shot"
    )

    p.add_argument(
        "--detector",
        default="yolo11s",
        choices=["yolo11n", "yolo11s", "yolo11m", "rtdetr-l"],
    )
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--face-conf", type=float, default=0.6)
    p.add_argument("--text-conf", type=float, default=0.5)
    p.add_argument("--scenes", type=int, default=3, help="scene labels kept per shot")

    p.add_argument(
        "--router-agreement",
        type=float,
        default=0.0,
        help="soft gate: fraction of a shot's keyframes a flag must hold in",
    )

    p.add_argument(
        "--map-size",
        type=int,
        default=512,
        help="longest side of every dense map written by 05",
    )
    p.add_argument("--expert-batch", type=int, default=8)
    p.add_argument(
        "--seg-model",
        default="segformer-b1",
        choices=["segformer-b0", "segformer-b1", "segformer-b2"],
    )
    p.add_argument(
        "--seg-top", type=int, default=10, help="classes or segments kept per keyframe"
    )
    p.add_argument(
        "--tags", type=int, default=15, help="zero-shot tags kept per keyframe"
    )

    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    p.add_argument(
        "--frame-source",
        choices=["video", "frames"],
        default="video",
        help="decode the video, or read 01-sampling's frames",
    )
    p.add_argument(
        "--det-stride",
        type=int,
        default=2,
        help="detect every Nth frame; the rest are interpolated",
    )
    p.add_argument("--track-low-conf", type=float, default=0.1)
    p.add_argument("--track-high-conf", type=float, default=0.25)
    p.add_argument("--track-new-conf", type=float, default=0.25)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--track-match", type=float, default=0.8)
    p.add_argument("--track-batch", type=int, default=16)
    p.add_argument("--crops", type=int, default=5, help="best crops kept per track")
    p.add_argument("--crop-pad", type=float, default=0.1)
    p.add_argument("--min-track", type=int, default=3)

    p.add_argument(
        "--series-stride",
        type=int,
        default=2,
        help="observed rows between series samples in 07",
    )
    p.add_argument(
        "--blendshapes", type=int, default=10, help="blendshapes kept per sample"
    )
    p.add_argument(
        "--attributes", type=int, default=8, help="zero-shot attributes kept per crop"
    )

    p.add_argument(
        "--consol-face-within",
        type=float,
        default=0.55,
        help="face cosine a link needs inside a segment",
    )
    p.add_argument(
        "--consol-face-across",
        type=float,
        default=0.45,
        help="face cosine a link needs across segments",
    )
    p.add_argument(
        "--consol-body-within",
        type=float,
        default=0.70,
        help="body cosine a link needs inside a segment",
    )
    p.add_argument(
        "--consol-body-across",
        type=float,
        default=0.60,
        help="body cosine a link needs across segments",
    )
    p.add_argument(
        "--consol-min-area",
        type=int,
        default=1600,
        help="crop pixel area below which a descriptor is not trusted",
    )
    p.add_argument(
        "--consol-min-front",
        type=float,
        default=0.15,
        help="frontality a face crop must reach to contribute a descriptor",
    )

    p.add_argument(
        "--agg-top", type=int, default=10, help="labels kept per distribution in 09"
    )
    p.add_argument(
        "--pose-vis",
        type=float,
        default=0.5,
        help="visibility a joint must reach to count in 09",
    )

    p.add_argument(
        "--attention-deg",
        type=float,
        default=30.0,
        help="cone width within which a head counts as attending someone",
    )
    p.add_argument(
        "--sync-min",
        type=int,
        default=5,
        help="shared affect samples a synchrony edge needs before it is computed",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")

    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        sys.exit(
            f"unknown stage(s): {', '.join(unknown)}\navailable: {', '.join(STAGES)}"
        )

    out_root = ROOT / "data" / "processed" / video.stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Stages read their input from out_root, so any stage can run on its own
    # as long as the previous one has already written its output.
    for name in sorted(args.stages, key=STAGES.index):
        cfg = Config(
            video=video,
            out_root=out_root,
            model_dir=ROOT / "models" / name,
            device=args.device,
            fps=args.fps,
            image_format=args.image_format,
            quality=args.quality,
            width=args.width,
            shot_mode=args.shot_mode,
            shot_overlap=args.shot_overlap,
            keyframes=args.keyframes,
            detector=args.detector,
            det_conf=args.det_conf,
            face_conf=args.face_conf,
            text_conf=args.text_conf,
            scenes=args.scenes,
            router_agreement=args.router_agreement,
            map_size=args.map_size,
            expert_batch=args.expert_batch,
            seg_model=args.seg_model,
            seg_top=args.seg_top,
            tags=args.tags,
            tracker=args.tracker,
            frame_source=args.frame_source,
            det_stride=args.det_stride,
            track_low_conf=args.track_low_conf,
            track_high_conf=args.track_high_conf,
            track_new_conf=args.track_new_conf,
            track_buffer=args.track_buffer,
            track_match=args.track_match,
            track_batch=args.track_batch,
            crops=args.crops,
            crop_pad=args.crop_pad,
            min_track=args.min_track,
            series_stride=args.series_stride,
            blendshapes=args.blendshapes,
            attributes=args.attributes,
            consol_face_within=args.consol_face_within,
            consol_face_across=args.consol_face_across,
            consol_body_within=args.consol_body_within,
            consol_body_across=args.consol_body_across,
            consol_min_area=args.consol_min_area,
            consol_min_front=args.consol_min_front,
            agg_top=args.agg_top,
            pose_vis=args.pose_vis,
            attention_deg=args.attention_deg,
            sync_min=args.sync_min,
        )
        download_models.ensure(name)
        print(f"[{name}] start")
        result = load_stage(name).run(cfg)
        print(f"[{name}] done -> {result}")


if __name__ == "__main__":
    main()
