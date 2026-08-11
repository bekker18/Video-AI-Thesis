"""Pipeline orchestrator. Each stage lives in src/<stage>/ and exposes run(cfg)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import download_models
from config import Config

ROOT = Path(__file__).resolve().parent

# Stages run in this order. Extended as modules are added.
STAGES = [
    "01-sampling",
    "02-segmentation",
    "03-profiler",
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
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video AI pipeline")
    p.add_argument("video", help="path to input .mp4")
    p.add_argument("--stages", nargs="+", default=STAGES,
                   help="subset of stages to run, e.g. --stages 02-segmentation")
    p.add_argument("--fps", type=float, default=None, help="sampling rate; default keeps every frame")
    p.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--quality", type=int, default=95, help="jpeg quality, 0 (worst) to 100 (best)")
    p.add_argument("--width", type=int, default=None, help="resize frames to this width")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--shot-mode", choices=["default", "clean_shot"], default="default",
                   help="clean_shot keeps only cuts and discards transitions")
    p.add_argument("--shot-overlap", type=int, default=30,
                   help="frames shared between adjacent inference windows")
    p.add_argument("--keyframes", type=int, default=3, help="representative frames kept per shot")

    p.add_argument("--detector", default="yolo11n",
                   choices=["yolo11n", "yolo11s", "yolo11m", "rtdetr-l"])
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--face-conf", type=float, default=0.6)
    p.add_argument("--text-conf", type=float, default=0.5)
    p.add_argument("--scenes", type=int, default=3, help="scene labels kept per shot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")

    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {', '.join(unknown)}\navailable: {', '.join(STAGES)}")

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
        )
        download_models.ensure(name)
        print(f"[{name}] start")
        result = load_stage(name).run(cfg)
        print(f"[{name}] done -> {result}")


if __name__ == "__main__":
    main()
