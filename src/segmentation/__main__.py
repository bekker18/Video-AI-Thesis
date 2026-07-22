"""CLI entry point."""
# python -m src.segmentation <frames_dir> [options]

import argparse
from pathlib import Path

from .download import download_models
from .pipeline import (
    DEFAULT_MOBILECLIP_WEIGHTS,
    DEFAULT_TRANSNET_WEIGHTS,
    segment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Segment sampled frames into shots with TransNetV2 and select a "
            "representative keyframe per shot with MobileCLIP medoids."
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
        "--transnet-weights",
        type=Path,
        default=DEFAULT_TRANSNET_WEIGHTS,
        help="TransNetV2 PyTorch weights.",
    )
    parser.add_argument(
        "--mobileclip-weights",
        type=Path,
        default=DEFAULT_MOBILECLIP_WEIGHTS,
        help="MobileCLIP checkpoint.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Transition probability threshold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device: cpu, cuda.",
    )
    args = parser.parse_args()

    download_models()

    shots = segment(
        args.frames_dir,
        output_dir=args.output,
        transnet_weights=args.transnet_weights,
        mobileclip_weights=args.mobileclip_weights,
        threshold=args.threshold,
        device=args.device,
    )

    out = (
        args.output
        if args.output is not None
        else args.frames_dir.parent / "segmentation"
    )
    print(f"Detected {len(shots)} shot(s); wrote {out}/segments.json and keyframes/")


if __name__ == "__main__":
    main()
