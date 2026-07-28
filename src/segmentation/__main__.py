"""CLI entry point."""
# python -m src.segmentation <frames_dir> [options]

import argparse
from pathlib import Path

from .download import download_models
from .pipeline import (
    DEFAULT_MOBILECLIP_WEIGHTS,
    DEFAULT_OMNISHOTCUT_WEIGHTS,
    segment,
)
from .shots import DEFAULT_OVERLAP


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Segment sampled frames into shots with OmniShotCut and select a "
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
        "--omnishotcut-weights",
        type=Path,
        default=DEFAULT_OMNISHOTCUT_WEIGHTS,
        help="OmniShotCut checkpoint.",
    )
    parser.add_argument(
        "--mobileclip-weights",
        type=Path,
        default=DEFAULT_MOBILECLIP_WEIGHTS,
        help="MobileCLIP checkpoint.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("clean_shot", "default"),
        default="clean_shot",
        help="clean_shot keeps general cuts only; default also labels transitions.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Overlap frames between adjacent inference windows.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for MobileCLIP: cpu, cuda. OmniShotCut always uses cuda.",
    )
    args = parser.parse_args()

    download_models()

    shots = segment(
        args.frames_dir,
        output_dir=args.output,
        omnishotcut_weights=args.omnishotcut_weights,
        mobileclip_weights=args.mobileclip_weights,
        mode=args.mode,
        overlap=args.overlap,
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
