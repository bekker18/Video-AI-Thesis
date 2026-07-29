import argparse
from pathlib import Path

from src.sampling.sampler import decode
from src.segmentation.download import download_models

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def ensure_data_dirs() -> None:
    # Create the data directory tree if it does not already exist.
    for directory in (RAW_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_data_dirs()
    download_models()

    parser = argparse.ArgumentParser(
        description="Decode a video into frames using OpenCV."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Input video.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Directory to write frames.",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=1,
        help="Save every step-th frame.",
    )
    args = parser.parse_args()

    video = args.video
    if video.parent == Path("."):
        video = RAW_DIR / video

    # Default output is data/processed/<video name>/frame.
    output = (
        args.output
        if args.output is not None
        else PROCESSED_DIR / video.stem / "frames"
    )

    saved = decode(video, output, step=args.step)
    print(f"Saved {saved} frame(s) to {output}")


if __name__ == "__main__":
    main()
