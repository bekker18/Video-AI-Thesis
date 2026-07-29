"""Shot-boundary detection over sampled frames using OmniShotCut."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# OmniShotCut resizes any input to the process resolution stored in its
# checkpoint; this is the fallback when the checkpoint does not advertise one.
DEFAULT_PROCESS_SIZE = (128, 96)  # (width, height)
# Overlap frames between adjacent inference windows, as used by the reference CLI.
DEFAULT_OVERLAP = 20


def load_frames(
    frames_dir: Path,
    size: tuple[int, int] | None = None,
    pattern: str = "frame_*.png",
) -> tuple[np.ndarray, list[Path]]:
    """Load sampled frames as a (T, H, W, 3) uint8 RGB array plus their paths."""

    import cv2

    paths = sorted(Path(frames_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {frames_dir}")

    frames: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if size is None and len({f.shape for f in frames}) > 1:
        raise ValueError(
            "Sampled frames have mixed resolutions; pass size=(width, height)."
        )

    return np.stack(frames), paths


class ShotDetector:
    """Wraps the OmniShotCut shot-query Transformer for inference on frames.

    OmniShotCut loads its weights onto CUDA, so a GPU is required.
    """

    def __init__(
        self,
        weights: Path,
        mode: str = "clean_shot",
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        import omnishotcut

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"OmniShotCut weights not found: {weights}\n")

        self.mode = mode
        self.overlap = overlap
        self.model = omnishotcut.load(str(weights))

    @property
    def process_size(self) -> tuple[int, int]:
        """(width, height) the loaded checkpoint resizes frames to."""
        args = getattr(self.model, "_model_args", None)
        width = getattr(args, "process_width", None)
        height = getattr(args, "process_height", None)
        if width is None or height is None:
            return DEFAULT_PROCESS_SIZE
        return int(width), int(height)

    def detect(
        self, frames: np.ndarray
    ) -> tuple[list[list[int]], list[str], list[str]]:
        """Return [start, end] (inclusive) shot ranges plus transition labels."""
        result = self.model.inference(frames, mode=self.mode, overlap=self.overlap)
        if self.mode == "clean_shot":
            ranges, intra_labels, inter_labels = result, [], []
        else:
            ranges, intra_labels, inter_labels = result

        if not ranges:
            # No boundary found: treat the whole input as a single shot.
            return [[0, len(frames) - 1]], [], []

        return [[int(s), int(e) - 1] for s, e in ranges], intra_labels, inter_labels
