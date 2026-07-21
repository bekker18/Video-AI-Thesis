"""Shot-boundary detection over sampled frames using TransNetV2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# TransNetV2 consumes 48x27 RGB frames in windows of 100 with a stride of 50, padding 25 frames at each end and keeping the central 50 predictions.
FRAME_WIDTH = 48
FRAME_HEIGHT = 27
WINDOW = 100
STRIDE = 50
PAD = 25


def load_frames(
    frames_dir: Path, pattern: str = "frame_*.png"
) -> tuple[np.ndarray, list[Path]]:
    """Load sampled frames as a (N, 27, 48, 3) uint8 RGB array plus their paths."""

    import cv2

    paths = sorted(Path(frames_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {frames_dir}")

    frames = np.empty((len(paths), FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return frames, paths


def predictions_to_scenes(
    predictions: np.ndarray, threshold: float = 0.5
) -> list[list[int]]:
    """Convert per-frame transition probabilities into [start, end] shot ranges."""

    preds = (np.asarray(predictions) > threshold).astype(np.uint8)

    scenes: list[list[int]] = []
    t = t_prev = 0
    start = 0
    for i, t in enumerate(preds):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if len(preds) > 0 and t == 0:
        scenes.append([start, len(preds) - 1])

    if not scenes:
        return [[0, len(preds) - 1]]
    return [[int(s), int(e)] for s, e in scenes]


class ShotDetector:
    """Wraps the TransNetV2 PyTorch model for inference on sampled frames."""

    def __init__(self, weights: Path, device: str = "cpu") -> None:
        import torch
        from transnetv2_pytorch import TransNetV2

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"TransNetV2 weights not found: {weights}\n")

        self._torch = torch
        self.device = device
        self.model = TransNetV2()
        state_dict = torch.load(str(weights), map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """Per-frame transition probabilities for (N, 27, 48, 3) uint8 frames."""
        torch = self._torch
        n = len(frames)

        remainder = n % STRIDE
        pad_end = PAD + (STRIDE - remainder if remainder != 0 else 0)
        padded = np.concatenate(
            [frames[:1]] * PAD + [frames] + [frames[-1:]] * pad_end, axis=0
        )

        chunks: list[np.ndarray] = []
        ptr = 0
        with torch.no_grad():
            while ptr + WINDOW <= len(padded):
                window = padded[ptr : ptr + WINDOW][np.newaxis]  # (1,100,27,48,3)
                single, _ = self.model(torch.from_numpy(window).to(self.device))
                single = torch.sigmoid(single).cpu().numpy()
                chunks.append(single[0, PAD : PAD + STRIDE, 0])
                ptr += STRIDE

        return np.concatenate(chunks)[:n]

    def detect(self, frames: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
        """Return [start, end] (inclusive) shot ranges over the frames."""
        return predictions_to_scenes(self.predict(frames), threshold)
