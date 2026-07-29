"""Representative keyframe selection per shot using MobileCLIP embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def medoid_index(embeddings: np.ndarray) -> int:
    """Index of the medoid embedding (max mean cosine similarity to the others)."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or len(emb) == 0:
        raise ValueError("embeddings must be a non-empty 2D array")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms

    sims = unit @ unit.T  # cosine similarity matrix
    n = len(emb)
    mean_sim = (sims.sum(axis=1) - 1.0) / max(n - 1, 1)  # exclude self-similarity
    return int(np.argmax(mean_sim))


class KeyframeSelector:
    """Selects the MobileCLIP medoid keyframe for each shot."""

    def __init__(
        self,
        weights: "Path",
        model_name: str = "MobileCLIP-S2",
        device: str = "cpu",
    ) -> None:
        import open_clip
        import torch

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"MobileCLIP weights not found: {weights}\n")

        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(weights)
        )
        self.model.eval().to(device)

    def embed(self, image_paths: list[Path]) -> np.ndarray:
        """MobileCLIP image embeddings for the given frame files."""
        import torch
        from PIL import Image

        tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
        return feats.cpu().numpy()

    def select(self, image_paths: list[Path], scenes: list[list[int]]) -> list[int]:
        """Return the global frame index of the keyframe for each [start, end] shot."""
        keyframes: list[int] = []
        for start, end in scenes:
            indices = list(range(start, end + 1))
            if len(indices) == 1:
                keyframes.append(indices[0])
                continue
            embeddings = self.embed([image_paths[i] for i in indices])
            keyframes.append(indices[medoid_index(embeddings)])
        return keyframes
