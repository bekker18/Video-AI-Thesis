"""Download the segmentation models into models/."""

from __future__ import annotations

import shutil
from pathlib import Path

MODELS_DIR = Path("models/segmentation")

_MODELS = [
    (
        "uva-cv-lab/OmniShotCut",
        "OmniShotCut_ckpt.pth",
        "OmniShotCut_ckpt.pth",
    ),
    (
        "apple/MobileCLIP-S2",
        "mobileclip_s2.pt",
        "mobileclip_s2.pt",
    ),
]


def download_models(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for repo_id, remote, local_name in _MODELS:
        target = models_dir / local_name
        if target.exists():
            print(f"exists {target}")
        else:
            from huggingface_hub import hf_hub_download

            print(f"download {repo_id}/{remote} -> {target}")
            cached = hf_hub_download(repo_id=repo_id, filename=remote)
            shutil.copyfile(cached, target)
        paths[local_name] = target
    return paths


if __name__ == "__main__":
    download_models()
