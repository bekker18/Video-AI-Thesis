"""Download the segmentation models into models/."""

from __future__ import annotations

import os
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


def _download_backbone(models_dir: Path) -> None:
    """Warm the torch.hub cache with OmniShotCut's ResNet-18 backbone.

    omnishotcut builds its backbone with torchvision's ``pretrained=True``, which
    fetches ImageNet weights on every run unless TORCH_HOME points somewhere
    persistent. Pull them once here so inference stays offline.
    """
    # The container sets TORCH_HOME already; this keeps local runs consistent.
    os.environ.setdefault("TORCH_HOME", str(models_dir))

    import torch
    from torchvision.models import ResNet18_Weights

    print(f"backbone cache {torch.hub.get_dir()}")
    ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)


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

    _download_backbone(models_dir)
    return paths


if __name__ == "__main__":
    download_models()
