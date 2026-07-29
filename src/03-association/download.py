"""Download the association models into models/association/."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

MODELS_DIR = Path("models/association")

_RTDETR_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"
)

# (source kind, source, remote name, local name)
_MODELS = [
    ("hf", "AdamCodd/YOLOv11n-face-detection", "model.pt", "yolov11n-face.pt"),
    ("hf", "Ultralytics/YOLO11", "yolo11n.pt", "yolo11n.pt"),
    ("url", _RTDETR_URL, "rtdetr-l.pt", "rtdetr-l.pt"),
]


def download_models(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for kind, source, remote, local_name in _MODELS:
        target = models_dir / local_name
        if target.exists():
            print(f"exists {target}")
        elif kind == "hf":
            from huggingface_hub import hf_hub_download

            print(f"download {source}/{remote} -> {target}")
            cached = hf_hub_download(repo_id=source, filename=remote)
            shutil.copyfile(cached, target)
        else:
            print(f"download {source} -> {target}")
            with urlopen(source) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        paths[local_name] = target
    return paths


if __name__ == "__main__":
    download_models()
