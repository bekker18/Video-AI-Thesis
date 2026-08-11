"""Downloads model weights into models/<stage>/. Already-present files are reused.

Runs automatically from main.py, or standalone:
    python3 download_models.py [stage ...]
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


@dataclass(frozen=True)
class Model:
    repo: str
    filename: str | None = None  # a single file, or the whole snapshot when None


@dataclass(frozen=True)
class File:
    url: str
    filename: str


MODELS: dict[str, tuple[Model, ...]] = {
    "02-segmentation": (
        Model("uva-cv-lab/OmniShotCut", "OmniShotCut_ckpt.pth"),
        Model("apple/MobileCLIP-S2-OpenCLIP"),
    ),
}

FILES: dict[str, tuple[File, ...]] = {
    "03-profiler": (
        File(f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
             "face_yunet.onnx"),
        # PP-OCRv3's DB detection stage, exported for OpenCV's dnn text detector.
        File(f"{OPENCV_ZOO}/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
             "text_ppocrv3.onnx"),
        File("https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
             "categories_places365.txt"),
    ),
}

# Fetched on demand, so switching detector does not pull every variant.
DETECTORS: dict[str, str] = {
    "yolo11n": f"{ULTRALYTICS}/yolo11n.pt",
    "yolo11s": f"{ULTRALYTICS}/yolo11s.pt",
    "yolo11m": f"{ULTRALYTICS}/yolo11m.pt",
    "rtdetr-l": f"{ULTRALYTICS}/rtdetr-l.pt",
}

# Backbone weights the model libraries fetch themselves. TORCH_HOME points into models/,
# so these are downloaded once instead of on every run.
BACKBONES: dict[str, tuple[str, ...]] = {
    "02-segmentation": ("https://download.pytorch.org/models/resnet18-f37072fd.pth",),
}


def stage_dir(stage: str) -> Path:
    path = ROOT / "models" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint(stage: str, filename: str) -> Path:
    return ROOT / "models" / stage / filename


def fetch(stage: str, url: str, filename: str) -> Path:
    target = stage_dir(stage) / filename
    if not target.exists():
        print(f"[{stage}] downloading {filename}")
        urllib.request.urlretrieve(url, target)
    return target


def detector(stage: str, name: str) -> Path:
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name}; choose from {', '.join(DETECTORS)}")
    return fetch(stage, DETECTORS[name], f"{name}.pt")


def ensure(stage: str) -> None:
    models = MODELS.get(stage)
    if models:
        from huggingface_hub import hf_hub_download, snapshot_download

        model_dir = stage_dir(stage)
        for model in models:
            if model.filename:
                if checkpoint(stage, model.filename).exists():
                    continue
                print(f"[{stage}] downloading {model.repo}/{model.filename}")
                hf_hub_download(model.repo, model.filename, local_dir=str(model_dir))
            else:
                # snapshot_download only fetches files missing from the cache.
                snapshot_download(model.repo, cache_dir=str(model_dir))

    for item in FILES.get(stage, ()):
        fetch(stage, item.url, item.filename)

    for url in BACKBONES.get(stage, ()):
        import torch

        torch.hub.load_state_dict_from_url(url, progress=False)  # no-op once cached


def main() -> None:
    stages = sys.argv[1:] or sorted({*MODELS, *FILES, *BACKBONES})
    for stage in stages:
        ensure(stage)
    if not sys.argv[1:]:
        detector("03-profiler", "yolo11n")


if __name__ == "__main__":
    main()
