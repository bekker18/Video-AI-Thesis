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
ANNOTATORS = "https://huggingface.co/lllyasviel/Annotators/resolve/main"
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"
HSEMOTION = (
    "https://github.com/av-savchenko/face-emotion-recognition/raw/main/models/"
    "affectnet_emotions/onnx"
)
INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7"


@dataclass(frozen=True)
class Model:
    repo: str
    filename: str | None = None  # a single file, or the whole snapshot when None


@dataclass(frozen=True)
class File:
    url: str
    filename: str


@dataclass(frozen=True)
class Archive:
    """A zip whose members are flattened into the stage directory."""

    url: str
    members: tuple[str, ...]


MODELS: dict[str, tuple[Model, ...]] = {
    "02-segmentation": (
        Model("uva-cv-lab/OmniShotCut", "OmniShotCut_ckpt.pth"),
        Model("apple/MobileCLIP-S2-OpenCLIP"),
    ),
    "05-global-experts": (
        Model("depth-anything/Depth-Anything-V2-Small-hf"),
        Model("nvidia/segformer-b1-finetuned-ade-512-512"),
        Model("facebook/mask2former-swin-tiny-coco-panoptic"),
    ),
}

PLACES365 = "http://places2.csail.mit.edu/models_places365"
RAM_TAGS = (
    "https://raw.githubusercontent.com/xinyu1205/recognize-anything/main/"
    "ram/data/ram_tag_list.txt"
)

FILES: dict[str, tuple[File, ...]] = {
    "03-profiler": (
        File(
            f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "face_yunet.onnx",
        ),
        # PP-OCRv3's DB detection stage, exported for OpenCV's dnn text detector.
        File(
            f"{OPENCV_ZOO}/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
            "text_ppocrv3.onnx",
        ),
    ),
    "05-global-experts": (
        File(f"{PLACES365}/resnet18_places365.pth.tar", "resnet18_places365.pth.tar"),
        File(
            "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
            "categories_places365.txt",
        ),
        File(RAM_TAGS, "ram_tag_list.txt"),
        File(f"{ANNOTATORS}/table5_pidinet.pth", "table5_pidinet.pth"),
    ),
    "07-conditional-experts": (
        # One pass gives mesh, head pose and blendshapes.
        File(
            f"{MEDIAPIPE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "face_landmarker.task",
        ),
        File(
            f"{MEDIAPIPE}/pose_landmarker/pose_landmarker_full/float16/1/"
            "pose_landmarker_full.task",
            "pose_landmarker_full.task",
        ),
        # Eight emotions plus valence and arousal from one EfficientNet-B0 pass.
        File(f"{HSEMOTION}/enet_b0_8_va_mtl.onnx", "emotion_enet_b0_va.onnx"),
        # CRNN recognition: the half of OCR 03 deliberately left out.
        File(
            f"{OPENCV_ZOO}/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx",
            "text_crnn_en.onnx",
        ),
        # Body appearance for 08-consolidation, which is the only descriptor most tracks get:
        # a face embedding exists for 4 of patrick.mp4's 69 tracks, a body crop for 68.
        File(
            f"{OPENCV_ZOO}/person_reid_youtureid/person_reid_youtu_2021nov.onnx",
            "reid_youtu.onnx",
        ),
    ),
}

# InsightFace ships its models as one release zip; two of the five are used.
ARCHIVES: dict[str, tuple[Archive, ...]] = {
    "07-conditional-experts": (
        Archive(
            f"{INSIGHTFACE}/buffalo_l.zip",
            ("w600k_r50.onnx", "genderage.onnx"),
        ),
    ),
}

# Fetched on demand, so switching detector does not pull every variant.
DETECTORS: dict[str, str] = {
    "yolo11n": f"{ULTRALYTICS}/yolo11n.pt",
    "yolo11s": f"{ULTRALYTICS}/yolo11s.pt",
    "yolo11m": f"{ULTRALYTICS}/yolo11m.pt",
    "rtdetr-l": f"{ULTRALYTICS}/rtdetr-l.pt",
}

# Backbone weights the model libraries fetch themselves.
# TORCH_HOME points into models/, so these are downloaded once instead of on every run.
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


def unpack(stage: str, archive: Archive) -> None:
    """Members are flattened, so callers reference them by basename like any other file."""
    import zipfile

    target = stage_dir(stage)
    wanted = [m for m in archive.members if not (target / Path(m).name).exists()]
    if not wanted:
        return

    zip_path = target / "_archive.zip"
    print(f"[{stage}] downloading {Path(archive.url).name}")
    urllib.request.urlretrieve(archive.url, zip_path)
    with zipfile.ZipFile(zip_path) as bundle:
        for member in wanted:
            with bundle.open(member) as src:
                (target / Path(member).name).write_bytes(src.read())
    zip_path.unlink()


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

    for archive in ARCHIVES.get(stage, ()):
        unpack(stage, archive)

    for url in BACKBONES.get(stage, ()):
        import torch

        torch.hub.load_state_dict_from_url(url, progress=False)  # no-op once cached


def main() -> None:
    stages = sys.argv[1:] or sorted({*MODELS, *FILES, *ARCHIVES, *BACKBONES})
    for stage in stages:
        ensure(stage)
    if not sys.argv[1:]:
        detector("03-profiler", "yolo11n")


if __name__ == "__main__":
    main()
