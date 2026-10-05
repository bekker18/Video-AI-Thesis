"""Downloads model weights into models/<stage>/, reusing what is there. Standalone: python3 download_models.py [stage ...]"""

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
    revision: str = "main"


@dataclass(frozen=True)
class File:
    url: str
    filename: str


@dataclass(frozen=True)
class Archive:
    """A zip whose members are flattened into the stage directory."""

    url: str
    members: tuple[str, ...]


# Selectable by --seg-model; B1 and B2 are pinned to the Hub's safetensors conversions.
SEGFORMERS: dict[str, Model] = {
    "segformer-b0": Model("nvidia/segformer-b0-finetuned-ade-512-512"),
    "segformer-b1": Model(
        "nvidia/segformer-b1-finetuned-ade-512-512",
        revision="97252cce3b3ef4a4e5c2599f043500ad49872d6c",
    ),
    "segformer-b2": Model(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        revision="4585665b1bf59b90b831d5145d3e25d6e0743d03",
    ),
}
DEFAULT_SEGFORMER = "segformer-b1"

MODELS: dict[str, tuple[Model, ...]] = {
    "02-segmentation": (
        Model("uva-cv-lab/OmniShotCut", "OmniShotCut_ckpt.pth"),
        Model("apple/MobileCLIP-S2-OpenCLIP"),
    ),
    "05-global-experts": (
        Model("depth-anything/Depth-Anything-V2-Small-hf"),
        SEGFORMERS[DEFAULT_SEGFORMER],
        Model("facebook/mask2former-swin-tiny-coco-panoptic"),
    ),
    # official FairFace checkpoint, mirrored from Google Drive
    "07-conditional-experts": (
        Model(
            "anning01/fairface",
            "res34_fair_align_multi_7_20190809.pt",
            revision="2f3694ee5c86e230f08d5f81d839a20f0f809b2d",
        ),
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
        File(
            "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
            "io_places365.txt",
        ),
        File(RAM_TAGS, "ram_tag_list.txt"),
        File(f"{ANNOTATORS}/table5_pidinet.pth", "table5_pidinet.pth"),
    ),
    "07-conditional-experts": (
        File(
            f"{MEDIAPIPE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "face_landmarker.task",
        ),
        File(
            f"{MEDIAPIPE}/pose_landmarker/pose_landmarker_full/float16/1/"
            "pose_landmarker_full.task",
            "pose_landmarker_full.task",
        ),
        File(f"{HSEMOTION}/enet_b0_8_va_mtl.onnx", "emotion_enet_b0_va.onnx"),
        File(
            f"{OPENCV_ZOO}/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx",
            "text_crnn_en.onnx",
        ),
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
DEFAULT_DETECTOR = "yolo11s"

# Also on demand: the 4B variant is ~9 GB and should not arrive unasked.
CAPTIONERS: dict[str, str] = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_CAPTIONER = "qwen3-vl-2b"

# Fetched by the libraries themselves; TORCH_HOME keeps them in models/.
BACKBONES: dict[str, tuple[str, ...]] = {
    "02-segmentation": ("https://download.pytorch.org/models/resnet18-f37072fd.pth",),
    "07-conditional-experts": (
        "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    ),
}

# Gaze-LLE's code comes through torch hub, pinned to a commit; it loads DINOv2's by name.
GAZE_REPO = "fkryan/gazelle:2d78f9f3bd2a5db360354954ec1a4c526ee8ab55"
GAZE_MODEL = "gazelle_dinov2_vitb14_inout"

# Torch hub code, fetched without GitHub API validation, which is rate-limited when unauthenticated.
HUB_REPOS: dict[str, tuple[str, ...]] = {
    "07-conditional-experts": ("facebookresearch/dinov2", GAZE_REPO),
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
        # renamed once complete, so an interrupted download is retried
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
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


def captioner(stage: str, name: str) -> str:
    """Returns the snapshot revision, recorded so a caption can be traced to its weights."""
    from huggingface_hub import snapshot_download

    if name not in CAPTIONERS:
        raise ValueError(f"unknown captioner {name}; choose from {', '.join(CAPTIONERS)}")
    return Path(snapshot_download(CAPTIONERS[name], cache_dir=str(stage_dir(stage)))).name


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
                hf_hub_download(
                    model.repo,
                    model.filename,
                    revision=model.revision,
                    local_dir=str(model_dir),
                )
            else:
                # snapshot_download only fetches files missing from the cache.
                snapshot_download(
                    model.repo, revision=model.revision, cache_dir=str(model_dir)
                )

    for item in FILES.get(stage, ()):
        fetch(stage, item.url, item.filename)

    for archive in ARCHIVES.get(stage, ()):
        unpack(stage, archive)

    for url in BACKBONES.get(stage, ()):
        import torch

        torch.hub.load_state_dict_from_url(url, progress=False)  # no-op once cached

    for repo in HUB_REPOS.get(stage, ()):
        import torch

        torch.hub.list(repo, trust_repo=True, skip_validation=True, verbose=False)


def main() -> None:
    stages = sys.argv[1:] or sorted({*MODELS, *FILES, *ARCHIVES, *BACKBONES, *HUB_REPOS})
    for stage in stages:
        ensure(stage)
    if not sys.argv[1:]:
        detector("03-profiler", DEFAULT_DETECTOR)
        captioner("11-caption", DEFAULT_CAPTIONER)


if __name__ == "__main__":
    main()
