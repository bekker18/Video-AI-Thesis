"""05-global-experts: the plastic, always-on branch.

Every expert here is person-independent and depends on no identity information, which is
why 04-router leaves them ungated. They run on the keyframes 02 selected.

Models are loaded one at a time and freed, so six of them never sit on the GPU at once.
Panoptic segment indices are per-keyframe geometry only: they are never matched across
frames and must not be read as identity, which comes solely from the gated branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

import download_models
from config import Config

STAGE = "05-global-experts"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SEMANTIC_MODELS = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}
PANOPTIC_MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"
CLIP_MODEL = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"

PLACES_WEIGHTS = "resnet18_places365.pth.tar"
PLACES_CATEGORIES = "categories_places365.txt"
TAG_LIST = "ram_tag_list.txt"
EDGE_WEIGHTS = "table5_pidinet.pth"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _free(model: Any) -> None:
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _keyframes(out_root: Path) -> tuple[list[str], list[Path], list[int]]:
    path = out_root / "segmentation.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run 02-segmentation first")
    meta = json.loads(path.read_text(encoding="utf-8"))

    keys: list[str] = []
    paths: list[Path] = []
    shots: list[int] = []
    for shot in meta["shots"]:
        for keyframe in shot["keyframes"]:
            name = Path(keyframe["path"]).stem  # "<shot>_<frame>"
            keys.append(name)
            paths.append(out_root / keyframe["path"])
            shots.append(int(shot["index"]))
    if not keys:
        raise RuntimeError("segmentation.json lists no keyframes")
    return keys, paths, shots


def _fit(image: Array, longest: int) -> Array:
    height, width = image.shape[:2]
    if max(height, width) <= longest:
        return image
    scale = longest / max(height, width)
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _write(path: Path, image: Array) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return f"global/{path.parent.name}/{path.name}"


def _classical(image: Array) -> dict[str, Any]:
    """Near-free descriptors: colour, contrast, focus, spatial organisation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue, green, red = (image[:, :, i].astype(np.float32) for i in range(3))

    # Hasler-Susstrunk colourfulness.
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colourfulness = float(
        np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    )

    edges = cv2.Canny(gray, 100, 200) > 0
    height, width = edges.shape
    half_h, half_w = height // 2, width // 2
    quadrants = [
        float(edges[:half_h, :half_w].mean()),
        float(edges[:half_h, half_w:].mean()),
        float(edges[half_h:, :half_w].mean()),
        float(edges[half_h:, half_w:].mean()),
    ]

    flipped = cv2.flip(gray, 1).astype(np.float32)
    symmetry = float(1.0 - np.abs(gray.astype(np.float32) - flipped).mean() / 255.0)

    return {
        "brightness": round(float(gray.mean()) / 255.0, 4),
        "contrast_rms": round(float(gray.std()) / 255.0, 4),
        "contrast_michelson": round(
            float(
                (int(gray.max()) - int(gray.min()))
                / max(1, int(gray.max()) + int(gray.min()))
            ),
            4,
        ),
        "saturation": round(
            float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0, 4
        ),
        "colourfulness": round(colourfulness, 3),
        "focus": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2),
        "mean_lab": [round(float(lab[:, :, i].mean()), 2) for i in range(3)],
        "edge_density": round(float(edges.mean()), 4),
        "edge_quadrants": [round(q, 4) for q in quadrants],
        "horizontal_symmetry": round(symmetry, 4),
    }


def _depth(images: list[Array], device: str, cache: Path, batch: int) -> list[Array]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=str(cache))
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL, cache_dir=str(cache)
    )
    model = model.to(device).eval()

    maps: list[Array] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in chunk]
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            predicted = model(**inputs).predicted_depth
            for j, im in enumerate(chunk):
                single = torch.nn.functional.interpolate(
                    predicted[j : j + 1].unsqueeze(1),
                    size=im.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                maps.append(single.squeeze().float().cpu().numpy())
    _free(model)
    return maps


def _normals(depth: Array) -> Array:
    """Surface normals by unprojecting the depth map into camera space and taking the cross
    product of neighbouring surface vectors.

    Depth Anything outputs relative inverse depth (disparity), so this converts to a
    pseudo-distance first and assumes a focal length. Taking gradients of the raw disparity
    instead yields a map dominated by depth discontinuities, with flat surfaces reading as
    featureless - orientation only shows up once the unprojection is done.
    """
    span = depth.max() - depth.min()
    disparity = (depth - depth.min()) / span if span else np.zeros_like(depth)
    z = 1.0 / (disparity.astype(np.float32) + 0.25)  # bounded pseudo-distance

    height, width = z.shape
    focal = float(max(height, width))
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float32) - width / 2.0,
        np.arange(height, dtype=np.float32) - height / 2.0,
    )
    points = np.dstack([us * z / focal, vs * z / focal, z])

    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
    if float(normal[:, :, 2].mean()) > 0:  # keep +Z pointing at the camera
        normal = -normal

    encoded: Array = ((normal + 1.0) * 127.5).astype(np.uint8)[
        :, :, ::-1
    ]  # BGR for imwrite
    return encoded


def _edges(images: list[Array], device: str, model_dir: Path) -> list[Array]:
    from controlnet_aux import PidiNetDetector
    from PIL import Image

    detector = PidiNetDetector.from_pretrained(str(model_dir), filename=EDGE_WEIGHTS)
    detector = detector.to(device)

    maps: list[Array] = []
    for image in images:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        out = detector(
            pil, detect_resolution=512, image_resolution=max(image.shape[:2])
        )
        edge = np.asarray(out.convert("L"))
        maps.append(
            cv2.resize(
                edge, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        )
    _free(detector)
    return maps


def _places_labels(path: Path) -> list[str]:
    """Lines look like '/a/airfield 0' or '/b/baseball_field 12'."""
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.append(line.split()[0].rsplit("/", 1)[-1].replace("_", " "))
    return labels


def _scene(
    images: list[Array], device: str, model_dir: Path, top: int, batch: int
) -> list[list[dict[str, Any]]]:
    import torch
    from torchvision.models import resnet18

    labels = _places_labels(model_dir / PLACES_CATEGORIES)
    model = resnet18(num_classes=len(labels))
    blob = torch.load(
        model_dir / PLACES_WEIGHTS, map_location="cpu", weights_only=False
    )
    state = {k.replace("module.", ""): v for k, v in blob["state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = [
                cv2.resize(
                    cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                    (224, 224),
                    interpolation=cv2.INTER_AREA,
                )
                for im in images[i : i + batch]
            ]
            tensor = torch.from_numpy(np.stack(chunk)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255).sub_(mean).div_(std)
            probs = model(tensor).softmax(dim=-1).cpu().numpy()
            for row in probs:
                order = np.argsort(-row)[:top]
                out.append(
                    [
                        {"label": labels[int(k)], "score": round(float(row[k]), 4)}
                        for k in order
                    ]
                )
    _free(model)
    return out


def _tags(
    embeddings: Array, device: str, model_dir: Path, clip_cache: Path, top: int
) -> list[list[dict[str, Any]]]:
    """Zero-shot against RAM's tag vocabulary, reusing the keyframe embeddings 02 saved.
    No image is decoded and no vision forward pass runs here."""
    import open_clip
    import torch

    vocabulary = [
        line.strip()
        for line in (model_dir / TAG_LIST).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, cache_dir=str(clip_cache)
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    chunks: list[Array] = []
    with torch.no_grad():
        for i in range(0, len(vocabulary), 256):
            tokens = tokenizer([f"a photo of {t}" for t in vocabulary[i : i + 256]]).to(
                device
            )
            features = model.encode_text(tokens).float()
            features /= features.norm(dim=-1, keepdim=True)
            chunks.append(features.cpu().numpy())
    _free(model)

    scores = embeddings @ np.concatenate(chunks).T
    out: list[list[dict[str, Any]]] = []
    for row in scores:
        order = np.argsort(-row)[:top]
        out.append(
            [
                {"label": vocabulary[int(k)], "score": round(float(row[k]), 4)}
                for k in order
            ]
        )
    return out


def _semantic(
    images: list[Array], device: str, cache: Path, name: str, batch: int
) -> tuple[list[Array], dict[int, str]]:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    repo = SEMANTIC_MODELS[name]
    processor = SegformerImageProcessor.from_pretrained(repo, cache_dir=str(cache))
    model = SegformerForSemanticSegmentation.from_pretrained(repo, cache_dir=str(cache))
    model = model.to(device).eval()
    names = {int(k): v for k, v in model.config.id2label.items()}

    maps: list[Array] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in chunk]
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            sizes = [im.shape[:2] for im in chunk]
            for j, size in enumerate(sizes):
                upscaled = torch.nn.functional.interpolate(
                    logits[j : j + 1], size=size, mode="bilinear", align_corners=False
                )
                maps.append(upscaled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy())
    _free(model)
    return maps, names


def _panoptic(
    images: list[Array], device: str, cache: Path, batch: int
) -> tuple[list[Array], list[list[dict[str, Any]]]]:
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(PANOPTIC_MODEL, cache_dir=str(cache))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        PANOPTIC_MODEL, cache_dir=str(cache)
    )
    model = model.to(device).eval()
    names = {int(k): v for k, v in model.config.id2label.items()}

    maps: list[Array] = []
    listings: list[list[dict[str, Any]]] = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in chunk]
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            outputs = model(**inputs)
            results = processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[im.shape[:2] for im in chunk]
            )

            for result in results:
                label_map = result["segmentation"].cpu().numpy()
                maps.append(np.clip(label_map + 1, 0, 255).astype(np.uint8))

                pixels = label_map.size
                segments: list[dict[str, Any]] = []
                for info in result["segments_info"]:
                    mask = label_map == info["id"]
                    if not mask.any():
                        continue
                    ys, xs = np.nonzero(mask)
                    segments.append(
                        {
                            # Scoped to this keyframe only. Never matched across frames.
                            "segment_index": int(info["id"]),
                            "label": names.get(
                                int(info["label_id"]), str(info["label_id"])
                            ),
                            "score": round(float(info.get("score", 1.0)), 3),
                            "area": round(float(mask.sum()) / pixels, 5),
                            "bbox": [
                                int(xs.min()),
                                int(ys.min()),
                                int(xs.max()),
                                int(ys.max()),
                            ],
                        }
                    )
                listings.append(sorted(segments, key=lambda s: -float(s["area"])))
    _free(model)
    return maps, listings


def _coverage(label_map: Array, names: dict[int, str], top: int) -> dict[str, float]:
    ids, counts = np.unique(label_map, return_counts=True)
    order = np.argsort(-counts)[:top]
    total = float(label_map.size)
    return {
        names.get(int(ids[k]), str(ids[k])): round(float(counts[k]) / total, 4)
        for k in order
    }


def run(cfg: Config) -> dict[str, Any]:
    device = _device(cfg.device)
    keys, paths, shots = _keyframes(cfg.out_root)
    clip_cache = download_models.stage_dir("02-segmentation")

    raw = [cv2.imread(str(p)) for p in paths]
    if any(im is None for im in raw):
        raise RuntimeError("some keyframes referenced by segmentation.json are missing")
    images = [_fit(im, cfg.map_size) for im in raw]
    batch = cfg.expert_batch

    records: list[dict[str, Any]] = [
        {
            "key": k,
            "shot": s,
            "frame": int(k.split("_")[1]),
            "classical": _classical(im),
        }
        for k, s, im in zip(keys, shots, images)
    ]

    root = cfg.out_root / "global"
    depth_maps = _depth(images, device, cfg.model_dir, batch)
    for record, depth in zip(records, depth_maps):
        low, high = float(depth.min()), float(depth.max())
        span = high - low
        scaled = ((depth - low) / span * 65535.0) if span else np.zeros_like(depth)
        record["depth"] = {
            "path": _write(
                root / "depth" / f"{record['key']}.png", scaled.astype(np.uint16)
            ),
            "min": round(low, 4),
            "max": round(high, 4),
            "relative": True,
        }
        record["normals"] = {
            "path": _write(root / "normals" / f"{record['key']}.png", _normals(depth)),
            "source": "depth gradient",
        }
    del depth_maps

    for record, edge in zip(records, _edges(images, device, cfg.model_dir)):
        record["edges"] = {
            "path": _write(root / "edges" / f"{record['key']}.png", edge),
            "density": round(float((edge > 128).mean()), 4),
        }

    for record, scene in zip(
        records, _scene(images, device, cfg.model_dir, cfg.scenes, batch)
    ):
        record["scene"] = scene

    embeddings: Array = np.load(cfg.out_root / "keyframe_embeddings.npy")
    for record, tags in zip(
        records, _tags(embeddings, device, cfg.model_dir, clip_cache, cfg.tags)
    ):
        record["tags"] = tags

    semantic_maps, semantic_names = _semantic(
        images, device, cfg.model_dir, cfg.seg_model, batch
    )
    for record, label_map in zip(records, semantic_maps):
        record["semantic"] = {
            "path": _write(root / "semantic" / f"{record['key']}.png", label_map),
            "label_space": "ade20k",
            "coverage": _coverage(label_map, semantic_names, cfg.seg_top),
        }
    del semantic_maps

    panoptic_maps, listings = _panoptic(images, device, cfg.model_dir, batch)
    for record, label_map, segments in zip(records, panoptic_maps, listings):
        record["panoptic"] = {
            "path": _write(root / "panoptic" / f"{record['key']}.png", label_map),
            "label_space": "coco_panoptic",
            "segment_count": len(segments),
            "segments": segments[: cfg.seg_top],
        }
    del panoptic_maps

    meta = {
        "video": str(cfg.video),
        "keyframe_count": len(records),
        "shot_count": len(set(shots)),
        "device": device,
        "map_size": cfg.map_size,
        "models": {
            "depth": DEPTH_MODEL,
            "edges": "PiDiNet",
            "scene": "places365-resnet18",
            "tags": CLIP_MODEL,
            "semantic": SEMANTIC_MODELS[cfg.seg_model],
            "panoptic": PANOPTIC_MODEL,
        },
        "label_spaces": {"semantic": "ade20k", "panoptic": "coco_panoptic"},
        "note": "panoptic segment_index is per-keyframe geometry, never an identity",
        "keyframes": records,
    }
    (cfg.out_root / "global.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "keyframes"}
