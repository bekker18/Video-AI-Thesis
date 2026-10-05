"""11-caption: one short caption for the whole video, from a VLM shown 02's keyframes and expert hints."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

import download_models
from config import Config

STAGE = "11-caption"
SEGMENTER = "02-segmentation"  # MobileCLIP cache is reused from there
GLOBAL = "05-global-experts"  # the panoptic label list comes from its model config
CLIP_MODEL = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"
PANOPTIC_MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

MAX_WORDS = 12
MAX_TOKENS = 40
TOP = 5  # labels kept per hint line
ADDRESS = 0.5  # share of a person's samples looking into the camera for the hint to say so
ANIMALS = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)

INSTRUCTION = (
    f"Write one caption for this video: a single sentence of at most {MAX_WORDS} words. "
    "Say who or what is in it, what they are doing, where, and the visual style if notable, "
    'for example "A chef cooks in a busy restaurant kitchen, in black and white." '
    "Give the gist only: no names, no numbers, no step-by-step account."
)
FRAMES_INTRO = "Frames from the video, in time order:"
HINTS_INTRO = "Automatic analysis of the whole video, which may contain errors:"


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read(json_dir: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = json_dir / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _pick(shots: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Nearest keyframe to evenly spaced instants, so long segments contribute more frames."""
    keyframes: list[dict[str, Any]] = []
    for shot in shots:
        for keyframe in shot["keyframes"]:
            # Row order matches keyframe_embeddings.npy, which 02 writes in this order.
            keyframes.append(
                {**keyframe, "segment": int(shot["index"]), "row": len(keyframes)}
            )

    duration = float(shots[-1]["end_time"])
    picked: dict[int, dict[str, Any]] = {}
    for i in range(max(1, count)):
        instant = (i + 0.5) * duration / max(1, count)
        shot = next((s for s in shots if float(s["end_time"]) > instant), shots[-1])
        best = min(
            (k for k in keyframes if k["segment"] == int(shot["index"])),
            key=lambda k: (abs(float(k["time"]) - instant), int(k["row"])),
        )
        picked[int(best["row"])] = best
    return [picked[row] for row in sorted(picked)]


def _ranked(weights: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]


def _hints(
    aggregated: dict[str, Any],
    profile: dict[str, Any],
    relations: dict[str, Any],
    cfg: Config,
) -> str:
    """The whole video in seven lines, each weighted by how long its segments last."""
    shots = aggregated["shots"]
    floor = cfg.identikit_attention
    mediums: dict[str, float] = {}
    tags: dict[str, float] = {}
    regions: dict[str, float] = {}
    objects: dict[str, float] = {}
    animals: dict[str, float] = {}
    moves: dict[str, float] = {}
    with_people = 0.0
    for shot in shots:
        seconds = float(shot["end_time"]) - float(shot["start_time"])
        plastic = shot["plastic"]
        keyframes = max(1, int(plastic.get("n", 0)))
        for tag in plastic.get("tags") or []:
            tags[tag["label"]] = (
                tags.get(tag["label"], 0.0) + seconds * int(tag["n"]) / keyframes
            )
        coverage = (plastic.get("semantic") or {}).get("coverage") or {}
        for name, share in coverage.items():
            name = str(name).strip()  # ADE20K has "bed "
            regions[name] = regions.get(name, 0.0) + seconds * float(share["mean"])
        for name, p in ((plastic.get("medium") or {}).get("distribution") or {}).items():
            mediums[name] = mediums.get(name, 0.0) + seconds * float(p)
        for found in shot["objects"]:
            label = found["label"]
            if label not in ANIMALS:
                objects[label] = objects.get(label, 0.0) + int(found["frame_count"])
            elif float(found["share"]) >= cfg.subject_min:  # as 12's subjects
                animals[label] = animals.get(label, 0.0) + seconds * float(found["share"])
        movement = (shot.get("camera") or {}).get("movement")
        if movement and movement != "unknown":
            moves[movement] = moves.get(movement, 0.0) + seconds
        if shot["person_ids"]:
            with_people += seconds

    at_once = max((int(s["counts"]["person_max"]) for s in profile["shots"]), default=0)
    if at_once:
        duration = sum(float(s["end_time"]) - float(s["start_time"]) for s in shots)
        presence = (
            "most of the video" if with_people >= duration / 2 else "part of the video"
        )
        visible = f"up to {at_once} visible at once" if at_once > 1 else "one at a time"
        people = f"{visible}, on screen for {presence}"
        shares = [
            [
                float(r["attention"][side].get("share") or 0.0)
                for side in ("a_to_b", "b_to_a")
            ]
            for r in relations["relations"]
        ]
        # Head orientation, not gaze, so the wording is about facing.
        if any(min(pair) >= floor for pair in shares):
            people += "; some face each other"
        elif any(max(pair) >= floor for pair in shares):
            people += "; some turn towards others"
        addressing = sum(
            1
            for p in aggregated["persons"]
            if int((p.get("address") or {}).get("n", 0)) >= 2
            and float(p["address"]["share"]) >= ADDRESS
        )
        if addressing:
            people += "; one looks into the camera" if addressing == 1 else "; some look into the camera"
    else:
        people = "none"

    return "\n".join(
        [
            f"Medium: {', '.join(_ranked(mediums)[:1]) or 'unknown'}",
            f"Scene tags: {', '.join(_ranked(tags)) or 'none'}",
            f"Main regions: {', '.join(_ranked(regions)) or 'none'}",
            f"Objects detected: {', '.join(_ranked(objects)) or 'none'}",
            f"Animals: {', '.join(_ranked(animals)) or 'none'}",
            f"People: {people}",
            f"Camera movement: {', '.join(_ranked(moves)) or 'unknown'}",
        ]
    )


def _image(path: Path, longest: int) -> Any:
    from PIL import Image

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"missing keyframe {path}")
    height, width = image.shape[:2]
    scale = min(1.0, longest / max(height, width))
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _messages(frames: list[dict[str, Any]], hints: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": FRAMES_INTRO}]
    for frame in frames:
        content.append(
            {"type": "text", "text": f"\nFrame at {float(frame['time']):.1f} s:"}
        )
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"\n{HINTS_INTRO}\n{hints}"})
    content.append({"type": "text", "text": f"\n{INSTRUCTION}"})
    return [{"role": "user", "content": content}]


def _sentence(raw: str) -> str:
    text = raw.strip().strip('"').strip()
    match = re.match(r"(.+?[.!?])(\s|$)", text, re.DOTALL)
    return (match.group(1) if match else text).strip().strip('"')


def _generate(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Any],
    device: str,
) -> str:
    import torch

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    new = out[:, inputs["input_ids"].shape[1] :]
    return str(processor.batch_decode(new, skip_special_tokens=True)[0]).strip()


def _plain(label: str) -> str:
    """COCO panoptic names like wall-other-merged, reduced to the word a caption would use."""
    return re.sub(r"-(merged|other|stuff)", "", label).replace("-", " ").strip()


def _vocabulary(cache: Path) -> set[str]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(PANOPTIC_MODEL, cache_dir=str(cache))
    labels: dict[int, str] = config.id2label or {}
    return {_plain(str(label)) for label in labels.values()}


def _found(global_meta: dict[str, Any], shots: list[dict[str, Any]]) -> set[str]:
    """Everything a detector or segmenter reported anywhere in the video."""
    found = {
        _plain(str(s["label"]))
        for keyframe in global_meta["keyframes"]
        for s in keyframe["panoptic"]["segments"]
    }
    for shot in shots:
        coverage = (shot["plastic"].get("semantic") or {}).get("coverage") or {}
        found |= {_plain(str(name)) for name in coverage}
        found |= {str(o["label"]) for o in shot["objects"]}
        if shot["person_ids"]:
            found.add("person")
    return found


def _unsupported(caption: str, vocabulary: set[str], found: set[str]) -> list[str]:
    """Objects the caption names that nothing in the pipeline detected."""
    text = caption.lower()
    named: list[str] = []
    # Longest first, and a match is consumed, so "teddy bear" does not also count as "bear".
    for term in sorted(vocabulary, key=lambda t: (-len(t), t)):
        pattern = rf"\b{re.escape(term)}(e?s)?\b"
        if re.search(pattern, text):
            named.append(term)
            text = re.sub(pattern, " ", text)
    return sorted(t for t in named if t not in found)


def _clip_score(caption: str, embeddings: Array, device: str, cache: Path) -> float:
    """Mean cosine between the caption and the shown keyframes' 02 embeddings."""
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, cache_dir=str(cache)
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()
    with torch.no_grad():
        text = model.encode_text(tokenizer([caption]).to(device)).float()
        text /= text.norm(dim=-1, keepdim=True)
    return round(float((embeddings @ text.cpu().numpy().T).mean()), 4)


def run(cfg: Config) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = _device(cfg.device)
    segmentation = _read(cfg.json_dir, "segmentation.json", "02-segmentation")
    profile = _read(cfg.json_dir, "profile.json", "03-profiler")
    global_meta = _read(cfg.json_dir, "global.json", "05-global-experts")
    aggregated = _read(cfg.json_dir, "aggregated.json", "09-aggregation")
    relations = _read(cfg.json_dir, "relations.json", "10-relations")

    frames = _pick(segmentation["shots"], cfg.caption_frames)
    hints = _hints(aggregated, profile, relations, cfg)
    images = [_image(cfg.out_root / f["path"], cfg.caption_size) for f in frames]

    repo = download_models.CAPTIONERS[cfg.caption_model]
    revision = download_models.captioner(STAGE, cfg.caption_model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(repo, cache_dir=str(cfg.model_dir))
    model: Any = AutoModelForImageTextToText.from_pretrained(
        repo, cache_dir=str(cfg.model_dir), dtype=dtype
    )
    model = model.to(device).eval()

    started = time.perf_counter()
    raw = _generate(model, processor, _messages(frames, hints), images, device)
    seconds = round(time.perf_counter() - started, 2)
    text = _sentence(raw)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    embeddings: Array = np.load(cfg.embeddings_dir / "keyframe_embeddings.npy")
    shown_rows = embeddings[[int(f["row"]) for f in frames]]
    vocabulary = _vocabulary(download_models.stage_dir(GLOBAL))
    found = _found(global_meta, aggregated["shots"])

    meta = {
        "video": segmentation["video"],
        "model": repo,
        "revision": revision,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        # 12 compares these with 09's segments and ignores a caption written for others.
        "segments": [
            [int(s["start_frame"]), int(s["end_frame"])] for s in segmentation["shots"]
        ],
        "frames": [
            {k: f[k] for k in ("segment", "frame", "time", "path")} for f in frames
        ],
        "frame_size": cfg.caption_size,
        "hints": hints,
        "caption": text,
        "raw": raw,
        "words": len(text.split()),
        "over_limit": len(text.split()) > MAX_WORDS,
        "clip_score": _clip_score(
            text, shown_rows, device, download_models.stage_dir(SEGMENTER)
        ),
        "unsupported": _unsupported(text, vocabulary, found),
        "seconds": seconds,
        "note": (
            "clip_score uses the MobileCLIP model 05's tags came from, so a caption that "
            "repeats the tags scores higher for that alone"
        ),
    }
    (cfg.json_dir / "caption.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {"model": repo, "frames": len(frames), "caption": text}
