"""People and topics across models per keyframe, after FRESCO's Tables 3-5, against prior expectations:
python3 src/validation/consistency.py"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Video: TypeAlias = dict[str, Any]  # name, folder, caption, keyframes
Totals: TypeAlias = dict[str, Array]

AVE, REELS = COLLECTIONS = ("ave", "reels")
NAMES = {AVE: "AVE film scenes", REELS: "Instagram reels"}
SHORT = {AVE: "AVE", REELS: "reels"}
MIN_N = 20  # videos needed to judge
FACE_INPUT = 640  # fixed YuNet input: a size per image makes it return phantom faces, so does 1280
FACE_CONF = 0.6  # 03's --face-conf
YUNET = ROOT / "models" / "03-profiler" / "face_yunet.onnx"
EXPERTS = ROOT / "models" / "05-global-experts"
PANOPTIC_MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"  # 05's
PANOPTIC_BATCH = 8  # 05's --expert-batch, so the same inputs
THINGS = 80  # COCO panoptic ids below this are countable objects
TEXT_CACHE = ROOT / "models" / "validation"  # CLIP ViT-L/14, OpenAI's weights
THRESHOLDS = (0.80, 0.85, 0.90)  # FRESCO's, on CLIP ViT-L/14 text cosine
CHANCE_DRAWS = 10  # times each first model's labels are dealt to other keyframes
PLANTED_DROP = 0.25  # share of the common topics a shuffle must remove; set after a 10-point bar missed
MOBILECLIP = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"  # the pipeline's encoder, 05's tags
MOBILECLIP_CACHE = ROOT / "models" / "02-segmentation"
PROBES = (
    "sedan",
    "land vehicle",
    "sport car",
)  # FRESCO: each scores 0.83 against "car"
JUDGED = ("C7", "C8", "C9", "C10", "C11", "C12")  # the expectations an encoder can move
MAX_GAP, MEAN_GAP = (
    0.05,
    0.02,
)  # encoders agree: largest and mean Table 5 share difference

# key, name, counts people (else presence only)
SOURCES: tuple[tuple[str, str, bool], ...] = (
    ("faces", "faces, YuNet on the whole keyframe", True),
    ("inbox", "faces, YuNet in YOLO's person boxes (03)", True),
    ("yolo", "persons, YOLO11s (03)", True),
    ("panoptic", "persons, Mask2Former panoptic, every segment", True),
    ("semantic", "person class, SegFormer ADE20K (05)", False),
    ("tags", "person words in the top 15 tags, MobileCLIP zero-shot (05)", False),
    ("tracked", "people tracked at the frame, ByteTrack (06)", True),
    ("persons", "persons at the frame, after 08", True),
    ("truth", "AVE's num-people label", True),
)
COUNTING = ("faces", "inbox", "yolo", "panoptic", "tracked", "persons")
BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-6", 3, 6),
    ("7-12", 7, 12),
    ("13-30", 13, 30),
    ("31+", 31, 10**9),
)
AVE_BINS = ("0", "1", "2", "3", "4", "5+")
TOPICS = ("tags", "objects", "semantic", "panoptic")
TOPIC_NAMES = {
    "tags": "tags (MobileCLIP top 15)",
    "objects": "objects (YOLO11s)",
    "semantic": "semantic (SegFormer)",
    "panoptic": "panoptic things (Mask2Former)",
}
PAIRS = (
    ("tags", "objects"),
    ("tags", "semantic"),
    ("tags", "panoptic"),
    ("objects", "semantic"),
    ("objects", "panoptic"),
    ("semantic", "panoptic"),
)
PARTS = ("first", "common", "second")

# Table 6: name, item, model a, model b; plausibility.py's checks and bands, set before its first run
AGREEMENTS = (
    (
        "smile",
        "face sample",
        "HSEmotion P(happiness) (07)",
        "MediaPipe smile blendshapes (07)",
    ),
    (
        "looking at the camera",
        "person, 3+ samples",
        "address: head pose and Gaze-LLE (09)",
        "MobileCLIP attribute (07)",
    ),
    (
        "outdoor",
        "keyframe",
        "Places365 outdoor probability (05)",
        "SegFormer sky coverage (05)",
    ),
    ("persons per keyframe", "keyframe", "YOLO11s (03)", "Mask2Former, every segment"),
    (
        "people per segment",
        "segment, 1+ visible",
        "most visible at once (03)",
        "persons followed (08)",
    ),
)
AGREE_PASS, AGREE_WEAK = 0.6, 0.4  # Spearman
SMILE = ("mouthSmileLeft", "mouthSmileRight")
LOOKING = "a person looking at the camera"
MIN_ADDRESS = 3  # samples behind an address share

# WordNet's person.n.01 by first noun sense over RAM's tags, reviewed by hand, plus everyday and group words
PERSON_WORDS = frozenset(
    {
        "actor", "actress", "adult", "airman", "animator", "architect", "artist", "astronaut", "astronomer",
        "athlete", "audience", "author", "baby", "baker", "ballerina", "ballet dancer", "barber",
        "bartender", "baseball player", "basketball coach", "basketball player", "bassist", "beekeeper",
        "bishop", "blacksmith", "bodybuilder", "boxer", "boy", "boyfriend", "bride", "bridesmaid",
        "brother", "brunette", "builder", "business executive", "business woman", "businessman",
        "businesswoman", "butcher", "captain", "carpenter", "catcher", "celebrity", "champion", "chap",
        "chef", "chemist", "child", "client", "climber", "coach", "college student", "commander",
        "commentator", "commuter", "competitor", "composer", "conductor", "construction worker", "cook",
        "couple", "cousin", "cowboy", "craftsman", "cricketer", "crowd", "customer", "dancer", "daughter",
        "defender", "dentist", "designer", "detective", "director", "disciple", "diver", "dj", "doctor",
        "driver", "drummer", "duchess", "dude", "educator", "electrician", "engineer", "entertainer",
        "equestrian", "explorer", "family", "farmer", "fashion designer", "fashion model", "father",
        "film director", "film producer", "fireman", "fisherman", "flight attendant", "florist",
        "flower girl", "football coach", "football player", "friend", "gardener", "geisha", "gentleman",
        "girl", "girlfriend", "gladiator", "goalkeeper", "golfer", "graduate", "grandfather",
        "grandmother", "grandparent", "groom", "guard", "guitarist", "guy", "gymnast", "hacker", "hero",
        "hiker", "historian", "hockey player", "host", "human", "husband", "individual", "jockey",
        "journalist", "judge", "kid", "king", "lady", "lawyer", "leader", "lifeguard", "lumberjack",
        "magician", "mahout", "maid", "man", "manager", "matador", "mechanic", "miner", "minister", "monk",
        "mother", "motorcyclist", "mountaineer", "musician", "nanny", "neighbor", "newborn", "newlywed",
        "nun", "nurse", "officer", "official", "painter", "paramedic", "paratrooper", "participant",
        "partner", "passenger", "patient", "pedestrian", "people", "performer", "person", "philosopher",
        "photographer", "physicist", "pianist", "pilgrim", "pilot", "pirate", "player", "plumber",
        "police officer", "policeman", "policewoman", "politician", "pope", "preacher", "president",
        "prince", "princess", "professor", "prophet", "protester", "queen", "rapper", "referee",
        "rescuer", "researcher", "rider", "rock climber", "runner", "sailor", "samurai", "santa claus",
        "saxophonist", "scientist", "scout", "sculptor", "secretary", "senior", "shepherd", "shopper",
        "sibling", "singer", "sister", "skateboarder", "skater", "skier", "snowboarder", "socialite",
        "soldier", "son", "spectator", "striker", "student", "supermodel", "supporter", "surfer",
        "surgeon", "swimmer", "tailor", "teacher", "team", "technician", "teenager", "tennis player",
        "toddler", "tour guide", "tourist", "trainer", "twin", "vendor", "vet", "veteran", "violinist",
        "violist", "volleyball player", "volunteer", "waiter", "warrior", "welder", "wife", "witch",
        "woman", "worker", "wrestler", "writer", "youth", "zombie",
    }
)  # fmt: skip
IRREGULAR = {"children": "child", "people": "people", "wives": "wife"}

# FRESCO's models (its Table 2), named in its reference rows
FRESCO_NAMES = {
    "faces": "faces (RetinaFace)",
    "yolo": "objects (UniDet)",
    "panoptic": "panoptic things (Mask2Former Swin-L)",
    "tags": "tags (RAM++)",
    "objects": "objects (UniDet)",
    "semantic": "semantic (Mask2Former Swin-L, ADE20K)",
}
# FRESCO's Table 3: (images with a person, people per image) on OpenImages and FFHQ in-the-wild
FRESCO_PEOPLE: dict[
    str, tuple[tuple[float, float | None], tuple[float, float | None]]
] = {
    "faces": ((0.9026, 2.07), (1.0, 1.75)),
    "yolo": ((0.8831, 1.85), (0.9921, 2.03)),
    "panoptic": ((0.9201, 2.14), (0.9984, 2.27)),
    "semantic": ((0.9555, None), (0.9997, None)),
    "tags": ((0.9610, None), (0.9999, None)),
    "caption": ((0.9535, None), (0.9998, None)),
}
# FRESCO's Table 4, bins 0/1, 2, 3-6, 7-12, 13-30, 31+
FRESCO_GROUPS: dict[str, dict[str, tuple[float, ...]]] = {
    "OpenImages": {
        "faces": (67.88, 15.78, 11.94, 2.85, 1.25, 0.30),
        "yolo": (60.84, 17.38, 18.63, 3.00, 0.15, 0.00),
        "panoptic": (61.04, 16.28, 16.18, 5.54, 0.95, 0.00),
    },
    "FFHQ in-the-wild": {
        "faces": (61.73, 21.80, 15.06, 1.17, 0.24, 0.00),
        "yolo": (52.70, 21.80, 23.56, 1.94, 0.00, 0.00),
        "panoptic": (51.99, 21.79, 20.60, 5.40, 0.22, 0.00),
    },
}
# FRESCO's Table 5 on OpenImages: (first, common, second) at 0.80, 0.85 and 0.90
FRESCO_TOPICS: dict[tuple[str, str], tuple[tuple[float, float, float], ...]] = {
    ("tags", "objects"): ((60.12, 29.86, 10.01), (71.91, 11.58, 16.52), (75.16, 5.47, 19.37)),
    ("tags", "semantic"): ((50.37, 30.09, 19.53), (64.40, 9.23, 26.37), (67.00, 4.63, 28.36)),
    ("tags", "panoptic"): ((72.03, 25.70, 2.27), (86.22, 7.82, 5.96), (88.21, 3.48, 8.31)),
    ("objects", "semantic"): ((23.16, 26.30, 50.54), (30.29, 16.21, 53.49), (30.83, 15.21, 53.97)),
    ("objects", "panoptic"): ((47.94, 43.38, 8.68), (62.09, 27.96, 9.95), (62.69, 26.87, 10.43)),
    ("semantic", "panoptic"): ((65.49, 34.51, 0.00), (71.84, 28.16, 0.00), (72.27, 27.73, 0.00)),
}  # fmt: skip


@cache
def _ade_names(repo: str) -> dict[int, str]:
    """SegFormer's class names: 05's semantic maps store class ids."""
    config = next(
        (EXPERTS / f"models--{repo.replace('/', '--')}").glob("snapshots/*/config.json")
    )
    labels = json.loads(config.read_text(encoding="utf-8"))["id2label"]
    return {int(k): str(v).strip() for k, v in labels.items()}


def _semantic(root: Path, path: str, names: dict[int, str]) -> dict[str, float]:
    """Every class on the map with its share of pixels; 05's coverage keeps only the 10 largest."""
    label_map = cv2.imread(str(root / path), cv2.IMREAD_UNCHANGED)
    if label_map is None:
        raise RuntimeError(f"missing semantic map {root / path}")
    ids, pixels = np.unique(label_map, return_counts=True)
    return {names[int(i)]: float(n) / label_map.size for i, n in zip(ids, pixels)}


def _read(root: Path, name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(
        (root / "json" / name).read_text(encoding="utf-8")
    )
    return found


def _singular(word: str) -> list[str]:
    forms = [word, IRREGULAR.get(word, word)]
    if word.endswith("men"):
        forms.append(word[:-3] + "man")
    if word.endswith("ies"):
        forms.append(word[:-3] + "y")
    if word.endswith("es"):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    return forms


def _names_person(text: str) -> bool:
    """A person word, or two, in the text, plurals reduced."""
    words = re.findall(r"[a-z]+", text.lower())
    grams = words + [f"{a} {b}" for a, b in pairwise(words)]
    return any(form in PERSON_WORDS for gram in grams for form in _singular(gram))


def _ave_shots(scene: dict[str, Any], fps: float) -> list[tuple[int, int, str]]:
    """AVE's shots as frame spans with one num-people label; 5 reads 5 or more."""
    out: list[tuple[int, int, str]] = []
    for shot in scene.values():
        labels = shot.get("num-people", [])
        if len(labels) == 1 and labels[0] in {"0", "1", "2", "3", "4", "5"}:
            start = round(float(shot["start-time"][0]) * fps)
            end = round(float(shot["end-time"][0]) * fps)
            out.append((start, end, "5+" if labels[0] == "5" else labels[0]))
    return out


def _video(root: Path, ave: dict[str, Any]) -> Video:
    segmentation = _read(root, "segmentation.json")
    profile = _read(root, "profile.json")
    global_meta = _read(root, "global.json")
    tracks = _read(root, "tracks.json")
    identity = _read(root, "identity.json")
    conditional = _read(root, "conditional.json")
    aggregated = _read(root, "aggregated.json")
    kit = json.loads((root / "identikit.json").read_text(encoding="utf-8"))
    caption = root / "json" / "caption.json"

    owner = {
        (int(t["segment"]), int(t["track_id"])): int(p["person_id"])
        for p in identity["persons"]
        for t in p["tracks"]
    }
    yields = {
        (int(t["segment"]), int(t["track_id"])): set(t.get("yields", []))
        for p in identity["persons"]
        for t in p["tracks"]
    }
    tracked: dict[int, int] = {}
    persons: dict[int, set[int]] = {}
    for shot in tracks["shots"]:
        for track in shot.get("tracks", []):
            person = owner[(int(shot["index"]), int(track["track_id"]))]
            for row in track["frames"]:
                frame = int(row["frame"])
                tracked[frame] = tracked.get(frame, 0) + 1
                persons.setdefault(frame, set()).add(person)
    profiled = {
        (int(s["index"]), int(k["frame"])): k
        for s in profile["shots"]
        for k in s["keyframes"]
    }
    spans = (
        _ave_shots(ave[root.name], float(segmentation["fps"]))
        if root.name in ave
        else []
    )

    names = _ade_names(str(global_meta["models"]["semantic"]))
    keyframes: list[dict[str, Any]] = []
    for k in global_meta["keyframes"]:
        frame = int(k["frame"])
        found = profiled[(int(k["shot"]), frame)]
        segments = k["panoptic"]["segments"]
        coverage = _semantic(root, str(k["semantic"]["path"]), names)
        tags = [str(t["label"]) for t in k["tags"]]
        truth = next((label for a, b, label in spans if a <= frame <= b), None)
        keyframes.append(
            {
                "key": str(k["key"]),
                "path": root / "keyframes" / f"{k['key']}.jpg",
                "yolo": int(found["person_count"]),
                "inbox": int(found["face_count"]),
                "kept": sum(1 for s in segments if s["label"] == "person"),  # 05's
                "segment_count": int(k["panoptic"]["segment_count"]),
                "cut": int(k["panoptic"]["segment_count"]) > len(segments),
                "semantic": "person" in coverage,
                "tags": any(t.lower() in PERSON_WORDS for t in tags),
                "tracked": tracked.get(frame, 0),
                "persons": len(persons.get(frame, set())),
                "truth": truth,
                "outdoor": float(k["outdoor"]),
                "sky": coverage.get("sky", 0.0),
                "labels": {
                    "tags": tags,
                    "objects": sorted(
                        {str(b["label"]) for b in found["persons"] + found["objects"]}
                    ),
                    "semantic": sorted(coverage),
                },
            }
        )

    # Table 6's pairs; a smile blendshape outside a sample's top ones counts as 0
    smile = [
        (
            float(s["emotion"]["scores"]["happiness"]),
            float(np.mean([(s.get("blendshapes") or {}).get(b, 0.0) for b in SMILE])),
        )
        for shot in conditional["shots"]
        for t in shot["tracks"]
        for s in t["series"]
        if s.get("emotion")
        and int(s["frame"])
        not in yields.get((int(shot["index"]), int(t["track_id"])), set())
    ]
    looking: list[tuple[float, float]] = []
    for p in aggregated["persons"]:
        scores = {str(a["label"]): float(a["score"]) for a in p.get("attributes") or []}
        if (
            int((p.get("address") or {}).get("n", 0)) >= MIN_ADDRESS
            and LOOKING in scores
        ):
            looking.append((float(p["address"]["share"]), scores[LOOKING]))
    return {
        "name": root.name,
        "folder": root.parent.name,
        "pass_zero": "same_face" in (identity.get("thresholds") or {}),
        "caption": json.loads(caption.read_text(encoding="utf-8"))["caption"]
        if caption.exists()
        else None,
        "keyframes": keyframes,
        "map_size": int(global_meta["map_size"]),
        "pairs": {
            "smile": smile,
            "looking at the camera": looking,
            "outdoor": [(k["outdoor"], k["sky"]) for k in keyframes],
            "people per segment": [
                (float(s["visible_at_once"]), float(s["followed"]))
                for s in kit["enunciative"]["segments"]
                if s["visible_at_once"]  # elsewhere 06-08 mostly never ran
            ],
        },
    }


def _load(
    processed: Path, folders: dict[str, str], ave: dict[str, Any]
) -> dict[str, list[Video]]:
    return {
        c: [
            _video(d, ave)
            for d in sorted((processed / folders[c]).iterdir())
            if (d / "identikit.json").exists()
        ]
        if (processed / folders[c]).is_dir()
        else []
        for c in COLLECTIONS
    }


def _faces(detector: Any, path: Path) -> int:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"missing keyframe {path}")
    height, width = image.shape[:2]
    scale = FACE_INPUT / max(height, width)
    small = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas: Array = np.zeros((FACE_INPUT, FACE_INPUT, 3), dtype=np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    _, found = detector.detect(canvas)
    if found is None:
        return 0
    return sum(1 for row in found if bool(np.isfinite(row[:4]).all()))


def _add_faces(videos: list[Video], cache: Path) -> None:
    """YuNet over each whole keyframe, cached beside the report for these videos only."""
    found: dict[str, int] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    kept: dict[str, int] = {}
    detector: Any = None
    for v in videos:
        for k in v["keyframes"]:
            name = f"{v['folder']}/{v['name']}/{k['key']}"
            if name not in found:
                if detector is None:
                    detector = cv2.FaceDetectorYN.create(
                        str(YUNET), "", (FACE_INPUT, FACE_INPUT), FACE_CONF, 0.3, 5000
                    )
                found[name] = _faces(detector, k["path"])
            k["faces"] = kept[name] = found[name]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(kept, sort_keys=True), encoding="utf-8")


def _fit(image: Array, longest: int) -> Array:
    """05's resize before its experts."""
    height, width = image.shape[:2]
    if max(height, width) <= longest:
        return image
    scale = longest / max(height, width)
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _things(
    v: Video, model: Any, processor: Any, names: dict[int, str]
) -> dict[str, Any]:
    """Every Mask2Former segment of a video's keyframes, run as 05 runs it."""
    import torch

    out: dict[str, Any] = {}
    keyframes = v["keyframes"]
    for start in range(0, len(keyframes), PANOPTIC_BATCH):
        chunk = keyframes[start : start + PANOPTIC_BATCH]
        images = [_fit(cv2.imread(str(k["path"])), v["map_size"]) for k in chunk]
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        with torch.no_grad():
            inputs = processor(images=rgb, return_tensors="pt").to(model.device)
            results = processor.post_process_panoptic_segmentation(
                model(**inputs), target_sizes=[image.shape[:2] for image in images]
            )
        for k, result in zip(chunk, results):
            present = set(np.unique(result["segmentation"].cpu().numpy()).tolist())
            labels = [
                int(s["label_id"])
                for s in result["segments_info"]
                if s["id"] in present
            ]
            out[str(k["key"])] = {
                "segments": len(labels),
                "things": sorted(names[i] for i in labels if i < THINGS),
            }
    return out


def _mask2former() -> tuple[Any, Any, dict[int, str]]:
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(
        PANOPTIC_MODEL, cache_dir=str(EXPERTS)
    )
    model: Any = Mask2FormerForUniversalSegmentation.from_pretrained(
        PANOPTIC_MODEL, cache_dir=str(EXPERTS)
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, processor, {int(i): str(n) for i, n in model.config.id2label.items()}


def _add_things(videos: list[Video], cache: Path) -> None:
    """Mask2Former's things on each keyframe, all of them: 05 keeps the 10 largest segments. Cached for
    these videos only."""
    found: dict[str, Any] = (
        json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    )
    kept: dict[str, Any] = {}
    loaded: tuple[Any, Any, dict[int, str]] | None = None
    for v in videos:
        name = f"{v['folder']}/{v['name']}"
        if name not in found:
            loaded = loaded or _mask2former()
            found[name] = _things(v, *loaded)
        kept[name] = found[name]
        for k in v["keyframes"]:
            seen = found[name][str(k["key"])]
            k["panoptic"] = seen["things"].count("person")
            k["segments"] = seen["segments"]
            k["labels"]["panoptic"] = sorted(set(seen["things"]))
        v["pairs"]["persons per keyframe"] = [
            (float(k["yolo"]), float(k["panoptic"])) for k in v["keyframes"]
        ]
    if loaded is not None:
        import torch

        del loaded
        torch.cuda.empty_cache()
    cache.write_text(json.dumps(kept, sort_keys=True), encoding="utf-8")


def _rerun_check(videos: list[Video]) -> dict[str, float]:
    """The rerun against 05's output: segment counts, and persons where 05 kept every segment."""
    keyframes = [k for v in videos for k in v["keyframes"]]
    whole = [k for k in keyframes if not k["cut"]]
    return {
        "segments_match": round(
            float(np.mean([k["segments"] == k["segment_count"] for k in keyframes])), 4
        ),
        "persons_match_uncut": round(
            float(np.mean([k["panoptic"] == k["kept"] for k in whole])), 4
        ),
    }


def _vocabulary(videos: list[Video]) -> list[str]:
    return sorted(
        {
            label
            for v in videos
            for k in v["keyframes"]
            for labels in k["labels"].values()
            for label in labels
        }
    )


def _embed(texts: list[str], mobile: bool = False) -> Array:
    """Unit text vectors of bare labels, as FRESCO's ClipTextScore: CLIP ViT-L/14, or the pipeline's MobileCLIP."""
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mobile:
        model, _, _ = open_clip.create_model_and_transforms(
            MOBILECLIP, cache_dir=str(MOBILECLIP_CACHE)
        )
        tokenizer = open_clip.get_tokenizer(MOBILECLIP, cache_dir=str(MOBILECLIP_CACHE))
    else:
        # OpenAI trained these weights with QuickGELU; plain GELU bends the vectors
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=str(TEXT_CACHE),
            force_quick_gelu=True,
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device).eval()
    rows: list[Array] = []
    with torch.no_grad():
        for start in range(0, len(texts), 256):
            features = model.encode_text(
                tokenizer(texts[start : start + 256]).to(device)
            ).float()
            rows.append((features / features.norm(dim=-1, keepdim=True)).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(rows)


def _topic_counts(
    first: list[str],
    second: list[str],
    sim: Array,
    index: dict[str, int],
    thresholds: tuple[float, ...],
) -> Array:
    """FRESCO's count per threshold: first's labels matched by any of second's, first's
    unmatched, second's unmatched."""
    out: Array = np.zeros((len(thresholds), 3))
    a = [index[x] for x in sorted(set(first))]
    b = [index[x] for x in sorted(set(second))]
    if not a or not b:
        out[:, 0], out[:, 2] = len(a), len(b)
        return out
    sub = sim[np.ix_(a, b)]
    for t, threshold in enumerate(thresholds):
        hit = sub >= threshold
        common = int(hit.any(axis=1).sum())
        out[t] = (len(a) - common, common, int((~hit.any(axis=0)).sum()))
    return out


def _deal(videos: list[Video], rng: np.random.Generator) -> None:
    """Each first model's labels dealt out to other keyframes of the collection: what pairs share by chance."""
    keyframes = [k for v in videos for k in v["keyframes"]]
    for a in sorted({a for a, _ in PAIRS}):
        pool = [k["labels"][a] for k in keyframes]
        draws = [rng.permutation(len(pool)) for _ in range(CHANCE_DRAWS)]
        for i, k in enumerate(keyframes):
            k.setdefault("chance", {})[a] = [pool[int(d[i])] for d in draws]


def _totals(
    v: Video,
    sim: Array,
    index: dict[str, int],
    thresholds: tuple[float, ...],
    chance: bool = False,
) -> dict[str, float]:
    """One video's counts; a collection's are their sum, so a resample sums resampled videos."""
    keyframes = v["keyframes"]
    out: dict[str, float] = {"videos": 1.0}
    for key, _, counts in SOURCES:
        found = [k[key] for k in keyframes if k[key] is not None]
        if key == "truth":
            values = [5 if x == "5+" else int(x) for x in found]
        else:
            values = [int(x) if counts else int(bool(x)) for x in found]
        out[f"{key}_n"] = len(values)
        out[f"{key}_present"] = sum(1 for x in values if x > 0)
        out[f"{key}_count"] = sum(values)
    named = v["caption"] is not None and _names_person(v["caption"])
    busy = (
        bool(keyframes) and sum(1 for k in keyframes if k["yolo"]) >= len(keyframes) / 2
    )
    out["caption_n"] = float(v["caption"] is not None)
    out["caption_present"] = float(named)
    out["busy"] = float(busy)
    out["busy_caption"] = float(busy and named)
    for a, b in PAIRS:
        counts = {"": np.zeros((len(thresholds), 3))}
        for k in keyframes:
            counts[""] += _topic_counts(
                k["labels"][a], k["labels"][b], sim, index, thresholds
            )
        if chance:
            zero = np.zeros((len(thresholds), 3))
            counts["_chance"] = (
                sum(
                    (
                        _topic_counts(labels, k["labels"][b], sim, index, thresholds)
                        for k in keyframes
                        for labels in k["chance"][a]
                    ),
                    zero,
                )
                / CHANCE_DRAWS
            )
            counts["_bare"] = sum(
                (
                    _topic_counts(
                        [x for x in k["labels"][a] if x != "person"],
                        [x for x in k["labels"][b] if x != "person"],
                        sim,
                        index,
                        thresholds,
                    )
                    for k in keyframes
                ),
                zero,
            )
        for kind, found in counts.items():
            for t in range(len(thresholds)):
                for j, part in enumerate(PARTS):
                    out[f"{a}-{b}@{t}_{part}{kind}"] = float(found[t, j])
    return out


def _matrix(
    videos: list[Video],
    sim: Array,
    index: dict[str, int],
    thresholds: tuple[float, ...],
    chance: bool = False,
) -> tuple[Array, list[str]]:
    rows = [_totals(v, sim, index, thresholds, chance) for v in videos]
    columns = sorted(rows[0])
    return np.asarray([[r[c] for c in columns] for r in rows], dtype=float), columns


def _sum(matrix: Array, columns: list[str], picks: Array | None = None) -> Totals:
    """Column totals over the videos, or over each resample's videos."""
    sums = (matrix if picks is None else matrix[picks]).sum(axis=-2)
    return {c: sums[..., j] for j, c in enumerate(columns)}


def _ratio(top: Array, bottom: Array) -> Array:
    out: Array = top / np.where(bottom > 0, bottom, np.nan)
    return out


def _presence(t: Totals, key: str) -> Array:
    return _ratio(t[f"{key}_present"], t[f"{key}_n"])


def _share(
    t: Totals, pair: tuple[str, str], at: int, part: str, kind: str = ""
) -> Array:
    """kind: "" as found, "_chance" with the first model's labels dealt at random, "_bare" without "person"."""
    a, b = pair
    cells = [t[f"{a}-{b}@{at}_{p}{kind}"] for p in PARTS]
    return _ratio(t[f"{a}-{b}@{at}_{part}{kind}"], cells[0] + cells[1] + cells[2])


def _own(t: Totals, b: str) -> Array:
    """The tags' own share against b, less the larger of the other two."""
    pair = ("tags", b)
    rest: Array = np.maximum(_share(t, pair, 0, "common"), _share(t, pair, 0, "second"))
    return _share(t, pair, 0, "first") - rest


@dataclass(frozen=True)
class Expectation:
    id: str
    claim: str
    value: Callable[[Totals], Array]
    bounds: tuple[float, float]  # pass when the 95% interval sits inside
    source: str = ""
    support: str = "videos"  # the total that must reach MIN_N


EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        "C1",
        "YOLO finds a person in more keyframes than full-frame YuNet finds a face",
        lambda t: _presence(t, "yolo") - _presence(t, "faces"),
        (0.0, math.inf),
        "backs and wide shots; FRESCO's datasets were filtered to faces",
    ),
    Expectation(
        "C2",
        "Mask2Former finds a person in as many keyframes as YOLO, within 5 points",
        lambda t: _presence(t, "panoptic") - _presence(t, "yolo"),
        (-0.05, 0.05),
        "FRESCO: +3.7 and +0.6 points",
    ),
    Expectation(
        "C3",
        "SegFormer finds a person in more keyframes than YOLO",
        lambda t: _presence(t, "semantic") - _presence(t, "yolo"),
        (0.0, math.inf),
        "FRESCO: 95.6% against 88.3%",
    ),
    Expectation(
        "C4",
        "The tags name a person in more keyframes than YOLO finds one",
        lambda t: _presence(t, "tags") - _presence(t, "yolo"),
        (0.0, math.inf),
        "FRESCO: 96.1% against 88.3%, from RAM++, a trained tagger; ours are zero-shot, not like for like",
    ),
    Expectation(
        "C5",
        "Mask2Former, every segment kept, counts more people per keyframe than YOLO",
        lambda t: _ratio(t["panoptic_count"] - t["yolo_count"], t["panoptic_n"]),
        (0.0, math.inf),
        "FRESCO: 2.14 against 1.85 per image",
    ),
    Expectation(
        "C6",
        "Captions name a person in 80% or more of the videos where YOLO sees one in most keyframes",
        lambda t: _ratio(t["busy_caption"], t["busy"]),
        (0.8, 1.0),
        "FRESCO: 95.4% of images",
        "busy",
    ),
    Expectation(
        "C7",
        "Against YOLO's objects, the tags' own topics are the largest share",
        lambda t: _own(t, "objects"),
        (0.0, math.inf),
        "FRESCO: 60.1%",
    ),
    Expectation(
        "C8",
        "Against SegFormer's classes, the tags' own topics are the largest share",
        lambda t: _own(t, "semantic"),
        (0.0, math.inf),
        "FRESCO: 50.4%",
    ),
    Expectation(
        "C9",
        "Against panoptic things, the tags' own topics are the largest share",
        lambda t: _own(t, "panoptic"),
        (0.0, math.inf),
        "FRESCO: 72.0%",
    ),
    Expectation(
        "C10",
        "SegFormer adds more topics to YOLO's objects than YOLO adds to SegFormer",
        lambda t: (
            _share(t, ("objects", "semantic"), 0, "second")
            - _share(t, ("objects", "semantic"), 0, "first")
        ),
        (0.0, math.inf),
        "FRESCO: 50.5% against 23.2%",
    ),
    Expectation(
        "C11",
        "Panoptic things add 5% or fewer topics to SegFormer's",
        lambda t: _share(t, ("semantic", "panoptic"), 0, "second"),
        (-math.inf, 0.05),
        "FRESCO: 0%",
    ),
    Expectation(
        "C12",
        "YOLO's objects and panoptic things share 60% or more topics at the highest threshold",
        lambda t: _share(t, ("objects", "panoptic"), 2, "common"),
        (0.6, 1.0),
        "same COCO names; FRESCO's mixed vocabularies gave 26.9%",
    ),
)


def _ci(stats: Array) -> list[float]:
    low, high = np.nanpercentile(stats, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def _inside(ci: list[float], low: float, high: float) -> str:
    if low <= ci[0] and ci[1] <= high:
        return "pass"
    return "fail" if ci[1] < low or ci[0] > high else "weak"


def _judge(
    e: Expectation,
    collection: str,
    matrix: Array,
    columns: list[str],
    rng: np.random.Generator,
    boot: int,
) -> dict[str, Any]:
    point = _sum(matrix, columns)
    out: dict[str, Any] = {
        "id": e.id,
        "collection": collection,
        "claim": e.claim,
        "bounds": [None if math.isinf(x) else x for x in e.bounds],
        "source": e.source,
        "support": int(point[e.support]),
    }
    estimate = float(e.value(point))
    if float(point[e.support]) < MIN_N or math.isnan(estimate):
        return {**out, "result": "too few"}
    picks = rng.integers(0, len(matrix), (boot, len(matrix)))
    ci = _ci(np.asarray(e.value(_sum(matrix, columns, picks)), dtype=float))
    return {
        **out,
        "estimate": round(estimate, 4),
        "ci": ci,
        "result": _inside(ci, *e.bounds),
    }


def _people(
    matrix: Array, columns: list[str], rng: np.random.Generator, boot: int
) -> dict[str, Any]:
    """Table 3: share of keyframes with a person, and people per keyframe."""
    point = _sum(matrix, columns)
    again = _sum(matrix, columns, rng.integers(0, len(matrix), (boot, len(matrix))))
    out: dict[str, Any] = {}
    for key, _, counts in (*SOURCES, ("caption", "", False)):
        n = float(point[f"{key}_n"])
        if not n:
            continue
        entry: dict[str, Any] = {
            "n": int(n),
            "with": round(float(_presence(point, key)), 4),
            "with_ci": _ci(_presence(again, key)),
        }
        if counts:
            entry["per_keyframe"] = round(float(point[f"{key}_count"]) / n, 4)
            entry["per_keyframe_ci"] = _ci(
                _ratio(again[f"{key}_count"], again[f"{key}_n"])
            )
        out[key] = entry
    return out


def _bin(value: int) -> str:
    return next(name for name, low, high in BINS if low <= value <= high)


def _groups(videos: list[Video]) -> dict[str, dict[str, float]]:
    """Table 4: share of keyframes per group size, in FRESCO's bins with 0 and 1 apart."""
    keyframes = [k for v in videos for k in v["keyframes"]]
    rows: dict[str, list[int]] = {
        key: [int(k[key]) for k in keyframes] for key in COUNTING
    }
    out: dict[str, dict[str, float]] = {}
    for key, values in rows.items():
        bins = [_bin(x) for x in values]
        out[key] = {name: round(bins.count(name) / len(bins), 4) for name, _, _ in BINS}
    return out


def _ave_groups(videos: list[Video]) -> dict[str, dict[str, float]]:
    """Table 4 against AVE's labels, in AVE's bins, on labelled keyframes only."""
    keyframes = [k for v in videos for k in v["keyframes"] if k["truth"] is not None]
    out: dict[str, dict[str, float]] = {}
    for key in (*COUNTING, "truth"):
        bins = [
            k[key] if key == "truth" else ("5+" if k[key] >= 5 else str(k[key]))
            for k in keyframes
        ]
        out[key] = {name: round(bins.count(name) / len(bins), 4) for name in AVE_BINS}
    return out


def _topics(
    matrix: Array, columns: list[str], rng: np.random.Generator, boot: int, levels: int
) -> dict[str, Any]:
    """Table 5: shares of first only, common and second only, per pair and threshold; with the
    common share by chance, above it, and without "person" where the matrix has them."""
    point = _sum(matrix, columns)
    again = _sum(matrix, columns, rng.integers(0, len(matrix), (boot, len(matrix))))
    out: dict[str, Any] = {}
    for pair in PAIRS:
        cells = []
        for t in range(levels):
            cell = {
                part: round(float(_share(point, pair, t, part)), 4) for part in PARTS
            }
            cell |= {f"{part}_ci": _ci(_share(again, pair, t, part)) for part in PARTS}
            if f"{'-'.join(pair)}@{t}_common_chance" in point:
                above = _share(point, pair, t, "common") - _share(
                    point, pair, t, "common", "_chance"
                )
                cell |= {
                    "chance": round(
                        float(_share(point, pair, t, "common", "_chance")), 4
                    ),
                    "above_chance": round(float(above), 4),
                    "above_chance_ci": _ci(
                        _share(again, pair, t, "common")
                        - _share(again, pair, t, "common", "_chance")
                    ),
                    "without_person": round(
                        float(_share(point, pair, t, "common", "_bare")), 4
                    ),
                }
            cells.append(cell)
        out["-".join(pair)] = cells
    return out


def _misses(videos: list[Video]) -> dict[str, Any]:
    """Keyframes with a face on the whole frame and no YOLO person: the router's silent misses."""
    keyframes = [(v["name"], k) for v in videos for k in v["keyframes"]]
    found = sorted(
        f"{name}/{k['key']}" for name, k in keyframes if k["faces"] and not k["yolo"]
    )
    return {
        "keyframes": len(found),
        "share": round(len(found) / len(keyframes), 4),
        "videos": len({f.split("/")[0] for f in found}),
        "examples": found[:30],
    }


def _planted(
    videos: list[Video],
    sim: Array,
    index: dict[str, int],
    thresholds: tuple[float, ...],
    seed: int,
    boot: int,
) -> list[dict[str, Any]]:
    """Faults the checks must catch, after Adebayo et al. (2018)."""
    rng = np.random.default_rng([seed, 999])
    matrix, columns = _matrix(videos, sim, index, thresholds)
    tags = [k["labels"]["tags"] for v in videos for k in v["keyframes"]]
    order = iter(rng.permutation(len(tags)).tolist())
    shuffled = [
        {
            **v,
            "keyframes": [
                {**k, "labels": {**k["labels"], "tags": tags[next(order)]}}
                for k in v["keyframes"]
            ],
        }
        for v in videos
    ]
    after, _ = _matrix(shuffled, sim, index, thresholds)
    before_t, after_t = _sum(matrix, columns), _sum(after, columns)
    common = {
        b: (
            float(_share(before_t, ("tags", b), 0, "common")),
            float(_share(after_t, ("tags", b), 0, "common")),
        )
        for b in ("objects", "semantic", "panoptic")
    }

    zeroed = matrix.copy()
    for column in ("yolo_present", "yolo_count"):
        zeroed[:, columns.index(column)] = 0.0
    found = {
        e.id: _judge(e, AVE, zeroed, columns, rng, boot)["result"]
        for e in EXPECTATIONS
        if e.id in ("C1", "C2")
    }
    return [
        {
            "fault": "AVE's tags shuffled between keyframes",
            "expected": f"topics shared with each other model fall by {PLANTED_DROP:.0%} or more of their share",
            "found": ", ".join(
                f"{b} -{100 * (x - y):.1f} points ({(x - y) / x:.0%})"
                for b, (x, y) in common.items()
            ),
            "caught": all(x - y >= PLANTED_DROP * x for x, y in common.values()),
        },
        {
            "fault": "AVE's YOLO person counts set to zero",
            "expected": "C1 and C2 fail",
            "found": ", ".join(f"{k} {r}" for k, r in found.items()),
            "caught": all(r == "fail" for r in found.values()),
        },
    ]


def _spearman(pairs: Array) -> float:
    if len(pairs) < 3:
        return math.nan
    a, b = rankdata(pairs[:, 0]), rankdata(pairs[:, 1])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0  # a constant orders nothing
    return float(np.corrcoef(a, b)[0, 1])


def _agreement(
    videos: list[Video], rng: np.random.Generator, boot: int
) -> dict[str, Any]:
    """Table 6: Spearman between two models on the same items."""
    out: dict[str, Any] = {}
    for name, *_ in AGREEMENTS:
        per = [np.asarray(v["pairs"][name], dtype=float).reshape(-1, 2) for v in videos]
        pooled = np.concatenate(per)
        if len(pooled) < MIN_N:
            out[name] = {"n": len(pooled), "result": "too few"}
            continue
        value = _spearman(pooled)
        picks = rng.integers(0, len(per), (boot, len(per)))
        stats = np.asarray(
            [_spearman(np.concatenate([per[i] for i in row])) for row in picks]
        )
        out[name] = {
            "n": len(pooled),
            "spearman": round(value, 4),
            "ci": _ci(stats),
            "result": "pass"
            if value >= AGREE_PASS
            else ("weak" if value >= AGREE_WEAK else "flag"),
        }
    return out


def _kappa(a: Array, b: Array) -> float:
    agree = float(np.mean(a == b))
    chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (agree - chance) / (1 - chance) if chance < 1 else 1.0


def _encoders(
    videos: dict[str, list[Video]],
    vocabulary: list[str],
    sim: Array,
    tables: dict[str, tuple[Array, list[str]]],
    seed: int,
    boot: int,
) -> dict[str, Any]:
    """CLIP at FRESCO's thresholds against the pipeline's MobileCLIP at thresholds that match as many label pairs."""
    index = {label: i for i, label in enumerate(vocabulary)}
    mobile = _embed(vocabulary, mobile=True)
    sims = {"clip": sim, "mobileclip": mobile @ mobile.T}
    probes = {}
    for name in sims:
        found = _embed(["car", *PROBES], mobile=name == "mobileclip")
        probes[name] = {
            p: round(float(found[0] @ found[i + 1]), 4) for i, p in enumerate(PROBES)
        }

    everyone = [v for c in COLLECTIONS for v in videos[c]]
    labels = {
        s: sorted({x for v in everyone for k in v["keyframes"] for x in k["labels"][s]})
        for s in TOPICS
    }
    size = len(vocabulary)
    keys = np.unique(
        np.concatenate(
            [
                np.asarray(
                    [
                        index[x] * size + index[y]
                        for x in labels[a]
                        for y in labels[b]
                        if x != y
                    ],
                    dtype=np.int64,
                )
                for a, b in PAIRS
            ]
        )
    )
    values = {name: s[keys // size, keys % size] for name, s in sims.items()}
    shares = [float(np.mean(values["clip"] >= t)) for t in THRESHOLDS]
    thresholds = {
        "clip": THRESHOLDS,
        "mobileclip": tuple(
            round(float(np.quantile(values["mobileclip"], 1 - q)), 4) for q in shares
        ),
    }

    topics: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, str]] = {}
    for name, found_sim in sims.items():
        topics[name], verdicts[name] = {}, {}
        for j, c in enumerate(COLLECTIONS):
            matrix, columns = (
                tables[c]
                if name == "clip"
                else _matrix(videos[c], found_sim, index, thresholds[name])
            )
            topics[name][c] = _topics(
                matrix,
                columns,
                np.random.default_rng([seed, 200 + j]),
                boot,
                len(THRESHOLDS),
            )
            for i, e in enumerate(EXPECTATIONS):
                if e.id in JUDGED:
                    found = _judge(
                        e, c, matrix, columns, np.random.default_rng([seed, i, j]), boot
                    )
                    verdicts[name][f"{e.id} {c}"] = (
                        f"{found['result']} {found.get('estimate')} {found.get('ci')}"
                    )
    gaps = [
        abs(topics["clip"][c][p][t][part] - topics["mobileclip"][c][p][t][part])
        for c in COLLECTIONS
        for p in topics["clip"][c]
        for t in range(len(THRESHOLDS))
        for part in PARTS
    ]
    same = all(
        verdicts["clip"][k].split()[0] == verdicts["mobileclip"][k].split()[0]
        for k in verdicts["clip"]
    )
    return {
        "rule": f"agree when C7-C12 verdicts match in both collections and Table 5 shares differ by at most "
        f"{MAX_GAP} (largest) and {MEAN_GAP} (mean)",
        "label_pairs": len(keys),
        "probes_against_car": probes,
        "thresholds": [
            {
                "clip": t,
                "mobileclip": m,
                "share_of_pairs": round(q, 5),
                "agreement": round(
                    float(
                        np.mean((values["clip"] >= t) == (values["mobileclip"] >= m))
                    ),
                    5,
                ),
                "kappa": round(
                    _kappa(values["clip"] >= t, values["mobileclip"] >= m), 4
                ),
            }
            for t, m, q in zip(THRESHOLDS, thresholds["mobileclip"], shares)
        ],
        "verdicts": verdicts,
        "verdicts_match": same,
        "largest_gap": round(max(gaps), 4),
        "mean_gap": round(float(np.mean(gaps)), 4),
        "agree": same and max(gaps) <= MAX_GAP and float(np.mean(gaps)) <= MEAN_GAP,
    }


# Pages, in the identikit page's ink and surfaces

PAGE_SIZE = (22.4, 13.5)  # inches at 100 dpi
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
PAGE, HAIRLINE = "#f9f9f7", "#e1e0d9"
STATUS = {
    "pass": "#0ca30c",
    "weak": "#fab219",
    "flag": "#ec835a",
    "fail": "#d03b3b",
    "too few": "#c3c2b7",
}


def _pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None or math.isnan(x) else f"{100 * x:.{digits}f}%"


def _cell(x: float, ci: list[float]) -> str:
    return f"{100 * x:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _page(title: str, subtitle: str) -> Figure:
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(0.018, 0.988, title, fontsize=26, weight="bold", color=INK, va="top")
    fig.text(0.018, 0.95, subtitle, fontsize=12, color=SECONDARY, va="top")
    return fig


def _table(
    fig: Figure,
    y: float,
    header: list[str],
    rows: list[list[str]],
    at: list[float],
    size: float = 10.0,
) -> float:
    """Rows of text at the given x positions; returns the y below the table."""
    step = size * 1.7 / (PAGE_SIZE[1] * 72)
    for text, x in zip(header, at):
        fig.text(x, y, text, fontsize=size - 0.5, color=MUTED, va="top")
    y -= step
    for row in rows:
        for i, (text, x) in enumerate(zip(row, at)):
            fig.text(
                x, y, text, fontsize=size, color=INK if i == 0 else SECONDARY, va="top"
            )
        y -= step
    return y - step / 2


def _heading(fig: Figure, y: float, text: str) -> float:
    fig.text(0.018, y, text, fontsize=14, weight="bold", color=INK, va="top")
    return y - 0.03


def _people_page(path: Path, report: dict[str, Any]) -> None:
    fig = _page(
        "Consistency: people across models",
        "After FRESCO's Table 3. Per keyframe: the share with a person, and people per keyframe; "
        "95% intervals from 1,000 resamples of whole videos. Captions are one per video.",
    )
    people = report["people"]
    at = [0.018, 0.27, 0.39, 0.47, 0.59, 0.67, 0.76, 0.83, 0.91]
    header = [
        "source",
        "AVE: with a person",
        "per keyframe",
        "reels: with a person",
        "per keyframe",
        "FRESCO OpenImages",
        "per image",
        "FRESCO FFHQ",
        "per image",
    ]
    rows: list[list[str]] = []
    for key, name, _ in (
        *SOURCES,
        ("caption", "person words in the caption, Qwen3-VL (11), per video", False),
    ):
        row = [name]
        for c in COLLECTIONS:
            found = people[c].get(key)
            row += (
                [
                    _cell(found["with"], found["with_ci"]),
                    f"{found['per_keyframe']:.2f}" if "per_keyframe" in found else "-",
                ]
                if found
                else ["-", "-"]
            )
        for with_person, per_image in FRESCO_PEOPLE.get(
            key, ((None, None), (None, None))
        ):
            row += [_pct(with_person), "-" if per_image is None else f"{per_image:.2f}"]
        rows.append(row)
    y = _table(fig, 0.9, header, rows, at, size=10.5)

    notes = [
        (
            "In-box faces exist only inside YOLO's boxes, and 06 and 08 run only in shots 03 found a person in, "
            "so those rows cannot disagree with YOLO about empty shots."
        ),
        (
            f"Mask2Former is run again here as 05 runs it, every segment kept: 05 keeps 10 a keyframe, and cut "
            f"the list in {_pct(report['collections'][AVE]['cut_share'])} of AVE keyframes and "
            f"{_pct(report['collections'][REELS]['cut_share'])} of reels keyframes. Where it did not, the rerun "
            f"finds the same persons in "
            f"{_pct(report['collections'][AVE]['mask2former_rerun']['persons_match_uncut'])} and "
            f"{_pct(report['collections'][REELS]['mask2former_rerun']['persons_match_uncut'])}."
        ),
        (
            "The tags are MobileCLIP's top 15 per keyframe over RAM's vocabulary, zero-shot with no score "
            "floor, where FRESCO's come from RAM++, a trained tagger."
        ),
        (
            "A person word is one of 259 WordNet person nouns reviewed by hand. AVE's label 5 means 5 or "
            "more and counts as 5."
        ),
    ]
    for c in COLLECTIONS:
        if report["collections"][c]["without_pass_zero"]:
            notes.append(
                f"{report['collections'][c]['without_pass_zero']} of the {NAMES[c]} predate 08's pass zero, "
                "so a person boxed twice there still counts twice."
            )
    for c in COLLECTIONS:
        m = report["router_misses"][c]
        notes.append(
            f"Router's silent misses, {NAMES[c]}: {m['keyframes']:,} keyframes ({_pct(m['share'], 2)}) "
            f"in {m['videos']} videos have a face on the whole frame and no YOLO person (open item 12)."
        )
    for note in notes:
        fig.text(0.018, y, note, fontsize=10, color=MUTED, va="top")
        y -= 0.024
    fig.savefig(path, facecolor=PAGE)


def _groups_page(path: Path, report: dict[str, Any]) -> None:
    fig = _page(
        "Consistency: group sizes across models",
        "After FRESCO's Table 4: the share of keyframes per number of people. FRESCO merges 0 and 1; "
        "here they are apart, and 0/1 is their sum.",
    )
    names = {key: name for key, name, _ in SOURCES}
    bins = ["0", "1", "0/1", "2", "3-6", "7-12", "13-30", "31+"]
    at = [0.018] + [0.3 + 0.075 * i for i in range(len(bins))]
    y = 0.9
    for c in COLLECTIONS:
        y = _heading(fig, y, NAMES[c])
        rows = []
        for key, shares in report["groups"][c].items():
            merged = dict(shares) | {"0/1": shares["0"] + shares["1"]}
            rows.append([names[key]] + [_pct(merged[b]) for b in bins])
        y = _table(fig, y, ["source", *bins], rows, at)
    y = _heading(fig, y, "FRESCO")
    rows = [
        [f"{dataset}: {FRESCO_NAMES[key]}"] + ["", ""] + [f"{x:.1f}%" for x in shares]
        for dataset, table in FRESCO_GROUPS.items()
        for key, shares in table.items()
    ]
    y = _table(fig, y, ["source", *bins], rows, at)

    y = _heading(
        fig, y, "AVE film scenes, against AVE's num-people label (labelled keyframes)"
    )
    rows = [
        [names[key]] + [_pct(shares[b]) for b in AVE_BINS]
        for key, shares in report["ave_groups"].items()
    ]
    _table(fig, y, ["source", *AVE_BINS], rows, at)
    fig.savefig(path, facecolor=PAGE)


def _topics_page(path: Path, report: dict[str, Any]) -> None:
    thresholds = report["thresholds"]
    fig = _page(
        "Consistency: topics across models",
        "After FRESCO's Table 5. Per keyframe, a label of the first model is shared when a label of the second "
        f"reaches the threshold ({report['encoder']}).",
    )
    at = [0.018] + [0.28 + 0.08 * i for i in range(9)]
    header = ["pair (first, second)"] + [
        f"{part} @ {t}"
        for t in thresholds
        for part in ("first only", "common", "second only")
    ]
    y = 0.9
    for c in COLLECTIONS:
        y = _heading(fig, y, NAMES[c])
        rows = [
            [f"{TOPIC_NAMES[a]}, {TOPIC_NAMES[b]}"]
            + [
                _pct(cell[part])
                for cell in report["topics"][c][f"{a}-{b}"]
                for part in PARTS
            ]
            for a, b in PAIRS
        ]
        y = _table(fig, y, header, rows, at)
    y = _heading(fig, y, "FRESCO, OpenImages, CLIP ViT-L/14 at 0.80, 0.85 and 0.90")
    rows = [
        [f"{FRESCO_NAMES[a]}, {FRESCO_NAMES[b]}"]
        + [f"{x:.1f}%" for cell in FRESCO_TOPICS[(a, b)] for x in cell]
        for a, b in PAIRS
    ]
    y = _table(fig, y, header, rows, at)

    y = _heading(
        fig,
        y,
        f"Common at {thresholds[0]} against chance: the first model's labels dealt to other keyframes of the "
        f"collection, {CHANCE_DRAWS} times",
    )
    header = ["pair (first, second)"] + [
        f"{SHORT[c]}: {column}" if i == 0 else column
        for c in COLLECTIONS
        for i, column in enumerate(
            ("common", "by chance", "above chance, points [CI]", 'without "person"')
        )
    ]
    rows = []
    for a, b in PAIRS:
        row = [f"{TOPIC_NAMES[a]}, {TOPIC_NAMES[b]}"]
        for c in COLLECTIONS:
            cell = report["topics"][c][f"{a}-{b}"][0]
            row += [
                _pct(cell["common"]),
                _pct(cell["chance"]),
                _cell(cell["above_chance"], cell["above_chance_ci"]),
                _pct(cell["without_person"]),
            ]
        rows.append(row)
    _table(
        fig, y, header, rows, [0.018, 0.25, 0.32, 0.39, 0.51, 0.61, 0.68, 0.75, 0.87]
    )
    fig.savefig(path, facecolor=PAGE)


def _agreement_page(path: Path, report: dict[str, Any]) -> None:
    fig = _page(
        "Consistency: two models measuring the same thing",
        "Spearman's rank correlation over the items, 95% interval from resampling whole videos; "
        f"pass from {AGREE_PASS}, weak from {AGREE_WEAK}, else flag.",
    )
    at = [0.018, 0.13, 0.225, 0.4, 0.56, 0.6, 0.725, 0.785, 0.825, 0.95]
    header = [
        "check",
        "item",
        "model a",
        "model b",
        "AVE: n",
        "Spearman [95% CI]",
        "result",
        "reels: n",
        "Spearman [95% CI]",
        "result",
    ]
    rows = []
    for name, item, a, b in AGREEMENTS:
        row = [name, item, a, b]
        for c in COLLECTIONS:
            found = report["agreement"][c][name]
            row += [
                f"{found['n']:,}",
                f"{found['spearman']:.3f} [{found['ci'][0]:.3f}, {found['ci'][1]:.3f}]"
                if "spearman" in found
                else "-",
                found["result"],
            ]
        rows.append(row)
    y = _table(fig, 0.9, header, rows, at, size=10.5)
    notes = [
        (
            "Smile: happiness and a smile are related, not the same; 07 keeps a sample's 10 "
            "strongest blendshapes, so a smile outside them counts as 0."
        ),
        "Looking at the camera: persons with 3 or more address samples whose 10 leading attributes include it.",
        "Outdoor: sky is a one-sided cue; interiors show none, and many exteriors show none either.",
        "Persons per keyframe: Mask2Former run again with every segment kept, where 05 keeps 10.",
        (
            "People per segment: followed counts everyone over the segment, visible the most in one keyframe; "
            "segments where 03 saw nobody are left out, as 06-08 run only where it saw someone."
        ),
    ]
    for note in notes:
        fig.text(0.018, y, note, fontsize=10, color=MUTED, va="top")
        y -= 0.024
    fig.savefig(
        path, facecolor=PAGE, bbox_inches="tight", pad_inches=0.3
    )  # one short table


def _summary_page(path: Path, report: dict[str, Any]) -> None:
    results = report["expectations"]
    fig = _page(
        "Consistency: expectations",
        "Written before the first run, mostly FRESCO's findings. 95% intervals from resampling whole videos: "
        "inside the bounds passes.",
    )
    for i, result in enumerate(("pass", "weak", "fail", "too few")):
        left = 0.62 + i * 0.09
        count = sum(1 for r in results if r["result"] == result)
        fig.text(
            left, 0.985, str(count), fontsize=28, weight="semibold", color=INK, va="top"
        )
        fig.text(
            left, 0.94, f"■ {result}", fontsize=10.5, color=STATUS[result], va="top"
        )

    step = min(0.027, 0.7 / max(1, len(results)))
    y = 0.9
    for header, x in (("check", 0.018), ("estimate [95% CI]", 0.62), ("result", 0.92)):
        fig.text(x, y, header, fontsize=9.5, color=MUTED, va="top")
    for r in results:
        y -= step
        fig.text(
            0.018,
            y,
            f"{r['id']} {SHORT[r['collection']]}",
            fontsize=10.5,
            weight="bold",
            color=INK,
            va="top",
        )
        fig.text(
            0.075,
            y,
            textwrap.shorten(r["claim"], 100),
            fontsize=10.5,
            color=INK,
            va="top",
        )
        estimate = (
            f"{r['estimate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
            if "estimate" in r
            else f"support {r['support']}"
        )
        fig.text(0.62, y, estimate, fontsize=10, color=SECONDARY, va="top")
        fig.text(
            0.92,
            y,
            f"■ {r['result']}",
            fontsize=10.5,
            color=STATUS[r["result"]],
            va="top",
        )
    y -= step * 1.6
    for fault in report["planted"]:
        mark = "caught" if fault["caught"] else "MISSED"
        line = f"Planted fault, {fault['fault']}: expected {fault['expected']}; found {fault['found']} - {mark}"
        fig.text(0.018, y, line, fontsize=10, color=SECONDARY, va="top")
        y -= step
    fig.savefig(path, facecolor=PAGE)


def main() -> None:
    p = argparse.ArgumentParser(
        description="People and topics across models, after FRESCO's Tables 3-5"
    )
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument(
        "--ave",
        default="ave_processed",
        help="the AVE scenes' folder under --processed",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "raw" / "ave" / "annotations.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "consistency" / "consistency.json",
        help="the report; its five pages and the model caches are written beside it",
    )
    p.add_argument(
        "--boot", type=int, default=1000, help="video resamples behind each interval"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--encoders",
        action="store_true",
        help="also compare CLIP with the pipeline's MobileCLIP, into <out>_encoders.json",
    )
    args = p.parse_args()

    ave = (
        json.loads(args.labels.read_text(encoding="utf-8"))
        if args.labels.exists()
        else {}
    )
    folders = {AVE: args.ave, REELS: REELS}
    videos = _load(args.processed, folders, ave)
    if any(len(videos[c]) < MIN_N for c in COLLECTIONS):
        sys.exit(
            f"needs {MIN_N} processed videos in each of {', '.join(folders.values())} under {args.processed}"
        )
    everyone = [v for c in COLLECTIONS for v in videos[c]]
    _add_faces(everyone, args.out.with_name("faces_yunet.json"))
    _add_things(everyone, args.out.with_name("things_mask2former.json"))
    vocabulary = _vocabulary(everyone)
    index = {label: i for i, label in enumerate(vocabulary)}
    embedded = _embed(vocabulary)
    sim: Array = embedded @ embedded.T

    for j, c in enumerate(COLLECTIONS):
        _deal(videos[c], np.random.default_rng([args.seed, 400 + j]))
    tables = {
        c: _matrix(videos[c], sim, index, THRESHOLDS, chance=True) for c in COLLECTIONS
    }
    results = [
        _judge(e, c, *tables[c], np.random.default_rng([args.seed, i, j]), args.boot)
        for i, e in enumerate(EXPECTATIONS)
        for j, c in enumerate(COLLECTIONS)
    ]
    report: dict[str, Any] = {
        "collections": {
            c: {
                "folder": folders[c],
                "videos": len(videos[c]),
                "without_pass_zero": sum(1 for v in videos[c] if not v["pass_zero"]),
                "keyframes": sum(len(v["keyframes"]) for v in videos[c]),
                "cut_share": round(
                    sum(k["cut"] for v in videos[c] for k in v["keyframes"])
                    / sum(len(v["keyframes"]) for v in videos[c]),
                    4,
                ),
                "captions": sum(1 for v in videos[c] if v["caption"] is not None),
                "mask2former_rerun": _rerun_check(videos[c]),
            }
            for c in COLLECTIONS
        },
        "encoder": "CLIP ViT-L/14 text, OpenAI's weights, FRESCO's thresholds 0.80, 0.85 and 0.90",
        "thresholds": list(THRESHOLDS),
        "vocabulary": len(vocabulary),
        "person_words": len(PERSON_WORDS),
        "resamples": args.boot,
        "seed": args.seed,
        "expectations": results,
        "planted": _planted(videos[AVE], sim, index, THRESHOLDS, args.seed, args.boot),
        "people": {
            c: _people(
                *tables[c], np.random.default_rng([args.seed, 100 + j]), args.boot
            )
            for j, c in enumerate(COLLECTIONS)
        },
        "groups": {c: _groups(videos[c]) for c in COLLECTIONS},
        "ave_groups": _ave_groups(videos[AVE]),
        "topics": {
            c: _topics(
                *tables[c],
                np.random.default_rng([args.seed, 200 + j]),
                args.boot,
                len(THRESHOLDS),
            )
            for j, c in enumerate(COLLECTIONS)
        },
        "router_misses": {c: _misses(videos[c]) for c in COLLECTIONS},
        "agreement": {
            c: _agreement(
                videos[c], np.random.default_rng([args.seed, 300 + j]), args.boot
            )
            for j, c in enumerate(COLLECTIONS)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pages = [
        args.out.with_suffix(".png"),
        args.out.with_name(f"{args.out.stem}_people.png"),
        args.out.with_name(f"{args.out.stem}_groups.png"),
        args.out.with_name(f"{args.out.stem}_topics.png"),
        args.out.with_name(f"{args.out.stem}_agreement.png"),
    ]
    _summary_page(pages[0], report)
    _people_page(pages[1], report)
    _groups_page(pages[2], report)
    _topics_page(pages[3], report)
    _agreement_page(pages[4], report)
    if args.encoders:
        encoders = _encoders(videos, vocabulary, sim, tables, args.seed, args.boot)
        args.out.with_name(f"{args.out.stem}_encoders.json").write_text(
            json.dumps(encoders, indent=2), encoding="utf-8"
        )
        print(
            f"encoders: agree {encoders['agree']}, verdicts match {encoders['verdicts_match']}, "
            f"largest gap {encoders['largest_gap']}, mean gap {encoders['mean_gap']}"
        )

    for c, found in report["collections"].items():
        print(f"{c}: {found}")
    for r in results:
        estimate = (
            f"{r['estimate']:.3f} {r['ci']}"
            if "estimate" in r
            else f"support {r['support']}"
        )
        print(
            f"{r['result']:8} {r['id']:4} {SHORT[r['collection']]:6} {r['claim']:90} {estimate}"
        )
    for c, checks in report["agreement"].items():
        for name, found in checks.items():
            print(
                f"agreement, {c}: {name:22} {found['result']:6} {found.get('spearman')} {found.get('ci')} n {found['n']}"
            )
    for fault in report["planted"]:
        print(
            f"planted: {fault['fault']}: {fault['found']} ({'caught' if fault['caught'] else 'MISSED'})"
        )
    for c, m in report["router_misses"].items():
        print(
            f"router misses, {c}: {m['keyframes']} keyframes ({m['share']:.2%}) in {m['videos']} videos"
        )
    print(f"-> {args.out}, pages {', '.join(p.name for p in pages)}")


if __name__ == "__main__":
    main()
