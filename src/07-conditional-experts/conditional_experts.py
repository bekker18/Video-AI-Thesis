"""07-conditional-experts: the gated figurative and enunciative experts.

Runs only where 04-router fired the matching expert, and consumes what 06 indexed rather
than the video wherever possible: attribute crops and text regions are already on disk.

Two sampling rates, because the consumers differ. Cheap, fast-moving experts (head pose,
blendshapes, emotion, pose) run on a stride across a track's observed rows and produce the
series 08 takes variance over. Expensive, slow-moving ones (identity embedding, age, gender,
attributes) run on the handful of ranked crops 06 already selected. Nothing runs on an
interpolated row - those are position estimates, not observations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np

import download_models
from config import Config

STAGE = "07-conditional-experts"
SEGMENTER = "02-segmentation"  # MobileCLIP cache is reused from there

FACE_MODEL = "face_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"
EMOTION_MODEL = "emotion_enet_b0_va.onnx"
ARCFACE_MODEL = "w600k_r50.onnx"
GENDERAGE_MODEL = "genderage.onnx"
OCR_MODEL = "text_crnn_en.onnx"
CLIP_MODEL = "hf-hub:apple/MobileCLIP-S2-OpenCLIP"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

EMOTIONS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
OCR_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

# insightface's 5-point template for a 112x112 aligned face.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

FACE_ATTRIBUTES = (
    "a person wearing eyeglasses",
    "a person wearing sunglasses",
    "a person with a beard",
    "a clean-shaven person",
    "a person with a moustache",
    "a person wearing a hat",
    "a person wearing a cap",
    "a person wearing a headscarf",
    "a person with long hair",
    "a person with short hair",
    "a person with curly hair",
    "a person with straight hair",
    "a bald person",
    "a person with blond hair",
    "a person with dark hair",
    "a person with grey hair",
    "a person wearing makeup",
    "a person smiling",
    "a person wearing earrings",
    "a person wearing a face mask",
    "a person looking at the camera",
    "a person looking away",
    "a person in profile view",
    "a person wearing headphones",
)


def _device(name: str) -> str:
    import torch

    if name == "cpu":
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no GPU is visible")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read(out_root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = out_root / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _crop(image: Array, box: list[int], pad: float = 0.0) -> Array:
    x1, y1, x2, y2 = box
    if pad:
        margin = round(pad * max(x2 - x1, y2 - y1))
        x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    out: Array = image[y1:y2, x1:x2]
    return out


def _series_rows(track: dict[str, Any], stride: int) -> list[dict[str, Any]]:
    """Every Nth observed row. Interpolated rows carry no observation to measure."""
    observed = [r for r in track["frames"] if not r["interpolated"]]
    return observed[:: max(1, stride)]


def _landmarkers(model_dir: Path, want_face: bool, want_pose: bool) -> tuple[Any, Any]:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    face = pose = None
    if want_face:
        face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_dir / FACE_MODEL)
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
    if want_pose:
        pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_dir / POSE_MODEL)
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
            )
        )
    return face, pose


def _mp_image(bgr: Array) -> Any:
    import mediapipe as mp

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
    )


def _euler(matrix: Array) -> dict[str, float]:
    """Head pose from the rotation block of the facial transformation matrix."""
    r = matrix[:3, :3]
    sy = math.sqrt(float(r[0, 0]) ** 2 + float(r[1, 0]) ** 2)
    if sy < 1e-6:
        pitch, yaw, roll = (
            math.atan2(-float(r[1, 2]), float(r[1, 1])),
            math.atan2(-float(r[2, 0]), sy),
            0.0,
        )
    else:
        pitch = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(-float(r[2, 0]), sy)
        roll = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return {
        "yaw": round(math.degrees(yaw), 2),
        "pitch": round(math.degrees(pitch), 2),
        "roll": round(math.degrees(roll), 2),
    }


def _face_pass(landmarker: Any, crop: Array, top: int) -> dict[str, Any] | None:
    result = landmarker.detect(_mp_image(crop))
    if not result.face_landmarks:
        return None

    out: dict[str, Any] = {}
    if result.facial_transformation_matrixes:
        out["head_pose"] = _euler(np.asarray(result.facial_transformation_matrixes[0]))
    if result.face_blendshapes:
        ranked = sorted(result.face_blendshapes[0], key=lambda c: -float(c.score))
        out["blendshapes"] = {
            str(c.category_name): round(float(c.score), 4) for c in ranked[:top]
        }
    out["landmark_count"] = len(result.face_landmarks[0])
    return out


def _five_points(landmarks: list[Any], size: tuple[int, int]) -> Array:
    """Eye, nose and mouth points for ArcFace alignment, ordered by image x so the
    template's left/right assignment holds regardless of landmark index semantics."""
    width, height = size
    eyes = (468, 473) if len(landmarks) >= 478 else (33, 263)

    def pick(i: int) -> tuple[float, float]:
        return landmarks[i].x * width, landmarks[i].y * height

    eye = sorted([pick(eyes[0]), pick(eyes[1])])
    mouth = sorted([pick(61), pick(291)])
    return np.array([eye[0], eye[1], pick(1), mouth[0], mouth[1]], dtype=np.float32)


def _pose_pass(landmarker: Any, crop: Array) -> dict[str, Any] | None:
    result = landmarker.detect(_mp_image(crop))
    if not result.pose_landmarks:
        return None
    points = [
        [round(p.x, 4), round(p.y, 4), round(float(getattr(p, "visibility", 0.0)), 3)]
        for p in result.pose_landmarks[0]
    ]
    world: list[list[float]] = []
    if result.pose_world_landmarks:
        world = [
            [round(p.x, 4), round(p.y, 4), round(p.z, 4)]
            for p in result.pose_world_landmarks[0]
        ]
    return {"keypoints": points, "world": world}


def _session(path: Path) -> Any:
    import onnxruntime

    # CPU only: these models are small, and onnxruntime's CUDA build would have to agree
    # with torch's on cuDNN, which is the conflict 03 avoided by not taking paddlepaddle.
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _emotion_pass(session: Any, crop: Array) -> dict[str, Any] | None:
    if not crop.size:
        return None
    rgb = (
        cv2.cvtColor(
            cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        / 255.0
    )
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    tensor = rgb.transpose(2, 0, 1)[None]
    raw = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])[0]

    logits = raw[: len(EMOTIONS)]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    out: dict[str, Any] = {
        "label": EMOTIONS[int(np.argmax(probs))],
        "scores": {name: round(float(p), 4) for name, p in zip(EMOTIONS, probs)},
    }
    if raw.shape[0] >= len(EMOTIONS) + 2:
        out["valence"] = round(float(raw[len(EMOTIONS)]), 4)
        out["arousal"] = round(float(raw[len(EMOTIONS) + 1]), 4)
    return out


def _embed_face(session: Any, aligned: Array) -> Array:
    tensor = (
        cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5
    ) / 127.5
    out = np.asarray(
        session.run(
            None, {session.get_inputs()[0].name: tensor.transpose(2, 0, 1)[None]}
        )[0]
    )[0]
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


def _gender_age(session: Any, aligned: Array) -> dict[str, Any]:
    resized = cv2.resize(aligned, (96, 96), interpolation=cv2.INTER_LINEAR)
    tensor = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5
    ) / 127.5
    raw = np.asarray(
        session.run(
            None, {session.get_inputs()[0].name: tensor.transpose(2, 0, 1)[None]}
        )[0]
    )[0]
    # The two gender outputs are unnormalised, so a raw max is not a confidence.
    exp = np.exp(raw[:2] - raw[:2].max())
    probs = exp / exp.sum()
    return {
        "gender": "male" if int(np.argmax(raw[:2])) == 1 else "female",
        "gender_score": round(float(probs.max()), 4),
        "age": round(float(raw[2]) * 100),
    }


def _recogniser(model_dir: Path) -> Any:
    """The CRNN's first conv takes one channel, so crops are fed as greyscale."""
    model = cv2.dnn.TextRecognitionModel(str(model_dir / OCR_MODEL))
    model.setDecodeType("CTC-greedy")
    model.setVocabulary(list(OCR_CHARSET))
    model.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5), False)
    return model


def _read_text(model: Any, crop: Array) -> str:
    """CRNN reads roughly ten characters at 100x32, but 03's detector returns whole caption
    lines running to 13:1, which squash to nothing. The crop is normalised to the model's
    height and read in chunks; boundaries cut words, so parts are joined with a space."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height = 32
    width = max(8, round(grey.shape[1] * height / grey.shape[0]))
    resized = cv2.resize(grey, (width, height), interpolation=cv2.INTER_LINEAR)

    parts: list[str] = []
    chunks = max(1, round(width / 100))
    for i in range(chunks):
        segment = resized[
            :, round(i * width / chunks) : round((i + 1) * width / chunks)
        ]
        if segment.shape[1] < 8:
            continue
        reading = str(model.recognize(segment)).strip()
        if reading:
            parts.append(reading)
    return " ".join(parts)


def _clip_attributes(
    crops: list[Array], device: str, cache: Path, top: int
) -> list[list[dict[str, Any]]]:
    """Zero-shot over a face-attribute vocabulary, the same trick 05 uses for tags."""
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, cache_dir=str(cache)
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(device).eval()

    crop_size = 256
    for step in getattr(preprocess, "transforms", []):
        size = getattr(step, "size", None)
        if isinstance(size, (tuple, list)):
            crop_size = int(size[0])

    with torch.no_grad():
        tokens = tokenizer(list(FACE_ATTRIBUTES)).to(device)
        text = model.encode_text(tokens).float()
        text /= text.norm(dim=-1, keepdim=True)

        batch = np.stack(
            [
                cv2.cvtColor(
                    cv2.resize(c, (crop_size, crop_size), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                for c in crops
            ]
        )
        tensor = (
            torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float().div_(255)
        )
        image = model.encode_image(tensor).float()
        image /= image.norm(dim=-1, keepdim=True)
        scores = (image @ text.T).cpu().numpy()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    out: list[list[dict[str, Any]]] = []
    for row in scores:
        order = np.argsort(-row)[:top]
        out.append(
            [
                {"label": FACE_ATTRIBUTES[int(i)], "score": round(float(row[i]), 4)}
                for i in order
            ]
        )
    return out


def _frames(video: Path, wanted: set[int]) -> Iterator[tuple[int, Array]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    last = max(wanted)
    index = 0
    while index <= last:
        if not capture.grab():
            break
        if index in wanted:
            ok, frame = capture.retrieve()
            if not ok:
                break
            yield index, frame
        index += 1
    capture.release()


def run(cfg: Config) -> dict[str, Any]:
    device = _device(cfg.device)
    segmentation = _read(cfg.out_root, "segmentation.json", "02-segmentation")
    routing = _read(cfg.out_root, "routing.json", "04-router")
    tracks_meta = _read(cfg.out_root, "tracks.json", "06-detection-tracking")
    profile = _read(cfg.out_root, "profile.json", "03-profiler")

    download_models.ensure(STAGE)
    model_dir = cfg.model_dir

    gates = {int(s["index"]): set(s["experts"]) for s in routing["shots"]}
    want_face = any("face" in e for e in gates.values())
    want_pose = any("body" in e for e in gates.values())
    want_ocr = any("ocr" in e for e in gates.values())

    # Which frames the series needs, and which track row each one belongs to.
    schedule: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    for shot in tracks_meta["shots"]:
        index = int(shot["index"])
        if not shot["tracked"] or not (gates.get(index, set()) & {"face", "body"}):
            continue
        for track in shot["tracks"]:
            for row in _series_rows(track, cfg.series_stride):
                schedule.setdefault(int(row["frame"]), []).append(
                    (index, int(track["track_id"]), row)
                )

    face_landmarker, pose_landmarker = _landmarkers(model_dir, want_face, want_pose)
    emotion = _session(model_dir / EMOTION_MODEL) if want_face else None

    series: dict[tuple[int, int], list[dict[str, Any]]] = {}
    if schedule:
        for index, image in _frames(cfg.video, set(schedule)):
            for shot, track_id, row in schedule[index]:
                experts = gates.get(shot, set())
                sample: dict[str, Any] = {"frame": index}

                if "face" in experts and row["face"] and face_landmarker is not None:
                    crop = _crop(image, row["face"], cfg.crop_pad)
                    found = (
                        _face_pass(face_landmarker, crop, cfg.blendshapes)
                        if crop.size
                        else None
                    )
                    if found:
                        sample.update(found)
                        # The emotion model has no "not a face" answer - it classifies whatever
                        # it is handed - so it runs only where the mesh confirms a face.
                        if emotion is not None:
                            feeling = _emotion_pass(emotion, crop)
                            if feeling:
                                sample["emotion"] = feeling

                if "body" in experts and pose_landmarker is not None:
                    crop = _crop(image, row["body"])
                    if crop.size:
                        found = _pose_pass(pose_landmarker, crop)
                        if found:
                            sample["pose"] = found

                if len(sample) > 1:
                    series.setdefault((shot, track_id), []).append(sample)

    # Attributes read 06's ranked crops straight off disk - no decoding at all.
    attribute_jobs: list[tuple[int, int, Path]] = []
    for shot in tracks_meta["shots"]:
        index = int(shot["index"])
        if not shot["tracked"] or "face" not in gates.get(index, set()):
            continue
        for track in shot["tracks"]:
            for crop in track["crops"]["face"]:
                if crop["path"]:
                    attribute_jobs.append(
                        (index, int(track["track_id"]), cfg.out_root / crop["path"])
                    )

    embeddings: list[Array] = []
    per_track: dict[tuple[int, int], dict[str, Any]] = {}
    if attribute_jobs and face_landmarker is not None:
        arcface = _session(model_dir / ARCFACE_MODEL)
        genderage = _session(model_dir / GENDERAGE_MODEL)

        clip_crops: list[Array] = []
        clip_keys: list[tuple[int, int]] = []
        for shot, track_id, path in attribute_jobs:
            image = cv2.imread(str(path))
            if image is None:
                continue
            result = face_landmarker.detect(_mp_image(image))
            if not result.face_landmarks:
                continue
            source = _five_points(
                result.face_landmarks[0], (image.shape[1], image.shape[0])
            )
            matrix, _ = cv2.estimateAffinePartial2D(
                source, ARCFACE_TEMPLATE, method=cv2.LMEDS
            )
            if matrix is None:
                continue
            aligned = cv2.warpAffine(
                image, matrix, (112, 112), borderValue=(0.0, 0.0, 0.0)
            )

            entry = per_track.setdefault(
                (shot, track_id), {"embedding_rows": [], "gender_age": [], "clip": []}
            )
            entry["embedding_rows"].append(len(embeddings))
            embeddings.append(_embed_face(arcface, aligned))
            entry["gender_age"].append(_gender_age(genderage, aligned))
            # CLIP reads the unaligned crop; the 112x112 warp is for ArcFace only.
            clip_crops.append(image)
            clip_keys.append((shot, track_id))

        if clip_crops:
            found = _clip_attributes(
                clip_crops, device, download_models.stage_dir(SEGMENTER), cfg.attributes
            )
            for key, labels in zip(clip_keys, found):
                per_track[key]["clip"].append(labels)

    text_by_shot: dict[int, list[dict[str, Any]]] = {}
    if want_ocr:
        recogniser = _recogniser(model_dir)
        for shot in profile["shots"]:
            index = int(shot["index"])
            if "ocr" not in gates.get(index, set()):
                continue
            for keyframe in shot["keyframes"]:
                regions = keyframe.get("text", [])
                if not regions:
                    continue
                path = (
                    cfg.out_root
                    / "keyframes"
                    / f"{index:04d}_{int(keyframe['frame']):06d}.jpg"
                )
                image = cv2.imread(str(path))
                if image is None:
                    continue
                for region in regions:
                    crop = _crop(image, region["box"])
                    if not crop.size:
                        continue
                    reading = _read_text(recogniser, crop)
                    if reading:
                        text_by_shot.setdefault(index, []).append(
                            {
                                "frame": int(keyframe["frame"]),
                                "box": region["box"],
                                "text": reading,
                                "detection_score": region["score"],
                            }
                        )

    shots: list[dict[str, Any]] = []
    for shot in tracks_meta["shots"]:
        index = int(shot["index"])
        experts = sorted(gates.get(index, set()))
        record: dict[str, Any] = {
            "index": index,
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
            "experts": [
                e for e in experts if e in {"face", "body", "ocr", "object_masks"}
            ],
            "tracks": [],
            "text": text_by_shot.get(index, []),
        }
        if shot["tracked"]:
            for track in shot["tracks"]:
                key = (index, int(track["track_id"]))
                samples = series.get(key, [])
                attrs = per_track.get(key)
                if not samples and not attrs:
                    continue
                record["tracks"].append(
                    {
                        "track_id": track["track_id"],
                        "tracker_id": track["tracker_id"],
                        "series": samples,
                        "attributes": attrs or {},
                    }
                )
        shots.append(record)

    for landmarker in (face_landmarker, pose_landmarker):
        if landmarker is not None:
            landmarker.close()

    if embeddings:
        np.save(cfg.out_root / "face_embeddings.npy", np.stack(embeddings))

    meta = {
        "video": segmentation["video"],
        "shot_count": len(shots),
        "device": device,
        "series_stride": cfg.series_stride,
        "series_samples": sum(len(v) for v in series.values()),
        "tracks_with_series": len(series),
        "face_embeddings": len(embeddings),
        "text_regions": sum(len(v) for v in text_by_shot.values()),
        "models": {
            "face_mesh_pose_blendshapes": FACE_MODEL,
            "body_pose": POSE_MODEL,
            "emotion": EMOTION_MODEL,
            "identity": ARCFACE_MODEL,
            "gender_age": GENDERAGE_MODEL,
            "ocr": OCR_MODEL,
            "attributes": CLIP_MODEL,
        },
        "note": "series samples are observations only; interpolated rows are never measured",
        # Recorded rather than silently skipped: the router fires object_masks, and nothing
        # here answers it. SAM 2 is a separate dependency and is not installed.
        "not_implemented": ["object_masks"],
        "shots": shots,
    }
    (cfg.out_root / "conditional.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "shots"}
