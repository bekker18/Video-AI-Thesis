"""Face, body and object detection over sampled frames.

Two detectors cover the three tracked streams: a dedicated face detector feeds
the "face" stream, and one general detector feeds both the "body" stream (the
person class) and the "object" stream (everything else). Sharing a single
general pass keeps the cost at two forward passes per detection frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO index of the person class, shared by YOLO and RT-DETR checkpoints.
PERSON_CLASS_ID = 0
# Low enough to keep ByteTrack's low-score boxes; the tracker applies the real thresholds itself (see trackers.TrackerConfig).
DEFAULT_CONF = 0.1
DEFAULT_IMGSZ = 640
# IoU above which two boxes are treated as the same thing detected twice.
# Deliberately high: real subjects standing close overlap far less than this, while duplicates of one subject overlap almost completely.
DEFAULT_NMS_IOU = 0.7
# Objects are held to a stricter floor than bodies. DEFAULT_CONF exists to feed ByteTrack's low-score association pass on the person stream; nothing in the
# object stream benefits from it, and at 0.1 a single confused frame is enough to open a spurious track.
DEFAULT_OBJECT_CONF = 0.3

STREAMS = ("face", "body", "object")


@dataclass(frozen=True)
class Detection:
    """One detected box in pixel coordinates on the source frame."""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class FrameDetections:
    """Detections for a single frame, split into the three tracked streams."""

    face: list[Detection]
    body: list[Detection]
    object: list[Detection]

    def stream(self, name: str) -> list[Detection]:
        return getattr(self, name)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def suppress_duplicates(
    detections: list[Detection],
    iou_threshold: float = DEFAULT_NMS_IOU,
) -> list[Detection]:
    """Class-agnostic non-maximum suppression over one frame's detections.

    RT-DETR is NMS-free by design, so nothing upstream removes near-identical
    boxes, and at DEFAULT_CONF the weak tail of its output is kept on purpose.
    The result is one subject reported two or three times, which the tracker
    faithfully turns into two or three identities.
    """
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.score):
        if all(iou(detection.xyxy, k.xyxy) < iou_threshold for k in kept):
            kept.append(detection)
    return kept


class _Model:
    """Thin wrapper over an ultralytics detector (YOLO or RT-DETR)."""

    def __init__(
        self,
        weights: Path,
        arch: str = "yolo",
        device: str = "cpu",
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
    ) -> None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {weights}\n")

        if arch == "rtdetr":
            from ultralytics import RTDETR as Arch
        elif arch == "yolo":
            from ultralytics import YOLO as Arch
        else:
            raise ValueError(f"Unknown detector arch: {arch!r}")

        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.weights_path = weights
        self.model = Arch(str(weights))
        self.names: dict[int, str] = dict(self.model.names)

    def __call__(self, image: np.ndarray) -> list[Detection]:
        """Detect on one BGR frame."""
        results = self.model.predict(
            image,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(
                xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(s),
                class_id=int(c),
                class_name=self.names.get(int(c), str(c)),
            )
            for b, s, c in zip(xyxy, scores, classes)
        ]


class MultiStreamDetector:
    """Produces face, body and object detections for a frame."""

    def __init__(
        self,
        face_weights: Path,
        general_weights: Path,
        general_arch: str = "yolo",
        device: str = "cpu",
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        detect_objects: bool = True,
        nms_iou: float = DEFAULT_NMS_IOU,
        object_conf: float = DEFAULT_OBJECT_CONF,
    ) -> None:
        self.face_model = _Model(
            face_weights, arch="yolo", device=device, imgsz=imgsz, conf=conf
        )
        self.general_model = _Model(
            general_weights, arch=general_arch, device=device, imgsz=imgsz, conf=conf
        )
        self.detect_objects = detect_objects
        self.nms_iou = nms_iou
        self.object_conf = object_conf

    @property
    def weights(self) -> dict[str, str]:
        return {
            "face": str(self.face_model.weights_path),
            "general": str(self.general_model.weights_path),
        }

    def detect(self, image: np.ndarray) -> FrameDetections:
        # Face checkpoints are single-class; relabel so the stream is readable
        # regardless of how the checkpoint names its class.
        faces = [
            Detection(d.xyxy, d.score, 0, "face")
            for d in suppress_duplicates(self.face_model(image), self.nms_iou)
        ]

        # Deduplicate before the split, not after: a duplicate only loses to the box that outscored it if the two are still in the same list.
        bodies: list[Detection] = []
        objects: list[Detection] = []
        for det in suppress_duplicates(self.general_model(image), self.nms_iou):
            if det.class_id == PERSON_CLASS_ID:
                bodies.append(det)
            elif self.detect_objects and det.score >= self.object_conf:
                objects.append(det)

        return FrameDetections(face=faces, body=bodies, object=objects)
