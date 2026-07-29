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
    ) -> None:
        self.face_model = _Model(
            face_weights, arch="yolo", device=device, imgsz=imgsz, conf=conf
        )
        self.general_model = _Model(
            general_weights, arch=general_arch, device=device, imgsz=imgsz, conf=conf
        )
        self.detect_objects = detect_objects

    @property
    def weights(self) -> dict[str, str]:
        return {
            "face": str(self.face_model.weights_path),
            "general": str(self.general_model.weights_path),
        }

    def detect(self, image: np.ndarray) -> FrameDetections:
        # Face checkpoints are single-class; relabel so the stream is readable
        # regardless of how the checkpoint names its class.
        faces = [Detection(d.xyxy, d.score, 0, "face") for d in self.face_model(image)]

        bodies: list[Detection] = []
        objects: list[Detection] = []
        for det in self.general_model(image):
            if det.class_id == PERSON_CLASS_ID:
                bodies.append(det)
            elif self.detect_objects:
                objects.append(det)

        return FrameDetections(face=faces, body=bodies, object=objects)
