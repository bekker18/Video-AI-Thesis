"""Per-stream multi-object tracking with ByteTrack and BoT-SORT(-ReID).

Detection runs every `detect_every` frames while tracking runs every frame. On
a detection frame the tracker performs a full associate-and-correct update; on
the frames in between it advances the Kalman motion model only, which keeps a
box on every frame without paying for a detector pass.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .detect import Detection

# Profiles from the architecture: the affordable path runs ByteTrack (motion only), the offline path runs BoT-SORT with ReID appearance features.
PROFILES = ("affordable", "offline")

# ByteTrack/BoT-SORT defaults, matching ultralytics' shipped bytetrack.yaml and botsort.yaml. track_buffer is overridden per run (see TrackerConfig.build).
_DEFAULTS = {
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.55,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": False,
    "model": "auto",
}


@dataclass(frozen=True)
class TrackedBox:
    """A box carrying a track id, on either a detected or a predicted frame."""

    track_id: int
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    # False on frames where the box comes from the motion model alone.
    detected: bool


class _Detections:
    """Duck-typed stand-in for an ultralytics Boxes object.

    The trackers only read confidences, classes and box coordinates off their
    `results` argument, and slice it when splitting detections into the high-
    and low-confidence sets, so a plain object with those attributes is enough
    and avoids building a full Results/Boxes tensor wrapper per frame.

    `xywhr` is deliberately absent: its presence is what switches the trackers
    into oriented-box mode. `angle` is present but None for the same reason,
    since the box format is chosen by testing it against None.
    """

    angle = None

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        # (center x, center y, width, height), the other format the trackers read.
        wh = xyxy[:, 2:] - xyxy[:, :2]
        self.xywh = np.concatenate([xyxy[:, :2] + wh / 2, wh], axis=1)

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> _Detections:
        if not detections:
            return cls(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return cls(
            np.array([d.xyxy for d in detections], dtype=np.float32),
            np.array([d.score for d in detections], dtype=np.float32),
            np.array([d.class_id for d in detections], dtype=np.float32),
        )

    def __getitem__(self, index) -> _Detections:
        return _Detections(self.xyxy[index], self.conf[index], self.cls[index])

    def __len__(self) -> int:
        return len(self.conf)


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker hyper-parameters derived from the run profile."""

    profile: str = "affordable"
    detect_every: int = 2
    # How long a track survives unmatched, in source frames.
    buffer_frames: int = 30
    reid_model: str = "auto"
    device: str = "cpu"

    def build(self) -> SimpleNamespace:
        if self.profile not in PROFILES:
            raise ValueError(
                f"Unknown profile: {self.profile!r}; use one of {PROFILES}"
            )

        args = SimpleNamespace(**_DEFAULTS)
        # The tracker's frame counter only advances on detection frames, so the
        # buffer has to be expressed in detection steps rather than in frames.
        args.track_buffer = max(
            1, round(self.buffer_frames / max(self.detect_every, 1))
        )
        args.device = self.device
        if self.profile == "offline":
            args.with_reid = True
            args.model = self.reid_model
        return args


def _build_tracker(config: TrackerConfig):
    """Instantiate the ultralytics tracker for the configured profile."""
    if config.profile == "offline":
        from ultralytics.trackers.bot_sort import BOTSORT as Tracker
    else:
        from ultralytics.trackers.byte_tracker import BYTETracker as Tracker

    args = config.build()
    # 8.3 takes a frame_rate used to scale the lost-track buffer; 8.4 dropped it
    # and uses track_buffer directly. We have already scaled the buffer for the
    # detection interval, so pass the rate that makes 8.3 a no-op.
    if "frame_rate" in inspect.signature(Tracker.__init__).parameters:
        return Tracker(args, frame_rate=30)
    return Tracker(args)


def _xyxy(track) -> tuple[float, float, float, float]:
    """Read a track's current box, tolerating the property rename in 8.4."""
    box = getattr(track, "xyxy", None)
    if box is None:
        tlwh = np.asarray(track.tlwh, dtype=float)
        box = np.concatenate([tlwh[:2], tlwh[:2] + tlwh[2:]])
    box = np.asarray(box, dtype=float).reshape(-1)[:4]
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


class StreamTracker:
    """Tracks one stream across a shot, with predict-only frames in between."""

    def __init__(
        self,
        stream: str,
        config: TrackerConfig,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.stream = stream
        self.config = config
        self.class_names = class_names or {}
        self._tracker = _build_tracker(config)
        # Last ReID embedding seen per track id, used for cross-shot linking.
        self.embeddings: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        """Clear all state at a shot boundary."""
        self._tracker.reset()
        self.embeddings.clear()

    def update(
        self, detections: list[Detection], image: np.ndarray
    ) -> list[TrackedBox]:
        """Associate detections with tracks on a detection frame."""
        results = _Detections.from_detections(detections)
        rows = np.asarray(self._tracker.update(results, image), dtype=float)
        self._harvest_embeddings()
        if rows.size == 0:
            return []

        boxes: list[TrackedBox] = []
        # Rows are [x1, y1, x2, y2, track_id, score, cls, detection_index].
        for row in rows.reshape(len(rows), -1):
            x1, y1, x2, y2, track_id, score, cls = (float(v) for v in row[:7])
            class_id = int(cls)
            boxes.append(
                TrackedBox(
                    track_id=int(track_id),
                    xyxy=(x1, y1, x2, y2),
                    score=score,
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, self.stream),
                    detected=True,
                )
            )
        return boxes

    def predict(self) -> list[TrackedBox]:
        """Advance the motion model one frame without running the detector."""
        from ultralytics.trackers.basetrack import TrackState

        pool = self._tracker.tracked_stracks + self._tracker.lost_stracks
        if not pool:
            return []
        self._tracker.multi_predict(pool)

        boxes: list[TrackedBox] = []
        for track in self._tracker.tracked_stracks:
            # Only confirmed, currently-tracked boxes are worth reporting;
            # lost tracks are propagated so they can be re-found, not emitted.
            if not track.is_activated or track.state != TrackState.Tracked:
                continue
            class_id = int(track.cls)
            boxes.append(
                TrackedBox(
                    track_id=int(track.track_id),
                    xyxy=_xyxy(track),
                    score=float(track.score),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, self.stream),
                    detected=False,
                )
            )
        return boxes

    def _harvest_embeddings(self) -> None:
        """Cache BoT-SORT's smoothed ReID feature for each live track."""
        if self.config.profile != "offline":
            return
        for track in self._tracker.tracked_stracks:
            feat = getattr(track, "smooth_feat", None)
            if feat is not None:
                self.embeddings[int(track.track_id)] = np.asarray(
                    feat, dtype=np.float32
                )
