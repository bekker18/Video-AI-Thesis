"""Field distributions per collection, after FRESCO's Figures 9-10, against prior expectations: python3 src/validation/distributions.py"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]
Video: TypeAlias = dict[str, Any]  # name, kit, tall, bars
Reader: TypeAlias = Callable[[Video], list[float]]
Mixer: TypeAlias = Callable[[Video], dict[str, float]]

AVE, REELS = COLLECTIONS = ("ave", "reels")
KNOWN = "tests"  # clips of known content
NAMES = {AVE: "AVE film scenes", REELS: "Instagram reels", KNOWN: "test clips"}
MIN_N = 20  # videos, or persons for a top check, needed to judge
BAR_FRAMES = 5  # keyframes checked for black bars
DARK = 16  # row mean grey below which a row is a black bar
BARS = 0.1  # bar share of the height that drops a video's brightness
CONSTANT = 0.01  # sd below this share of the range reads constant
COLLAPSE = 0.95  # one answer's share from which a head reads collapsed
OUTSIDE = 0.01  # share outside a fixed range that flags a field

AGES = (
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
)  # FairFace's
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
GROUPS = (
    "White",
    "Black",
    "Latino/Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
)


def _edges(means: Array) -> float:
    """Share of the lines that are black at either end."""
    lit = np.flatnonzero(means >= DARK)
    if not len(lit):
        return 1.0
    return float(lit[0] + len(means) - 1 - lit[-1]) / len(means)


def _bars(gray: Array) -> float:
    """Black bars, top and bottom or left and right: the larger share."""
    return max(_edges(gray.mean(axis=1)), _edges(gray.mean(axis=0)))


def _video(root: Path) -> Video:
    kit = json.loads((root / "identikit.json").read_text(encoding="utf-8"))
    shots = json.loads(
        (root / "json" / "segmentation.json").read_text(encoding="utf-8")
    )["shots"]
    paths = [root / k["path"] for s in shots for k in s["keyframes"]]
    picks = (
        np.linspace(0, len(paths) - 1, min(BAR_FRAMES, len(paths))).round().astype(int)
    )
    grays = [
        g
        for i in picks
        if (g := cv2.imread(str(paths[int(i)]), cv2.IMREAD_GRAYSCALE)) is not None
    ]
    height, width = grays[0].shape if grays else (1, 1)
    return {
        "name": root.name,
        "kit": kit,
        "tall": height / min(height, width),  # frame height over its short side
        "bars": float(np.median([_bars(g) for g in grays])) if grays else 0.0,
    }


def _get(found: Any, *path: str) -> Any:
    for key in path:
        found = found.get(key) if isinstance(found, dict) else None
    return found


def _seconds(kit: dict[str, Any]) -> dict[int, float]:
    return {int(s["segment"]): float(s["seconds"]) for s in kit["timeline"]["segments"]}


def _described(kit: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in kit["figurative"]["persons"] if p.get("status") == "described"]


def _watchers(kit: dict[str, Any]) -> list[dict[str, Any]]:
    ids = {int(p["person_id"]) for p in _described(kit)}
    return [p for p in kit["enunciative"]["persons"] if int(p["person_id"]) in ids]


def _ratio(pairs: list[tuple[float, float]]) -> float | None:
    total = sum(w for _, w in pairs)
    return sum(x * w for x, w in pairs) / total if total > 0 else None


def _sum(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Weighted sum and total weight, for the band checks to pool."""
    total = sum(w for _, w in pairs)
    return (sum(x * w for x, w in pairs), total) if total > 0 else None


# Readers: one value per segment, person or video


def _each(level: str, *path: str) -> Reader:
    def read(v: Video) -> list[float]:
        return [
            float(x)
            for s in v["kit"][level]["segments"]
            if (x := _get(s, *path)) is not None
        ]

    return read


def _person(*path: str) -> Reader:
    def read(v: Video) -> list[float]:
        return [
            float(x) for p in _described(v["kit"]) if (x := _get(p, *path)) is not None
        ]

    return read


def _watcher(*path: str, floor: int = 1) -> Reader:
    def read(v: Video) -> list[float]:
        return [
            float(x)
            for p in _watchers(v["kit"])
            if (x := _get(p, *path)) is not None
            and int(_get(p, *path[:-1], "n") or 0) >= floor
        ]

    return read


def _one(read: Callable[[Video], float | None]) -> Reader:
    def wrap(v: Video) -> list[float]:
        found = read(v)
        return [] if found is None else [found]

    return wrap


def _things(key: str) -> Reader:
    """Mask2Former's person segments, pooled per segment by 09."""

    def read(v: Video) -> list[float]:
        return [
            float(c[key])
            for s in v["kit"]["plastic"]["segments"]
            for c in s.get("centralities") or []
            if c["label"] == "person"
        ]

    return read


def _person_depth(v: Video) -> list[float]:
    return [
        float(o["disparity"])
        for s in v["kit"]["plastic"]["segments"]
        for o in (s.get("depth") or {}).get("objects", [])
        if o["label"] == "person"
    ]


def _background_depth(v: Video) -> list[float]:
    return [
        float(x)
        for s in v["kit"]["plastic"]["segments"]
        if (x := _get(s, "depth", "background")) is not None
    ]


def _shot_lengths(v: Video) -> list[float]:
    return [
        math.log10(float(s["seconds"]))
        for s in v["kit"]["timeline"]["segments"]
        if float(s["seconds"]) > 0
    ]


def _face_height(v: Video) -> list[float]:
    """Largest face over the short side, so vertical and horizontal frames compare."""
    return [
        float(d["face_height"]) * float(v["tall"])
        for s in v["kit"]["enunciative"]["segments"]
        if (d := s.get("camera_distance")) and float(d["face_height"]) > 0
    ]


def _face_ratio(v: Video) -> list[float]:
    """Segments with a face, as for face height: 0 means no face, not a small one."""
    return [
        float(x)
        for s in v["kit"]["enunciative"]["segments"]
        if (x := s.get("face_background_ratio"))
    ]


def _people_time(v: Video) -> float | None:
    return _ratio(
        [
            (1.0 if s["person_ids"] else 0.0, float(s["seconds"]))
            for s in v["kit"]["timeline"]["segments"]
        ]
    )


def _women(v: Video) -> float | None:
    return _ratio(
        [
            (float(p["gender"]["distribution"]["female"]), float(p["seconds"]))
            for p in _described(v["kit"])
            if p.get("gender")
        ]
    )


def _looks(v: Video) -> float | None:
    relations = v["kit"]["summary"]["relations"]
    pairs = int(relations["co_present"])
    return int(relations["with_evidence"]) / pairs if pairs else None


def _address(v: Video) -> float | None:
    return _ratio(
        [
            (float(p["address"]["share"]), float(p["address"]["n"]))
            for p in _watchers(v["kit"])
            if p.get("address")
        ]
    )


def _gaze_in(v: Video) -> float | None:
    return _ratio(
        [
            (float(p["gaze"]["in_frame_share"]), float(p["gaze"]["n"]))
            for p in _watchers(v["kit"])
            if p.get("gaze")
        ]
    )


def _happiness(v: Video) -> float | None:
    return _ratio(
        [
            (float(p["emotion"]["distribution"]["happiness"]), float(p["seconds"]))
            for p in _described(v["kit"])
            if p.get("emotion")
        ]
    )


def _age(v: Video) -> float | None:
    """Mean bracket index (0 for 0-2, 8 for 70+), by screen time."""
    return _ratio(
        [
            (
                sum(i * float(p["age"]["distribution"][b]) for i, b in enumerate(AGES)),
                float(p["seconds"]),
            )
            for p in _described(v["kit"])
            if p.get("age")
        ]
    )


def _portrait(v: Video) -> float | None:
    seconds = _seconds(v["kit"])
    return _ratio(
        [
            (
                1.0 if s["portrait_scene"] == "portrait" else 0.0,
                seconds[int(s["segment"])],
            )
            for s in v["kit"]["enunciative"]["segments"]
            if s.get("portrait_scene")
        ]
    )


def _single(v: Video) -> float | None:
    seconds = _seconds(v["kit"])
    return _ratio(
        [
            (1.0 if s["crowdedness"] == "single" else 0.0, seconds[int(s["segment"])])
            for s in v["kit"]["figurative"]["segments"]
            if s["crowdedness"] != "none"
        ]
    )


def _brightness(v: Video) -> float | None:
    """05 measured black bars too, so a barred video is left out."""
    found = _get(v["kit"], "plastic", "video", "brightness", "mean")
    return float(found) if found is not None and v["bars"] < BARS else None


def _outdoor(v: Video) -> float | None:
    found = _get(v["kit"], "figurative", "video", "indoor_outdoor", "outdoor")
    return None if found is None else float(found)


def _off_centre(v: Video) -> float | None:
    found = _things("x")(v)
    return float(np.mean([abs(x - 0.5) for x in found])) if found else None


def _yaw_mean(v: Video) -> tuple[float, float] | None:
    return _sum([(y, 1.0) for y in _watcher("head_pose", "yaw", "mean")(v)])


def _yaw_side(v: Video) -> tuple[float, float] | None:
    return _sum(
        [(1.0 if y > 0 else 0.0, 1.0) for y in _watcher("head_pose", "yaw", "mean")(v)]
    )


def _centre_x(v: Video) -> tuple[float, float] | None:
    return _sum([(x, 1.0) for x in _things("x")(v)])


def _women_pooled(v: Video) -> tuple[float, float] | None:
    return _sum(
        [
            (float(p["gender"]["distribution"]["female"]), float(p["seconds"]))
            for p in _described(v["kit"])
            if p.get("gender")
        ]
    )


def _texted(v: Video) -> tuple[float, float]:
    return (1.0 if int(v["kit"]["summary"]["text"]["readings"]) else 0.0, 1.0)


def _interior(v: Video) -> tuple[float, float] | None:
    seconds = _seconds(v["kit"])
    return _sum(
        [
            (
                1.0 if s["indoor_outdoor"]["label"] == "indoor" else 0.0,
                seconds[int(s["segment"])],
            )
            for s in v["kit"]["figurative"]["segments"]
            if s.get("indoor_outdoor")
        ]
    )


def _modal_emotion(v: Video) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in _described(v["kit"]):
        if p.get("emotion"):
            out[p["emotion"]["modal"]] = out.get(p["emotion"]["modal"], 0.0) + 1.0
    return out


def _medium(v: Video) -> str:
    return str(_get(v["kit"], "plastic", "video", "medium", "label"))


def _subjects(v: Video) -> list[str]:
    return [str(s["label"]) for s in v["kit"]["figurative"]["subjects"]]


def _places(v: Video) -> list[str]:
    return [
        str(p["label"])
        for p in (_get(v["kit"], "figurative", "video", "place") or [])[:2]
    ]


def _io(v: Video) -> str:
    return str(_get(v["kit"], "figurative", "video", "indoor_outdoor", "label"))


def _readings(v: Video) -> int:
    return int(v["kit"]["summary"]["text"]["readings"])


def _described_count(v: Video) -> int:
    return int(v["kit"]["summary"]["persons"]["described"])


# Figure 9: name, level, unit, fixed range (None: pooled 0.5-99.5 percentiles), reader
NUMERIC: tuple[tuple[str, str, str, tuple[float, float] | None, Reader], ...] = (
    (
        "brightness",
        "plastic",
        "segment",
        (0.0, 1.0),
        _each("plastic", "visual", "brightness"),
    ),
    (
        "saturation",
        "plastic",
        "segment",
        (0.0, 1.0),
        _each("plastic", "visual", "saturation"),
    ),
    ("chroma", "plastic", "segment", None, _each("plastic", "visual", "chroma")),
    ("warmth (b*)", "plastic", "segment", None, _each("plastic", "visual", "warmth")),
    ("tint (a*)", "plastic", "segment", None, _each("plastic", "visual", "tint")),
    (
        "contrast",
        "plastic",
        "segment",
        None,
        _each("plastic", "visual", "contrast_rms"),
    ),
    (
        "edge density",
        "plastic",
        "segment",
        None,
        _each("plastic", "visual", "edge_density"),
    ),
    (
        "symmetry",
        "plastic",
        "segment",
        None,
        _each("plastic", "visual", "horizontal_symmetry"),
    ),
    ("shot length (log10 s)", "plastic", "segment", None, _shot_lengths),
    ("person position, horizontal", "plastic", "segment", (0.0, 1.0), _things("x")),
    ("person position, vertical", "plastic", "segment", (0.0, 1.0), _things("y")),
    ("person centrality", "plastic", "segment", (0.0, 1.0), _things("centrality")),
    ("person depth (disparity)", "plastic", "segment", (0.0, 1.0), _person_depth),
    (
        "background depth (disparity)",
        "plastic",
        "segment",
        (0.0, 1.0),
        _background_depth,
    ),
    (
        "outdoor probability",
        "figurative",
        "segment",
        (0.0, 1.0),
        _each("figurative", "indoor_outdoor", "outdoor"),
    ),
    ("people on screen", "figurative", "video", (0.0, 1.0), _one(_people_time)),
    ("women, by screen time", "figurative", "video", (0.0, 1.0), _one(_women)),
    ("valence", "figurative", "person", (-1.0, 1.0), _person("valence", "mean")),
    ("arousal", "figurative", "person", (-1.0, 1.0), _person("arousal", "mean")),
    ("body movement", "figurative", "person", None, _person("movement", "mean")),
    (
        "face height, of the short side",
        "enunciative",
        "segment",
        (0.0, 1.0),
        _face_height,
    ),
    ("face to background ratio", "enunciative", "segment", None, _face_ratio),
    (
        "head yaw (degrees)",
        "enunciative",
        "person",
        (-90.0, 90.0),
        _watcher("head_pose", "yaw", "mean"),
    ),
    (
        "head pitch (degrees)",
        "enunciative",
        "person",
        (-90.0, 90.0),
        _watcher("head_pose", "pitch", "mean"),
    ),
    (
        "head roll (degrees)",
        "enunciative",
        "person",
        (-90.0, 90.0),
        _watcher("head_pose", "roll", "mean"),
    ),
    (
        "gaze in-frame probability",
        "enunciative",
        "person",
        (0.0, 1.0),
        _watcher("gaze", "in_frame", "mean"),
    ),
    (
        "looks into the camera",
        "enunciative",
        "person",
        (0.0, 1.0),
        _watcher("address", "share", floor=2),
    ),
    ("looks between people", "enunciative", "video", (0.0, 1.0), _one(_looks)),
)


def _mix_persons(key: str) -> Mixer:
    """Described persons' distributions summed, one vote each."""

    def mix(v: Video) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in _described(v["kit"]):
            for label, share in (_get(p, key, "distribution") or {}).items():
                out[label] = out.get(label, 0.0) + float(share)
        return out

    return mix


def _mix_time(level: str, *path: str) -> Mixer:
    def mix(v: Video) -> dict[str, float]:
        seconds = _seconds(v["kit"])
        out: dict[str, float] = {}
        for s in v["kit"][level]["segments"]:
            label = _get(s, *path)
            if label and label != "unknown":
                out[str(label)] = out.get(str(label), 0.0) + seconds[int(s["segment"])]
        return out

    return mix


# Figure 10: name, unit, label order, mixer, answered by one model head
CATEGORICAL: tuple[tuple[str, str, tuple[str, ...], Mixer, bool], ...] = (
    ("emotion", "share of described persons", EMOTIONS, _mix_persons("emotion"), True),
    ("age", "share of described persons", AGES, _mix_persons("age"), True),
    (
        "gender",
        "share of described persons",
        ("male", "female"),
        _mix_persons("gender"),
        True,
    ),
    (
        "perceived ethnicity",
        "share of described persons, live action only",
        GROUPS,
        _mix_persons("perceived_ethnicity"),
        True,
    ),
    (
        "medium",
        "share of running time",
        ("live action", "2D animation", "CGI"),
        _mix_time("plastic", "medium", "modal"),
        False,
    ),
    (
        "shot scale",
        "share of running time with someone followed",
        ("extreme close-up", "close-up", "medium", "long", "extreme long"),
        _mix_time("enunciative", "camera_distance", "scale"),
        False,
    ),
    (
        "camera movement",
        "share of running time",
        ("static", "handheld", "pan", "tilt", "zoom"),
        _mix_time("enunciative", "camera", "movement"),
        False,
    ),
    (
        "crowdedness",
        "share of running time",
        ("none", "single", "couple", "group", "crowd"),
        _mix_time("figurative", "crowdedness"),
        False,
    ),
)


@dataclass(frozen=True)
class Expectation:
    id: str
    claim: str
    kind: str  # higher, equal, band, top, known
    where: tuple[str, ...]  # (high, low), (a, b), (collection,) or (video,), by kind
    read: Callable[[Video], Any]
    bounds: tuple[float, float] = (0.0, 0.0)  # band's interval, or equal's margin
    label: str = ""  # top: the expected commonest label
    source: str = ""


NAVARRO = "Navarro-Guere 2024, 398 Instagram Reels from a feed"
EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        "E1",
        "Reels look into the camera more than film scenes",
        "higher",
        (REELS, AVE),
        _address,
    ),
    Expectation(
        "E2",
        "Film scenes keep gaze inside the frame more than reels",
        "higher",
        (AVE, REELS),
        _gaze_in,
    ),
    Expectation(
        "E3",
        "Reels show more happiness than film scenes",
        "higher",
        (REELS, AVE),
        _happiness,
        source="FRESCO: FFHQ peaks at happy, OpenImages at neutral",
    ),
    Expectation(
        "E4",
        "Reels spend more of their time in portrait framing",
        "higher",
        (REELS, AVE),
        _portrait,
    ),
    Expectation(
        "E5",
        "Reels spend more of their peopled time on one person",
        "higher",
        (REELS, AVE),
        _single,
    ),
    Expectation(
        "E6",
        "Reels are brighter than film scenes, videos without black bars",
        "higher",
        (REELS, AVE),
        _brightness,
        source="FRESCO: the least similar images are darker",
    ),
    Expectation(
        "E7",
        "Reels and film scenes are as outdoor, within 0.10",
        "equal",
        (REELS, AVE),
        _outdoor,
        bounds=(-0.1, 0.1),
        source=f"{NAVARRO}: 37% exterior; AVE's labels: 36%",
    ),
    Expectation(
        "E8",
        "Film scenes show older people than reels",
        "higher",
        (AVE, REELS),
        _age,
        source="Selfiecity: median age 25",
    ),
    Expectation(
        "S1",
        "Head yaw averages within 5 degrees of the lens, film scenes",
        "band",
        (AVE,),
        _yaw_mean,
        (-5.0, 5.0),
    ),
    Expectation(
        "S2",
        "Head yaw averages within 5 degrees of the lens, reels",
        "band",
        (REELS,),
        _yaw_mean,
        (-5.0, 5.0),
    ),
    Expectation(
        "S3",
        "Heads turn either way about equally (45-55%), film scenes",
        "band",
        (AVE,),
        _yaw_side,
        (0.45, 0.55),
    ),
    Expectation(
        "S4",
        "Heads turn either way about equally (45-55%), reels",
        "band",
        (REELS,),
        _yaw_side,
        (0.45, 0.55),
    ),
    Expectation(
        "S5",
        "Persons sit near the centre horizontally (mean 0.4-0.6), film scenes",
        "band",
        (AVE,),
        _centre_x,
        (0.4, 0.6),
        source="FRESCO: centroids mostly central",
    ),
    Expectation(
        "S6",
        "Persons sit near the centre horizontally (mean 0.4-0.6), reels",
        "band",
        (REELS,),
        _centre_x,
        (0.4, 0.6),
        source="FRESCO: centroids mostly central",
    ),
    Expectation(
        "S7",
        "Persons sit further off-centre in film scenes than in reels",
        "higher",
        (AVE, REELS),
        _off_centre,
    ),
    Expectation(
        "S8",
        "Neutral is the commonest emotion of persons in film scenes",
        "top",
        (AVE,),
        _modal_emotion,
        label="neutral",
        source="FRESCO: OpenImages peaks at neutral",
    ),
    Expectation(
        "A1",
        "Women hold 25-40% of described persons' screen time, film scenes",
        "band",
        (AVE,),
        _women_pooled,
        (0.25, 0.4),
        source="USC Annenberg: 29.9-34.6% of speaking characters 2007-2022; GD-IQ: about 36% of screen presence",
    ),
    Expectation(
        "A2",
        "30-65% of reels carry on-screen text",
        "band",
        (REELS,),
        _texted,
        (0.3, 0.65),
        source=f"{NAVARRO}: 46%",
    ),
    Expectation(
        "A3",
        "Reels are indoors 40-70% of their running time",
        "band",
        (REELS,),
        _interior,
        (0.4, 0.7),
        source=f"{NAVARRO}: 49% interior, 10% mixed",
    ),
    Expectation(
        "K1",
        "anime reads 2D animation",
        "known",
        ("anime",),
        lambda v: (_medium(v) == "2D animation", _medium(v)),
    ),
    Expectation(
        "K2",
        "cat1 has a cat as subject",
        "known",
        ("cat1",),
        lambda v: ("cat" in _subjects(v), ", ".join(_subjects(v))),
    ),
    Expectation(
        "K3",
        "cat2 has a cat as subject",
        "known",
        ("cat2",),
        lambda v: ("cat" in _subjects(v), ", ".join(_subjects(v))),
    ),
    Expectation(
        "K4",
        "cat2 has no described person",
        "known",
        ("cat2",),
        lambda v: (_described_count(v) == 0, f"{_described_count(v)} described"),
    ),
    Expectation(
        "K5",
        "messi's two leading places include soccer or stadium",
        "known",
        ("messi",),
        lambda v: (
            any("soccer" in p or "stadium" in p for p in _places(v)),
            ", ".join(_places(v)),
        ),
    ),
    Expectation(
        "K6",
        "messi has on-screen text read",
        "known",
        ("messi",),
        lambda v: (_readings(v) > 0, f"{_readings(v)} readings"),
    ),
    Expectation(
        "K7",
        "patrick reads outdoor",
        "known",
        ("patrick",),
        lambda v: (_io(v) == "outdoor", _io(v)),
    ),
    Expectation(
        "K8",
        "thanos looks into the camera in half its samples or more",
        "known",
        ("thanos",),
        lambda v: ((_address(v) or 0.0) >= 0.5, f"{_address(v) or 0.0:.2f}"),
    ),
    Expectation(
        "K9",
        "thanos reads live action: the known CGI miss (STATE item 27)",
        "known",
        ("thanos",),
        lambda v: (_medium(v) == "live action", _medium(v)),
    ),
)


def _values(e: Expectation, videos: dict[str, list[Video]]) -> dict[str, list[Any]]:
    if e.kind == "known":
        return {
            e.where[0]: [
                e.read(v) for v in videos.get(KNOWN, []) if v["name"] == e.where[0]
            ]
        }
    return {
        c: [x for v in videos.get(c, []) if (x := e.read(v)) is not None]
        for c in e.where
    }


def _auc(high: Array, low: Array) -> float:
    """P(first > second), ties counting half."""
    ordered = np.sort(low)
    below = np.searchsorted(ordered, high, side="left")
    equal = np.searchsorted(ordered, high, side="right") - below
    return float((below + 0.5 * equal).sum() / (len(high) * len(low)))


def _ci(stats: Array) -> list[float]:
    low, high = np.nanpercentile(stats, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def _inside(ci: list[float], low: float, high: float) -> str:
    if low <= ci[0] and ci[1] <= high:
        return "pass"
    return "fail" if ci[1] < low or ci[0] > high else "weak"


def _judge(
    e: Expectation, values: dict[str, list[Any]], rng: np.random.Generator, boot: int
) -> dict[str, Any]:
    """Each interval resamples whole videos."""
    out: dict[str, Any] = {
        "id": e.id,
        "claim": e.claim,
        "kind": e.kind,
        "where": list(e.where),
        "n": [len(values[c]) for c in e.where],
        "source": e.source,
    }
    if e.kind == "known":
        found = values[e.where[0]]
        if not found:
            return {**out, "result": "missing"}
        held, seen = found[0]
        return {**out, "seen": seen, "result": "pass" if held else "fail"}

    if e.kind in ("higher", "equal"):
        a, b = (np.asarray(values[c], dtype=float) for c in e.where)
        if min(len(a), len(b)) < MIN_N:
            return {**out, "result": "too few"}
        rows = zip(
            rng.integers(0, len(a), (boot, len(a))),
            rng.integers(0, len(b), (boot, len(b))),
        )
        out["means"] = [round(float(a.mean()), 4), round(float(b.mean()), 4)]
        if e.kind == "higher":
            out["estimate"] = round(_auc(a, b), 4)
            out["ci"] = _ci(np.asarray([_auc(a[i], b[j]) for i, j in rows]))
            out["result"] = (
                "pass"
                if out["ci"][0] > 0.5
                else ("fail" if out["ci"][1] < 0.5 else "weak")
            )
        else:
            out["estimate"] = round(float(a.mean() - b.mean()), 4)
            out["ci"] = _ci(np.asarray([a[i].mean() - b[j].mean() for i, j in rows]))
            out["result"] = _inside(out["ci"], *e.bounds)
        return out

    found = values[e.where[0]]
    if e.kind == "band":
        if len(found) < MIN_N:
            return {**out, "result": "too few"}
        pairs = np.asarray(found, dtype=float)
        picked = pairs[rng.integers(0, len(pairs), (boot, len(pairs)))].sum(axis=1)
        out["estimate"] = round(float(pairs[:, 0].sum() / pairs[:, 1].sum()), 4)
        out["ci"] = _ci(picked[:, 0] / np.where(picked[:, 1] > 0, picked[:, 1], np.nan))
        out["result"] = _inside(out["ci"], *e.bounds)
        return out

    labels = sorted({k for counts in found for k in counts})
    matrix = np.asarray(
        [[counts.get(k, 0.0) for k in labels] for counts in found], dtype=float
    )
    if not len(matrix) or matrix.sum() < MIN_N:
        return {**out, "result": "too few"}
    total = matrix.sum(axis=0)
    tops = (
        matrix[rng.integers(0, len(matrix), (boot, len(matrix)))]
        .sum(axis=1)
        .argmax(axis=1)
    )
    share = float(np.mean([labels[int(t)] == e.label for t in tops]))
    out["leader"] = labels[int(total.argmax())]
    out["estimate"] = (
        round(float(total[labels.index(e.label)] / total.sum()), 4)
        if e.label in labels
        else 0.0
    )
    out["top_in_resamples"] = round(share, 4)
    out["result"] = "pass" if share >= 0.95 else ("fail" if share <= 0.05 else "weak")
    return out


def _span(values: Array) -> tuple[float, float]:
    if not len(values):
        return (0.0, 1.0)
    low, high = (float(x) for x in np.percentile(values, [0.5, 99.5]))
    return (low, high) if high > low else (low, low + 1.0)


def _numeric_flags(values: Array, span: tuple[float, float], fixed: bool) -> list[str]:
    flags: list[str] = []
    if len(values) >= MIN_N and float(values.std()) < CONSTANT * (span[1] - span[0]):
        flags.append("constant")
    if (
        fixed
        and len(values)
        and float(((values < span[0]) | (values > span[1])).mean()) > OUTSIDE
    ):
        flags.append("out of range")
    return flags


def _categorical_flags(weights: dict[str, float], head: bool) -> list[str]:
    total = sum(weights.values())
    if head and total >= MIN_N and max(weights.values()) / total >= COLLAPSE:
        return ["collapsed"]
    return []


def _numeric(videos: dict[str, list[Video]]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for name, level, unit, fixed, read in NUMERIC:
        per = {c: [read(v) for v in vs] for c, vs in videos.items()}
        values = {
            c: np.asarray([x for found in lists for x in found], dtype=float)
            for c, lists in per.items()
        }
        span = fixed or _span(np.concatenate([values[c] for c in COLLECTIONS]))
        fields.append(
            {
                "name": name,
                "level": level,
                "unit": unit,
                "fixed": fixed is not None,
                "span": span,
                "values": values,
                "without": {
                    c: sum(1 for found in lists if not found)
                    for c, lists in per.items()
                },
            }
        )
    return fields


def _categorical(videos: dict[str, list[Video]]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for name, unit, labels, mix, head in CATEGORICAL:
        weights: dict[str, dict[str, float]] = {}
        for c in COLLECTIONS:
            total: dict[str, float] = {}
            for v in videos[c]:
                for label, w in mix(v).items():
                    total[label] = total.get(label, 0.0) + w
            weights[c] = total
        seen = {k for t in weights.values() for k in t}
        order = [k for k in labels] + sorted(seen - set(labels))
        fields.append(
            {
                "name": name,
                "unit": unit,
                "labels": order,
                "weights": weights,
                "head": head,
            }
        )
    return fields


def _planted(
    table: dict[str, dict[str, list[Any]]],
    results: list[dict[str, Any]],
    numeric: list[dict[str, Any]],
    categorical: list[dict[str, Any]],
    seed: int,
    boot: int,
) -> list[dict[str, Any]]:
    """Faults the checks must catch, after Adebayo et al. (2018)."""
    rng = np.random.default_rng([seed, 999])
    between = [
        e for e in EXPECTATIONS if e.kind == "higher" and set(e.where) == {AVE, REELS}
    ]
    passing = []
    for e in between:
        pooled = table[e.id][e.where[0]] + table[e.id][e.where[1]]
        order = rng.permutation(len(pooled))
        cut = len(table[e.id][e.where[0]])
        shuffled = {
            e.where[0]: [pooled[i] for i in order[:cut]],
            e.where[1]: [pooled[i] for i in order[cut:]],
        }
        if _judge(e, shuffled, rng, boot)["result"] == "pass":
            passing.append(e.id)

    # only a passing check can show that swapped data fails it
    passed = {r["id"] for r in results if r["result"] == "pass"}
    first = next((e for e in between if e.id in passed), None)
    swap = "no between-collection expectation passed"
    if first is not None:
        swapped = {
            first.where[0]: table[first.id][first.where[1]],
            first.where[1]: table[first.id][first.where[0]],
        }
        swap = _judge(first, swapped, rng, boot)["result"]

    bright = next(f for f in numeric if f["name"] == "brightness")
    constant = _numeric_flags(
        np.full(len(bright["values"][AVE]), 0.5), bright["span"], bright["fixed"]
    )

    gender = next(f for f in categorical if f["name"] == "gender")
    collapsed = _categorical_flags({"male": sum(gender["weights"][AVE].values())}, True)
    return [
        {
            "fault": "collections shuffled between videos",
            "expected": "no between-collection expectation passes, bar chance",
            "found": f"{len(passing)} of {len(between)} pass: {', '.join(passing) or 'none'}",
            "caught": len(passing) <= 1,
        },
        {
            "fault": f"{first.id if first else 'a passing check'}'s values swapped between collections",
            "expected": "fail",
            "found": swap,
            "caught": swap == "fail",
        },
        {
            "fault": "every AVE segment's brightness set to 0.5",
            "expected": "constant",
            "found": ", ".join(constant) or "no flag",
            "caught": "constant" in constant,
        },
        {
            "fault": "every AVE person read male",
            "expected": "collapsed",
            "found": ", ".join(collapsed) or "no flag",
            "caught": "collapsed" in collapsed,
        },
    ]


# Pages, in the identikit page's ink and surfaces

PAGE_SIZE = (22.4, 13.5)  # inches at 100 dpi
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
PAGE, SURFACE, HAIRLINE, BASELINE = "#f9f9f7", "#fcfcfb", "#e1e0d9", "#c3c2b7"
SERIES = {AVE: "#2a78d6", REELS: "#eb6834"}  # palette slots 1 and 2
STATUS = {"pass": "#0ca30c", "weak": "#fab219", "fail": "#d03b3b"}
GRID = np.linspace(0.0, 1.0, 241)


def _density(x: Array) -> Array:
    """Gaussian KDE on GRID, over a fine histogram for speed."""
    width = max(1.06 * float(x.std()) * len(x) ** -0.2, 0.012)
    counts, edges = np.histogram(x, bins=480, range=(0.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2
    out: Array = (
        np.exp(-0.5 * ((GRID[:, None] - centres[None, :]) / width) ** 2) @ counts
    )
    return out


def _short(x: float) -> str:
    return f"{x:.3g}"


def _legend(
    fig: Figure, videos: dict[str, list[Video]], known: bool, x: float, y: float
) -> None:
    for c in COLLECTIONS:
        fig.text(x, y, "■", fontsize=14, color=SERIES[c], va="center")
        fig.text(
            x + 0.012,
            y,
            f"{NAMES[c]} · {len(videos[c])} videos",
            fontsize=11,
            color=SECONDARY,
            va="center",
        )
        x += 0.2
    if known:
        fig.text(x, y, f"| {NAMES[KNOWN]}", fontsize=11, color=INK, va="center")


def _ridges(
    fig: Figure,
    rect: tuple[float, float, float, float],
    title: str,
    rows: list[dict[str, Any]],
) -> None:
    x, y, w, h = rect
    fig.text(
        x, y + h + 0.006, title, fontsize=14, weight="bold", color=INK, va="bottom"
    )
    ax = fig.add_axes((x + 0.15, y, w - 0.2, h))
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, len(rows) - 0.05)
    for i, field in enumerate(rows):
        base = len(rows) - 1 - i
        low, high = field["span"]
        ax.axhline(base, color=HAIRLINE, linewidth=0.8)
        for c in COLLECTIONS:
            values = field["values"][c]
            if len(values) < 2:
                continue
            density = _density(np.clip((values - low) / (high - low), 0.0, 1.0))
            if density.max() <= 0:
                continue
            top = base + 0.85 * density / density.max()
            ax.fill_between(GRID, base, top, color=SERIES[c], alpha=0.16, linewidth=0)
            ax.plot(GRID, top, color=SERIES[c], linewidth=1.4)
        known = field["values"].get(KNOWN)
        if known is not None and len(known):
            ticks = np.clip((known - low) / (high - low), 0.0, 1.0)
            ax.plot(
                ticks, np.full(len(ticks), base + 0.07), "|", color=INK, markersize=7
            )
        ax.text(
            -0.02,
            base + 0.42,
            field["name"],
            ha="right",
            va="center",
            fontsize=10.5,
            color=INK,
        )
        ax.text(
            -0.02,
            base + 0.12,
            f"per {field['unit']}",
            ha="right",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
        ax.text(
            1.02,
            base + 0.3,
            f"{_short(low)} to {_short(high)}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )


def _numeric_page(
    path: Path, numeric: list[dict[str, Any]], videos: dict[str, list[Video]]
) -> None:
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(
        0.018,
        0.988,
        "Distributions: numeric fields",
        fontsize=26,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.018,
        0.95,
        "After FRESCO's Figure 9. Each ridge is one field's density in a collection, at its own peak height; "
        "fixed ranges where the field has one, else the 0.5-99.5 percentiles of both collections.",
        fontsize=12,
        color=SECONDARY,
        va="top",
    )
    _legend(fig, videos, KNOWN in videos, 0.52, 0.926)
    rows = {
        level: [f for f in numeric if f["level"] == level]
        for level in ("plastic", "figurative", "enunciative")
    }
    top, bottom = 0.89, 0.03
    unit = (top - bottom - 0.05) / max(
        len(rows["plastic"]), len(rows["figurative"]) + len(rows["enunciative"])
    )
    _ridges(
        fig,
        (0.02, top - unit * len(rows["plastic"]), 0.47, unit * len(rows["plastic"])),
        "Plastic",
        rows["plastic"],
    )
    high = unit * len(rows["figurative"])
    _ridges(fig, (0.51, top - high, 0.47, high), "Figurative", rows["figurative"])
    low = unit * len(rows["enunciative"])
    _ridges(
        fig,
        (0.51, top - high - 0.05 - low, 0.47, low),
        "Enunciative",
        rows["enunciative"],
    )
    fig.savefig(path, facecolor=PAGE)


def _categorical_page(
    path: Path, categorical: list[dict[str, Any]], videos: dict[str, list[Video]]
) -> None:
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(
        0.018,
        0.988,
        "Distributions: categorical fields",
        fontsize=26,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.018,
        0.95,
        "After FRESCO's Figure 10. Person fields: each described person's distribution, one person one vote; "
        "segment fields: share of running time.",
        fontsize=12,
        color=SECONDARY,
        va="top",
    )
    _legend(fig, videos, False, 0.55, 0.945)
    grid = fig.add_gridspec(
        2, 4, left=0.04, right=0.985, top=0.85, bottom=0.12, wspace=0.22, hspace=0.75
    )
    for k, field in enumerate(categorical):
        ax = fig.add_subplot(grid[k // 4, k % 4])
        ax.set_facecolor(SURFACE)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(BASELINE)
        ax.tick_params(length=0, labelsize=8.5, labelcolor=MUTED)
        ax.grid(axis="y", color=HAIRLINE, linewidth=1)
        ax.set_axisbelow(True)
        labels = field["labels"]
        for i, c in enumerate(COLLECTIONS):
            weights = field["weights"][c]
            total = sum(weights.values()) or 1.0
            ax.bar(
                np.arange(len(labels)) + (i - 0.5) * 0.38,
                [weights.get(label, 0.0) / total for label in labels],
                width=0.36,
                color=SERIES[c],
            )
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(
            field["name"], loc="left", fontsize=12, weight="bold", color=INK, pad=18
        )
        ax.text(
            0.0, 1.03, field["unit"], transform=ax.transAxes, fontsize=8.5, color=MUTED
        )
    fig.savefig(path, facecolor=PAGE)


def _estimate(r: dict[str, Any]) -> str:
    if r["kind"] == "known":
        return f"seen: {r.get('seen', '-')}"
    if "estimate" not in r:
        return f"n {r['n']}"
    if r["kind"] == "top":
        return f"leader {r['leader']}; {r['estimate']:.0%} of persons; top in {r['top_in_resamples']:.0%} of resamples"
    name = {"higher": "AUC", "equal": "difference", "band": "value"}[r["kind"]]
    return f"{name} {r['estimate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] · n {', '.join(str(n) for n in r['n'])}"


def _summary_page(path: Path, report: dict[str, Any]) -> None:
    results = report["expectations"]
    fig = Figure(figsize=PAGE_SIZE, dpi=100, facecolor=PAGE)
    fig.text(
        0.018,
        0.988,
        "Distributions: expectations",
        fontsize=26,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.018,
        0.95,
        "Written before the first run. Between collections: AUC, P(first > second); bands and equivalence: the 95% "
        f"interval must sit inside. Intervals from {report['resamples']:,} resamples of whole videos.",
        fontsize=12,
        color=SECONDARY,
        va="top",
    )
    for i, result in enumerate(("pass", "weak", "fail", "too few")):
        left = 0.62 + i * 0.09
        count = sum(1 for r in results if r["result"] == result)
        fig.text(
            left, 0.985, str(count), fontsize=28, weight="semibold", color=INK, va="top"
        )
        fig.text(
            left,
            0.94,
            f"■ {result}",
            fontsize=10.5,
            color=STATUS.get(result, BASELINE),
            va="top",
        )

    step = min(0.027, 0.66 / max(1, len(results)))
    y = 0.9
    for header, x in (("check", 0.018), ("estimate [95% CI]", 0.5), ("result", 0.93)):
        fig.text(x, y, header, fontsize=9.5, color=MUTED, va="top")
    for r in results:
        y -= step
        fig.text(0.018, y, r["id"], fontsize=10.5, weight="bold", color=INK, va="top")
        fig.text(
            0.045,
            y,
            textwrap.shorten(r["claim"], 85),
            fontsize=10.5,
            color=INK,
            va="top",
        )
        fig.text(
            0.5,
            y,
            textwrap.shorten(_estimate(r), 90),
            fontsize=10,
            color=SECONDARY,
            va="top",
        )
        fig.text(
            0.93,
            y,
            f"■ {r['result']}",
            fontsize=10.5,
            color=STATUS.get(r["result"], BASELINE),
            va="top",
        )

    y -= step * 1.6
    flags = [
        f"{name} ({c}): {', '.join(flags)}"
        for kind in ("numeric", "categorical")
        for name, field in report[kind].items()
        for c, found in field["collections"].items()
        if (flags := found.get("flags"))
    ]
    fig.text(
        0.018,
        y,
        "Fields flagged: " + ("; ".join(flags) or "none"),
        fontsize=10,
        color=SECONDARY,
        va="top",
    )
    for fault in report["planted"]:
        y -= step
        mark = "caught" if fault["caught"] else "MISSED"
        line = f"Planted fault, {fault['fault']}: expected {fault['expected']}, found {fault['found']} - {mark}"
        fig.text(0.018, y, line, fontsize=10, color=SECONDARY, va="top")
    fig.savefig(path, facecolor=PAGE)


def _numeric_report(field: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c, values in field["values"].items():
        entry: dict[str, Any] = {
            "n": len(values),
            "videos_without": field["without"][c],
        }
        if len(values):
            quantiles = np.percentile(values, [5, 25, 50, 75, 95])
            entry.update(
                mean=round(float(values.mean()), 4),
                sd=round(float(values.std()), 4),
                quantiles={
                    f"p{q}": round(float(v), 4)
                    for q, v in zip((5, 25, 50, 75, 95), quantiles)
                },
            )
        if c in COLLECTIONS:
            entry["flags"] = _numeric_flags(values, field["span"], field["fixed"])
        out[c] = entry
    return {
        "level": field["level"],
        "unit": field["unit"],
        "range": [round(v, 4) for v in field["span"]],
        "fixed_range": field["fixed"],
        "collections": out,
    }


def _categorical_report(field: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c, weights in field["weights"].items():
        total = sum(weights.values())
        shares = (
            {k: round(weights.get(k, 0.0) / total, 4) for k in field["labels"]}
            if total
            else {}
        )
        out[c] = {
            "total": round(total, 2),
            "shares": shares,
            "flags": _categorical_flags(weights, field["head"]),
        }
    return {"unit": field["unit"], "collections": out}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Distributions of identikit fields per collection, and expectations about them"
    )
    p.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    p.add_argument(
        "--ave",
        default="ave_processed",
        help="the AVE scenes' folder under --processed",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "distributions" / "distributions.json",
        help="the report; its three pages are written beside it",
    )
    p.add_argument(
        "--boot", type=int, default=1000, help="video resamples behind each interval"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    folders = {AVE: args.ave, REELS: REELS, KNOWN: KNOWN}
    videos = {
        c: [
            _video(d)
            for d in sorted((args.processed / folder).iterdir())
            if (d / "identikit.json").exists()
        ]
        for c, folder in folders.items()
        if (args.processed / folder).is_dir()
    }
    if any(len(videos.get(c, [])) < MIN_N for c in COLLECTIONS):
        sys.exit(
            f"needs {MIN_N} processed videos in each of {args.ave}, {REELS} under {args.processed}"
        )

    numeric = _numeric(videos)
    categorical = _categorical(videos)
    table = {e.id: _values(e, videos) for e in EXPECTATIONS}
    results = [
        _judge(e, table[e.id], np.random.default_rng([args.seed, index]), args.boot)
        for index, e in enumerate(EXPECTATIONS)
    ]
    report = {
        "collections": {
            c: {
                "folder": folders[c],
                "videos": len(vs),
                "minutes": round(
                    sum(float(v["kit"]["summary"]["duration_s"]) for v in vs) / 60, 1
                ),
                "barred_videos": sum(1 for v in vs if v["bars"] >= BARS),
                "vertical_videos": sum(1 for v in vs if v["tall"] > 1),
            }
            for c, vs in videos.items()
        },
        "resamples": args.boot,
        "seed": args.seed,
        "floors": {
            "min_n": MIN_N,
            "bars": BARS,
            "constant": CONSTANT,
            "collapse": COLLAPSE,
            "outside": OUTSIDE,
        },
        "expectations": results,
        "planted": _planted(table, results, numeric, categorical, args.seed, args.boot),
        "numeric": {f["name"]: _numeric_report(f) for f in numeric},
        "categorical": {f["name"]: _categorical_report(f) for f in categorical},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pages = [
        args.out.with_suffix(".png"),
        args.out.with_name(f"{args.out.stem}_numeric.png"),
        args.out.with_name(f"{args.out.stem}_categorical.png"),
    ]
    _summary_page(pages[0], report)
    _numeric_page(pages[1], numeric, videos)
    _categorical_page(pages[2], categorical, videos)

    for c, found in report["collections"].items():
        print(f"{c}: {found}")
    for r in results:
        print(f"{r['result']:8} {r['id']:3} {r['claim']:70} {_estimate(r)}")
    for fault in report["planted"]:
        print(
            f"planted: {fault['fault']}: {fault['found']} ({'caught' if fault['caught'] else 'MISSED'})"
        )
    print(f"-> {args.out}, pages {', '.join(p.name for p in pages)}")


if __name__ == "__main__":
    main()
