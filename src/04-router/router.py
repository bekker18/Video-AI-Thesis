"""04-router: deterministic mapping from 03-profiler's presence flags to the active expert set.

Every expert records why it fired or did not, so any routing decision can be replayed and
explained from routing.json alone. A shot with no people skips the whole human pipeline,
including detection and tracking, while still receiving the plastic experts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Config

# Scene-global and person-independent: these run on every shot, always.
PLASTIC = ("depth", "edges", "scene", "tags")

# Object classes whose downstream expert needs a stable identity across frames. Anything
# outside this set still triggers object_masks, just not detection and tracking.
IDENTITY_CLASSES = frozenset({
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "frisbee", "skateboard", "skis", "snowboard", "sports ball", "surfboard",
    "tennis racket", "kite", "baseball bat",
})


@dataclass(frozen=True)
class Facts:
    """What the router is allowed to look at, resolved once per shot."""
    person: bool
    persons: int
    text: bool
    object: bool
    objects: int
    identity: tuple[str, ...]


def _read_profile(out_root: Path) -> dict[str, Any]:
    path = out_root / "profile.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}; run 03-profiler first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _facts(shot: dict[str, Any], floor: float) -> Facts:
    """`floor` is the optional soft gate: a flag must also be seen in this fraction of the
    shot's keyframes. At 0 it is off and any single sighting counts."""
    flags = shot["flags"]
    counts = shot["counts"]
    agreement = shot.get("agreement", {})
    per_class = agreement.get("objects", {})

    def gated(name: str) -> bool:
        return bool(flags.get(name)) and float(agreement.get(name, 1.0)) >= floor

    classes = [c for c in counts.get("objects", {}) if float(per_class.get(c, 1.0)) >= floor]
    person = gated("person")
    return Facts(
        person=person,
        persons=int(counts.get("person_max", 0)) if person else 0,
        text=gated("text"),
        object=gated("object") and bool(classes),
        objects=len(classes),
        identity=tuple(sorted(c for c in classes if c in IDENTITY_CLASSES)),
    )


def _decisions(facts: Facts) -> list[dict[str, Any]]:
    """The routing function. Order is fixed so two runs are byte-identical."""
    rules: list[tuple[str, str, str, bool, str]] = [
        (name, "plastic", "always active", True, "scene-global, person-independent")
        for name in PLASTIC
    ]

    identity = ", ".join(facts.identity) or "none"
    rules += [
        ("detection_tracking", "conditional",
         "a person is present, or an object that needs a stable identity downstream",
         facts.person or bool(facts.identity),
         f"person={facts.person}, identity objects: {identity}"),

        ("face", "conditional", "a person is present",
         facts.person, f"person={facts.person}, person_max={facts.persons}"),

        ("body", "conditional", "a person is present",
         facts.person, f"person={facts.person}, person_max={facts.persons}"),

        ("pairwise_relations", "conditional",
         "two or more people co-occur in at least one profiled keyframe",
         facts.persons >= 2, f"person_max={facts.persons}"),

        ("ocr", "conditional", "text is present",
         facts.text, f"text={facts.text}"),

        ("object_masks", "conditional", "objects are present",
         facts.object, f"object={facts.object}, classes={facts.objects}"),
    ]

    return [
        {"expert": name, "kind": kind, "fired": fired, "rule": rule, "evidence": evidence}
        for name, kind, rule, fired, evidence in rules
    ]


def run(cfg: Config) -> dict[str, Any]:
    profile = _read_profile(cfg.out_root)

    shots: list[dict[str, Any]] = []
    fired_counts: dict[str, int] = {}
    human_skipped = 0

    for shot in profile["shots"]:
        facts = _facts(shot, cfg.router_agreement)
        decisions = _decisions(facts)
        active = [d["expert"] for d in decisions if d["fired"]]

        for name in active:
            fired_counts[name] = fired_counts.get(name, 0) + 1
        if not facts.person:
            human_skipped += 1

        shots.append({
            "index": shot["index"],
            "start_frame": shot["start_frame"],
            "end_frame": shot["end_frame"],
            "start_time": shot["start_time"],
            "end_time": shot["end_time"],
            "experts": active,
            "skipped": [d["expert"] for d in decisions if not d["fired"]],
            "decisions": decisions,
        })

    total = len(shots)
    meta = {
        "video": profile["video"],
        "shot_count": total,
        "agreement_floor": cfg.router_agreement,
        "experts": sorted({e for s in shots for e in s["experts"]}),
        "fired_shots": dict(sorted(fired_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "fired_share": {k: round(v / total, 3) for k, v in
                        sorted(fired_counts.items(), key=lambda kv: (-kv[1], kv[0]))} if total else {},
        "human_pipeline_skipped": human_skipped,
        "shots": shots,
    }
    (cfg.out_root / "routing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {k: v for k, v in meta.items() if k != "shots"}
