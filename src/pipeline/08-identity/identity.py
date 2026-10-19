"""08-identity: 06's segment-local tracks into global person ids by face and body appearance."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from config import Config

STAGE = "08-identity"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

Key: TypeAlias = tuple[int, int]  # (segment, track_id)

# One person boxed twice, set on the reels (STATE.md, One person tracked twice)
SAME_FACE_SHARE = 0.8
SAME_FACE_FRAMES = 2
SAME_FACE_IOU = 0.5
SAME_FACE_VETO = 0.26  # same-person face cosine, 1st percentile
SEATED_SIDE = 0.25  # face centre off the body midline, in box widths
SEATED_TOP = 0.5  # face centre in this top share of the body box


def _read(json_dir: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = json_dir / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _load(embeddings_dir: Path, name: str) -> Array:
    path = embeddings_dir / name
    if not path.exists():
        return np.zeros((0, 0), dtype=np.float32)
    out: Array = np.load(path)
    return out


def _usable(crop: dict[str, Any], cfg: Config, kind: str) -> bool:
    """Absolute floors, not 06's per-track `score`; sharpness has no clip-independent scale."""
    if float(crop["area"]) < cfg.identity_min_area:
        return False
    if kind == "face":
        frontality = crop["frontality"]
        if frontality is None or float(frontality) < cfg.identity_min_front:
            return False
    return True


def _descriptor(
    rows: list[dict[str, Any]], keep: set[int], table: Array
) -> Array | None:
    """Mean of the crop vectors that clear the floors, renormalised."""
    picked = [int(r["row"]) for r in rows if int(r["frame"]) in keep]
    if not picked or not table.size:
        return None
    vector = table[picked].mean(axis=0)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    out: Array = vector / norm
    return out


def _link(
    a: dict[str, Any], b: dict[str, Any], face_min: float, body_min: float
) -> dict[str, Any] | None:
    """The best modality clearing its own threshold; one only a track has is no evidence."""
    best: dict[str, Any] | None = None
    for kind, floor in (("face", face_min), ("body", body_min)):
        left, right = a[kind], b[kind]
        if left is None or right is None:
            continue
        similarity = float(left @ right)
        if similarity < floor:
            continue
        if best is None or similarity - floor > float(best["margin"]):
            best = {
                "modality": kind,
                "similarity": round(similarity, 4),
                "threshold": floor,
                "margin": round(similarity - floor, 4),
            }
    return best


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(a: list[int], b: list[int]) -> float:
    wide = min(a[2], b[2]) - max(a[0], b[0])
    tall = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or tall <= 0:
        return 0.0
    overlap = wide * tall
    return overlap / (_area(a) + _area(b) - overlap)


def _seated(face: list[int], body: list[int]) -> bool:
    """A body's own face: near the top of its box and its midline."""
    width, height = body[2] - body[0], body[3] - body[1]
    if width <= 0 or height <= 0:
        return False
    side = abs((face[0] + face[2]) / 2 - (body[0] + body[2]) / 2) / width
    top = ((face[1] + face[3]) / 2 - body[1]) / height
    return side <= SEATED_SIDE and top <= SEATED_TOP


def _same_face(
    a: dict[str, Any], b: dict[str, Any], face_min: float, body_min: float
) -> dict[str, Any] | None:
    """One face box seated in both bodies, in most frames both have a face; disagreeing faces veto."""
    rows_a, rows_b = a["observed"], b["observed"]
    both = [
        f
        for f in rows_a.keys() & rows_b.keys()
        if rows_a[f]["face"] and rows_b[f]["face"]
    ]
    same = sum(
        _iou(rows_a[f]["face"], rows_b[f]["face"]) > SAME_FACE_IOU
        and _seated(rows_a[f]["face"], rows_a[f]["body"])
        and _seated(rows_b[f]["face"], rows_b[f]["body"])
        for f in both
    )
    if same < SAME_FACE_FRAMES or same < SAME_FACE_SHARE * len(both):
        return None
    if (
        a["face"] is not None
        and b["face"] is not None
        and float(a["face"] @ b["face"]) < SAME_FACE_VETO
    ):
        return None
    share = same / len(both)
    return {
        "modality": "same_face",
        "similarity": round(share, 4),
        "threshold": SAME_FACE_SHARE,
        "margin": round(share - SAME_FACE_SHARE, 4),
        "shared_faces": same,
    }


def _cluster(
    units: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    face_min: float,
    body_min: float,
    link: Any = None,
) -> list[dict[str, Any]]:
    """Complete linkage in descending similarity; `link` replaces the appearance rule."""
    link = link or _link
    scored: list[tuple[float, int, int, dict[str, Any]]] = []
    for i, j in pairs:
        found = link(units[i], units[j], face_min, body_min)
        if found is not None:
            scored.append((float(found["similarity"]), i, j, found))
    # ties on keys, so two runs agree
    scored.sort(key=lambda s: (-s[0], units[s[1]]["keys"][0], units[s[2]]["keys"][0]))

    allowed = {(i, j) for i, j in pairs} | {(j, i) for i, j in pairs}
    members = {i: [i] for i in range(len(units))}
    owner = list(range(len(units)))
    links: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(units))}

    for _, i, j, found in scored:
        left, right = owner[i], owner[j]
        if left == right:
            continue
        joined = [(x, y) for x in members[left] for y in members[right]]
        if any((x, y) not in allowed for x, y in joined):
            continue
        if any(link(units[x], units[y], face_min, body_min) is None for x, y in joined):
            continue
        members[left] += members[right]
        links[left] += links[right] + [
            {
                "a": units[i]["keys"][0],
                "b": units[j]["keys"][0],
                **found,
            }
        ]
        for index in members[right]:
            owner[index] = left
        del members[right], links[right]

    merged: list[dict[str, Any]] = []
    for root, group in members.items():
        keys = sorted(k for index in group for k in units[index]["keys"])
        frames: set[int] = set()
        for index in group:
            frames |= units[index]["frames"]
        merged.append(
            {
                "keys": keys,
                "frames": frames,
                "links": [link for index in group for link in units[index]["links"]]
                + links[root],
                "face": _mean([units[i]["face"] for i in group]),
                "body": _mean([units[i]["body"] for i in group]),
                "face_crops": sum(units[i]["face_crops"] for i in group),
                "body_crops": sum(units[i]["body_crops"] for i in group),
            }
        )
    merged.sort(key=lambda u: u["keys"][0])
    return merged


def _yields(
    keys: list[Key], bodies: dict[Key, dict[int, list[int]]]
) -> dict[Key, list[int]]:
    """Frames each track leaves to another of the person: the larger body box keeps a frame."""
    owner: dict[int, Key] = {}
    for key in keys:
        for frame, body in bodies[key].items():
            held = owner.get(frame)
            if held is None or _area(body) > _area(bodies[held][frame]):
                owner[frame] = key
    return {key: sorted(f for f in bodies[key] if owner[f] != key) for key in keys}


def _mean(vectors: list[Array | None]) -> Array | None:
    present = [v for v in vectors if v is not None]
    if not present:
        return None
    stacked = np.stack(present).mean(axis=0)
    norm = float(np.linalg.norm(stacked))
    if norm < 1e-6:
        return None
    out: Array = stacked / norm
    return out


def run(cfg: Config) -> dict[str, Any]:
    segmentation = _read(cfg.json_dir, "segmentation.json", "02-segmentation")
    tracks_meta = _read(cfg.json_dir, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.json_dir, "conditional.json", "07-conditional-experts")
    faces = _load(cfg.embeddings_dir, "face_embeddings.npy")
    bodies = _load(cfg.embeddings_dir, "body_embeddings.npy")

    rows_by_track: dict[Key, dict[str, Any]] = {}
    for shot in conditional["shots"]:
        for track in shot["tracks"]:
            rows_by_track[(int(shot["index"]), int(track["track_id"]))] = track.get(
                "attributes", {}
            )

    tracks: list[dict[str, Any]] = []
    body_boxes: dict[Key, dict[int, list[int]]] = {}
    for shot in tracks_meta["shots"]:
        if not shot["tracked"]:
            continue
        index = int(shot["index"])
        for track in shot["tracks"]:
            key = (index, int(track["track_id"]))
            attrs = rows_by_track.get(key, {})
            descriptors: dict[str, Array | None] = {}
            counts: dict[str, int] = {}
            for kind, field, table in (
                ("face", "embedding_rows", faces),
                ("body", "body_rows", bodies),
            ):
                keep = {
                    int(c["frame"])
                    for c in track["crops"][kind]
                    if _usable(c, cfg, kind)
                }
                rows = list(attrs.get(field, []))
                descriptors[kind] = _descriptor(rows, keep, table)
                counts[kind] = len([r for r in rows if int(r["frame"]) in keep])

            body_boxes[key] = {int(r["frame"]): r["body"] for r in track["frames"]}
            tracks.append(
                {
                    "keys": [list(key)],
                    "frames": set(body_boxes[key]),
                    "observed": {
                        int(r["frame"]): r
                        for r in track["frames"]
                        if not r["interpolated"]
                    },
                    "links": [],
                    "face": descriptors["face"],
                    "body": descriptors["body"],
                    "face_crops": counts["face"],
                    "body_crops": counts["body"],
                }
            )

    # Pass zero: one person boxed twice, which shares frames the passes below never join
    co_present = [
        (i, j)
        for i, j in itertools.combinations(range(len(tracks)), 2)
        if tracks[i]["keys"][0][0] == tracks[j]["keys"][0][0]
        and tracks[i]["frames"] & tracks[j]["frames"]
    ]
    merged = _cluster(tracks, co_present, 0.0, 0.0, link=_same_face)
    same_face_merges = len(tracks) - len(merged)
    units = [u for u in merged if u["face"] is not None or u["body"] is not None]
    abstained = [u for u in merged if u["face"] is None and u["body"] is None]

    # Pass one, inside a segment; tracks sharing a frame are two people
    by_segment: dict[int, list[int]] = {}
    for i, unit in enumerate(units):
        by_segment.setdefault(int(unit["keys"][0][0]), []).append(i)
    within_pairs = [
        (i, j)
        for group in by_segment.values()
        for i, j in itertools.combinations(group, 2)
        if not (units[i]["frames"] & units[j]["frames"])
    ]
    after_within = _cluster(
        units, within_pairs, cfg.identity_face_within, cfg.identity_body_within
    )
    within_merges = len(units) - len(after_within)

    # Pass two, across segments only: a merge pass one refused is not re-offered
    across_pairs = [
        (i, j)
        for i, j in itertools.combinations(range(len(after_within)), 2)
        if {k[0] for k in after_within[i]["keys"]}.isdisjoint(
            {k[0] for k in after_within[j]["keys"]}
        )
        and not (after_within[i]["frames"] & after_within[j]["frames"])
    ]
    final = _cluster(
        after_within, across_pairs, cfg.identity_face_across, cfg.identity_body_across
    )
    across_merges = len(after_within) - len(final)

    # abstained tracks stay singletons
    everyone = final + abstained
    everyone.sort(key=lambda u: (min(k[0] for k in u["keys"]), u["keys"][0][1]))

    persons: list[dict[str, Any]] = []
    for person_id, unit in enumerate(everyone, start=1):
        keys = [[int(a), int(b)] for a, b in unit["keys"]]
        # 09 and 10 skip a track's yielded frames
        yields = _yields([(a, b) for a, b in keys], body_boxes)
        persons.append(
            {
                "person_id": person_id,
                "tracks": [
                    {"segment": a, "track_id": b}
                    | ({"yields": yields[(a, b)]} if yields[(a, b)] else {})
                    for a, b in keys
                ],
                "segments": sorted({a for a, _ in keys}),
                "first_frame": min(unit["frames"]) if unit["frames"] else None,
                "last_frame": max(unit["frames"]) if unit["frames"] else None,
                "frame_count": len(unit["frames"]),
                "linked": len(keys) > 1,
                "abstained": unit["face"] is None and unit["body"] is None,
                "descriptors": {
                    "face_crops": unit["face_crops"],
                    "body_crops": unit["body_crops"],
                },
                # each merge keeps its modality and margin
                "links": [
                    {
                        **link,
                        "a": {"segment": link["a"][0], "track_id": link["a"][1]},
                        "b": {"segment": link["b"][0], "track_id": link["b"][1]},
                    }
                    for link in unit["links"]
                ],
            }
        )

    source = {
        "detector": tracks_meta["detector"],
        "tracker": tracks_meta["tracker"],
        "det_stride": tracks_meta["det_stride"],
        "thresholds": tracks_meta["thresholds"],
        "min_track": cfg.min_track,
        "crops": cfg.crops,
    }
    meta = {
        "video": segmentation["video"],
        "track_count": len(tracks),
        "person_count": len(persons),
        "linked_persons": sum(1 for p in persons if p["linked"]),
        "abstained_tracks": sum(
            1 for t in tracks if t["face"] is None and t["body"] is None
        ),
        "same_face_merges": same_face_merges,
        "within_merges": within_merges,
        "across_merges": across_merges,
        "face_descriptors": sum(1 for t in tracks if t["face"] is not None),
        "body_descriptors": sum(1 for t in tracks if t["body"] is not None),
        "thresholds": {
            "face_within": cfg.identity_face_within,
            "face_across": cfg.identity_face_across,
            "body_within": cfg.identity_body_within,
            "body_across": cfg.identity_body_across,
            "min_area": cfg.identity_min_area,
            "min_frontality": cfg.identity_min_front,
            "same_face": {
                "share": SAME_FACE_SHARE,
                "frames": SAME_FACE_FRAMES,
                "iou": SAME_FACE_IOU,
                "veto": SAME_FACE_VETO,
                "seated_side": SEATED_SIDE,
                "seated_top": SEATED_TOP,
            },
        },
        "source": source,
        "note": (
            "person ids are video-scoped; a track with no usable crop is left unlinked "
            "rather than assigned to the nearest cluster; two tracks sharing a frame are one "
            "person only when boxed twice around one face, and then a track's `yields` are "
            "the frames the other describes"
        ),
        "persons": persons,
    }
    (cfg.json_dir / "identity.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "persons"}
