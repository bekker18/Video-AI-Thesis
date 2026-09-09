"""08-consolidation: resolve segment-local tracks into global person ids.

Identity is resolved at three widening scopes along the pipeline: presence flags at 03,
segment-local track ids at 06, and global person ids here. This is the main resolver,
not a repair step - within-segment association is motion-only and survives about two seconds,
so a single busy segment fragments one person into many tracks.
Most of the identity in a video is established here.

A pure function like 04, 09 and 10: no models, no GPU, byte-identical across runs.
07 does the embedding, this does the clustering, which is what makes threshold tuning cost a second.

Two passes, because the priors differ. Inside a segment lighting and pose are stable,
so the threshold can be strict and the only merges available are the ones motion could not make.
Across a cut nothing is stable and the threshold must be looser. One pass would conflate them.

Abstention is permitted and is the point: a track whose crops are all blurred,
tiny or non-frontal is left unlinked rather than forced into the nearest cluster.
A wrong link merges two people's demographics, emotion distributions and gaze statistics into one fictitious person,
and nothing downstream can detect that it happened.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from config import Config

STAGE = "08-consolidation"

Array: TypeAlias = np.ndarray[Any, np.dtype[Any]]

Key: TypeAlias = tuple[int, int]  # (segment, track_id)


def _read(out_root: Path, name: str, produced_by: str) -> dict[str, Any]:
    path = out_root / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run {produced_by} first")
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _load(out_root: Path, name: str) -> Array:
    """A modality with no rows is normal - cat1.mp4 yields no face at all."""
    path = out_root / name
    if not path.exists():
        return np.zeros((0, 0), dtype=np.float32)
    out: Array = np.load(path)
    return out


def _usable(crop: dict[str, Any], cfg: Config, kind: str) -> bool:
    """Absolute floors, not 06's `score`, which is normalised within its own track and so
    rates the best of five blurred non-faces highly.

    Sharpness is deliberately not one of them. It is Laplacian variance, and its scale is
    a property of the footage rather than of the crop: the median across these clips runs from
    10 on thanos.mp4 to 1139 on messi.mp4, so any absolute floor that catches a blurred crop
    in one clip discards every crop in another. Area is pixels and frontality is a normalised
    ratio; both mean the same thing in every clip.
    """
    if float(crop["area"]) < cfg.consol_min_area:
        return False
    if kind == "face":
        frontality = crop["frontality"]
        if frontality is None or float(frontality) < cfg.consol_min_front:
            return False
    return True


def _descriptor(
    rows: list[dict[str, Any]], keep: set[int], table: Array
) -> Array | None:
    """Mean of a track's surviving crop vectors, renormalised. One bad crop in the mean is
    exactly what produces a wrong link, so gated crops are dropped before averaging."""
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
    """The best modality that clears its own threshold. Two tracks sharing no modality carry
    no evidence about each other and are never linked on the strength of the other's."""
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


def _cluster(
    units: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    face_min: float,
    body_min: float,
) -> list[dict[str, Any]]:
    """Complete linkage: a unit joins a cluster only if it clears the threshold against every member.
    Single linkage would chain distinct people together through one weak intermediate,
    and the frame-disjointness constraint would stop applying to the cluster as a whole.

    Merging in descending similarity is the standard agglomerative order,
    and it makes the result independent of the order the tracks happen to arrive in.
    """
    scored: list[tuple[float, int, int, dict[str, Any]]] = []
    for i, j in pairs:
        found = _link(units[i], units[j], face_min, body_min)
        if found is not None:
            scored.append((float(found["similarity"]), i, j, found))
    # Ties broken on the unit keys, so two runs agree.
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
        if any(
            _link(units[x], units[y], face_min, body_min) is None for x, y in joined
        ):
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
    segmentation = _read(cfg.out_root, "segmentation.json", "02-segmentation")
    tracks_meta = _read(cfg.out_root, "tracks.json", "06-detection-tracking")
    conditional = _read(cfg.out_root, "conditional.json", "07-conditional-experts")
    faces = _load(cfg.out_root, "face_embeddings.npy")
    bodies = _load(cfg.out_root, "body_embeddings.npy")

    rows_by_track: dict[Key, dict[str, Any]] = {}
    for shot in conditional["shots"]:
        for track in shot["tracks"]:
            rows_by_track[(int(shot["index"]), int(track["track_id"]))] = track.get(
                "attributes", {}
            )

    # One unit per track, carrying its frame set, its two descriptors, and why it has none.
    units: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
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

            unit = {
                "keys": [list(key)],
                "frames": {int(r["frame"]) for r in track["frames"]},
                "links": [],
                "face": descriptors["face"],
                "body": descriptors["body"],
                "face_crops": counts["face"],
                "body_crops": counts["body"],
            }
            if descriptors["face"] is None and descriptors["body"] is None:
                abstained.append(unit)
            else:
                units.append(unit)

    # Pass one, inside a segment.
    # Two tracks sharing a frame cannot be one person, whatever they look like.
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
        units, within_pairs, cfg.consol_face_within, cfg.consol_body_within
    )
    within_merges = len(units) - len(after_within)

    # Pass two, across segments only.
    # Same-segment pairs are deliberately not offered a second chance at the looser threshold:
    # complete linkage then also refuses to merge them transitively through a cross-segment link,
    # so a merge the strict threshold rejected cannot be laundered by the loose one.
    across_pairs = [
        (i, j)
        for i, j in itertools.combinations(range(len(after_within)), 2)
        if {k[0] for k in after_within[i]["keys"]}.isdisjoint(
            {k[0] for k in after_within[j]["keys"]}
        )
        and not (after_within[i]["frames"] & after_within[j]["frames"])
    ]
    final = _cluster(
        after_within, across_pairs, cfg.consol_face_across, cfg.consol_body_across
    )
    across_merges = len(after_within) - len(final)

    # Abstained tracks are people too - the identikit must tolerate singletons - they are simply never candidates for a link, and say so.
    everyone = final + abstained
    everyone.sort(key=lambda u: (min(k[0] for k in u["keys"]), u["keys"][0][1]))

    persons: list[dict[str, Any]] = []
    for person_id, unit in enumerate(everyone, start=1):
        keys = [[int(a), int(b)] for a, b in unit["keys"]]
        persons.append(
            {
                "person_id": person_id,
                "tracks": [{"segment": a, "track_id": b} for a, b in keys],
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
                # Every merge carries the modality that made it and how far it cleared its threshold,
                # so a person can be argued with from this file alone.
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
        "track_count": len(units) + len(abstained),
        "person_count": len(persons),
        "linked_persons": sum(1 for p in persons if p["linked"]),
        "abstained_tracks": len(abstained),
        "within_merges": within_merges,
        "across_merges": across_merges,
        "face_descriptors": sum(1 for u in units if u["face"] is not None),
        "body_descriptors": sum(1 for u in units if u["body"] is not None),
        "thresholds": {
            "face_within": cfg.consol_face_within,
            "face_across": cfg.consol_face_across,
            "body_within": cfg.consol_body_within,
            "body_across": cfg.consol_body_across,
            "min_area": cfg.consol_min_area,
            "min_frontality": cfg.consol_min_front,
        },
        "source": source,
        "note": (
            "person ids are video-scoped; a track with no usable crop is left unlinked "
            "rather than assigned to the nearest cluster"
        ),
        "persons": persons,
    }
    (cfg.out_root / "consolidated.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "persons"}
