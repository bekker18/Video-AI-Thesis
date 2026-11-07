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

FACE_VETO = 0.331  # faces under it never link; fitted on VPCD (2026-11-07)
NEAR = 2  # segments a body-only link may reach across cuts: A-B-A dialogue
LARGE_FACE = 16_000  # px; a face this large needs no frontality

# One person boxed twice, set on the reels (HISTORY.md, 2026-10-19)
SAME_FACE_SHARE = 0.8
SAME_FACE_FRAMES = 2
SAME_FACE_IOU = 0.5
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
        if frontality is None:
            return False
        if (
            float(frontality) < cfg.identity_min_front
            and float(crop["area"]) < LARGE_FACE
        ):
            return False
    return True


def _own_face(row: dict[str, Any] | None) -> bool:
    """A face counts for a track only when seated in its body box, not a neighbour's caught inside it."""
    return bool(row and row.get("face") and _seated(row["face"], row["body"]))


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


def _faces_disagree(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["face"] is None or b["face"] is None:
        return False
    return float(a["face"] @ b["face"]) < FACE_VETO


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


def _same_face(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
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
    if _faces_disagree(a, b):
        return None
    share = same / len(both)
    return {
        "modality": "same_face",
        "similarity": round(share, 4),
        "threshold": SAME_FACE_SHARE,
        "margin": round(share - SAME_FACE_SHARE, 4),
        "shared_faces": same,
    }


def _pass_zero(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One person boxed twice: tracks of a segment sharing frames and one seated face join, in descending
    share, only if every pair across the two groups does (complete linkage); ties go by key."""
    found = {
        (i, j): link
        for i, j in itertools.combinations(range(len(tracks)), 2)
        if tracks[i]["keys"][0][0] == tracks[j]["keys"][0][0]
        and tracks[i]["frames"] & tracks[j]["frames"]
        and (link := _same_face(tracks[i], tracks[j])) is not None
    }
    order = sorted(
        found,
        key=lambda p: (
            -found[p]["similarity"],
            tracks[p[0]]["keys"][0],
            tracks[p[1]]["keys"][0],
        ),
    )
    members = {i: [i] for i in range(len(tracks))}
    links: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(tracks))}
    owner = list(range(len(tracks)))
    for i, j in order:
        left, right = owner[i], owner[j]
        if left == right or any(
            (min(x, y), max(x, y)) not in found
            for x in members[left]
            for y in members[right]
        ):
            continue
        members[left] += members[right]
        links[left] += links[right] + [
            {"a": tracks[i]["keys"][0], "b": tracks[j]["keys"][0], **found[(i, j)]}
        ]
        for index in members[right]:
            owner[index] = left
        del members[right], links[right]
    merged = [_combine(tracks, group, links[root]) for root, group in members.items()]
    return sorted(merged, key=lambda u: u["keys"][0])


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


def _links(
    f: Array, b: Array, face_min: float, body_min: float, body_ok: Array | None
) -> tuple[Array, Array]:
    """The link test on arrays of similarities: the modality clearing its threshold by more, -inf where none does
    or the faces disagree; and whether that modality is the body."""
    with np.errstate(invalid="ignore"):
        has_f, has_b = ~np.isnan(f), ~np.isnan(b)
        ok_f = has_f & (f >= face_min)
        ok_b = has_b & (b >= body_min)
        if body_ok is not None:
            ok_b &= body_ok
        by_body = np.where(ok_b, b - body_min, -np.inf) > np.where(
            ok_f, f - face_min, -np.inf
        )
        sim = np.where(
            (ok_f | ok_b) & ~(has_f & (f < FACE_VETO)), np.where(by_body, b, f), -np.inf
        )
    return sim, by_body


def _centres(groups: list[list[int]], vectors: Array) -> tuple[Array, Array]:
    has = ~np.isnan(vectors[:, 0])
    clean = np.where(has[:, None], vectors, 0.0)
    sums = np.stack([clean[g].sum(axis=0) for g in groups])
    return sums, np.array([int(has[g].sum()) for g in groups])


def _unit(sums: Array, counts: Array) -> Array:
    out: Array = sums / np.maximum(np.linalg.norm(sums, axis=-1, keepdims=True), 1e-12)
    out[counts == 0] = np.nan
    return out


def _average(
    groups: list[list[int]],
    found: list[list[dict[str, Any]]],
    space: dict[str, Any],
    offered: Array,
    near: Array | None,
    face_min: float,
    body_min: float,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    """Groups merge by the mean of their tracks' vectors, the best link first; a merged group is compared anew."""
    fs, fc = _centres(groups, space["face"])
    bs, bc = _centres(groups, space["body"])
    f, b = _unit(fs, fc), _unit(bs, bc)
    allowed = offered.copy()
    score, _ = _links(f @ f.T, b @ b.T, face_min, body_min, near)
    score[~allowed] = -np.inf
    groups, found = [list(g) for g in groups], [list(x) for x in found]
    alive = np.ones(len(groups), dtype=bool)
    keys = space["keys"]
    while len(groups) > 1:
        a, c = divmod(int(np.argmax(score)), len(groups))
        if not np.isfinite(score[a, c]):
            break
        _, by_body = _links(
            f[c : c + 1] @ f[a], b[c : c + 1] @ b[a], face_min, body_min, None
        )
        threshold = body_min if by_body[0] else face_min
        found[a] += found[c] + [
            {
                "a": min(keys[i] for i in groups[a]),
                "b": min(keys[i] for i in groups[c]),
                "modality": "body" if by_body[0] else "face",
                "similarity": round(float(score[a, c]), 4),
                "threshold": threshold,
                "margin": round(float(score[a, c]) - threshold, 4),
            }
        ]
        groups[a] += groups[c]
        alive[c] = False
        for sums, counts in ((fs, fc), (bs, bc)):
            sums[a] += sums[c]
            counts[a] += counts[c]
        allowed[a] &= allowed[c]
        allowed[:, a] = allowed[a]
        allowed[c] = allowed[:, c] = False
        if near is not None:
            near[a] |= near[c]
            near[:, a] = near[a]
        f[a] = _unit(fs[a : a + 1], fc[a : a + 1])[0]
        b[a] = _unit(bs[a : a + 1], bc[a : a + 1])[0]
        f[c] = b[c] = np.nan
        row, _ = _links(
            f @ f[a], b @ b[a], face_min, body_min, None if near is None else near[a]
        )
        row[~allowed[a]] = -np.inf
        score[a] = score[:, a] = row
        score[c] = score[:, c] = -np.inf
    keep = np.flatnonzero(alive)
    return [groups[k] for k in keep], [found[k] for k in keep]


def _pass(
    groups: list[list[int]],
    found: list[list[dict[str, Any]]],
    space: dict[str, Any],
    scope: str,
    face_min: float,
    body_min: float,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    """Inside a segment, or across segments: groups sharing a frame, or a segment across cuts, never merge."""
    if len(groups) < 2:
        return groups, found
    segments = [{int(space["segment"][i]) for i in g} for g in groups]
    offered = np.zeros((len(groups), len(groups)), dtype=bool)
    near = offered.copy() if scope == "across" else None
    for i, j in itertools.combinations(range(len(groups)), 2):
        fits = (
            segments[i] == segments[j]
            if scope == "within"
            else segments[i].isdisjoint(segments[j])
        )
        if fits and not space["blocked"][np.ix_(groups[i], groups[j])].any():
            offered[i, j] = offered[j, i] = True
        if near is not None:
            gap = min(abs(s - t) for s in segments[i] for t in segments[j])
            near[i, j] = near[j, i] = gap <= NEAR
    return _average(groups, found, space, offered, near, face_min, body_min)


def _two_passes(
    members: list[int], space: dict[str, Any], cfg: Config
) -> tuple[list[list[int]], list[list[dict[str, Any]]], int, int]:
    groups: list[list[int]] = [[i] for i in members]
    found: list[list[dict[str, Any]]] = [[] for _ in members]
    within, found = _pass(
        groups,
        found,
        space,
        "within",
        cfg.identity_face_within,
        cfg.identity_body_within,
    )
    across, found = _pass(
        within,
        found,
        space,
        "across",
        cfg.identity_face_across,
        cfg.identity_body_across,
    )
    return across, found, len(groups) - len(within), len(within) - len(across)


def _attach(
    groups: list[list[int]],
    found: list[list[dict[str, Any]]],
    faceless: list[int],
    space: dict[str, Any],
    cfg: Config,
) -> list[int]:
    """Each face-less track joins the group of the closest body with a face in its own or a neighbouring segment,
    if it clears the body threshold and shares no frame with the group; returns the rest."""
    owner = {i: g for g, members in enumerate(groups) for i in members}
    bodies, segment = space["body"], space["segment"]
    faced = [i for i in owner if not np.isnan(bodies[i, 0])]
    rest = []
    for u in faceless:
        near = [i for i in faced if abs(int(segment[i]) - int(segment[u])) <= NEAR]
        sims = bodies[near] @ bodies[u] if near else np.zeros(0)
        for k in np.argsort(-sims, kind="stable"):
            i, g = near[k], owner[near[k]]
            floor = (
                cfg.identity_body_within
                if segment[i] == segment[u]
                else cfg.identity_body_across
            )
            if sims[k] >= floor and not space["blocked"][u, groups[g]].any():
                groups[g].append(u)
                found[g].append(
                    {
                        "a": space["keys"][i],
                        "b": space["keys"][u],
                        "modality": "body",
                        "similarity": round(float(sims[k]), 4),
                        "threshold": floor,
                        "margin": round(float(sims[k]) - floor, 4),
                    }
                )
                break
        else:
            rest.append(u)
    return rest


def _matrix(units: list[dict[str, Any]], kind: str) -> Array:
    width = next((len(u[kind]) for u in units if u[kind] is not None), 1)
    out: Array = np.full((len(units), width), np.nan)
    for i, unit in enumerate(units):
        if unit[kind] is not None:
            out[i] = unit[kind]
    return out


def _shared(units: list[dict[str, Any]]) -> Array:
    """Which units share a frame."""
    spans = [(min(u["frames"]), max(u["frames"])) for u in units]
    order = sorted(range(len(units)), key=lambda i: spans[i][0])
    out = np.zeros((len(units), len(units)), dtype=bool)
    for n, i in enumerate(order):
        for j in order[n + 1 :]:
            if spans[j][0] > spans[i][1]:
                break
            if units[i]["frames"] & units[j]["frames"]:
                out[i, j] = out[j, i] = True
    return out


def _combine(
    units: list[dict[str, Any]], members: list[int], found: list[dict[str, Any]]
) -> dict[str, Any]:
    frames: set[int] = set()
    for i in members:
        frames |= units[i]["frames"]
    return {
        "keys": sorted(k for i in members for k in units[i]["keys"]),
        "frames": frames,
        "links": [link for i in members for link in units[i]["links"]] + found,
        "face": _mean([units[i]["face"] for i in members]),
        "body": _mean([units[i]["body"] for i in members]),
        "face_crops": sum(units[i]["face_crops"] for i in members),
        "body_crops": sum(units[i]["body_crops"] for i in members),
    }


def _resolve(
    units: list[dict[str, Any]], cfg: Config
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Faces first, by group means, inside a segment and then across segments; face-less units join a face's group
    by a nearby body, and the rest group by body alone (2026-11-07)."""
    space: dict[str, Any] = {
        "keys": [tuple(u["keys"][0]) for u in units],
        "segment": np.array([int(u["keys"][0][0]) for u in units], dtype=np.int64),
        "blocked": _shared(units),
        "face": _matrix(units, "face"),
        "body": _matrix(units, "body"),
    }
    faced = [i for i, u in enumerate(units) if u["face"] is not None]
    faceless = [i for i, u in enumerate(units) if u["face"] is None]
    groups, found, within, across = _two_passes(
        faced, space | {"body": np.full_like(space["body"], np.nan)}, cfg
    )
    rest = _attach(groups, found, faceless, space, cfg)
    more, more_found, within_more, across_more = _two_passes(
        rest, space | {"face": np.full_like(space["face"], np.nan)}, cfg
    )
    final = [_combine(units, g, x) for g, x in zip(groups + more, found + more_found)]
    return final, within + within_more, across + across_more, len(faceless) - len(rest)


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
            observed = {
                int(r["frame"]): r for r in track["frames"] if not r["interpolated"]
            }
            for kind, field, table in (
                ("face", "embedding_rows", faces),
                ("body", "body_rows", bodies),
            ):
                keep = {
                    int(c["frame"])
                    for c in track["crops"][kind]
                    if _usable(c, cfg, kind)
                    and (kind == "body" or _own_face(observed.get(int(c["frame"]))))
                }
                rows = list(attrs.get(field, []))
                descriptors[kind] = _descriptor(rows, keep, table)
                counts[kind] = len([r for r in rows if int(r["frame"]) in keep])

            body_boxes[key] = {int(r["frame"]): r["body"] for r in track["frames"]}
            tracks.append(
                {
                    "keys": [list(key)],
                    "frames": set(body_boxes[key]),
                    "observed": observed,
                    "links": [],
                    "face": descriptors["face"],
                    "body": descriptors["body"],
                    "face_crops": counts["face"],
                    "body_crops": counts["body"],
                }
            )

    merged = _pass_zero(tracks)
    same_face_merges = len(tracks) - len(merged)
    units = [u for u in merged if u["face"] is not None or u["body"] is not None]
    abstained = [u for u in merged if u["face"] is None and u["body"] is None]

    final, within_merges, across_merges, attached_merges = _resolve(units, cfg)

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
        "attached_merges": attached_merges,
        "face_descriptors": sum(1 for t in tracks if t["face"] is not None),
        "body_descriptors": sum(1 for t in tracks if t["body"] is not None),
        "thresholds": {
            "face_within": cfg.identity_face_within,
            "face_across": cfg.identity_face_across,
            "body_within": cfg.identity_body_within,
            "body_across": cfg.identity_body_across,
            "min_area": cfg.identity_min_area,
            "min_frontality": cfg.identity_min_front,
            "face_veto": FACE_VETO,
            "joining": "group means, faces first",
            "body_near_segments": NEAR,
            "large_face": LARGE_FACE,
            "face_seated_in_body": True,
            "same_face": {
                "share": SAME_FACE_SHARE,
                "frames": SAME_FACE_FRAMES,
                "iou": SAME_FACE_IOU,
                "seated_side": SEATED_SIDE,
                "seated_top": SEATED_TOP,
            },
        },
        "source": source,
        "note": (
            "person ids are video-scoped; a track with no usable crop is left unlinked "
            "rather than assigned to the nearest cluster; two tracks whose faces score under "
            "face_veto are never linked; two tracks sharing a frame are one "
            "person only when boxed twice around one face, and then a track's `yields` are "
            "the frames the other describes"
        ),
        "persons": persons,
    }
    (cfg.json_dir / "identity.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {k: v for k, v in meta.items() if k != "persons"}
