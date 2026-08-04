"""Association of per-stream tracks into stable identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

# Fraction of the face box that must fall inside the body box to count as a containment vote. Faces sit well inside a body box, so this can be strict.
DEFAULT_CONTAINMENT = 0.6
# Minimum number of voting frames before a face is bound to a body, which rejects pairings built on one or two frames of coincidental overlap.
DEFAULT_MIN_VOTES = 3
# Cosine similarity above which two body tracks in different shots are the same person.
DEFAULT_REID_SIMILARITY = 0.65
# The same, for face tracks. Stricter than the body threshold: the encoder is a general-purpose backbone rather than a face recognition model, so its face
# embeddings are usable but noisier, and two different faces framed alike are more similar to each other than two different bodies are.
DEFAULT_FACE_SIMILARITY = 0.7
# Streams that carry appearance worth comparing across a cut. Objects are left out: "the same chair in two shots" is not an identity claim this stage makes.
LINKED_STREAMS = ("face", "body")

# Evidence a track needs before it is allowed to become an identity. A handful of low-confidence frames is what a detector artifact looks like, and promoting one invents a person who was never there.
DEFAULT_MIN_DETECTIONS = 3
DEFAULT_MIN_TRACK_SCORE = 0.3


@dataclass
class Track:
    """One track's lifetime within a shot."""

    key: str
    stream: str
    shot_id: int
    track_id: int
    class_id: int
    class_name: str
    start_frame: int
    end_frame: int
    num_frames: int = 0
    num_detected: int = 0
    score_sum: float = 0.0
    embedding: np.ndarray | None = None
    identity_id: str | None = None

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.num_frames if self.num_frames else 0.0


@dataclass
class Identity:
    """A stable identity spanning one or more tracks."""

    identity_id: str
    kind: str  # "person" or "object"
    face_tracks: list[str] = field(default_factory=list)
    body_tracks: list[str] = field(default_factory=list)
    object_tracks: list[str] = field(default_factory=list)
    shots: list[int] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    class_name: str = ""

    @property
    def track_keys(self) -> list[str]:
        return self.face_tracks + self.body_tracks + self.object_tracks


def containment(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """Fraction of the `inner` box's area that lies inside `outer`."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return overlap / area if area > 0 else 0.0


def bind_faces_to_bodies(
    votes: dict[tuple[str, str], int],
    min_votes: int = DEFAULT_MIN_VOTES,
) -> dict[str, str]:
    """Resolve containment votes into a one-to-one face-to-body binding.

    `votes` maps (face_key, body_key) to the number of frames in which the face
    was contained in that body. Pairs are taken greedily from the highest vote
    count down, so the strongest evidence wins, and each track is used once -
    one face belongs to one body.
    """
    binding: dict[str, str] = {}
    taken_bodies: set[str] = set()

    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    for (face_key, body_key), count in ranked:
        if count < min_votes:
            break
        if face_key in binding or body_key in taken_bodies:
            continue
        binding[face_key] = body_key
        taken_bodies.add(body_key)
    return binding


def is_supported(
    track: Track,
    min_detections: int = DEFAULT_MIN_DETECTIONS,
    min_score: float = DEFAULT_MIN_TRACK_SCORE,
) -> bool:
    """Whether a track carries enough evidence to stand for something real."""
    return track.num_detected >= min_detections and track.mean_score >= min_score


def cannot_link_pairs(
    records: list[dict],
    tracks_by_key: dict[str, Track],
) -> set[tuple[str, str]]:
    """Identity pairs that were visible in the same frame."""
    per_frame: defaultdict[int, set[str]] = defaultdict(set)
    for record in records:
        identity_id = tracks_by_key[record["track_key"]].identity_id
        if identity_id and identity_id.startswith("person"):
            per_frame[record["frame_index"]].add(identity_id)

    pairs: set[tuple[str, str]] = set()
    for present in per_frame.values():
        pairs.update(combinations(sorted(present), 2))
    return pairs


def _identity_embeddings(
    identities: dict[str, Identity],
    tracks: list[Track],
) -> dict[str, dict[str, np.ndarray]]:
    """One averaged embedding per person identity per stream."""
    by_key = {t.key: t for t in tracks}
    result: dict[str, dict[str, np.ndarray]] = {}

    for identity in identities.values():
        if identity.kind != "person":
            continue
        per_stream: dict[str, np.ndarray] = {}
        for stream in LINKED_STREAMS:
            vectors, weights = [], []
            for key in getattr(identity, f"{stream}_tracks"):
                track = by_key.get(key)
                if track is None or track.embedding is None:
                    continue
                vectors.append(
                    np.asarray(track.embedding, dtype=np.float32).reshape(-1)
                )
                weights.append(float(max(track.num_detected, 1)))
            if not vectors:
                continue
            weighted = np.array(weights, dtype=np.float32)[:, None]
            total = (np.stack(vectors) * weighted).sum(0)
            norm = float(np.linalg.norm(total))
            if norm > 0:
                per_stream[stream] = total / norm
        if per_stream:
            result[identity.identity_id] = per_stream
    return result


def _link_margin(
    group_a: set[str],
    group_b: set[str],
    embeddings: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> float | None:
    """How far the best-agreeing stream clears its threshold, or None."""
    best: float | None = None
    for stream, threshold in thresholds.items():
        sims = [
            float(embeddings[a][stream] @ embeddings[b][stream])
            for a in group_a
            for b in group_b
            if stream in embeddings[a] and stream in embeddings[b]
        ]
        if not sims:
            continue
        margin = sum(sims) / len(sims) - threshold
        if best is None or margin > best:
            best = margin
    return best


def link_across_shots(
    identities: dict[str, Identity],
    tracks: list[Track],
    cannot_link: set[tuple[str, str]] | None = None,
    similarity: float = DEFAULT_REID_SIMILARITY,
    face_similarity: float = DEFAULT_FACE_SIMILARITY,
) -> dict[str, str]:
    """Cluster person identities that appearance says are the same person."""
    embeddings = _identity_embeddings(identities, tracks)
    if len(embeddings) < 2:
        return {}

    thresholds = {"body": similarity, "face": face_similarity}
    cannot_link = cannot_link or set()

    # Representative -> members. Reps are the lowest id in the cluster, and ids are numbered by first appearance, which is the ordering merge_identities
    # relies on to build a representative before anything folds into it.
    clusters: dict[str, set[str]] = {i: {i} for i in embeddings}
    blocked: dict[str, set[str]] = {
        i: {b for a, b in cannot_link if a == i} | {a for a, b in cannot_link if b == i}
        for i in embeddings
    }

    while len(clusters) > 1:
        best: tuple[float, str, str] | None = None
        for a, b in combinations(sorted(clusters), 2):
            if clusters[b] & blocked[a]:
                continue
            margin = _link_margin(clusters[a], clusters[b], embeddings, thresholds)
            if margin is not None and (best is None or margin > best[0]):
                best = (margin, a, b)
        if best is None or best[0] < 0:
            break

        _, keep, absorb = best
        clusters[keep] |= clusters[absorb]
        blocked[keep] |= blocked[absorb]
        del clusters[absorb], blocked[absorb]

    return {
        member: rep
        for rep, members in clusters.items()
        for member in members
        if member != rep
    }


def assign_identities(
    tracks: list[Track],
    bindings: dict[str, str],
) -> dict[str, Identity]:
    """Give every track an identity id, merging each bound face-body pair.

    Person ids are numbered by first appearance so that reading the manifest
    top to bottom follows the video.
    """
    by_key = {t.key: t for t in tracks}

    # Group track keys into identities: a bound face and body share one group.
    groups: dict[str, list[str]] = {}
    group_of: dict[str, str] = {}
    for track in sorted(tracks, key=lambda t: (t.start_frame, t.stream, t.track_id)):
        if track.key in group_of:
            continue
        members = [track.key]
        if track.stream == "face" and track.key in bindings:
            members.append(bindings[track.key])
        elif track.stream == "body":
            for face_key, body_key in bindings.items():
                if body_key == track.key and face_key not in group_of:
                    members.append(face_key)
        for key in members:
            group_of[key] = track.key
        groups[track.key] = members

    identities: dict[str, Identity] = {}
    counters = {"person": 0, "object": 0}
    for root in sorted(groups, key=lambda k: (by_key[k].start_frame, k)):
        members = [by_key[k] for k in groups[root]]
        kind = "object" if members[0].stream == "object" else "person"
        counters[kind] += 1
        identity_id = f"{kind}_{counters[kind]:04d}"

        identity = Identity(
            identity_id=identity_id,
            kind=kind,
            shots=sorted({m.shot_id for m in members}),
            start_frame=min(m.start_frame for m in members),
            end_frame=max(m.end_frame for m in members),
            class_name=members[0].class_name if kind == "object" else "person",
        )
        for member in members:
            member.identity_id = identity_id
            getattr(identity, f"{member.stream}_tracks").append(member.key)
        identities[identity_id] = identity

    return identities


def merge_identities(
    identities: dict[str, Identity],
    tracks: list[Track],
    merges: dict[str, str],
) -> dict[str, Identity]:
    """Apply cross-shot merges, folding each identity into its representative.

    Representatives are the lowest id in each merged set, and ids were numbered
    by first appearance, so walking identities in that same order guarantees a
    representative is created before anything folds into it.
    """
    if not merges:
        return identities

    merged: dict[str, Identity] = {}
    # Ids are zero-padded and numbered by first appearance, so sorting them
    # lexicographically is the same order union-find picked representatives in.
    for identity_id in sorted(identities):
        identity = identities[identity_id]
        target_id = merges.get(identity_id, identity_id)
        target = merged.get(target_id)
        if target is None:
            identity.identity_id = target_id
            merged[target_id] = identity
            continue
        target.face_tracks += identity.face_tracks
        target.body_tracks += identity.body_tracks
        target.object_tracks += identity.object_tracks
        target.shots = sorted(set(target.shots) | set(identity.shots))
        target.start_frame = min(target.start_frame, identity.start_frame)
        target.end_frame = max(target.end_frame, identity.end_frame)

    for track in tracks:
        if track.identity_id in merges:
            track.identity_id = merges[track.identity_id]

    return merged


def renumber_identities(
    identities: dict[str, Identity],
    tracks: list[Track],
) -> dict[str, Identity]:
    """Renumber ids contiguously by first appearance.

    Merging folds identities into their representative and leaves the absorbed
    numbers unused, so a clip can end up reporting person_0001 and person_0005
    and nothing between. Reading the manifest should not raise the question of
    where the missing three went.
    """
    mapping: dict[str, str] = {}
    renumbered: dict[str, Identity] = {}
    counters = {"person": 0, "object": 0}

    for identity in sorted(
        identities.values(), key=lambda i: (i.start_frame, i.identity_id)
    ):
        counters[identity.kind] += 1
        new_id = f"{identity.kind}_{counters[identity.kind]:04d}"
        mapping[identity.identity_id] = new_id
        identity.identity_id = new_id
        renumbered[new_id] = identity

    for track in tracks:
        if track.identity_id in mapping:
            track.identity_id = mapping[track.identity_id]

    return renumbered


def count_people(
    identities: dict[str, Identity],
    tracks_by_key: dict[str, Track],
    shot_id: int,
) -> list[str]:
    """Person identities present in a shot, for the gate's headcount."""
    present: set[str] = set()
    for identity in identities.values():
        if identity.kind != "person":
            continue
        if any(tracks_by_key[k].shot_id == shot_id for k in identity.track_keys):
            present.add(identity.identity_id)
    return sorted(present)


def collect_votes(
    face_boxes: list[tuple[str, tuple[float, ...]]],
    body_boxes: list[tuple[str, tuple[float, ...]]],
    votes: defaultdict[tuple[str, str], int],
    threshold: float = DEFAULT_CONTAINMENT,
) -> None:
    """Add one frame's face-in-body containment votes.

    Each face votes for at most one body - the one containing it best - so a
    face overlapping two bodies in a crowd does not reinforce both pairings.
    """
    for face_key, face_box in face_boxes:
        best_key, best_score = None, threshold
        for body_key, body_box in body_boxes:
            score = containment(face_box, body_box)
            if score >= best_score:
                best_key, best_score = body_key, score
        if best_key is not None:
            votes[(face_key, best_key)] += 1
