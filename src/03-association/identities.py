"""Association of per-stream tracks into stable identities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# Fraction of the face box that must fall inside the body box to count as a containment vote. Faces sit well inside a body box, so this can be strict.
DEFAULT_CONTAINMENT = 0.6
# Minimum number of voting frames before a face is bound to a body, which rejects pairings built on one or two frames of coincidental overlap.
DEFAULT_MIN_VOTES = 3
# Cosine similarity above which two body tracks in different shots are the same person.
DEFAULT_REID_SIMILARITY = 0.65


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


class _UnionFind:
    """Merges identities that turn out to be the same person across shots."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Keep the earlier identity as the representative so ids stay ordered by first appearance.
            first, second = sorted((root_a, root_b))
            self._parent[second] = first


def link_across_shots(
    tracks: list[Track],
    similarity: float = DEFAULT_REID_SIMILARITY,
) -> dict[str, str]:
    """Merge person identities across shots using body ReID embeddings.

    Returns a mapping from identity id to its merged representative. Only body
    tracks are compared: the ReID encoder is trained on whole people, so its
    features are meaningful for bodies and not for face crops or objects.
    """
    candidates = [
        t
        for t in tracks
        if t.stream == "body" and t.embedding is not None and t.identity_id
    ]
    if len(candidates) < 2:
        return {}

    matrix = np.stack([t.embedding for t in candidates]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (matrix / norms) @ (matrix / norms).T

    union = _UnionFind()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            # Same shot means the tracker already decided these are different people; only bridge across a cut.
            if candidates[i].shot_id == candidates[j].shot_id:
                continue
            if sims[i, j] >= similarity:
                union.union(candidates[i].identity_id, candidates[j].identity_id)

    return {
        t.identity_id: union.find(t.identity_id)
        for t in candidates
        if union.find(t.identity_id) != t.identity_id
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
