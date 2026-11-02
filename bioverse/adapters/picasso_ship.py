"""Synthetic ship-motif point clouds for equivariant pattern-matching benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import awkward as ak
import numpy as np
from scipy.spatial import cKDTree

from ..adapter import Adapter
from ..data import Assets, Split
from ..transforms.scene_split import stratified_split_index
from ..utilities import IteratorWithLength, batched

NEGATIVE_TYPES = ("scramble", "chirality", "picasso", "tilt")

# Part labels stored on motif nodes.
PART_BODY = 0
PART_MAST = 1
PART_SAIL = 2


class PicassoVariant(str, Enum):
    SAIL_WRONG_ANCHOR = "sail_wrong_anchor"


@dataclass(frozen=True)
class ShipTemplate:
    """Canonical ship motif centered at the origin."""

    positions: np.ndarray  # (N, 2)
    parts: np.ndarray  # (N,) int
    edges: np.ndarray  # (E, 2) int
    body_nodes: np.ndarray  # body-only nodes (excludes shared mast anchor)
    mast_anchor_node: int  # shared body/mast junction at (2, 1)
    mast_nodes: np.ndarray  # mast above anchor: (2, 2), (2, 3)
    sail_nodes: np.ndarray  # sail triangle: (2, 2), (2, 3), (3, 2)

    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    @property
    def mast_indices(self) -> np.ndarray:
        return self.mast_nodes

    @property
    def body_indices(self) -> np.ndarray:
        return self.body_nodes

    @property
    def sail_indices(self) -> np.ndarray:
        return self.sail_nodes

    def anchor_index(self) -> int:
        """Body/mast junction used as the tilt pivot and mast base."""
        return int(self.mast_anchor_node)

    def mast_sail_indices(self) -> np.ndarray:
        """Mast (above anchor) and sail nodes moved/rotated together."""
        return np.unique(np.concatenate([self.mast_nodes, self.sail_nodes]))


BODY_WALK = "(2|1)-(1|1)-(0|1)-(1|0)-(2|0)-(3|1)-(2|1)"
MAST_WALK = "(2|1)-(2|2)-(2|3)"
SAIL_WALK = "(2|3)-(3|2)-(2|2)"
SHIP_WALK = "(2|1)-(1|1)-(0|1)-(1|0)-(2|0)-(3|1)-(2|1)-(2|2)-(2|3)-(3|2)-(2|2)"


def _unique_undirected_edges(edges: np.ndarray) -> np.ndarray:
    if len(edges) == 0:
        return edges
    normalized = np.sort(edges, axis=1)
    return np.unique(normalized, axis=0)


def _walk_coords(spec: str) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for token in spec.split("-"):
        x_str, y_str = token.strip("()").split("|")
        coords.append((float(x_str), float(y_str)))
    return coords


def _unique_walk_coords(spec: str) -> set[tuple[float, float]]:
    return set(_walk_coords(spec))

def _parse_ship_walk(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``(x|y)-(x|y)-...`` into unique node positions and path edges."""
    walk_coords = _walk_coords(spec)
    coord_to_idx: dict[tuple[float, float], int] = {}
    walk_indices: list[int] = []
    for xy in walk_coords:
        if xy not in coord_to_idx:
            coord_to_idx[xy] = len(coord_to_idx)
        walk_indices.append(coord_to_idx[xy])

    positions = np.array(
        [xy for xy, _ in sorted(coord_to_idx.items(), key=lambda item: item[1])],
        dtype=float,
    )
    edges = np.array(
        [[walk_indices[i], walk_indices[i + 1]] for i in range(len(walk_indices) - 1)],
        dtype=int,
    )
    return positions, _unique_undirected_edges(edges)


def _assign_parts(
    positions: np.ndarray,
    *,
    body_nodes: np.ndarray,
    mast_anchor_node: int,
    mast_nodes: np.ndarray,
    sail_nodes: np.ndarray,
) -> np.ndarray:
    parts = np.full(len(positions), PART_BODY, dtype=int)
    parts[body_nodes] = PART_BODY
    parts[mast_anchor_node] = PART_MAST
    parts[mast_nodes] = PART_MAST
    parts[sail_nodes] = PART_SAIL
    return parts


def _node_indices_for_coords(
    positions: np.ndarray, coords: set[tuple[float, float]]
) -> np.ndarray:
    indices = [
        i
        for i, (x, y) in enumerate(positions)
        if (float(x), float(y)) in coords
    ]
    return np.array(indices, dtype=int)


def _build_template() -> ShipTemplate:
    """Build the ship motif from ``SHIP_WALK`` and center it at the origin."""
    positions, edges = _parse_ship_walk(SHIP_WALK)
    mast_anchor_coords = _unique_walk_coords(MAST_WALK) & _unique_walk_coords(BODY_WALK)
    mast_only_coords = _unique_walk_coords(MAST_WALK) - mast_anchor_coords
    body_only_coords = _unique_walk_coords(BODY_WALK) - mast_anchor_coords
    sail_coords = _unique_walk_coords(SAIL_WALK)

    mast_anchor_node = int(_node_indices_for_coords(positions, mast_anchor_coords)[0])
    body_nodes = _node_indices_for_coords(positions, body_only_coords)
    mast_nodes = _node_indices_for_coords(positions, mast_only_coords)
    mast_nodes = mast_nodes[np.argsort(positions[mast_nodes, 1])]
    sail_nodes = _node_indices_for_coords(positions, sail_coords)
    parts = _assign_parts(
        positions,
        body_nodes=body_nodes,
        mast_anchor_node=mast_anchor_node,
        mast_nodes=mast_nodes,
        sail_nodes=sail_nodes,
    )
    return ShipTemplate(
        positions=positions - positions.mean(axis=0),
        parts=parts,
        edges=edges,
        body_nodes=body_nodes,
        mast_anchor_node=mast_anchor_node,
        mast_nodes=mast_nodes,
        sail_nodes=sail_nodes,
    )


TEMPLATE = _build_template()


def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _apply_rotation(points: np.ndarray, angle: float, center: np.ndarray | None = None) -> np.ndarray:
    if center is None:
        center = np.zeros(2, dtype=float)
    return (points - center) @ _rotation_matrix(angle).T + center


def _mirror_points_across_line(points: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
    direction = line_direction / (np.linalg.norm(line_direction) + 1e-8)
    rel = points - line_point
    projected = np.outer(rel @ direction, direction)
    return line_point + 2 * projected - rel


def _knn_edges(points: np.ndarray, k: int) -> np.ndarray:
    n = len(points)
    if n <= 1:
        return np.empty((0, 2), dtype=int)
    k = min(k, n - 1)
    _, neighbors = cKDTree(points).query(points, k=k + 1)
    neighbors = np.atleast_2d(neighbors)
    edges = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs[1:]:
            edges.append([i, int(j)])
    return _unique_undirected_edges(np.array(edges, dtype=int))


def _merge_edges(*edge_sets: np.ndarray) -> np.ndarray:
    if not edge_sets:
        return np.empty((0, 2), dtype=int)
    merged = np.concatenate([e for e in edge_sets if len(e) > 0], axis=0) if any(len(e) for e in edge_sets) else np.empty((0, 2), dtype=int)
    if len(merged) == 0:
        return merged
    return _unique_undirected_edges(merged)


def _sample_rotation_angle(
    rng: np.random.Generator,
    min_deg: float,
    max_deg: float,
) -> float:
    return np.deg2rad(rng.uniform(min_deg, max_deg))


# Mast leans 45° to the left (counter-clockwise from +y).
TILT_LEFT_DEG = 45.0


def _sample_disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    angles = rng.uniform(0.0, 2 * np.pi, size=n)
    radii = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def _sample_points_in_disk(
    rng: np.random.Generator,
    n: int,
    radius: float,
    *,
    min_distance: float = 0.0,
    existing: np.ndarray | None = None,
    max_attempts: int = 20_000,
) -> np.ndarray:
    """Sample ``n`` disk points, optionally keeping a minimum pairwise distance.

    ``existing`` points are exclusion sites only and are not returned. Motif
    geometry is never moved; only newly sampled points are constrained.
    """
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    if min_distance <= 0.0 and (existing is None or len(existing) == 0):
        return _sample_disk(rng, n, radius)

    refs = (
        np.empty((0, 2), dtype=float)
        if existing is None or len(existing) == 0
        else np.asarray(existing, dtype=float)
    )
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < n:
        remaining = n - len(accepted)
        batch = _sample_disk(rng, max(remaining * 8, 16), radius)
        for candidate in batch:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to place {n} points with min_distance={min_distance} "
                    f"in a disk of radius {radius} after {max_attempts} attempts"
                )
            if len(refs) == 0 and not accepted:
                accepted.append(candidate)
                continue
            occupied = refs if not accepted else np.vstack([refs, np.asarray(accepted)])
            if float(np.min(np.linalg.norm(occupied - candidate, axis=1))) >= min_distance:
                accepted.append(candidate)
                if len(accepted) >= n:
                    break
    return np.asarray(accepted, dtype=float)


def _perturb_scramble(
    rng: np.random.Generator,
    template: ShipTemplate,
    noise_radius: float,
    knn_k: int,
    min_node_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _sample_points_in_disk(
        rng,
        template.n_nodes,
        noise_radius,
        min_distance=min_node_distance,
    )
    return positions, _knn_edges(positions, knn_k)


def _perturb_chirality(template: ShipTemplate) -> tuple[np.ndarray, np.ndarray]:
    positions = template.positions.copy()
    anchor = template.anchor_index()
    mast_axis = positions[template.mast_nodes[-1]] - positions[anchor]
    positions[template.sail_indices] = _mirror_points_across_line(
        positions[template.sail_indices], positions[anchor], mast_axis
    )
    return positions, template.edges.copy()


def _is_bottom_body_node(template: ShipTemplate, node: int) -> bool:
    """True if ``node`` sits on the hull below the canonical mast deck."""
    return float(template.positions[node, 1]) < float(
        template.positions[template.mast_anchor_node, 1]
    ) - 1e-8


def _bottom_body_nodes(template: ShipTemplate) -> np.ndarray:
    """Hull nodes below the deck; Picasso negatives attach the mast here."""
    return np.array(
        [int(node) for node in template.body_nodes if _is_bottom_body_node(template, int(node))],
        dtype=int,
    )


def _perturb_picasso(
    template: ShipTemplate,
    rng: np.random.Generator,
    wrong_anchor: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    positions = template.positions.copy()
    edges = template.edges.copy()
    anchor = template.anchor_index()
    movable = template.mast_sail_indices()
    mast_base = int(template.mast_nodes[0])

    if wrong_anchor is None:
        bottom = _bottom_body_nodes(template)
        if bottom.size == 0:
            raise RuntimeError("Picasso negatives need at least one bottom hull node")
        wrong_anchor = int(rng.choice(bottom))
    else:
        wrong_anchor = int(wrong_anchor)
    offset = positions[wrong_anchor] - positions[anchor]
    positions[movable] += offset
    if _is_bottom_body_node(template, wrong_anchor):
        # Original mast points +y through the hull; flip so it points away.
        positions[movable] = _apply_rotation(
            positions[movable], np.pi, positions[wrong_anchor]
        )

    kept = []
    for u, v in edges:
        if (u == anchor and v == mast_base) or (v == anchor and u == mast_base):
            continue
        kept.append([u, v])
    edges = np.array(kept, dtype=int) if kept else np.empty((0, 2), dtype=int)
    edges = np.vstack([edges, np.array([[wrong_anchor, mast_base]], dtype=int)])

    return positions, _unique_undirected_edges(edges), PicassoVariant.SAIL_WRONG_ANCHOR.value


def _perturb_tilt(template: ShipTemplate) -> tuple[np.ndarray, np.ndarray]:
    """Tilt mast+sail 45° left about the deck anchor. Body stays put."""
    positions = template.positions.copy()
    anchor = template.anchor_index()
    rotate_idx = template.mast_sail_indices()
    positions[rotate_idx] = _apply_rotation(
        positions[rotate_idx],
        np.deg2rad(TILT_LEFT_DEG),
        positions[anchor],
    )
    return positions, template.edges.copy()


def _build_motif(
    rng: np.random.Generator,
    template: ShipTemplate,
    *,
    present: bool,
    negative_type: str,
    noise_radius: float,
    knn_k: int,
    min_tilt_deg: float,
    max_tilt_deg: float,
    min_tilt_neg_deg: float,
    max_tilt_neg_deg: float,
    min_node_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    if present:
        return template.positions.copy(), template.edges.copy(), None

    if negative_type == "scramble":
        pos, edges = _perturb_scramble(
            rng, template, noise_radius, knn_k, min_node_distance=min_node_distance
        )
        return pos, edges, None
    if negative_type == "chirality":
        pos, edges = _perturb_chirality(template)
        return pos, edges, None
    if negative_type == "picasso":
        return _perturb_picasso(template, rng)
    if negative_type == "tilt":
        pos, edges = _perturb_tilt(template)
        return pos, edges, None
    raise ValueError(f"Unknown negative_type {negative_type!r}; expected one of {NEGATIVE_TYPES}")


def _build_graph(
    motif_positions: np.ndarray,
    pattern_edges: np.ndarray,
    noise_positions: np.ndarray,
    *,
    knn_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge pattern edges with kNN among noise, plus motif–noise bridges.

    Extra kNN edges between two motif nodes are dropped so the motif's
    combinatorial skeleton stays exactly the pattern graph.
    """
    n_motif = len(motif_positions)
    positions = (
        motif_positions
        if len(noise_positions) == 0
        else np.concatenate([motif_positions, noise_positions], axis=0)
    )

    pattern_edges = _unique_undirected_edges(pattern_edges)
    pattern_pairs = {tuple(sorted(map(int, e))) for e in pattern_edges}

    edge_sets = [pattern_edges]
    if len(noise_positions) > 0:
        knn = _knn_edges(positions, knn_k)
        if len(knn) > 0:
            keep = ~((knn[:, 0] < n_motif) & (knn[:, 1] < n_motif))
            knn = knn[keep]
        edge_sets.append(knn)

    edges = _merge_edges(*edge_sets)
    is_motif_edge = np.zeros(len(edges), dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        normalized = tuple(sorted((int(u), int(v))))
        both_motif = u < n_motif and v < n_motif
        is_motif_edge[i] = int(both_motif and normalized in pattern_pairs)

    pattern_edge_is_motif = np.ones(len(pattern_edges), dtype=np.int8)
    return edges, is_motif_edge, pattern_edges, pattern_edge_is_motif


def _to_record(
    sample_id: str,
    positions: np.ndarray,
    edges: np.ndarray,
    edge_is_motif: np.ndarray,
    pattern_edges: np.ndarray,
    pattern_edge_is_motif: np.ndarray,
    is_motif_node: np.ndarray,
    present: bool,
    negative_type: str,
    picasso_variant: str | None,
    rotated: bool = False,
) -> ak.Record:
    pos_3d = np.pad(positions, ((0, 0), (0, 1)), constant_values=0.0).tolist()
    atom_labels = np.where(is_motif_node, "M", "N").tolist()
    return ak.Record(
        {
            "molecule_id": [[sample_id]],
            "molecule_present": [[int(present)]],
            "molecule_rotated": [[int(rotated)]],
            "molecule_negative_type": [[negative_type if not present else ""]],
            "molecule_picasso_variant": [[picasso_variant or ""]],
            "atom_label": [[[[atom_labels]]]],
            "atom_is_motif": [[[[is_motif_node.astype(np.int8).tolist()]]]],
            "atom_pos": [[[[pos_3d]]]],
            "molecule_pattern_edges": [[pattern_edges.astype(int).tolist()]],
            "molecule_pattern_edge_is_motif": [[pattern_edge_is_motif.astype(np.int8).tolist()]],
            "molecule_edges": [[edges.astype(int).tolist()]],
            "molecule_edge_is_motif": [[edge_is_motif.astype(np.int8).tolist()]],
        }
    )


def generate_sample(
    rng: np.random.Generator,
    template: ShipTemplate,
    sample_index: int,
    *,
    present: bool,
    negative_type: str,
    n_noise: int,
    noise_radius: float,
    knn_k: int,
    min_node_distance: float = 0.5,
    min_rotation_deg: float = 30.0,
    max_rotation_deg: float = 330.0,
    min_tilt_deg: float = 20.0,
    max_tilt_deg: float = 70.0,
    min_tilt_neg_deg: float = -70.0,
    max_tilt_neg_deg: float = -20.0,
    rotate: bool,
) -> ak.Record:
    label = "pos" if present else negative_type
    sample_id = f"ship_{label}_{sample_index:06d}"

    motif_pos, pattern_edges, picasso_variant = _build_motif(
        rng,
        template,
        present=present,
        negative_type=negative_type,
        noise_radius=noise_radius,
        knn_k=knn_k,
        min_node_distance=min_node_distance,
        min_tilt_deg=min_tilt_deg,
        max_tilt_deg=max_tilt_deg,
        min_tilt_neg_deg=min_tilt_neg_deg,
        max_tilt_neg_deg=max_tilt_neg_deg,
    )
    if (not present) and negative_type == "picasso" and len(motif_pos) > 0:
        motif_pos = motif_pos - motif_pos.mean(axis=0)

    noise_pos = (
        _sample_points_in_disk(
            rng,
            n_noise,
            noise_radius,
            min_distance=min_node_distance,
            existing=motif_pos,
        )
        if n_noise > 0
        else np.empty((0, 2))
    )
    all_pos = (
        motif_pos if len(noise_pos) == 0 else np.concatenate([motif_pos, noise_pos], axis=0)
    )
    if rotate:
        angle = _sample_rotation_angle(rng, min_rotation_deg, max_rotation_deg)
        all_pos = _apply_rotation(all_pos, angle)

    motif_pos = all_pos[: len(motif_pos)]
    noise_pos = all_pos[len(motif_pos) :]

    edges, edge_is_motif, pattern_edges, pattern_edge_is_motif = _build_graph(
        motif_pos,
        pattern_edges,
        noise_pos,
        knn_k=knn_k,
    )
    is_motif_node = np.concatenate(
        [np.ones(len(motif_pos), dtype=np.int8), np.zeros(len(noise_pos), dtype=np.int8)]
    )

    return _to_record(
        sample_id,
        all_pos,
        edges,
        edge_is_motif,
        pattern_edges,
        pattern_edge_is_motif,
        is_motif_node,
        present,
        negative_type,
        picasso_variant,
        rotated=rotate,
    )


class PicassoShipAdapter(Adapter):
    """Generate synthetic ship-motif classification point clouds.

    Each release is a balanced binary dataset: positives contain the canonical
    ship motif; negatives apply one selected perturbation
    (``scramble``, ``chirality``, ``picasso``, or ``tilt``). Optional
    modifiers add Gaussian clutter (``n_noise``) and global rotation
    (``rotate``). When ``rotate`` is true, train and val scenes stay upright
    and only the test partition is randomly rotated. Tilt negatives always
    rotate the mast 45° left. Picasso negatives attach the mast only to a
    bottom hull node (flipped away from the body) and are re-centered before
    noise is sampled.

    Parameters
    ----------
    n_samples_per_class : int
        Number of positive and negative scenes (total ``2 * n_samples_per_class``).
    negative_type : str
        Perturbation applied to negative samples.
    n_noise : int
        Number of additional noise nodes (modifier M1).
    noise_radius : float
        Disk radius for sampling noise node positions.
    knn_k : int
        Number of nearest neighbors for geometric graph construction on noise
        and scrambled motif nodes.
    min_node_distance : float
        Minimum Euclidean distance between sampled nodes (noise, and scrambled
        motif nodes). Existing motif geometry is never moved.
    min_rotation_deg, max_rotation_deg : float
        Global rotation range in degrees when ``rotate`` is enabled (test split).
    min_tilt_deg, max_tilt_deg, min_tilt_neg_deg, max_tilt_neg_deg : float
        Ignored. Tilt negatives always use a 45° left mast rotation.
    rotate : bool
        If true, apply a random global rotation to **test** scenes only.
    seed : int
        Random seed for reproducibility.
    """

    @classmethod
    def download(
        cls,
        n_samples_per_class: int = 1000,
        negative_type: str = "scramble",
        n_noise: int = 0,
        noise_radius: float = 3.0,
        knn_k: int = 2,
        min_node_distance: float = 0.5,
        min_rotation_deg: float = 30.0,
        max_rotation_deg: float = 330.0,
        min_tilt_deg: float = 20.0,
        max_tilt_deg: float = 70.0,
        min_tilt_neg_deg: float = -70.0,
        max_tilt_neg_deg: float = -20.0,
        rotate: bool = False,
        seed: int = 0,
    ) -> tuple[Iterator, Split, Assets]:
        if negative_type not in NEGATIVE_TYPES:
            raise ValueError(f"negative_type must be one of {NEGATIVE_TYPES}, got {negative_type!r}")
        if n_samples_per_class <= 0:
            raise ValueError("n_samples_per_class must be positive")

        rng = np.random.default_rng(seed)
        template = TEMPLATE
        n_total = 2 * n_samples_per_class
        present_flags = np.array([True] * n_samples_per_class + [False] * n_samples_per_class)
        order = rng.permutation(n_total)
        present_flags = present_flags[order]
        labels = present_flags.astype(int)
        index = stratified_split_index(
            labels,
            train_size=0.8,
            test_size=0.1,
            val_size=0.1,
            seed=seed,
        )

        records = []
        for sample_index, (present, partition) in enumerate(zip(present_flags, index)):
            records.append(
                generate_sample(
                    rng,
                    template,
                    sample_index,
                    present=bool(present),
                    negative_type=negative_type,
                    n_noise=n_noise,
                    noise_radius=noise_radius,
                    knn_k=knn_k,
                    min_node_distance=min_node_distance,
                    min_rotation_deg=min_rotation_deg,
                    max_rotation_deg=max_rotation_deg,
                    min_tilt_deg=min_tilt_deg,
                    max_tilt_deg=max_tilt_deg,
                    min_tilt_neg_deg=min_tilt_neg_deg,
                    max_tilt_neg_deg=max_tilt_neg_deg,
                    rotate=bool(rotate) and partition == "test",
                )
            )

        split = Split({"random_scene_split": index}, default="random_scene_split")
        return (
            batched(IteratorWithLength(iter(records), n_total)),
            split,
            Assets(
                {
                    "motif_template": {
                        "walk": SHIP_WALK,
                        "positions": template.positions.tolist(),
                        "parts": template.parts.tolist(),
                        "edges": template.edges.tolist(),
                        "body_nodes": template.body_nodes.tolist(),
                        "mast_anchor_node": template.mast_anchor_node,
                        "mast_nodes": template.mast_nodes.tolist(),
                        "sail_nodes": template.sail_nodes.tolist(),
                    },
                    "generation_config": {
                        "n_samples_per_class": n_samples_per_class,
                        "negative_type": negative_type,
                        "n_noise": n_noise,
                        "noise_radius": noise_radius,
                        "knn_k": knn_k,
                        "min_node_distance": min_node_distance,
                        "min_rotation_deg": min_rotation_deg,
                        "max_rotation_deg": max_rotation_deg,
                        "min_tilt_deg": min_tilt_deg,
                        "max_tilt_deg": max_tilt_deg,
                        "min_tilt_neg_deg": min_tilt_neg_deg,
                        "max_tilt_neg_deg": max_tilt_neg_deg,
                        "rotate": rotate,
                        "rotate_split": "test" if rotate else None,
                        "seed": seed,
                    },
                }
            ),
        )
