import awkward as ak
import numpy as np
import pytest
from scipy.spatial import cKDTree

from bioverse.adapters.picasso_ship import (
    SHIP_WALK,
    TEMPLATE,
    TILT_LEFT_DEG,
    PicassoShipAdapter,
    _apply_rotation,
    _is_bottom_body_node,
    _knn_edges,
    _parse_ship_walk,
    _perturb_picasso,
    _perturb_tilt,
    generate_sample,
)


def test_template_matches_ship_walk():
    positions, edges = _parse_ship_walk(SHIP_WALK)
    assert len(positions) == 9
    assert len(edges) == 10
    assert TEMPLATE.n_nodes == 9
    assert len(TEMPLATE.edges) == 10
    assert np.allclose(TEMPLATE.positions.mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("negative_type", ["scramble", "chirality", "picasso", "tilt"])
def test_balanced_classes(negative_type):
    batches, split, assets = PicassoShipAdapter.download(
        n_samples_per_class=4,
        negative_type=negative_type,
        seed=0,
    )
    batch = next(iter(batches))
    labels = ak.to_numpy(ak.ravel(batch.molecule_present))
    assert labels.shape == (8,)
    assert np.sum(labels == 1) == 4
    assert np.sum(labels == 0) == 4


def test_rotate_only_test_split():
    batches, split, assets = PicassoShipAdapter.download(
        n_samples_per_class=20,
        negative_type="scramble",
        n_noise=0,
        rotate=True,
        seed=0,
    )
    batch = next(iter(batches))
    rotated = ak.to_numpy(ak.ravel(batch.molecule_rotated)).astype(bool)
    assert rotated[split["test"]].all()
    assert not rotated[split["train"]].any()
    assert not rotated[split["val"]].any()
    assert assets["generation_config"]["rotate_split"] == "test"

    upright, _, _ = PicassoShipAdapter.download(
        n_samples_per_class=8,
        negative_type="scramble",
        n_noise=0,
        rotate=False,
        seed=1,
    )
    upright_batch = next(iter(upright))
    assert not ak.to_numpy(ak.ravel(upright_batch.molecule_rotated)).any()


def test_shuffled_stratified_split():
    batches, split, assets = PicassoShipAdapter.download(
        n_samples_per_class=20,
        negative_type="scramble",
        seed=0,
    )
    batch = next(iter(batches))
    labels = ak.to_numpy(ak.ravel(batch.molecule_present))
    assert not np.all(labels[:20] == 1), "records should be shuffled, not pos-then-neg"
    names = split.names["random_scene_split"]
    for part in ("train", "val", "test"):
        mask = split[part]
        part_labels = labels[mask]
        n_pos = int((part_labels == 1).sum())
        n_neg = int((part_labels == 0).sum())
        assert n_pos == n_neg, f"{part} is not class-balanced: {n_pos} pos, {n_neg} neg"
        assert part in names


def test_positive_has_template_edges():
    rng = np.random.default_rng(0)
    record = generate_sample(
        rng,
        TEMPLATE,
        0,
        present=True,
        negative_type="scramble",
        n_noise=0,
        noise_radius=2.0,
        knn_k=2,
        rotate=False,
    )
    edges = np.array(record.molecule_edges[0][0])
    template_pairs = {tuple(sorted(e)) for e in TEMPLATE.edges}
    edge_pairs = {tuple(sorted(map(int, e))) for e in edges}
    assert template_pairs.issubset(edge_pairs)
    assert np.all(np.array(record.atom_is_motif[0][0][0][0]) == 1)


def test_scramble_uses_pattern_knn_graph():
    rng = np.random.default_rng(1)
    record = generate_sample(
        rng,
        TEMPLATE,
        1,
        present=False,
        negative_type="scramble",
        n_noise=0,
        noise_radius=2.0,
        knn_k=2,
        rotate=False,
    )
    assert int(record.molecule_present[0][0]) == 0
    pos = np.array(record.atom_pos[0][0][0][0])[:, :2]
    pattern_edges = np.array(record.molecule_pattern_edges[0][0])
    expected = _knn_edges(pos, 2)
    pattern_pairs = {tuple(sorted(map(int, e))) for e in pattern_edges}
    assert pattern_pairs == {tuple(sorted(map(int, e))) for e in expected}
    assert np.all(np.array(record.molecule_pattern_edge_is_motif[0][0]) == 1)
    assert np.all(np.array(record.molecule_edge_is_motif[0][0]) == 1)
    assert len(record.molecule_edges[0][0]) > 0


def test_scramble_pattern_edges_preserved_with_noise():
    rng = np.random.default_rng(6)
    record = generate_sample(
        rng,
        TEMPLATE,
        6,
        present=False,
        negative_type="scramble",
        n_noise=10,
        noise_radius=2.0,
        knn_k=2,
        rotate=False,
    )
    pattern_edges = {tuple(sorted(map(int, e))) for e in record.molecule_pattern_edges[0][0]}
    merged_edges = {tuple(sorted(map(int, e))) for e in record.molecule_edges[0][0]}
    assert pattern_edges.issubset(merged_edges)
    assert len(merged_edges) > len(pattern_edges)
    motif_edge_flags = np.array(record.molecule_edge_is_motif[0][0])
    assert np.any(motif_edge_flags == 0)
    assert np.any(motif_edge_flags == 1)


def test_positive_stores_template_pattern_edges():
    rng = np.random.default_rng(0)
    record = generate_sample(
        rng,
        TEMPLATE,
        0,
        present=True,
        negative_type="scramble",
        n_noise=0,
        noise_radius=2.0,
        knn_k=2,
        rotate=False,
    )
    pattern_edges = {tuple(sorted(e)) for e in TEMPLATE.edges}
    stored = {tuple(sorted(map(int, e))) for e in record.molecule_pattern_edges[0][0]}
    assert pattern_edges == stored


def test_noise_nodes_and_edges():
    rng = np.random.default_rng(2)
    record = generate_sample(
        rng,
        TEMPLATE,
        2,
        present=True,
        negative_type="scramble",
        n_noise=10,
        noise_radius=2.0,
        knn_k=2,
        rotate=False,
    )
    is_motif = np.array(record.atom_is_motif[0][0][0][0])
    assert np.sum(is_motif == 1) == TEMPLATE.n_nodes
    assert np.sum(is_motif == 0) == 10
    assert np.any(np.array(record.molecule_edge_is_motif[0][0]) == 0)


def test_knn_adds_noise_bridges_but_not_extra_motif_motif():
    rng = np.random.default_rng(9)
    record = generate_sample(
        rng,
        TEMPLATE,
        9,
        present=True,
        negative_type="scramble",
        n_noise=20,
        noise_radius=3.0,
        knn_k=3,
        min_node_distance=0.5,
        rotate=False,
    )
    n_motif = TEMPLATE.n_nodes
    pattern = {tuple(sorted(map(int, e))) for e in record.molecule_pattern_edges[0][0]}
    merged = [tuple(sorted(map(int, e))) for e in record.molecule_edges[0][0]]
    motif_motif = {e for e in merged if e[0] < n_motif and e[1] < n_motif}
    bridges = [e for e in merged if (e[0] < n_motif) != (e[1] < n_motif)]
    noise_noise = [e for e in merged if e[0] >= n_motif and e[1] >= n_motif]
    assert motif_motif == pattern
    assert len(bridges) > 0
    assert len(noise_noise) > 0


def test_knn_adds_noise_bridges_but_not_extra_motif_motif():
    rng = np.random.default_rng(9)
    record = generate_sample(
        rng,
        TEMPLATE,
        9,
        present=True,
        negative_type="scramble",
        n_noise=20,
        noise_radius=3.0,
        knn_k=3,
        min_node_distance=0.5,
        rotate=False,
    )
    n_motif = TEMPLATE.n_nodes
    pattern = {tuple(sorted(map(int, e))) for e in record.molecule_pattern_edges[0][0]}
    merged = [tuple(sorted(map(int, e))) for e in record.molecule_edges[0][0]]
    motif_motif = {e for e in merged if e[0] < n_motif and e[1] < n_motif}
    bridges = [e for e in merged if (e[0] < n_motif) != (e[1] < n_motif)]
    noise_noise = [e for e in merged if e[0] >= n_motif and e[1] >= n_motif]
    assert motif_motif == pattern
    assert len(bridges) > 0
    assert len(noise_noise) > 0


def test_rotation_generates_valid_sample():
    rng = np.random.default_rng(3)
    record = generate_sample(
        rng,
        TEMPLATE,
        3,
        present=True,
        negative_type="scramble",
        n_noise=5,
        noise_radius=2.0,
        knn_k=2,
        rotate=True,
    )
    pos = np.array(record.atom_pos[0][0][0][0])[:, :2]
    assert pos.shape[0] == TEMPLATE.n_nodes + 5
    assert np.isfinite(pos).all()


def test_min_node_distance_for_sampled_points():
    rng = np.random.default_rng(7)
    min_distance = 0.5
    record = generate_sample(
        rng,
        TEMPLATE,
        7,
        present=True,
        negative_type="scramble",
        n_noise=15,
        noise_radius=2.5,
        knn_k=3,
        min_node_distance=min_distance,
        rotate=False,
    )
    pos = np.array(record.atom_pos[0][0][0][0])[:, :2]
    is_motif = np.array(record.atom_is_motif[0][0][0][0]).astype(bool)
    noise = pos[~is_motif]
    for i, point in enumerate(noise):
        others = np.concatenate([pos[is_motif], noise[:i], noise[i + 1 :]], axis=0)
        assert np.min(np.linalg.norm(others - point, axis=1)) >= min_distance - 1e-8

    scramble = generate_sample(
        rng,
        TEMPLATE,
        8,
        present=False,
        negative_type="scramble",
        n_noise=15,
        noise_radius=2.5,
        knn_k=3,
        min_node_distance=min_distance,
        rotate=False,
    )
    scramble_pos = np.array(scramble.atom_pos[0][0][0][0])[:, :2]
    tree_d = np.min(
        cKDTree(scramble_pos).query(scramble_pos, k=2)[0][:, 1]
    )
    assert tree_d >= min_distance - 1e-8


def test_tilt_is_fixed_45_degrees_left():
    positions, edges = _perturb_tilt(TEMPLATE)
    anchor = TEMPLATE.anchor_index()
    body = TEMPLATE.body_nodes
    rotate_idx = TEMPLATE.mast_sail_indices()

    assert np.array_equal(edges, TEMPLATE.edges)
    assert np.allclose(positions[body], TEMPLATE.positions[body])
    assert np.allclose(positions[anchor], TEMPLATE.positions[anchor])
    expected = _apply_rotation(
        TEMPLATE.positions[rotate_idx],
        np.deg2rad(TILT_LEFT_DEG),
        TEMPLATE.positions[anchor],
    )
    assert np.allclose(positions[rotate_idx], expected)
    mast_tip = int(TEMPLATE.mast_nodes[-1])
    before = TEMPLATE.positions[mast_tip] - TEMPLATE.positions[anchor]
    after = positions[mast_tip] - positions[anchor]
    # +y mast leans toward -x (left).
    assert after[0] < 0
    assert np.dot(before, after) == pytest.approx(
        np.linalg.norm(before) * np.linalg.norm(after) * np.cos(np.deg2rad(TILT_LEFT_DEG)),
        rel=1e-6,
    )


def test_picasso_motif_is_recentered_before_noise():
    rng = np.random.default_rng(5)
    record = generate_sample(
        rng,
        TEMPLATE,
        5,
        present=False,
        negative_type="picasso",
        n_noise=10,
        noise_radius=3.0,
        knn_k=3,
        rotate=False,
    )
    pos = np.array(record.atom_pos[0][0][0][0])[:, :2]
    is_motif = np.array(record.atom_is_motif[0][0][0][0]).astype(bool)
    assert np.allclose(pos[is_motif].mean(axis=0), 0.0, atol=1e-10)
    assert int(is_motif.sum()) == TEMPLATE.n_nodes
    assert int((~is_motif).sum()) == 10


def test_picasso_transplants_mast_and_sail():
    rng = np.random.default_rng(5)
    positions, edges, _ = _perturb_picasso(TEMPLATE, rng)
    anchor = TEMPLATE.anchor_index()
    body = TEMPLATE.body_nodes
    mast = TEMPLATE.mast_nodes
    sail = TEMPLATE.sail_nodes

    assert np.allclose(positions[body], TEMPLATE.positions[body])
    assert np.allclose(positions[anchor], TEMPLATE.positions[anchor])
    assert not np.allclose(positions[mast], TEMPLATE.positions[mast])

    mast_base = int(mast[0])
    body_set = {int(n) for n in body}
    edge_pairs = {tuple(sorted(map(int, e))) for e in edges}
    assert tuple(sorted((anchor, mast_base))) not in edge_pairs
    attached = [
        p for p in edge_pairs
        if mast_base in p and ((p[0] in body_set) or (p[1] in body_set))
    ]
    assert attached
    other = attached[0][0] if attached[0][1] == mast_base else attached[0][1]
    assert _is_bottom_body_node(TEMPLATE, other)


def test_picasso_samples_only_bottom_hull_nodes():
    rng = np.random.default_rng(1)
    mast_base = int(TEMPLATE.mast_nodes[0])
    body_set = {int(n) for n in TEMPLATE.body_nodes}
    seen = set()
    for _i in range(40):
        _positions, edges, _ = _perturb_picasso(TEMPLATE, rng)
        edge_pairs = {tuple(sorted(map(int, e))) for e in edges}
        attached = next(
            p for p in edge_pairs
            if mast_base in p and ((p[0] in body_set) or (p[1] in body_set))
        )
        other = attached[0] if attached[1] == mast_base else attached[1]
        assert _is_bottom_body_node(TEMPLATE, other)
        seen.add(int(other))
    assert seen == {int(n) for n in TEMPLATE.body_nodes if _is_bottom_body_node(TEMPLATE, int(n))}


def test_picasso_flips_mast_when_attached_to_bottom():
    rng = np.random.default_rng(0)
    anchor = TEMPLATE.anchor_index()
    mast_tip = int(TEMPLATE.mast_nodes[-1])
    original_mast_dir = TEMPLATE.positions[mast_tip] - TEMPLATE.positions[anchor]
    body_centroid = TEMPLATE.positions[TEMPLATE.body_nodes].mean(axis=0)

    for node in TEMPLATE.body_nodes:
        positions, edges, _ = _perturb_picasso(TEMPLATE, rng, wrong_anchor=int(node))
        mast_dir = positions[mast_tip] - positions[int(node)]
        toward_body = body_centroid - positions[int(node)]
        if _is_bottom_body_node(TEMPLATE, int(node)):
            assert np.dot(mast_dir, original_mast_dir) < 0
            assert np.dot(mast_dir, toward_body) < 0
        else:
            assert np.dot(mast_dir, original_mast_dir) > 0
        mast_base = int(TEMPLATE.mast_nodes[0])
        edge_pairs = {tuple(sorted(map(int, e))) for e in edges}
        assert tuple(sorted((int(node), mast_base))) in edge_pairs


def test_invalid_negative_type():
    with pytest.raises(ValueError):
        PicassoShipAdapter.download(negative_type="unknown")
