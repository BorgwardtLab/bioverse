import awkward as ak
import numpy as np
import pytest

from bioverse.data import Batch
from bioverse.transforms.filter_chains import FilterChains


def _two_chain_batch() -> Batch:
    return Batch(
        {
            "molecule_id": ak.Array([[["complex"]]]),
            "chain_label": ak.Array([[["A", "B"]]]),
            "residue_label": ak.Array([[[["A", "R"], ["N", "D"]]]]),
            "residue_pos": ak.Array(
                [[[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]]
            ),
        }
    )


def _leaf_strings(value) -> list[str]:
    items: list[str] = []

    def walk(node):
        if isinstance(node, str):
            items.append(node)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(ak.to_list(value))
    return items


def _chain_letters(batch: Batch) -> list[str]:
    return _leaf_strings(batch.data["chain_label"])


def test_keep_one_chain():
    out = FilterChains(keep="A").transform_batch(_two_chain_batch())
    assert _chain_letters(out) == ["A"]
    assert _leaf_strings(out.data["residue_label"]) == ["A", "R"]


def test_drop_one_chain():
    out = FilterChains(drop="B").transform_batch(_two_chain_batch())
    assert _chain_letters(out) == ["A"]


def test_keep_and_drop():
    out = FilterChains(keep=("A", "B"), drop="B").transform_batch(_two_chain_batch())
    assert _chain_letters(out) == ["A"]


def test_keep_all_is_identity():
    src = _two_chain_batch()
    out = FilterChains(keep=("A", "B")).transform_batch(src)
    assert _chain_letters(out) == ["A", "B"]
    assert out.data["chain_label"].ndim == src.data["chain_label"].ndim
    assert "residue_pos" in src.data


def test_keep_preserves_leading_axes():
    src = _two_chain_batch()
    out = FilterChains(keep="A").transform_batch(src)
    assert out.data["chain_label"].ndim == src.data["chain_label"].ndim
    assert out.data["residue_label"].ndim == src.data["residue_label"].ndim


def test_does_not_mutate_input():
    src = _two_chain_batch()
    FilterChains(keep="A").transform_batch(src)
    assert _chain_letters(src) == ["A", "B"]


def test_requires_keep_or_drop():
    with pytest.raises(ValueError, match="keep and/or drop"):
        FilterChains()


def test_empty_selection_raises():
    with pytest.raises(ValueError, match="No chains"):
        FilterChains(keep="Z").transform_batch(_two_chain_batch())
