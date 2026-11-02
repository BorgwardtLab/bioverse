"""Peptide-chain flags and protein-pocket masks for protein–peptide complexes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .constants import PROTEIN_ALPHABET


MASK_INDEX = len(PROTEIN_ALPHABET)


def is_residue_list(labels):
    return bool(labels) and isinstance(labels[0], str)


def peptide_flags(labels, types, peptide_type="peptide", last_is_peptide=False):
    """Nested booleans aligned with ``residue_label``: True on peptide residues."""
    if is_residue_list(labels):
        if types is None:
            is_pep = bool(last_is_peptide)
        else:
            is_pep = str(types).lower() == peptide_type
        return [is_pep] * len(labels)
    if types is None:
        n = len(labels)
        return [
            peptide_flags(lab, None, peptide_type, last_is_peptide=(i == n - 1))
            for i, lab in enumerate(labels)
        ]
    return [
        peptide_flags(lab, typ, peptide_type) for typ, lab in zip(types, labels)
    ]


def flatten_flags(flags):
    out = []

    def walk(tree):
        if not tree:
            return
        if isinstance(tree[0], (bool, np.bool_)):
            out.extend(bool(flag) for flag in tree)
            return
        for child in tree:
            walk(child)

    walk(flags)
    return out


def _is_bool_list(value):
    return bool(value) and isinstance(value[0], (bool, np.bool_))


def _is_chain_group(flags):
    return bool(flags) and isinstance(flags[0], list) and _is_bool_list(flags[0])


def _nearest_to_peptide(flags, positions):
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    pep = np.asarray(flags, dtype=bool)
    n = len(flags)
    if coords.shape[0] != n:
        raise ValueError(
            f"pocket positions {coords.shape[0]} != residue flags {n}"
        )
    if (not pep.any()) or pep.all() or n == 0:
        return np.full(n, np.inf, dtype=np.float64)
    nearest = cdist(coords, coords[pep]).min(axis=1)
    nearest = np.asarray(nearest, dtype=np.float64)
    nearest[pep] = np.inf
    return nearest


def _pocket_leaf(flags, positions, radius):
    nearest = _nearest_to_peptide(flags, positions)
    pep = np.asarray(flags, dtype=bool)
    return ((~pep) & np.isfinite(nearest) & (nearest <= radius)).tolist()


def _distance_leaf(flags, positions, radius=None):
    return _nearest_to_peptide(flags, positions).tolist()


def pocket_mask(labels, types, positions, radius, peptide_type="peptide"):
    """Protein residues whose Cα is within ``radius`` Å of any peptide Cα."""
    flags = peptide_flags(labels, types, peptide_type)
    return _map_peptide_tree(flags, positions, _pocket_leaf, float(radius))


def pocket_distances(labels, types, positions, peptide_type="peptide"):
    """Nearest peptide Cα distance (Å) for each protein residue; inf on peptide."""
    flags = peptide_flags(labels, types, peptide_type)
    return _map_peptide_tree(flags, positions, _distance_leaf, None)


def _map_peptide_tree(flags, positions, leaf_fn, radius):
    if _is_bool_list(flags):
        return leaf_fn(flags, positions, radius)
    if _is_chain_group(flags):
        lengths = [len(chain) for chain in flags]
        flat_flags = [flag for chain in flags for flag in chain]
        flat_pos = [pos for chain in positions for pos in chain]
        values = leaf_fn(flat_flags, flat_pos, radius)
        out = []
        start = 0
        for length in lengths:
            out.append(values[start : start + length])
            start += length
        return out
    return [
        _map_peptide_tree(flag, pos, leaf_fn, radius)
        for flag, pos in zip(flags, positions)
    ]
