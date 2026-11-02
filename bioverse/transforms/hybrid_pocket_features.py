"""Typed residue features for protein residues + peptide atom nodes."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import element_from_atom_name


ELEMENT_ALPHABET = ("C", "N", "O", "S", "P", "H", "X")
GEOM_WIDTH = 12
FEATURE_WIDTH = GEOM_WIDTH + len(ELEMENT_ALPHABET) + 2


def _flat_bool(batch, name):
    if name not in batch.data:
        return None
    return np.asarray([bool(x) for x in ak.to_list(ak.ravel(batch.data[name]))], dtype=bool)


def _flat_labels(batch, name):
    return [str(x) for x in ak.to_list(ak.ravel(batch.data[name]))]


def _flat_geom(batch, name, n_res, width):
    if name not in batch.data:
        return np.zeros((n_res, width), dtype=np.float32)
    values = np.asarray(ak.to_numpy(getattr(batch, name)), dtype=np.float32)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 1:
        values = values.reshape(n_res, -1)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.shape[0] != n_res:
        raise ValueError(f"{name} shape {values.shape} != ({n_res}, *)")
    if values.shape[1] < width:
        pad = np.zeros((n_res, width), dtype=np.float32)
        pad[:, : values.shape[1]] = values
        return pad
    return values[:, :width]


class HybridPocketFeatures(Transform):
    """Write ``residue_features``: protein geometry + peptide element one-hot."""

    def __init__(self, element_alphabet=ELEMENT_ALPHABET):
        self.element_alphabet = tuple(element_alphabet)
        self.element_index = {sym: i for i, sym in enumerate(self.element_alphabet)}
        self.unk = self.element_index.get("X", len(self.element_alphabet) - 1)

    def transform_batch(self, batch):
        labels = _flat_labels(batch, "residue_label")
        n_res = len(labels)
        is_atom = _flat_bool(batch, "residue_is_atom")
        if is_atom is None:
            is_atom = np.zeros(n_res, dtype=bool)
        dih = _flat_geom(batch, "residue_dihedrals", n_res, 6)
        ang = _flat_geom(batch, "residue_bond_angles", n_res, 6)
        features = np.zeros((n_res, FEATURE_WIDTH), dtype=np.float32)
        features[:, :6] = dih
        features[:, 6:12] = ang
        for i, (atom, label) in enumerate(zip(is_atom, labels)):
            if atom:
                element = element_from_atom_name(label)
                features[i, GEOM_WIDTH + self.element_index.get(element, self.unk)] = 1.0
                features[i, -1] = 1.0
            else:
                features[i, -2] = 1.0
        batch.residue_features = features
        return batch
