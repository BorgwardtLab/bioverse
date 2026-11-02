"""Masked prediction of protein residues that contact a bound peptide."""

import awkward as ak
import numpy as np

from ..task import Task
from ..transforms.hide_peptide_residues import HidePeptideResidues
from ..transforms.residue_positions import ResiduePositions
from ..utilities import PROTEIN_ALPHABET
from ..utilities.peptide import flatten_flags, pocket_mask


def _mol_residue_counts(batch):
    counts = batch.toc["residue"]
    if getattr(counts, "ndim", 0) == 0:
        return np.asarray([int(counts)], dtype=np.int64)
    per_mol = ak.sum(counts, axis=-1)
    return np.asarray(ak.to_numpy(ak.ravel(per_mol)), dtype=np.int64)


def _hybrid_pocket_flags(is_atom, names, pos, radius, mol_counts):
    n_res = int(is_atom.shape[0])
    flags = np.zeros(n_res, dtype=bool)
    names = np.asarray(names)
    if mol_counts is None or int(np.sum(mol_counts)) != n_res:
        mol_counts = np.asarray([n_res], dtype=np.int64)
    start = 0
    for n in mol_counts:
        n = int(n)
        sl = slice(start, start + n)
        protein = ~is_atom[sl]
        pep_ca = is_atom[sl] & (names[sl] == "CA")
        if protein.any() and pep_ca.any():
            nearest = np.linalg.norm(
                pos[sl][protein, None, :] - pos[sl][None, pep_ca, :], axis=-1
            ).min(axis=1)
            flags[start + np.flatnonzero(protein)] = nearest <= radius
        start += n
    return flags


class MaskedPocketPredictionTask(Task):
    """Mask protein pocket residues and predict their amino-acid identities.

    The pocket is every protein residue whose Cα is within ``radius`` Å of any
    peptide Cα. Peptide letters are optional context: set ``hide_peptide`` to
    replace them with the masking token after the targets are extracted.
    """

    def __init__(self, radius=8.0, hide_peptide=False, peptide_type="peptide"):
        codes = np.char.encode(
            ak.ravel(list(PROTEIN_ALPHABET)).to_list(), "utf-8"
        ).view(np.uint8)
        self.lookup = np.zeros((256,), dtype=np.int32)
        self.lookup[codes] = np.arange(len(PROTEIN_ALPHABET)).astype(np.int64)
        self.radius = float(radius)
        self.hide_peptide = bool(hide_peptide)
        self.peptide_type = peptide_type

    def _with_positions(self, X):
        if "residue_pos" not in X:
            X = ResiduePositions(mode="CA").transform_batch(X)
        return X

    def _labels_and_mask(self, X):
        targets = X.molecules.residue_label
        n_res = int(ak.num(ak.ravel(targets), axis=0))
        if "residue_is_pocket" in X.data:
            flags = np.asarray(
                [bool(x) for x in ak.to_list(ak.ravel(X.data["residue_is_pocket"]))],
                dtype=bool,
            )
            if flags.shape[0] == n_res:
                return targets, flags.tolist()
        if "residue_is_atom" in X.data:
            is_atom = np.asarray(
                [bool(x) for x in ak.to_list(ak.ravel(X.residue_is_atom))], dtype=bool
            )
            names = (
                [str(x).strip() for x in ak.to_list(ak.ravel(X.residue_atom_name))]
                if "residue_atom_name" in X.data
                else [""] * n_res
            )
            pos = np.asarray(ak.to_list(ak.ravel(X.residue_pos)), dtype=np.float64).reshape(-1, 3)
            if pos.shape[0] != n_res or is_atom.shape[0] != n_res:
                raise ValueError("hybrid pocket fields do not align with residue_label")
            flags = _hybrid_pocket_flags(
                is_atom, names, pos, self.radius, _mol_residue_counts(X)
            )
            return targets, flags.tolist()
        labels = ak.to_list(X.data["residue_label"])
        types = ak.to_list(X.data["chain_type"]) if "chain_type" in X.data else None
        nested_mask = pocket_mask(
            labels,
            types,
            ak.to_list(X.data["residue_pos"]) if "residue_pos" in X.data else ak.to_list(X.residue_pos),
            self.radius,
            peptide_type=self.peptide_type,
        )
        return targets, flatten_flags(nested_mask)

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = "residue"
        X = self._with_positions(X)
        targets, flat_mask = self._labels_and_mask(X)
        n_res = int(ak.num(ak.ravel(targets), axis=0))
        if len(flat_mask) != n_res:
            raise ValueError(f"pocket mask {len(flat_mask)} != n residues {n_res}")
        mask = ak.unflatten(ak.Array(flat_mask), ak.num(targets, axis=-1))
        codes = np.char.encode(ak.ravel(targets).to_list(), "utf-8").view(np.uint8)
        tokens = self.lookup[codes]
        y_tokens = ak.unflatten(tokens, ak.num(targets, axis=-1))[mask]
        while getattr(y_tokens, "ndim", 1) > 2:
            y_tokens = ak.flatten(y_tokens, axis=-1)
        y = ak.Array({"target": y_tokens})
        y["sizes"] = ak.num(y["target"], axis=-1)
        X.molecules.residue_mask = mask
        if self.hide_peptide:
            X = HidePeptideResidues(peptide_type=self.peptide_type).transform_batch(X)
        return X, y
