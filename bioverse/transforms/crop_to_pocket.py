"""Keep peptide residues and the protein pocket; drop the rest of the protease."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..data import Batch
from ..transform import Transform
from ..utilities.peptide import flatten_flags, peptide_flags, pocket_mask


def _or_nested(left, right):
    if not left:
        return right
    if isinstance(left[0], (bool, np.bool_)):
        return [bool(a) or bool(b) for a, b in zip(left, right)]
    return [_or_nested(a, b) for a, b in zip(left, right)]


def _as_flat_bool(batch, name):
    if name not in batch.data:
        return None
    return np.asarray([bool(x) for x in ak.to_list(ak.ravel(batch.data[name]))], dtype=bool)


def _mol_residue_counts(batch):
    counts = batch.toc["residue"]
    if getattr(counts, "ndim", 0) == 0:
        return np.asarray([int(counts)], dtype=np.int64)
    per_mol = ak.sum(counts, axis=-1)
    return np.asarray(ak.to_numpy(ak.ravel(per_mol)), dtype=np.int64)


def _ensure_nonempty_molecules(keep, mol_counts):
    keep = np.asarray(keep, dtype=bool).copy()
    start = 0
    for n in mol_counts:
        n = int(n)
        if n > 0 and not keep[start : start + n].any():
            keep[start : start + n] = True
        start += n
    return keep


def _ensure_protein_kept(keep, protein, mol_counts):
    """Do not leave a molecule as peptide-only; that yields empty IF targets."""
    keep = np.asarray(keep, dtype=bool).copy()
    protein = np.asarray(protein, dtype=bool)
    if protein.shape[0] != keep.shape[0]:
        return keep
    start = 0
    for n in mol_counts:
        n = int(n)
        sl = slice(start, start + n)
        if protein[sl].any() and not (keep[sl] & protein[sl]).any():
            keep[sl] = keep[sl] | protein[sl]
        start += n
    return keep


def _fill_like_labels(labels, flat, idx):
    if not isinstance(labels, list):
        return labels
    if labels and isinstance(labels[0], str):
        n = len(labels)
        out = [bool(flat[idx[0] + j]) for j in range(n)]
        idx[0] += n
        return out
    return [_fill_like_labels(lab, flat, idx) for lab in labels]


def _filter_by_labels(values, labels, keep, idx):
    """Drop residue leaves in ``values`` using the ``residue_label`` tree as schema."""
    if not isinstance(labels, list):
        return values
    if labels and isinstance(labels[0], str):
        n = len(labels)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(
                f"residue leaf length {0 if not isinstance(values, list) else len(values)} != {n}"
            )
        out = [values[j] for j in range(n) if keep[idx[0] + j]]
        idx[0] += n
        return out
    if not isinstance(values, list):
        raise ValueError("crop column structure does not match residue_label")
    if len(values) == len(labels):
        return [_filter_by_labels(v, lab, keep, idx) for v, lab in zip(values, labels)]
    if len(values) == 1:
        return [_filter_by_labels(values[0], labels, keep, idx)]
    raise ValueError("crop column structure does not match residue_label")


class CropToPocket(Transform):
    """Drop non-pocket protein residues.

    Remaining nodes are the peptide (or peptide atoms, after explode) plus every
    protein residue whose Cα is within ``radius`` Å of a peptide Cα. Apply this
    *before* kNN so the graph is built on the cropped complex.
    """

    def __init__(self, radius=8.0, peptide_type="peptide"):
        self.radius = float(radius)
        self.peptide_type = peptide_type

    def _keep_hybrid_slice(self, is_atom, names, pos):
        protein = ~is_atom
        pep_ca = is_atom & (np.asarray(names) == "CA")
        if (not protein.any()) or (not pep_ca.any()):
            return np.ones(is_atom.shape[0], dtype=bool)
        nearest = np.linalg.norm(pos[protein, None, :] - pos[None, pep_ca, :], axis=-1).min(axis=1)
        keep = np.zeros(is_atom.shape[0], dtype=bool)
        keep[np.flatnonzero(protein)] = nearest <= self.radius
        keep |= is_atom
        if not keep.any():
            keep[:] = True
        return keep

    def _keep_hybrid(self, batch, n_res):
        is_atom = _as_flat_bool(batch, "residue_is_atom")
        names = (
            [str(x).strip() for x in ak.to_list(ak.ravel(batch.residue_atom_name))]
            if "residue_atom_name" in batch.data
            else [""] * n_res
        )
        pos = np.asarray(ak.to_list(ak.ravel(batch.residue_pos)), dtype=np.float64).reshape(-1, 3)
        if is_atom is None or pos.shape[0] != n_res:
            return None
        mol_counts = _mol_residue_counts(batch)
        if int(mol_counts.sum()) != n_res:
            return self._keep_hybrid_slice(is_atom, names, pos)
        keep = np.zeros(n_res, dtype=bool)
        start = 0
        for n in mol_counts:
            n = int(n)
            sl = slice(start, start + n)
            keep[sl] = self._keep_hybrid_slice(is_atom[sl], names[sl], pos[sl])
            start += n
        return keep

    def _keep_residue(self, batch, n_res):
        if "residue_pos" not in batch.data:
            return None
        labels = ak.to_list(batch.data["residue_label"])
        types = ak.to_list(batch.data["chain_type"]) if "chain_type" in batch.data else None
        positions = ak.to_list(batch.data["residue_pos"])
        pep = peptide_flags(labels, types, self.peptide_type)
        pocket = pocket_mask(labels, types, positions, self.radius, peptide_type=self.peptide_type)
        keep = flatten_flags(_or_nested(pocket, pep))
        if len(keep) != n_res:
            pep_flat = _as_flat_bool(batch, "residue_is_peptide")
            pos = np.asarray(ak.to_list(ak.ravel(batch.residue_pos)), dtype=np.float64).reshape(-1, 3)
            if pep_flat is None or pos.shape[0] != n_res:
                return None
            if (not pep_flat.any()) or pep_flat.all():
                return None
            mol_counts = _mol_residue_counts(batch)
            keep_arr = pep_flat.copy()
            if int(mol_counts.sum()) != n_res:
                nearest = np.linalg.norm(pos[~pep_flat, None, :] - pos[None, pep_flat, :], axis=-1).min(axis=1)
                keep_arr[~pep_flat] = nearest <= self.radius
                return keep_arr
            start = 0
            for n in mol_counts:
                n = int(n)
                sl = slice(start, start + n)
                pep_sl = pep_flat[sl]
                if pep_sl.any() and (not pep_sl.all()):
                    nearest = np.linalg.norm(pos[sl][~pep_sl, None, :] - pos[sl][None, pep_sl, :], axis=-1).min(axis=1)
                    keep_sl = pep_sl.copy()
                    keep_sl[~pep_sl] = nearest <= self.radius
                    keep_arr[sl] = keep_sl
                start += n
            return keep_arr
        return np.asarray(keep, dtype=bool)

    def _pocket_flags(self, batch, keep):
        keep = np.asarray(keep, dtype=bool)
        if "residue_is_atom" in batch.data:
            is_atom = _as_flat_bool(batch, "residue_is_atom")
            if is_atom is not None and is_atom.shape[0] == keep.shape[0]:
                return keep & ~is_atom
        pep = _as_flat_bool(batch, "residue_is_peptide")
        if pep is not None and pep.shape[0] == keep.shape[0]:
            return keep & ~pep
        labels = ak.to_list(batch.data["residue_label"])
        types = ak.to_list(batch.data["chain_type"]) if "chain_type" in batch.data else None
        pep_f = np.asarray(flatten_flags(peptide_flags(labels, types, self.peptide_type)), dtype=bool)
        if pep_f.shape[0] == keep.shape[0]:
            return keep & ~pep_f
        return keep.copy()

    def _apply_keep_tree(self, batch, keep, pocket=None):
        n_res = int(keep.shape[0])
        labels = ak.to_list(batch.data["residue_label"])
        residue_level = batch.prefixes.index("residue")
        new_data = {}
        for col, arr in batch.data.items():
            prefix = col.split("_")[0]
            if prefix not in batch.prefixes or batch.prefixes.index(prefix) < residue_level:
                new_data[col] = arr
                continue
            idx = [0]
            filtered = _filter_by_labels(ak.to_list(arr), labels, keep, idx)
            if idx[0] != n_res:
                raise ValueError(f"{col} consumed {idx[0]} residue slots, expected {n_res}")
            new_data[col] = ak.Array(filtered)
        if pocket is not None:
            filtered_labels = ak.to_list(new_data["residue_label"])
            kept_pocket = np.asarray(pocket, dtype=bool)[keep]
            idx = [0]
            nested = _fill_like_labels(filtered_labels, kept_pocket, idx)
            new_data["residue_is_pocket"] = ak.Array(nested)
        return Batch(new_data, prefixes=list(batch.prefixes), resolution=batch.resolution)

    def _apply_keep(self, batch, keep, pocket=None):
        keep = np.asarray(keep, dtype=bool)
        return self._apply_keep_tree(batch, keep, pocket=pocket)

    def transform_batch(self, batch):
        if "residue_pos" not in batch.data:
            from .residue_positions import ResiduePositions

            batch = ResiduePositions(mode="CA").transform_batch(batch)
        n_res = int(ak.num(ak.ravel(batch.residue_label), axis=0))
        if n_res == 0:
            return batch
        if "residue_is_atom" in batch.data:
            keep = self._keep_hybrid(batch, n_res)
        else:
            keep = self._keep_residue(batch, n_res)
        if keep is None:
            return batch
        mol_counts = _mol_residue_counts(batch)
        keep = _ensure_nonempty_molecules(keep, mol_counts)
        protein = None
        if "residue_is_atom" in batch.data:
            is_atom = _as_flat_bool(batch, "residue_is_atom")
            if is_atom is not None and is_atom.shape[0] == keep.shape[0]:
                protein = ~is_atom
        if protein is None:
            pep = _as_flat_bool(batch, "residue_is_peptide")
            if pep is not None and pep.shape[0] == keep.shape[0]:
                protein = ~pep
        if protein is not None:
            keep = _ensure_protein_kept(keep, protein, mol_counts)
        if int(keep.sum()) == 0:
            return batch
        pocket = self._pocket_flags(batch, keep)
        if int(keep.sum()) == n_res:
            batch.residues.residue_is_pocket = ak.Array(np.asarray(pocket, dtype=bool).tolist())
            return batch
        return self._apply_keep(batch, keep, pocket=pocket)
