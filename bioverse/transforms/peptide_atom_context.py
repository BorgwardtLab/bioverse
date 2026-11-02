"""Turn peptide residues into atom-level pseudo-residues for hybrid graphs."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..data import Batch
from ..transform import Transform
from ..transforms.explode_ligand_atoms import _map_chains
from ..transforms.filter_chains import _chain_ids
from ..utilities import element_from_atom_name
from ..utilities.peptide import MASK_INDEX, flatten_flags, peptide_flags


BACKBONE = {"N", "CA", "C", "O", "OXT"}
GEOM_FIELDS = ("residue_dihedrals", "residue_bond_angles")


def _is_hydrogen(name, element):
    name = str(name).strip()
    if element == "H":
        return True
    if name.startswith("H"):
        return True
    return len(name) > 1 and name[0].isdigit() and "H" in name


def _keep_atom(name, element, sidechains):
    if _is_hydrogen(name, element):
        return False
    if sidechains:
        return True
    return str(name).strip() in BACKBONE


def _as_flat_bool(batch, name):
    if name not in batch.data:
        return None
    return np.asarray([bool(x) for x in ak.to_list(ak.ravel(batch.data[name]))], dtype=bool)


def _as_flat_array(batch, name):
    if name not in batch.data:
        return None
    values = np.asarray(ak.to_numpy(getattr(batch, name)))
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    return values


def _select_chains(batch, indices):
    if not indices:
        return None
    labels = batch.data["chain_label"]
    chain_axis = labels.ndim - 1
    slicer = (slice(None),) * chain_axis + (indices,)
    chain_level = batch.prefixes.index("chain")
    data = {}
    for col, value in batch.data.items():
        prefix = col.split("_")[0]
        if prefix in batch.prefixes and batch.prefixes.index(prefix) >= chain_level:
            data[col] = value[slicer]
        else:
            data[col] = value
    return Batch(data, prefixes=list(batch.prefixes), resolution=batch.resolution)


def _concat_chains(left, right):
    chain_axis = left.data["chain_label"].ndim - 1
    chain_level = left.prefixes.index("chain")
    data = {}
    for col in set(left.data) | set(right.data):
        if col not in left.data:
            data[col] = right.data[col]
            continue
        if col not in right.data:
            data[col] = left.data[col]
            continue
        prefix = col.split("_")[0]
        if prefix in left.prefixes and left.prefixes.index(prefix) >= chain_level:
            data[col] = ak.concatenate([left.data[col], right.data[col]], axis=chain_axis)
        else:
            data[col] = left.data[col]
    return Batch(data, prefixes=list(left.prefixes), resolution=left.resolution)


class PeptideAtomContext(Transform):
    """Explode peptide residues into heavy-atom nodes.

    Protein residues stay in Awkward. Only the peptide chain is materialized
    in Python. Set ``sidechains=False`` for backbone atoms only.
    """

    def __init__(self, sidechains=True, peptide_type="peptide"):
        self.sidechains = bool(sidechains)
        self.peptide_type = peptide_type

    def _peptide_mask(self, batch):
        flags = _as_flat_bool(batch, "residue_is_peptide")
        if flags is not None:
            return flags
        types = ak.to_list(batch.chains.chain_type) if "chain_type" in batch.data else None
        labels = ak.to_list(batch.chains.residue_label)
        return np.asarray(flatten_flags(peptide_flags(labels, types, self.peptide_type)), dtype=bool)

    def _snapshot_protein_geom(self, batch, peptide):
        snaps = {}
        for field in GEOM_FIELDS:
            values = _as_flat_array(batch, field)
            if values is None:
                continue
            if values.shape[0] != peptide.shape[0]:
                raise ValueError(f"{field} length {values.shape[0]} != n residues {peptide.shape[0]}")
            snaps[field] = values[~peptide]
        return snaps

    def _explode_peptide(self, batch):
        residue_cols = [key for key in batch.data if key.startswith("residue_")]
        atom_cols = [key for key in batch.data if key.startswith("atom_")]
        extra = [key for key in ("chain_type",) if key in batch.data]
        trees = {key: ak.to_list(batch.data[key]) for key in residue_cols + atom_cols + extra}
        chain_depth = batch.prefixes.index("chain")

        def explode_chains(nodes):
            out = {key: [] for key in nodes}
            out["chain_type"] = nodes["chain_type"]
            for chain_idx, chain_type in enumerate(nodes["chain_type"]):
                exploded = {key: [] for key in residue_cols + atom_cols}
                atom_labels = nodes["atom_label"][chain_idx]
                atom_pos = nodes["atom_pos"][chain_idx]
                atom_names = nodes["atom_name"][chain_idx] if "atom_name" in nodes else atom_labels
                parent = {key: nodes[key][chain_idx] for key in residue_cols}
                for res_idx, labels in enumerate(atom_labels):
                    names = atom_names[res_idx]
                    for atom_idx, label in enumerate(labels):
                        name = names[atom_idx] if atom_idx < len(names) else label
                        element = element_from_atom_name(label)
                        if not _keep_atom(name, element, self.sidechains):
                            continue
                        stripped = str(name).strip()
                        for key in residue_cols:
                            if key == "residue_label":
                                exploded[key].append(element)
                            elif key == "residue_pos":
                                exploded[key].append(atom_pos[res_idx][atom_idx])
                            elif key == "residue_token":
                                exploded[key].append(MASK_INDEX)
                            elif key == "residue_is_atom":
                                exploded[key].append(True)
                            elif key == "residue_atom_name":
                                exploded[key].append(stripped)
                            elif key == "residue_mask":
                                exploded[key].append(False)
                            else:
                                exploded[key].append(parent[key][res_idx])
                        for key in atom_cols:
                            value = nodes[key][chain_idx][res_idx][atom_idx]
                            if key == "atom_label":
                                exploded[key].append([element])
                            elif key == "atom_name":
                                exploded[key].append([stripped])
                            else:
                                exploded[key].append([value])
                for key in residue_cols + atom_cols:
                    out[key].append(exploded[key])
            return out

        out = _map_chains(trees, 0, chain_depth, explode_chains)
        for key in residue_cols + atom_cols:
            batch.data[key] = ak.Array(out[key])
        return batch

    def transform_batch(self, batch):
        if "chain_type" not in batch or "atom_label" not in batch:
            return batch
        peptide = self._peptide_mask(batch)
        geom = self._snapshot_protein_geom(batch, peptide)
        n_protein = int((~peptide).sum())
        types = _chain_ids(batch.data["chain_type"])
        pep_idx = [i for i, typ in enumerate(types) if typ == self.peptide_type]
        prot_idx = [i for i, typ in enumerate(types) if typ != self.peptide_type]
        if not pep_idx:
            return batch
        protein = _select_chains(batch, prot_idx) if prot_idx else None
        peptide_batch = _select_chains(batch, pep_idx)
        n_pep = int(ak.num(ak.ravel(peptide_batch.residue_label), axis=0))
        if "residue_is_atom" not in peptide_batch.data:
            peptide_batch.residue_is_atom = ak.Array([False] * n_pep)
        if "residue_atom_name" not in peptide_batch.data:
            peptide_batch.residue_atom_name = ak.Array([""] * n_pep)
        peptide_batch = self._explode_peptide(peptide_batch)
        merged = _concat_chains(protein, peptide_batch) if protein is not None else peptide_batch
        merged.toc["residue"] = ak.num(merged.data["residue_label"], axis=merged.prefixes.index("residue"))
        if "atom_label" in merged.data:
            merged.toc["atom"] = ak.num(merged.data["atom_label"], axis=merged.prefixes.index("atom"))
        n_res = int(ak.num(ak.ravel(merged.residue_label), axis=0))
        n_atom = n_res - n_protein
        if n_atom < 0:
            raise RuntimeError(f"Peptide explode shrank the graph: {n_res} < {n_protein} protein residues")
        is_atom = np.zeros(n_res, dtype=bool)
        is_atom[n_protein:] = True
        pep_names = [str(x).strip() for x in ak.to_list(ak.ravel(peptide_batch.residue_atom_name))]
        if len(pep_names) != n_atom:
            pep_names = [""] * n_atom
        names = [""] * n_protein + pep_names
        merged.residue_is_atom = ak.Array(is_atom.tolist())
        merged.residue_atom_name = ak.Array(names)
        pep_flags = np.zeros(n_res, dtype=bool)
        pep_flags[n_protein:] = True
        merged.residue_is_peptide = ak.Array(pep_flags.tolist())
        for field, values in geom.items():
            width = values.shape[1]
            padded = np.zeros((n_res, width), dtype=values.dtype)
            padded[:n_protein] = values
            setattr(merged, field, padded)
        return merged
