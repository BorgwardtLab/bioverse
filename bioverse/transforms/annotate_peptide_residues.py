"""Broadcast chain-level peptide vs protein labels onto residues."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities.peptide import flatten_flags, peptide_flags


class AnnotatePeptideResidues(Transform):
    """Write ``residue_is_peptide`` from ``chain_type`` (last chain if missing)."""

    def __init__(self, peptide_type="peptide"):
        self.peptide_type = peptide_type

    def transform_batch(self, batch):
        labels = ak.to_list(batch.data["residue_label"])
        types = ak.to_list(batch.data["chain_type"]) if "chain_type" in batch.data else None
        flags = np.asarray(
            flatten_flags(peptide_flags(labels, types, self.peptide_type)),
            dtype=bool,
        )
        n_res = int(ak.num(ak.ravel(batch.data["residue_label"]), axis=0))
        if len(flags) != n_res:
            raise ValueError(
                f"residue_is_peptide length {len(flags)} != n residues {n_res}"
            )
        batch.residues.residue_is_peptide = ak.Array(flags)
        return batch
