"""Remove peptide residues so a receptor encoder never sees the ligand."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..transform import Transform
from ..transforms.crop_to_pocket import CropToPocket, _mol_residue_counts
from ..utilities.peptide import flatten_flags, peptide_flags


class DropPeptideResidues(Transform):
    """Keep protein residues only. Contact labels computed earlier are filtered too."""

    def __init__(self, peptide_type="peptide"):
        self.peptide_type = peptide_type
        self._crop = CropToPocket(radius=8.0, peptide_type=peptide_type)

    def transform_batch(self, batch):
        n_res = int(ak.num(ak.ravel(batch.residue_label), axis=0))
        if n_res == 0:
            return batch
        labels = ak.to_list(batch.data["residue_label"])
        types = ak.to_list(batch.data["chain_type"]) if "chain_type" in batch.data else None
        pep = np.asarray(
            flatten_flags(peptide_flags(labels, types, self.peptide_type)), dtype=bool
        )
        if pep.size != n_res:
            return batch
        if not pep.any():
            return batch
        keep = ~pep
        mol_counts = _mol_residue_counts(batch)
        start = 0
        keep = keep.copy()
        for n in mol_counts:
            n = int(n)
            sl = slice(start, start + n)
            if not keep[sl].any():
                keep[sl] = True
            start += n
        if int(keep.sum()) == n_res:
            return batch
        return self._crop._apply_keep(batch, keep)
