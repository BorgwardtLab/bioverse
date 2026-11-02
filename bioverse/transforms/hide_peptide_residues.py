"""Replace peptide residue identity with the masking token. Structure stays."""

from __future__ import annotations

import awkward as ak

from ..transform import Transform
from ..utilities import MASKING_TOKEN
from ..utilities.peptide import MASK_INDEX, flatten_flags, peptide_flags


class HidePeptideResidues(Transform):
    """Hide peptide sequence while leaving coordinates and protein letters intact."""

    def __init__(self, mask=MASKING_TOKEN, peptide_type="peptide"):
        self.mask = mask
        self.peptide_type = peptide_type

    def transform_batch(self, batch):
        types = (
            ak.to_list(batch.chains.chain_type) if "chain_type" in batch.data else None
        )
        labels = ak.to_list(batch.chains.residue_label)
        flags = flatten_flags(peptide_flags(labels, types, self.peptide_type))
        residue_labels = list(batch.residue_label)
        if len(flags) != len(residue_labels):
            raise ValueError(
                f"peptide flags {len(flags)} != n residues {len(residue_labels)}"
            )
        batch.residue_label = ak.Array(
            [self.mask if flag else label for flag, label in zip(flags, residue_labels)]
        )
        if "residue_token" in batch:
            tokens = list(batch.residue_token)
            batch.residue_token = ak.Array(
                [
                    MASK_INDEX if flag else int(token)
                    for flag, token in zip(flags, tokens)
                ]
            )
        return batch
