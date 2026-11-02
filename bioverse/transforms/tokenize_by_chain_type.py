import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import BIOCHEMICAL_ATOM_ALPHABET, PROTEIN_ALPHABET, UNK_TOKEN


class TokenizeByChainType(Transform):
    """Tokenize residues with disjoint protein and ligand alphabets.

    Protein chains use ``protein_alphabet`` (plus UNK). Ligand chains use
    ``ligand_alphabet`` with indices offset so cysteine ``C`` and carbon
    ``C`` are different tokens. Writes matching ``residue_features``.
    """

    def __init__(
        self,
        protein_alphabet=PROTEIN_ALPHABET,
        ligand_alphabet=BIOCHEMICAL_ATOM_ALPHABET,
        protein_type="protein",
        ligand_type="ligand",
        unk=UNK_TOKEN,
    ):
        self.protein_alphabet = protein_alphabet
        self.ligand_alphabet = ligand_alphabet
        self.protein_type = protein_type
        self.ligand_type = ligand_type
        self.unk = unk

        protein = list(protein_alphabet)
        if unk not in protein:
            protein.append(unk)
        ligand = list(ligand_alphabet)
        if unk not in ligand:
            ligand.append(unk)

        self.protein_tokens = {label: idx for idx, label in enumerate(protein)}
        self.ligand_offset = len(protein)
        self.ligand_tokens = {
            label: self.ligand_offset + idx for idx, label in enumerate(ligand)
        }
        self.protein_unk = self.protein_tokens[unk]
        self.ligand_unk = self.ligand_tokens[unk]
        self.n_tokens = self.ligand_offset + len(ligand)

    def transform_assets(self, assets):
        assets["residue_features"] = np.eye(self.n_tokens).tolist()
        return assets

    def transform_batch(self, batch):
        if "chain_type" in batch:
            types = ak.to_list(batch.chains.chain_type)
            labels = ak.to_list(batch.chains.residue_label)
            tokens = []
            for chain_type, residue_labels in zip(types, labels):
                if chain_type == self.ligand_type:
                    table, default = self.ligand_tokens, self.ligand_unk
                else:
                    table, default = self.protein_tokens, self.protein_unk
                tokens.extend(table.get(label, default) for label in residue_labels)
        else:
            tokens = [
                self.protein_tokens.get(label, self.protein_unk)
                for label in batch.residue_label
            ]
        batch.residue_token = ak.Array(tokens)
        return batch
