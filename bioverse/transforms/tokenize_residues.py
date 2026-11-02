import awkward as ak

from ..transform import Transform
from ..utilities import PROTEIN_ALPHABET, UNK_TOKEN


class TokenizeResidues(Transform):

    """Tokenize residues into integer indices using dataset assets."""

    def __init__(self, alphabet=PROTEIN_ALPHABET, unk=UNK_TOKEN):
        self.alphabet = alphabet
        self.unk = unk
        labels = list(alphabet)
        self.tokens = {aa: idx for idx, aa in enumerate(labels)}
        self.unk_index = self.tokens.setdefault(unk, len(labels))

    def transform_batch(self, batch):
        batch.residue_token = ak.Array(
            [self.tokens.get(label, self.unk_index) for label in batch.residue_label]
        )
        return batch
