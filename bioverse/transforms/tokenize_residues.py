import awkward as ak

from ..transform import Transform
from ..utilities import PROTEIN_ALPHABET


class TokenizeResidues(Transform):

    def __init__(self, alphabet=PROTEIN_ALPHABET):
        self.alphabet = alphabet
        self.tokens = {aa: idx for idx, aa in enumerate(alphabet)}

    def transform_batch(self, batch):
        batch.residue_token = ak.Array(
            [self.tokens.get(label, -1) for label in batch.residue_label]
        )
        return batch
