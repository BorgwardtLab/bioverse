import awkward as ak

from ..transform import Transform
from ..utilities import BIOCHEMICAL_ATOM_ALPHABET


class TokenizeAtoms(Transform):

    def __init__(self, alphabet=BIOCHEMICAL_ATOM_ALPHABET):
        self.alphabet = alphabet
        self.tokens = {aa: idx for idx, aa in enumerate(alphabet)}

    def transform_batch(self, batch):
        batch.atom_token = ak.Array(
            [self.tokens.get(label, -1) for label in batch.atom_label]
        )
        return batch
