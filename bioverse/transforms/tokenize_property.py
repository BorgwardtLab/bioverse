import awkward as ak

from ..transform import Transform


class TokenizeProperty(Transform):

    """Tokenize properties into integer indices using dataset assets."""

    def __init__(self, field, alphabet):
        self.field = field
        self.alphabet = alphabet
        labels = alphabet if isinstance(alphabet, list) else list(alphabet)
        self.tokens = {str(label): idx for idx, label in enumerate(labels)}

    def transform_batch(self, batch):
        values = batch.__getattr__(self.field)
        flat = ak.flatten(values, axis=None)
        tokenized = ak.Array([self.tokens.get(str(label), -1) for label in flat])
        tokenized = ak.unflatten(tokenized, ak.num(values))
        if tokenized.ndim > 1 and ak.all(ak.num(tokenized, axis=-1) == 1):
            tokenized = ak.flatten(tokenized, axis=-1)
        batch.__setattr__(self.field, tokenized)
        return batch
