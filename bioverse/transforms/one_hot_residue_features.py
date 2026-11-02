import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import PROTEIN_ALPHABET, UNK_TOKEN


class OneHotResidueFeatures(Transform):

    """One-hot encode residues features.

    ``transform_assets`` stores the identity table for VirtualBatch.featurize
    after an offline apply. ``transform_batch`` writes ``residue_onehot`` so
    live graphs can concatenate it with geometry.
    """

    def __init__(self, alphabet=PROTEIN_ALPHABET, unk=UNK_TOKEN):
        self.alphabet = alphabet
        self.unk = unk

    def _labels(self):
        labels = list(self.alphabet)
        if self.unk not in labels:
            labels.append(self.unk)
        return labels

    def transform_assets(self, assets):
        n = len(self._labels())
        if "features" not in assets:
            assets["residue_features"] = np.eye(n).tolist()
        else:
            assets["residue_features"] = np.concatenate(
                [
                    assets["residue_features"],
                    np.eye(n),
                ],
                axis=1,
            ).tolist()
        return assets

    def transform_batch(self, batch):
        if "residue_token" not in batch.data:
            return batch
        n = len(self._labels())
        tokens = np.asarray(ak.to_list(ak.ravel(batch.residue_token)), dtype=np.int64)
        if tokens.size == 0:
            return batch
        tokens = np.clip(tokens, 0, n - 1)
        batch.residues.residue_onehot = ak.Array(np.eye(n, dtype=np.float32)[tokens])
        return batch
