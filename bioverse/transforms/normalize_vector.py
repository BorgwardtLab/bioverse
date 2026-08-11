import awkward as ak
import numpy as np

from ..transform import Transform


def _vector_norm(vectors):
    norm = np.sqrt(ak.sum(vectors * vectors, axis=-1))
    return ak.where(norm == 0, 1, norm)


class NormalizeVector(Transform):

    """L2-normalize vector features."""

    def __init__(self, field):
        self.field = field

    def transform_batch(self, batch):
        vectors = batch.__getattr__(self.field)
        norm = _vector_norm(vectors)
        batch.__setattr__(self.field, vectors / norm[..., np.newaxis])
        return batch
