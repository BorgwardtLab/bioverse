import awkward as ak
import numpy as np

from ..metric import Metric


def _unit_vectors(vectors):
    norm = np.sqrt(ak.sum(vectors * vectors, axis=-1))
    norm = ak.where(norm == 0, 1, norm)
    return vectors / norm[..., np.newaxis]


class MeanAngularErrorMetric(Metric):
    """Mean angular error between predicted and true vectors."""

    better = "lower"

    def __init__(self, name="Angular MAE", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = _unit_vectors(y_true)
        y_pred = _unit_vectors(y_pred)
        cosine = ak.sum(y_true * y_pred, axis=-1)
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))
