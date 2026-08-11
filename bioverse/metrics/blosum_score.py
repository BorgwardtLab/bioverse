import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities import BLOSUM62


class BlosumScoreMetric(Metric):
    """Average BLOSUM62 substitution score between true and predicted residues."""

    better = "higher"

    def __init__(self, name: str = "Blosum Score", on: int = 2, per: int = 1, **kwargs):
        super().__init__(name=name, on=on, per=per, **kwargs)

    def compute(self, y_true: ak.Array, y_pred: ak.Array):
        if y_pred.ndim > y_true.ndim:
            y_pred = ak.argmax(y_pred, axis=-1)

        true_flat = ak.to_numpy(ak.flatten(y_true, axis=-1))
        pred_flat = ak.to_numpy(ak.flatten(y_pred, axis=-1))
        scores_flat = BLOSUM62[true_flat, pred_flat]

        lengths = ak.to_numpy(ak.num(y_true, axis=-1))
        scores = ak.unflatten(scores_flat, lengths)
        return ak.mean(scores, axis=-1)
