import awkward as ak
import numpy as np

from ..metric import Metric


class PerplexityMetric(Metric):
    better = "lower"

    def __init__(self, name="Perplexity", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_pred = ak.softmax(y_pred, axis=-1)
        cross_entropy = -ak.sum(y_true * np.log(y_pred), axis=-1)
        total = ak.sum(y_true, axis=-1)
        return np.exp(cross_entropy / total)
