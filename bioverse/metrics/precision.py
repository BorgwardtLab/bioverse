import awkward as ak

from ..metric import Metric


class PrecisionMetric(Metric):
    better = "higher"

    def __init__(self, name="Precision", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_pred = y_pred > 0.5
        y_true = ak.values_astype(y_true, bool)
        tp = ak.sum(y_true & y_pred, axis=-1)
        fp = ak.sum(~y_true & y_pred, axis=-1)
        precision = tp / (tp + fp)
        return precision
