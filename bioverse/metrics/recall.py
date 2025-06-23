import awkward as ak

from ..metric import Metric


class RecallMetric(Metric):
    better = "higher"

    def __init__(self, name="Recall", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_pred = y_pred > 0.5
        y_true = ak.values_astype(y_true, bool)
        tp = ak.sum((y_true == 1) & (y_pred == 1), axis=-1)
        fn = ak.sum((y_true == 1) & (y_pred == 0), axis=-1)
        recall = tp / (tp + fn)
        return recall
