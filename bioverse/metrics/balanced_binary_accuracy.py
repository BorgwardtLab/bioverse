import awkward as ak

from ..metric import Metric


class BalancedBinaryAccuracyMetric(Metric):
    better = "higher"

    def __init__(self, name="Bal.Acc.", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, bool)
        y_pred = y_pred >= self.threshold
        tp = ak.sum(y_true & y_pred, axis=-1)
        tn = ak.sum(~y_true & ~y_pred, axis=-1)
        fp = ak.sum(~y_true & y_pred, axis=-1)
        fn = ak.sum(y_true & ~y_pred, axis=-1)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return (sensitivity + specificity) / 2
