import awkward as ak

from ..metric import Metric


class BalancedMultiClassAccuracyMetric(Metric):
    better = "higher"

    def __init__(self, name="Bal.Acc.", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_pred = ak.argmax(y_pred, axis=-1)
        tp = ak.sum(y_true == y_pred, axis=-1)
        fn = ak.sum((y_true != y_pred) & (y_true == 1), axis=-1)
        tn = ak.sum(y_true == y_pred, axis=-1)
        fp = ak.sum((y_true != y_pred) & (y_true == 0), axis=-1)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return (sensitivity + specificity) / 2
