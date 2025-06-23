import awkward as ak

from ..metric import Metric


class MultiLabelAccuracyMetric(Metric):
    better = "higher"

    def __init__(self, name="Accuracy", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold

    def compute(self, y_true, y_pred):
        y_pred = y_pred >= self.threshold
        y_true = ak.values_astype(y_true, bool)
        correct = ak.sum(y_true == y_pred, axis=-1)
        total = ak.num(y_true, axis=-1)
        accuracy = correct / total
        return accuracy
