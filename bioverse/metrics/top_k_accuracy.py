import awkward as ak

from ..metric import Metric


class TopKAccuracyMetric(Metric):
    better = "higher"

    def __init__(self, name="Accuracy", k=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k

    def compute(self, y_true, y_pred):
        topk_preds = ak.argsort(y_pred, axis=-1, ascending=False)[:, :, : self.k]
        total = ak.num(y_true, axis=-1)
        y_true = ak.broadcast_arrays(y_true, topk_preds)[0]
        correct = ak.sum(ak.any(y_true == topk_preds, axis=-1), axis=-1)
        topk_accuracy = correct / total
        return topk_accuracy
