import awkward as ak

from ..metric import Metric
from ..utilities.array import cumsum, diff


class AurocMetric(Metric):
    better = "higher"

    def __init__(self, name="AUROC", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, int)
        # Sort by predicted values
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        y_true_sorted = y_true[order]

        # Calculate true positives, false positives, and true negatives
        tp = cumsum(y_true_sorted)
        fp = cumsum(1 - y_true_sorted)
        tn = ak.sum(1 - y_true_sorted, axis=-1, keepdims=True) - fp

        # Calculate false positive rate and true positive rate
        fpr = fp / (fp + tn)
        tpr = tp / ak.sum(y_true_sorted, axis=-1, keepdims=True)

        # Calculate AUROC using the trapezoidal rule
        delta_fpr = diff(fpr)
        tpr_mid = (tpr[..., :-1] + tpr[..., 1:]) / 2
        auroc = ak.sum(delta_fpr * tpr_mid, axis=-1)
        return auroc
