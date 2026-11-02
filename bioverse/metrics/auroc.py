import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities.array import cumsum, diff


def _as_numpy(array):
    return np.asarray(ak.to_list(array), dtype=float).reshape(-1)


class AurocMetric(Metric):
    """Area under the receiver operating characteristic curve."""

    better = "higher"

    def __init__(self, name="AUROC", reduction="nanmean", **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, int)
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        y_true_sorted = y_true[order]
        n_pos = ak.sum(y_true_sorted, axis=-1)
        n_neg = ak.sum(1 - y_true_sorted, axis=-1)
        tp = cumsum(y_true_sorted)
        fp = cumsum(1 - y_true_sorted)
        tn = ak.sum(1 - y_true_sorted, axis=-1, keepdims=True) - fp
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr = fp / (fp + tn)
            tpr = tp / ak.sum(y_true_sorted, axis=-1, keepdims=True)
            delta_fpr = diff(fpr)
            tpr_mid = (tpr[..., :-1] + tpr[..., 1:]) / 2
            auroc = ak.sum(delta_fpr * tpr_mid, axis=-1)
        auroc = _as_numpy(auroc)
        valid = (_as_numpy(n_pos) > 0) & (_as_numpy(n_neg) > 0)
        auroc = np.where(valid, auroc, np.nan)
        return auroc if auroc.size > 1 else float(auroc[0])
