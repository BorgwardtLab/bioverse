import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities.array import cumsum, diff


def _as_numpy(array):
    return np.asarray(ak.to_list(array), dtype=float).reshape(-1)


class AuprcMetric(Metric):
    """Area under the precision-recall curve."""

    better = "higher"

    def __init__(self, name="AUPRC", reduction="nanmean", **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, int)
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        y_true_sorted = y_true[order]
        n_pos = ak.sum(y_true_sorted, axis=-1)
        tp = cumsum(y_true_sorted)
        fp = cumsum(1 - y_true_sorted)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = tp / (tp + fp)
            recall = tp / ak.sum(y_true_sorted, axis=-1, keepdims=True)
            precision = ak.concatenate(
                [ak.ones_like(precision[..., :1]), precision], axis=-1
            )
            recall = ak.concatenate([ak.zeros_like(recall[..., :1]), recall], axis=-1)
            delta_recall = diff(recall)
            precision_mid = (precision[..., :-1] + precision[..., 1:]) / 2
            auprc = ak.sum(delta_recall * precision_mid, axis=-1)
        auprc = _as_numpy(auprc)
        auprc = np.where(_as_numpy(n_pos) > 0, auprc, np.nan)
        return auprc if auprc.size > 1 else float(auprc[0])
