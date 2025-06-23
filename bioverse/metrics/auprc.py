import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities.array import cumsum, diff


class AuprcMetric(Metric):
    better = "higher"

    def __init__(self, name="AUPRC", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, int)
        # Sort by predicted values
        desc_order = ak.argsort(y_pred, axis=-1, ascending=False)
        y_true_sorted = y_true[ak.unzip(desc_order)]

        # Calculate true positives and false positives
        tp = cumsum(y_true_sorted)
        fp = cumsum(1 - y_true_sorted)
        total_pos = ak.sum(y_true_sorted, axis=-1, keepdims=True)
        precision = tp / (tp + fp)
        recall = tp / total_pos

        # Prepend the start point (recall=0, precision=1)
        precision = ak.concatenate(
            [ak.ones_like(precision[..., :1]), precision], axis=-1
        )
        recall = ak.concatenate([ak.zeros_like(recall[..., :1]), recall], axis=-1)

        # Calculate AUPRC using the trapezoidal rule
        delta_recall = diff(recall)
        precision_mid = (precision[..., :-1] + precision[..., 1:]) / 2
        auprc = ak.sum(delta_recall * precision_mid, axis=-1)

        # Handle case where there are no positive samples
        n_pos = ak.sum(y_true_sorted, axis=-1)
        auprc = auprc * (n_pos > 0)

        return auprc
