import awkward as ak
import numpy as np

from ..metric import Metric


class SpearmansRhoMetric(Metric):
    """Spearman rank correlation between predictions and targets."""

    better = "higher"

    def __init__(self, name="Spearman", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.to_numpy(ak.ravel(y_true))
        y_pred = ak.to_numpy(ak.ravel(y_pred))

        def rank(values):
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
            return ranks

        y_true_rank = rank(y_true)
        y_pred_rank = rank(y_pred)
        d = y_true_rank - y_pred_rank
        n = len(y_true)
        if n < 2:
            return float("nan")
        return float(1 - 6 * np.sum(d * d) / (n * (n * n - 1)))
