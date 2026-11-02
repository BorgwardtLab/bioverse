import awkward as ak
import numpy as np

from ..metric import Metric


def _average_rank(values):
    """1-based ranks; ties receive the mean of their ordinal positions."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _pearson(left, right):
    left = left - left.mean()
    right = right - right.mean()
    denom = np.sqrt(np.sum(left * left) * np.sum(right * right))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(left * right) / denom)


class SpearmansRhoMetric(Metric):
    """Spearman rank correlation between predictions and targets.

    Ranks use the mid-rank (average) convention for ties. The coefficient is
    Pearson's r on those ranks, so a constant input is undefined (``nan``)
    rather than an artifact of table order.
    """

    better = "higher"

    def __init__(self, name="Spearman", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.to_numpy(ak.ravel(y_true))
        y_pred = ak.to_numpy(ak.ravel(y_pred))
        if len(y_true) < 2:
            return float("nan")
        return _pearson(_average_rank(y_true), _average_rank(y_pred))
