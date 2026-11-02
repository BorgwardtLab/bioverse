import awkward as ak
import numpy as np

from ..metric import Metric


class _Fenwick:
    def __init__(self, n: int):
        self.n = int(n)
        self.tree = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, index: int, value: int = 1) -> None:
        i = int(index) + 1
        while i <= self.n:
            self.tree[i] += value
            i += i & -i

    def prefix(self, end: int) -> int:
        """Sum of ``[0, end)``."""
        i = int(end)
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return int(total)


def pairwise_ranking_accuracy(true, pred) -> float:
    """Fraction of ordered label pairs whose predicted ranking matches.

    Ties in ``true`` are skipped. A pair with ``true[i] > true[j]`` is correct
    when ``pred[i] > pred[j]``. Predicted ties count as incorrect, so a
    constant predictor scores 0. Runs in ``O(n log n)`` memory and time.
    """
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    n = true.size
    if n < 2:
        return float("nan")
    order = np.argsort(true, kind="mergesort")
    true = true[order]
    pred = pred[order]
    pred_rank = np.unique(pred, return_inverse=True)[1]
    fenwick = _Fenwick(int(pred_rank.max()) + 1)
    comparable = 0
    correct = 0
    inserted = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and true[j] == true[i]:
            j += 1
        for k in range(i, j):
            comparable += inserted
            correct += fenwick.prefix(int(pred_rank[k]))
        for k in range(i, j):
            fenwick.add(int(pred_rank[k]))
        inserted += j - i
        i = j
    if comparable == 0:
        return float("nan")
    return float(correct / comparable)


class PairwiseRankingAccuracyMetric(Metric):
    """Fraction of ordered pairs whose predicted ranking matches the labels.

    Ties in ``y_true`` are skipped. A pair with ``y_true[i] > y_true[j]`` is
    correct when ``y_pred[i] > y_pred[j]``. Constant predictions score 0.
    """

    better = "higher"

    def __init__(self, name="Pairwise ranking", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        return pairwise_ranking_accuracy(
            ak.to_numpy(ak.ravel(y_true)),
            ak.to_numpy(ak.ravel(y_pred)),
        )
