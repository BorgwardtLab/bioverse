import numpy as np

from ..metric import Metric


class MeanActiveRankMetric(Metric):
    """Mean rank of actives in virtual screening."""

    better = "higher"

    def __init__(self, name="Mean Active Rank", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mars = []
        for labels, scores in zip(y_true, y_pred):
            labels = np.asarray(labels, dtype=bool)
            scores = np.asarray(scores, dtype=float)
            if labels.sum() == 0:
                continue
            order = np.argsort(scores)[::-1]
            screened = labels[order]
            ranks = [i for i, active in enumerate(screened) if active]
            mars.append(1 - (np.mean(ranks) / len(labels)))
        return float(np.mean(mars)) if len(mars) else 0.0
