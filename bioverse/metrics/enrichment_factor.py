import numpy as np

from ..metric import Metric


class EnrichmentFactorMetric(Metric):
    better = "higher"

    def __init__(self, name="Enrichment Factor", cutoff_fraction=0.2, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cutoff_fraction = cutoff_fraction

    def compute(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=object)
        y_pred = np.asarray(y_pred, dtype=object)
        efs = []
        for labels, scores in zip(y_true, y_pred):
            labels = np.asarray(labels, dtype=bool)
            scores = np.asarray(scores, dtype=float)
            n_actives = int(labels.sum())
            if n_actives == 0 or len(labels) == 0:
                continue
            order = np.argsort(scores)[::-1]
            screened = labels[order]
            sel = max(int(len(labels) * self.cutoff_fraction), 1)
            n_actives_in_sel = int(screened[:sel].sum())
            efs.append((n_actives_in_sel / sel) / (n_actives / len(labels)))
        return float(np.mean(efs)) if len(efs) else 0.0
