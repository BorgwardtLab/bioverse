import awkward as ak

from ..metric import Metric


class SpearmansRhoMetric(Metric):
    better = "higher"

    def __init__(self, name="Spearman", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        print("metric")

        def rank(data):
            sorter = ak.argsort(data, axis=-1)
            inv = ak.argsort(sorter, axis=-1)
            ranks = ak.local_index(data, axis=-1) + 1
            return ranks[inv]

        y_true_rank = rank(y_true)
        y_pred_rank = rank(y_pred)

        d = y_true_rank - y_pred_rank
        d_squared = d * d

        n = ak.num(y_true, axis=-1)
        numerator = 6 * ak.sum(d_squared, axis=-1)
        denominator = n * (n * n - 1)

        spearmans_rho = 1 - numerator / denominator

        return spearmans_rho
