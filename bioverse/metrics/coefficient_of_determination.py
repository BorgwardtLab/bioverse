import awkward as ak

from ..metric import Metric


class CoefficientOfDeterminationMetric(Metric):
    better = "higher"

    def __init__(self, name="R2", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        ss_res = ak.sum((y_true - y_pred) ** 2, axis=-1)
        ss_tot = ak.sum(
            (y_true - ak.mean(y_true, axis=-1, keepdims=True)) ** 2, axis=-1
        )
        r2 = 1 - ss_res / ss_tot
        return r2
