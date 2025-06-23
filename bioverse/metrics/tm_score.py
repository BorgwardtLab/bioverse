import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities import alignment, flatten, parallelize


class TMScoreMetric(Metric):
    better = "higher"

    def __init__(self, name="TM-Score", type="Protein", **kwargs):
        super().__init__(name=name, **kwargs)
        if type == "Protein":
            self.anchor = "CA"
        elif type == "RNA":
            self.anchor = "C3'"
        else:
            raise ValueError(f"Invalid type: {type}")

    def before_compute(self, y_true, y_pred):
        def job(item):
            yt, yp = item["0"], item["1"]
            pos_true, pos_pred = alignment(yt["residue_pos"], yp["target"])
            pos_true = flatten(pos_true, exclude=-1)
            pos_pred = flatten(pos_pred, exclude=-1)
            return pos_true, pos_pred

        pos = parallelize(job, ak.zip([y_true, y_pred]), progress=False)
        pos_true, pos_pred = list(zip(*pos))
        return ak.Array({"pos": pos_true}), ak.Array({"pos": pos_pred})

    def compute(self, y_true, y_pred):
        L = ak.num(y_true, axis=1)
        d0 = 0.6 * (L - 0.5) ** 0.5 - 2.5
        d0 = ak.where(L < 30, 0.7, d0)
        d0 = ak.where(L < 24, 0.6, d0)
        d0 = ak.where(L < 20, 0.5, d0)
        d0 = ak.where(L < 16, 0.4, d0)
        d0 = ak.where(L < 12, 0.3, d0)
        di = np.sqrt(ak.sum((y_true - y_pred) ** 2, axis=-1))
        tm_score = ak.max(ak.sum(1 / (1 + (di / d0) ** 2), axis=-1) / L, axis=-1)
        return tm_score
