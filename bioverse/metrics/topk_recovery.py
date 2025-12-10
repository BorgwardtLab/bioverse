import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities import BLOSUM62


class TopkRecoveryMetric(Metric):
    """
    Recovery metric that gives credit when the predicted residue is within the
    top-k BLOSUM62 substitutions for the *true* residue.

    Ranking is based on BLOSUM62 scores per true residue, with ties sharing
    the same rank (i.e. identical scores are treated as the same rank).

    For k=1 this reduces to standard Recovery, since the true residue has the
    highest BLOSUM62 score with itself in the 20×20 amino-acid sub-matrix.
    """

    better = "higher"

    def __init__(
        self,
        name: str = "Top-k Recovery",
        k: int = 1,
        on: int = 2,
        per: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        k : int
            Number of BLOSUM-based substitution ranks to treat as correct.
            k=1 is equivalent to standard Recovery (exact match).
        on, per :
            Passed through to the base `Metric` to match `RecoveryMetric`
            behaviour (defaults: on=2, per=1).
        """
        super().__init__(name=name, on=on, per=per, **kwargs)
        if k < 1:
            raise ValueError("TopkRecoveryMetric requires k >= 1.")
        self.k = int(k)
        self.name = f"Top-{k} Recovery"

        # Precompute acceptable predicted labels for each true label based on
        # BLOSUM62 rows in PROTEIN_ALPHABET order.
        scores = np.asarray(BLOSUM62, dtype=float)  # [20, 20]
        num_labels = scores.shape[0]
        acceptable = np.zeros_like(scores, dtype=bool)

        for i in range(num_labels):
            row = scores[i]
            unique_vals = np.unique(row)[::-1]  # descending unique scores
            k_eff = min(self.k, len(unique_vals))
            thresh = unique_vals[k_eff - 1]
            acceptable[i] = row >= thresh

        self.acceptable = acceptable  # shape [20, 20], bool

    def compute(self, y_true: ak.Array, y_pred: ak.Array):
        """
        y_true: integer residue tokens in PROTEIN_ALPHABET order.
        y_pred: logits or probabilities over the same alphabet.
        """
        # Predicted class (top-1 by probability/logit)
        y_pred_idx = ak.argmax(y_pred, axis=-1)  # same ragged structure as y_true

        # Flatten over sequence positions for vectorised table lookup
        true_flat = ak.to_numpy(ak.flatten(y_true, axis=-1))
        pred_flat = ak.to_numpy(ak.flatten(y_pred_idx, axis=-1))

        # Lookup correctness from precomputed BLOSUM-based accept table
        correct_flat = self.acceptable[true_flat, pred_flat]  # [N_flat] bool

        # Restore per-sequence structure
        lengths = ak.to_numpy(ak.num(y_true, axis=-1))  # [N_seq]
        correct = ak.unflatten(correct_flat, lengths)  # [N_seq, L]

        num_correct = ak.sum(correct, axis=-1)
        total = ak.num(y_true, axis=-1)
        return num_correct / total
