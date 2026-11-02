"""WT protease graph + mutation operator + substrate window → Δactivity."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..task import Task
from ..tasks.mutation_effect_prediction import (
    _aa_tokens,
    _first_unique_ids,
    _letters_at_sites,
    _mutation_spec,
)
from ..tasks.protease_substrate_compatibility import _leaf
from ..utilities.cleavage import WINDOW_LEN, tokenize_window


def _mutation_table(scenes, field, row):
    table = getattr(scenes, field)[row]
    while table.ndim > 1 and len(table) == 1:
        table = table[0]
    return table


class ProteaseMutationDeltaTask(Task):
    """Keep the wild-type fold and emit mutation + P6–P4′ fields.

    The geometric model scores ``f(P_mut, S) - f(P_WT, S)`` against the
    synthetic (or measured) ``Δactivity`` stored in ``molecule_mutation_effect``.
    Coordinates stay wild-type; mutant identity is spliced at train time.
    """

    def __init__(self, resolution="residue", window_len: int = WINDOW_LEN) -> None:
        super().__init__()
        self.resolution = resolution
        self.window_len = int(window_len)

    def __call__(self, vbatch, assets, index):
        uniq, graph_ids = _first_unique_ids(index["scene"])
        X = vbatch[index["scene"][uniq], index["frame"][uniq], index["molecule"][uniq]]
        X.resolution = self.resolution
        idx, amino_acid, pos, effects = _mutation_spec(X, index, rows=graph_ids)
        mutations = np.asarray(ak.to_numpy(ak.ravel(index["mutation"])), dtype=int)
        gap = "-" * self.window_len
        windows, kinds = [], []
        for i, mut in enumerate(mutations):
            row = int(graph_ids[i])
            window, kind = gap, "synthetic"
            try:
                window = _leaf(
                    _mutation_table(X.scenes, "molecule_mutation_substrate", row)[
                        int(mut)
                    ],
                    gap,
                )
            except Exception:
                window = gap
            try:
                kind = _leaf(
                    _mutation_table(X.scenes, "molecule_mutation_kind", row)[int(mut)],
                    "synthetic",
                )
            except Exception:
                kind = "synthetic"
            windows.append(str(window) if window is not None else gap)
            kinds.append(str(kind) if kind is not None else "synthetic")
        tokens = [tokenize_window(window, self.window_len) for window in windows]
        y = ak.Array(
            {
                "target": effects,
                "position": pos,
                "wt": _aa_tokens(_letters_at_sites(X, idx)),
                "mut": _aa_tokens(ak.to_list(ak.ravel(amino_acid))),
                "graph": graph_ids,
                "substrate": tokens,
                "window": windows,
                "kind": kinds,
            }
        )
        return X, y
