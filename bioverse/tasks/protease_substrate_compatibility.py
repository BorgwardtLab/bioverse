"""Score protease structure × substrate-window compatibility."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..task import Task
from ..tasks.mutation_effect_prediction import (
    _apply_residue_mutations,
    _retokenize_residues,
)
from ..utilities.cleavage import WINDOW_LEN, tokenize_window


def _peel_singleton_axes(table):
    """Drop leading length-1 wrappers without eating the mutation axis."""
    while getattr(table, "ndim", 0) > 1 and len(table) == 1:
        inner = table[0]
        try:
            inner_len = len(inner)
        except TypeError:
            return inner
        if inner_len == 0:
            break
        table = inner
    return table


def _leaf(value, default=None):
    if isinstance(value, ak.Array) and value.ndim == 0:
        try:
            return value.item()
        except Exception:
            return ak.to_list(value)
    if isinstance(value, ak.Array):
        value = ak.to_list(value)
    if value is None:
        return default
    return value


def _mutation_row(batch, field, row, mut, default=None):
    if field not in batch.data:
        return default
    table = _peel_singleton_axes(batch.data[field][row])
    mut = int(mut)
    try:
        n = len(table)
    except TypeError:
        return _leaf(table, default)
    if n == 0:
        return default
    if mut < 0 or mut >= n:
        return default
    return _leaf(table[mut], default)


def _mutation_sites(batch, row, mut):
    pos = _mutation_row(batch, "molecule_mutation_pos", row, mut, default=[])
    lab = _mutation_row(batch, "molecule_mutation_label", row, mut, default=[])
    if pos is None:
        pos = []
    if lab is None:
        lab = []
    if not isinstance(pos, list):
        pos = [pos]
    if not isinstance(lab, list):
        lab = [lab]
    sites, aas = [], []
    for site, aa in zip(pos, lab):
        try:
            idx = int(site)
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        token = str(aa)
        if token in {"", "-", "None", "nan"}:
            continue
        sites.append(idx)
        aas.append(token)
    return sites, aas


def _refeaturize(batch, assets):
    if "residue_token" not in batch:
        return batch
    _retokenize_residues(batch)
    if "residue_features" not in assets:
        return batch
    table = np.asarray(assets["residue_features"])
    tokens = np.asarray(ak.to_numpy(ak.ravel(batch.residue_token)), dtype=int)
    tokens = np.clip(tokens, 0, len(table) - 1)
    batch.residue_features = table[tokens]
    return batch


class ProteaseSubstrateCompatibilityTask(Task):
    """Return a protease fold and a tokenized P6–P4′ substrate window.

    Protease point mutations from ``molecule_mutation_*`` are applied to
    residue identity (coordinates stay wild-type). The paired substrate
    window is written to ``y["substrate"]`` as integer tokens so a two-tower
    model can score ``f(P, S)``.
    """

    def __init__(self, resolution="residue", window_len: int = WINDOW_LEN) -> None:
        super().__init__()
        self.resolution = resolution
        self.window_len = int(window_len)

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        mutations = np.asarray(ak.to_numpy(ak.ravel(index["mutation"])), dtype=int)
        idx_rows, aa_rows, effects, windows, kinds = [], [], [], [], []
        gap = "-" * self.window_len
        for i, mut in enumerate(mutations):
            effect = _mutation_row(X, "molecule_mutation_effect", i, mut, default=1.0)
            try:
                effects.append(float(effect))
            except (TypeError, ValueError):
                effects.append(1.0)
            window = _mutation_row(
                X, "molecule_mutation_substrate", i, mut, default=gap
            )
            windows.append(str(window) if window is not None else gap)
            kind = _mutation_row(X, "molecule_mutation_kind", i, mut, default="pair")
            kinds.append(str(kind) if kind is not None else "pair")
            sites, aas = _mutation_sites(X, i, mut)
            n_res = int(len(ak.ravel(X.data["residue_label"][i])))
            for site, aa in zip(sites, aas):
                if site >= n_res:
                    continue
                idx_rows.append([i, site])
                aa_rows.append(aa)
        if idx_rows:
            X.molecules.residue_label = _apply_residue_mutations(
                X.molecules.residue_label,
                ak.Array(idx_rows),
                ak.Array(aa_rows),
                n_molecules=len(mutations),
            )
            X = _refeaturize(X, assets)
        tokens = [tokenize_window(window, self.window_len) for window in windows]
        y = ak.Array(
            {
                "target": effects,
                "substrate": tokens,
                "window": windows,
                "kind": kinds,
            }
        )
        return X, y
