import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import PROTEIN_ALPHABET


def _mutation_table(scenes, field, row):
    table = getattr(scenes, field)[row]
    while table.ndim > 1 and len(table) == 1:
        table = table[0]
    return table


def _is_letter_list(x):
    return isinstance(x, list) and x and isinstance(x[0], str)


def _is_chain_list(x):
    return isinstance(x, list) and x and _is_letter_list(x[0])


def _as_list(x):
    return x if isinstance(x, list) else ak.to_list(x)


def _collect_one_molecule(x):
    """Parse one molecule into a list of chains (each a list of 1-letter codes)."""
    x = _as_list(x)
    if _is_letter_list(x):
        return [list(x)]
    if _is_chain_list(x):
        return [list(chain) for chain in x]
    if isinstance(x, list) and len(x) == 1:
        return _collect_one_molecule(x[0])
    raise ValueError(f"Cannot parse molecule residue labels of type {type(x)!r}")


def _collect_chain_groups(x, n_molecules):
    """Return one chain-group per selected molecule.

    A batch of single-chain proteins is ``[seq, seq, ...]``. A single
    multi-chain complex is ``[chain_a, chain_b]``. ``n_molecules`` (from the
    mutation index) distinguishes those two cases.
    """
    x = _as_list(x)
    while (
        isinstance(x, list)
        and len(x) == 1
        and n_molecules != 1
        and not _is_letter_list(x)
    ):
        x = x[0]
    if n_molecules == 1:
        return [_collect_one_molecule(x)]
    if not isinstance(x, list) or len(x) != n_molecules:
        raise ValueError(
            f"Expected {n_molecules} molecules in residue labels, "
            f"got {len(x) if isinstance(x, list) else type(x)!r}"
        )
    return [_collect_one_molecule(item) for item in x]


def _replace_one_molecule(x, chains):
    x = _as_list(x)
    if _is_letter_list(x):
        return chains[0] if len(chains) == 1 else chains
    if _is_chain_list(x):
        return chains
    if isinstance(x, list) and len(x) == 1:
        return [_replace_one_molecule(x[0], chains)]
    raise ValueError(f"Cannot write molecule residue labels of type {type(x)!r}")


def _replace_chain_groups(x, groups, n_molecules):
    x = _as_list(x)
    if (
        isinstance(x, list)
        and len(x) == 1
        and n_molecules != 1
        and not _is_letter_list(x)
    ):
        return [_replace_chain_groups(x[0], groups, n_molecules)]
    if n_molecules == 1:
        return _replace_one_molecule(x, groups[0])
    return [
        _replace_one_molecule(item, group) for item, group in zip(x, groups)
    ]


def _apply_residue_mutations(labels, idx, amino_acid, n_molecules=None):
    template = ak.to_list(labels)
    idx_np = np.asarray(ak.to_numpy(idx))
    if idx_np.ndim == 1:
        idx_np = idx_np.reshape(-1, 1)
    batch_i = (
        idx_np[:, 0].astype(int)
        if idx_np.shape[1] > 1
        else np.zeros(len(idx_np), dtype=int)
    )
    n_from_idx = int(np.max(batch_i)) + 1 if len(batch_i) else 1
    if n_molecules is None:
        n_molecules = n_from_idx
    groups = _collect_chain_groups(template, n_molecules)
    positions = idx_np[:, -1].astype(int)
    values = [str(v) for v in ak.to_list(ak.ravel(amino_acid))]
    for row, pos, aa in zip(batch_i, positions, values):
        concat = [res for chain in groups[int(row)] for res in chain]
        concat[int(pos)] = aa
        restored = []
        start = 0
        for chain in groups[int(row)]:
            restored.append(concat[start : start + len(chain)])
            start += len(chain)
        groups[int(row)] = restored
    return ak.Array(_replace_chain_groups(template, groups, n_molecules))


_AA_TOKEN = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}


def _tokenize_label_tree(x):
    x = _as_list(x)
    if _is_letter_list(x):
        return [_AA_TOKEN.get(str(aa), -1) for aa in x]
    if isinstance(x, list):
        return [_tokenize_label_tree(item) for item in x]
    return _AA_TOKEN.get(str(x), -1)


def _retokenize_residues(batch):
    batch.molecules.residue_token = _tokenize_label_tree(batch.molecules.residue_label)
    return batch


def _as_float(value):
    if isinstance(value, ak.Array):
        value = ak.to_list(ak.ravel(value))
    if isinstance(value, (list, tuple)):
        while isinstance(value, (list, tuple)):
            if not value:
                return 0.0
            value = value[0]
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    flat = np.ravel(arr)
    return float(flat[0]) if flat.size else 0.0


def _first_unique_ids(values):
    values = np.asarray(ak.to_numpy(ak.ravel(values)), dtype=np.int64)
    seen = {}
    uniq = []
    inverse = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values.tolist()):
        found = seen.get(value)
        if found is None:
            found = len(uniq)
            seen[value] = found
            uniq.append(i)
        inverse[i] = found
    return np.asarray(uniq, dtype=np.int64), inverse


def _mutation_spec(batch, index, rows=None):
    mutations = ak.to_numpy(index["mutation"])
    if rows is None:
        rows = np.arange(len(mutations), dtype=np.int64)
    else:
        rows = np.asarray(rows, dtype=np.int64)
    amino_acid = ak.Array(
        [
            _mutation_table(batch.scenes, "molecule_mutation_label", int(rows[i]))[int(m)]
            for i, m in enumerate(mutations)
        ]
    )
    pos = ak.Array(
        [
            _mutation_table(batch.scenes, "molecule_mutation_pos", int(rows[i]))[int(m)]
            for i, m in enumerate(mutations)
        ]
    )
    idx_rows, aa_rows = [], []
    for i, (sites, aas) in enumerate(zip(ak.to_list(pos), ak.to_list(amino_acid))):
        if not isinstance(sites, list):
            sites = [sites]
        if not isinstance(aas, list):
            aas = [aas]
        graph = int(rows[i])
        for site, aa in zip(sites, aas):
            idx_rows.append([graph, int(site)])
            aa_rows.append(aa)
    effects = ak.Array(
        [
            _as_float(
                _mutation_table(batch.scenes, "molecule_mutation_effect", int(rows[i]))[
                    int(m)
                ]
            )
            for i, m in enumerate(mutations)
        ]
    )
    return ak.Array(idx_rows), ak.Array(aa_rows), pos, effects


def _letters_at_sites(batch, idx):
    idx_np = np.asarray(ak.to_numpy(idx))
    if idx_np.ndim == 1:
        idx_np = idx_np.reshape(-1, 1)
    batch_i = (
        idx_np[:, 0].astype(int)
        if idx_np.shape[1] > 1
        else np.zeros(len(idx_np), dtype=int)
    )
    n_molecules = int(np.max(batch_i)) + 1 if len(batch_i) else 1
    groups = _collect_chain_groups(batch.molecules.residue_label, n_molecules)
    letters = []
    for row, pos in zip(batch_i, idx_np[:, -1].astype(int)):
        concat = [res for chain in groups[int(row)] for res in chain]
        letters.append(str(concat[int(pos)]))
    return letters


def _aa_tokens(letters):
    return [_AA_TOKEN.get(str(aa), 0) for aa in letters]


class MutationEffectPredictionTask(Task):

    """Predict the effect of amino-acid mutations.

    ``mutate_structure=True`` writes the mutant residue into ``X`` (ProteinGym
    fitness / inverse-folding). ``False`` leaves the wild-type graph and only
    records ``position`` / ``wt`` / ``mut`` on ``y`` so the model can splice
    one-hot features and re-encode, matching Huber eval.
    """

    def __init__(self, resolution="residue", mutate_structure=True) -> None:
        super().__init__()
        self.resolution = resolution
        self.mutate_structure = bool(mutate_structure)

    def __call__(self, vbatch, assets, index):
        graph_ids = None
        if not self.mutate_structure:
            uniq, graph_ids = _first_unique_ids(index["scene"])
            X = vbatch[index["scene"][uniq], index["frame"][uniq], index["molecule"][uniq]]
        else:
            X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        idx, amino_acid, pos, effects = _mutation_spec(X, index, rows=graph_ids)
        y_fields = {
            "target": effects,
            "position": pos,
            "wt": _aa_tokens(_letters_at_sites(X, idx)),
            "mut": _aa_tokens(ak.to_list(ak.ravel(amino_acid))),
        }
        if graph_ids is not None:
            y_fields["graph"] = graph_ids
        y = ak.Array(y_fields)
        if self.mutate_structure:
            idx_np = np.asarray(ak.to_numpy(idx))
            if idx_np.size:
                if idx_np.ndim == 1:
                    idx_np = idx_np.reshape(-1, 1)
                n_molecules = int(idx_np[:, 0].max()) + 1
                X.molecules.residue_label = _apply_residue_mutations(
                    X.molecules.residue_label, idx, amino_acid, n_molecules=n_molecules
                )
                X = _retokenize_residues(X)
        return X, y
