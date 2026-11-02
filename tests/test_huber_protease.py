from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
import yaml

from bioverse.adapters.huber_protease import (
    WILDTYPE_PROTEASE,
    WILDTYPE_SUBSTRATE,
    _find_single_tsv,
    _sequence_record,
    mutation_tables,
)
from bioverse.factory import BenchmarkFactory
from bioverse.tasks.mutation_effect_prediction import _apply_residue_mutations
from bioverse.transforms.structure_prediction import (
    StructurePredictionTransform,
    _peel_to_chains,
)
from bioverse.utilities import ONE_TO_THREE, IteratorWithLength, batched, config
from bioverse.utilities.constants import PROTEIN_ALPHABET


def fake_boltz_predictor(self, input_dir, out_dir):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    for yaml_path in sorted(input_dir.glob("*.yaml")):
        spec = yaml.safe_load(yaml_path.read_text())
        stem = yaml_path.stem
        dest = out_dir / f"boltz_results_{input_dir.name}" / "predictions" / stem
        dest.mkdir(parents=True, exist_ok=True)
        lines = [
            "HEADER    FAKE BOLTZ PREDICTION\n",
            "REMARK   1 ALPHAFOLD\n",
        ]
        atom_i = 1
        chain_offset = 0.0
        for item in spec["sequences"]:
            protein = item["protein"]
            chain_id = protein["id"]
            if isinstance(chain_id, list):
                chain_id = chain_id[0]
            sequence = protein["sequence"]
            for res_i, aa in enumerate(sequence, start=1):
                res = ONE_TO_THREE[aa]
                x0 = chain_offset + res_i * 3.8
                for name, dx, element in (
                    ("N", 0.0, "N"),
                    ("CA", 1.5, "C"),
                    ("C", 2.5, "C"),
                    ("O", 3.2, "O"),
                ):
                    atom = f" {name:<3s}"
                    lines.append(
                        f"ATOM  {atom_i:5d} {atom} {res} {chain_id[-1]}"
                        f"{res_i:4d}    {x0 + dx:8.3f}{0.0:8.3f}{0.0:8.3f}"
                        f"  1.00 90.00           {element:>2s}\n"
                    )
                    atom_i += 1
            chain_offset += 400.0
        (dest / f"{stem}_model_0.pdb").write_text("".join(lines) + "END\n")


@pytest.fixture(scope="module")
def huber_benchmark():
    config.workers = 1
    old_dataset = config.dataset_path
    test_root = Path(config.root) / "test_datasets"
    config.dataset_path = test_root
    dest = Path(config.raw_path) / "MLDEEP"
    cached = Path("/tmp/mldeep_raw")
    if not dest.exists() and cached.is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(cached)

    def _cache_dir(self):
        return test_root / "boltz2"

    try:
        with patch.object(
            StructurePredictionTransform, "run_predictor", fake_boltz_predictor
        ), patch.object(
            StructurePredictionTransform, "_resolved_cache_dir", _cache_dir
        ):
            yield BenchmarkFactory("B_HUBPRT")
    finally:
        config.dataset_path = old_dataset


def _peel_letters(out):
    peeled = ak.to_list(out)
    while isinstance(peeled, list) and peeled and isinstance(peeled[0], list):
        peeled = peeled[0]
    return "".join(peeled)


def test_apply_single_chain_mutation():
    labels = ak.Array([[list("ARND")]])
    idx = ak.Array([[0, 1]])
    amino_acid = ak.Array([["G"]])
    out = _apply_residue_mutations(labels, idx, amino_acid)
    assert _peel_letters(out) == "AGND"


def test_apply_two_chain_molecule_concatenated_positions():
    labels = ak.Array([[list("ARND"), list("EFGH")]])
    idx = ak.Array([[0, 1], [0, 5]])
    amino_acid = ak.Array([["X"], ["Y"]])
    out = _apply_residue_mutations(labels, idx, amino_acid)
    chains = ak.to_list(out)
    while isinstance(chains, list) and chains and isinstance(chains[0][0], list):
        chains = chains[0]
    assert ["".join(chain) for chain in chains] == ["AXND", "EYGH"]


def test_apply_batch_of_single_chain_molecules():
    labels = ak.Array([list("ARND"), list("ARND"), list("ARND")])
    idx = ak.Array([[0, 1], [1, 2], [2, 0]])
    amino_acid = ak.Array([["G"], ["C"], ["V"]])
    out = _apply_residue_mutations(labels, idx, amino_acid)
    assert ["".join(seq) for seq in ak.to_list(out)] == ["AGND", "ARCD", "VRND"]


def test_apply_batch_mutations_when_trailing_rows_are_unmutated():
    labels = ak.Array([list("ARND")] * 16)
    idx = ak.Array([[0, 1], [14, 2]])
    amino_acid = ak.Array([["G"], ["C"]])
    out = _apply_residue_mutations(labels, idx, amino_acid, n_molecules=16)
    seqs = ["".join(seq) for seq in ak.to_list(out)]
    assert len(seqs) == 16
    assert seqs[0] == "AGND"
    assert seqs[14] == "ARCD"
    assert seqs[15] == "ARND"


def test_structure_prediction_transform_parses_coordinates(tmp_path):
    record = _sequence_record("ARND", WILDTYPE_SUBSTRATE)
    batches = batched(IteratorWithLength(iter([record]), 1))
    batch = next(iter(batches))
    transform = StructurePredictionTransform(
        model="boltz2",
        use_msa_server=False,
        cache_dir=tmp_path / "boltz",
    )
    with patch.object(
        StructurePredictionTransform, "run_predictor", fake_boltz_predictor
    ):
        out = transform.transform_batch(batch)
    assert "atom_pos" in out
    chains = [
        "".join(chain)
        for chain in ak.to_list(_peel_to_chains(out.data["residue_label"][0]))
    ]
    assert chains == ["ARND", WILDTYPE_SUBSTRATE]
    assert len(_peel_to_chains(out.data["atom_pos"][0])) == 2


def test_dataset_is_single_test_complex(huber_benchmark):
    dataset = huber_benchmark.dataset
    assert len(dataset.toc) == 1
    assert int(dataset.toc["mutations"][0]) > 1000
    assert dataset.split.names["huber_mutation_split"] == ["test"]
    batch = next(iter(dataset.shards))
    chains = [
        "".join(chain)
        for chain in ak.to_list(_peel_to_chains(batch.data["residue_label"][0]))
    ]
    protease = dataset.assets["protease_sequence"]
    substrate = dataset.assets["substrate_sequence"]
    assert chains == [WILDTYPE_PROTEASE, WILDTYPE_SUBSTRATE]
    assert protease == WILDTYPE_PROTEASE
    assert substrate == WILDTYPE_SUBSTRATE
    assert "HHHHHH" not in protease
    assert all(aa in PROTEIN_ALPHABET for aa in protease)


def test_loader_reuses_structure_and_varies_both_chains(huber_benchmark):
    dataset = huber_benchmark.dataset
    n_protease = len(dataset.assets["protease_sequence"])
    n_substrate = len(dataset.assets["substrate_sequence"])
    _, _, expected_auc = mutation_tables(_find_single_tsv(config.raw_path / "MLDEEP"))
    loader = huber_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="mutations",
        shuffle=False,
        progress=False,
    )
    coords = []
    positions = []
    targets = []
    for i, ((X, y), _) in enumerate(loader):
        pos = np.array(ak.to_list(y["position"][0]), dtype=int)
        positions.append(pos)
        targets.append(float(ak.to_numpy(ak.ravel(y["target"]))[0]))
        xyz = np.asarray(ak.to_numpy(ak.ravel(X.atom_pos)[:12]))
        coords.append(xyz)
        if i >= 31:
            break

    assert len(coords) == 32
    for later in coords[1:]:
        np.testing.assert_allclose(later, coords[0])
    np.testing.assert_allclose(targets, expected_auc[:32], rtol=0, atol=1e-10)

    assert any(np.any(pos < n_protease) for pos in positions)
    assert any(np.any(pos >= n_protease) for pos in positions)
    assert any(len(pos) == 2 for pos in positions)
    both = next(pos for pos in positions if len(pos) == 2)
    assert both[0] < n_protease
    assert n_protease <= both[1] < n_protease + n_substrate
