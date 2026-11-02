from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
import yaml

from bioverse.adapters.starr_rbd import (
    SITE_OFFSET,
    WILDTYPE_ACE2,
    WILDTYPE_RBD,
    _find_scores_csv,
    _sequence_record,
    mutation_tables,
)
from bioverse.factory import BenchmarkFactory
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
def starr_benchmark():
    config.workers = 1
    old_dataset = config.dataset_path
    test_root = Path(config.root) / "test_datasets"
    config.dataset_path = test_root
    dest = Path(config.raw_path) / "SARS_CoV_2_RBD_DMS" / "final_variant_scores.csv"
    cached = Path("/tmp/rbd_dms/final_variant_scores.csv")
    if not dest.exists() and cached.is_file():
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
            yield BenchmarkFactory("B_STRRBD")
    finally:
        config.dataset_path = old_dataset


def test_structure_prediction_transform_parses_coordinates(tmp_path):
    record = _sequence_record("NITNL", WILDTYPE_ACE2[:7])
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
    assert chains == ["NITNL", WILDTYPE_ACE2[:7]]
    assert len(_peel_to_chains(out.data["atom_pos"][0])) == 2


def test_dataset_is_single_test_complex(starr_benchmark):
    dataset = starr_benchmark.dataset
    assert len(dataset.toc) == 1
    assert int(dataset.toc["mutations"][0]) > 1000
    assert dataset.split.names["starr_mutation_split"] == ["test"]
    batch = next(iter(dataset.shards))
    chains = [
        "".join(chain)
        for chain in ak.to_list(_peel_to_chains(batch.data["residue_label"][0]))
    ]
    rbd = dataset.assets["rbd_sequence"]
    ace2 = dataset.assets["ace2_sequence"]
    assert chains == [WILDTYPE_RBD, WILDTYPE_ACE2]
    assert rbd == WILDTYPE_RBD
    assert ace2 == WILDTYPE_ACE2
    assert "HHHHHH" not in rbd
    assert "HHHHHH" not in ace2
    assert all(aa in PROTEIN_ALPHABET for aa in rbd)
    assert all(aa in PROTEIN_ALPHABET for aa in ace2)
    assert len(rbd) == 201
    assert rbd[0] == "N"
    assert rbd[484 - SITE_OFFSET] == "E"
    assert rbd[501 - SITE_OFFSET] == "N"


def test_loader_reuses_structure_and_mutates_rbd_only(starr_benchmark):
    dataset = starr_benchmark.dataset
    n_rbd = len(dataset.assets["rbd_sequence"])
    n_ace2 = len(dataset.assets["ace2_sequence"])
    _, _, expected = mutation_tables(
        _find_scores_csv(config.raw_path / "SARS_CoV_2_RBD_DMS")
    )
    loader = starr_benchmark.loader(
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
    np.testing.assert_allclose(targets, expected[:32], rtol=0, atol=1e-10)

    assert all(np.all((pos >= 0) & (pos < n_rbd)) for pos in positions)
    assert n_ace2 == len(WILDTYPE_ACE2)
