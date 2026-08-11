import shutil
from pathlib import Path
import json

import awkward as ak
import numpy as np
import pytest
from fastavro import reader, writer

from bioverse.adapters.proteinshake import convert_protein
from bioverse.factory import BenchmarkFactory
from bioverse.utilities import config

DATASET = "ProteinLigandInterfaceDataset"


def _truncate_protein(protein: dict, max_residues: int = 12) -> dict:
    residue_numbers = protein["atom"]["residue_number"]
    keep_residues = sorted(set(residue_numbers))[:max_residues]
    keep = [i for i, r in enumerate(residue_numbers) if r in keep_residues]
    protein = {
        "protein": dict(protein["protein"]),
        "atom": {key: [values[i] for i in keep] for key, values in protein["atom"].items()},
    }
    protein["protein"]["ID"] = "1ABC"
    protein["protein"]["ligand_smiles"] = "CCO"
    protein["protein"]["neglog_aff"] = 5.0
    protein["protein"]["random_split"] = "train"
    binding_residues = set(keep_residues[:2])
    res_nums = protein["atom"]["residue_number"]
    protein["atom"]["binding_site"] = [
        int(res_nums[i] in binding_residues) for i in range(len(res_nums))
    ]
    return protein


def _extend_atom_schema(schema: dict) -> dict:
    schema = json.loads(json.dumps(schema))
    for field in schema["fields"]:
        if field["name"] != "atom":
            continue
        atom_type = field["type"]
        if isinstance(atom_type, list):
            record = atom_type[1]
        else:
            record = atom_type
        names = {item["name"] for item in record["fields"]}
        if "binding_site" not in names:
            record["fields"].append(
                {"name": "binding_site", "type": {"type": "array", "items": "int"}}
            )
    return schema


@pytest.fixture(scope="module")
def binding_site_avro(tmp_path_factory):
    source = (
        config.raw_path
        / "ProteinShake"
        / "ProteinProteinInterfaceDataset"
        / "ProteinProteinInterfaceDataset.atom.avro"
    )
    if not source.exists():
        pytest.skip("ProteinProteinInterfaceDataset required to build test fixture")

    with open(source, "rb") as file:
        avro_reader = reader(file)
        schema = _extend_atom_schema(avro_reader.writer_schema)
        protein = _truncate_protein(next(avro_reader))

    raw_root = tmp_path_factory.mktemp("proteinshake")
    raw_dir = raw_root / "ProteinShake" / DATASET
    raw_dir.mkdir(parents=True)
    avro_path = raw_dir / f"{DATASET}.atom.avro"
    with open(avro_path, "wb") as file:
        writer(file, schema, [protein])

    original_raw_path = config.raw_path
    config.raw_path = raw_root
    dataset_root = config.dataset_path / "D_PSLINT"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    yield avro_path, protein
    config.raw_path = original_raw_path
    dataset_root = config.dataset_path / "D_PSLINT"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)


@pytest.fixture(scope="module")
def benchmark(binding_site_avro):
    config.workers = 1
    return BenchmarkFactory("B_PSLINT")


def test_convert_protein_exposes_binding_site(binding_site_avro):
    _, protein = binding_site_avro
    record = convert_protein(protein)
    labels = ak.ravel(record.residue_binding_site)
    assert len(labels) > 0
    assert set(ak.to_numpy(labels)) <= {0, 1}
    assert ak.sum(labels) > 0


def test_task_returns_binding_site_targets(benchmark, binding_site_avro):
    _, protein = binding_site_avro
    expected = ak.to_numpy(
        ak.ravel(convert_protein(protein).residue_binding_site)
    ).astype(float)

    loader = benchmark.loader(
        partition="train",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))

    targets = ak.to_numpy(ak.ravel(y["target"])).astype(float)
    assert len(targets) == len(expected)
    np.testing.assert_array_equal(targets, expected)
    assert X.molecules.residue_label is not None
