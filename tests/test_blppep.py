from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest

from bioverse.adapters.biolip import (
    BioLipAdapter,
    build_peptide_complex_record,
    is_peptide_ligand,
    load_structure_atoms,
    parse_annotations,
)
from bioverse.factory import BenchmarkFactory
from bioverse.utilities import PROTEIN_ALPHABET, config
from tests.test_biolip import _annotation_line, _atom_line

config.workers = 1


def _write_peptide_complex_pdb(
    path: Path,
    protein_chain="A",
    peptide_chain="B",
):
    lines = ["HEADER    BLPPEP TEST\n"]
    serial = 1
    x0 = 0.0
    for resname, resi in (("ALA", 11), ("ARG", 12), ("SER", 13)):
        for name, dx, element in (("N", 0.0, "N"), ("CA", 1.5, "C")):
            lines.append(
                _atom_line(
                    serial,
                    name,
                    resname,
                    protein_chain,
                    resi,
                    x0 + dx,
                    0.0,
                    0.0,
                    element,
                )
            )
            serial += 1
        x0 += 3.8
    x0 = 3.0
    for resname, resi in (("GLY", 1), ("LYS", 2), ("PHE", 3)):
        for name, dx, element in (("N", 0.0, "N"), ("CA", 1.5, "C")):
            lines.append(
                _atom_line(
                    serial,
                    name,
                    resname,
                    peptide_chain,
                    resi,
                    x0 + dx,
                    4.0,
                    0.0,
                    element,
                )
            )
            serial += 1
        x0 += 3.8
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


def _write_blppep_raw(root: Path, n: int = 12) -> Path:
    raw = root / "BioLiP"
    structures = raw / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        pdb_id = f"p{i:03d}"
        lines.append(
            _annotation_line(
                pdb_id=pdb_id,
                ligand="peptide",
                ligand_chain="B",
                site="A11 S13",
                moad="",
                lig_resi="1",
            )
        )
        _write_peptide_complex_pdb(structures / f"{pdb_id}.pdb")
    lines.append(
        _annotation_line(
            pdb_id="sml0",
            ligand="FMN",
            moad="Kd=1nM",
        )
    )
    lines.append(
        _annotation_line(
            pdb_id="dna0",
            ligand="dna",
            ligand_chain="B",
            moad="",
        )
    )
    (raw / "BioLiP.txt").write_text("".join(lines))
    return raw


def test_peptide_ligand_helper():
    assert is_peptide_ligand("peptide")
    assert is_peptide_ligand("PEPTIDE")
    assert not is_peptide_ligand("dna")
    assert not is_peptide_ligand("FMN")


def test_parse_annotations_keeps_peptides_without_affinity(tmp_path):
    path = tmp_path / "BioLiP.txt"
    path.write_text(
        "".join(
            [
                _annotation_line(
                    pdb_id="1pep",
                    ligand="peptide",
                    ligand_chain="B",
                    moad="",
                ),
                _annotation_line(pdb_id="1bkj", moad="Kd=0.17uM"),
                _annotation_line(
                    pdb_id="1dna",
                    ligand="dna",
                    ligand_chain="B",
                    moad="Kd=1nM",
                ),
            ]
        )
    )
    entries = parse_annotations(path, ligand="peptide")
    assert [entry["pdb_id"] for entry in entries] == ["1pep"]
    assert entries[0]["molecule_id"] == "1pep_A_peptide_B_1"
    assert "kd" not in entries[0]
    assert "pkd" not in entries[0]


def test_build_peptide_complex_record_two_chains_no_labels(tmp_path):
    pdb_path = _write_peptide_complex_pdb(tmp_path / "1pep.pdb")
    (tmp_path / "ann.txt").write_text(
        _annotation_line(pdb_id="1pep", ligand="peptide", ligand_chain="B", moad="")
    )
    entry = parse_annotations(tmp_path / "ann.txt", ligand="peptide")[0]
    record = build_peptide_complex_record(load_structure_atoms(pdb_path), entry)
    assert record is not None
    types = ak.to_list(record.chain_type[0][0])
    labels = ak.to_list(record.residue_label[0][0])
    assert types == ["protein", "peptide"]
    assert labels[0] == ["A", "R", "S"]
    assert labels[1] == ["G", "K", "F"]
    assert "residue_contact" not in record.fields
    assert "molecule_pkd" not in record.fields
    assert (
        build_peptide_complex_record(load_structure_atoms(pdb_path), entry, max_length=3)
        is None
    )
    assert "molecule_kd" not in record.fields
    assert ak.to_list(record.molecule_pdb_id) == [["1pep"]]


def test_adapter_download_peptide_uses_local_raw(tmp_path):
    _write_blppep_raw(tmp_path, n=4)
    with patch.object(config, "raw_path", tmp_path):
        batches, split, assets = BioLipAdapter.download(ligand="peptide")
        batch = next(iter(batches))
    assert assets["n_complexes"] == 4
    assert assets["n_annotations"] == 4
    assert assets["ligand"] == "peptide"
    assert len(batch) == 4
    types = ak.to_list(batch.data["chain_type"][0][0][0])
    assert types == ["protein", "peptide"]
    assert "molecule_pkd" not in batch.data
    assert "residue_contact" not in batch.data


def test_adapter_download_peptide_parallel(tmp_path):
    _write_blppep_raw(tmp_path, n=4)
    with patch.object(config, "raw_path", tmp_path), patch.object(config, "workers", 2):
        batches, _, assets = BioLipAdapter.download(ligand="peptide")
        batch = next(iter(batches))
    assert assets["n_complexes"] == 4
    assert len(batch) == 4


@pytest.fixture(scope="module")
def blppep_benchmark(tmp_path_factory):
    raw_root = tmp_path_factory.mktemp("blppep_raw")
    data_root = tmp_path_factory.mktemp("blppep_data")
    _write_blppep_raw(raw_root, n=20)
    old_raw = config.raw_path
    old_data = config.dataset_path
    old_workers = config.workers
    config.raw_path = raw_root
    config.dataset_path = data_root
    config.workers = 1
    try:
        yield BenchmarkFactory("B_BLPPEP")
    finally:
        config.raw_path = old_raw
        config.dataset_path = old_data
        config.workers = old_workers


def test_dataset_random_split_80_10_10(blppep_benchmark):
    dataset = blppep_benchmark.dataset
    assert dataset.split.names["random_scene_split"]
    counts = {part: int(dataset.split[part].sum()) for part in ("train", "val", "test")}
    n = sum(counts.values())
    assert n == 20
    assert counts["test"] == int(n * 0.1)
    assert counts["val"] == int(n * 0.1)
    assert counts["train"] == n - counts["test"] - counts["val"]
    batch = next(iter(dataset.shards))
    assert "molecule_pkd" not in batch
    assert "residue_contact" not in batch
    types = ak.to_list(batch.data["chain_type"][0])
    while isinstance(types[0], list):
        types = types[0]
    assert types == ["protein", "peptide"]


def test_benchmark_loader_masked_residues(blppep_benchmark):
    loader = blppep_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    found = None
    for (X, y), _ in loader:
        targets = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
        if targets.size:
            found = (X, y, targets)
            break
    assert found is not None, "benchmark loader yielded no masked residues"
    X, y, targets = found
    protein_idx = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}
    assert set(targets.tolist()) <= {protein_idx["A"], protein_idx["R"], protein_idx["S"]}
    assert protein_idx["G"] not in set(targets.tolist())
    chains = ak.to_list(X.data["chain_type"])
    while isinstance(chains[0], list):
        chains = chains[0]
    assert set(chains) == {"protein", "peptide"}
    assert "atom_pos" in X
    mask = ak.to_numpy(ak.ravel(X.molecules.residue_mask)).astype(bool)
    assert int(mask.sum()) == targets.size
    assert int(mask.sum()) == 3
    assert mask.tolist() == [True, True, True, False, False, False]
