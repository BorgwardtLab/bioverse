import math
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest

from bioverse.adapters.biolip import (
    R_KCAL,
    T_REF,
    BioLipAdapter,
    _imap_threads,
    build_complex_record,
    is_polymer_ligand,
    load_structure_atoms,
    parse_affinity_fields,
    parse_annotations,
    parse_binding_site,
    parse_kd,
)
from bioverse.adapters import BioLipAdapter as ImportedBioLipAdapter
from bioverse.data import Batch
from bioverse.factory import BenchmarkFactory
from bioverse.transforms import ExplodeLigandAtoms, TokenizeByChainType
from bioverse.utilities import PROTEIN_ALPHABET, config

config.workers = 1


def _annotation_line(
    pdb_id="1bkj",
    receptor="A",
    ligand="FMN",
    ligand_chain="A",
    serial="1",
    site="H11 R12 S13",
    manual="",
    moad="Kd=0.17uM",
    pdbbind="",
    bindingdb="",
    lig_resi="241",
):
    cols = [
        pdb_id,
        receptor,
        "1.90",
        "BS01",
        ligand,
        ligand_chain,
        serial,
        site,
        "H1 R2 S3",
        "",
        "",
        "1.1.1.1",
        "0000001",
        manual,
        moad,
        pdbbind,
        bindingdb,
        "P00000",
        "12345678",
        lig_resi,
        "ARNDCEQ",
    ]
    return "\t".join(cols) + "\n"


def _atom_line(
    serial,
    name,
    resname,
    chain,
    resi,
    x,
    y,
    z,
    element,
    het=False,
    insertion=" ",
):
    kind = "HETATM" if het else "ATOM  "
    atom = f"{name:>3s}" if len(name) < 4 else name
    return (
        f"{kind}{serial:5d} {atom:<4s} {resname:>3s} {chain}"
        f"{resi:4d}{insertion}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00           {element:>2s}\n"
    )


def _write_complex_pdb(
    path: Path,
    protein_chain="A",
    ligand_chain="A",
    ligand="FMN",
    ligand_resi=241,
    contacts=(11, 13),
    insertion_resi=None,
):
    lines = ["HEADER    BIOLIP TEST\n"]
    serial = 1
    residues = [("ALA", 11), ("ARG", 12), ("SER", 13)]
    if insertion_resi is not None:
        residues.append(("PHE", insertion_resi))
    x0 = 0.0
    for resname, resi in residues:
        insertion = "A" if insertion_resi is not None and resi == insertion_resi else " "
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
                    insertion=insertion,
                )
            )
            serial += 1
        x0 += 3.8
    for name, dx, element in (("C1", 0.0, "C"), ("N1", 1.2, "N"), ("O1", 2.4, "O")):
        lines.append(
            _atom_line(
                serial,
                name,
                ligand,
                ligand_chain,
                ligand_resi,
                20.0 + dx,
                1.0,
                0.0,
                element,
                het=True,
            )
        )
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


def _write_biolip_raw(root: Path, n: int = 12) -> Path:
    raw = root / "BioLiP"
    structures = raw / "structures"
    structures.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        pdb_id = f"t{i:03d}"
        kd = 10 ** (-5 - 0.15 * i)
        site = "A11 S13" if i % 2 == 0 else "A11 R12"
        lines.append(
            _annotation_line(
                pdb_id=pdb_id,
                site=site,
                moad=f"Kd={kd}M",
                lig_resi="241",
            )
        )
        _write_complex_pdb(structures / f"{pdb_id}.pdb")
    lines.append(
        _annotation_line(
            pdb_id="skip",
            ligand="dna",
            moad="Kd=1nM",
        )
    )
    lines.append(
        _annotation_line(
            pdb_id="noki",
            moad="Ki=2nM",
        )
    )
    (raw / "BioLiP.txt").write_text("".join(lines))
    return raw


def test_parse_kd_units_and_bounds():
    assert math.isclose(parse_kd("Kd=0.17uM"), 1.7e-7)
    assert math.isclose(parse_kd("kd=4.6nM"), 4.6e-9)
    assert math.isclose(parse_kd("Kd=2mM"), 0.002)
    assert math.isclose(parse_kd("Kd=1.6pM"), 1.6e-12)
    assert math.isclose(parse_kd("Kd=0.000000064M"), 6.4e-8)
    assert parse_kd("Kd<10uM") is None
    assert parse_kd("Kd>1nM") is None
    assert parse_kd("Ki=23nM") is None
    assert parse_kd("") is None
    assert parse_kd("remove_due_to_license_issue") is None


def test_parse_affinity_prefers_manual_over_moad():
    aff = parse_affinity_fields(["Kd=2nM", "Kd=0.17uM", "", ""])
    assert math.isclose(aff["kd"], 2e-9)
    assert aff["affinity_source"] == "manual"
    assert math.isclose(aff["pkd"], 8.69897, rel_tol=1e-5)
    assert math.isclose(aff["dg"], R_KCAL * T_REF * math.log(2e-9), rel_tol=1e-10)

    aff = parse_affinity_fields(["", "Kd=0.17uM", "", "Ki=4.6nM"])
    assert aff["affinity_source"] == "moad"
    assert math.isclose(aff["kd"], 1.7e-7)


def test_parse_binding_site_insertions():
    assert parse_binding_site("H11 R12 S13") == [
        ("H", 11, ""),
        ("R", 12, ""),
        ("S", 13, ""),
    ]
    assert parse_binding_site("Y33 F100A M100B") == [
        ("Y", 33, ""),
        ("F", 100, "A"),
        ("M", 100, "B"),
    ]
    assert parse_binding_site("") == []


def test_polymer_ligands_are_excluded():
    assert is_polymer_ligand("dna")
    assert is_polymer_ligand("RNA")
    assert is_polymer_ligand("peptide")
    assert not is_polymer_ligand("FMN")
    assert not is_polymer_ligand("ATP")


def test_parse_annotations_filters_kd_and_polymers(tmp_path):
    path = tmp_path / "BioLiP_nr.txt"
    path.write_text(
        "".join(
            [
                _annotation_line(pdb_id="1bkj", moad="Kd=0.17uM"),
                _annotation_line(pdb_id="1dna", ligand="dna", moad="Kd=1nM"),
                _annotation_line(pdb_id="1kii", moad="Ki=2nM"),
                _annotation_line(pdb_id="1c9h", moad="", bindingdb="Kd=0.200000nM"),
            ]
        )
    )
    entries = parse_annotations(path)
    assert [entry["pdb_id"] for entry in entries] == ["1bkj", "1c9h"]
    assert entries[0]["molecule_id"] == "1bkj_A_FMN_A_1"
    assert math.isclose(entries[1]["kd"], 2e-10)


def test_build_complex_record_two_chains_and_contacts(tmp_path):
    pdb_path = _write_complex_pdb(tmp_path / "1bkj.pdb")
    (tmp_path / "ann.txt").write_text(_annotation_line(site="A11 S13", moad="Kd=0.17uM"))
    entry = parse_annotations(tmp_path / "ann.txt")[0]
    record = build_complex_record(load_structure_atoms(pdb_path), entry)
    assert record is not None
    chains = ak.to_list(record.chain_label[0][0])
    types = ak.to_list(record.chain_type[0][0])
    labels = ak.to_list(record.residue_label[0][0])
    contacts = ak.to_list(record.residue_contact[0][0])
    assert chains[0] == "A"
    assert types == ["protein", "ligand"]
    assert labels[0] == ["A", "R", "S"]
    assert labels[1] == ["FMN"]
    assert contacts[0] == [1, 0, 1]
    assert contacts[1] == [0]
    protein_atoms = ak.to_list(record.atom_label[0][0][0])
    ligand_atoms = ak.to_list(record.atom_label[0][0][1])
    assert protein_atoms[0] == ["N", "CA"]
    assert ligand_atoms == [["C", "N", "O"]]
    assert math.isclose(float(ak.to_numpy(ak.ravel(record.molecule_pkd))[0]), -math.log10(1.7e-7))


def test_explode_and_chain_type_tokenizer(tmp_path):
    pdb_path = _write_complex_pdb(tmp_path / "1bkj.pdb")
    (tmp_path / "ann.txt").write_text(_annotation_line(site="A11 S13", moad="Kd=0.17uM"))
    entry = parse_annotations(tmp_path / "ann.txt")[0]
    record = build_complex_record(load_structure_atoms(pdb_path), entry)
    batch = ExplodeLigandAtoms().transform_batch(Batch(ak.Array([record])))
    labels = ak.to_list(batch.data["residue_label"])
    while isinstance(labels[0][0], list):
        labels = labels[0]
    assert labels[0] == ["A", "R", "S"]
    assert labels[1] == ["C", "N", "O"]
    tokenizer = TokenizeByChainType()
    batch = tokenizer.transform_batch(batch)
    tokens = np.asarray(ak.to_list(ak.ravel(batch.residue_token)))
    assert tokens[0] == PROTEIN_ALPHABET.index("A")
    assert tokens[3] == tokenizer.ligand_tokens["C"]
    assert tokens[3] != PROTEIN_ALPHABET.index("C")


def test_shared_chain_id_renames_ligand(tmp_path):
    pdb_path = _write_complex_pdb(tmp_path / "1bkj.pdb", ligand_chain="A")
    (tmp_path / "ann.txt").write_text(_annotation_line(ligand_chain="A"))
    entry = parse_annotations(tmp_path / "ann.txt")[0]
    record = build_complex_record(load_structure_atoms(pdb_path), entry)
    chains = ak.to_list(record.chain_label[0][0])
    assert chains[0] == "A"
    assert chains[1] != "A"
    assert ak.to_list(record.chain_type[0][0]) == ["protein", "ligand"]


def test_insertion_code_contact(tmp_path):
    pdb_path = _write_complex_pdb(
        tmp_path / "1ct8.pdb",
        insertion_resi=100,
        ligand="TAA",
        ligand_resi=550,
    )
    (tmp_path / "ann.txt").write_text(
        _annotation_line(
            pdb_id="1ct8",
            ligand="TAA",
            ligand_chain="A",
            site="A11 F100A",
            lig_resi="550",
        )
    )
    entry = parse_annotations(tmp_path / "ann.txt")[0]
    record = build_complex_record(load_structure_atoms(pdb_path), entry)
    labels = ak.to_list(record.residue_label[0][0][0])
    insertions = ak.to_list(record.residue_insertion[0][0][0])
    contacts = ak.to_list(record.residue_contact[0][0][0])
    assert labels[-1] == "F"
    assert insertions[-1] == "A"
    assert contacts[0] == 1
    assert contacts[-1] == 1


def test_adapter_download_uses_local_raw(tmp_path):
    _write_biolip_raw(tmp_path, n=4)
    with patch.object(config, "raw_path", tmp_path):
        batches, split, assets = BioLipAdapter.download()
        batch = next(iter(batches))
    assert assets["n_complexes"] == 4
    assert assets["n_annotations"] == 4
    assert len(batch) == 4
    types = ak.to_list(batch.data["chain_type"][0][0][0])
    assert types == ["protein", "ligand"]
    contacts = np.asarray(ak.to_list(batch.data["residue_contact"][0][0][0][0]))
    assert set(contacts.tolist()) <= {0, 1}
    assert int(contacts.sum()) >= 1
    pkd = np.asarray(ak.to_numpy(ak.ravel(batch.data["molecule_pkd"])))
    assert pkd.shape == (4,)
    assert np.all(np.isfinite(pkd))


def test_imap_threads_preserves_order():
    assert _imap_threads(lambda x: x * 10, [1, 2, 3, 4], "scale", workers=2) == [
        10,
        20,
        30,
        40,
    ]


def test_adapter_download_parallel(tmp_path):
    _write_biolip_raw(tmp_path, n=4)
    with patch.object(config, "raw_path", tmp_path), patch.object(config, "workers", 2):
        batches, _, assets = BioLipAdapter.download()
        batch = next(iter(batches))
    assert assets["n_complexes"] == 4
    assert len(batch) == 4


@pytest.fixture(scope="module")
def biolip_benchmark(tmp_path_factory):
    raw_root = tmp_path_factory.mktemp("biolip_raw")
    data_root = tmp_path_factory.mktemp("biolip_data")
    _write_biolip_raw(raw_root, n=20)
    old_raw = config.raw_path
    old_data = config.dataset_path
    old_workers = config.workers
    config.raw_path = raw_root
    config.dataset_path = data_root
    config.workers = 1
    try:
        yield BenchmarkFactory("B_BIOLIP")
    finally:
        config.raw_path = old_raw
        config.dataset_path = old_data
        config.workers = old_workers


def test_dataset_factory_heterogeneous_molecules(biolip_benchmark):
    dataset = biolip_benchmark.dataset
    assert dataset.split.names["random_scene_split"]
    for part in ("train", "val", "test"):
        assert int(dataset.split[part].sum()) > 0
    batch = next(iter(dataset.shards))
    assert "molecule_pkd" in batch
    assert "residue_contact" in batch
    assert "chain_type" in batch
    types = ak.to_list(batch.data["chain_type"][0])
    while isinstance(types[0], list):
        types = types[0]
    assert types == ["protein", "ligand"]
    labels = ak.to_list(batch.data["residue_label"][0])
    while isinstance(labels[0][0], list):
        labels = labels[0]
    assert labels[0] == ["A", "R", "S"]
    assert labels[1] == ["FMN"]


def test_benchmark_loader_predicts_pkd(biolip_benchmark):
    loader = biolip_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))
    target = float(ak.to_numpy(ak.ravel(y["target"]))[0])
    assert np.isfinite(target)
    assert 4.0 < target < 8.0
    chains = ak.to_list(X.data["chain_type"])
    while isinstance(chains[0], list):
        chains = chains[0]
    assert set(chains) == {"protein", "ligand"}
    assert "atom_pos" in X


def test_factory_exports_adapter():
    assert ImportedBioLipAdapter is BioLipAdapter


def test_benchmark_metrics_update(biolip_benchmark):
    loader = biolip_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    for (X, y), _ in loader:
        pred = ak.Array({"target": ak.to_numpy(ak.ravel(y["target"]))})
        biolip_benchmark.update(y, pred)
    rows = {row["Metric"]: row["Value"] for row in biolip_benchmark.metric.result().data}
    assert any("MAE" in name for name in rows)
    assert any("Spearman" in name for name in rows)
    for name, value in rows.items():
        if "MAE" in name:
            assert np.isclose(value, 0.0, atol=1e-8)
        if "Spearman" in name:
            assert np.isclose(value, 1.0, atol=1e-8)
