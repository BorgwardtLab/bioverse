import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from bioverse.adapters.alpha_fold_inv_bench import AlphaFoldInvBenchAdapter
from bioverse.utilities.id_mapping import (
    alphafold_candidates,
    build_alphafold_index,
    fetch_alphafold_structure,
    fetch_alphafold_structures,
    parse_invbench_name,
    resolve_alphafold_structure,
)


def test_parse_invbench_name():
    assert parse_invbench_name("12as.A") == ("12AS", "A")
    assert parse_invbench_name("2fyz.A") == ("2FYZ", "A")


def test_build_alphafold_index(tmp_path):
    pdb_path = tmp_path / "AF-P00963-F1-model_v4.pdb"
    pdb_path.write_text("END\n")
    index = build_alphafold_index(tmp_path, version="v4")
    assert index["P00963"] == pdb_path

    gz_path = tmp_path / "AF-P01000-F1-model_v4.pdb.gz"
    gz_path.write_text("END\n")
    index = build_alphafold_index(tmp_path, version="v4")
    assert index["P01000"] == gz_path


def test_alphafold_candidates():
    sifts = {("12AS", "A"): ["P00963", "P00000"]}
    index = {"P00963": Path("/tmp/AF-P00963-F1-model_v4.pdb")}
    candidates = alphafold_candidates("12as", "A", sifts, index)
    assert candidates == [("P00963", index["P00963"])]


def test_fetch_alphafold_structure_uses_cache(tmp_path):
    cached = tmp_path / "AF-Q8DIT2-F1-model_v6.pdb"
    shutil.copy(Path(__file__).parent / "dummy" / "dummy.pdb", cached)
    assert fetch_alphafold_structure("Q8DIT2", tmp_path) == cached


def test_fetch_alphafold_structure_downloads_direct_url(tmp_path):
    pdb_bytes = (Path(__file__).parent / "dummy" / "dummy.pdb").read_bytes()

    class FakeResponse:
        status_code = 200
        content = pdb_bytes

    with patch(
        "bioverse.utilities.id_mapping._get_session"
    ) as mock_session:
        mock_session.return_value.get.return_value = FakeResponse()
        path = fetch_alphafold_structure("Q8DIT2", tmp_path)

    assert path == tmp_path / "AF-Q8DIT2-F1-model_v6.pdb"
    assert path.exists()


def test_fetch_alphafold_structures_bulk(tmp_path):
    cached = tmp_path / "AF-P00963-F1-model_v6.pdb"
    shutil.copy(Path(__file__).parent / "dummy" / "dummy.pdb", cached)
    pdb_bytes = cached.read_bytes()

    class FakeResponse:
        status_code = 200
        content = pdb_bytes

    with patch(
        "bioverse.utilities.id_mapping._get_session"
    ) as mock_session:
        mock_session.return_value.get.return_value = FakeResponse()
        index = fetch_alphafold_structures(
            ["P00963", "Q8DIT2"], tmp_path, workers=2
        )

    assert index["P00963"] == cached
    assert index["Q8DIT2"] == tmp_path / "AF-Q8DIT2-F1-model_v6.pdb"


def test_resolve_alphafold_structure_prefers_local(tmp_path):
    local = tmp_path / "local" / "AF-P00963-F1-model_v4.pdb"
    local.parent.mkdir(parents=True)
    local.write_text("END\n")
    index = {"P00963": local}
    api_cache = tmp_path / "api"
    resolved = resolve_alphafold_structure("P00963", index, api_cache)
    assert resolved == (local, "local")


@pytest.fixture
def afcath_fixture(tmp_path, monkeypatch):
    from bioverse.utilities import config

    raw_path = tmp_path / "raw"
    invbench_path = raw_path / "ProteinInvBench" / "data" / "cath4.2"
    invbench_path.mkdir(parents=True)
    (invbench_path / "chain_set_splits.json").write_text(
        '{"train": ["12as.A"], "validation": [], "test": []}'
    )
    (invbench_path / "chain_set.jsonl").write_text(
        '{"name": "12as.A", "seq": "ACDEFGHIK", "CATH": ["3.30.930"], '
        '"coords": {"N": [[0,0,0]], "CA": [[0,0,0]], "C": [[0,0,0]], "O": [[0,0,0]]}}\n'
    )

    sifts_path = raw_path / "SIFTS"
    sifts_path.mkdir(parents=True)
    (sifts_path / "pdb_chain_uniprot.csv").write_text(
        "PDB,CHAIN,SP_PRIMARY,RES_BEG,RES_END,PDB_BEG,PDB_END,SP_BEG,SP_END\n"
        "12as,A,P00963,1,9,,9,1,9\n"
    )

    af_path = raw_path / "AlphaFoldDB" / "v4" / "swissprot_pdb"
    af_path.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent / "dummy" / "dummy.pdb",
        af_path / "AF-P00963-F1-model_v4.pdb",
    )

    monkeypatch.setattr(config, "raw_path", raw_path)
    return raw_path


def test_adapter_maps_invbench_to_alphafold(afcath_fixture):
    from bioverse.utilities import config

    config.workers = 1
    batches, split, assets = AlphaFoldInvBenchAdapter.download()
    batch = next(iter(batches))

    assert assets["mapping_summary"]["mapped"] == 1
    assert assets["mapping_summary"]["mapped_local"] == 1
    assert batch.data["molecule_id"][0][0][0] == "12as.A"
    assert batch.data["molecule_uniprot"][0][0][0] == "P00963"
    assert batch.data["molecule_structure_source"][0][0][0] == "alphafold"
    assert (
        split.names["ProteinInvBench_scene_split"][
            split.data["ProteinInvBench_scene_split"][0]
        ]
        == "train"
    )


def test_adapter_api_fallback(afcath_fixture):
    from bioverse.utilities import config

    config.workers = 1
    api_cache = afcath_fixture / "AlphaFoldDB/api"
    api_cache.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent / "dummy" / "dummy.pdb",
        api_cache / "AF-P00963-F1-model_v6.pdb",
    )

    invbench_path = afcath_fixture / "ProteinInvBench/data/cath4.2/chain_set.jsonl"
    invbench_path.write_text(
        '{"name": "12as.A", "seq": "WRONGSEQ", "CATH": ["3.30.930"], '
        '"coords": {"N": [[0,0,0]], "CA": [[0,0,0]], "C": [[0,0,0]], "O": [[0,0,0]]}}\n'
    )

    local_path = afcath_fixture / "AlphaFoldDB/v4/swissprot_pdb/AF-P00963-F1-model_v4.pdb"
    local_path.unlink()

    batches, split, assets = AlphaFoldInvBenchAdapter.download(api_workers=1)
    batch = next(iter(batches))
    assert assets["mapping_summary"]["mapped"] == 1
    assert assets["mapping_summary"]["mapped_api"] == 1
    assert batch.data["molecule_id"][0][0][0] == "12as.A"
