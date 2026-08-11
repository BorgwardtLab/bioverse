import shutil
from pathlib import Path

import pytest

from bioverse.processors import PdbProcessor
from bioverse.utilities.id_mapping import (
    uniprot_from_alphafold_path,
    uniprot_from_foldseek_target,
)


def test_uniprot_from_alphafold_path():
    assert uniprot_from_alphafold_path("AF-P00963-F1-model_v4.pdb") == "P00963"
    assert uniprot_from_alphafold_path("AF-P00963-F1-model_v4.pdb.gz") == "P00963"
    assert uniprot_from_alphafold_path("dummy.pdb") is None


def test_uniprot_from_foldseek_target():
    assert uniprot_from_foldseek_target("AF-P00963-F1-model_v4") == "P00963"


def test_pdb_processor_exclude(tmp_path):
    included = tmp_path / "AF-P00963-F1-model_v4.pdb"
    excluded = tmp_path / "AF-P01000-F1-model_v4.pdb"
    shutil.copy(Path(__file__).parent / "dummy" / "dummy.pdb", included)
    shutil.copy(Path(__file__).parent / "dummy" / "dummy.pdb", excluded)

    files = list(
        PdbProcessor.process(tmp_path, shuffle=False, exclude={"P01000"})
    )
    assert len(files) == 1
    assert PdbProcessor.exclude_key(included) not in {"P01000"}


def test_compile_foldseek_exclusions(tmp_path, monkeypatch):
    from bioverse.utilities import config
    from scripts.compile_foldseek_exclusions import compile_exclusions

    invbench = tmp_path / "ProteinInvBench/data/cath4.2"
    invbench.mkdir(parents=True)
    (invbench / "chain_set_splits.json").write_text(
        '{"train": [], "validation": [], "test": ["12as.A"]}'
    )
    sifts = tmp_path / "SIFTS"
    sifts.mkdir()
    (sifts / "pdb_chain_uniprot.csv").write_text(
        "PDB,CHAIN,SP_PRIMARY,RES_BEG,RES_END,PDB_BEG,PDB_END,SP_BEG,SP_END\n"
        "12as,A,P00963,1,9,,9,1,9\n"
    )
    monkeypatch.setattr(config, "raw_path", tmp_path)

    hits = tmp_path / "hits.m8"
    hits.write_text("query\tAF-P00734-F1-model_v4\t0.8\t0.9\t0.001\n")
    out = tmp_path / "exclusions.json"
    payload = compile_exclusions(hits, out, threshold=0.5)

    assert "P00963" in payload["excluded_uniprots"]
    assert "P00734" in payload["excluded_uniprots"]
    assert payload["n_excluded"] == 2


def test_adapter_exclusion_list_loaded():
    from bioverse.adapters.alpha_fold_exclusion import (
        EXCLUDED_UNIPROTS,
        EXCLUSIONS_PATH,
    )

    assert EXCLUSIONS_PATH.exists()
    assert len(EXCLUDED_UNIPROTS) > 1000


def test_adapter_download_excludes_structures(tmp_path, monkeypatch):
    import bioverse.adapters.alpha_fold_exclusion as adapter_mod
    from bioverse.utilities import config

    raw_path = tmp_path / "raw"
    af_path = raw_path / "AlphaFoldDB/v4/swissprot_pdb"
    af_path.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent / "dummy" / "dummy.pdb",
        af_path / "AF-P00963-F1-model_v4.pdb",
    )
    shutil.copy(
        Path(__file__).parent / "dummy" / "dummy.pdb",
        af_path / "AF-P01000-F1-model_v4.pdb",
    )
    monkeypatch.setattr(config, "raw_path", raw_path)
    config.workers = 1
    monkeypatch.setattr(adapter_mod, "EXCLUDED_UNIPROTS", frozenset({"P01000"}))
    batches, split, assets = adapter_mod.AlphaFoldExclusionAdapter.download()
    batch = next(iter(batches))

    assert assets["exclusions"]["count"] == 1
    assert len(batch) == 1
