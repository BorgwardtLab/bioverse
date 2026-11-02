import numpy as np
import pandas as pd
import pytest

from bioverse.adapters.huber_compatibility import compatibility_tables
from bioverse.adapters.huber_protease import WILDTYPE_SUBSTRATE
from bioverse.adapters.merops import (
    _assign_window_splits,
    _window_split_counts,
    expand_profile_windows,
    parse_cleavages,
    parse_combinatorial_profiles,
)
from bioverse.utilities.cleavage import (
    WINDOW_LEN,
    huber_window as pad_huber,
    merops_p4p4_window,
    prefer_uniprot,
    tokenize_window,
)


def test_merops_p4p4_window_left_pads_p6_p5():
    window = merops_p4p4_window(["Arg", "Thr", "Ile", "Ala", "Val", "Asn", "Tyr", "Thr"])
    assert window == "--RTIAVNYT" or window.startswith("--")
    assert len(window) == WINDOW_LEN
    assert window[2:] == "RTIAVNYT"


def test_huber_window_right_pads_primed_sites():
    window = pad_huber("ENLYFQS")
    assert window.startswith("ENLYFQS")
    assert len(window) == WINDOW_LEN
    assert window[7:] == "---"


def test_parse_cleavages_keeps_canonical_octamers():
    rows = [
        [
            "CLE1",
            "A01.001",
            "sub",
            "cleavage",
            "Arg",
            "Thr",
            "Ile",
            "Ala",
            "Val",
            "Asn",
            "Tyr",
            "Thr",
        ],
        [
            "CLE2",
            "A01.001",
            "syn",
            "cleavage",
            "-",
            "-",
            "Ac",
            "Phe",
            "TyI",
            "-",
            "-",
            "-",
        ],
        ["CLE3", "not-a-code", "sub", "cleavage", "A", "A", "A", "A", "A", "A", "A", "A"],
    ]
    windows = parse_cleavages(rows, min_canonical=4)
    assert list(windows) == ["A01.001"]
    assert windows["A01.001"] == ["--RTIAVNYT"]


def test_prefer_uniprot_swissprot():
    assert prefer_uniprot(["A0A087QPT2", "P0DJD8", "-"]) == "P0DJD8"


def test_combinatorial_profile_expansion_labels_allowed_residues():
    rows = [
        [
            "C04.004",
            "wild type",
            "-",
            "-",
            "-",
            "Q",
            "S",
            "-",
            "-",
            "-",
            "xxxQ*Sxxx",
            "ACC",
            "ref",
            "TEV",
        ]
    ]
    profiles = parse_combinatorial_profiles(rows)
    item = profiles["C04.004"][0]
    labeled = list(expand_profile_windows(item))
    positives = {window: label for window, label, *_ in labeled if label == 1.0}
    assert any("Q" in window for window in positives)
    assert all(len(window) == WINDOW_LEN for window, *_ in labeled)


def test_huber_compatibility_tables_encode_substrate_windows():
    df = pd.DataFrame(
        {
            "TEVp": ["WT", "H46A"],
            "TEVs": ["ENLYFQS", "ENLYFQS"],
            "n_mut_TEVp": [0, 1],
            "n_mut_TEVs": [0, 0],
            "TEVp_pos": [np.nan, 46],
            "TEVp_AA": [np.nan, "A"],
            "TEVs_pos": [np.nan, np.nan],
            "TEVs_AA": [np.nan, np.nan],
            "AUC": [1.0, 0.2],
        }
    )
    positions, labels, effects, windows, kinds = compatibility_tables(df)
    assert kinds == ["wt", "protease"]
    assert positions[0] == []
    assert positions[1] == [45]
    assert labels[1] == ["A"]
    assert windows[0] == pad_huber(WILDTYPE_SUBSTRATE)
    assert effects == pytest.approx([1.0, 0.2])


def test_tokenize_window_maps_gaps_to_unk():
    tokens = tokenize_window("ENLYFQS")
    assert len(tokens) == WINDOW_LEN
    assert tokens[0] != tokens[-1]
    assert tokens[-1] == 20


def _fake_protease_record(sequence, molecule_id):
    import awkward as ak

    n = len(sequence)
    backbone = ["N", "CA", "C", "O"]
    atom_label = [[backbone for _ in range(n)]]
    atom_pos = [
        [
            [
                [float(i) * 3.8, 0.0, 0.0],
                [float(i) * 3.8 + 1.5, 0.0, 0.0],
                [float(i) * 3.8 + 2.5, 0.0, 0.0],
                [float(i) * 3.8 + 3.2, 0.0, 0.0],
            ]
            for i in range(n)
        ]
    ]
    data = {
        "chain_label": ak.Array(["A"])[np.newaxis, np.newaxis],
        "residue_label": ak.Array([list(sequence)])[np.newaxis, np.newaxis],
        "residue_number": ak.Array([np.arange(1, n + 1)])[np.newaxis, np.newaxis],
        "atom_label": ak.Array(atom_label)[np.newaxis, np.newaxis],
        "atom_pos": ak.Array(atom_pos)[np.newaxis, np.newaxis],
        "molecule_id": ak.Array([[molecule_id]]),
    }
    return ak.Record(data)


def test_merops_adapter_holds_out_huber_and_cieplak(tmp_path, monkeypatch):
    from bioverse.adapters.merops import MeropsAdapter
    from bioverse.utilities import config

    config.raw_path = tmp_path
    root = tmp_path / "MEROPS"
    root.mkdir()
    (root / "Substrate_search.txt").write_text(
        "CLE1\tA01.001\tsub\tcleave\tArg\tThr\tIle\tAla\tVal\tAsn\tTyr\tThr\n"
        "CLE1b\tA01.001\tsub\tcleave\tPhe\tThr\tIle\tAla\tVal\tAsn\tTyr\tThr\n"
        "CLE2\tC04.004\tsub\tcleave\tGlu\tAsn\tLeu\tTyr\tPhe\tGln\tSer\tGly\n"
    )
    (root / "MID_2_UniProt.txt").write_text(
        '"Code","PepName","UniProt"\n'
        '"A01.001","pepsin A","P0DJD8"\n'
        '"C04.004","TEV","P04517"\n'
    )

    def fake_resolve(uniprot, **kwargs):
        return (
            _fake_protease_record("ARNDCEQGHILKMFPSTWYVARND", kwargs["molecule_id"]),
            "alphafold",
        )

    monkeypatch.setattr(
        "bioverse.adapters.merops.resolve_protease_structure", fake_resolve
    )
    batches, split, assets = MeropsAdapter.download(
        max_windows=4, max_length=80, seed=0
    )
    batch = next(iter(batches))
    assert "molecule_mutation_substrate" in batch
    assert "heldout" in split.names["random_scene_split"]
    assert "heldout" in split.names["random_mutation_split"]
    assert "train" in split.names["random_mutation_split"]
    assert split.default == "random_mutation_split"
    assert len(split.data["random_mutation_split"]) == 3
    assert assets["exclude_codes"] == ["C04.004", "M10.003", "M10.004"]


def test_window_split_stratifies_per_protease():
    assert _window_split_counts(1) == (1, 0, 0)
    assert _window_split_counts(2) == (1, 1, 0)
    assert _window_split_counts(10) == (8, 1, 1)
    labels = _assign_window_splits(
        [4, 1],
        ["A01.001", "C04.004"],
        exclude={"C04.004"},
        seed=0,
        train=0.8,
        val=0.1,
    )
    assert labels[-1] == "heldout"
    assert labels.count("heldout") == 1
    assert len(labels) == 5
    assert "train" in labels
    assert "val" in labels


def test_compatibility_task_tokenizes_substrate_and_mutates_protease():
    import awkward as ak

    from bioverse.adapters.merops import _attach_pairs
    from bioverse.data import Assets, Batch
    from bioverse.tasks.protease_substrate_compatibility import (
        ProteaseSubstrateCompatibilityTask,
    )
    from bioverse.transforms import OneHotResidueFeatures, TokenizeResidues
    from bioverse.utilities import IteratorWithLength, batched

    sequence = "ARNDCEQGHILKMFPSTWYV"
    record = _fake_protease_record(sequence, "TEVp")
    record = _attach_pairs(record, ["ENLYFQS---"], [0.4])
    data = {field: record[field] for field in record.fields}
    data["molecule_mutation_pos"] = [[ [[1]] ]]
    data["molecule_mutation_label"] = [[ [["G"]] ]]
    record = ak.Record(data)
    batch = next(iter(batched(IteratorWithLength(iter([record]), 1))))
    tokenizer = TokenizeResidues()
    batch = tokenizer.transform_batch(batch)
    assets = Assets()
    assets = OneHotResidueFeatures().transform_assets(assets)
    index = ak.Array(
        {
            "scene": np.array([0]),
            "frame": ak.Array([[0]]),
            "molecule": ak.Array([[0]]),
            "mutation": np.array([0]),
        }
    )
    X, y = ProteaseSubstrateCompatibilityTask()(batch, assets, index)
    letters = ak.to_list(X.molecules.residue_label)
    while isinstance(letters, list) and letters and isinstance(letters[0], list):
        letters = letters[0]
    assert "".join(letters).startswith("AGND")
    assert "substrate" in y.fields
    assert "window" in y.fields
    assert str(y["window"][0]) == "ENLYFQS---"
    assert len(y["substrate"][0]) == WINDOW_LEN
    assert float(y["target"][0]) == pytest.approx(0.4)


def test_batch_getitem_keeps_per_scene_mutation_tables():
    import awkward as ak

    from bioverse.adapters.merops import _attach_pairs
    from bioverse.data import Assets, Batch
    from bioverse.tasks.protease_substrate_compatibility import (
        ProteaseSubstrateCompatibilityTask,
    )
    from bioverse.transforms import OneHotResidueFeatures, TokenizeResidues
    from bioverse.utilities import IteratorWithLength, batched

    records = [
        _attach_pairs(
            _fake_protease_record("ARNDCEQGHILKMFPSTWYV", "A01.001"),
            [f"--RTIAVNY{i}"[:10] for i in range(45)],
            [1.0] * 45,
        ),
        _attach_pairs(
            _fake_protease_record("ARNDCEQGHILKMFPSTWYV", "A01.002"),
            [f"--EYGLFQIN"] * 64,
            [0.5] * 64,
        ),
    ]
    batch = next(iter(batched(IteratorWithLength(iter(records), 2))))
    batch = TokenizeResidues().transform_batch(batch)
    assets = OneHotResidueFeatures().transform_assets(Assets())
    index = ak.Array(
        {
            "scene": np.array([0, 1]),
            "frame": ak.Array([[0], [0]]),
            "molecule": ak.Array([[0], [0]]),
            "mutation": np.array([0, 51]),
        }
    )
    selected = batch[index["scene"], index["frame"], index["molecule"]]
    ids = ak.to_list(selected.data["molecule_id"])
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = [row[0] if isinstance(row, list) else row for row in ids]
    assert ids[0] == "A01.001"
    assert ids[1] == "A01.002"
    X, y = ProteaseSubstrateCompatibilityTask()(batch, assets, index)
    assert str(y["window"][0]).startswith("--RTIAVNY")
    assert str(y["window"][1]) == "--EYGLFQIN"
    assert float(y["target"][1]) == pytest.approx(0.5)


def test_compatibility_task_applies_sparse_protease_mutations_in_a_batch():
    import awkward as ak

    from bioverse.data import Assets
    from bioverse.tasks.protease_substrate_compatibility import (
        ProteaseSubstrateCompatibilityTask,
    )
    from bioverse.transforms import OneHotResidueFeatures, TokenizeResidues
    from bioverse.utilities import IteratorWithLength, batched

    sequence = "ARNDCEQGHILKMFPSTWYV"
    records = []
    for molecule_id, pos, lab, window in (
        ("TEVp", [1], ["G"], "ENLYFQS---"),
        ("TEVp", [], [], "ANLYFQS---"),
        ("TEVp", [2], ["C"], "ENLYFQS---"),
    ):
        record = _fake_protease_record(sequence, molecule_id)
        data = {field: record[field] for field in record.fields}
        data["molecule_mutation_effect"] = [[[0.4]]]
        data["molecule_mutation_substrate"] = [[[window]]]
        data["molecule_mutation_pos"] = [[[pos]]]
        data["molecule_mutation_label"] = [[[lab]]]
        records.append(ak.Record(data))
    batch = next(iter(batched(IteratorWithLength(iter(records), 3))))
    batch = TokenizeResidues().transform_batch(batch)
    assets = OneHotResidueFeatures().transform_assets(Assets())
    index = ak.Array(
        {
            "scene": np.array([0, 1, 2]),
            "frame": ak.Array([[0], [0], [0]]),
            "molecule": ak.Array([[0], [0], [0]]),
            "mutation": np.array([0, 0, 0]),
        }
    )
    X, y = ProteaseSubstrateCompatibilityTask()(batch, assets, index)
    letters = ak.to_list(X.molecules.residue_label)
    while (
        isinstance(letters, list)
        and letters
        and isinstance(letters[0], list)
        and letters[0]
        and isinstance(letters[0][0], list)
    ):
        letters = [row[0] for row in letters]
    seqs = ["".join(row) if isinstance(row[0], str) else "".join(row[0]) for row in letters]
    assert seqs[0].startswith("AGND")
    assert seqs[1].startswith("ARND")
    assert seqs[2].startswith("ARCD")
    assert str(y["window"][1]) == "ANLYFQS---"


def test_compatibility_task_reads_windows_when_mutation_tables_are_empty():
    import awkward as ak

    from bioverse.data import Assets, Batch
    from bioverse.tasks.protease_substrate_compatibility import (
        ProteaseSubstrateCompatibilityTask,
    )
    from bioverse.transforms import OneHotResidueFeatures, TokenizeResidues
    from bioverse.utilities import IteratorWithLength, batched

    sequence = "ARNDCEQGHILKMFPSTWYV"
    record = _fake_protease_record(sequence, "A01.001")
    windows = [f"--RTIAVNY{i}"[:10] for i in range(45)]
    data = {field: record[field] for field in record.fields}
    data["molecule_mutation_effect"] = [[ [1.0] * 45 ]]
    data["molecule_mutation_substrate"] = [[ windows ]]
    data["molecule_mutation_label"] = [[ [[ ] for _ in windows] ]]
    data["molecule_mutation_pos"] = [[ [[ ] for _ in windows] ]]
    record = ak.Record(data)
    batch = next(iter(batched(IteratorWithLength(iter([record]), 1))))
    batch = TokenizeResidues().transform_batch(batch)
    assets = OneHotResidueFeatures().transform_assets(Assets())
    index = ak.Array(
        {
            "scene": np.array([0, 0]),
            "frame": ak.Array([[0], [0]]),
            "molecule": ak.Array([[0], [0]]),
            "mutation": np.array([0, 51]),
        }
    )
    X, y = ProteaseSubstrateCompatibilityTask()(batch, assets, index)
    assert str(y["window"][0]) == windows[0]
    assert float(y["target"][0]) == pytest.approx(1.0)
    assert str(y["window"][1]) == "-" * WINDOW_LEN


def test_keep_longest_chain_wraps_string_chain_labels():
    import awkward as ak

    from bioverse.adapters.protease_structures import keep_longest_chain

    short = list("ALA")
    long = list("CDEFGHIK")
    record = ak.Record(
        {
            "chain_label": ak.Array(["A", "B"])[np.newaxis, np.newaxis],
            "residue_label": ak.Array([short, long])[np.newaxis, np.newaxis],
            "residue_number": ak.Array(
                [list(range(1, 4)), list(range(1, 9))]
            )[np.newaxis, np.newaxis],
            "atom_label": ak.Array(
                [[["CA"]] * 3, [["CA"]] * 8]
            )[np.newaxis, np.newaxis],
            "atom_pos": ak.Array(
                [
                    [[[float(i), 0.0, 0.0]] for i in range(3)],
                    [[[float(i), 1.0, 0.0]] for i in range(8)],
                ]
            )[np.newaxis, np.newaxis],
            "molecule_id": ak.Array([["prot"]]),
        }
    )
    cropped = keep_longest_chain(record)
    labels = ak.to_list(cropped["residue_label"])
    while isinstance(labels, list) and labels and isinstance(labels[0], list):
        labels = labels[0]
    assert "".join(labels) == "CDEFGHIK"
    chain = ak.to_list(cropped["chain_label"])
    while isinstance(chain, list) and len(chain) == 1:
        chain = chain[0]
    assert chain == "B"


def test_as_protease_record_batches_single_and_multi_chain():
    import awkward as ak

    from bioverse.adapters.merops import _attach_pairs
    from bioverse.adapters.protease_structures import as_protease_record
    from bioverse.data import Batch

    short = list("ALA")
    long = list("CDEFGHIK")
    multi = ak.Record(
        {
            "chain_label": ak.Array(["A", "B"])[np.newaxis, np.newaxis],
            "residue_label": ak.Array([short, long])[np.newaxis, np.newaxis],
            "residue_number": ak.Array(
                [list(range(1, 4)), list(range(1, 9))]
            )[np.newaxis, np.newaxis],
            "atom_label": ak.Array(
                [[["CA"]] * 3, [["CA"]] * 8]
            )[np.newaxis, np.newaxis],
            "atom_pos": ak.Array(
                [
                    [[[float(i), 0.0, 0.0]] for i in range(3)],
                    [[[float(i), 1.0, 0.0]] for i in range(8)],
                ]
            )[np.newaxis, np.newaxis],
            "molecule_id": ak.Array([["multi"]]),
        }
    )
    single = _fake_protease_record("ARNDCEQGHILKMFPSTWYV", "single")
    records = [
        _attach_pairs(as_protease_record(single, molecule_id="single"), ["--ENLYFQSG"]),
        _attach_pairs(as_protease_record(multi, molecule_id="multi"), ["--RTIAVNYT"]),
    ]
    batch = Batch(ak.Array(records))
    assert len(batch) == 2
    assert "residue" in batch.toc
    assert int(ak.sum(batch.toc["residue"])) == 20 + 8


def test_peel_sequence_maps_nan_residue_labels():
    import awkward as ak

    from bioverse.adapters.merops import _peel_sequence

    labels = list("ARNDCEQGHILKMFPSTWYV")
    labels[5] = float("nan")
    record = ak.Record(
        {"residue_label": ak.Array([labels])[np.newaxis, np.newaxis]}
    )
    seq = _peel_sequence(record)
    assert len(seq) == 20
    assert seq[5] == "X"
    assert seq[:5] == "ARNDC"


def test_cieplak_parser_reads_hexamer_column_not_caption():
    from pathlib import Path

    from bioverse.adapters.cieplak import parse_cieplak_xlsx
    from bioverse.utilities import config

    path = Path(config.raw_path) / "CIEPLK" / "cieplak_s10.xlsx"
    if not path.exists():
        pytest.skip("Cieplak workbook not cached")
    pairs = parse_cieplak_xlsx(path)
    assert len(pairs["MMP2"]) == 100
    assert len(pairs["MMP9"]) == 100
    window, value = pairs["MMP2"][0]
    assert len(window) == WINDOW_LEN
    assert value > 0


def test_huber_aligns_when_pdb_is_shorter_than_query():
    from bioverse.adapters.huber_compatibility import _align_offset
    from bioverse.adapters.huber_protease import WILDTYPE_PROTEASE

    tagged = "GHHHHHHH" + WILDTYPE_PROTEASE[:220]
    overlap = _align_offset(WILDTYPE_PROTEASE, tagged)
    assert overlap is not None
    q0, s0, length = overlap
    assert q0 == 0
    assert s0 == 8
    assert length == 220


def test_mutation_sampler_empty_index():
    import awkward as ak

    from bioverse.samplers.mutation import MutationSampler

    toc = ak.Array(
        {"scene": np.array([0, 1]), "mutations": np.array([0, 0])}
    )
    index = MutationSampler().index(toc, np.array([True, True]))
    assert len(index["scene"]) == 0
    assert len(index["mutation"]) == 0

