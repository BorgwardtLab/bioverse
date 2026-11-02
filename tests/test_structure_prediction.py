from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
import yaml

from bioverse.adapters.huber_protease import WILDTYPE_SUBSTRATE, _sequence_record
from bioverse.transforms.structure_prediction import (
    StructurePredictionTransform,
    _peel_to_chains,
)
from bioverse.utilities import IteratorWithLength, batched
from tests.test_huber_protease import fake_boltz_predictor


def _predict(tmp_path, record, predictor=fake_boltz_predictor):
    batch = next(iter(batched(IteratorWithLength(iter([record]), 1))))
    transform = StructurePredictionTransform(
        model="boltz2",
        use_msa_server=False,
        cache_dir=tmp_path / "boltz",
    )
    with patch.object(StructurePredictionTransform, "run_predictor", predictor):
        return transform.transform_batch(batch)


def _omit_oxygen(self, input_dir, out_dir):
    fake_boltz_predictor(self, input_dir, out_dir)
    for path in Path(out_dir).rglob("*_model_0.pdb"):
        kept = [
            line
            for line in path.read_text().splitlines(keepends=True)
            if not line.startswith("ATOM") or " O " not in line
        ]
        path.write_text("".join(kept))


def test_overlay_uses_molecule_axis_and_residue_fields(tmp_path):
    out = _predict(tmp_path, _sequence_record("ARND", WILDTYPE_SUBSTRATE))
    residue_label = out.data["residue_label"]
    assert residue_label.ndim == 5
    assert [len(residue_label[0, 0, 0, i]) for i in range(2)] == [4, len(WILDTYPE_SUBSTRATE)]
    assert "residue_pos" in out
    assert "residue_token" in out
    assert "residue_backbone" in out
    pos = np.asarray(ak.to_numpy(ak.ravel(out.data["residue_pos"])))
    assert pos.size == (4 + len(WILDTYPE_SUBSTRATE)) * 3
    backbone = ak.to_list(_peel_to_chains(out.data["residue_backbone"][0]))
    assert np.asarray(backbone[0]).shape == (4, 4, 3)


def test_molecule_index_keeps_every_chain(tmp_path):
    out = _predict(tmp_path, _sequence_record("ARND", WILDTYPE_SUBSTRATE))
    sliced = out[np.asarray([0]), np.asarray([0]), np.asarray([0])]
    chains = [
        "".join(chain)
        for chain in ak.to_list(_peel_to_chains(sliced.data["residue_label"]))
    ]
    assert chains == ["ARND", WILDTYPE_SUBSTRATE]


def test_missing_backbone_atom_raises(tmp_path):
    with pytest.raises(RuntimeError, match="missing backbone atoms"):
        _predict(
            tmp_path,
            _sequence_record("ARND", WILDTYPE_SUBSTRATE),
            predictor=_omit_oxygen,
        )


def test_residue_count_mismatch_raises(tmp_path):
    def extra_residue(self, input_dir, out_dir):
        input_dir = Path(input_dir)
        for yaml_path in sorted(input_dir.glob("*.yaml")):
            spec = yaml.safe_load(yaml_path.read_text())
            spec["sequences"][0]["protein"]["sequence"] += "A"
            yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False))
        fake_boltz_predictor(self, input_dir, out_dir)

    with pytest.raises(RuntimeError, match="predicted residues"):
        _predict(
            tmp_path,
            _sequence_record("ARND", WILDTYPE_SUBSTRATE),
            predictor=extra_residue,
        )
