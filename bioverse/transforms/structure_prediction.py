from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import yaml

from ..data import Assets, Batch, Split
from ..processors import CifProcessor, PdbProcessor
from ..transform import Transform
from ..utilities import (
    CHAIN_ALPHABET,
    PROTEIN_ALPHABET,
    IteratorWithLength,
    config,
    note,
    progressbar,
)

BACKBONE_ATOMS = ("N", "CA", "C", "O")
AA_INDEX = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}
STRUCTURE_FIELDS = (
    "atom_pos",
    "atom_label",
    "atom_b_factor",
    "residue_label",
    "residue_number",
    "residue_pLDDT",
    "residue_pos",
    "residue_token",
    "residue_backbone",
    "chain_label",
    "molecule_id",
)

_MODEL_ALIASES = {
    "boltz2": "boltz2",
    "boltz-2": "boltz2",
    "boltz 2": "boltz2",
}


def _normalize_model(model: str) -> str:
    key = str(model).strip().lower().replace("_", "-")
    if key not in _MODEL_ALIASES:
        supported = ", ".join(sorted(_MODEL_ALIASES))
        raise ValueError(
            f"Unknown structure prediction model {model!r}. "
            f"Supported models: {supported}."
        )
    return _MODEL_ALIASES[key]


def _peel_to_chains(array: ak.Array) -> ak.Array:
    """Drop leading singleton scene/frame/molecule wrappers, leaving [chain, ...]."""
    array = ak.Array(array)
    while array.ndim > 1 and len(array) == 1:
        array = array[0]
    return array


def _nest_molecule(chain_level) -> ak.Array:
    """Wrap chain-level values as [scene, frame, molecule, chain, ...]."""
    array = ak.Array(chain_level)
    return array[np.newaxis, np.newaxis, np.newaxis]


def _chain_ids(value) -> list[str]:
    return [str(item) for item in ak.to_list(_peel_to_chains(value))]


def _reorder_chains(chain_level, pred_ids: list[str], want_ids: list[str]):
    """Align predicted chains to the input chain order."""
    array = _peel_to_chains(chain_level)
    if list(pred_ids) == list(want_ids):
        return array
    index = {chain: i for i, chain in enumerate(pred_ids)}
    if all(chain in index for chain in want_ids):
        return ak.Array([array[index[chain]] for chain in want_ids])
    if len(pred_ids) != len(want_ids):
        raise RuntimeError(
            f"Predicted chains {pred_ids} do not match input chains {want_ids}."
        )
    return array


def _dense_backbone(atom_label, atom_pos, sequences: list[str], chain_ids: list[str]):
    """Build one N/CA/C/O frame per input residue. Never drop residues."""
    labels = ak.to_list(_peel_to_chains(atom_label))
    coords = ak.to_list(_peel_to_chains(atom_pos))
    if len(labels) != len(sequences):
        raise RuntimeError(
            f"Predicted {len(labels)} chains, expected {len(sequences)} ({chain_ids})."
        )
    residue_pos, residue_backbone, residue_token, residue_label = [], [], [], []
    for chain_i, (seq, chain_id) in enumerate(zip(sequences, chain_ids)):
        lab_chain = labels[chain_i]
        pos_chain = coords[chain_i]
        if len(lab_chain) != len(seq):
            raise RuntimeError(
                f"Chain {chain_id} has {len(lab_chain)} predicted residues, "
                f"expected {len(seq)} from the input sequence."
            )
        pos, backbone, tokens, letters = [], [], [], []
        for res_i, aa in enumerate(seq):
            names = [str(name).strip() for name in lab_chain[res_i]]
            xyz = np.asarray(pos_chain[res_i], dtype=np.float32)
            lookup = {name: xyz[j] for j, name in enumerate(names)}
            missing = [atom for atom in BACKBONE_ATOMS if atom not in lookup]
            if missing:
                raise RuntimeError(
                    f"Chain {chain_id} residue {res_i} ({aa}) is missing "
                    f"backbone atoms {missing}."
                )
            pos.append(lookup["CA"])
            backbone.append(
                np.stack([lookup[atom] for atom in BACKBONE_ATOMS], axis=0)
            )
            tokens.append(AA_INDEX.get(aa, -1))
            letters.append(aa)
        residue_pos.append(np.stack(pos, axis=0))
        residue_backbone.append(np.stack(backbone, axis=0))
        residue_token.append(np.asarray(tokens, dtype=np.int64))
        residue_label.append(letters)
    return residue_pos, residue_backbone, residue_token, residue_label


def _molecule_chains(batch: Batch, scene: int) -> tuple[list[str], list[str]]:
    residue_label = _peel_to_chains(batch.data["residue_label"][scene])
    sequences = ["".join(ak.to_list(chain)) for chain in residue_label]
    if "chain_label" in batch.data:
        chain_label = _peel_to_chains(batch.data["chain_label"][scene])
        while getattr(chain_label, "ndim", 0) > 1:
            chain_label = chain_label[0]
        chain_ids = [str(x) for x in ak.to_list(chain_label)]
    else:
        chain_ids = list(CHAIN_ALPHABET[: len(sequences)])
    if len(chain_ids) < len(sequences):
        chain_ids.extend(CHAIN_ALPHABET[len(chain_ids) : len(sequences)])
    return chain_ids[: len(sequences)], sequences


def _sequence_key(sequences: list[str]) -> str:
    return hashlib.sha256("\n".join(sequences).encode()).hexdigest()[:16]


def _find_prediction(out_dir: Path, stem: str) -> Path:
    patterns = (
        f"{stem}_model_0.cif",
        f"{stem}_model_0.pdb",
        f"{stem}.cif",
        f"{stem}.pdb",
    )
    for pattern in patterns:
        matches = sorted(out_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No Boltz prediction found for {stem!r} under {out_dir}."
    )


def _parse_structure(path: Path) -> ak.Record:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".pdb") or suffix.endswith(".pdb.gz"):
        return PdbProcessor.process_file(path, pLDDT=True)
    return CifProcessor.process_file(path, pLDDT=True)


def _overlay_structures(
    batch: Batch,
    records: list[ak.Record],
    molecules: list[tuple[list[str], list[str]]],
) -> Batch:
    """Write predicted coordinates into standard scene/frame/molecule nesting.

    Sequence identity comes from the parent batch so mutation indices stay
    aligned. Missing backbone atoms or a residue-count change raise instead of
    dropping residues.
    """
    data = {
        field: batch.data[field]
        for field in batch.data
        if field not in STRUCTURE_FIELDS
    }
    stacked: dict[str, list[ak.Array]] = {field: [] for field in STRUCTURE_FIELDS}
    for scene, (record, (chain_ids, sequences)) in enumerate(zip(records, molecules)):
        pred_ids = _chain_ids(record["chain_label"])
        atom_label = _reorder_chains(record["atom_label"], pred_ids, chain_ids)
        atom_pos = _reorder_chains(record["atom_pos"], pred_ids, chain_ids)
        pos, backbone, tokens, letters = _dense_backbone(
            atom_label, atom_pos, sequences, chain_ids
        )
        stacked["atom_label"].append(_nest_molecule(atom_label))
        stacked["atom_pos"].append(_nest_molecule(atom_pos))
        stacked["chain_label"].append(_nest_molecule(chain_ids))
        stacked["residue_label"].append(_nest_molecule(letters))
        stacked["residue_pos"].append(_nest_molecule(pos))
        stacked["residue_token"].append(_nest_molecule(tokens))
        stacked["residue_backbone"].append(_nest_molecule(backbone))
        if "residue_number" in batch.data:
            stacked["residue_number"].append(
                _nest_molecule(_peel_to_chains(batch.data["residue_number"][scene]))
            )
        elif "residue_number" in record.fields:
            stacked["residue_number"].append(
                _nest_molecule(_reorder_chains(record["residue_number"], pred_ids, chain_ids))
            )
        if "molecule_id" in batch.data:
            molecule_id = ak.to_list(_peel_to_chains(batch.data["molecule_id"][scene]))
            stacked["molecule_id"].append(ak.Array([[[str(molecule_id[0])]]]))
        elif "molecule_id" in record.fields:
            molecule_id = ak.to_list(record["molecule_id"])
            stacked["molecule_id"].append(ak.Array([[[str(molecule_id[0])]]]))

    for field, values in stacked.items():
        if values:
            data[field] = ak.concatenate(values, axis=0)
    return Batch(data)


class StructurePredictionTransform(Transform):
    """Predict 3D coordinates for each molecule with a folding model.

    Currently supports Boltz 2. Each batch is written as a directory of YAML
    inputs and predicted in a single ``boltz predict`` call. Parsed coordinates
    are written as a standard scene / frame / molecule / chain batch, including
    dense ``residue_pos`` / ``residue_token`` / ``residue_backbone``. Mutation
    tables and other molecule annotations are preserved. Residue counts must
    match the input sequences.
    """

    def __init__(
        self,
        model: str = "boltz2",
        use_msa_server: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.model = _normalize_model(model)
        self.use_msa_server = use_msa_server
        self.cache_dir = cache_dir

    def _resolved_cache_dir(self) -> Path:
        if self.cache_dir is None:
            return config.raw_path / self.model
        return Path(self.cache_dir)

    def transform(
        self, batches, split: Split, assets: Assets
    ) -> tuple[IteratorWithLength, Split, Assets]:
        assets = self.transform_assets(assets)
        split = self.transform_split(split)

        def generate():
            for batch in progressbar(
                batches,
                description=self.__class__.__name__,
                total=len(batches) if hasattr(batches, "__len__") else None,
            ):
                yield self.transform_batch(batch)

        return (
            IteratorWithLength(
                generate(),
                length=len(batches) if hasattr(batches, "__len__") else None,
            ),
            split,
            assets,
        )

    def transform_batch(self, batch: Batch) -> Batch:
        molecules = [
            _molecule_chains(batch, scene) for scene in range(len(batch))
        ]
        cache_dir = self._resolved_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        records: list[ak.Record | None] = [None] * len(molecules)
        missing: list[int] = []
        for scene, (chain_ids, sequences) in enumerate(molecules):
            cached = cache_dir / f"{_sequence_key(sequences)}.cif"
            if not cached.exists():
                cached = cache_dir / f"{_sequence_key(sequences)}.pdb"
            if cached.exists():
                records[scene] = _parse_structure(cached)
            else:
                missing.append(scene)

        if missing:
            predicted = self._predict_missing(
                [molecules[i] for i in missing],
                cache_dir,
            )
            for scene, record in zip(missing, predicted):
                records[scene] = record

        if any(record is None for record in records):
            raise RuntimeError("Structure prediction did not return a model for every scene.")
        return _overlay_structures(batch, records, molecules)  # type: ignore[arg-type]

    def _predict_missing(
        self,
        molecules: list[tuple[list[str], list[str]]],
        cache_dir: Path,
    ) -> list[ak.Record]:
        with tempfile.TemporaryDirectory(prefix="bioverse-boltz-") as tmp:
            tmp = Path(tmp)
            input_dir = tmp / "inputs"
            out_dir = tmp / "out"
            input_dir.mkdir()
            out_dir.mkdir()
            stems = []
            for scene, (chain_ids, sequences) in enumerate(molecules):
                stem = f"{scene:04d}"
                stems.append(stem)
                payload = {
                    "version": 1,
                    "sequences": [
                        self._protein_entry(chain_id, sequence)
                        for chain_id, sequence in zip(chain_ids, sequences)
                    ],
                }
                with open(input_dir / f"{stem}.yaml", "w") as handle:
                    yaml.safe_dump(payload, handle, sort_keys=False)

            self.run_predictor(input_dir, out_dir)

            records = []
            for stem, (_, sequences) in zip(stems, molecules):
                path = _find_prediction(out_dir, stem)
                cached = cache_dir / f"{_sequence_key(sequences)}{path.suffix}"
                shutil.copy2(path, cached)
                records.append(_parse_structure(cached))
            return records

    def _protein_entry(self, chain_id: str, sequence: str) -> dict:
        entry = {"protein": {"id": chain_id, "sequence": sequence}}
        if not self.use_msa_server:
            entry["protein"]["msa"] = "empty"
        return entry

    def run_predictor(self, input_dir: Path, out_dir: Path) -> None:
        """Run Boltz 2 on a directory of YAML inputs."""
        if self.model != "boltz2":
            raise ValueError(f"Unsupported model {self.model!r}.")
        if shutil.which("boltz") is None:
            raise RuntimeError(
                "Boltz is required for StructurePredictionTransform. "
                "Install it with `pip install boltz` and ensure `boltz` is on PATH."
            )
        command = [
            "boltz",
            "predict",
            str(input_dir),
            "--out_dir",
            str(out_dir),
            "--model",
            "boltz2",
            "--override",
            "--no_kernels",
        ]
        if self.use_msa_server:
            command.append("--use_msa_server")
        note("Running Boltz 2 structure prediction.")
        subprocess.run(command, check=True)
