import re

import awkward as ak
import numpy as np
from fastavro import reader as avro_reader

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config, download, load

REPOSITORY_URL = "https://zenodo.org/records/15259912/files"

DATASETS = [
    "RCSBDataset",
    "SCOPDataset",
    "TMAlignDataset",
    "GeneOntologyDataset",
    "EnzymeCommissionDataset",
    "ProteinFamilyDataset",
    "ProteinProteinInterfaceDataset",
    "ProteinLigandDecoysDataset",
    "AlphaFoldDataset_swissprot",
]

ADDITIONAL_FILES = {
    "TMAlignDataset": [
        "TMAlignDataset.tmscore.npy",
        "TMAlignDataset.rmsd.npy",
        "TMAlignDataset.gdt.npy",
        "TMAlignDataset.lddt.npy",
    ],
    "GeneOntologyDataset": ["GeneOntologyDataset.godag.obo"],
    "ProteinProteinInterfaceDataset": [
        "ProteinProteinInterfaceDataset.interfaces.json"
    ],
}

SPLIT_KEYS = (
    ["random_split"]
    + [f"sequence_split_{c}" for c in ("0.5", "0.6", "0.7", "0.8", "0.9")]
    + [
        f"structure_split_{c}"
        for c in ("0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
    ]
)

OPTIONAL_FIELDS = (
    ("SASA", "residue_SASA"),
    ("RSA", "residue_RSA"),
    ("pLDDT", "residue_pLDDT"),
    ("is_interface", "residue_is_interface"),
)


def split_name(key: str) -> str:
    if key == "random_split":
        return "random_scene_split"
    if key.startswith("sequence_split_"):
        return f"sequence_{int(float(key.rsplit('_', 1)[-1]) * 100)}_scene_split"
    if key.startswith("structure_split_"):
        return f"structure_{int(float(key.rsplit('_', 1)[-1]) * 100)}_scene_split"
    return f"{key.removesuffix('_split')}_scene_split"


def convert_protein(protein: dict) -> ak.Record:
    meta = protein["protein"]
    atom = protein["atom"]
    data = {"molecule_id": ak.Array([[meta["ID"]]])}
    for key, value in meta.items():
        if key in {"ID", "sequence"} or key in SPLIT_KEYS:
            continue
        field = "molecule_" + re.sub(r"[^0-9a-zA-Z]+", "_", key).strip("_").lower()
        data[field] = ak.Array([[value]])
        if key == "EC":
            levels = value.split(".")
            for i in range(1, len(levels) + 1):
                data[f"molecule_ec{i}"] = ak.Array([[ ".".join(levels[:i]) ]])

    n = len(atom["x"])
    chain_id = atom.get("chain_id", ["A"] * n)
    chain_sizes = ak.run_lengths(ak.Array(chain_id))
    residue_sizes = ak.run_lengths(ak.Array(atom["residue_number"]))
    pos = np.stack([atom["x"], atom["y"], atom["z"]], axis=-1)
    unflat = lambda x: ak.Array(x).unflatten(chain_sizes).unflatten(residue_sizes, 1)
    data |= {
        "chain_label": unflat(chain_id).firsts(2).firsts(1),
        "residue_number": unflat(atom["residue_number"]).firsts(2),
        "residue_label": unflat(atom["residue_type"]).firsts(2),
        "atom_label": unflat(atom["atom_type"]),
        "atom_pos": unflat(pos),
    }
    for key, target in OPTIONAL_FIELDS:
        if key in atom:
            data[target] = unflat(atom[key]).firsts(2)

    return ak.Record({k: v[np.newaxis, np.newaxis] for k, v in data.items()})


def residue_count(protein: dict) -> int:
    return len(ak.run_lengths(ak.Array(protein["atom"]["residue_number"])))


class ProteinShakeAdapter(Adapter):
    """Adapter for ProteinShake precomputed datasets hosted on Zenodo."""

    @classmethod
    def download(cls, dataset: str):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}")

        path = config.raw_path / "ProteinShake" / dataset
        avro = path / f"{dataset}.atom.avro"
        if not avro.exists():
            download(
                f"{REPOSITORY_URL}/{dataset}.atom.avro.gz",
                path / f"{dataset}.atom",
            )

        protein_ids, chain_lengths, split_data, split_keys = [], {}, {}, None

        def generator():
            nonlocal split_keys, split_data
            with open(avro, "rb") as file:
                for protein in avro_reader(file):
                    meta = protein["protein"]
                    protein_id = meta["ID"]
                    protein_ids.append(protein_id)
                    chain_lengths[protein_id] = residue_count(protein)

                    if split_keys is None:
                        split_keys = [key for key in SPLIT_KEYS if key in meta]
                        split_keys += [
                            key
                            for key in meta
                            if key.endswith("_split") and key not in split_keys
                        ]
                        split_data = {split_name(key): [] for key in split_keys}
                    for key in split_keys:
                        split_data[split_name(key)].append(meta[key])

                    yield convert_protein(protein)

        records = list(generator())

        default = split_name("random_split")
        split = (
            Split(
                split_data,
                default=(
                    default if default in split_data else next(iter(split_data), None)
                ),
            )
            if split_data
            else Split()
        )

        assets = Assets({"protein_ids": protein_ids})
        for filename in ADDITIONAL_FILES.get(dataset, []):
            meta_path = path / filename
            if not meta_path.exists():
                download(f"{REPOSITORY_URL}/{filename}.gz", path / meta_path.stem)
            key = filename.removeprefix(f"{dataset}.")
            for suffix in (".npy", ".json", ".obo"):
                key = key.removesuffix(suffix)
            if meta_path.suffix == ".json":
                interfaces = load(meta_path)
                assets["interfaces"] = interfaces
                assets["interface_contacts"] = {
                    "path": str(meta_path),
                    "chain_lengths": chain_lengths,
                }
            elif meta_path.suffix == ".npy":
                assets[key] = {
                    "path": str(meta_path),
                    "index": {protein_id: i for i, protein_id in enumerate(protein_ids)},
                }
            else:
                assets[key] = str(meta_path)

        return (
            batched(IteratorWithLength(iter(records), len(records))),
            split,
            assets,
        )
