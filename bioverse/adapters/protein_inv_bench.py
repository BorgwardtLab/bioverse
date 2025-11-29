import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import (
    IteratorWithLength,
    batched,
    config,
    download,
    interleave,
    load,
    progressbar,
)


class ProteinInvBenchAdapter(Adapter):
    """Adapter for ProteinInvBench."""

    @classmethod
    def download(cls):
        path = config.raw_path / "ProteinInvBench"
        download(
            "https://github.com/A4Bio/ProteinInvBench/releases/download/dataset_release/data.tar.gz",
            path,
        )
        split_lookup = load(path / "data" / "cath4.2" / "chain_set_splits.json")
        split_lookup = {
            v: [i]
            for i, k in enumerate(["train", "validation", "test"])
            for v in split_lookup[k]
        }
        proteins = load(path / "data" / "cath4.2" / "chain_set.jsonl")
        proteins = np.array([item for item in proteins if item["name"] in split_lookup])
        split = np.array([split_lookup[item["name"]] for item in proteins])

        # sort for optimal data loading
        sort_index = np.argsort(np.ravel(split))
        proteins = proteins[sort_index]
        split = split[sort_index]

        def generator():
            for item in proteins:
                atom_pos = np.array(
                    interleave(
                        item["coords"]["N"],
                        item["coords"]["CA"],
                        item["coords"]["C"],
                        item["coords"]["O"],
                    )
                )
                atom_label = np.array(["N", "CA", "C", "O"] * len(item["seq"]))
                atom_mask = ~np.any(np.isnan(atom_pos), axis=-1)
                atom_counts = ak.sum(ak.unflatten(atom_mask, 4), axis=-1)
                atom_pos = ak.unflatten(atom_pos[atom_mask], atom_counts)
                atom_label = ak.unflatten(atom_label[atom_mask], atom_counts)
                residue_label = np.array([aa for aa in item["seq"]])
                data = {
                    "molecule_id": [[item["name"]]],
                    "molecule_cath": [[item["CATH"]]],
                    "residue_label": [[[residue_label]]],
                    "atom_pos": [[[atom_pos]]],
                    "atom_label": [[[atom_label]]],
                }
                yield ak.Record(data)

        batches = batched(IteratorWithLength(generator(), len(proteins)))
        return batches, Split(split, names=["train", "val", "test"]), Assets({})
