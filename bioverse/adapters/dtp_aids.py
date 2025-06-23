from itertools import tee

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config, download, unzip_file


class DtpAidsAdapter(Adapter):
    """Adapter for DTP AIDS Antiviral Screen located at https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS+Antiviral+Screen+Data. Uses OGB splits. Note: Missing some data compared to the OGB dataset based on 2D structures, Train: 100, Val: 100, Test: 100"""

    @classmethod
    def download(cls):
        path = config.raw_path / "DTP-AIDS"

        # download splits from OGB
        download(
            "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
            path,
        )

        # unpack and parse class labels, smiles, and splits
        unzip_file(path / "hiv" / "mapping" / "mol.csv.gz")
        with open(path / "hiv" / "mapping" / "mol.csv") as file:
            lines = file.readlines()[1:]
        active, smiles, _ = zip(*[line.split(",") for line in lines])
        smiles = np.array(list(smiles))
        active = np.array(list(active)).astype(int)

        # construct split
        split = np.array([[-1]] * len(lines))
        original_split_len = {}
        for name in ["train", "test", "valid"]:
            unzip_file(path / "hiv" / "split" / "scaffold" / f"{name}.csv.gz")
            with open(path / "hiv" / "split" / "scaffold" / f"{name}.csv") as file:
                split_index = np.array([int(line) for line in file.readlines()])
            original_split_len[name] = len(split_index)
            split[split_index] = {"train": [0], "valid": [1], "test": [2]}[name]
        split_lookup = {k: v for k, v in zip(smiles, split)}
        split = np.array([split_lookup[smile] for smile in smiles])

        # sort by split for more effective data loading
        sort_index = np.argsort([int(s[0]) for s in split])
        smiles = smiles[sort_index]
        active = active[sort_index]
        split = split[sort_index]

        # create data
        def generator():
            for smile, label in zip(smiles, active):
                data = {"molecule_smiles": [[smile]], "molecule_label": [[label]]}
                yield ak.Record(data)

        batches = batched(IteratorWithLength(generator(), len(smiles)))
        return batches, Split(split, names=["train", "val", "test"]), Assets({})
