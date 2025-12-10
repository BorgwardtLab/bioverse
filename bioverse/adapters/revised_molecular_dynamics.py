from glob import glob
from pathlib import Path

import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import ATOM_ALPHABET, IteratorWithLength, batched, config, download


class RevisedMolecularDynamicsAdapter(Adapter):
    """Adapter for rMD17."""

    @classmethod
    def download(cls, name):
        assert name in [
            "aspirin",
            "paracetamol",
            "malonaldehyde",
            "naphthalene",
            "ethanol",
            "salicylic",
            "benzene",
            "toluene",
            "azobenzene",
            "uracil",
        ]
        path = config.raw_path / "RevisedMolecularDynamics17"
        download(
            "https://figshare.com/ndownloader/files/23950376",
            path,
            extension=".tar.bz2",
        )
        with open(path / "rmd17" / "splits" / "index_test_01.csv") as f:
            test_split = np.array(f.read().splitlines())
        with open(path / "rmd17" / "splits" / "index_train_01.csv") as f:
            train_split = np.array(f.read().splitlines())
        train_split, val_split = train_split[:-100], train_split[-100:]

        def generator():
            data = np.load(path / "rmd17" / "npz_data" / f"rmd17_{name}.npz")
            n, f = data["coords"].shape[1], 1000
            index = rng.choice(data["coords"].shape[0], size=f, replace=False)
            energies = data["energies"][index]
            charges = data["nuclear_charges"][index]
            data = {
                "frame_id": np.arange(f),
                "molecule_id": np.array([[name]] * f),
                "molecule_energy": energies.reshape(f, 1),
                "atom_pos": data["coords"].reshape(f, 1, 1, 1, n, 3),
                "atom_force": data["forces"].reshape(f, 1, 1, 1, n, 3),
                "atom_label": np.array(ATOM_ALPHABET)[charges - 1]
                .reshape(1, 1, 1, 1, n)
                .repeat(f, axis=0),
            }
            yield data

        batches = batched(IteratorWithLength(generator(), 1))
        return batches, Split([]), Assets({})
