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

        def generator():
            data = np.load(path / "rmd17" / "npz_data" / f"rmd17_{name}.npz")
            f, n, _ = data["coords"].shape
            data = {
                "frame_id": np.arange(f),
                "molecule_id": np.array([[name]] * f),
                "molecule_energy": data["energies"].reshape(f, 1),
                "atom_pos": data["coords"].reshape(f, 1, 1, 1, n, 3),
                "atom_force": data["forces"].reshape(f, 1, 1, 1, n, 3),
                "atom_label": np.array(ATOM_ALPHABET)[data["nuclear_charges"] - 1]
                .reshape(1, 1, 1, 1, n)
                .repeat(f, axis=0),
            }
            yield data

        batches = batched(IteratorWithLength(generator(), 1))
        return batches, Split([]), Assets({})
