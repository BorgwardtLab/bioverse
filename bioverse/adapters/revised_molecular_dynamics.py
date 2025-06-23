from glob import glob
from pathlib import Path

import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config, download, progressbar


class RevisedMolecularDynamicsAdapter(Adapter):
    """Adapter for rMD17."""

    @classmethod
    def download(cls):
        path = config.raw_path / "RevisedMolecularDynamics17"
        download(
            "https://figshare.com/ndownloader/files/23950376",
            path,
            extension=".tar.bz2",
        )
        files = glob(str(path / "rmd17" / "npz_data" / "*.npz"))

        def generator():
            for file in progressbar(files, description="Processing"):
                data = np.load(file)
                name = Path(file).name.split(".")[0].split("_")[1]
                data = {
                    "frame_id": [0],
                    "molecule_id": [name],
                    "molecule_energy": [[[[data["energies"]]]]],
                    "atom_pos": [[[[data["coords"]]]]],
                    "atom_force": [[[[data["forces"]]]]],
                    "atom_charge": [[[[data["nuclear_charges"]]]]],
                }
                yield data

        batches = batched(IteratorWithLength(generator(), len(files)))
        return batches, Split([]), Assets({})
