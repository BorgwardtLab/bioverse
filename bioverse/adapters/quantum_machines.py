import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import XyzProcessor
from ..utilities import HARTREE_TO_EV, batched, config, download

# fmt: off
QM9_PROPS = ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]
# fmt: on


class QuantumMachinesAdapter(Adapter):
    """Adapter for QM9."""

    @classmethod
    def download(cls):
        path = config.raw_path / "QuantumMachines"
        download(
            "https://figshare.com/ndownloader/files/3195389", path, extension=".tar.bz2"
        )
        batches = batched(XyzProcessor.process(path))
        converter = np.ones(15)
        converter[[5, 6, 7, 9, 10, 11, 12, 13]] = HARTREE_TO_EV

        # convert the Hartee values to eV
        def modify(batch):
            batch.molecule_labels = batch.molecule_labels * converter[np.newaxis]
            for i, name in enumerate(QM9_PROPS):
                batch.__setattr__(f"molecule_{name}", batch.molecule_labels[:, i])
            batch.data.pop("molecule_labels")
            return batch

        return batches.map(modify), Split([]), Assets({})
