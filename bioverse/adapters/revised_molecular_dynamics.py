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
        data = np.load(path / "rmd17" / "npz_data" / f"rmd17_{name}.npz")
        n, m = data["coords"].shape[1], data["coords"].shape[0]
        with open(path / "rmd17" / "splits" / "index_test_01.csv") as f:
            test_index = np.array(list(map(int, f.read().splitlines())))
        with open(path / "rmd17" / "splits" / "index_train_01.csv") as f:
            train_index = np.array(list(map(int, f.read().splitlines())))
        rng = np.random.default_rng(0)
        rng.shuffle(train_index)
        n_val = 100
        train_index, val_index = train_index[:-n_val], train_index[-n_val:]
        index = np.concatenate([train_index, val_index, test_index])
        f = len(index)
        split = np.array(
            ["train"] * len(train_index)
            + ["val"] * len(val_index)
            + ["test"] * len(test_index)
        )
        split = split[np.argsort(index)]

        def generator():
            yield {
                "frame_id": np.arange(f),
                "molecule_id": np.array([[name]] * f),
                "molecule_energy": data["energies"][index].reshape(f, 1),
                "atom_pos": data["coords"][index].reshape(f, 1, 1, 1, n, 3),
                "atom_force": data["forces"][index].reshape(f, 1, 1, 1, n, 3),
                "atom_label": np.array(ATOM_ALPHABET)[data["nuclear_charges"] - 1]
                .reshape(1, 1, 1, 1, n)
                .repeat(f, axis=0),
            }

        batches = batched(IteratorWithLength(generator(), 1))
        return (
            batches,
            Split({"rmd17_frame_split": split}, default="rmd17_frame_split"),
            Assets({}),
        )
