from glob import glob
from pathlib import Path

import pandas as pd

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import IteratorWithLength, batched, config, download


class ProteinGymAdapter(Adapter):
    """Adapter for ProteinGym."""

    @classmethod
    def download(cls):
        path = config.raw_path / "ProteinGym"
        download(
            "https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip",
            path / "mutations",
        )
        download(
            "https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip",
            path / "structures",
        )
        structures = PdbProcessor.process(path / "structures")
        structures = {item["molecule_id"][0, 0]: item for item in structures}
        paths = glob(str(path / "mutations" / "DMS_ProteinGym_substitutions" / "*.csv"))

        def generator():
            for path in paths:
                name = Path(path).stem
                if not name in structures:
                    continue
                df = pd.read_csv(path)
                df["mutant"] = df["mutant"].map(
                    lambda m: [(int(x[1:-1]), x[-1]) for x in m.split(":")]  # type: ignore
                )  # type: ignore
                item = structures[name]
                item["molecule_mutations"] = [[df["mutant"].tolist()]]
                item["molecule_mutation_effect"] = [[df["DMS_score"].tolist()]]
                yield item

        batches = batched(IteratorWithLength(generator(), len(paths)))
        return batches, Split([[0]] * len(paths), names=["test"]), Assets({})
