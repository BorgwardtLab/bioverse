from glob import glob
from pathlib import Path

import awkward as ak
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
            "https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip",
            path / "structures",
        )
        download(
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip",
            path / "mutations",
        )
        structures = PdbProcessor.process(path / "structures")
        structures = {item["molecule_id"][0, 0]: item for item in structures}
        paths = glob(
            str(path / "mutations" / "cv_folds_singles_substitutions" / "*.csv")
        )
        split, items = [], []
        for path in paths:
            name = Path(path).stem
            if not name in structures:
                continue
            df = pd.read_csv(path)
            pos = df["mutant"].map(
                lambda m: [int(x[1:-1]) - 1 for x in m.split(":")]  # type: ignore
            )  # type: ignore
            wt = df["mutant"].map(
                lambda m: [x[0] for x in m.split(":")]  # type: ignore
            )  # type: ignore
            label = df["mutant"].map(
                lambda m: [x[-1] for x in m.split(":")]  # type: ignore
            )  # type: ignore
            item = structures[name]
            assert ak.all(
                ak.Array(pos).ravel() < ak.num(item["residue_label"], axis=3).ravel()[0]
            )
            item["molecule_mutation_label"] = [[label]]
            item["molecule_mutation_pos"] = [[pos]]
            item["molecule_mutation_effect"] = [[df["DMS_score"].tolist()]]
            split.extend(
                "test" if v == 0 else "val" if v == 1 else "train"
                for v in df["fold_random_5"]
            )
            items.append(item)

        def generator():
            for item in items:
                yield item

        batches = batched(IteratorWithLength(generator(), len(paths)))
        return (
            batches,
            Split({"random_mutation_split": split}, default="random_mutation_split"),
            Assets({}),
        )
