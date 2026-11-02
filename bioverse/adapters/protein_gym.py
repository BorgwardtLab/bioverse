from glob import glob
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import IteratorWithLength, batched, config, download, note

REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
    "reference_files/DMS_substitutions.csv"
)
DEFAULT_DMS_TYPES = ("Activity", "Binding", "Stability")


def _is_identity_mutant(text):
    parts = str(text).split(":")
    return all(len(part) >= 3 and part[0] == part[-1] for part in parts)


def centered_effects(df, how="wt", scale="none"):
    """WT-center DMS scores, optionally divide by a per-assay robust scale.

    ``scale="mad"`` divides by ``1.4826 * MAD`` so assays share a unit that
    matches a standard deviation for roughly Gaussian effects. A tiny floor
    falls back to the sample std, then to 1.0.
    """
    scores = pd.to_numeric(df["DMS_score"], errors="coerce")
    if how in (None, "none", False):
        centered = scores.fillna(0.0)
    else:
        wt_mask = df["mutant"].map(_is_identity_mutant)
        if bool(wt_mask.any()):
            wt = float(scores[wt_mask].mean())
        else:
            wt = float(scores.median())
        centered = (scores - wt).fillna(0.0)
    if scale in (None, "none", False):
        return centered.tolist()
    values = np.asarray(centered, dtype=float)
    if scale in ("mad", "robust"):
        mad = float(np.median(np.abs(values - np.median(values))))
        denom = 1.4826 * mad
    else:
        denom = float(np.std(values))
    if not np.isfinite(denom) or denom < 1e-6:
        denom = float(np.std(values))
    if not np.isfinite(denom) or denom < 1e-6:
        denom = 1.0
    return (centered / denom).tolist()


def keep_assay(name, dms_types, exclude_substrings, reference):
    text = str(name).upper()
    if exclude_substrings:
        if any(str(token).upper() in text for token in exclude_substrings):
            return False
    if reference is None or dms_types is None:
        return True
    row = reference[reference["DMS_id"].astype(str) == str(name)]
    if row.empty:
        # Keep assays missing from the reference file unless excluded above.
        return True
    kind = str(row.iloc[0].get("selection_type", "") or row.iloc[0].get("DMS_type", ""))
    uni = str(row.iloc[0].get("UniProt_ID", "") or "")
    blob = f"{name} {uni} {kind}".upper()
    if exclude_substrings and any(str(token).upper() in blob for token in exclude_substrings):
        return False
    allowed = {str(item) for item in dms_types}
    return kind in allowed or not kind


class ProteinGymAdapter(Adapter):
    """Download deep mutational scanning benchmarks from ProteinGym."""

    @classmethod
    def download(
        cls,
        assays: list[str] | None = None,
        dms_types: list[str] | None = None,
        exclude_substrings: list[str] | None = None,
        center: str = "none",
        scale: str = "none",
    ):
        path = config.raw_path / "ProteinGym"
        download(
            "https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip",
            path / "structures",
        )
        download(
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip",
            path / "mutations",
        )
        reference = None
        if dms_types is not None or exclude_substrings:
            ref_path = path / "DMS_substitutions.csv"
            download(REFERENCE_URL, ref_path)
            # download() may append the URL suffix if the dest has none.
            if not ref_path.exists():
                alt = ref_path.with_suffix(".csv")
                ref_path = alt if alt.exists() else ref_path
            try:
                reference = pd.read_csv(ref_path)
            except Exception:
                reference = None
                note("ProteinGym reference file unreadable; falling back to name filters.")
        structures = PdbProcessor.process(path / "structures")
        structures = {item["molecule_id"][0, 0]: item for item in structures}
        paths = glob(
            str(path / "mutations" / "cv_folds_singles_substitutions" / "*.csv")
        )
        split, items = [], []
        skipped = 0
        for csv_path in paths:
            name = Path(csv_path).stem
            if assays is not None and name not in assays:
                continue
            if not name in structures:
                continue
            if not keep_assay(name, dms_types, exclude_substrings, reference):
                skipped += 1
                continue
            df = pd.read_csv(csv_path)
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
            item["molecule_mutation_effect"] = [
                [centered_effects(df, how=center, scale=scale)]
            ]
            split.extend(
                "test" if v == 0 else "val" if v == 1 else "train"
                for v in df["fold_random_5"]
            )
            items.append(item)

        def generator():
            for item in items:
                yield item

        if skipped:
            note(f"ProteinGym: skipped {skipped} assays by type/homology filter.")
        batches = batched(IteratorWithLength(generator(), len(items)))
        return (
            batches,
            Split({"random_mutation_split": split}, default="random_mutation_split"),
            Assets({"n_assays": len(items), "n_skipped": skipped}),
        )

