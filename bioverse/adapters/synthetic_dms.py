"""Synthetic protease DMS matrices from MEROPS, BioLiP, and ProteinGym priors."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import PROTEIN_ALPHABET, config, note
from ..utilities.cleavage import prefer_uniprot
from ..utilities.synthetic_dms import (
    ComposerConfig,
    MutationContext,
    MutationEffectPrior,
    RewirePrior,
    SubstratePwmPrior,
    burial_bins,
    cleft_pocket_prob,
    generate_matrix,
    scale_rows,
    HELD_OUT_FAMILIES,
)
from .biolip import ensure_annotations, parse_annotations
from .merops import (
    DEFAULT_EXCLUDE,
    _assign_window_splits,
    _family,
    load_uniprot_map,
    merops_path,
    parse_cleavages,
)
from .protease_structures import (
    flatten_atom_names,
    flatten_atom_pos,
    flatten_residue_numbers,
    records_to_batches,
    residue_letters_from,
    resolve_protease_structure,
)
from .protein_gym import centered_effects, keep_assay

HELD_OUT_UNIPROT = frozenset({"P04517", "P08253", "P14780"})
HUB_EXCLUDE = set(DEFAULT_EXCLUDE)


def _ca_coords(record) -> tuple[str, np.ndarray, list[int]]:
    letters = residue_letters_from(record["residue_label"])
    atoms = (
        flatten_atom_names(record["atom_label"])
        if "atom_label" in record.fields
        else [["CA"] for _ in letters]
    )
    pos = (
        flatten_atom_pos(record["atom_pos"])
        if "atom_pos" in record.fields
        else [[[float(i), 0.0, 0.0]] for i in range(len(letters))]
    )
    n = min(len(letters), len(atoms), len(pos))
    letters, atoms, pos = letters[:n], atoms[:n], pos[:n]
    numbers = flatten_residue_numbers(
        record["residue_number"] if "residue_number" in record.fields else None,
        n,
    )[:n]
    xyz = []
    seq = []
    kept_numbers = []
    for letter, names, coords, number in zip(letters, atoms, pos, numbers):
        if not coords:
            continue
        if "CA" in names:
            xyz.append(coords[names.index("CA")])
        else:
            xyz.append(np.mean(np.asarray(coords, dtype=float), axis=0).tolist())
        seq.append(letter)
        kept_numbers.append(int(number))
    return "".join(seq), np.asarray(xyz, dtype=np.float64), kept_numbers


def _attach_synthetic(record, rows):
    data = {field: record[field] for field in record.fields}
    data["molecule_mutation_label"] = [[[row.mut] for row in rows]]
    data["molecule_mutation_pos"] = [[[int(row.pos)] for row in rows]]
    data["molecule_mutation_effect"] = [[float(row.delta) for row in rows]]
    data["molecule_mutation_substrate"] = [[str(row.window) for row in rows]]
    data["molecule_mutation_kind"] = [[str(row.kind) for row in rows]]
    data["molecule_mutation_pocket"] = [[float(row.pocket) for row in rows]]
    return ak.Record(data)


def load_biolip_peptide_sites() -> dict[str, set[int]]:
    """UniProt → PDB residue numbers contacted by a peptide in BioLiP."""
    path = ensure_annotations(config.raw_path / "BioLiP")
    sites: dict[str, set[int]] = defaultdict(set)
    for entry in parse_annotations(path, ligand="peptide"):
        uniprot = str(entry.get("uniprot_id") or "").strip().upper()
        if not uniprot or uniprot in HELD_OUT_UNIPROT:
            continue
        for _chain, resi, _ins in entry.get("binding_site") or []:
            sites[uniprot].add(int(resi))
    return dict(sites)


def pocket_probability(coords, residue_numbers, uniprot, biolip_sites) -> np.ndarray:
    q = cleft_pocket_prob(coords)
    if uniprot and uniprot in biolip_sites and residue_numbers:
        hits = biolip_sites[uniprot]
        for i, number in enumerate(residue_numbers):
            if int(number) in hits:
                q[i] = max(float(q[i]), 0.92)
    return np.clip(q, 0.0, 1.0)


def fit_mutation_prior(
    *,
    dms_types=("Activity", "Binding", "Stability"),
    exclude_substrings=("TEV", "TEVP", "NIa", "P04517", "TOBACCO_ETCH"),
    max_per_assay: int = 256,
    seed: int = 0,
) -> MutationEffectPrior:
    """Fit a stratified ProteinGym effect prior. Does not copy assay scores."""
    from glob import glob

    from ..processors import PdbProcessor
    from ..utilities import download

    rng = np.random.default_rng(int(seed))
    path = config.raw_path / "ProteinGym"
    download(
        "https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip",
        path / "structures",
    )
    download(
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip",
        path / "mutations",
    )
    structures = {
        item["molecule_id"][0, 0]: item
        for item in PdbProcessor.process(path / "structures")
    }
    reference = None
    ref_path = path / "DMS_substitutions.csv"
    try:
        from .protein_gym import REFERENCE_URL

        download(REFERENCE_URL, ref_path)
        if ref_path.exists():
            import pandas as pd

            reference = pd.read_csv(ref_path)
    except Exception:
        reference = None
    contexts, effects = [], []
    csvs = sorted(
        glob(str(path / "mutations" / "cv_folds_singles_substitutions" / "*.csv"))
    )
    for csv_path in csvs:
        name = Path(csv_path).stem
        if name not in structures:
            continue
        if not keep_assay(name, dms_types, exclude_substrings, reference):
            continue
        import pandas as pd

        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        record = structures[name]
        sequence, coords, _numbers = _ca_coords(record)
        if len(sequence) < 8 or coords.shape[0] != len(sequence):
            continue
        burials = burial_bins(coords)
        values = np.asarray(centered_effects(df, how="wt", scale="mad"), dtype=float)
        take = min(int(max_per_assay), len(df))
        pick = rng.choice(len(df), size=take, replace=False) if len(df) > take else np.arange(len(df))
        for row_i in pick:
            mutant = str(df.iloc[int(row_i)]["mutant"])
            parts = [part for part in mutant.split(":") if len(part) >= 3]
            if len(parts) != 1:
                continue
            wt, mut = parts[0][0], parts[0][-1]
            try:
                pos = int(parts[0][1:-1]) - 1
            except ValueError:
                continue
            if pos < 0 or pos >= len(sequence):
                continue
            if str(sequence[pos]).upper() != wt.upper():
                continue
            if wt not in PROTEIN_ALPHABET or mut not in PROTEIN_ALPHABET:
                continue
            contexts.append(MutationContext.from_mutation(wt, mut, burials[pos]))
            effects.append(float(values[int(row_i)]))
    prior = MutationEffectPrior().fit(contexts, effects)
    note(f"Synthetic DMS: ProteinGym prior fit on {prior.n_fit} substitutions.")
    return prior


def fit_rewire_prior(
    *,
    exclude_families=HELD_OUT_FAMILIES,
    exclude_codes=None,
    min_canonical: int = 4,
    pwm_weight: float = 0.5,
) -> RewirePrior:
    """Fit the pocket–substrate rewire table on MEROPS, excluding Huber families."""
    exclude_codes = set(HUB_EXCLUDE if exclude_codes is None else exclude_codes)
    exclude_families = {str(fam).upper() for fam in exclude_families}
    pairs = parse_cleavages(min_canonical=min_canonical)
    windows = []
    for code, items in pairs.items():
        if code in exclude_codes:
            continue
        if _family(code) in exclude_families:
            continue
        windows.extend(items)
    prior = RewirePrior().fit(windows, pwm_weight=pwm_weight)
    note(f"Synthetic DMS: MEROPS rewire prior fit on {prior.n_fit} windows.")
    return prior


def iter_merops_proteases(
    *,
    exclude=HUB_EXCLUDE,
    max_length: int = 400,
    max_windows: int | None = 64,
    max_proteases: int | None = None,
    min_windows: int = 2,
    min_canonical: int = 4,
):
    """Yield resolved MEROPS proteases. Huber/Cieplak identifiers are skipped."""
    uniprot_map = load_uniprot_map()
    pairs = parse_cleavages(min_canonical=min_canonical)
    pdb_dir = merops_path() / "pdb"
    af_dir = merops_path() / "alphafold"
    mapping_dir = merops_path() / "pdbe"
    n = 0
    for code in sorted(code for code in pairs if code in uniprot_map):
        if code in exclude:
            continue
        if max_proteases is not None and n >= max_proteases:
            break
        uniprot = prefer_uniprot(uniprot_map[code])
        if uniprot is None or str(uniprot).upper() in HELD_OUT_UNIPROT:
            continue
        try:
            resolved = resolve_protease_structure(
                uniprot,
                pdb_dir=pdb_dir,
                af_dir=af_dir,
                mapping_dir=mapping_dir,
                molecule_id=code,
            )
        except Exception as exc:
            note(f"Failed structure for {code} ({uniprot}): {exc}")
            continue
        if resolved is None:
            continue
        record, source_name = resolved
        sequence, coords, numbers = _ca_coords(record)
        if not sequence or coords.shape[0] != len(sequence):
            continue
        if len(sequence) == 0 or len(sequence) > max_length:
            continue
        windows = list(pairs[code])
        if max_windows is not None:
            windows = windows[: max_windows]
        if len(windows) < min_windows:
            continue
        n += 1
        yield {
            "code": code,
            "uniprot": str(uniprot).upper(),
            "source": source_name,
            "record": record,
            "sequence": sequence,
            "coords": coords,
            "numbers": numbers,
            "windows": windows,
        }


def _split_by_mutation(rows_per_code, codes, seed, train=0.8, val=0.1):
    """Assign train/val/test by unique (pos, mut); all substrates follow."""
    n_rows = []
    labels_by_code = {}
    for code, rows in zip(codes, rows_per_code):
        keys = [(int(row.pos), str(row.mut)) for row in rows]
        unique = list(dict.fromkeys(keys))
        assignment = _assign_window_splits(
            [len(unique)],
            [code],
            exclude=set(),
            seed=seed,
            train=train,
            val=val,
        )
        key_to_split = {key: assignment[i] for i, key in enumerate(unique)}
        labels_by_code[code] = [key_to_split[key] for key in keys]
        n_rows.append(len(rows))
    labels = []
    for code in codes:
        labels.extend(labels_by_code[code])
    return labels


def build_synthetic_records(
    entries,
    *,
    mutation_prior: MutationEffectPrior,
    substrate_prior: SubstratePwmPrior,
    biolip_sites: dict[str, set[int]] | None,
    cfg: ComposerConfig,
    rng: np.random.Generator,
    n_pocket: int = 28,
    n_offpocket: int = 12,
    n_cognate: int = 8,
    n_decoy: int = 4,
):
    """Compose A/B/C synthetic matrices onto MEROPS protease records."""
    biolip_sites = biolip_sites or {}
    window_pool = [w for item in entries for w in item["windows"]]
    records, used, row_groups = [], [], []
    for item in entries:
        q = pocket_probability(
            item["coords"], item["numbers"], item["uniprot"], biolip_sites
        )
        decoys = [w for w in window_pool if w not in item["windows"]]
        rng.shuffle(decoys)
        rows = generate_matrix(
            sequence=item["sequence"],
            pocket_q=q,
            windows=item["windows"],
            decoys=decoys,
            mutation_prior=mutation_prior,
            substrate_prior=substrate_prior,
            protease_id=item["code"],
            cfg=cfg,
            rng=rng,
            coords=item["coords"],
            n_pocket=n_pocket,
            n_offpocket=n_offpocket,
            n_cognate=n_cognate,
            n_decoy=n_decoy,
        )
        rows = scale_rows(rows)
        if not rows:
            continue
        record = _attach_synthetic(item["record"], rows)
        data = {field: record[field] for field in record.fields}
        data["molecule_uniprot"] = ak.Array([[item["uniprot"]]])
        data["molecule_structure_source"] = ak.Array([[item.get("source", "unknown")]])
        data["molecule_merops_id"] = ak.Array([[item["code"]]])
        data["molecule_family"] = ak.Array([[_family(item["code"])]])
        data["molecule_pocket"] = ak.Array([[q.tolist()]])
        record = ak.Record(data)
        records.append(record)
        used.append(item["code"])
        row_groups.append(rows)
    return records, used, row_groups


class SyntheticDmsAdapter(Adapter):
    """Synthetic protease × mutation × substrate Δactivity matrices.

    Formulations (``formulation``):

    * ``mutation`` — ProteinGym mutation prior only
    * ``mutation_pocket`` — mutation × BioLiP/cleft pocket weight
    * ``mutation_pocket_substrate`` — also MEROPS PWM substrate specificity

    Huber TEV and Cieplak MMPs are excluded from proteases, windows, and the
    ProteinGym prior. Generation is deterministic given ``seed``.
    """

    @classmethod
    def download(
        cls,
        formulation: str = "mutation_pocket_substrate",
        seed: int = 0,
        max_length: int = 400,
        max_windows: int | None = 64,
        max_proteases: int | None = None,
        min_windows: int = 2,
        n_pocket: int = 28,
        n_offpocket: int = 12,
        n_cognate: int = 8,
        n_decoy: int = 4,
        pocket_floor: float = 0.15,
        substrate_gain: float = 0.75,
        noise_scale: float = 0.15,
        train_size: float = 0.8,
        val_size: float = 0.1,
        pocket_source: str = "biolip",
        pocket_teacher: str = "",
    ):
        cfg = ComposerConfig(
            formulation=formulation,
            pocket_floor=float(pocket_floor),
            substrate_gain=float(substrate_gain),
            noise_scale=float(noise_scale),
        )
        rng = np.random.default_rng(int(seed))
        entries = list(
            iter_merops_proteases(
                max_length=max_length,
                max_windows=max_windows,
                max_proteases=max_proteases,
                min_windows=min_windows,
            )
        )
        if not entries:
            raise RuntimeError("Synthetic DMS adapter found no MEROPS proteases.")
        mutation_prior = fit_mutation_prior(seed=seed)
        substrate_prior = SubstratePwmPrior().fit(
            {item["code"]: item["windows"] for item in entries}
        )
        biolip_sites = {}
        if pocket_source in {"biolip", "teacher", "both"}:
            try:
                biolip_sites = load_biolip_peptide_sites()
            except Exception as exc:
                note(f"BioLiP peptide sites unavailable ({exc}); using cleft prior.")
        if pocket_source == "teacher" and pocket_teacher:
            note(
                "pocket_teacher is recorded but BioLiP/cleft q_i are used at "
                "label time so dataset release stays deterministic without GPU."
            )
        records, used, row_groups = build_synthetic_records(
            entries,
            mutation_prior=mutation_prior,
            substrate_prior=substrate_prior,
            biolip_sites=biolip_sites,
            cfg=cfg,
            rng=rng,
            n_pocket=n_pocket,
            n_offpocket=n_offpocket,
            n_cognate=n_cognate,
            n_decoy=n_decoy,
        )
        if not records:
            raise RuntimeError("Synthetic DMS adapter produced no labeled proteases.")
        mutation_splits = _split_by_mutation(row_groups, used, seed, train_size, val_size)
        scene_splits = ["train"] * len(used)
        n_val = max(1, int(round(len(used) * val_size))) if len(used) > 2 else 0
        if n_val:
            scene_splits[-n_val:] = ["val"] * n_val
        batches = records_to_batches(records)
        split = Split(
            {
                "random_scene_split": scene_splits,
                "random_mutation_split": mutation_splits,
            },
            default="random_mutation_split",
        )
        n_rows = int(sum(len(rows) for rows in row_groups))
        assets = Assets(
            {
                "window_len": 10,
                "formulation": formulation,
                "seed": int(seed),
                "exclude_codes": sorted(HUB_EXCLUDE),
                "n_proteases": len(records),
                "n_rows": n_rows,
                "n_pg_prior": int(mutation_prior.n_fit),
                "n_biolip_uniprot": len(biolip_sites),
                "pocket_source": pocket_source,
                "pocket_teacher": str(pocket_teacher or ""),
                "pocket_floor": float(pocket_floor),
                "substrate_gain": float(substrate_gain),
                "noise_scale": float(noise_scale),
            }
        )
        note(
            f"Synthetic DMS {formulation}: {len(records)} proteases, {n_rows} rows."
        )
        return batches, split, assets
